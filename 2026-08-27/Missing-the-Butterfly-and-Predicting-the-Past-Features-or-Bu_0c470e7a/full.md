# Missing the Butterfly and Predicting the Past: Features or Bugs of Accurate AI Weather Models?

Pedram Hassanzadeh<sup>1,2,3†∗</sup>

Weidong Li<sup>1†</sup> Y. Qiang Sun<sup>4,5,1∗</sup> Jiangdi Wang<sup>2</sup>

Alexander Wikner<sup>1,3</sup>

Justin Finkel<sup>3</sup>

Jonathan Q. Weare<sup>6</sup>

<sup>1</sup>Department of Geophysical Sciences, University of Chicago, Chicago, USA <sup>2</sup>Committee on Computational and Applied Mathematics, University of Chicago, Chicago, USA. <sup>3</sup>Data Science Institute, University of Chicago, Chicago, USA.

<sup>4</sup>State Key Laboratory of Severe Weather Meteorological Science and Technology, Nanjing University, Nanjing, China. <sup>5</sup>Key Laboratory of Mesoscale Severe Weather/Ministry of Education, Nanjing University, Nanjing, China.   
<sup>6</sup>Courant Institute School of Mathematics, Computing, and Data Science, New York University, New York City, USA.

<sup>∗</sup>Corresponding authors: pedramh@uchicago.edu, qiangsun@nju.edu.cn <sup>†</sup>These authors contributed equally to this work.

## Abstract

AI weather prediction (AIWP) models rival physics-based models, yet the sources oftheir unexpected forecast accuracy and the degree oftheir physical fidelity remain unclear. Here, across a hierarchy spanning observation-based reanalysis, a general circulation model, and the multi-scale Lorenz system, we show that AI models can be trained to skillfully predict the past (backcast), though backcasts are systematically less accurate than forecasts. However, skillful backcasting appears to violate the second law of thermodynamics, and all these forecasting and backcasting models miss the butterfly efect. We trace the surprising forecast accuracy, missing butterfly, and skillful backcasting to a single cause: inevitable coarse-graining of training data, which removes fast, small scales and/or some variables. From the Lorenz system to oficial Pangu-Weather models, reducing coarse-graining makes AI predictions more physics-like (arrow of time and butterfly-like efects emerge), but forecast accuracy declines. Results ofer an explanation for AIWP models’ forecast skill: unlike physics-based models, they implicitly learn how fast, small scales afect large scales without inheriting their rapid error growth. Broader implications are that AI models’ proliferation calls for revisiting predictability theories and long-term climate emulation strategies, and backcasting ofers a useful, new lens for such analyses.

## 1 Introduction

Artificial intelligence (AI) weather prediction (AIWP) models have transformed short- to mediumrange (days to two weeks) forecasting [1–3]: they outperform the best physics-based numerical weather prediction (NWP) models across a wide range of common skill metrics at orders ofmagnitude lower computational cost [4–7]. State-of-the-art AIWP models are trained in a broadly similar fashion: to advance the atmosphere’s 3D global state x from time � to � + Δ� using a deep neural network NN, i.e.,

$$
\mathbf { x } ( t + \Delta t ) = N N _ { \theta } \big ( \mathbf { x } ( t ) \big ) ,\tag{1}
$$

where Δ� is often 6-24 hours, � represents the learnable parameters, and the training data come from ∼ 40 years of observation-based reanalysis products like ERA5 [8]. Longer forecasts are produced autoregressively. Given their accuracy and computational eficiency, AIWP models are now being operationalized by leading forecasting centers [9, 10] and increasingly used to provide actionable weather information to those most in need [11]. More recently, these models have been pushed to subseasonal-to-seasonal time scales and even to longer-term climate projections, with promising early results [12, 13].

Yet, a growing number of studies have identified several failure modes of AIWP models, e.g., their inability to forecast unprecedented “gray swan” weather extremes [14–16] and to reproduce the small-scales’ statistics [17, 18]. Most surprisingly, these highly skillful models were found to lack the butterfly efect [19]: the rapid growth of ensemble spread, which accelerates as initial-condition perturbations vanish, in multi-scale chaotic systems [20–22]. This deficiency appears shared across otherwise very diferent deterministic and generative AI architectures [23]. While this failure has generated concerns, its root cause and implications remain unclear. Meanwhile, the sources of AIWP models’ surprising accuracy are also poorly understood, hindering trust and further improvements. Indeed, this accuracy comes despite long-standing skepticism about skillful data-driven weather forecasting, e.g., from Lorenz [24] and others, including an estimated requirement of $\sim 1 0 ^ { 3 0 }$ years of training data [25]; see Discussion. Underlying these puzzles, fundamental questions about the chaotic atmosphere’s predictability limits, and the butterfly efect’s practical relevance, remain a subject of extensive research and debate [26–30].

Here, we investigate the missing butterfly efect and surprising forecast accuracy of AIWP models. First, since nothing in these AIWP models’ design restricts them to the arrow of time, we also train “backcasting” models: neural networks that predict the past (Fig. 1A), i.e.,

$$
\mathbf { x } ( t - \Delta t ) = N N _ { \phi } \big ( \mathbf { x } ( t ) \big ) ,\tag{2}
$$

the Eq. (1) “forecast” model’s time-reversed counterpart, with its own (independent) learnable parameters �. As shown below, autoregressive backcasting ofers a new, useful dimension for exploring AIWP models and fundamentals of atmospheric predictability.

Across a hierarchy spanning ERA5 reanalysis, an intermediate-complexity general circulation model (GCM), and multi-scale Lorenz 96, we show that, while skillful, these backcasting AI models consistently behave in ways sharply at odds with fundamental understanding of dynamical systems and their numerical integrations. Simply put, skillful backcasting appears to violate the second law of thermodynamics. We also show that across the hierarchy, the butterfly efect is missing from these AI forecasting and backcasting models alike. We then explicitly connect all three (surprising AI forecast skills, the missing butterfly efect, skillful AI backcasting), showing they are symptoms of the same underlying cause. We examine whether each symptom is a feature or a bug of skillful AIWP models.

## 2 Results

For each system in our hierarchy, the backcasting AI models (Eq. (2)) are architecturally identical to the forecasting models (Eq. (1)) and are trained on the same data, optimizer, and loss function (often the mean absolute or squared error, MAE/MSE); the sole diference is the reversed input–output pairing (Fig. 1A). We have trained a number of deterministic and generative AIWP models based on state-of-the-art architectures (see Methods and Data).

## 2.1 Skillful AI backcasting and persistent asymmetry: ERA5 and PlaSim (GCM)

Figure 1B shows the anomaly correlation coeficient (ACC) of 500-hPa geopotential (Z500) as a function of lead time. On ERA5, the backcasting Transformer (deterministic) remains skillful to 6.5 days into the past while the forecast is skillful for 9.1 days. It might be surprising that the past is harder to predict (asymmetry = 9.1/6.5 > 1). However, backcasting should be practically impossible for dissipative dynamical systems such as the atmosphere, because, simply put, it leads to an apparent violation ofthe second law ofthermodynamics! As further explained in the Supplementary Materials, time reversal turns dissipation (which is prevalent in the atmosphere across scales) into anti-dissipation (and reverses the sign of Lyapunov exponents), so unavoidable small errors grow explosively, fastest at the smallest scales. Accordingly, backcasting using numerical integrations fails catastrophically; see examples from the heat equation to Lorenz 63 to Lorenz 96 in the Supplementary Materials and Fig. 2A. This is not the case for the AIWP model of ERA5.

The asymmetry > 1 is not an artifact of ERA5, which, as an assimilation product continually updated by observations, is not a closed dynamical system: PlaSim, which is one (a GCM), exhibits even larger asymmetries, including in generative and neural-operator AIWP models (Figs. 1B and S1- S2). The asymmetry is robust across variables, pressure levels, and skill metrics in both ERA5 and PlaSim (Figs. S3-S4). Geographically, the asymmetry is overall larger in the tropics than in the extratropics (Table S1); likely a reflection of the stronger diabatic and dissipative processes (higher entropy production rate) there [31, 32], so that the tropical atmosphere is, in this sense, further from reversibility (see the Supplementary Materials).

Two robust findings therefore need explaining: that backcasting works at all, and that the past is nevertheless consistently harder to predict than the future. But first, let us also examine the error growth and butterfly efect in these forecasting and backcasting AIWP models.

## 2.2 Missing butterflies in AI forecasting and backcasting alike: ERA5 and PlaSim (GCM)

In a multi-scale chaotic system like the atmosphere, ensemble spread from vanishingly small initial perturbations should grow rapidly, fed by upscale error transfer from the small scales, and saturate after a finite time. This is the “real” butterfly efect [20–22, 28], which sets the upper limit of atmospheric predictability. We quantify the spread with the diference kinetic energy (DKE), the area-weighted ensemble variance of the horizontal winds (Methods and Data). Figure 1C shows that numerical models, ICON (NWP) and the PlaSim GCM, reproduce this behavior; however, no AIWP model does, in forecasting or backcasting, for ERA5 or PlaSim.

The growth of DKE in the numerical integrations of physics-based forecast models behaves as expected: in both ICON and PlaSim GCM, the smallest-amplitude ensembles grow by orders of magnitude within a day or two, showing diminishing return as the initial-condition errors shrink (Fig. 1C). This is the butterfly efect [21, 22]. For the large-amplitude perturbations, ensemble forecasts from numerical integrations and AIWP models behave similarly, showing exponential growth in DKE over several days (unlike the butterfly efect, the growth rate here is independent of the perturbation amplitude). The forecasting results in ERA5 agree with those of Selz and Craig in their pioneering papers [19, 23]. Here, we show the same behavior in the AIWP models of a GCM, but most strikingly, in backcasting AIWP models. The fundamental understanding of time-reversal physics (Supplementary Materials) and numerical experiments with Lorenz 96 (Fig. 3A) suggest even a more explosive growth of DKE in backcasting. However, we robustly see a near symmetry between the forecasting and backcasting DKE growth across the hierarchy.

We are thus left with three puzzles about AIWP models: on one hand, forecast accuracy that longstanding arguments deemed unattainable, and on the other, skillful backcasting and a missing butterfly efect that disagree with the fundamental understanding of multi-scale chaotic dynamics, robustly confirmed by numerical integrations of physics-based models. To further probe these findings, we extend the hierarchy to a canonical testbed of multi-scale, chaotic dynamics, the Lorenz 96 system.

## 2.3 Training-data coarse-graining dramatically alters forecasts, backcasts, and butterflies: Lorenz 96

The two-scale Lorenz 96 system (Eqs. (S7)-(S8)) couples 8 slow, large-amplitude variables $X _ { i }$ (representing large-scale circulation) to 256 fast, small-amplitude variables $Y _ { i , j }$ whose characteristic time scale is an order of magnitude shorter (representing small-scale, often subgrid-scale processes); see Methods and Data. Its numerical integration displays the time irreversibility that should make any backcasting impractical: Fig. 2A shows that integrated backward, the trajectory blows up rapidly due to anti-dissipation. As demonstrated in Fig. 3, the numerical ensemble forecasts of this system exhibit the butterfly efect, and the overall DKE growth curves show a behavior similar to those from numerical integration of physics-based models like ICON and PlaSim GCM as the perturbation amplitudes are varied (compare to Fig. 1C).

The AI forecasting and backcasting models can exhibit very diferent behaviors depending on the degree of coarse-graining of the training data: How many spatio-temporal scales the training data includes. We first train AI models with data closest to the “real-world” regime: only using the large scales and after removing the fastest variabilities (�-only, $\Delta t = 1 0 \delta t$ where �� is the time step of the numerical solver/ground truth). As shown in the top-left corners of Figs. 2B-C and 3, the forecasts and backcasts of these AI models have remarkably similar characteristics as those of AIWP models of ERA5 and PlaSim (Figs. 1B-C and S3-S4): Accurate forecasts, skillful backcasts with asymmetry > 1, amplitude-independent exponential growth in DKE, and missing butterflies. Hovmoller diagrams and DKE curves of backcasts demonstrate behavior that is markedly diferent¨ from that of numerical integrations.

A  
![](images/23cc932fb8847d87d4c1d532ef26fe4ff1f2f36ef50a7ceff69e7de41de1c918.jpg)

B  
![](images/16334baf58486ef35a702ae7e6f2b80b8cd6feec6c478cbb5cca515346c89c01.jpg)

![](images/f0ce279434a74f16047f577f31cd7aaafc38e12deac945b5a3632d89ed2c2e61.jpg)

C  
![](images/5cbd06081aea23b53aba73470c94bb698fe70bf5b8cd0b9f5994918e5f6384a2.jpg)

![](images/95571091dfd989875b565ffd48245dce29ca6194dfe2980557844ae169c62c66.jpg)  
Figure 1: Missing butterfly and forecast-backcast asymmetry in ERA5 and GCM AIWP models. (A) Schematic of the autoregressive forecasting and backcasting models. The forecast network $N N _ { \theta }$ advances the atmospheric state (Eq. (1)); an independently trained backcast network $N N _ { \phi }$ steps it backward (Eq. (2)). (B) Anomaly correlation coeficient (ACC) of Z500 versus lead time for ERA5 (left; Transformer (deterministic), based on Pangu-Weather [2]) and the PlaSim GCM (right; Transformer and conditional Difusion (stochastic), similar to GenCast [4]), averaged over all initial conditions from the test sets. $\Delta t = 2 4 \mathrm { ~ h ~ }$ h and the horizontal resolution is ∼ 100 km (ERA5) and ∼ 300 km (PlaSim). Shading marks the $\mathrm { A C C } > 0 . 6$ skill window; annotations give the crossing times and their forecast/backcast ratio, the asymmetry. See Figs. S1-S4 for other variables, metrics, Δ�, and neural-operator and generative architectures. (C) Diference kinetic energy (DKE) for perturbations of diferent amplitudes, decreasing from a reference value, � . We also show DKE from numerical integrations of physics-based models (ICON 20 km and PlaSim GCM). See Methods and Data for details about the AIWP models, physics-based models, perturbations, and metrics.

A  
![](images/81a0c46c88dc5c973aa928938eaa9c5ed8e9b4d331869cb57d119887bf1116d3.jpg)

![](images/f37b1dee8c65660f1c54f2552f8d142e750545857f175972021099bf736fcc8e.jpg)

![](images/eca54ec5f35de9da6be35bc7adeaa1386918ee624aa54395463f0c016952b8ed.jpg)

B  
![](images/22c8c28b6730610d2ef3d66a922e8a44fa45f985efebbffe5cf464f0d47c3add.jpg)

![](images/9588f83d93096396058aec83fa37342788c718aac2f32ffeee4665224bbce442.jpg)

![](images/de0d4dca0cfa2cf53291a5e030611697da62d334f000a22434a2bad924b89f4b.jpg)

![](images/9f2948f43a4e9fab5f716e13b8b22bcbd5c1cec64e6f523100c11c8f398cd249.jpg)

![](images/c51ea448778152aa8eb3fe165ad57f690c2c4f4dd4d9f8c200f131debc8d4a95.jpg)

![](images/dbb7e1863976956ab2cb17063bee203c7bb5fdba86deab9ae9ce0f21e0b07f66.jpg)

C  
![](images/c7b795da91606a2bab60149e723c27c757d219b63911ccbc956caa413fedd0c3.jpg)

![](images/f6545ad305cc83a415f95a2942be68228a7eef45cb19d74cb4de281e65dbc916.jpg)

![](images/d78644a8caaa7a3b7a6e41e535c739d854324bc6b6abe64e2111a1ca06f55248.jpg)

![](images/167b48d871baa42879d7ec0b945c5955f74904f05c0f696c745f7c5c4e7a03bb.jpg)

![](images/73cf25f7b9dbda5f601d015c92f84b57cd44ba08efbcc160f3db773bf29a4b7f.jpg)

![](images/fb2eaeae8254b7bacbca231add9b746ca2bc4b97268498a5612a41b26da8cdff.jpg)  
Figure 2: Efects of training data coarse-graining on forecasting and backcasting AI models of Lorenz 96. All results (RMSE, asymmetry, and Hovmoller diagrams) are based on the large-scale¨ variable, $X _ { i } ( i \ : = \ : 1 \ : . \ : . 8 )$ . (A) The numerical-model reference trajectory alongside forecasts and backcasts from numerical integrations of Eqs. (S7)-(S8) initialized at lead time $0 ;$ saturated dark red/blue regions mark numerical blow-up in backcasts. Single floating-point precision calculations are shown to match the precision used in AI models’ inference; it does not make any noticeable diference. (B) and (C) show RMSE curves (averaged over 100 initial conditions from the test set) and Hovmoller diagrams (from the same initial condition as in (A)) for 12 independent AI¨ models trained on only the large-scale $X _ { i }$ and on both scales $( X _ { i }$ and $Y _ { i , j } \left( j = 1 \ldots 3 2 \right) \stackrel { - } { ( }$ ) with time interval $\Delta t = 1 0 \delta t , 5 \delta t$ , and 1�� (�� is the time step of the numerical solver). Shading indicates the RMSE interquartile range. Annotation boxes report forecast RMSE at 4.5 days and the asymmetry (backcast/forecast RMSE ratio) at 1.5 days. Note the diference between the RMSE ranges in rows (B) and (C), as well as between the lead-time ranges, chosen for clearer illustration. See Methods and Data for more details of the system and AI models.

Note that the missing butterfly efect in the real-world-regime AI model of two-scale Lorenz 96 and in the AIWP models of ERA5 and PlaSim (Figs. 1C and 3) does not mean that these models miss chaos. Their predictions are sensitive to initial-condition perturbations across perturbation amplitudes. In fact, their amplitude-independent exponential DKE growth closely resembles that of numerical integrations of Lorenz 63 (Fig. S5), the hallmark example of deterministic chaos, but itself misses the real butterfly efect because it lacks multi-scale dynamics (see Palmer [21, 22] for an illustrative discussion of Lorenz 63 versus Lorenz 96).

Next, we develop additional (independent) forecasting and backcasting AI models trained on data with more spatial and/or temporal scales: using � or (�, �) and Δ� = 1, 5, or 10 �� (see Methods and

Data for details of extensive model design exploration). At the bottom-right corner of Figs. 2B-C and 3, we show the results of the AI models trained in the “perfect-data” regime: these models have seen all the scales the numerical solution/ground truth contains. Compared with the models trained in the $\mathrm { \ddot { \ s } _ { r e a l - w o r l d } \mathrm { \vec { \Omega } _ { \mathrm { ~ } } , } }$ regime, the AI models trained in the perfect-data regime have a worse forecast skill (by a factor of $\sim 7 )$ and larger asymmetry (by a factor of ∼ 4) but also blown-up backcasting and rapid DKE growth that closely mimics the butterfly efect. The last two in particular show that the perfect-data-regime AI models are consistent with the fundamental understanding of multi-scale chaotic systems and their numerical integrations.

We observe these trends in forecast skill, asymmetry, DKE growth, and Hovmoller diagrams con-¨ sistently as we move from the “real-world” regime corner to the “perfect-data” regime corner. Therefore, we propose the following three hypotheses supported by these Lorenz 96 experiments: coarse-graining used in producing the training data is the common cause of H1) Unexpected AI forecast skill (for large scales), H2) Missing butterfly efect, and H3) Skillful backcasting. We broadly define coarse-graining, which is inevitable in practice, as filtering of small/fast scales and/or ignoring some state variables.

The emergence of H1 from coarse-graining can be further seen in the analysis of leading finite-time Lyapunov exponents of the large and small scales, $\lambda ^ { X }$ and $\lambda ^ { Y }$ (Table S2). In the ground truth, $\lambda ^ { X }$ and $\setminus { \lambda ^ { Y } }$ are the same (within the uncertainty range), reflecting the fact that the error growth in the large scales is determined by the small scales. The perfect-data-regime AI model reproduces these Lyapunov exponents fairly well. However, the real-world-regime AI model, which lacks small scales, has $\mathbf { \chi } ^ { \bullet } \mathbf { \chi } ^ { X }$ around 8 times smaller than the ground truth/perfect-data-regime AI model, leading to its enhanced forecast skills. We note that here, H1 refers to the skill for large scales, which is relevant for practical forecasting and the basis of common verification metrics (see Discussion).

As for H2, Fig. 3 shows the dramatic impact of training-data coarse-graining on the DKE growth of the 12 AI forecasting and backcasting models, which share an identical architecture and MSE loss. In particular, when trained on complete or nearly complete data, these MSE-trained models exhibit a butterfly-like efect: rapid, amplitude-dependent DKE growth comparable to, or even faster than, that of the numerical integrations (Fig. 3B). This argues against the suggestion that optimizing toward a conditional mean is the primary cause of the missing butterfly efect (see the Discussion). These results raise the question of whether a butterfly-like efect can likewise emerge in a state-of-the-art AIWP model when the coarse-graining of its training data is reduced, which we address next.

## 2.4 As coarse-graining is reduced, a butterfly-like efect emerges and forecast skill degrades: Pangu-Weather

We use the oficial Pangu-Weather models [2]: four independent deterministic neural networks with time steps $\Delta t = 2 4 .$ , 6, 3, and 1 h, all trained with an MAE loss on $0 . 2 5 ^ { \circ }$ ERA5, the native spatial resolution of the dataset. We only perform inference (forecasts) with these models; we have not trained, fine-tuned, or otherwise modified them. Figure 4A shows that as Δ� decreases, the small-amplitude ensembles transition from slow, amplitude-independent exponential DKE growth for the 24-h Pangu-Weather, consistent with [19, 23] and our $1 ^ { \circ }$ Transformer (Fig. 1C), to rapid, amplitude-dependent growth for the 1-h Pangu-Weather that approaches the butterfly-like behavior of the ICON integrations. Meanwhile, the skillful lead time drops from 9.3 to 2.0 days. Consistent with the Lorenz 96 experiments (Figs. 2-3) and hypotheses H1 and H2, reducing (here, temporal) coarse-graining degrades forecast skill and strengthens butterfly-like growth.

This behavior has not been reported before because the $\Delta t = 1$ h and 3 h Pangu-Weather models had not been run continuously. Selz and Craig [19] used all four networks, but through hierarchical temporal aggregation, in which the $\Delta t = 2 4$ h model takes the longest steps and the shorter-Δ� models only fill the intermediate hours; the drops in DKE at network switches that they reported, largest when the 24-h model takes over, are consistent with Fig. 4A. Their follow-up study [23] ran only the 6- and 24-h versions.

The spectra of the ensemble perturbations (Fig. 4B) further characterize the emerging DKE growth. For $\Delta t = 2 4$ and 6 h, the small-amplitude spectra jump within the first step and then grow slowly, with their shape roughly preserved, remaining orders of magnitude below the ERA5 spectrum even at 3 days. For $\Delta t = 1 \mathrm { ~ h ~ }$ , perturbation energy instead grows fastest at the small scales, saturates there first, and progressively fills in the larger scales toward the ERA5 spectrum: the upscale error growth expected in multi-scale chaotic systems [20, 21]; see Fig. S6. However, at the smallest scales, the late-time spread overshoots the ERA5 spectrum, beyond the factor of 2 allowed by physical saturation; the 1-h model generates variance at scales that ERA5 itself barely contains, a warning sign discussed in Methods and Data.

![](images/d059c3a9a93d0a8a948b6f7156a9a3b565c1cf1f06e8c5b602f194d7c77247f2.jpg)  
Figure 3: Efects of training-data coarse-graining on ensemble growth and the butterfly efect in AI models of Lorenz 96. Curves show DKE of the large-scale variable, $X _ { i } ( i = 1 \ldots 8 )$ for 12 independent, separate AI models that have an identical architecture (including the same MSE loss) trained on (A) only the large-scale $X _ { i }$ and (B) on both scales $( X _ { i }$ and $Y _ { i , j } \left( j = 1 \ldots 3 2 \right)$ ) with time interval $\Delta t = 1 0 \delta t$ , 5��, and 1�� (�� is the time step of the numerical solver). Gray curves show forecasts and backcasts from numerical integrations of the two-scale Lorenz 96 equations (S7)– (S8); crosses mark numerical blow-up in backcasts. Curves are averaged over the same 100 initial conditions from the test set. Perturbations are of diferent amplitudes, decreasing from a reference value, $\epsilon _ { 0 } .$ See Methods and Data for details about the perturbations and metrics.

We emphasize that we are not claiming that this is the (real) butterfly efect. Even at its native resolution, ERA5 (and especially its subset of variables used for training) is highly coarse-grained: the reanalysis is produced by a ∼31-km NWP model with parameterized subgrid processes, it misses state variables, and $\Delta t = 1$ h remains far longer than the time step of the NWP model’s numerical solver. The Lorenz 96 hierarchy shows what to expect in this situation: with small spatial scales fully missing (�-only), AI models overall remain in the real-world-regime behavior even when trained at the solver’s own time step (Fig. 3, top-right corner, where $\Delta t = 1 \delta t )$ . Even in the perfect-data regime, we call the behavior in Fig. 3 butterfly-like: the DKE grows faster than the numerical model’s near $t = 0$ , whereas the real-world-regime AI models match the early numerical growth rates better but never accelerate as time progresses forward or backward. Returning to the Pangu-Weather model with $\Delta t = 1$ h, unlike what numerical experiments show [33, 34, 19], the largest early DKE growth is not necessarily collocated with precipitation (Fig. S7). However, we ofer a word of caution and argue that assessing the butterfly efect, and more broadly, the physicality of error growth, in AI models requires deeper investigation and a revisiting of what these terms mean, as the current understanding heavily relies on numerical experiments, with their own known shortcomings. Disagreement with a numerical model does not, by itself, make an AIWP model wrong: these data-driven models might be learning, and forecasting, weather diferently.

One must also be cautious about hardware artifacts [23] and nonphysical forecasts while examining butterflies in AI models; see Methods and Data for how these are controlled for and assessed. What is robust here is the trade-of: within a single family of state-of-the-art AIWP models, reducing temporal coarse-graining results in butterfly-like growth and degrades forecast skill, further supporting H1 and H2.

A  
![](images/a053148ae21a6f13e027df501b286de4b554326915250623d91463194d4489b4.jpg)  
B

![](images/4fa1c9c10af9014b9d4dc6ae99b314806dc14a11098b6c7da2eefeeb92c09f81.jpg)  
Figure 4: Efects of temporal coarse-graining of training data on ensemble growth and the butterfly efect in the Pangu-Weather forecasts. Columns show, from left to right, forecasts from the independent, separate oficial Pangu-Weather models [2] with time step $\Delta t = 2 4 , 6 , 3$ , and 1 h. (A) DKE versus lead time, as in Fig. 1C, for three amplitudes of initial perturbations. DKE is averaged over 60<sup>o</sup>S-60<sup>o</sup>N. See Fig. S10 for results with global averaging and lower GPU precision. Gray curves show the DKE from ICON forecasts (from [19]). The number in each panel is the skillful lead time, $\mathrm { i . e . }$ , when ACC drops to 0.6, as in Fig. 1B. (B) Power spectra $S _ { \nu } ( n )$ of the 300-hPa meridional wind perturbation $\nu _ { 3 0 0 } ^ { \prime }$ (each member minus ensemble mean) as a function of total wavenumber $n ,$ for the initial-condition perturbations with the smallest $( \epsilon _ { 0 } / 1 0 ^ { 3 }$ , upper row) and largest $( \epsilon _ { 0 } ,$ , lower row) amplitudes. These curves are averaged over the same ensemble members and initial conditions as the ERA5 Transformer in Fig. 1C. Colors denote (nonuniform) lead times from +1 to +72 h; the gray curve is the imposed initial Perlin noise perturbation at $t = 0 ,$ , and the black curve is the spectrum of the full ERA5 field for reference. The top axis denotes the corresponding wavelength. Figure S6 shows the spectra of the forecast errors as a function of lead time. See Methods and Data for more information.

## 3 Discussion

Across the hierarchy of ERA5, PlaSim GCM, and Lorenz 96, we find that the coarse-graining of the training data is the common cause of the following (seemingly unrelated) key features of AI prediction models:

H1. The surprising forecast accuracy,

H2. The missing butterfly efect,

H3. The skillful backcasting.

In short, the more completely the training data represent the true dynamical system spatially and temporally, the more the AI model behaves like the system, but the worse it forecasts the large scales, which are often the desired target of prediction and most verification metrics.

Before discussing H1-H3 and their implications, we note that these results ofer a step toward answering a central question about AIWP models: Do they learn atmospheric dynamics, or merely learn the evolution of weather the way one learns a video? A video plays backward as easily as forward. Skillful backcasting (H3), taken alone, might appear to support this video interpretation, but only in the strongly coarse-grained regime characteristic of the AIWP models’ common training datasets. As the training data become more complete, the same architecture, trained with the same loss on pairs one time step apart, becomes markedly more physics-like: backcasting skill disappears, the forecast–backcast asymmetry grows, and a butterfly-like efect emerges. We are not suggesting that AIWP models trained on strongly coarse-grained data (e.g., ERA5 with Δ� = 24 h) learn weather much like a video: dynamical tests and other analyses have demonstrated that they learn aspects of atmospheric dynamics [14, 15, 35–38]. Our findings, however, show that with more complete training data, these models behave more like the physical system, including its arrow of time, although they become less accurate for practical forecasting. A major question then is whether there is a downside to having AIWP models that are more accurate but less physics-like; we return to this later.

Regarding H1, coarse-graining removes the fast, small scales through which errors grow upscale to the large scales [20, 28, 33, 34]. Therefore, the efective system that the AI models learn has much slower large-scale error growth than the true system (quantified for Lorenz 96 via Lyapunov exponents; Table S2). That lower spatio-temporal resolution improves forecast accuracy might seem counterintuitive, but this intuition is built on numerical simulations, in which higher resolution improves accuracy. This is because of lower discretization errors and, more importantly, reduced reliance on “explicit” subgrid-scale parameterization schemes, the leading source of structural/epistemic uncertainty in these models. AI models, in contrast, learn subgrid-scale parameterizations “implicitly”: they can account for the averaged efects of the fast, small scales on the large scales’ evolution without explicitly representing the former, which can otherwise drive rapid error growth. Consistent with this implicit parameterization, AI climate emulators, trained on coarse-grained outputs of high-resolution physics-based simulations can closely reproduce key climate statistics of the original simulations [39]. Moreover, even in NWP models, increasing resolution yields diminishing returns in large-scale forecast accuracy [40], in part because the newly resolved fast, small scales drive rapid error growth [29, 34, 41]. Taken together, we argue that this implicit parameterization in AIWP trained with coarse-grained data significantly enhances their forecast skill and is a major reason fo their surprising accuracy.

We should clarify that the surprising accuracy is not necessarily about comparison with the NWP models. Such a comparison would have to account for the quality of the physics-based models subgrid-scale parameterization schemes, a complex question. Rather, H1 concerns the AIWP models’ skill, which is unexpected given earlier studies arguing that data-driven weather forecasting requires an enormous amount of training data [24], with $\mathbf { \bar { 1 0 ^ { 3 0 } } }$ years as one estimate [25]. Our findings suggest that such estimates need to be revisited in light of the efects of AIWP models training-data coarse-graining on error growth and the butterfly efect, key contributors to those high data requirements. Other factors likely contribute to the gap between these estimates and practice, including the underlying assumptions, e.g., about the attractor dimension [42, 43], and, perhaps more importantly, the learning algorithm. Analog forecasting in those studies is essentially nearestneighbor regression, whose scaling sufers from the curse of dimensionality [44, 45]. Current AIWP models instead use deep neural networks (over-parameterized nonlinear regression) to learn a smooth Δ� flow map [46–48]. Furthermore, AIWP models have been shown to “translocate”, i.e., learn from dynamically similar events in other regions, which significantly enrich their training set [15, 14]. Developing theoretical scaling laws relating training-data size to AIWP model accuracy, beyond emerging empirical estimates [49, 50], remains an important open research direction and can build on our findings.

As for the missing butterfly efect (H2), it is a symptom of coarse-grained training data, not of the architecture, the loss function, or deterministic-versus-generative design. The Lorenz 96 experiments isolate the cause cleanly (Fig. 3): all 12 networks share one architecture and one MSE loss, yet the butterfly efect appears or disappears with the content of the training data alone. The Pangu-Weather experiments with varying Δ� (Fig. 4) show the same dependence in a state-of-the-art AIWP model. Spatially, however, ERA5 (and PlaSim) are already highly coarse-grained at their native resolutions, leaving no room to informatively further vary the spatial scales in either direction; future explorations should focus on high-resolution global NWP models and GCMs. These findings argue against optimizing toward a conditional mean or blurring, which is due in part to spectral bias [17], as the primary cause: MSE-trained networks reproduce the butterfly efect in Lorenz 96, while the

Difusion AIWP models of PlaSim studied here and GenCast and FourCastNet2 examined in [23] miss butterflies despite their little spectral bias (Figs. 1C, S2, and S8).

Note that after showing the lack of butterfly efect across AIWP models of diferent architectures and loss functions, [23] suggested a training-data origin, the remaining common factor (although all these AIWP models also share the same learning principle: supervised prediction of the short-Δ� evolution). However, the problem has been attributed to the specific nature of the data-assimilated analysis products used for training and fine-tuning [23, 22]. The hierarchy here, whose PlaSim and Lorenz 96 training sets involve no data assimilation, tests the training-data origin directly and identifies the key property: its spatial and temporal coarse-graining. Our results also suggest that restoring butterflies can come at the cost of forecast accuracy; still, if one views their absence as a bug rather than a feature of accurate AIWP models, the question becomes how best to restore them (see below).

Skillful backcasting (H3) is made possible by the same coarse-graining. The learned efective system is smoother, with smaller leading Lyapunov exponents and less of the fast, small scales that dominate the anti-dissipative directions responsible for explosive backward integration (see Supplementary Materials). The apparent violation of the second law of thermodynamics is thereby resolved: the learned efective system is less irreversible than the true one, but still irreversible. This residua irreversibility is manifested in the asymmetry remaining above one, which is largest in the tropics, consistent with the higher entropy production there [31, 32], and in preliminary explorations in which long-term AI backcasting of Lorenz 96, PlaSim, and ERA5 leads to unstable or unphysical states, a sign of irreversibility, though much slower than the true system.

This work introduces backcasting as a new, useful lens for investigating the fundamentals of predictability and better understanding AIWP models. Backcasting can also provide practical tools of its own. Speculatively, a single AI model trained both forward and backward might internalize more of the underlying dynamics, perhaps even rudiments of causality. That said, our preliminary bidirectional forecasting-backcasting model (Methods and Data) shows no improvement so far, but the design space is essentially unexplored. Backcast models might also help sample the rarest weather extremes and find their dynamical precursors, tasks traditionally addressed with limited linear tools such as adjoints. AI-based approaches have recently emerged, including AI-boosted rare-event sampling [51], iterative use of AIWP models’ adjoints [52], and sampling emulators like cBottle [53, 54]. Backcasting could augment these as a stable, fully nonlinear, autoregressive counterpart of the adjoints.

Growing eforts focus on identifying the physics that AI models miss [14, 16–19, 23, 55] and on making them more physics-like, e.g., through inductive biases such as constraints in architectures and loss functions [56–59]. Our results prompt a question: do we actually want AIWP models to fully behave like physics-based models even if restoring the physics costs performance, e.g., accuracy of practical forecasting (Figs. 2 and 4)? This trade-of may prove solvable, and there is precedent for excluding physics for the benefit of practical skill: sound waves, computationally costly to resolve yet largely irrelevant to weather, have been filtered out of the equations solved by many NWP models and GCMs [60]. However, the distinction is that while sound waves carry little of what matters for weather and climate prediction, the fast, small scales filtered by coarse-graining carry much of it. Thus, the deeper question is what these AI models are missing because coarse-graining has removed physics from their training data. Is this absence part of why AIWP models struggle with gray swans [14, 16]? What does it imply for probabilistic forecasts, especially at subseasonal-to-seasonal lead times, which rely on perturbation growth and must be well calibrated [11, 61]? The question might be even more pressing for long-term climate emulators: AI models of the emerging global km-scale, physics-based simulations [62] are typically trained on output first heavily coarse-grained, e.g., to ∼100 km [39], to Δ� of hours to a day, and even to daily-averaged state variables [13], which can discard significant physics. Consistent with this concern, idealized data-driven models with partial state vectors (missing variables) fail to reproduce forced responses [63]. Such failures ofer a caution for emulators built for sampling internal variability and projecting climate change [56]. Answering these questions requires a deeper understanding of what AI weather and climate models learn, do not learn, and why.

To conclude, the development of AI weather (and climate) models has concentrated on architectures and objectives: deeper and larger networks, customized loss functions, and embedded physical constraints. Those directions address the optimization and expressiveness of AI models. Our results point instead to the ingredient that no architecture or loss function can substitute for: the information content of the training data. Two remedies might come to mind immediately: hybrid AI-physics modeling, and foundation modeling, in which heterogeneous datasets are combined through largescale pretraining. However, state-of-the-art examples of each, NeuralGCM [5] and Aurora [64], miss the butterfly efect [23]; restoring at least this feature appears to require other approaches. Coarsegraining does entail an irreversible loss of information, yet some of what is lost can, in principle, be represented rather than resolved, as memory and stochasticity: the Mori–Zwanzig formalism [65, 66] and its data-driven descendants [67–69] (see Supplementary Materials) might provide a principled starting point. For predicting the large scales of weather days ahead, coarse-graining is a feature, the source of AIWP skill and of learnable backcasting; for representing the physics of the atmosphere, its error growth, irreversibility, and predictability limits, it is a bug, likely most consequentially for long-term emulation. The content of the training data is therefore not a preprocessing detail but the central design decision of an AI weather or climate model.

## Acknowledgments and Disclosure of Funding

We thank Kerry Emanuel and Fabrizio Falasca for insightful discussions about atmospheric predictability and time reversal, and Boris Bonev and Daniel Boscu for helpful comments on GPU precision. Computational resources were provided by NSF ACCESS (allocation ATM170020), NCAR’s CISL (allocations URIC0009 and UCHI0018), and the University of Chicago’s Research Computing Center and Data Science Institute (DSI). Claude Fable 5 has been used for editing and proofreading the text, and for generating the visualization code for Movie S1.

Funding: This work was supported by NSF grant AGS-2531264, Schmidt Sciences LLC (through the InMOS project), and the University of Chicago’s Data Science Institute (DSI) and the Institute for Climate and Sustainable Growth (through the AI for Climate Initiative) to PH. AW and JF are grateful to DSI for an Eric and Wendy Schmidt AI in Science and an AI for Climate postdoctoral fellowship, respectively. JQW was supported by a Pritzker AI+Science Visiting Scholarship from DSI.

Author contributions: PH conceived the idea and wrote the paper. PH and YQS supervised research. PH, YQS, and JQW designed the study. WL and YQS (ERA5 and PlaSim) and JW (Lorenz systems) trained AI models, analyzed data, generated the figures, and wrote the Methods and Data section. AW developed AI models and generated the PlaSim data. All authors interpreted the result and edited the paper.

Competing interests: There are no competing interests to declare.

Data and code availability: We downloaded the ERA5 dataset from Copernicus https: //cds.climate.copernicus.eu/ and regridded it with CDO https://code.mpimet.mpg. de/projects/cdo/. The ICON data are provided by [19]. The PlaSim GCM is available at https://github.com/HartmutBorth/PLASIM. The oficial Pangu-Weather models are obtained from https://github.com/198808xc/Pangu-Weather. Movie S1 is available at https: //doi.org/10.5281/zenodo.22062019. We will make all trained AI models and their inference data public upon publication.

## References

[1] Jaideep Pathak, Shashank Subramanian, Peter Harrington, Sanjeev Raja, Ashesh Chattopad hyay, Morteza Mardani, Thorsten Kurth, David Hall, Zongyi Li, Kamyar Azizzadenesheli, Pedram Hassanzadeh, Karthik Kashinath, and Animashree Anandkumar. FourCastNet: A global data-driven high-resolution weather model using adaptive Fourier neural operators. arXiv preprint arXiv:2202.11214, 2022. doi: 10.48550/arXiv.2202.11214.

[2] Kaifeng Bi, Lingxi Xie, Hengheng Zhang, Xin Chen, Xiaotao Gu, and Qi Tian. Accurate medium-range global weather forecasting with 3d neural networks. Nature, 619(7970):533– 538, 2023.

[3] Remi Lam, Alvaro Sanchez-Gonzalez, Matthew Willson, Peter Wirnsberger, Meire Fortunato, Ferran Alet, Suman Ravuri, Timo Ewalds, Zach Eaton-Rosen, Weihua Hu, et al. Learning skillful medium-range global weather forecasting. Science, 382(6677):1416–1421, 2023.

[4] Ilan Price, Alvaro Sanchez-Gonzalez, Ferran Alet, Tom R Andersson, Andrew El-Kadi, Dominic Masters, Timo Ewalds, Jacklynn Stott, Shakir Mohamed, Peter Battaglia, et al. Probabilistic weather forecasting with machine learning. Nature, 637(8044):84–90, 2025.

[5] Dmitrii Kochkov, Janni Yuval, Ian Langmore, Peter Norgaard, Jamie Smith, Grifin Mooers, Milan Klower, James Lottes, Stephan Rasp, Peter D¨uben, et al. Neural general circulation¨ models for weather and climate. Nature, 632(8027):1–7, 2024.

[6] Stephan Rasp, Stephan Hoyer, Alexander Merose, Ian Langmore, Peter Battaglia, Tyler Russell, Alvaro Sanchez-Gonzalez, Vivian Yang, Rob Carver, Shreya Agrawal, et al. Weatherbench 2: A benchmark for the next generation of data-driven global weather models. Journal of Advances in Modeling Earth Systems, 16(6):e2023MS004019, 2024.

[7] Rajat Masiwal, Colin Aitken, Adam Marchakitus, Mayank Gupta, Katherine Kowal, Hamid A Pahlavan, Tyler Yang, Y Qiang Sun, Michael Kremer, Amir Jina, et al. Decision-oriented benchmarking to transform ai weather forecast access: Application to the indian monsoon. arXiv preprint arXiv:2602.03767, 2026.

[8] Hans Hersbach, Bill Bell, Paul Berrisford, Shoji Hirahara, Andras Hor´ anyi, Joaqu´ ´ın Munoz-˜ Sabater, Julien Nicolas, Carole Peubey, Raluca Radu, Dinand Schepers, Adrian Simmons, Cornel Soci, Saleh Abdalla, Xavier Abellan, Gianpaolo Balsamo, Peter Bechtold, Gionata Biavati, Jean Bidlot, Massimo Bonavita, Giovanna De Chiara, Per Dahlgren, Dick Dee, Michail Diamantakis, Rossana Dragani, Johannes Flemming, Richard Forbes, Manuel Fuentes, Alan Geer, Leo Haimberger, Sean Healy, Robin J. Hogan, El´ıas Holm, Marta Janiskov´ a, Sarah´ Keeley, Patrick Laloyaux, Philippe Lopez, Cristina Lupu, Gabor Radnoti, Patricia de Rosnay, Iryna Rozum, Freja Vamborg, Sebastien Villaume, and Jean-Noel Th¨ epaut. The ERA5 global´ reanalysis. Quarterly Journal of the Royal Meteorological Society, 146(730):1999–2049, 2020. ISSN 1477-870X.

[9] Eric G Daub, Tom Dunstan, Thusal Bennett, Matthew Burnand, James Chappell, Alejandro Coca-Castro, Noushin Eftekhari, J Scott Hosking, Manvendra Janmaijaya, Jon Lillis, et al. Technical overview and architecture of the fastnet machine learning weather prediction model, version 1.0. arXiv preprint arXiv:2509.17658, 2025.

[10] Gabriel Moldovan, Ewan Pinnington, Ana Prieto Nemesio, Simon Lang, Zied Ben Bouallegue,\` Jesper Dramsch, Mihai Alexe, Mario Santa Cruz, Sara Hahner, Harrison Cook, et al. Aifs single 1.1. 0: an update to ecmwf’s machine-learned weather forecast model aifs. Geoscientific Model Development, 19(10):4703–4724, 2026.

[11] Colin Aitken, Rajat Masiwal, Adam Marchakitus, Katherine Kowal, Mayank Gupta, Tyler Yang, Amir Jina, Pedram Hassanzadeh, William R Boos, and Michael Kremer. Designing probabilistic ai monsoon forecasts to inform agricultural decision-making. arXiv preprint arXiv:2603.07893, 2026.

[12] Oliver Watt-Meyer, Brian Henn, Jeremy McGibbon, Spencer K Clark, Anna Kwa, W Andre Perkins, Elynn Wu, Lucas Harris, and Christopher S Bretherton. Ace2: accurately learning subseasonal to decadal atmospheric variability and forced responses. npj Climate and Atmospheric Science, 8(1):205, 2025.

[13] Brian Henn, Christopher S Bretherton, Nikolay Kodunov, Christian Lessig, Maria J Molina, Troy Arcomano, Oliver Watt-Meyer, Guillaume Couairon, Renu Singh, Robert Brunstein, et al. AIMIP Phase 1: systematic evaluations of AI weather and climate models. arXiv preprint arXiv:2605.06944, 2026.

[14] Y Qiang Sun, Pedram Hassanzadeh, Mohsen Zand, Ashesh Chattopadhyay, Jonathan Weare, and Dorian S Abbot. Can ai weather models predict out-of-distribution gray swan tropical cyclones? Proceedings ofthe National Academy ofSciences, 122(21):e2420914122, 2025.

[15] Y Qiang Sun, Pedram Hassanzadeh, Tifany Shaw, Hamid A Pahlavan, and Adam Marchakitus. Predicting regional gray swans via translocation: AI weather models and Dubai’s unprecedented 2024 rainfall. Science Advances (in press), 2026. doi: arXiv:2505.10241.

[16] Zhongwei Zhang, Erich Fischer, Jakob Zscheischler, and Sebastian Engelke. Physics-based models outperform ai weather forecasts of record-breaking extremes. Science Advances, 12 (18):eaec1433, 2026.

[17] A Chattopadhyay, Y Q Sun, and P Hassanzadeh. Challenges of learning multi-scale dynamics with ai weather models: Implications for stability and one solution. arXiv preprint arXiv:2304.07029, 2023.

[18] Massimo Bonavita. On some limitations of current machine learning weather prediction models. Geophysical Research Letters, 51(12):e2023GL107377, 2024.

[19] Tobias Selz and George C Craig. Can artificial intelligence-based weather prediction models simulate the butterfly efect? Geophysical Research Letters, 50(20):e2023GL105747, 2023.

[20] Edward N Lorenz. The predictability of a flow which possesses many scales of motion. Tellus, 21(3):289–307, 1969.

[21] Tim N Palmer, Andreas Doring, and Gregory Seregin. The real butterfly efect. ¨ Nonlinearity, 27(9):R123–R141, 2014.

[22] Tim Palmer. The real butterfly efect and maggoty apples. Physics Today, 77(5):30–35, 2024.

[23] T Selz and GC Craig. Can ai-based weather prediction models simulate the butterfly efect? the role of architecture and implementation. Journal of Geophysical Research: Machine Learning and Computation, 3(3):e2025JH001180, 2026.

[24] Edward N Lorenz. Atmospheric predictability as revealed by naturally occurring analogues. Journal ofAtmospheric Sciences, 26(4):636–646, 1969.

[25] HM Van den Dool. Searching for analogues, how long must we wait? Tellus A, 46(3):314–324, 1994.

[26] Edward N Lorenz. Predictability: A problem partly solved. In Proc. Seminar on predictability, volume 1, pages 1–18, 1996.

[27] Dale R Durran and Mark Gingrich. Atmospheric predictability: Why butterflies are not of practical importance. Journal ofthe Atmospheric Sciences, 71(7):2476–2488, 2014.

[28] Y Qiang Sun and Fuqing Zhang. Intrinsic versus practical limits of atmospheric predictability and the significance of the butterfly efect. Journal of the Atmospheric Sciences, 73(3): 1419–1438, 2016.

[29] Fuqing Zhang, Y Qiang Sun, Linus Magnusson, Roberto Buizza, Shian-Jiann Lin, Jan-Huey Chen, and Kerry Emanuel. What is the predictability limit of midlatitude weather? Journal ofthe Atmospheric Sciences, 76(4):1077–1091, 2019.

[30] P Trent Vonich and Gregory J Hakim. Atmospheric predictability beyond 30 days with machine learning. Artificial Intelligencefor the Earth Systems, 5(3):260009, 2026.

[31] Jose Pinto Peixoto, Abraham H Oort, M´ ario De Almeida, and Ant´ onio Tom´ e. Entropy budget´ of the atmosphere. Journal of Geophysical Research: Atmospheres, 96(D6):10981–10988, 1991.

[32] Olivier Pauluis and Isaac M Held. Entropy budget of an atmosphere in radiative–convective equilibrium. part ii: Latent heat transport and moist processes. Journal of the Atmospheric Sciences, 59(2):140–149, 2002.

[33] Fuqing Zhang, Chris Snyder, and Richard Rotunno. Efects of moist convection on mesoscale predictability. Journal ofthe Atmospheric Sciences, 60(9):1173–1185, 2003.

[34] Tobias Selz and George C Craig. Upscale error growth in a high-resolution simulation of a summertime weather event over europe. Monthly Weather Review, 143(3):813–827, 2015.

[35] Senne Van Loon, Maria Rugenstein, and Elizabeth A Barnes. Reanalysis-based global radiative response to sea surface temperature patterns: Evaluating the ai2 climate emulator. Geophysical Research Letters, 52(14):e2025GL115432, 2025.

[36] Elynn Wu, Finn Rebassoo, Pappu Paul, Cristian Proistosescu, Jacqueline Nugent, Daniel McCoy, Peter Caldwell, and Christopher S Bretherton. Applying the ace2 emulator to sst green’s functions for the e3smv3 global atmosphere model. Journal of Geophysical Research: Machine Learning and Computation, 2(3):e2025JH000774, 2025.

[37] Gregory J Hakim and Sanjit Masanam. Dynamical tests of a deep learning weather prediction model. Artificial Intelligence for the Earth Systems, 3(3), 2024.

[38] George Craig, Tobias Selz, Matthias Beylich, and Kirsten I Tempest. The physics of ai weather models. arXiv preprint arXiv:2605.23778, 2026.

[39] W Andre Perkins, Anna Kwa, Jeremy McGibbon, Troy Arcomano, Spencer K Clark, Oliver Watt-Meyer, Christopher S Bretherton, and Lucas M Harris. Hiro-ace: Fast and skillful ai emulation and downscaling trained on a 3 km global storm-resolving model. arXiv preprint arXiv:2512.18224, 2025.

[40] Roberto Buizza. Horizontal resolution impact on short-and long-range forecast error. Quarterly Journal ofthe Royal Meteorological Society, 136(649):1020–1035, 2010.

[41] Falko Judt. Insights into atmospheric predictability through global convection-permitting model simulations. Journal ofthe Atmospheric Sciences, 75(5):1477–1497, 2018.

[42] C Nicolis. Atmospheric analogs and recurrence time statistics: Toward a dynamical formulation. Journal ofthe Atmospheric Sciences, 55(3):465–475, 1998.

[43] Paul Platzer, Pascal Yiou, Philippe Naveau, Pierre Tandeo, J-F Filipot, Pierre Ailliot, and Yicun Zhen. Using local dynamics to explain analog forecasting of chaotic systems. Journal ofthe Atmospheric Sciences, 78(7):2117–2133, 2021.

[44] Charles J Stone. Optimal global rates of convergence for nonparametric regression. The annals ofstatistics, pages 1040–1053, 1982.

[45] Laszl´ o Gy´ orfi, Michael Kohler, Adam Krzy˙zak, and Harro Walk.¨ A distribution-free theory ofnonparametric regression. Springer, 2002.

[46] Andrew R Barron. Universal approximation bounds for superpositions of a sigmoidal function. IEEE Transactions on Information Theory, 39(3):930–945, 1993.

[47] Tomaso Poggio, Hrushikesh Mhaskar, Lorenzo Rosasco, Brando Miranda, and Qianli Liao. Why and when can deep-but not shallow-networks avoid the curse of dimensionality: a review. International Journal ofAutomation and Computing, 14(5):503–519, 2017.

[48] Mikhail Belkin, Daniel Hsu, Siyuan Ma, and Soumik Mandal. Reconciling modern machinelearning practice and the classical bias–variance trade-of. Proceedings of the National Academy of Sciences, 116(32):15849–15854, 2019.

[49] Yuejiang Yu, Langwen Huang, Alexandru Calotoiu, and Torsten Hoefler. Scaling laws of global weather models. arXiv preprint arXiv:2602.22962, 2026.

[50] Shashank Subramanian, Alexander Kiefer, Arnur Nigmetov, Amir Gholami, Dmitriy Morozov, and Michael W Mahoney. On neural scaling laws for weather emulation through continual training. arXiv preprint arXiv:2603.25687, 2026.

[51] Amaury Lancelin, Alexander Wikner, Laurent Dubus, Clement Le Priol, Dorian S Abbot, ´ Freddy Bouchet, Pedram Hassanzadeh, and Jonathan Weare. Ai-boosted rare event sampling to characterize extreme weather. Physical Review Letters, 137(6):064201, 2026.

[52] P Trent Vonich and Gregory J Hakim. Predictability limit of the 2021 pacific northwest heatwave from deep-learning sensitivity analysis. Geophysical Research Letters, 51(19): e2024GL110651, 2024.

[53] Noah D Brenowitz, Tao Ge, Akshay Subramaniam, Peter Manshausen, Aayush Gupta, David M Hall, Morteza Mardani, Arash Vahdat, Karthik Kashinath, and Michael S Pritchard. Climate in a bottle: Towards a generative foundation model for the kilometer-scale global atmosphere. arXiv preprint arXiv:2505.06474, 2025.

[54] Jerry Lin, Mu-Ting Chien, Mansi Sakarvadia, and Elizabeth A Barnes. Extremes on rewind: Generating 1,000-member ensembles initialized at a final condition. arXiv preprint arXiv:2608.19008, 2026.

[55] Hisu Kim, Jihun Ryu, Seok-Woo Son, Jee-Hoon Jeong, Hyungjun Kim, and Jin-Ho Yoon. A spectral test of the butterfly efect and physical consistency in the difusion-based gencast’s ensembles. npj Climate and Atmospheric Science, 9(1):110, 2026.

[56] Ching-Yao Lai, Pedram Hassanzadeh, Aditi Sheshadri, Maike Sonnewald, Rafaele Ferrari, and Venkatramani Balaji. Machine learning for climate physics and simulations. Annual Review ofCondensed Matter Physics, 16(1):343–365, 2025.

[57] Zongyi Li, Hongkai Zheng, Nikola Kovachki, David Jin, Haoxuan Chen, Burigede Liu, Kamyar Azizzadenesheli, and Anima Anandkumar. Physics-informed neural operator for learning partial diferential equations. ACM/IMS Journal of Data Science, 1(3):1–27, 2024.

[58] Yogesh Verma, Markus Heinonen, and Vikas Garg. Climode: Climate and weather forecasting with physics-informed neural odes. In International Conference on Learning Representations, volume 2024, pages 8408–8430, 2024.

[59] Yingkai Sha, John S Schreck, William Chapman, and David John Gagne. Improving ai weather prediction models using global mass and energy conservation schemes. Journal ofAdvances in Modeling Earth Systems, 17(11):e2025MS005138, 2025.

[60] Geofrey K Vallis. Atmospheric and oceanic fluid dynamics. Cambridge University Press, 2006.

[61] Anna Asch, Raphael Rossellini, Pedram Hassanzadeh, and Rebecca Willett. Rigorous uncertainty quantification of probabilistic ai weather forecasts with conformal prediction. arXiv preprint arXiv:2606.19642, 2026.

[62] Bjorn Stevens, Masaki Satoh, Ludovic Auger, Joachim Biercamp, Christopher S Bretherton, Xi Chen, Peter D¨uben, Falko Judt, Marat Khairoutdinov, Daniel Klocke, et al. Dyamond: the dynamics of the atmospheric general circulation modeled on non-hydrostatic domains. Progress in Earth and Planetary Science, 6(1):61, 2019.

[63] Fabrizio Falasca. Probing forced responses and causality in data-driven climate emulators: Conceptual limitations and the role of reduced-order models. Physical Review Research, 7 (4):043314, 2025.

[64] Cristian Bodnar, Wessel P Bruinsma, Ana Lucic, Megan Stanley, Anna Allen, Johannes Brandstetter, Patrick Garvan, Maik Riechert, Jonathan A Weyn, Haiyu Dong, et al. A foundation model for the earth system. Nature, 641(8065):1180–1187, 2025.

[65] Hazime Mori. Transport, collective motion, and brownian motion. Progress of theoretical physics, 33(3):423–455, 1965.

[66] Robert Zwanzig. Ensemble method in the theory of irreversibility. The Journal of Chemical Physics, 33(5):1338–1341, 1960.

[67] Jeroen Wouters and Valerio Lucarini. Multi-level dynamical systems: Connecting the ruelle response theory and the mori-zwanzig approach. Journal of Statistical Physics, 151(5):850– 860, 2013.

[68] Dmitri Kondrashov, Mickael D Chekroun, and Michael Ghil. Data-driven non-markovian¨ closure models. Physica D: Nonlinear Phenomena, 297:33–55, 2015.

[69] Fabrizio Falasca and Laure Zanna. Physics and causally constrained discrete-time neural models of turbulent dynamical systems. arXiv preprint arXiv:2602.13847, 2026.

[70] Klaus Fraedrich, Henk Jansen, E. Kirk, Uwe Luksch, and Frank Lunkeit. The planet simulator: Towards a user friendly model. Meteorologische Zeitschrift, 14(3):299–304, 2005. doi: 10.1127/0941-2948/2005/0043.

[71] Boris Bonev, Thorsten Kurth, Christian Hundt, Jaideep Pathak, Maximilian Baust, Karthik Kashinath, and Anima Anandkumar. Spherical fourier neural operators: Learning stable dynamics on the sphere. In International conference on machine learning, pages 2806–2823. PMLR, 2023.

[72] William Peebles and Saining Xie. Scalable difusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.

[73] Francesco Ragone, Jeroen Wouters, and Freddy Bouchet. Computation of extreme heat waves in climate models using a large deviation algorithm. Proceedings of the National Academy of Sciences, 115(1):24–29, 2018.

[74] Daniel S. Wilks. Efects of stochastic parametrizations in the Lorenz ’96 system. Quarterly Journal of the Royal Meteorological Society, 131(606):389–407, 2005. doi: 10.1256/qj.04.03.

[75] Tobias Thornes, Peter D¨uben, and Tim Palmer. On the use of scale-dependent precision in earth system modelling. Quarterly Journal of the Royal Meteorological Society, 143(703): 897–908, 2017. doi: 10.1002/qj.2974.

[76] Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama. Optuna: A next-generation hyperparameter optimization framework. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 2623– 2631, 2019. doi: 10.1145/3292500.3330701.

[77] Lisha Li, Kevin Jamieson, Giulia DeSalvo, Afshin Rostamizadeh, and Ameet Talwalkar. Hyperband: A novel bandit-based approach to hyperparameter optimization. Journal of Machine Learning Research, 18(185):1–52, 2018.

[78] Giancarlo Benettin, Luigi Galgani, Antonio Giorgilli, and Jean-Marie Strelcyn. Lyapunov characteristic exponents for smooth dynamical systems and for Hamiltonian systems; a method for computing all of them. Part 1: Theory. Meccanica, 15:9–20, 1980. doi: 10.1007/BF02128236.

[79] Alan Wolf, Jack B. Swift, Harry L. Swinney, and John A. Vastano. Determining Lyapunov exponents from a time series. Physica D: Nonlinear Phenomena, 16(3):285–317, 1985. doi: 10.1016/0167-2789(85)90011-9.

[80] TN Krishnamurti, K Rajendran, TSV Vijaya Kumar, Stephen Lord, Zoltan Toth, Xiaolei Zou, Steven Cocke, Jon E Ahlquist, and I Michael Navon. Improved skill for the anomaly correlation of geopotential heights at 500 hpa. Monthly weather review, 131(6):1082–1102, 2003.

[81] Jacques Hadamard. Lectures on Cauchy’s problem in linear partial diferential equations. Courier Corporation, 2014.

[82] Lawrence Edward Payne. Improperly posed problems in partial diferential equations. SIAM, 1975.

[83] E. N. Lorenz. Deterministic nonperiodic flow. Journal of the Atmospheric Sciences, 20(2): 130–141, 1963.

[84] Divakar Viswanath. Lyapunov exponents from random Fibonacci sequences to the Lorenz equations. Cornell University, 1998.

[85] V. I. Oseledets. A multiplicative ergodic theorem. Lyapunov characteristic numbers for dynamical systems. Transactions ofthe Moscow Mathematical Society, 19:197–231, 1968.

[86] Lai-Sang Young. Mathematical theory of lyapunov exponents. Journal of Physics A: Mathematical and Theoretical, 46(25):254001, jun 2013.

[87] Francesco Ginelli, Hugues Chate, Roberto Livi, and Antonio Politi. Covariant lyapunov´ vectors. Journal ofPhysics A: Mathematical and Theoretical, 46(25):254005, jun 2013.

[88] William G. Hoover, Carol G. Tull, and Harald A. Posch. Negative Lyapunov exponents for dissipative systems. Physics Letters A, 131(3):211–215, 1988. doi: 10.1016/0375-9601(88) 90072-2.

[89] Michael Ghil and Valerio Lucarini. The physics of climate variability and climate change. Reviews of Modern Physics, 92(3):035002, 2020.

[90] J-P Eckmann and David Ruelle. Ergodic theory of chaos and strange attractors. Reviews of Modern Physics, 57(3):617, 1985.

[91] James L. Kaplan and James A. Yorke. Chaotic behavior of multidimensional diference equations. In Functional Diferential Equations and Approximation of Fixed Points, volume 730 of Lecture Notes in Mathematics, pages 204–227. Springer, 1979.

[92] A. Karimi and M. R. Paul. Extensive chaos in the Lorenz-96 model. Chaos, 20:043105, 2010. doi: 10.1063/1.3496397.

[93] Mallory Carlu, Francesco Ginelli, Valerio Lucarini, and Antonio Politi. Lyapunov analysis of multiscale dynamics: the slow bundle of the two-scale Lorenz 96 model. Nonlinear Processes in Geophysics, 26:73–89, 2019.

[94] Lesley De Cruz, Sebastian Schubert, Jonathan Demaeyer, Valerio Lucarini, and Stephane´ Vannitsem. Exploring the Lyapunov instability properties of high-dimensional atmospheric and climate models. Nonlinear Processes in Geophysics, 25:387–412, 2018.

[95] Stephane Vannitsem. Predictability of large-scale atmospheric motions: Lyapunov exponents´ and error dynamics. Chaos, 27:032101, 2017. doi: 10.1063/1.4979042.

[96] Robert Lattes and Jacques Louis Lions. The method of quasi-reversibility: applications to\` partial diferential equations. American Elsevier, 1969.

[97] A. N. Tikhonov and V. Y. Arsenin. Solutions ofIll-Posed Problems. V. H. Winston & Sons, 1977. Translated from the Russian; translation editor F. John. Distributed by Halsted Press (Wiley), New York.

[98] Kerry A Emanuel, J David Neelin, and Christopher S Bretherton. On large-scale circulations in convecting atmospheres. Quarterly Journal of the Royal Meteorological Society, 120(519): 1111–1143, 1994.

[99] Alireza Seif, Mohammad Hafezi, and Christopher Jarzynski. Machine learning the thermodynamic arrow of time. Nature Physics, 17(1):105–113, 2021.

[100] Robert Zwanzig. Nonequilibrium statistical mechanics. Oxford university press, 2001.

[101] Joel L Lebowitz. Boltzmann’s entropy and time’s arrow. Physics Today, 46(9):32–38, 1993.

[102] Heinz Dieter Zeh. The physical basis of the direction of time, volume 4. Springer, 2007.

[103] Crispin Gardiner. Stochastic methods, volume 4. Springer Berlin Heidelberg, 2009.

[104] Brian DO Anderson. Reverse-time difusion equation models. Stochastic Processes and their Applications, 12(3):313–326, 1982.

[105] Ulrich G Haussmann and Etienne Pardoux. Time reversal of difusions. The Annals of Probability, pages 1188–1205, 1986.

[106] Alexandre J Chorin, Ole H Hald, and Raz Kupferman. Optimal prediction and the mori– zwanzig representation of irreversible processes. Proceedings of the National Academy of Sciences, 97(7):2968–2973, 2000.

[107] Andrew J Majda, Ilya Timofeyev, and Eric Vanden Eijnden. A mathematical framework for stochastic climate models. Communications on Pure and Applied Mathematics: A Journal Issued by the Courant Institute ofMathematical Sciences, 54(8):891–974, 2001.

[108] C. Franzke, T. J. O’Kane, J. Berner, P. D. Williams, and V. Lucarini. Stochastic climate theory and modelling. Wiley Interdisciplinary Reviews Climate Change, 6:63–78, 2015.

## A Methods and Data

Throughout the Methods and Data, �� denotes the numerical integration time step used to generate training and testing data (in PlaSim and Lorenz 96), Δ� denotes the input–output prediction interval used to train and infer with an AI model, and � denotes the forecast or backcast lead time. With the exception of Fig. S10C, all results and analyses presented in this paper are from AI models trained and inferred with single precision (FP32) on the NCAR CISL Derecho cluster using A100 GPUs (more details below).

We provide details of the following in this section:

• The ERA5 dataset and the AI models trained on it,

• The ICON NWP model and data,

• The PlaSim GCM, the dataset generated from it, and the AI models trained on this data,

• The two-scale Lorenz 96 system, the dataset generated from it, and the AI models trained on this data,

• The evaluation metrics.

## The Supplementary Materials includes

• Discussion on time reversal in multi-scale, dissipative dynamical systems,

• Movie S1 visualizing the 3D phase space of numerical forecasts and backcasts in Lorenz 63,

• More details on the Transformer-, SFNO-, and Difusion-based AI model architectures.

## A.1 Observation-based data: ERA5

Data

ERA5 [8] is the fifth-generation global atmospheric reanalysis of the European Centre for Medium-Range Weather Forecasts (ECMWF) and serves as the observationally constrained atmospheric dataset. We use five upper-air variables on 17 pressure levels and a selection of surface variables as our prognostic state x (see Table S3 for details). Each AIWP model evolves the prognostic variables in time conditioned on prescribed time-varying forcings b(�) and static boundary fields c, which are supplied as inputs but are not evolved.

As described below, we have also trained independent, separate forecasting and backcasting AIWP models with a transformer architecture that is based on Pangu-Weather. These deterministic models (Transformers, hereafter), used in Fig. 1, are trained on data remapped conservatively to a uniform $1 ^ { \circ } \times 1 ^ { \circ }$ latitude–longitude grid. We have trained independent, separate models with Δ� = 6 and 24 h. We train on 6-hourly data from 1979–2018 (40 years) and validate and select checkpoints on 2019.

The oficial Pangu-Weather model, which we obtained from [2], is trained on data with $0 . 2 5 ^ { \circ } \times 0 . 2 5 ^ { \circ }$ latitude–longitude grid; there are independent, separate versions trained with $\Delta t = 1 , 3 , 6$ and 24 h. We have only performed inference for “forecasting” with the Pangu-Weather model (Fig. 4).

These Transformer AIWP models and the Pangu-Weather model are tested using the same initial conditions taken once a month in 2020 and 2021.

## Forecasting and backcasting AIWP models: Transformer

The forecast and backcast AI models are formulated based on Eqs. (1)–(2), though here, the NN takes two additional inputs b(�) and c:

$$
\begin{array} { r l r } { { \mathbf x } ( t + \Delta t ) } & { { } = } & { N N _ { \theta } ( { \mathbf x } ( t ) , { \mathbf b } ( t ) , { \mathbf c } ) , } \end{array}\tag{S1}
$$

$$
\begin{array} { r l r } { { \mathbf x } \big ( t - \Delta t \big ) } & { { } = } & { N N _ { \phi } \big ( { \mathbf x } ( t ) , { \mathbf b } ( t ) , { \mathbf c } \big ) , } \end{array}\tag{S2}
$$

where b(�) is the time-varying forcing top-of-atmosphere incident solar radiation and c includes the land–sea mask and surface geopotential field (topography) through channel-wise concatenation. Table S3 lists all of the variables used in x, b, and c. We have trained independent, separate models for Δ� = 6 and 24 h, all on the $1 ^ { \circ } \times 1 ^ { \circ }$ horizontal resolution for computational feasibility.

The deterministic forecasting and backcasting neural networks $( N N _ { \theta }$ and $\mathcal { N N } _ { \phi } )$ have identical architectures, based on the Pangu-Weather architecture [2], hereafter referred to as the “Transformer”. The full architecture description is provided in the Supplementary Materials. The sole diference between $N N _ { \theta }$ and $N N _ { \phi }$ is the reversed input–output pairing, $[ \mathbf { x } ( t ) \ \to \ \mathbf { x } ( t - \Delta t ) ]$ instead of $[ \mathbf { x } ( t )  \mathbf { x } ( t + \Delta t ) ]$ , drawn from the same training dataset. Note that backcasting here should not be confused with the common meteorological terms “hindcast” or “reforecast”, which refer to aforecast initialized retrospectively for a historical date (see Fig. 1). Note that this autoregressive backcasting is also diferent from the backward prediction in cBottle [53], a novel non-autoregressive sampling emulator, which can produce a backward trajectory of a pre-defined length at once.

For training, each field is standardized by its mean and standard deviation computed from the training set. Both models are trained on single-time-step Δ� predictions; multi-step trajectories (rollouts) are generated autoregressively at inference by feeding each predicted state back in as input.

The Transformer minimizes a weighted mean absolute error (MAE) loss. Below, � represents an ERA5 training sample and ˆ� denotes the AI model prediction; the subscripts $a , z ,$ and � index upper-air variables, pressure levels, and surface variables, respectively $( \mathrm { e } . \mathrm { g } . , x _ { s }$ includes surface variables and $\hat { x } _ { a , z }$ denotes predicted upper-air variables). All losses are latitude weighted, with $\begin{array} { r } { \langle f \rangle _ { w } \equiv \sum _ { \varphi , \lambda } w ( \varphi ) f / \sum _ { \varphi , \lambda } \bar { w } ( \varphi ) } \end{array}$ denoting the latitude-weighted spatial mean and $w ( \varphi ) = \cos \varphi$ (here, � and � are longitude and latitude, respectively). The MAE loss is:

$$
\mathcal { L } _ { \mathrm { T r a n s f o r m e r } } = \frac { 1 } { N _ { a } N _ { z } } \sum _ { a , z } \Bigl \langle \vert \hat { x } _ { a , z } - x _ { a , z } \vert \Bigr \rangle _ { w } + \frac { \alpha } { N _ { s } } \sum _ { s } \Bigl \langle \vert \hat { x } _ { s } - x _ { s } \vert \Bigr \rangle _ { w } ,\tag{S3}
$$

where $\alpha = 1 / 4$ gives diferential weighting between upper-air and surface variables. $N _ { z } ~ = ~ 1 7 ,$ $N _ { a } = 5 .$ , and $N _ { s } = 9$ , are the number of vertical levels, upper-air variables, and surface variables, respectively.

The ERA5 forecasting and backcasting Transformers are trained with the AdamW optimizer and a OneCycleLR learning-rate schedule. We have explored a range of hyperparameters, $\mathrm { e . g . }$ , the peak learning rate, learning-rate schedule, and number of epochs, and selected the best-performing configuration based on the one-time-step (1Δ�) validation MAE. At the end, we have found the same hyperparameters for the best-performing forecasting and backcasting models, which have, overall, similar 1Δ� accuracy, comparable to that of the oficial Pangu-Weather forecast, at the same Δ� (Fig. S9). The complete architectural and training hyperparameters are listed in Table S4.

## Ensemble generation

Ensembles are generated by adding perturbations to the initial conditions. Perlin noise following [2] is added to all standardized fields of the initial condition $( T , u , \nu , q , Z$ on all pressure levels, and the surface variables). Each member is drawn from an independent random seed, and the perturbation amplitude is a function of a reference amplitude $\epsilon _ { 0 } = 0 . 1$ . The initial conditions are taken once a month in 2020 and 2021, and a 16-member ensemble is generated for each. Each two-dimensional perturbation field is a 3-octave sum on the $1 8 0 \times 3 6 0$ grid with periods of (4, 8, 16) cycles across the domain (spatial scales of roughly $9 0 ^ { \circ } , 4 5 ^ { \circ }$ , and $2 2 . { \bar { 5 } } ^ { \circ }$ per cycle) and octave weights of (0.2, 0.1, 0.05).

Hardware (GPU) precision and details can matter in small-amplitude perturbation experiments [23]: apparent butterflies can be hardware artifacts, as the rapid DKE growth of several AIWP models has been shown to be GPU numerical noise that vanishes or shrinks on CPUs. Therefore, we have carefully ensured that all training of and inference with our AI models (for ERA5, PlaSim, and Lorenz 96) use single floating-point precision (FP32). We have disabled TF32 and conducted all experiments on the same cluster (CISL Derecho) and same GPUs (A100). We have also confirmed that setting $\epsilon _ { 0 } = 0$ leads to $\mathrm { D K E } = 0$ over time in Pangu-Weather used in Fig. 4. To further see the efect of hardware, Fig. S10 presents experiments with Pangu-Weather and lower precision (TF32), which show noticeably larger DKE growth. This is due to state-dependent random perturbations that are injected into the calculations over time due to lower precision, consistent with the results reported in [23].

We have further verified that the DKE growth in Fig. 4A is robust to the averaging domain (global versus $6 0 ^ { \circ } \mathrm { S } { - } 6 0 ^ { \circ } \mathrm { N } )$ and to the perturbation type (Perlin versus Gaussian); see Fig. S10. Fig. 4A excludes latitudes poleward of ${ \bar { 6 } } 0 ^ { \circ }$ , where the freely running Pangu-Weather (with small Δ�) can develop unphysical behaviors. Part of the rapid DKE growth of the $\Delta t = 1$ h model at high latitudes (included in total wavenumbers in Figs. 4B and S6) reflects such instabilities rather than learned dynamics. Cleanly disentangling the two requires further study.

## Forecasting-backcasting AI model: Preliminary exploration

We have conducted preliminary exploration of building a single AI model that performs both forecasting and backcasting, as such a model might better learn dynamics. This bidirectional AI model can be represented as

$$
\begin{array} { r l r } { { \mathbf x } ( t + a \Delta t ) } & { { } = } & { N N _ { \vartheta } \left( { \mathbf x } ( t ) , a , { \mathbf b } ( t ) , { \mathbf c } \right) , } \end{array}\tag{S4}
$$

where $\textit { a } = \mathrel { + } 1$ (forecast) or −1 (backcast) determines the direction of the time evolution. This single neural network is trained with all training samples, $[ \mathbf { x } ( t )  \mathbf { x } ( t + \Delta t ) , a = + 1 ]$ and $[ { \bf x } ( t ) $ $\mathbf { x } ( t - \Delta t ) , a = - 1 ]$ , combined. The time indicator � can be fed into the model in diferent ways. For the Transformer, we use a learned directional embedding, a vector of the token dimension added to the patch embedding of every token, as is done for the positional embedding. Apart from this embedding, the architecture, the weighted MAE loss of Eq. (S3), and all training settings are the same as before (Table S4). We have only trained this model for $\Delta t = 2 4 \mathrm { ~ h ~ }$ . This preliminary investigation has not shown any noticeable improvement in accuracy or any change in DKE growth, though the design space for encoding � remains largely unexplored. As described later, we reach the same conclusion with a preliminary exploration in PlaSim, for which we also tried a cyclic loss.

## A.2 Numerical weather prediction (NWP) model: ICON

As a physics-based NWP model reference, we use ICON, an operational global model from the German Weather Service and Max Planck Institute for Meteorology, which solves the discretized governing diferential equations on an icosahedral grid. The ICON data (the DKE in Figs. 1C and 4A) used in this study are provided by [19], where the ICON ensembles are run at both 2.5 km and 20 km horizontal resolution.

## A.3 Atmosphere-only GCM: PlaSim

## GCM setup and data

PlaSim [70] is an intermediate-complexity global climate model (GCM) that couples a spectral atmospheric dynamical core, which solves the primitive equations for vorticity, divergence, temperature, and humidity, to a simplified land model. Sea surface temperature (SST) and sea ice are prescribed as boundary conditions that repeat identically each year. Numerical integrations are performed at T42 spectral truncation mapped onto a 64 × 128 Gaussian grid with 10 native vertical sigma levels. We have performed 110 years of integration. The first 60 years are discarded as spin-up (admittedly, unnecessarily too long). After integration, the 3D atmospheric state is interpolated onto 13 fixed pressure levels (Table S3). Six-hourly data from simulation years 61–100 (40 years) are used for training and after a 4-year gap, year 105 is used for validation and checkpoint selection. Initial conditions taken every 50 days in Years 106–109 are employed for testing.

We train separate, independent AIWP models with $\Delta t = 6$ and 24 h for Transformer and $\Delta t = 2 4$ h for SFNO and Difusion.

## Forecasting and backcasting AIWP models: Transformer, Neural Operator, and Difusion

For PlaSim, we train forecast and backcast AIWP models based on three architectures: the same Transformer used for ERA5, a spherical Fourier neural operator (SFNO) [71], and a difusion transformer (Difusion, hereafter) [72]. Full architecture descriptions are provided in the Supplementary Materials. The same Eqs. (S1)-(S2), single-time-step training, autoregressive inference, and standardization described for ERA5 are used.

Beyond the prognostic state x, the Transformer and SFNO receive the time-varying forcings b and static fields c (Table S3) through channel-wise concatenation, whereas the Difusion uses them as cross-attention context and is additionally conditioned on day of year and hour of day. For the forecasting Difusion model, Eq. (S1) represents a sample from the learned conditional distribution $p _ { \theta } ( \mathbf { x } ( t + \Delta t ) \mid \mathbf { x } ( t ) , \mathbf { b } ( t ) , \mathbf { c } )$ rather than a deterministic mapping to a unique future state. The same applies to the backcasting Difusion model.

All three architectures minimize a latitude-weighted loss using the notation defined earlier for ERA5. The Transformer minimizes the weighted MAE loss of Eq. (S3). The SFNO minimizes a latitudeweighted mean squared error (MSE) on the predicted state:

$$
\mathcal { L } _ { \mathrm { S F N O } } = \frac { 1 } { N _ { a } N _ { z } + N _ { s } } \left[ \sum _ { a , z } \Bigl \langle ( \hat { x } _ { a , z } - x _ { a , z } ) ^ { 2 } \Bigr \rangle _ { w } + \sum _ { s } \Bigl \langle ( \hat { x } _ { s } - x _ { s } ) ^ { 2 } \Bigr \rangle _ { w } \right] ,\tag{S5}
$$

where atmospheric and surface fields share a single normalizer $N _ { a } N _ { z } + N _ { s } . N _ { z } = 1 3 , N _ { a } = 5$ , and $N _ { s } = 4 .$ are the number of vertical levels, upper-air variables, and surface variables, respectively. The Difusion AIWP model is trained by denoising score matching and minimizes the same latitudeweighted MSE, applied to the predicted noise ˆ� :

$$
\mathcal { L } _ { \mathrm { D i f f u s i o n } } = \frac { 1 } { N _ { a } N _ { z } + N _ { s } } \left[ \sum _ { a , z } \Bigl \langle ( \hat { \varepsilon } _ { a , z } - \varepsilon _ { a , z } ) ^ { 2 } \Bigr \rangle _ { w } + \sum _ { s } \Bigl \langle ( \hat { \varepsilon } _ { s } - \varepsilon _ { s } ) ^ { 2 } \Bigr \rangle _ { w } \right] ,\tag{S6}
$$

where $\varepsilon$ is the noise injected at difusion step $\tau _ { d } \in \{ 1 , \dots , N _ { \mathrm { d i f f } } \}$ (�<sub>�</sub> is the difusion step index, distinct from the forecast lead time �), with $N _ { \mathrm { d i f f } } = 1 0 0$ and a cosine noise schedule.

All PlaSim models are trained with the AdamW optimizer; the learning-rate schedules and other training hyperparameters are listed in Table S4. As for ERA5, we have explored a range of hyperparameters (e.g., peak learning rate, learning-rate schedule, and number of epochs). For Transformers, the forecasting and backcasting models are tuned independently based on their single-step (1Δ�) validation MAE, which yields the same hyperparameters except for the peak learning rate (the 1Δ� errors of the forecast and backcast models are overall comparable, especially for $\Delta t = 2 4$ h; see Fig. S9). We also conducted hyperparameter optimization of the Transformer models against the error of the day-3 prediction, but this does not change any key results or conclusions, so all results reported in the paper are from models whose hyperparameters are chosen based on the 1Δ� loss. For the SFNO and Difusion models, hyperparameters are tuned on the forecasting model only, and the backcasting model adopts the same configuration.

## Ensemble generation

For all AIWP models of PlaSim, ensembles are generated by adding Perlin noise perturbations to initial conditions, following what we did for the ERA5 AIWP models. Because the Difusion model is stochastic and can otherwise generate ensemble spread intrinsically through its sampling process, we run it in a “quasi-deterministic” mode following [23]: the sampling seed is kept constant across ensemble members, so that ensemble spread arises solely from the explicit initial perturbations (with pre-defined amplitude) rather than from random sampling. Initial conditions are taken every 50 days during simulation years 106–109, and a 16-member ensemble is generated for each. Each two-dimensional noise field is constructed as for ERA5.

For the numerical integration of the physics-based PlaSim model, we use the PlaSim’s perturbation scheme implemented in [73]. We generate a 16-member ensemble for each initial condition by adding random white noise of small amplitude $( 1 0 ^ { - 8 } )$ to the spectral coeficients of the logarithmic surface pressure. Each perturbed state is advanced with the $\bar { \delta t } = 2 0$ min time step. The first 6 h is discarded as spin-up to allow rapid dynamical adjustment of the unphysical noise; at 6 h, we then subtract the ensemble mean to obtain the perturbations, which are scaled so that the initial ensemble spread (DKE) matches that of the corresponding AIWP models at 300 hPa (as shown in Fig. 1C).

## Forecasting-backcasting AI model: Preliminary exploration

As done for ERA5, we have explored a single bidirectional forecasting-backcasting Transformer for PlaSim, following Eq. (S4). Here, we have additionally experimented with a cycle-consistency constraint in the loss function, which requires that one Δ� forward $( a = + 1 )$ followed by one Δ backward $( a = - 1 )$ returns the original state. However, in these preliminary explorations, neither approach resulted in any improvement in accuracy or any change in the DKE growth.

## A.4 Canonical multi-scale, chaotic system: Two-scale Lorenz 96

## Equations and Data

We use the two-scale Lorenz 96 model [26], a canonical testbed for multi-scale, chaotic dynamics.

Slow- and large-scale variables $X _ { i }$ and fast- and small-scale variables $Y _ { i , j }$ are governed by

$$
\frac { d X _ { i } } { d t } = X _ { i - 1 } ( X _ { i + 1 } - X _ { i - 2 } ) - X _ { i } + F - \frac { h \gamma } { \beta } \sum _ { j = 1 } ^ { J } Y _ { i , j } ,\tag{S7}
$$

$$
\frac { d Y _ { i , j } } { d t } = - \gamma \beta Y _ { i , j + 1 } ( Y _ { i , j + 2 } - Y _ { i , j - 1 } ) - \gamma Y _ { i , j } + \frac { h \gamma } { \beta } X _ { i } ,\tag{S8}
$$

with periodic boundary conditions $( i = 1 , \dots , K ; j = 1 , \dots , J )$ . We adopt standard parameters [74] $K = 8 , J = 3 2 , F = 2 0 , h = 1$ , and $\beta = \gamma = 1 0 $ , giving a 264-dimensional state vector and chaotic dynamics. Time in this system is commonly measured in model time units (MTU). Based on the error doubling time of the large scales, past studies [26, 75] have estimated that with the current parameters, $1 \mathrm { M T U } \approx 5 $ days, which we have adopted in all figures showing results from the Lorenz 96 system.

The forward dynamics is chaotic and dissipative. In this system, dissipation is stronger at the smaller scales by a factor of $\gamma = 1 0$

To generate ground truth and training data, Eqs. (S7)-(S8) are integrated with a fourth-order Runge– Kutta scheme at $\delta t = 0 . 0 0 5$ MTU in double floating-point precision (FP64). The initial condition is drawn with independent standard normal values for all 264 components; after a 5,000-step (25 MTU) spin-up to discard transients, the subsequent $1 0 ^ { 7 }$ steps are saved at every �� in double precision.

The $1 0 ^ { 7 } .$ -step dataset is sequentially partitioned into 90% training and 10% validation, and all variables are standardized using the training set’s statistics. The validation segment is used for hyperparameter pruning, the learning-rate schedule, early stopping, and model selection (more details below). After model selection, an additional independent test trajectory equal in length to the validation segment is generated. The 100 evaluation initial conditions are drawn from this test trajectory and are spaced 1,000 �� (5 MTU, more than 100 Lyapunov times) apart, far beyond the system’s decorrelation time.

For the numerical results in Fig. 2A, we numerically re-integrate the equations from a reference state on the archived trajectory both forward and backward in time, the latter using the same Runge–Kutta scheme with a negated time step −��, in both single (FP32) and double (FP64) precision (the former is used to match the precision of AI models’ training and inference). To prevent floating-point overflow, state values in the numerical integration are capped $\mathrm { a t \pm 1 0 ^ { 1 0 } }$ ; the backward trajectory reaches this cap within a fraction of an MTU. In Fig. 2A, the saturated color regions correspond to $\left| X _ { i } \right|$ exceeding the color-scale range as the backcast solution grows without bound; in Fig. 3, crosses mark the last lead time at which the ensemble DKE of the backcasting numerical integration remains below $1 0 ^ { 3 }$ (the plotted range).

## Forecasting and backcasting AI models: Multilayer perceptron

Following Eqs. (1) and (2), we train independent forecast and backcast AI models for the Lorenz 96 system. Each is a multilayer perceptron with ReLU activations, of width � (neurons per hidden layer) and depth � (number of hidden layers), that outputs the full state x at $t \pm \Delta t$ directly. The AI-model prediction interval is $\Delta t = n \delta t$ with $n \in \{ 1 , 5 , 1 \bar { 0 } \}$ (reminder that $\delta t = 0 . 0 0 5$ MTU is the numerical solver’s time step). Note that the largest time step of AI models, $1 0 \delta t \mathrm { ~ = ~ } 0 . 0 5 \mathrm { ~ M T U } = 0 . 2 5$ day, is comparable to one Lyapunov time $( 1 / \lambda \approx 0 . 0 4$ MTU, from the measured leading finite-time Lyapunov exponent, FTLE, in Table S2).

The Lorenz 96 results are based on 12 independently trained AI models:

• Forecast and backcast models,

• Models with $\mathbf { x } = X _ { i }$ and $\mathbf { x } = ( X _ { i } , Y _ { i , j } )$ (from concatenation),

• Models with Δ� = ��� where � = 1<sub>,</sub> 5<sub>,</sub> or 10.

For each �, training pairs $\big ( \mathbf { x } ( t ) , \mathbf { x } ( t \pm n \delta t ) \big )$ use all saved samples and each model is trained on the full $9 \times 1 0 ^ { 6 }$ -sample training set. All models are trained with the Adam optimizer by minimizing the MSE of the predicted state.

We have selected the best performing AI models based on extensive hyperparameter optimization, though the key results and conclusions are robust with respect to these choices. We have explored a range of width $W \in \{ 5 1 2 , 1 0 2 4 , 2 0 4 8 \}$ and depth $D \in \{ 3 , 5 , 7 \}$ and four weight initialization seeds. Then for each fixed (�, �) configuration and seed, the learning rate and batch size have been tuned independently for each of the twelve models with the Optuna framework [76] and a Hyperband pruner [77] (Table S5). All results are reported using the best-performing AI models with $W = 1 0 2 4$ and $D = 7$ , selected based on the criterion of using the same $( W , D )$ for forecast and backcast that produce similar lowest single-step validation error. Single time-step (one Δ�) forecast and backcast RMSE are comparable for all models (backcast-to-forecast RMSE ratios of 0.90 − 1.13 except for one case; see Table S6), indicating that the forecast–backcast asymmetry in Figs. 2 and 3 is dominated by autoregressive rollout rather than by diferences in the single time-step accuracy of the AI models.

## Ensemble generation

Ensembles are generated by adding random perturbations to initial conditions: independent Gaussian noise of amplitude $\epsilon _ { 0 } = 0$ .1 is added to all components of the initial state x to produce a 100-member ensemble. Reduced-amplitude ensembles are obtained by scaling down �<sub>0</sub>.

## Computation of leading Lyapunov exponents in Lorenz 96

In multi-scale chaotic systems like Lorenz 96, the leading global Lyapunov exponent is set by the fast, small-scale variables (�): a generic infinitesimal perturbation asymptotically grows at this fastest rate, so the infinite-time measure is uninformative about the large scales (�). Predictability measures such as the error doubling time [75] are informative about the large scales, but only because they track finite-amplitude errors, whose fast, small-scale components saturate while the large-scale error continues to grow. We instead require the growth rate of infinitesimal perturbations as they impact the large-scale (�) variables alone. Thus, for the numerical and AI models, we estimate a projected FTLE: continuous perturbation rescaling keeps the perturbation infinitesimal, following the classical algorithm of Benettin et al. [78] as implemented by Wolf et al. [79], and the growth rate is measured, over a finite window, from the � components alone. Because the estimator requires only evaluations of the model itself (no tangent-linear equations), it is applied identically to the numerical and AI models.

A random initial state $\mathbf { x } _ { 0 }$ and a perturbed state $\mathbf { x } _ { 0 } ^ { \prime } = \mathbf { x } _ { 0 } + \delta _ { 0 } \mathbf { v } _ { 0 }$ (where $\mathbf { v } _ { 0 }$ is a random unit vector in the full state space and $\delta _ { 0 } = 1 0 ^ { - 4 } )$ are integrated through the AI model (time step Δ�) or the numerical model (time step ��). At each step $m ,$ , the full perturbation vector $\Delta \mathbf { x } _ { m } = \mathbf { x } _ { m } ^ { \prime } - \mathbf { x } _ { m }$ is recorded, and the perturbed state is re-normalized to magnitude $\delta _ { 0 }$ along the new divergence vector $\mathbf v _ { m } = \Delta \mathbf x _ { m } / \lVert \Delta \mathbf x _ { m } \rVert$ . The projected forward FTLE of the large-scale variables, $\lambda ^ { ( X ) }$ , is the time average over a finite window of $N _ { \mathrm { s t e p } }$ steps:

$$
\boldsymbol { \lambda } ^ { ( X ) } = \frac { 1 } { { { N _ { \mathrm { { s t e p } } } } \Delta t } } \sum _ { m = 1 } ^ { N _ { \mathrm { { s t e p } } } } { \ln \left( { \frac { { \lVert { \boldsymbol { \Delta } { \bf { x } } _ { m } ^ { ( X ) } } \rVert } } { { \delta _ { 0 } \lVert { \bf { v } _ { m - 1 } ^ { ( X ) } } \rVert } } } \right) } ,\tag{S9}
$$

where, to isolate the predictability of the large scales, the logarithmic growth rate uses exclusively the norm of the � components. The fast-variable exponents $\lambda ^ { ( \bar { Y } ) }$ in Table S2 are computed similarly but using the � components. The averaging window spans 0.1–0.3 MTU after a 0.1-MTU transient, so $\overline { { N _ { \mathrm { s t e p } } } } = 0 . 2 \mathrm { M T U } / \Delta t$ (40 steps for the numerical model and the 1�� AI models; 4 steps at $\Delta t = 1 0 \delta t )$

To capture the variability of predictability across the reference trajectory, reported values represent the mean and standard deviation evaluated over 20 independent initial conditions drawn from the reference trajectory.

The $\lambda ^ { ( X ) }$ and $\lambda ^ { ( Y ) }$ for the ground truth and some of the key AI models are reported in Table S2.

## A.5 Evaluation metrics

Forecast and backcast skills are assessed using two complementary metrics: the anomaly correlation coeficient (ACC) and the root mean square error (RMSE), which measure the accuracy of trajectory predictions. We also compute the diference kinetic energy (DKE), which quantifies the growth of initial perturbations and characterizes the chaotic behavior of the system. ACC and RMSE are evaluated for several predicted variables, whereas DKE is evaluated from the horizontal wind components. For Lorenz 96, RMSE and DKE are both based only on the large-scale variable, �.

Let � denote forecast or backcast lead time, $i = 1 , \ldots , N$ the grid point index, and $w _ { i } = \cos \varphi _ { i }$ the area weight for latitude $\varphi _ { i }$ . For deterministic verification, ${ \hat { x } } _ { i } ( \tau )$ and $x _ { i } ( \tau )$ are the predicted and ground truth values, and $\bar { x } _ { i } ^ { \mathrm { c } }$ is the climatological mean at grid point �. For ensemble verification, $u _ { i } ^ { ( m ) } ( \tau )$ and $\nu _ { i } ^ { ( m ) } ( \tau )$ are the zonal and meridional winds of member $m = 1 , \ldots , M$ , with instantaneous ensemble means $\tilde { u } _ { i } ( \tau )$ and $\tilde { \nu } _ { i } ( \tau )$ . All metrics are averaged over the evaluation cases (inference from many initial conditions in the test set) for each system to yield lead-time-dependent skill curves.

## Anomaly correlation coeficient (ACC)

ACC measures the area-weighted pattern similarity between predicted and target anomalies relative to climatology (defined as the calendar-day mean over 1979–2018 for ERA5 and years 61–100 for PlaSim),

$$
\operatorname { A C C } ( \tau ) = \frac { \displaystyle \sum _ { i = 1 } ^ { N } w _ { i } \left( \hat { x } _ { i } ( \tau ) - \bar { x } _ { i } ^ { \mathrm { c } } \right) \left( x _ { i } ( \tau ) - \bar { x } _ { i } ^ { \mathrm { c } } \right) } { \displaystyle \sqrt { \sum _ { i = 1 } ^ { N } w _ { i } \left( \hat { x } _ { i } ( \tau ) - \bar { x } _ { i } ^ { \mathrm { c } } \right) ^ { 2 } } ~ \sqrt { \sum _ { i = 1 } ^ { N } w _ { i } \left( x _ { i } ( \tau ) - \bar { x } _ { i } ^ { \mathrm { c } } \right) ^ { 2 } } } .\tag{S10}
$$

Note that ACC primarily measures the large-scale forecast accuracy. The conventional skill threshold for weather prediction is $\mathrm { A C C } > 0 . 6 [ 8 0 ]$ . For all ERA5 and PlaSim results, we define the asymmetry as the ratio of forecast to backcast lead times at $\mathrm { A C C } = 0 . 6$

Root mean square error (RMSE)

RMSE quantifies prediction error as

$$
\mathrm { R M S E } ( \tau ) = \sqrt { \frac { \displaystyle \sum _ { i = 1 } ^ { N } w _ { i } \left( \hat { x } _ { i } ( \tau ) - x _ { i } ( \tau ) \right) ^ { 2 } } { \displaystyle \sum _ { i = 1 } ^ { N } w _ { i } } } .\tag{S11}
$$

For Lorenz 96, RMSE is defined as in Eq. (S11) with uniform weights over the $K = 8$ large, slow variables �. The asymmetry in Lorenz 96 is then defined as the backcast/forecast RMSE ratio at a fixed lead (1.5 days; Fig. 2).

## Diference kinetic energy (DKE)

DKE measures ensemble spread rather than error against a target. It is the area-weighted mean of half the ensemble variance of the horizontal wind components, computed about the instantaneous ensemble mean $( \tilde { u } _ { i } ( \tau ) , \tilde { \nu } _ { i } ( \tau ) )$ , distinct from the calendar-day climatological mean $\bar { x } _ { i } ^ { \mathrm { c } }$ used in ACC. Analogous to the kinetic component of the diference total energy of [33], it quantifies spread directly in dynamical units:

$$
\mathbf { D K E } ( \tau ) = \frac { \displaystyle \sum _ { i = 1 } ^ { N } w _ { i } \frac { 1 } { 2 } \big [ \sigma _ { u , i } ^ { 2 } ( \tau ) + \sigma _ { \nu , i } ^ { 2 } ( \tau ) \big ] } { \displaystyle \sum _ { i = 1 } ^ { N } w _ { i } } , \qquad \sigma _ { u , i } ^ { 2 } ( \tau ) = \frac { 1 } { M - 1 } \sum _ { m = 1 } ^ { M } \bigl ( u _ { i } ^ { ( m ) } ( \tau ) - \tilde { u } _ { i } ( \tau ) \bigr ) ^ { 2 } ,\tag{S12}
$$

where $\sigma _ { u , \iota } ^ { 2 }$ and $\sigma _ { \nu , i } ^ { 2 }$ are the unbiased ensemble variances of the two horizontal wind components at grid point $i , w _ { i }$ is the area weight, � is the ensemble size, and � the number of grid points. Units are $\mathrm { m } ^ { 2 } \mathrm { s } ^ { - 2 }$ . Globally averaged DKE at 300 hPa is reported, unless specified otherwise.

For the Lorenz 96 system, DKE is analogously defined as half the sum of the ensemble variances of the large-scale variables:

$$
\mathrm { D K E } _ { \mathrm { L o r e n z } 9 6 } ( \tau ) = \frac { 1 } { 2 } \sum _ { i = 1 } ^ { K } \frac { 1 } { M - 1 } \sum _ { m = 1 } ^ { M } \bigl ( X _ { i } ^ { ( m ) } ( \tau ) - \tilde { X } _ { i } ( \tau ) \bigr ) ^ { 2 } ,\tag{S13}
$$

where $X _ { i } ^ { ( m ) }$ is the large-scale state of ensemble member � and $\tilde { X } _ { i }$ is the instantaneous ensemble mean.

## B Supplementary Materials

## B.1 Time reversal in multi-scale, dissipative, chaotic systems and the $2 ^ { \mathrm { n d } }$ law of thermodynamics

This section further describes why predicting the past of a multi-scale, dissipative, chaotic system, such as the atmosphere, should be practically impossible. It then explains why backcasting AI models can nevertheless be skillful without violating any of these arguments, or the second law of thermodynamics. We use three examples of increasing relevance: the heat equation, Lorenz $^ { 6 3 }$ , and the two-scale Lorenz 96 system (Methods and Data).

## Multi-scale dynamics: Heat equation and Hadamard ill-posedness

The linear, one-dimensional heat equation, $\partial T / \partial t = \kappa \partial ^ { 2 } T / \partial x ^ { 2 }$ with difusivity $\kappa > 0$ and spatial coordinate �, evolves each Fourier mode of wavenumber � independently:

$$
\hat { T } ( k , t ) = \hat { T } ( k , 0 ) e ^ { - \kappa k ^ { 2 } t } .\tag{S14}
$$

Forward in time $( t > 0 )$ , perturbations decay, most quickly at the smallest spatial scales; the analytical and numerical solutions are stable to initial-condition errors. Reversing time $( t ^ { * } = - t > 0 )$ flips the exponent: a perturbation of amplitude � at wavenumber $k ,$ , unavoidable in any real computation or measurement, grows backward as $\boldsymbol { \varepsilon } e ^ { + \kappa k ^ { 2 } t ^ { * } }$ and reaches order unity by $\begin{array} { r } { t _ { e } ^ { * } ( k ) = \frac { \ln ( 1 / \varepsilon ) } { \kappa k ^ { 2 } } } \end{array}$ . Even at double floating-point precision $( \varepsilon \sim 1 0 ^ { - 1 6 }$ , ln $( 1 / \varepsilon ) \approx 3 7 ) , t _ { e } ^ { * }$ quickly collapses quadratically with wavenumber: the smallest scales dominate the backcasting solution first, and in a system with a broad range of scales, they do so almost immediately. This is the classical statement that the backward heat problem is ill-posed in the sense of Hadamard [81, 82]. Note that here, the exponents in forecasting and backcasting have the same amplitudes (as a function of length scale) but opposite signs $( - \kappa k ^ { 2 } \mathrm { \ v e r s u s } + \kappa k ^ { 2 } )$

## Chaotic dynamics: Lorenz 63

Chaotic dissipative systems further refine the picture of backcasting. The nonlinear, single-scale Lorenz 63 system [83],

$$
\frac { d x } { d t } = \sigma ( y - x ) ,\tag{S15}
$$

$$
\frac { d y } { d t } = x ( \rho - z ) - y ,\tag{S16}
$$

$$
\frac { d z } { d t } = x y - \beta z ,\tag{S17}
$$

with standard parameters $( \sigma = 1 0 , \rho = 2 8 , \beta = 8 / 3 )$ has Lyapunov exponents $( \lambda _ { 1 } , \lambda _ { 2 } , \lambda _ { 3 } )$ ≈ (0.91, 0, −14.57) [84]. Their sum equals the phase-space divergence:

$$
\sum _ { i = 1 } ^ { 3 } \lambda _ { i } = - ( 1 + \sigma + \beta ) \approx - 1 3 . 6 7 ,\tag{S18}
$$

i.e., volumes contract, and trajectories settle onto an attractor, the Lorenz strange attractor, while nearby states separate exponentially at the modest rate $\lambda _ { 1 } \approx + 0 . 9 1$ (Movie S1). Reversing time negates the Lyapunov spectrum, which reorders $\mathbf { t o } \approx ( + 1 4 . 5 7 , 0 , - 0 . 9 1 )$ ; see below for more discussion on this reordering.

Two major consequences follow, and Movie S1 shows both. First, the reversed system is dramatically more unstable along trajectories: infinitesimal errors �-fold $1 4 . 5 7 / 0 . 9 1 \approx 1 6$ times faster in backcasting compared with forecasting, so an initial error reaches order one 16× faster. Second, and more damning, the sum of the reversed exponents is

$$
- \sum _ { i = 1 } ^ { 3 } \lambda _ { i } = + ( 1 + \sigma + \beta ) \approx + 1 3 . 6 7 .\tag{S19}
$$

Backward in time, phase-space volumes expand, and the attractor of the forward dynamics is a repeller of the backward dynamics. Any perturbation transverse to the attractor, which forward dynamics would harmlessly contract away, is exponentially amplified backward at rates up to $e ^ { + 1 4 . 5 7 t ^ { * } }$ and trajectories are expelled from the attractor altogether (see Movie S1).

As a result, numerical backcasting of single-scale Lorenz 63 fails for two distinct reasons: errors grow faster and into regions the dynamics never visits. Note that here, unlike for the heat equation, time-reversal not only negates the signs of exponents but also changes the amplitude of the leading exponent, leading to 16× faster growth of errors in backcasting. These two failures follow from two distinct properties of Lorenz $6 3 \mathrm { { : } s }$ Lyapunov spectrum under time reversal: A) The sign reversal of all exponents, and B) $| \lambda _ { 3 } | > \lambda _ { 1 }$ . Next, we discuss the conditions under which (A) and (B) apply to more complex systems such as the atmosphere.

As for (A), if the dynamics is invertible, admits an ergodic invariant probability measure, and satisfies a mild integrability condition on the tangent dynamics, the Oseledets multiplicative ergodic theorem guarantees a well-defined Lyapunov spectrum, and the exponents of the time-reversed dynamics are the forward exponents with reversed signs [85–88]. For dissipative systems such as the atmosphere [89], the invariant measure is supported on the attractor, so the reversal applies to trajectories on the attractor itself [90]; e.g., the initial conditions that we choose from forward numerical integrations of Lorenz 96 and PlaSim, or from ERA5 reanalysis. Of-attractor states diverge under the reversed flow and have no backward Lyapunov exponents [90]; such states are not available from ERA5 reanalysis or any forward numerical integration (Lorenz 96 or PlaSim) and are not of concern here. Therefore, (A) is broadly true.

Let us order the � Lyapunov exponents of a system from the most positive $\lambda _ { 1 }$ (leading) to the most negative $\lambda _ { L }$ . Then, property (B) is that $| \lambda _ { L } | > \lambda _ { 1 }$ , so that upon reversal and reordering, the leading backward exponent exceeds the leading forward one, making backward error growth exponentially faster than forward. In three-dimensional dissipative, chaotic systems such as Lorenz 63 this is automatic: the exponent sum equals the (negative) phase-space divergence and the intermediate exponent vanishes, so $\left| \lambda _ { 3 } \right| = \lambda _ { 1 } + \left( \sigma + 1 + \beta \right) > \lambda _ { 1 }$ (equivalently, the Kaplan–Yorke dimension lies below three [91]). In higher dimensions, however, a negative exponent sum only guarantees that the contracting directions collectively outweigh the expanding ones, so $| \lambda _ { L } | > \lambda _ { 1 }$ is not guaranteed in general. But the picture changes once we consider multi-scale systems. In such systems, dissipation acts most strongly on the smallest and fastest scales, leading to $| \lambda _ { L } | > \lambda _ { 1 }$ , as documented in computed Lyapunov spectra of Lorenz 96 [92, 93] and of atmospheric and coupled climate models [94, 95]. Therefore, both (A) and (B) are expected to apply to the atmosphere.

Considering the above discussion on the heat equation and Lorenz 63, backcasting of multi-scale, dissipative, chaotic systems should be practically impossible. Next, we discuss a canonical system that embodies all three characteristics, the two-scale Lorenz 96.

## Multi-scale, chaotic dynamics: Two-scale Lorenz 96

This system (Eqs. (S7)-(S8)) adds the ingredient the atmosphere has and Lorenz 63 lacks: multi-scale dynamics, specifically, faster dissipation at smaller scales $Y _ { i , j }$ . Under time reversal, the damping terms $- X _ { i }$ and $- \gamma Y _ { i , j }$ become anti-dissipative, with the fast variables, growing $\gamma ~ = ~ 1 0$ times more strongly, blowing up first and dragging the large scales $X _ { i }$ with them through the coupling. Consequently, backcasting via numerical integration diverges rapidly at both single and double floating-point precision (Fig. 2A).

## The second law of thermodynamics, the arrow of time, and the asymmetry

It may seem that a skillful AI backcasting conflicts with the second law of thermodynamics: entropy increase selects a direction (arrow) of time, and the atmosphere, a forced-dissipative system full of mixing and irreversible diabatic processes, plainly should follow that arrow. However, coarse-graining removes the spatio-temporal scales at which the strongest dissipation happens, leading to a training set that is closer to reversible. Seen in a diferent way, coarse-graining mimics the classical remedy to Hadamard ill-posedness: regularization by truncating/damping the high wavenumbers, after which the time-reversed integrations become stable over a finite window [96, 97]. Furthermore, the training data contain only states on (or extremely near) the attractor, so the AI model never learns, and is never asked to represent, the explosive of-attractor directions (and as described above, because of the coarse-graining, the on-attractor instability the training data inherits is milder as well). A learned backcast does not integrate the reversed vector field; it estimates, by regression, the attractor state

Δ� earlier that is most compatible with the current coarse state. It operates on the attractor, at the “resolved” scales, or not at all.

In the end, the skillful AI backcasting does not violate the second law of thermodynamics; it just operates on a dataset that is closer to reversible due to coarse-graining and on-attractor sampling. The forecast-backcast asymmetry that we see in the hierarchy (Figs. 1B and 2B-C) is then a measure of the residual irreversibility (dynamical dissipation and, for the atmosphere, thermodynamic entropy production) that survives coarse-graining and is retained in the training data. This interpretation makes a testable prediction: the asymmetry should be largest where irreversible processes are strongest in the resolved dynamics of the atmosphere. The larger asymmetry we find in the tropics, where diabatic heating and convective dissipation dominate [31, 32, 98], compared with the extratropics (Table S1) is consistent with this prediction, and warrants further investigation.

Finally, it is worth mentioning the double role “coarse-graining” might appear to play, and the distinction between the two operations this term refers to can be instructive. Microscopic dynamics (at the molecular level) are time-reversal symmetric; irreversibility and entropy growth emerge at the level of coarse-grained, macroscopic descriptions<sup>1</sup>, when information about the discarded degrees of freedom is given up, as formalized from Boltzmann’s molecular chaos assumption to the projection-operator formalisms of Mori and Zwanzig [66, 65, 100]. Thus, coarse-graining from molecular (microscopic) to continuous (macroscopic) dynamics is the operation through which the latter acquires its arrow of time and time-irreversibility [101, 102]. In the context of the work here, coarse-graining refers to the removal of the fast and small scales of the macroscopic dynamics, which brings the training data closer to time-reversible.

## The role ofstochasticity and the Mori-Zwanzig Formalism

All of the arguments above concern deterministic dynamics. Stochastic dynamics behave qualitatively diferently under time reversal. Consider the linear ordinary diferential equation $d x / d t = - a x$ with $a > 0$ , which, for example, governs the Fourier mode of the heat equation with $a = \kappa k ^ { 2 }$ . The forecast solution is $x ( t ) \sim e ^ { - a t }$ , which exponentially decays. The time-reversed equation is $d x / d t ^ { * } = + a x .$ where $t ^ { * } = - t > 0$ . Thus, the backcast solution is $x ( t ^ { * } ) \sim e ^ { + a t ^ { * } }$ , which exponentially blows up.

Now let’s consider the stochastic version driven by Gaussian white noise �(�):

$$
\frac { d x } { d t } = - a x + \eta ( t ) ,\tag{S20}
$$

i.e., an Ornstein–Uhlenbeck process [103]. When the noise is additive and the process is statistically stationary, the equation obeyed by the time-reversed process [104, 105] is:

$$
\frac { d x } { d t ^ { * } } = - a x + \tilde { \eta } ( t ^ { * } ) ,\tag{S21}
$$

where $\tilde { \eta }$ is another realization of the same white noise. This is basically the same equation as (S20), showing that the stationary stochastic process is time-reversible. The past is exactly as predictable as thefuture, which is dramatically diferentfrom the behavior ofthe same equation when the noise vanishes $( \eta = 0 )$

The Mori–Zwanzig formalism [65, 66, 100] reconciles the coarse-graining versus stochasticity explanations. By the Mori–Zwanzig formalism, eliminating the fast, small scales of a deterministic multiscale system (coarse-graining) leaves resolved-scale dynamics that contain memory and a stochastic forcing [106]. This observation underlies stochastic climate modeling [107, 108, 69] and stochastic parameterizations [67, 68]. Coarse-grained deterministic data are thus statistically of the same kind as realizations of a stochastic process: the stochastic-process view of ERA5 and the coarse-graining explanation of the main text are one interpretation expressed in two languages.

## B.2 More details on neural network architectures

## The Transformer model

The Transformer follows the three-dimensional Earth-specific transformer architecture of [2]. A 3D patch embedding groups every 2×2×2 block in (level, latitude, longitude) into a 240-dimensional token. These tokens pass through a hierarchical encoder–decoder of Swin-style shifted-window self-attention blocks in four stages of depth 2, 6, 6, and 2, each attending within a local window spanning two pressure levels, with 6, 12, 12, and 6 attention heads per stage. The encoder halves the horizontal resolution at each stage and the decoder restores it; a sub-pixel deconvolution head and a recovery head reconstruct the full-resolution prognostic state. The ERA5 Transformer uses a window size of 2×6×10. The PlaSim Transformer is identical except for a window size of 2×6×12, matching the T42 grid; see Table S4 for details.

## The spherical Fourier Neural Operator (SFNO) model

The SFNO [71] uses spherical harmonic transform (SHT), so that spectral filtering respects the Earth’s spherical geometry. In plain terms, the model learns transformations in spherical spectral space rather than directly on the latitude–longitude grid (used in FourCastNet [1]). A point-wise encoder lifts the input to a 384-dimensional channel space. Twelve spectral blocks each apply a forward SHT, a learned degree-dependent linear filter, and an inverse SHT, followed by a two-layer GELU multilayer perceptron (expansion ratio 2) with a residual connection and per-channel instance normalization. A global residual connection links encoder to decoder.

## The Difusion model

This is a conditional score-based difusion model. This generative model learns to reverse an artificial noising process and generates a target state through iterative denoising. A forward process adds Gaussian noise to the target state over $N _ { \mathrm { d i f f } } { = } 1 0 0$ steps with a cosine noise schedule. The denoiser is a vision transformer that tokenizes the input via a 2×2 patch embedding into 1024-dimensional tokens, to which a learned spherical-harmonics positional embedding is added. Twelve self-attention blocks (16 heads each) operate under adaptive layer normalization conditioned on the difusion noise level and calendar position (day-of-year and hour-of-day), setting per-block shift, scale, and gate parameters. After each of the first 6 blocks, a cross-attention layer allows state tokens to attend to boundary-forcing context tokens. A noise-level-modulated unpatchify head projects back to the prognostic channels.

Table S1: Regional variation of the forecast-backcast asymmetry. As defined in the main text, the asymmetry is the ratio of the lead times at which the forecast and backcast ACC drops to 0.6. Asymmetry > 1 means that backcasting is less skillful than forecasting. Asymmetry values are reported globally and separately for the extratropical Northern Hemisphere (NH: 30<sup>◦</sup>N–70<sup>◦</sup>N), Tropics: 30<sup>◦</sup>S–30<sup>◦</sup>N, and extratropical Southern Hemisphere (SH: 70<sup>◦</sup>S–30<sup>◦</sup>S) for both ERA5 and PlaSim and for a variety of architectures and Δ�. For each variable and AI model, the largest value among the three regions is shown in bold. For ERA5, consistent across both Δ� values and variables at 3 vertical levels, the asymmetry is noticeably largest in the tropics. For PlaSim, the same trend is seen for Z500 and U250, but the trend is not robust for U850.
<table><tr><td colspan="2"></td><td colspan="2">ERA5</td><td colspan="3">PlaSim</td></tr><tr><td colspan="2">Architecture</td><td colspan="2">Transformer</td><td colspan="2">Transformer SFNO</td><td>Diffusion</td></tr><tr><td>Variable</td><td>Region</td><td>24 h 6h</td><td>24 h</td><td>6h</td><td>24h</td><td>24 h</td></tr><tr><td rowspan="4">U250</td><td>Global</td><td>1.50</td><td>1.81 1.78</td><td>1.98</td><td>1.77</td><td>2.03</td></tr><tr><td>NH</td><td>1.36 1.66</td><td>1.68</td><td>1.84</td><td>1.68</td><td>1.94</td></tr><tr><td>Tropics</td><td>1.91 2.23</td><td>2.04</td><td>2.15</td><td>2.00</td><td>2.32</td></tr><tr><td>SH</td><td>1.42 1.74</td><td>1.72</td><td>1.87</td><td>1.79</td><td>1.99</td></tr><tr><td rowspan="4">Z500</td><td>Global</td><td>1.40</td><td>1.63</td><td>1.63 1.76</td><td>1.59</td><td>1.94</td></tr><tr><td>NH</td><td>1.35 1.64</td><td>1.59</td><td>1.73</td><td>1.48</td><td>1.89</td></tr><tr><td>Tropics</td><td>1.53</td><td>1.97 2.03</td><td>2.75</td><td>1.82</td><td>2.17</td></tr><tr><td>SH</td><td>1.39 1.65</td><td>1.63</td><td>1.72</td><td>1.70</td><td>1.94</td></tr><tr><td rowspan="4">U850</td><td>Global</td><td>1.47</td><td>2.03</td><td>1.82</td><td>2.35 1.77</td><td>2.19</td></tr><tr><td>NH</td><td>1.38 2.07</td><td>1.80</td><td>2.34</td><td>1.75</td><td>2.22</td></tr><tr><td>Tropics</td><td>1.79</td><td>2.52 1.79</td><td>2.25</td><td>1.71</td><td>2.27</td></tr><tr><td>SH</td><td>1.33</td><td>1.84 1.80</td><td>2.24</td><td>1.83</td><td>2.06</td></tr></table>

Table S2: Leading Lyapunov exponents of Lorenz 96 numerical and AI models. The leading finite-time Lyapunov exponents (FTLE) � are estimated using the rescaling method (see Methods and Data). The reported values are the mean and standard deviation over 20 independent initial conditions on the reference trajectory, each with an independent random initial perturbation of amplitude $\delta _ { 0 } ~ = ~ 1 0 ^ { - 4 }$ , computed over the window 0.1–0.3 MTU (after discarding a 0.1-MTU transient). The FTLE values are projected onto the slow (�) or fast (�) variables.
<table><tr><td>System / AI model</td><td>Projection</td><td> $\overline { { \lambda ( \mathbf { M T U } ^ { - 1 } ) } }$ </td></tr><tr><td>Ground truth</td><td>X</td><td> $2 1 . 9 \pm 6 . 3$ </td></tr><tr><td>Ground truth</td><td>Y</td><td> $2 4 . 3 \pm 4 . 0$ </td></tr><tr><td>XY, 1δt AI model</td><td>X</td><td> $2 6 . 8 \pm 6 . 5$ </td></tr><tr><td>XY, 1δt AI model</td><td>Y</td><td> $2 6 . 8 \pm 5 . 9$ </td></tr><tr><td>X, 1δt AI model</td><td>X</td><td> $3 . 0 5 \pm 1 . 8 4$ </td></tr><tr><td>X, 10δt AI model</td><td>X</td><td> $2 . 8 1 \pm 1 . 5 9$ </td></tr></table>

Table S3: Variable inventories for the Pangu-Weather, ERA5, and PlaSim AIWP models. Upperair prognostic state $\mathbf { X } _ { \mathrm { U a } } .$ , surface prognostic state $\mathbf { X } _ { \mathrm { s f c } }$ , time-varying boundary forcings b, and static fields c are listed. Checkmarks indicate inclusion; dashes indicate not applicable. The Pangu-Weather column describes the pretrained model on its native $0 . 2 5 ^ { \circ }$ grid, which we use for inference only. The ERA5 Transformer column describes the data for the 1<sup>◦</sup> ERA5 Transformer. The PlaSim AIWP column describes data for the Transformer, SFNO, and Difusion AIWP models. <sup>†</sup>We have trained versions of the PlaSim Transformer with sea-surface temperature as a prognostic variable (in x) and the key observations and conclusions from Fig. 1B-C remain the same.
<table><tr><td>Category</td><td>Variable</td><td>Pangu-Weather [2]</td><td>ERA5 Transformer</td><td>PlaSim AIWP</td></tr><tr><td rowspan="5">Upper-air  $\mathbf { X } _ { \mathrm { U a } }$ </td><td>Temperature (T)</td><td>13 levels</td><td>17 levels</td><td>13 levels</td></tr><tr><td>Zonal wind (u)</td><td>13 levels</td><td>17 levels</td><td>13 levels</td></tr><tr><td>Meridional wind (v)</td><td>13 levels</td><td>17 levels</td><td>13 levels</td></tr><tr><td>Specific humidity (q)</td><td>13 levels</td><td>17 levels</td><td>13 levels</td></tr><tr><td>Geopotential (Z)</td><td>13 levels</td><td>17 levels</td><td>13 levels</td></tr><tr><td rowspan="8">Surface Xsfc</td><td>2-m temperature</td><td></td><td>√</td><td></td></tr><tr><td>10-m zonal &amp; meridional wind</td><td></td><td>√</td><td></td></tr><tr><td>Mean sea-level pressure</td><td></td><td>√</td><td></td></tr><tr><td>Surface pressure</td><td></td><td>√</td><td></td></tr><tr><td>Skin temperature</td><td></td><td>√</td><td>V</td></tr><tr><td>0–7 cm soil water &amp; soil temp.</td><td></td><td>√</td><td></td></tr><tr><td>Soil moisture</td><td></td><td></td><td>√</td></tr><tr><td>Sea-surface temperature</td><td></td><td>√</td><td></td></tr><tr><td rowspan="3">Time-varying forcings b</td><td>top-of-atmosphere incident solar radiation</td><td></td><td>√</td><td>V</td></tr><tr><td>Sea-surface temperature</td><td></td><td></td><td> $\checkmark ^ { \dagger }$ </td></tr><tr><td>Sea-ice concentration</td><td></td><td></td><td>√</td></tr><tr><td rowspan="2">Static fields c</td><td>Land-sea mask</td><td></td><td>√</td><td>√</td></tr><tr><td>Surface geopotential Horizontal grid</td><td></td><td>√</td><td>√</td></tr><tr><td rowspan="5"></td><td>Pressure levels (hPa)</td><td>721 × 1440 (0.25°) 50, 100, 150, 200, 250, 300, 400, 500, 600, 700,</td><td>180 × 360 (1°) 5, 10, 20, 30, 50, 70, 100, 150, 250, 300, 400,</td><td>64 × 128 (T42) 50, 100, 150, 200, 250,</td></tr><tr><td></td><td>850, 925, 1000</td><td>500, 600, 700, 850,</td><td>300, 400, 500, 600, 700, 850, 925, 1000</td></tr><tr><td>Time interval ∆t</td><td>1 h, 3 h, 6 h, 24 h</td><td>925,1000 6 h, 24 h</td><td>6 h, 24 h</td></tr><tr><td>Training period</td><td>1979–2017 (39 years)</td><td>1979–2018 (40 years)</td><td>61–100 (40 years)</td></tr><tr><td>Validation period Test period</td><td>2019 (1 year) 2020-2021 (2 years)</td><td>2019 (1 year) 2020-2021 (2 years)</td><td>105 (1 year) 106–109 (4 years)</td></tr></table>

Table S4: Architectural and training hyperparameters for ERA5 and PlaSim AIWP models. Entries marked with a dash are not applicable. <sup>†</sup> Forecast / backcast peak learning rates.
<table><tr><td></td><td></td><td>ERA5 Transformer</td><td>PlaSim Transformer</td><td>PlaSim SFNO</td><td>PlaSim Diffusion</td></tr><tr><td rowspan="5">Architecture</td><td>Embedding dim Demb</td><td>240</td><td>240</td><td>384</td><td>1024</td></tr><tr><td>Depths / blocks</td><td>2+6+6+2</td><td>2+6+6+2</td><td>12</td><td>12 self-attn. + 6 cross-attn.</td></tr><tr><td>Attention heads</td><td>6/12/12/6</td><td>6/12/12/6</td><td></td><td>16</td></tr><tr><td>Window size</td><td>2×6×10</td><td>2×6×12</td><td></td><td></td></tr><tr><td>Patch size</td><td>2×2×2</td><td>2×2×2</td><td></td><td>2×2</td></tr><tr><td rowspan="8">Training</td><td>Optimizer</td><td>AdamW</td><td>AdamW</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Peak learning rate</td><td>5×10⁻4</td><td>6×10−4/2×10-4†</td><td>6×10-4</td><td>5×10-5</td></tr><tr><td>LR schedule</td><td>OneCycleLR</td><td>OneCycleLR</td><td>ReduceLROnPlateau</td><td>warmup + cosine decay</td></tr><tr><td>Weight decay</td><td>0.01</td><td>0.01</td><td>0.01</td><td>0.05</td></tr><tr><td>Batch size</td><td>32</td><td>32</td><td>32</td><td>32</td></tr><tr><td>Epochs</td><td>120</td><td>120</td><td>200</td><td>800</td></tr><tr><td>Loss function</td><td>weighted MAE</td><td>weighted MAE</td><td>spherical MSE</td><td>weighted MSE</td></tr><tr><td>Steps Ndiff Noise schedule</td><td></td><td></td><td></td><td>100</td></tr></table>

Table S5: Architectural and training hyperparameters for the Lorenz 96 AI models. Set notation indicates discrete values explored during the hyperparameter optimization phase.
<table><tr><td>Phase</td><td>Hyperparameter</td><td>Value / Search space</td></tr><tr><td rowspan="4">Architecture</td><td>Architecture</td><td>multilayer perceptron, ReLU activations</td></tr><tr><td>Width W</td><td>{512, 1024, 2048}</td></tr><tr><td>Depth D</td><td>{3, 5, 7} hidden layers</td></tr><tr><td>Input dimension</td><td>8 (X only) or 264 (XY)</td></tr><tr><td rowspan="4">Hyperparameter search</td><td>Optimizer</td><td>Adam</td></tr><tr><td>Learning rate</td><td>Log-uniform  $[ 1 0 ^ { - 4 } , 1 0 ^ { - 2 } ]$ </td></tr><tr><td>Batch size</td><td>{256, 512, 1024, 2048, 4096}</td></tr><tr><td>Pruning</td><td>Hyperband (150 epochs/trial)</td></tr><tr><td rowspan="4">Final training</td><td>Max epochs</td><td>500</td></tr><tr><td>LR schedule</td><td>ReduceLROnPlateau (factor 0.5, patience 15)</td></tr><tr><td>Early stopping</td><td>patience 50 epochs</td></tr><tr><td>Training loss</td><td>mean squared error (MSE)</td></tr><tr><td></td><td>Evaluation metric</td><td>mean absolute error (MAE)</td></tr></table>

Table S6: One-time-step RMSE of the forecast and backcast AI models. RMSE of a single prediction step of length Δ�, computed over the � variables against the reference trajectory, for the selected AI models with $W { = } 1 0 2 4$ and �=7. Values are the mean ± standard deviation over the same 100 initial conditions as Fig. 2.
<table><tr><td>X</td><td>∆t</td><td>Forecast RMSE</td><td>Backcast RMSE</td></tr><tr><td>X</td><td>1δt</td><td> $\overline { { 0 . 0 0 5 6 \pm 0 . 0 0 2 0 } }$ </td><td> $\overline { { 0 . 0 0 5 6 \pm 0 . 0 0 2 1 } }$ </td></tr><tr><td>X</td><td>5δt</td><td> $0 . 0 2 6 \pm 0 . 0 0 9 3$ </td><td> $0 . 0 2 7 \pm 0 . 0 1 1$ </td></tr><tr><td>X</td><td>10δt</td><td> $0 . 0 4 8 \pm 0 . 0 1 6$ </td><td> $0 . 0 4 6 \pm 0 . 0 1 8$ </td></tr><tr><td>XY</td><td>1δt</td><td> $0 . 0 3 8 \pm 0 . 0 1 2$ </td><td> $0 . 0 4 3 \pm 0 . 0 1 5$ </td></tr><tr><td>XY</td><td>5δt</td><td> $0 . 1 6 \pm 0 . 0 5 4$ </td><td> $0 . 2 3 \pm 0 . 0 6 7$ </td></tr><tr><td>XY</td><td>10δt</td><td> $0 . 2 9 \pm 0 . 0 7 3$ </td><td> $0 . 2 6 \pm 0 . 0 6 6$ </td></tr></table>

![](images/1feea5c40cdef35489eb5a2bd8ba9b4431d20a65b1ff4cb11b6988f0fcf546b9.jpg)

![](images/82cc75a48fd3544552953f2ab1ebab36d69504fb710922730766eb9845368277.jpg)

![](images/7fb3d9e6264bafb445e6d5b4a52d059042f7b62e509ae4943f33559b3c217a98.jpg)

![](images/bab985e1d45662182eebb696237e468d12589798e0614bba24867cd15ba92bcd.jpg)  
Figure S1: Missing butterfly and forecast-backcast asymmetry in the SFNO-based AIWP models of PlaSim GCM. As in Fig. 1B–C, but for independently trained forecasting and backcasting AIWP models, based on spherical Fourier neural operator (SFNO) architecture [71], with $\Delta t = 2 4$ h (see Methods and Data). ACC of �850, �250, and �500 versus lead time are shown, as well as the DKE at 300 hPa for ensembles initialized with perturbations of amplitude $\epsilon _ { 0 } = 0 . 1 , \epsilon _ { 0 } / 1 0$ , and $\epsilon _ { 0 } / 1 0 ^ { 3 }$ . All curves are averaged over the same initial conditions from the test set as in Fig. 1.

![](images/6612c685142346c03dc08ef65ba3616591214ef19880fd99c9c60f6bd36e799b.jpg)

![](images/a9fa37ffc4fd106e890124f1e8e39de679d5db25405083bde08005f2731d2aad.jpg)

![](images/61687d539688537e096013d7677987b7fe6ab6133d9f9204e7fa2e0805e5aa58.jpg)

![](images/e7090d901cc9b39135cd8e1a8b0a4e581b412ad47ff4f7b4c8751ea3296d9892.jpg)  
Figure S2: Missing butterfly and forecast-backcast asymmetry in the Difusion-based AIWP models of PlaSim GCM. As in Fig. S1, but for independently trained forecasting and backcasting AIWP models, based on conditional difusion models, with $\Delta t = 2 4 \mathrm { h }$ (see Methods and Data). ACC curves are shown for the 16-member ensemble mean (solid) and for a single member (dashed). In the DKE panel, solid curves are obtained in the quasi-deterministic mode used throughout the paper (see Methods and Data), whereas the dashed curve is the spread generated by stochastic sampling alone, with unperturbed initial conditions.

![](images/e6a997ae41d1c9c131d930506dc151bc84c6be8fc392ac7c4b440c5d6cbe455f.jpg)  
Figure S3: Forecast-backcast asymmetry across variables and Δ� in Transformer-based AIWP models of ERA5 and PlaSim GCM. As in Fig. 1B, but for U850 (top), Z500 (middle), and U250 (bottom) for ERA5 and PlaSim Transformers $( \Delta t = 2 4$ and 6 h). Colors denote Δ� (legend at the bottom); the color-matched annotations show the asymmetry. Curves are averaged over the test set initial conditions (same as in Fig. 1).

![](images/100524c36c9b9f1c4ed28b2776c34a86499ca0d581b063dad627cc67c35751fa.jpg)  
Figure S4: RMSE-based forecast and backcast accuracy and asymmetry across variables and Δ� in Transformer-based AIWP models of ERA5 and PlaSim GCM. As in Fig. S3, but showing RMSE rather than ACC for the same models, variables, and initial conditions.

![](images/9a2302d4e86536f65e84275fa3ea6853b38537ec1a8bdb98e8f3b1815b84df36.jpg)  
Figure S5: DKE growth in the numerical and AI models of Lorenz 63. Curves show DKE, computed over all three variables, of forecast ensembles for the numerical model (based on integration of Eqs. (S15)–(S17)) and the AI model. The AI model is trained similarly to the one used for Lorenz 96 (Methods and Data): a multilayer perceptron trained on pairs of all three variables with $\Delta t = 1 0 \delta t$ Perturbations are Gaussian, with amplitudes decreasing from a reference value, $\epsilon _ { 0 } .$ . Curves are averaged over 10, 000 initial conditions drawn from a held-out test trajectory, each with a 100- member ensemble; the same initial conditions and perturbations are used for both models. The AI model closely tracks the numerical model’s DKE curves at all amplitudes. They both lack the “real” butterfly efect [21, 22].

![](images/96470f52bf426dd8105c862c9d9e07e54c177961d0ad5cef0f3a9546214ff278.jpg)  
Figure S6: Scale dependence of the forecast errors in the Pangu-Weather models with diferent Δ�. Power spectra �(�) of the 300-hPa error field (prediction minus ERA5) as a function of total wavenumber � and lead time (colors, nonuniformly from 1 h to 3 days; see legend), compared with the spectrum of the full field (black). The top axis gives the corresponding wavelength. Error saturates first at small scales and progressively fills in the larger scales; at a fixed lead time, the error is larger at all scales for the models trained with smaller Δ�. Curves are averaged over the test set initial conditions.

![](images/91620578a1a6509428c1872d8d7369653e800bbf0cf3451c1bcae6264ae85f85.jpg)  
Figure S7: Spatial distribution of Pangu-Weather forecasts’ early DKE growth and the ERA5 precipitation field. For the rotated figure, rows show lead time from 1 h (first row) to 24 h (last row). The first four columns from the left show the Δ� of the oficial Pangu-Weather model. Panels marked “No data” correspond to lead times shorter than the model’s Δ�. As an example, maps of the 300-hPa DKE over parts of the tropical western Pacific are shown for ensembles with the smallest perturbation amplitude $\ \dot { ( \epsilon _ { 0 } / 1 0 ^ { 3 } ) }$ for one representative initial condition. The Δ� = 1 h ensembles show coherent DKE patterns rather than unphysical noise. However, unlike what numerical experiments suggest [19, 33, 34], the regions of largest early DKE growth in the 1-h Pangu-Weather forecasts do not necessarily collocate with precipitation. Taking the 6-h lead time as an example, the DKE pattern in the 1-h model instead resembles a mixture of small-scale features amplified in the 6-h model unde the influence of precipitation.

![](images/ea14db8bf3d85b6016f8ed80c769b6396472015d427cd90455703f1e8d5102c1.jpg)  
Figure S8: Spectra of one-time-step forecasts in the AIWP models of ERA5 and PlaSim GCM. Power spectra of the 300-hPa kinetic energy, � (�), as a function of total wavenumber �, computed with the spherical harmonic transform, for one-time-step (1Δ�) forecasts. Left: The four oficial Pangu-Weather models trained on ERA5 (0.25<sup>◦</sup>) from [2]. Right: Transformer-, SFNO-, and Difusion-based AIWP models of PlaSim GCM, which has a numerical resolution of T42. Curves are averaged over the test set initial conditions. Bottom row zooms into the high wavenumber end for better visualization. Black curves are the reference spectra of the full ERA5 and PlaSim fields. Vertical dashed lines mark each dataset’s spectral truncation (T639 for ERA5, T42 for PlaSim); gray dotted and dashed lines are −5/3 and −3 slope references.

![](images/15f46b5862027416b3ca336d9439c6ed987fdce9b11532e9c20ba78fc553d825.jpg)  
Figure S9: One-time-step error of the forecasting and backcasting AIWP models of ERA5 and PlaSim GCM. RMSE of one-time-step prediction (1Δ�) for U850 (top), Z500, and U250, evaluated over the test set initial conditions (see Methods and Data). Left: ERA5, which also shows forecast accuracy of the oficial Pangu-Weather models (gray; $\Delta t = 1 , 3 , 6$ , and 24 h [2]) and our forecasting (blue) and backcasting (red) Transformers $( \Delta t = 6$ and 24 h). Right: the PlaSim forecasting and backcasting Transformers $( \Delta t = 6$ and 24 h). Boxes span the interquartile range, whiskers the 5th– 95th percentiles, and the horizontal line the median.

![](images/1e5b829396a29fc1b66b54a428bdcf708dda5d59985249ea919b80bf63829a05.jpg)  
Figure S10: Sensitivity of the Pangu-Weather forecasts’ DKE growth to the averaging domain, GPU precision, and initial-perturbation structure. Each panel shows the ensemble DKE at 300 hPa. Columns correspond to the four oficial Pangu-Weather models [2] with $\Delta t = 2 4 , 6 , 3$ and 1 h. (A) The same as Fig. 4A, the reference, which uses DKE averaged over 60<sup>o</sup>S-60<sup>o</sup>N, single precision (FP32) inference, and Perlin-noise initial perturbations. (B) as in (A) but averaged globally. (C) as in (A) but without strict FP32 inference, so that the GPU carries out the internal matrix operations in lower precision (TF32). (D) as in (A) but with Gaussian noise instead of Perlin noise. Colors denote the initial-perturbation amplitudes. Gray curves are the ICON reference simulations, solid for 2.5 km and dashed for 20 km resolutions. Panels (B) and (C) show that the faster DKE growth for small-amplitude perturbations might be observed for the small-Δ� AIWP models, but they can be due to high-latitude unphysical instabilities or noise from lower-precision GPU calculations. (D) shows little sensitivity to the structure of the initial-condition perturbation.

## Caption for Movie S1. Numerical forecasts and backcasts on the Lorenz 63 attractor.

Animation of the Lorenz 63 (Eqs. (S15)-(S17)) states numerically integrated forward and backward from 10 initial conditions that are slightly diferent (same numerical methods as those used for Lorenz 96; see Methods and Data). The Lorenz attractor (thin gray lines) is visualized in the �– �–� phase space. The initial conditions are on the attractor. The forecast trajectories remain on the attractor (due to the phase-space contraction), though they diverge significantly in the long term due to the +0.91 leading Lyapunov exponent. The backcast trajectories quickly leave the attractor, which is now a repeller (due to the phase-space expansion). See the Supplementary Materials for more discussion. The visualization code is generated by Claude Fable 5. The movie is available at https://doi.org/10.5281/zenodo.22062019.