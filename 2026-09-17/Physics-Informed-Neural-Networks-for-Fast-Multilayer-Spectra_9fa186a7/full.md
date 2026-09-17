# Physics-Informed Neural Networks for Fast Multilayer Spectral Inversion of Hα 6562.8 <sup>˚</sup>A and Ca II 8542.1 <sup>˚</sup>A Spectra

Ziyang Zhang,<sup>1</sup> Qin Li,<sup>2</sup> Vasyl B. Yurchyshyn,<sup>2,</sup> <sup>3</sup> Kangwoo Yi,<sup>2</sup> Haimin Wang,<sup>2,</sup> <sup>3</sup> Wenda Cao,<sup>2,</sup> <sup>3</sup> and Bo Shen<sup>1,</sup> <sup>4</sup>

<sup>1</sup>Department of Mechanical and Industrial Engineering, New Jersey Institute of Technology, Newark, NJ, USA

<sup>2</sup>Department of Physics, New Jersey Institute of Technology, Newark, NJ, USA

<sup>3</sup>Big Bear Solar Observatory, New Jersey Institute of Technology, Big Bear City, CA, USA

<sup>4</sup>Department of Data Science, New Jersey Institute of Technology, Newark, NJ, USA

## ABSTRACT

Strong chromospheric absorption lines such as Hα 6562.8 <sup>˚</sup>A and Ca II 8542.1 <sup>˚</sup>A provide important diagnostics of plasma dynamics and thermal structure in the solar chromosphere. Multilayer spectral inversion (MLSI) ofers a physically interpretable framework for modeling these lines with a finite number of radiative-transfer layers, but conventional MLSI still requires pixel-by-pixel nonlinear least-squares fitting and is therefore costly for large imaging spectroscopic data sets. In this work, we introduce a physics-informed neural-network (PINN) framework to accelerate MLSI while retaining its analytic radiative-transfer formulation. The network predicts MLSI parameters from observed line profiles, then passes the predicted parameters through the diferentiable MLSI forward model to synthesize spectra. We train the model in two stages: the first stage uses only the spectral reconstruction loss, followed by a second fine-tuning stage that combines spectral consistency with parameter-space supervision from conventional MLSI results for a single reference image. This strategy reduces the need for large precomputed training sets while preserving the physical interpretability of the MLSI parameters. We apply the method to FISS Hα 6562.8 <sup>˚</sup>A and Ca II 8542.1 <sup>˚</sup>A observations of both quiet-Sun and active-region targets. The MLSI-PINN parameter maps reproduce the main spatia structures of direct MLSI inversions, taken with the Fast Imaging Solar Spectrograph (FISS) of the Goode Solar Telescope (GST). Across all parameter comparisons shown for the representative quiet-Sun and active-region rasters, the arithmetic mean of the pixel-wise Pearson correlation coeficients is 0.933. The reconstructed spectra also remain close to both the observed profiles and conventional MLSI reconstructions. With the implementations and hardware used in this study, MLSI-PINN requires approximately 5–15 s to process one raster after training, compared with approximately 3–5 min for conventional MLSI, corresponding to an inference speedup of about 12–60 times. These results show that the two-stage physics-informed neural network provides a substantially faster, physically constrained approximation to MLSI without a dominant loss in reconstruction quality, improving the feasibility of applying MLSI to large chromospheric imaging-spectroscopy data sets.

Keywords: Solar chromosphere, Radiative transfer equation, Spectroscopy, Neural networks

## 1. INTRODUCTION

High-resolution spectroscopy of chromospheric lines provides diagnostics of the thermal and dynamic structure of the lower solar atmosphere through analysis of line profiles. Among the commonly used diagnostics, Hα and Ca II 8542.1 <sup>˚</sup>A lines are particularly important because their profiles sample atmospheric conditions from the upper photosphere into the chromosphere and carry complementary information on opacity, line-of-sight motions, and line broadening (Leenaarts et al. 2009, 2012; Carlsson et al. 2019). The Fast Imaging Solar Spectrograph (FISS) at the Goode Solar Telescope provides a complete spectral profile over a defined field of view (Goode & Cao 2012; Chae et al.

2013, 2021). Such observations contain substantially more information than monochromatic images, but quantitative analysis of the atmospheric parameters requires an appropriate inversion model.

Inversions of Hα and Ca II spectra have been used to investigate the temperature and velocity structure of umbral flashes (Henriques et al. 2017; Felipe & Esteban Pozuelo 2019). Related analyses have constrained the thermal, velocity, and line-broadening properties of chromospheric fibrils and transient brightenings (Kianfar et al. 2020; Vissers et al. 2019). With full-Stokes Ca II observations, spectropolarimetric inversions can also constrain magnetic fields in plage (Pietrow et al. 2020). Multiline spectropolarimetric inversions of DKIST/ViSP observations have also been used to diagnose the thermal and dynamic structure of flare ribbons (Yadav et al. 2025). These applications show that the shape of the spectral line carries thermal and dynamic information in addition to morphology alone. The detailed shape of the spectral line reflects the combined efects of temperature, velocity, opacity, and radiative transfer, which also provides constraints on the atmospheric state. At the same time, chromospheric line formation is suficiently complex so that extracting this information is rarely a simple measurement problem. More physically detailed inversion models generally require more expensive radiative-transfer calculations and nonlinear optimization, which can become a limiting factor for large spectroscopic datasets.

These computational demands motivate a practical compromise between the physical detail of the adopted atmospheric model and computational feasibility. Compact analytic approaches, including Milne–Eddington formulations (Unno 1956; Skumanich & Lites 1987) and cloud-model inversions (Beckers 1964; Tziotziou 2007), represent the atmosphere with a small number of parameters and can be applied eficiently to large data volumes. Such models have also been used to analyze chromospheric structures (Chae et al. 2014). Their eficiency, however, comes from restrictive assumptions about the atmospheric structure. More flexible stratified inversion codes, including SIR and NICOLE (Ruiz Cobo & del Toro Iniesta 1992; Socas-Navarro et al. 2015) and STiC and SNAPI (de la Cruz Rodr´ıguez et al. 2019; Mili´c & van Noort 2018), can retrieve height-dependent atmospheric quantities. Similar non-local thermodynamic equilibrium methods have also been applied to Mg II spectra (de la Cruz Rodr´ıguez et al. 2016). All iterative inversion methods require repeated forward-model evaluations; the main diferences in computational cost arise from the complexity of the forward model and the number of free parameters. (Ruiz Cobo & del Toro Iniesta 1992; Socas-Navarro et al. 2015; de la Cruz Rodr´ıguez et al. 2016; de la Cruz Rodr´ıguez et al. 2019; Mili´c & van Noort 2018). For high-cadence rasters and long observing sequences, this repeated pixel-by-pixel fitting can dominate the analysis cost.

Multilayer spectral inversion (MLSI) was developed to model strong absorption lines that form over a broad height range from the photosphere to the chromosphere. It combines a Milne–Eddington description of the photospheric contribution with a cloud-model description of the chromospheric contribution within a finite number of radiativetransfer layers (Chae et al. 2020). The method returns parameters with explicit physical meanings, including source functions, Doppler velocities, Doppler widths, opacity-related quantities, damping, and total chromospheric optica thickness. A subsequent extension allowed the chromospheric absorption profile to vary with height (Chae et al. 2021). For Hα and Ca II 8542, this formulation retains an analytic forward model while allowing more vertical structure than a single-layer cloud model. Nevertheless, conventional MLSI still relies on constrained nonlinear least-squares fitting at each pixel. Thus, although MLSI is computationally lighter than stratified inversion under non-local thermodynamic equilibrium, applying it repeatedly to many FISS rasters or long time sequences remains costly.

Machine-learning methods provide a possible route to accelerate this type of inverse problem. Once trained, a neural network can replace repeated iterative fitting with a direct evaluation of the inverse mapping. Neural networks have been used to approximate spectroscopic and spectropolarimetric inversion mappings, and machine-learning methods are increasingly used in solar diagnostic applications more broadly (Asensio Ramos & D´ıaz Baso 2019; Sainz Dalda et al. 2019; Osborne et al. 2019; Asensio Ramos et al. 2023). The SPIn4D project provides radiative-MHD simulations and synthetic Stokes profiles to support the development of deep-learning spectropolarimetric inversions (Yang et al. 2024). Fast numerical approaches have also been developed for related inverse problems, such as diferential emission measure reconstruction (Cheung et al. 2015). In the MLSI context, Lee et al. (2022) showed that a supervised neural network can reproduce MLSI outputs for Hα and Ca II 8542 much faster than direct nonlinear fitting. Although its inference is fast, the model is trained exclusively against precomputed MLSI parameter labels and therefore requires a suficiently large and representative set of conventional inversion results. Its performance consequently depends on the coverage and consistency of the training labels, and ambiguities or systematic biases in the reference inversions may be inherited by the network.

Physics-informed neural networks (PINN) provide a way to include the forward model more directly in the learning process (Raissi et al. 2019; Karniadakis et al. 2021; Gnanasambandam et al. 2023). Instead of training only against parameter labels, the predicted parameters can be evaluated through the physical model that maps them back to observables. In solar physics, related ideas have been applied to coronal magnetic-field modeling, neural-field-based spectropolarimetric inference, and Milne–Eddington inversion (Jarolim et al. 2023; D´ıaz Baso et al. 2025; Jarolim et al. 2025; Li et al. 2025). Physics-informed learning has also been used to reconstruct three-dimensional magnetic fields from spectropolarimetric inversion results (Yang et al. 2025), whereas our work focuses on intensity-only spectroscopic inversion. MLSI is a natural candidate for this strategy because its forward calculation is analytic and diferentiable. The MLSI parameters predicted by a neural network can be passed through the multilayer radiative-transfer mode during training, so that the loss is defined at least in part by the mismatch between observed and synthesized spectra.

In this work, we develop an MLSI-PINN framework for fast inversion of FISS Hα and Ca II 8542 spectra taken with the Fast Imaging Solar Spectrograph (FISS) of Goode Solar Telescope (GST). The network takes an observed spectral profile as input and predicts an MLSI parameter vector. The embedded MLSI forward model then synthesizes the corresponding emergent line profile, and the spectral reconstruction error provides the primary physics-based constraint. To reduce degeneracy among multilayer solutions, we use a second training stage in which a limited conventional MLSI reference raster supplies parameter-space supervision for selected quantities. This design is intended to preserve the interpretability of MLSI parameters while reducing the need for repeated pixel-by-pixel nonlinear optimization. The method should therefore be viewed not as a replacement for detailed radiative-transfer inversions, but as an acceleration strategy for MLSI-style analysis of large FISS data sets and time-dependent observations.

The remainder of this paper is organized as follows. Section 2 describes the MLSI forward model and the two-stage MLSI-PINN training framework. Section 3 presents the FISS observations and preprocessing procedures. Section 4 evaluates spectral reconstruction, parameter-space agreement, temporal behavior, and computational performance. Section 5 summarizes the implications and limitations of the proposed approach.

## 2. METHOD

## 2.1. MLSI Forward Model

The analytic forward model of multilayer spectral inversion (MLSI) provides the physical component of our method. In the MLSI-PINN framework, a neural network maps an observed spectral profile to a set of MLSI parameters, which are then passed through the analytic radiative-transfer operator to produce a synthetic spectrum. Direct comparison between the synthesized and observed profiles provides the physics-based training signal. In this way, the network can be optimized eficiently while the inferred quantities retain the physical interpretation of the original MLSI parameters.

To establish the notation used below, we briefly summarize the MLSI forward calculation adopted in this work, following the multilayer formulation of Chae et al. (2020, 2021). The calculation begins with the formal solution of the radiative-transfer equation,

$$
I _ { \lambda } = \int _ { 0 } ^ { \infty } S ( t _ { \lambda } ) e ^ { - \tau _ { \lambda } } d \tau _ { \lambda } ,\tag{1}
$$

where $I _ { \lambda }$ is the emergent specific intensity, $S ( \cdot )$ is the source function, and $\tau _ { \lambda }$ is the optical depth at wavelength λ. MLSI approximates this continuous atmosphere with a small number of layers, allowing the emergent spectrum to be expressed analytically in terms of a finite set of physically interpretable parameters.

The atmosphere is represented by a photospheric layer and two chromospheric layers. The photospheric background profile is described by a Voigt-like absorption profile,

$$
r ( \lambda ) = 1 + \eta \mathcal { V } \big ( \lambda ; \Delta \lambda ( v _ { p } ) , w _ { p } , a \big ) ,\tag{2}
$$

which gives the background intensity

$$
I _ { 2 } ( \lambda ) = S _ { 2 } + \frac { S _ { p } - S _ { 2 } } { r ( \lambda ) } .\tag{3}
$$

Here $r ( \lambda )$ is the total-to-continuum absorption factor, $\mathcal { V } ( \cdot )$ is the normalized Voigt profile, and $\Delta \lambda ( v _ { p } )$ is the wavelength shift associated with the photospheric line-of-sight velocity $v _ { p }$ . The quantities $\eta , w _ { p }$ , and a denote the line-to-continuum opacity ratio, the photospheric Doppler width, and the dimensionless damping parameter, respectively. The continuum optical-depth coordinate is denoted by $\tau _ { c } ,$ with $S _ { 2 }$ and $S _ { p }$ representing the source functions at $\tau _ { c } = 0$ and $\tau _ { c } = 1$ respectively.

The two chromospheric layers are subsequently applied to this photospheric background. Let $t _ { 0 }$ denote the total chromospheric optical thickness at line center. The source functions $S _ { 0 } , \ S _ { 1 }$ , and $S _ { 2 }$ are specified at the line-center optical-depth positions $0 , \tau _ { 0 } / 2$ , and $\tau _ { 0 } .$ respectively. The upper chromospheric layer is characterized by the line-of-sight velocity $v _ { 1 }$ and Doppler width $\omega _ { 1 }$ , whereas the lower chromospheric layer is characterized by v<sub>2</sub> and $\omega _ { 2 }$ . Gaussian-like absorption profiles are adopted for both chromospheric layers.

![](images/a09f38cc5fa06cd834e0b31a39f280c1e8a5d0fe2d5b70629156c47f113886aa.jpg)  
Figure 1. Schematic diagram of the two-stage MLSI-PINN framework. In stage 1, the observed spectral profile is mapped to MLSI parameters by a physics-informed neural network and passed through the analytic MLSI forward model to synthesize a spectrum. The network is trained using the spectral reconstruction loss between the observed and synthesized profiles. In stage 2, the stage-1 checkpoint is fine-tuned using a combined loss that includes both the spectral reconstruction loss and a parameter-space loss computed from selected conventional MLSI reference parameters.

The complete 13-dimensional MLSI parameter vector is written as

$$
p = ( v _ { p } , \log \eta , \log w _ { p } , \log a , \log S _ { p } , \log S _ { 2 } , \log \tau _ { 0 } , v _ { 2 } , v _ { 1 } , \log w _ { 2 } , \log w _ { 1 } , \log S _ { 1 } , \log S _ { 0 } ) .\tag{4}
$$

Using this parameter vector, the final emergent spectrum can be expressed compactly as

$$
I _ { 0 } ( \lambda ) = \mathcal { F } _ { \mathrm { M L S I } } ( \pmb { p } ) ,\tag{5}
$$

where $\mathcal { F } _ { \mathrm { M L S I } } ( \cdot )$ denotes the complete deterministic MLSI forward operator. Herein, the photospheric background $I _ { 2 } ( \lambda )$ is first propagated through the lower chromospheric layer to obtain the intermediate profile $I _ { 1 } ( \lambda )$ and then through the upper chromospheric layer to obtain the final emergent profile $I _ { 0 } ( \lambda )$

## 2.2. Physics-Informed Neural Network Framework

As illustrated in Figure 1, we formulate the acceleration of MLSI as a physics-informed inverse problem. Instead of solving an independent nonlinear least-squares problem for every spatial pixel, a neural network learns an approximate mapping from observed spectral profiles to MLSI parameters. The analytic MLSI forward model remains embedded

in the training procedure, allowing the predicted parameters to be evaluated through the spectra they produce rather than solely through an empirical regression target. Once training is complete, the model can infer the MLSI parameters of an observed profile with a single network forward pass.

We denote the neural network by $\mathcal { N } _ { \theta }$ , where θ represents its trainable weights. Given a normalized observed spectra profile $I _ { \mathrm { o b s } } ( \lambda )$ , the stage-1 network predicts the complete MLSI parameter vector,

$$
\pmb { p } ^ { ( 1 ) } = \mathcal { N } _ { \theta } \big ( I _ { \mathrm { o b s } } ( \lambda ) \big ) .\tag{6}
$$

Before entering the MLSI forward operator, the predicted parameters are mapped to their physically allowed ranges. Positive quantities, including optical thicknesses, Doppler widths, and opacity ratios, are represented in logarithmic form, while bounded quantities are restricted using scaled nonlinear transformations. These constraints prevent the network from exploring nonphysical regions of parameter space and preserve the interpretation of its outputs as MLSI atmospheric parameters.

The spectrum corresponding to the predicted parameter vector is synthesized using the analytic MLSI forward model,

$$
I _ { \mathrm { s y n } } ( \lambda ) = \mathcal { F } _ { \mathrm { M L S I } } \big ( \boldsymbol { p } ^ { ( 1 ) } \big ) ,\tag{7}
$$

where $I _ { \mathrm { s y n } } ( \lambda )$ is the synthesized emergent profile. The spectral reconstruction loss is then defined as

$$
\mathcal { L } _ { \mathrm { s p e c } } = \frac { 1 } { N _ { \lambda } } \sum _ { \lambda } \left[ I _ { \mathrm { s y n } } ( \lambda ) - I _ { \mathrm { o b s } } ( \lambda ) \right] ^ { 2 } ,\tag{8}
$$

where $N _ { \lambda }$ is the number of wavelength samples included in the loss. In practice, the loss is evaluated over a central wavelength range around the line core because the far-wing and continuum samples can contain isolated spikes or residual calibration errors that are not adequately represented by the MLSI line-formation model. The selected wavelength range contains the strongest chromospheric contribution and therefore provides the most relevant constraint on the MLSI parameters.

The spectral reconstruction loss alone does not always determine every MLSI parameter uniquely, since diferent parameter combinations may produce similar emergent profiles. To reduce this ambiguity, we train the model in two stages. During the first stage, the network is optimized using only the physics-based spectral reconstruction loss,

$$
\mathcal { L } ^ { ( 1 ) } = \mathcal { L } _ { \mathrm { s p e c } } .\tag{9}
$$

This stage establishes a physics-informed baseline mapping from the observed spectra to the MLSI parameter space without requiring a large set of precomputed MLSI reference parameters.

In the second stage, the stage-1 model provides the initialization, and its objective is augmented with a parameterspace supervision term derived from conventional MLSI inversions of a single reference raster. The refinement focuses on parameters for which the stage-1 predictions exhibit systematic deviations from the conventional inversion. Let Ω denote the selected subset of MLSI parameters. The parameter loss is evaluated only over this subset,

$$
\mathcal { L } _ { \mathrm { p a r a m } } = \frac { 1 } { \left| \Omega \right| } \sum _ { i \in \Omega } \left( \pmb { p } _ { i } - \pmb { p } _ { i } ^ { \mathrm { M L S I } } \right) ^ { 2 } ,\tag{10}
$$

where $\mathbf { \nabla } _ { \pmb { p } _ { i } }$ is the stage-2 prediction of parameter $i , p _ { i } ^ { \mathrm { M L S I } }$ is the corresponding conventional MLSI reference value, and $| \Omega |$ is the number of selected parameters. The predicted and reference subvectors are denoted by $\pmb { p } _ { \Omega } ^ { ( 2 ) }$ and $p _ { \Omega } ^ { \mathrm { M L S I } }$ respectively.

The complete stage-2 objective is

$$
\begin{array} { r } { \mathcal { L } ^ { ( 2 ) } = \alpha \mathcal { L } _ { \mathrm { s p e c } } + \beta \mathcal { L } _ { \mathrm { p a r a m } } , } \end{array}\tag{11}
$$

where α and $\beta$ control the relative contributions of the spectral and parameter-space constraints.

This two-stage procedure retains the spectral constraint imposed by the analytic forward model while using a limited amount of conventional MLSI output to refine selected parameters. As a representative implementation, the neural network used for the Hα active-region data is a fully connected multilayer perceptron with hidden layers containing 256, 128, and 64 neurons. Batch normalization and SiLU activations are used in the hidden layers, and a Tanh activation is applied after the third hidden layer. The final linear layer returns the raw MLSI parameter vector. The

Ca II 8542 and quiet-Sun models follow the same general architecture and training procedure, with minor adjustments to the loss windows and parameter ranges to account for diferences in spectral sampling and line properties.

For this representative Hα setting, stage 1 is trained for 50 epochs using the Adam optimizer with a learning rate of $2 \times 1 0 ^ { - 4 }$ , a batch size of $2 5 6 ,$ a weight decay of $1 0 ^ { - 5 }$ , and a dropout rate of 0.0. The stage-1 spectral reconstruction loss is evaluated over wavelength indices 180–490. Stage 2 is initialized from the stage-1 checkpoint and fine-tuned for 20 epochs using a learning rate of $2 \times 1 0 ^ { - 5 }$ and zero weight decay. During this stage, the spectral loss is evaluated over a narrower line-core window corresponding to indices 235–345 and is combined with the parameter-space loss using equal weights. The refinement subset is

$$
\Omega = \{ \log S _ { 1 } , \log S _ { 0 } , v _ { 1 } , v _ { 2 } , \log w _ { 1 } , \log w _ { 2 } \} .\tag{12}
$$

Parameters outside Ω retain their stage-1 values, whereas those within Ω are replaced by the refined predictions $ { \boldsymbol { p } } _ { \Omega } ^ { ( 2 ) }$ The photospheric velocity $v _ { p }$ is handled separately: it is estimated from the observed profile using the conventional line-center proxy and inserted into the final parameter vector rather than being freely refined by the network. The assembled parameter vector is then passed through $\mathcal { F } _ { \mathrm { M L S I } }$ to produce the reconstructed spectrum.

Training is performed with mini-batches of spectral profiles sampled from the reference raster, without a validation split or early stopping. After training, the model can be applied to additional observations of the same spectral line obtained on the same observing day. Inference requires only a neural-network forward pass, followed by the analytic MLSI forward calculation when reconstructed spectra are required. This replaces repeated pixel-by-pixel nonlinear fitting with a fixed trained inverse mapping while preserving the physical interpretation of the MLSI parameters.

## 3. DATA

## 3.1. FISS Observations

The observational data used in this study were obtained with the Fast Imaging Solar Spectrograph FISS (Chae et al. 2013) installed on the 1.6 m Goode Solar Telescope at Big Bear Solar Observatory (BBSO). FISS is a dualchannel imaging spectrograph that simultaneously records the Hα and Ca II 8542 <sup>˚</sup>A spectral bands. For the observing configurations used in this study, the spatial sampling is approximately $0 . 1 6 ^ { \prime \prime }$ per pixel, and the raster cadence ranged from approximately 20 to 35 s.

We analyze two FISS observing sequences. The first sequence is a quiet-Sun observation obtained on 2021 August 7 from 16:40 to 17:23 UT. It contains 105 image frames, and the processed spectral cubes used in this work have dimensions of $5 0 2 \times 2 4 6 \times 2 0 0$ , where the three axes correspond to wavelength, slit position, and scan position, respectively. The second sequence is an active-region observation obtained on 2023 August 14 from 17:25 to 18:08 UT. It contains 83 image frames, with processed cube dimensions of $5 1 2 \times 2 5 6 \times 1 5 0$ . Both sequences include simultaneous Hα and Ca II 8542 spectra.

The spectral coverage is approximately 9.7 <sup>˚</sup>A for Hα and 12.9 <sup>˚</sup>A for Ca II 8542, with spectral samplings of about 0.019 <sup>˚</sup>A and 0.026 <sup>˚</sup>A, respectively. Each image therefore contains on the order of $1 0 ^ { 4 } – 1 0 ^ { 5 }$ spatially resolved line profiles. These data provide both quiet-Sun and active-region examples for evaluating the proposed MLSI-PINN framework. A summary of the observing sequences is given in Table 1.

Table 1. Summary of the FISS observing sequences used in this study.
<table><tr><td>Obs.</td><td>Target</td><td>Date</td><td>UT</td><td>Frames</td><td>Cube Size</td></tr><tr><td>1</td><td>Quiet Sun</td><td>2021 Aug 7</td><td>16:40-17:23</td><td>105</td><td> $5 0 2 \times 2 4 6 \times 2 0 0$ </td></tr><tr><td>2</td><td>Active Region</td><td>2023 Aug 14</td><td>17:25-18:08</td><td>83</td><td> $5 1 2 \times 2 5 6 \times 1 5 0$ </td></tr></table>

Note. Both observing sequences include simultaneous Hα and Ca II 8542 spectra. Cube size is given as $N _ { \lambda } \times N _ { y } \times N _ { x }$

## 3.2. Data Reduction and Preprocessing

The raw FISS data are reduced with the standard FISS reduction pipeline implemented in the FISSPy package, following the procedures described by Chae et al. (2013). The reduction first removes detector signatures through dark and bias subtraction and flat-field correction. It then corrects the spectrogram geometry, including the spectra tilt and the geometric distortion between the dispersion and slit directions, so that the wavelength and spatial axes are placed on a rectified grid. Wavelength calibration is applied to assign a physical wavelength scale to each spectra channel. The output of this procedure is a calibrated spectral cube for each image frame and for each spectral line.

Because Hα and Ca II 8542 have diferent wavelength coverages and spectral samplings, the two lines are processed and modeled separately. For each line, the reduced spectra are represented on a fixed wavelength grid before being supplied to the neural network. This keeps the network input dimension fixed and ensures that the MLSI forward operator is evaluated on the same wavelength samples used by the observations. Only the intensity profile is used as the network input in this work.

For each observing sequence and spectral line, one reference image is selected for training. The spectra from this image are used to train the first-stage physics-informed model. The same reference image is also inverted with the conventional MLSI procedure, and the resulting MLSI parameter maps are used as parameter-space supervision targets in the second training stage. The remaining image frames are not used to optimize the network; they are reserved for inference and evaluation after the model has been trained.

Before training, each spectral profile is normalized by its continuum intensity,

$$
\tilde { I } ( \lambda ) = \frac { I ( \lambda ) } { \langle I ( \lambda ) \rangle _ { \mathrm { c o n t i n u u m } } } ,\tag{13}
$$

where $\langle I ( \lambda ) \rangle _ { \mathrm { c o n t i n u u m } }$ is the mean intensity over selected line-continuum wavelength samples. The same normalization is applied to the spectra used in the reconstruction loss and during inference. This profile-wise normalization brings the input intensities to a comparable scale, typically close to unity in the line continuums, and improves the numerica conditioning of neural-network training.

## 4. RESULTS

In this section, we evaluate the MLSI-PINN results using the FISS Hα and Ca II 8542 observations described in Section 3. The conventional MLSI inversion is used as the reference solution for parameter-space comparison, while the observed spectra provide the reference for spectral-profile reconstruction. We first describe the spectral morphology of the quiet-Sun and active-region observations, then compare the inferred MLSI parameters in map space and pixelby-pixel parameter space. We subsequently examine representative spectral reconstructions and temporal evolution, and finally summarize the computational performance of the proposed framework.

## 4.1. FISS Observations and Spectral Morphology

Figure 2 presents representative images from the quiet-Sun and active-region FISS observations. For both Hα and Ca II 8542, the panels sample the same seven wavelength ofsets from −4 to +4 <sup>˚</sup>A. The line-continuum images mainly trace photospheric and lower-atmospheric structures, whereas the line-center images emphasize chromospheric morphology.

The quiet-Sun observations show relatively moderate contrast and spatially distributed fine structure. In comparison, the active-region images contain a prominent sunspot, stronger intensity gradients, and more structured line-core features. The active-region profiles therefore span a broader range of line depths, Doppler shifts, and asymmetries, providing a more demanding test of the inversion model. The wavelength-dependent changes in both regions also illustrate why the full spectral profile, rather than a single monochromatic image, is required for quantitative inversion.

Having established the observational morphology and spectral complexity of the two data sets, we next examine whether the MLSI-PINN model preserves the physical-parameter structure obtained from conventional MLSI.

## 4.2. Parameter-Space Agreement with Conventional MLSI

We first compare the spatial distributions of the inferred MLSI parameters. This comparison is important because a good spectral reconstruction alone does not guarantee that the inferred parameters reproduce the spatially coherent structures obtained by conventional MLSI.

Figure 3 shows the quiet-Sun parameter maps for Hα and Ca II 8542. The MLSI-PINN predictions reproduce the large-scale morphology of the conventional MLSI maps for both lines. The source-function and Doppler-width parameters preserve the principal quiet-Sun contrast patterns, while the velocity-related quantities show somewhat larger local diferences, consistent with their sensitivity to small line-center shifts and multilayer parameter degeneracy.

The active-region parameter maps are shown in Figure 4. Compared with the quiet-Sun case, the active-region maps contain sharper structures and stronger local variations, particularly in the line-core source functions and velocityrelated parameters. Even under these more complex conditions, the MLSI-PINN results recover the dominant activeregion morphology in both spectral lines. The agreement is strongest for parameters controlling the overall line depth and width, whereas the remaining diferences are most apparent in quantities associated with local line asymmetries and small line-core shifts.

![](images/04b9cf698454f14c12f051a74ab61595a932cddf9a83d5ac92e776c5c3933862.jpg)  
Figure 2. Representative quiet-Sun and active-region FISS monochromatic images. Rows (A) and (B) show quiet-Sun Hα and Ca II 8542, respectively, while rows (C) and (D) show the corresponding active-region observations. Each row samples wavelength ofsets of −4, −1, −0.5, 0, +0.5, +1, and +4 <sup>˚</sup>A relative to the corresponding line center. Display ranges are selected to retain the morphological structure in both data sets and are not intended for quantitative comparison of absolute intensit between rows.

To quantify the pixel-by-pixel agreement, Figures 5 and 6 compare the MLSI-PINN predictions with conventiona MLSI parameters for representative quiet-Sun and active-region rasters. In each panel, the horizontal axis gives the direct MLSI value, the vertical axis gives the MLSI-PINN prediction, and the dashed diagonal indicates the oneto-one relation. The correlation coeficients reported in these figures are calculated from individual representative rasters, whereas Table 2 reports statistics aggregated over all evaluated inference rasters. Small diferences between the figure-level and sequence-level coeficients are therefore expected.

Figure 5 combines the quiet-Sun Hα and Ca II 8542 comparisons. The source-function parameters form compact, positively correlated distributions, with CC values of 0.883–0.938 for Hα and 0.929–0.980 for Ca II 8542. Lower agreement is found for Hα $v _ { 2 }$ and log $w _ { 2 }$ , with CC values of 0.871 and 0.825, respectively, and for Ca II log w<sub>1</sub>, with a CC of 0.774. The chromospheric velocities and selected Doppler widths exhibit broader distributions, consistent with their greater sensitivity to line-core shifts and degeneracy among multilayer parameters. Although the Dopplerwidth parameters generally have small absolute errors, their relatively narrow intrinsic ranges make the correlation and $R ^ { 2 }$ values more sensitive to small systematic ofsets. The high agreement of $v _ { p } ,$ with CC values of 0.998 for Hα and 1.000 for Ca $\mathrm { I I } ,$ primarily reflects the shared line-center proxy treatment rather than an independently learned neural-network prediction.

![](images/468dea405adf201bd13e404fdd10b60a020216a8065a4c45b90c91972362e1cf.jpg)  
Figure 3. Quiet-Sun parameter-map comparison for Hα and Ca II 8542. For each parameter, the conventional MLSI result and the MLSI-PINN prediction are shown using the same color scale.

![](images/62b71d5a28fde0aa06574d30044be0e219650fab8b72ca3ef232401e3cbf17f6.jpg)  
Figure 4. Active-region parameter-map comparison for Hα and Ca II 8542. For each parameter, the conventional MLSI result is compared with the MLSI-PINN prediction using a shared color scale. The stronger spatial gradients in the active-region maps provide a more stringent test of whether the neural-network inversion preserves the parameter morphology of conventional MLSI.

Quiet Sun Hα Parameter Correlations  
![](images/16f926acea90b57daeba4e998391faa8ae4cf1642152a50761499f5bf3495777.jpg)  
Figure 5. Quiet-Sun parameter-space comparison between MLSI-PINN and conventional MLSI for one representative inference raster. The upper two rows show the Hα parameters, and the lower two rows show the Ca II 8542 parameters. The horizontal and vertical axes give the conventional MLSI and MLSI-PINN values, respectively. The dashed diagonal marks the one-to-one relation, and each panel reports the pixel-by-pixel Pearson correlation coeficient. The high agreement of $v _ { p }$ primarily reflects the shared line-center proxy treatment.

Figure 6 presents the corresponding active-region comparisons. The distributions span broader parameter ranges than in the quiet Sun, reflecting the stronger line-profile variability and spatial gradients of the active region. The source-function parameters retain high CC values of 0.938–0.984 for Hα and 0.972–0.993 for Ca II 8542. Greater scatter occurs in Ca II $v _ { 1 } ,$ log $w _ { 2 } .$ , and log $w _ { 1 }$ , whose CC values are 0.855, 0.853, and 0.812, respectively, while Hα log $w _ { 1 }$ has a CC of 0.843. This behavior is consistent with the reduced uniqueness of these chromospheric parameters within the multilayer forward model. Nevertheless, clear correlations remain for most parameters, particularly the source functions and photospheric Doppler widths. Overall, the active-region results indicate that the network preserves the main MLSI parameter-space structure even when the observed profiles contain stronger line asymmetries and a broader range of Doppler shifts.

Active Region Hα Parameter Correlations  
![](images/b986c811bdf1707c5d1fbb474e1783800b245087213da8514bd9ba35ea9c6941.jpg)  
Figure 6. Active-region parameter-space comparison between MLSI-PINN and conventional MLSI for one representative inference raster. The upper two rows show the Hα parameters, and the lower two rows show the Ca II 8542 parameters. The dashed diagonal indicates the one-to-one relation. The broader distributions relative to the quiet-Sun case reflect the larger range of line depths, Doppler shifts, and profile asymmetries in the active region.

Together, Figures 5 and 6 show that MLSI-PINN reproduces the parameter-space structure of conventional MLSI for both spectral lines and solar targets. The CC values range from 0.774 to 1.000 for the representative quiet-Sun raster and from 0.812 to 1.000 for the active-region raster. The comparatively lower agreement is concentrated in chromospheric velocities and selected Doppler widths, which are less uniquely constrained by the emergent spectra profiles.

Table 2 summarizes the parameter statistics and quantitative agreement with the conventional MLSI solution. The means and standard deviations are computed from the MLSI-PINN parameter maps, while the mean absolute error (MAE), Pearson correlation coeficient (CC), and coeficient of determination (R<sup>2</sup> score) are computed relative to the direct MLSI parameters. The table includes all evaluated inference rasters, whereas the CC values in Figures 5 and 6 correspond to single representative rasters.

The source-function parameters are recovered most consistently across both spectral lines and both solar targets. For the quiet-Sun case, the CC values of log $S _ { p } ,$ log $S _ { 2 } .$ log $S _ { 1 }$ , and log $S _ { 0 }$ are generally close to or above 0.9 for both Hα and Ca II 8542, and the corresponding MAEs remain small compared with the parameter dispersions. The same trend is observed in the active-region case, where the source-function parameters remain strongly correlated with the direct MLSI solution despite the broader range of line depths and local spectral asymmetries. This indicates that the spectral reconstruction loss and limited parameter supervision preserve the source-function structure inferred by conventional MLSI.

The photospheric velocity $v _ { p }$ also shows very high agreement, but this parameter should be interpreted separately. In the MLSI-PINN output, $v _ { p }$ is estimated from the observed line-center proxy, whereas in conventional MLSI it is obtained as part of the full-profile fitting. Both quantities are controlled primarily by the photospheric line-center shift and therefore remain highly consistent. However, because the two estimates are not produced by exactly the same calculation, the CC is not necessarily equal to unity. Small diferences can arise from line-center measurement uncertainty, residual wavelength-calibration errors, finite spectral sampling, and local profile asymmetries.

The chromospheric velocity parameters exhibit larger scatter than the source-function parameters. This behavior is expected because $v _ { 1 }$ and $v _ { 2 }$ mainly afect line-core shifts and asymmetries, which can be partially compensated by changes in Doppler width, opacity, and source-function gradients in a multilayer model. Even so, the velocity correlations remain positive in all cases, showing that the MLSI-PINN captures the dominant velocity structure. The active-region Hα case gives particularly strong agreement for $v _ { 2 } .$ , while the Ca II 8542 velocity parameters show larger errors, consistent with the broader distributions in Figure $6 .$

The Doppler-width parameters show a diferent behavior. Their MAEs are generally small in absolute value, particularly for log $w _ { p } .$ , while some of their $R ^ { 2 }$ values are more modest. This does not necessarily imply a large physical discrepancy. Instead, it reflects the narrow intrinsic dynamic range of several width parameters: when the reference quantity varies only weakly across the field of view, even a small residual ofset can noticeably reduce $R ^ { 2 }$ . The width parameters should therefore be assessed using the MAE and CC together with $R ^ { 2 }$ , rather than from the coeficient of determination alone.

Table 2. Means and standard deviations of the MLSI-PINN physical parameters, with MAE, CC, and $R ^ { 2 }$ score relative to direct MLSI calculations.
<table><tr><td rowspan="2">Region</td><td rowspan="2">Physical Parameter</td><td colspan="2">Mean ± Standard Deviation</td><td colspan="2">MAE</td><td colspan="2">CC</td><td colspan="2"> $R ^ { 2 }$  Score</td></tr><tr><td>Hα</td><td> $\mathrm { { C a ~ I I } }$ </td><td>Hα</td><td> $\mathrm { { C a ~ I I } }$ </td><td>Hα</td><td>Ca II</td><td>Hα</td><td>Ca II</td></tr><tr><td rowspan="9"> $\mathrm { Q S }$ </td><td>log  $S _ { p } \ [ I _ { 0 } ]$ </td><td> $0 . 0 3 7 \pm 0 . 0 2 8$ </td><td> $- 0 . 0 0 6 \pm 0 . 0 2 0$ </td><td>0.008</td><td>0.006</td><td>0.943</td><td>0.937</td><td>0.887</td><td>0.857</td></tr><tr><td> $\log S _ { 2 } \ [ I _ { 0 } ]$ </td><td> $- 0 . 2 4 0 \pm 0 . 0 2 0$ </td><td> $- 0 . 3 4 0 \pm 0 . 0 2 7$ </td><td>0.008</td><td>0.010</td><td>0.919</td><td>0.983</td><td>0.816</td><td>0.858</td></tr><tr><td> $\log S _ { 1 } \ [ I _ { 0 } ]$ </td><td> $- 0 . 4 8 8 \pm 0 . 0 3 5$ </td><td> $- 0 . 4 4 6 \pm 0 . 0 4 9$ </td><td>0.014</td><td>0.014</td><td>0.897</td><td>0.956</td><td>0.769</td><td>0.874</td></tr><tr><td> $\log S _ { 0 } \ \big [ I _ { 0 } \big ]$ </td><td> $- 1 . 0 2 0 \pm 0 . 0 5 5$ </td><td> $- 1 . 3 7 4 \pm 0 . 1 5 4$ </td><td>0.021</td><td>0.033</td><td>0.912</td><td>0.966</td><td>0.826</td><td>0.933</td></tr><tr><td> $v _ { p } \ [ \mathrm { k m \ s ^ { - 1 } } ]$ </td><td> $- 0 . 0 1 1 \pm 0 . 6 2 6$ </td><td> $- 0 . 7 4 7 \pm 0 . 9 5 7$ </td><td>0.028</td><td>0.008</td><td>0.998</td><td>1.000</td><td>0.996</td><td>1.000</td></tr><tr><td> $v _ { 2 } \ [ \mathrm { k m \ s ^ { - 1 } } ]$ </td><td> $0 . 9 7 0 \pm 1 . 6 2 1$ </td><td> $0 . 0 1 8 \pm 2 . 6 1 2$ </td><td>0.628</td><td>0.845</td><td>0.907</td><td>0.972</td><td>0.756</td><td>0.664</td></tr><tr><td> $v _ { 1 } ~ [ \mathrm { k m ~ s ^ { - 1 } } ]$ </td><td> $0 . 3 8 6 \pm 2 . 5 4 9$ </td><td> $0 . 5 2 8 \pm 2 . 0 3 2$ </td><td>0.629</td><td>0.390</td><td>0.964</td><td>0.986</td><td>0.884</td><td>0.911</td></tr><tr><td> $\log w _ { p } \ [ \AA ]$ </td><td> $- 0 . 6 5 5 \pm 0 . 0 0 3$ </td><td> $- 1 . 2 7 6 \pm 0 . 0 0 2$ </td><td>0.003</td><td>0.001</td><td>0.961</td><td>0.894</td><td>0.550</td><td>0.233</td></tr><tr><td> $\mathrm { l o g \ } w _ { 2 } \ \mathrm { [ \AA ] }$ </td><td> $- 0 . 4 0 9 \pm 0 . 0 2 4$ </td><td> $- 0 . 5 6 6 \pm 0 . 0 8 3$ </td><td>0.014</td><td>0.034</td><td>0.855</td><td>0.938</td><td>0.644</td><td>0.750</td></tr><tr><td rowspan="8">AR</td><td> $\underline { { \mathrm { l o g } \ : w _ { 1 } \ : [ \mathrm { \AA } ] } }$ </td><td> $- 0 . 4 8 3 \pm 0 . 0 3 0$ </td><td> $- 0 . 8 9 5 \pm 0 . 0 5 3$ </td><td>0.018</td><td>0.030</td><td>0.913</td><td>0.799</td><td>0.676</td><td>0.638</td></tr><tr><td> $\log S _ { p } \ [ I _ { 0 } ]$ </td><td> $0 . 0 3 4 \pm 0 . 0 5 6$ </td><td> $- 0 . 0 2 4 \pm 0 . 0 5 9$ </td><td>0.015</td><td>0.006</td><td>0.969</td><td>0.992</td><td>0.922</td><td>0.981</td></tr><tr><td> $\log S _ { 2 } \ [ I _ { 0 } ]$ </td><td> $- 0 . 1 4 7 \pm 0 . 0 4 3$ </td><td> $- 0 . 2 8 1 \pm 0 . 0 5 7$ </td><td>0.009</td><td>0.006</td><td>0.985</td><td>0.995</td><td>0.933</td><td>0.984</td></tr><tr><td> $\log S _ { 1 } \ [ I _ { 0 } ]$ </td><td> $- 0 . 3 8 2 \pm 0 . 0 4 3$ </td><td> $- 0 . 3 1 5 \pm 0 . 0 5 7$ </td><td>0.013</td><td>0.010</td><td>0.951</td><td>0.971</td><td>0.888</td><td>0.943</td></tr><tr><td> $\log S _ { 0 } \ \big [ I _ { 0 } \big ]$ </td><td> $- 0 . 7 5 3 \pm 0 . 0 6 5$ </td><td> $- 0 . 6 3 9 \pm 0 . 1 2 9$ </td><td>0.023</td><td>0.015</td><td>0.938</td><td>0.989</td><td>0.727</td><td>0.975</td></tr><tr><td> $v _ { p } \ [ \mathrm { k m \ s ^ { - 1 } } ]$ </td><td> $0 . 0 0 0 \pm 0 . 3 6 8$ </td><td> $- 0 . 2 5 0 \pm 0 . 7 5 7$ </td><td>0.080</td><td>0.113</td><td>0.975</td><td>1.000</td><td>0.913</td><td>0.936</td></tr><tr><td> $v _ { 2 } \ [ \mathrm { k m \ s ^ { - 1 } } ]$ </td><td> $1 . 2 5 8 \pm 1 . 4 6 2$ </td><td> $0 . 0 2 5 \pm 2 . 1 1 6$ </td><td>0.317</td><td>0.723</td><td>0.977</td><td>0.922</td><td>0.932</td><td>0.659</td></tr><tr><td> $v _ { 1 } ~ [ \mathrm { k m ~ s ^ { - 1 } } ]$ </td><td> $- 0 . 8 8 9 \pm 1 . 4 5 8$ </td><td> $0 . 4 0 4 \pm 0 . 8 7 7$ </td><td>0.417</td><td>0.321</td><td>0.946</td><td>0.846</td><td>0.805</td><td>0.643</td></tr><tr><td></td><td> $\log w _ { p } \ [ \AA ]$ </td><td> $- 0 . 6 5 3 \pm 0 . 0 0 6$  一</td><td> $- 1 . 2 7 8 \pm 0 . 0 0 6$ </td><td>0.002</td><td>0.001</td><td>0.986</td><td>0.994</td><td>0.845</td><td>0.984</td></tr><tr><td></td><td> $\mathrm { l o g \it w _ { 2 } \ [ \AA ] }$ </td><td> $- 0 . 3 8 2 \pm 0 . 0 4 2$ </td><td> $- 0 . 5 8 0 \pm 0 . 1 0 4$  一</td><td>0.017</td><td>0.048</td><td>0.906</td><td>0.857</td><td>0.799</td><td>0.733</td></tr><tr><td></td><td>log w1 [Å]</td><td> $- 0 . 4 6 1 \pm 0 . 0 3 0$ </td><td> $- 0 . 8 4 7 \pm 0 . 0 8 5$ </td><td>0.014</td><td>0.046</td><td>0.851</td><td>0.809</td><td>0.705</td><td>0.651</td></tr></table>

## 4.3. Spectral Reconstruction and Temporal Evolution

The parameter-space comparison demonstrates that the network preserves the main MLSI parameter structure. We now examine whether these inferred parameters also reproduce the observed spectral profiles and their tempora evolution.

![](images/97bf17ac36675e18c429128dbd6ef5363fe2f2aa6102e38d60b07e29985cb285.jpg)  
Figure 7. Quiet-Sun spectral-profile comparison for representative internetwork and network pixels. The left panels show the observed Hα and Ca II 8542 line-center intensity maps with the selected locations marked. The right panels compare the observed profiles with the MLSI-PINN emergent profile $I _ { 0 } ,$ the intermediate profiles $I _ { 1 }$ and $I _ { 2 } ,$ and the conventional MLSI reconstruction. The residual panels show $I _ { 0 } - ($ Observed and MLSI − Observed. The wavelength ofset is defined relative to the corresponding line center of each transition.

We first examine the reconstructed spectral profiles at representative quiet-Sun and active-region locations in Figures $7$ and $^ { 8 , }$ respectively. For each spectral line, the left panel shows the observed line-center intensity map from which representative pixels were selected. The quiet-Sun pixels represent internetwork and network regions, whereas the active-region pixels represent sunspot umbral and penumbral regions. Each plotted line profile is extracted from the individual pixel marked in the corresponding map, without spatial averaging. The right panels compare the observed line profile with the final MLSI-PINN emergent profile $I _ { 0 } ,$ the intermediate layer profiles $I _ { 1 }$ and $I _ { 2 }$ , and the conventional MLSI reconstruction. The intermediate profiles are not fitted to the observations independently; they illustrate how the embedded multilayer forward model constructs the final emergent line profile from the predicted atmospheric parameters. The wavelength axis is expressed as $\lambda - \lambda _ { 0 }$ separately for each spectral line.

Figure 7 shows the quiet-Sun profile comparison for representative internetwork and network locations. For both Hα and Ca II 8542, the final MLSI-PINN line profile reproduces the main absorption profile and closely follows the conventional MLSI reconstruction, particularly around the line core. To avoid contamination from the outer continuum regions, the residual statistics were calculated only over the central wavelength interval from −200 to +200 pm. Defining the residual as the reconstructed intensity minus the observed intensity, the Hα mean residua averaged over the two selected pixels is $2 . 3 1 \times 1 0 ^ { - 3 }$ for MLSI-PINN and $- 5 . 6 6 \times 1 0 ^ { - 4 }$ for conventional MLSI. The corresponding mean absolute residuals are $1 . 2 4 \times 1 0 ^ { - 2 }$ and $1 . 0 5 \times 1 0 ^ { - 2 }$ in normalized-intensity units. For Ca II 8542, the mean residuals are $2 . 1 0 \times 1 0 ^ { - 3 }$ for MLSI-PINN and $9 . 2 8 \times 1 0 ^ { - 5 }$ for conventional MLSI, while the mean absolute residuals are $9 . 9 5 \times 1 0 ^ { - 3 }$ and $7 . 5 9 \times 1 0 ^ { - 3 }$ , respectively. The positive MLSI-PINN mean residuals indicate a small average upward intensity bias in the quiet-Sun reconstructions, although the ofset is not uniform across every wavelength or selected pixel. The slightly lower residuals of conventional MLSI are expected because its parameters are optimized independently for each observed line profile. Localized residual features appearing in both reconstructions are more likely associated with observational spectral structure and the finite flexibility of the compact MLSI forward model.

![](images/b555692668a4b2f3d04ff1bb481c3f5ab3a5aa587c0a032a612b4b42bad118ca.jpg)  
Figure 8. Active-region spectral-profile comparison for representative sunspot umbral and penumbral pixels. The left panels show the observed Hα and Ca II 8542 line-center intensity maps with the selected locations marked. The right panels compare the observed profiles with the MLSI-PINN emergent profile $I _ { 0 } ,$ the intermediate profiles $I _ { 1 }$ and $I _ { 2 } ,$ and the conventional MLSI reconstruction. The residual panels show I<sub>0</sub> − Observed and MLSI − Observed. The wavelength ofset is defined relative to the corresponding line center of each transition.

Figure 8 shows the corresponding active-region comparison for representative sunspot umbral and penumbral locations. Compared with the quiet-Sun profiles, the active-region spectra exhibit stronger local variations in their line cores and wings. All residual panels in Figures 7 and 8 use the same vertical range of −0.17 to 0.17, allowing their amplitudes to be compared directly. Using the same $- 2 0 0 \mathrm { t o + 2 0 0 p m }$ interval, the Hα mean residuals are $- 5 . 0 2 \times 1 0 ^ { - 3 }$ for MLSI-PINN and $- 5 . 6 0 \times 1 0 ^ { - 4 }$ for conventional MLSI, with mean absolute residuals of 1 $. . 2 7 \times 1 0 ^ { - 2 }$ and $8 . 6 7 \times 1 0 ^ { - 3 }$ respectively. For Ca II 8542, the corresponding mean residuals are $- 5 . 1 8 \times 1 0 ^ { - 3 }$ and $- 1 . 3 0 \times 1 0 ^ { - 3 }$ , while the mean absolute residuals are $1 . 0 8 \times 1 0 ^ { - 2 }$ and $7 . 0 3 \times 1 0 ^ { - 3 }$ . In contrast to the small positive bias found in the quiet-Sun examples, the selected active-region profiles show a modest negative MLSI-PINN intensity bias. Nevertheless, the mean absolute residuals remain of the same order for the two data sets, and MLSI-PINN continues to recover the dominant absorption morphology in both spectral lines. These results indicate that the network retains the physically interpretable MLSI representation with a modest additional reconstruction error relative to independently optimized conventional MLSI.

We also examine the temporal evolution of the reconstructed spectra and physical parameters in Figures 9-12. For each data set, the time–wavelength diagrams compare the observed line profile, the conventional MLSI reconstruction, the MLSI-PINN reconstruction, and their residuals. The corresponding parameter time series show the upper-layer line-of-sight velocities $v _ { 1 }$ inferred from Hα and Ca II 8542, together with two derived quantities, the temperature $T$ and nonthermal velocity ξ.

The temperature and nonthermal velocity are not independent MLSI-PINN output parameters. They are derived from the upper-chromospheric Doppler widths of the two lines, assuming that the Hα and $\mathrm { C a }$ II 8542 upper-layer widths sample the same plasma and that the line broadening can be decomposed into thermal and nonthermal components:

$$
\left( \frac { c w _ { j } } { \lambda _ { j } } \right) ^ { 2 } = \frac { 2 k _ { \mathrm { B } } T } { m _ { j } } + \xi ^ { 2 } , \qquad j = \mathrm { H } \alpha , \mathrm { ~ C a ~ I I } .\tag{14}
$$

Using $w _ { \mathrm { H } }$ and $w _ { \mathrm { { C a } } }$ for the upper-layer Doppler widths $w _ { 1 }$ of Hα and Ca II 8542, respectively, this gives

$$
T = 8 1 0 0 \left( \frac { w _ { \mathrm { H } } } { 0 . 0 2 5 ~ \mathrm { n m } } \right) ^ { 2 } \left[ 1 - 0 . 5 9 \left( \frac { w _ { \mathrm { C a } } } { w _ { \mathrm { H } } } \right) ^ { 2 } \right] \mathrm { ~ K } ,\tag{15}
$$

and

$$
\xi = 5 . 4 0 \left( \frac { w _ { \mathrm { C a } } } { 0 . 0 1 5 ~ \mathrm { n m } } \right) \left[ 1 - 0 . 0 4 2 \left( \frac { w _ { \mathrm { H } } } { w _ { \mathrm { C a } } } \right) ^ { 2 } \right] ^ { 1 / 2 } \mathrm { k m ~ s } ^ { - 1 } .\tag{16}
$$

Figure 9 shows the quiet-Sun time–wavelength comparison. Both Hα and Ca II 8542 show that the MLSI-PINN reconstruction follows the observed temporal evolution of the line profile and remains close to the conventional MLSI reconstruction.

Figure 10 shows the quiet-Sun temporal evolution of $v _ { 1 } , T$ , and ξ. The MLSI-PINN results closely track the MLSI reference for both line-of-sight velocities and for the derived thermal and nonthermal quantities, indicating that the network preserves the temporal behavior of the MLSI solution.

Figures 11 and 12 show the corresponding active-region results. The active-region sequence contains observational gaps, which are retained in the time axis rather than interpolated. These missing intervals appear as white horizontal bands in the time–wavelength diagrams and as breaks in the parameter time series. Within the observed intervals, MLSI-PINN follows the main time-dependent line-profile changes and reproduces the MLSI temporal trends in the upper-layer velocities and derived quantities.

To provide a more direct quantitative comparison of the reconstruction accuracy, Figure 13 summarizes the residuals shown in Figures 9 and 11. The conventional MLSI mean absolute residuals range from approximately $5 . 9 \times 1 0 ^ { - 3 }$ to $7 . 7 \times 1 0 ^ { - 3 }$ , while the MLSI-PINN values range from approximately $1 . 1 \times 1 0 ^ { - 2 } \mathrm { ~ t o ~ } 1 . 5 \times 1 0 ^ { - 2 }$ . The error bars indicate the temporal standard deviation of the wavelength-averaged absolute residual, with the largest variation occurring for active-region Hα. Although the MLSI-PINN residuals are systematically higher, both methods remain on the same order of magnitude for the two spectral lines and solar targets. As seen in Figures 9 and 11, the stronger MLSI-PINN residuals remain concentrated near similar wavelength positions throughout the sequences, particularly along the steep line-core flanks. A small diference in the reconstructed line-center position or profile width can produce a relatively large pointwise intensity residual where $| \partial I / \partial \lambda |$ is large. Consequently, the observed and reconstructed profiles can remain close in their overall morphology even when a slight spectral displacement produces coherent residual bands. Because the residual is evaluated at fixed wavelength samples, this metric penalizes small spectra shifts more strongly than a comparison based primarily on the overall profile shape. This distinction is important when interpreting the residual magnitude because it separates a small wavelength-position ofset from a substantia error in reproducing the evolution of the line profile. This sensitivity is most evident for active-region Hα, which exhibits both the largest mean MLSI-PINN residual and the greatest temporal variation among the four cases. The residual structure is therefore consistent with small systematic diferences in the inferred velocity or width parameters rather than purely random temporal reconstruction errors. Overall, MLSI-PINN maintains reconstruction accuracy comparable in scale to conventional MLSI, although the direct nonlinear optimization provides a closer pointwise fi to the observed spectra.

![](images/6272d9a811d6867c1bf3d8b51e941535ea07c8b3a0334b2f6508936a4b8a9fc2.jpg)  
Figure 9. Quiet-Sun time–wavelength spectral comparison for Hα and Ca II 8542. The columns show the observed spectra, the conventional MLSI reconstruction, the MLSI-PINN reconstruction, and the corresponding MLSI and MLSI-PINN residuals.

![](images/c97778ed9be7b911dea87bcfded7c51f42fea2e74ffec5bcddc2c2e7538db858.jpg)  
Figure 10. Quiet-Sun temporal evolution of the upper-layer velocities v<sub>1</sub> from Hα and Ca II 8542, and the derived temperature T and nonthermal velocity ξ.

![](images/954ebf0b16abb5d853ec968017fd89727d85d7facd9563fc96a5b8deac9ca0d0.jpg)  
Figure 11. Active-region time–wavelength spectral comparison for Hα and Ca II 8542. The columns show the observed spectra, the conventional MLSI reconstruction, the MLSI-PINN reconstruction, and the corresponding MLSI and MLSI-PINN residuals. White horizontal bands indicate missing observing intervals.

![](images/7db7d6e90bf2fbb86c8fc7c3736c8e4d3949ead7b8163bcb7901ded2a4a221c5.jpg)  
Figure 12. Active-region temporal evolution of $v _ { 1 } , T ,$ and $\xi .$ The curves are broken across missing observing intervals, so no interpolation is implied across the gaps.

![](images/b9e6b6ccd80a6db58b7abfaabf16b0c88b6af0df0094200e07be47eeb891d675.jpg)

![](images/8bcf77cf9a0e11cd6ab8f6f0e311c65bd1f9b574c3be6db8b1c1264b6b82dad7.jpg)

![](images/c0b09ab1bdcd10b28164bcc5c89fdee229129e092dac4ef0c1ab93d8a564f5a4.jpg)

![](images/351ebc89e59511d20df2c95e473ac4b5b5c5753d98ffdf960d08435a50ca513c.jpg)  
Figure 13. Summary of the spectral reconstruction residuals in Figures 9 and 11. The four panels show quiet-Sun Hα, quiet-Sun Ca II 8542, active-region Hα, and active-region Ca II 8542, respectively. For each time step, the mean absolute residual is calculated over the displayed wavelength interval of $| \lambda - \lambda _ { 0 } | \leq 1 2 0$ pm. The bar height gives the temporal mean, and the error bar indicates one standard deviation over the observed sequence.

## 4.4. Computational Performance

The preceding comparisons show that the MLSI-PINN model retains the main parameter-space and spectral reconstruction behavior of conventional MLSI. The practical value of the proposed framework also depends on its computational cost. We therefore record approximate wall-clock times for conventional MLSI and for the MLSI-PINN workflow. The conventional MLSI baseline is CPU-based and performs pixel-by-pixel nonlinear least-squares fitting in dependently for each raster. In contrast, MLSI-PINN requires a one-time two-stage training step on a reference raster, after which each additional raster is processed by neural-network inference followed by the analytic MLSI forward calculation. The reported MLSI-PINN inference time includes parameter prediction and spectral forward synthesis, but excludes file I/O.

The timing comparison should be interpreted as a practical wall-clock comparison rather than a hardwareindependent algorithmic benchmark. Conventional MLSI is implemented as a CPU-based pixel-by-pixel nonlinear fitting workflow, whereas MLSI-PINN uses GPU-accelerated neural-network inference after a one-time training step. A fully hardware-matched comparison is therefore not straightforward. Nevertheless, this comparison reflects the practical deployment scenario: direct MLSI fitting requires several minutes per raster, while the trained MLSI-PINN model processes each additional raster in several seconds. For observing sequences containing tens to hundreds o rasters, the one-time training cost is amortized over many frames, making the proposed framework substantially more eficient for same-day FISS time-series analysis.

Table 3. Approximate wall-clock computational cost of conventional MLSI and MLSI-PINN.
<table><tr><td>Method</td><td>Hardware</td><td>Training time</td><td>Inference time per raster</td><td>Notes</td></tr><tr><td>Conventional MLSI</td><td>CPU-based</td><td></td><td>3–5 min</td><td>Pixel-by-pixel nonlinear fitting</td></tr><tr><td>MLSI-PINN</td><td>RTX 5090 GPU</td><td>~15 min</td><td>5-15 s</td><td>Two-stage training, then forward inference</td></tr></table>

Overall, the parameter maps, density diagrams, spectral reconstructions, temporal comparisons, and quantitative metrics give a consistent picture. The MLSI-PINN model reproduces the main spatial structures and parameter scales of the conventional MLSI inversion for both quiet-Sun and active-region observations. The agreement is strongest for the source-function parameters, while the $v _ { p }$ agreement mainly reflects the consistency between the line-center proxy and the conventional MLSI photospheric-velocity estimate. The chromospheric velocity and Doppler-width parameters show larger deviations because they are more sensitive to line asymmetries, narrow dynamic ranges, and multilayer parameter degeneracy. Despite these limitations, the reconstructed spectra remain close to the observations and show residuals comparable to conventional MLSI forward reconstruction, indicating that the model preserves the main physical constraints of the MLSI formulation while avoiding repeated pixel-by-pixel nonlinear fitting.

## 4.5. Spectral Reconstruction Using the Institut f¨ur Astrophysik G¨ottingen (IAG) Solar Flux Atlas

As an additional test of spectral reconstruction beyond the FISS observations, we applied the stage-1 MLSI-PINN procedure to the Hα and Ca II 8542 profiles extracted from the IAG solar flux atlas (Reiners et al. 2016). The IAG profiles were resampled onto the corresponding FISS wavelength grids before training. As no reference MLSI parameters are available for these spectra, only the spectral reconstruction loss was used, without the stage-2 parameter-space supervision.

![](images/2a254bce6e9357ea904120b5fcf8ac21a3ac0acc2cc0e55b65dd450e0f562750.jpg)  
Figure 14. Stage-1 MLSI-PINN reconstruction of the Hα and Ca II 8542 profiles extracted from the IAG solar flux atlas.

Figure 14 shows the observed IAG profiles together with the reconstructed emergent spectrum $I _ { 0 }$ and the intermediate profiles $I _ { 1 }$ and $I _ { 2 }$ . For both lines, $I _ { 0 }$ reproduces the broad absorption profile and the line core. Within the clean wavelength samples over $\pm 4 \textup { \AA }$ , the mean residuals, defined as $\left. I _ { 0 } - I _ { \mathrm { o b s } } \right.$ , are $4 . 9 9 \times 1 0 ^ { - 3 }$ for Hα and $- 6 . 5 3 \times 1 0 ^ { - 3 }$ for Ca II 8542. The prominent localized residual peaks, particularly in Hα, occur where narrow absorption features are not represented by the compact MLSI forward model. These features occupy only a small fraction of the wavelength interval and therefore do not characterize the overall reconstruction agreement.

This test demonstrates that the MLSI-PINN framework can be adapted to independently obtained, high-resolution solar spectra while retaining the same analytic multilayer construction. However, because the IAG atlas provides diskintegrated solar flux rather than reference atmospheric parameters, this comparison evaluates spectral representability rather than the accuracy of the inferred MLSI parameters.

## 5. SUMMARY AND DISCUSSION

In this paper, we have developed a two-stage physics-informed neural-network framework for accelerating multilayer spectral inversion of Hα and Ca II 8542 spectra. The method keeps the analytic MLSI forward model inside the training loop, so the network is not trained as a purely empirical mapping from spectra to labels. Instead, the predicted parameters are required to reproduce the observed spectra through the same multilayer radiative-transfer formulation used by MLSI. Applied to quiet-Sun and active-region FISS observations, the method reproduces the main parameter structures of conventional MLSI and generates reconstructed spectra with residuals comparable to direct MLSI forward synthesis.

The two-stage design is important for reducing parameter degeneracy. In stage 1, the model is trained only through the spectral reconstruction loss, which allows it to learn a fast inverse mapping directly from the observed profiles. However, the emergent line profile does not uniquely determine every MLSI parameter. Diferent combinations of source functions, velocities, optical depths, and Doppler widths can produce similar line profiles. Stage 2 therefore introduces limited parameter-space supervision from conventional MLSI results for a selected subset of parameters while retaining the spectral reconstruction loss. This provides additional guidance for parameters that are less strongly constrained by the line profile alone, without requiring a large precomputed training set.

The parameter-dependent performance follows the expected sensitivity of the MLSI forward model. Source-function parameters are more robust because they control the line depth and the broad profile shape. Chromospheric velocities and some Doppler-width parameters are more dificult because they afect subtler profile asymmetries and can be partially degenerate with other layer parameters. The photospheric velocity $v _ { p }$ should be interpreted separately, since it is inserted from a line-center proxy rather than independently learned by the network. Its high agreement with conventional MLSI therefore reflects the consistency of the proxy estimate with the fitted MLSI photospheric velocity.

The present validation is focused on same-day or same-sequence application. This is a practical scope for FISS timeseries analysis, because the wavelength sampling, calibration, and intensity normalization are relatively consistent within a sequence. Since the two-stage training cost is modest compared with repeated pixel-by-pixel MLSI fitting over many images, retraining or fine-tuning for a new observing day remains practical. The method should therefore be viewed as an eficient sequence-level acceleration strategy rather than a universal inversion model that can be transferred without adjustment across instruments, observing conditions, or substantially diferent calibration states.

However, there are still several limitations. The parameter supervision in stage 2 inherits the assumptions and possible biases of the conventional MLSI reference inversion. In addition, MLSI itself is a simplified analytic radiativetransfer model and cannot replace full NLTE inversions when detailed atmospheric stratification, magnetic-field information, or more realistic radiative transfer is required. Future work should test the framework on a broader set of observing days and solar targets, quantify uncertainty for parameters afected by degeneracy, and examine its transferability across diferent observations. As the core inversion pipeline “mapping an observed spectrum to MLSI parameters and constraining them through the embedded analytic forward model” remains consistent across data sets, the framework is not inherently limited to the FISS sequences examined here. This consistency suggests that the method could, in principle, be extended to high-volume observations from DKIST. Investigating this possibility is a promising direction for future work, since rapid raster-scale inference could make MLSI-type analysis of large DKIST rasters and time sequences more practical. Despite these limitations, the present results show that MLSI-PINN can substantially reduce the practical cost of MLSI analysis while preserving the physical interpretability of the multilaye inversion parameters.

## ACKNOWLEDGMENTS

We gratefully acknowledge the use of data from the Goode Solar Telescope (GST) at Big Bear Solar Observatory (BBSO). BBSO operation is supported by NSF grant AGS-2309939 and the New Jersey Institute of Technology. GST operation is partly supported by the Korea Astronomy and Space Science Institute and Seoul National University. This work was supported by NSF grants AGS-2309939, 2401229, and 2408174, and NASA grants 80NSSC24M0174, 80NSSC24K0258, and 80NSSC26K1202. We also acknowledge the computational resources and support provided by Wulver, the high-performance computing cluster at the New Jersey Institute of Technology (NJIT).

## REFERENCES

Asensio Ramos, A., Cheung, M. C. M., Chifu, I., & Gafeira, R. 2023, Living Reviews in Solar Physics, 20, 4, doi: 10.1007/s41116-023-00038-x

Asensio Ramos, A., & D´ıaz Baso, C. J. 2019, Astronomy & Astrophysics, 626, A102, doi: 10.1051/0004-6361/201935628

Beckers, J. M. 1964, PhD thesis, Sacramento Peak Observatory, Air Force Cambridge Research Laboratories, Massachusetts, USA

Carlsson, M., De Pontieu, B., & Hansteen, V. H. 2019, Annual Review of Astronomy and Astrophysics, 57, 189, doi: 10.1146/annurev-astro-081817-052044

Chae, J., Cho, K., Kang, J., et al. 2021, Journal of the Korean Astronomical Society, 54, 139, doi: 10.5303/JKAS.2021.54.5.139

Chae, J., Madjarska, M. S., Kwak, H., & Cho, K. 2020, Astronomy & Astrophysics, 640, A45, doi: 10.1051/0004-6361/202038141

Chae, J., Yang, H., Park, H., et al. 2014, The Astrophysical Journal, 789, 108, doi: 10.1088/0004-637x/789/2/108

Chae, J., Park, H.-M., Ahn, K., et al. 2013, Solar Physics, 288, 1, doi: 10.1007/s11207-012-0147-x

Cheung, M. C. M., Boerner, P., Schrijver, C. J., et al. 2015, The Astrophysical Journal, 807, 143, doi: 10.1088/0004-637x/807/2/143

de la Cruz Rodr´ıguez, J., Leenaarts, J., & Asensio Ramos, A. 2016, The Astrophysical Journal Letters, 830, L30, doi: 10.3847/2041-8205/830/2/L30

de la Cruz Rodr´ıguez, J., Leenaarts, J., Danilovic, S., & Uitenbroek, H. 2019, Astronomy & Astrophysics, 623, A74, doi: 10.1051/0004-6361/201834464

D´ıaz Baso, C. J., Asensio Ramos, A., de la Cruz Rodr´ıguez, J., da Silva Santos, J. M., & Rouppe van der Voort, L. 2025, Astronomy & Astrophysics, 693, A170, doi: 10.1051/0004-6361/202452172

Felipe, T., & Esteban Pozuelo, S. 2019, Astronomy & Astrophysics, 632, A75, doi: 10.1051/0004-6361/201936679

Gnanasambandam, R., Shen, B., Chung, J., Yue, X., & Kong, Z. 2023, IEEE Transactions on Pattern Analysis and Machine Intelligence, 45, 15588

Goode, P. R., & Cao, W. 2012, in Proceedings of SPIE, Vol. 8444, Ground-based and Airborne Telescopes IV, ed. L. M. Stepp, R. Gilmozzi, & H. J. Hall (SPIE), 844403, doi: 10.1117/12.925494

Henriques, V. M. J., Mathioudakis, M., Socas-Navarro, H., & de la Cruz Rodr´ıguez, J. 2017, The Astrophysical Journal, 845, 102, doi: 10.3847/1538-4357/aa7ca4

Jarolim, R., Molnar, M. E., Tremblay, B., Centeno, R., & Rempel, M. 2025, The Astrophysical Journal Letters, 985, L7, doi: 10.3847/2041-8213/add342

Jarolim, R., Thalmann, J. K., Veronig, A. M., & Podladchikova, T. 2023, Nature Astronomy, 7, 1171, doi: 10.1038/s41550-023-02030-9

Karniadakis, G. E., Kevrekidis, I. G., Lu, L., et al. 2021, Nature Reviews Physics, 3, 422, doi: 10.1038/s42254-021-00314-5

Kianfar, S., Leenaarts, J., Danilovic, S., de la Cruz Rodr´ıguez, J., & D´ıaz Baso, C. J. 2020, Astronomy & Astrophysics, 637, A1, doi: 10.1051/0004-6361/202037572

Lee, K.-S., Chae, J., Park, E., et al. 2022, The Astrophysical Journal, 940, 147, doi: 10.3847/1538-4357/ac9c60

Leenaarts, J., Carlsson, M., Hansteen, V., & Rouppe van der Voort, L. 2009, The Astrophysical Journal, 694, L128, doi: 10.1088/0004-637x/694/2/l128

Leenaarts, J., Carlsson, M., & Rouppe van der Voort, L. 2012, The Astrophysical Journal, 749, 136, doi: 10.1088/0004-637x/749/2/136

Li, Q., Shen, B., Jiang, H., et al. 2025, arXiv preprint arXiv:2507.09430

Mili´c, I., & van Noort, M. 2018, Astronomy & Astrophysics, 617, A24, doi: 10.1051/0004-6361/201833382

Osborne, C. M. J., Armstrong, J. A., & Fletcher, L. 2019, The Astrophysical Journal, 873, 128, doi: 10.3847/1538-4357/ab07b4

Pietrow, A. G. M., Kiselman, D., de la Cruz Rodr´ıguez, J., et al. 2020, Astronomy & Astrophysics, 644, A43, doi: 10.1051/0004-6361/202038750

Raissi, M., Perdikaris, P., & Karniadakis, G. 2019, Journal of Computational Physics, 378, 686, doi: 10.1016/j.jcp.2018.10.045

Reiners, A., Mrotzek, N., Lemke, U., Hinrichs, J., & Reinsch, K. 2016, Astronomy & Astrophysics, 587, A65, doi: 10.1051/0004-6361/201527530

Ruiz Cobo, B., & del Toro Iniesta, J. C. 1992, The Astrophysical Journal, 398, 375, doi: 10.1086/171862

Sainz Dalda, A., de la Cruz Rodr´ıguez, J., De Pontieu, B., & Goˇsi´c, M. 2019, The Astrophysical Journal Letters, 875, L18, doi: 10.3847/2041-8213/ab15d9

Skumanich, A., & Lites, B. W. 1987, The Astrophysical Journal, 322, 473, doi: 10.1086/165743

Socas-Navarro, H., de la Cruz Rodr´ıguez, J., Asensio Ramos, A., Trujillo Bueno, J., & Ruiz Cobo, B. 2015, Astronomy & Astrophysics, 577, A7, doi: 10.1051/0004-6361/201424860

Tziotziou, K. 2007, in Astronomical Society of the Pacific Conference Series, Vol. 368, The Physics of Chromospheric Plasmas, ed. P. Heinzel, I. Dorotoviˇc, & R. J. Rutten, 217, doi: 10.48550/arXiv.0704.1558

Unno, W. 1956, PASJ, 8, 108, doi: 10.1093/pasj/8.3-4.108

Vissers, G. J. M., de la Cruz Rodr´ıguez, J., Libbrecht, T., et al. 2019, Astronomy & Astrophysics, 627, A101, doi: 10.1051/0004-6361/201833560

Yadav, R., Kazachenko, M. D., Cauzzi, G., et al. 2025, The Astrophysical Journal, 989, 183, doi: 10.3847/1538-4357/adf4c1

Yang, K. E., Tarr, L. A., Rempel, M., et al. 2024, The Astrophysical Journal, 976, 204, doi: 10.3847/1538-4357/ad865b

Yang, K. E., Sun, X., Tarr, L. A., et al. 2025, The Astrophysical Journal, 995, 146, doi: 10.3847/1538-4357/ae12ef