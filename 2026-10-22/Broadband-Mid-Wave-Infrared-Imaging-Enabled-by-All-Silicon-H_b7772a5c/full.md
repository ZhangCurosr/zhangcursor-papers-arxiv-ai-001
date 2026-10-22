# Broadband Mid-Wave Infrared Imaging Enabled by All-Silicon Hyperboloid Metalens

Rose Johnson<sup>1,#</sup> , Debabrata Mandal<sup>2,#</sup>, Zihao Zou<sup>2</sup>, Jessie Wei<sup>3</sup>, Anna Wirth-Singh<sup>3</sup>, Vladimir

Yarmolik<sup>1</sup>, Johannes Froech<sup>2</sup>, Arka Majumdar<sup>1,3,\*</sup> Praneeth Chakravarthula<sup>2,\*</sup>

<sup>1</sup> Department of Electrical and Computer Engineering, University of Washington, Seattle, WA 98195, USA.

<sup>2</sup> Department of Computer Science, University of North Carolina at Chapel Hill, Chapel Hill, NC 27517, USA.

<sup>3</sup> Department of Physics, University of Washington, Seattle, WA 98195, USA.

Corresponding authors. Email: arka@uw.edu, cpk@cs.unc.edu

\# These authors contributed equally

## Abstract

The mid-wave infrared (MWIR, 3 – 5 μm) is indispensable for thermal sensing, gas detection, long-range imaging, and free-space optical communications.<sup>1,2</sup> Meta-optics offer a compact, lightweight, and potentially low-cost alternative to conventional refractive optics, making them attractive for size, weight, and power (SWaP)-constrained systems. However, their strong chromatic dispersion has largely restricted imaging to narrow spectral bands.<sup>3</sup> Here, we demonstrate broadband imaging across the full 3–5 μm MWIR band using a single-layer silicon metasurface coupled with computational reconstruction. Remarkably, the reconstruction model is adapted using only four image captures, overcoming large training data requirements and enabling broadband image recovery despite the strongly wavelength-dependent optical response. In contrast to approaches that engineer complex, dispersion-optimized metalens profiles for broadband operation, our method leverages the wavelength-dependent information encoded by a simple 2- cm-diameter hyperboloid metalens. We show that this metasurface captures sufficient broadband intensity information for computational reconstruction across the entire MWIR band. These results establish a simple route toward broadband computational meta-optics without requiring broadband achromatic metalens designs, offering a promising platform for compact MWIR imaging systems.

## Introduction

Imaging in the mid-wave infrared (MWIR, wavelength band 3 – 5 μm) is central to a broad range of scientific, industrial, and defense applications, ranging from long-range imaging and remote sensing to free-space optical communication and adaptive optical wavefront sensing through atmospheric turbulence.<sup>3–7</sup> Conventional MWIR imaging systems typically rely on refractive lenses made from a limited set of infrared-transparent materials, such as germanium and chalcogenide glasses. These optics provide high image quality, but they are often bulky and heavy. This becomes a major limitation for platforms such as aircrafts, drones, and other systems where size, weight, and power (SWaP) are tightly constrained.

Flat meta-optics offer a fundamentally different approach. Instead of shaping light with a thick, curved lens, a metasurface uses a thin layer of subwavelength structures to control the optical wavefront.<sup>8,9</sup> This can dramatically reduce the volume and mass of an imaging system. MWIR metasurfaces have been demonstrated using several material platforms, including chalcogenide glasses such as ZnSe and ZnS, Si on sapphire or CaF<sub>2</sub>, single-layer Si, and double-layer ZnS/Si structures.<sup>2,3,10–12</sup> Silicon is particularly attractive because all-silicon metasurface can be readily fabricated using high volume semiconductor manufacturing techniques.

However, the major obstacle to broadband meta-optic imaging is chromatic aberration. A conventional hyperboloid metalens is designed to focus one wavelength, and its focal response changes rapidly away from that design wavelength because of the strong wavelength dependence of its phase profile.<sup>13,14</sup> As a result, an image that is sharp at one wavelength can become severely blurred at another. This problem becomes especially challenging when a single metasurface must operate across a broad spectral band such as the full 3–5 μm MWIR window.

Two main approaches have emerged to address this challenge. The first is dispersion engineering, in which the individual meta-atoms are designed to produce the desired phase response across multiple wavelengths. In the visible, this approach has enabled broadband achromatic metalenses with diffraction-limited focusing, and related designs have been demonstrated at narrower bandwidths in the infrared.<sup>15–21</sup> However, the achievable bandwidth, numerical aperture (NA), and aperture size are fundamentally coupled.<sup>13,22,23</sup> As a result, many demonstrated achromatic metalenses operate with relatively small apertures or low NA, limited to only a few phase wraps. This limitation becomes increasingly important for practical imaging systems, where a moderateto-large aperture can require thousands of phase wraps across the lens. In the MWIR, broadband metalenses have been experimentally demonstrated, but typically over only part of the full 3–5 μm band or at relatively small aperture and NA.<sup>3,6,17</sup>

The second approach is computational imaging, where rather than requiring the optics alone to form a sharp image at every wavelength, the optical design and reconstruction algorithm work together. For example, inverse-designed meta-optics jointly optimized with a computational backend using an image-quality metric such as the modulation transfer function (MTF) have been demonstrated over the desired spectral range.<sup>24–26</sup> Such designs generally trade the best possible performance at any single wavelength for a more uniform response across the full bandwidth. A computational backend then removes the remaining errors. This strategy has produced compelling broadband imaging results in the visible and has also been extended to the long-wave infrared (LWIR, 8 – 12 μm), where inverse designed all-Silicon meta-optics enabled broadband thermal

imaging.<sup>24,27</sup>

Interestingly, complex dispersion engineering is not always necessary for computational imaging. Broadband computational imaging has also been demonstrated in the visible using a conventional hyperboloid metalens, even though its point spread function (PSF) becomes highly distorted away from the design wavelength.<sup>28</sup> In that case, information shared among the red, green, and blue (RGB) channels was used by a learned reconstruction model to correct chromatic blur and vignetting. This raises an important question: can a similarly simple metalens enable broadband computational imaging in the MWIR, where no RGB color channels are available?

The answer is not obvious because visible and MWIR cameras sample spectral information differently. Visible cameras typically use RGB color filters, which divide the visible spectrum into narrower spectral channels.<sup>29,30</sup> Apart from preserving color information, these filters also reduce the spectral bandwidth that contributes to each measurement, effectively making the operational wavelength range narrower. In contrast, a broadband MWIR detector can collect photons across the entire 3–5 μm band in a single channel. The optics must therefore tolerate a much larger fractional bandwidth, making broadband optical optimization substantially more difficult. At the same time, MWIR intensity imaging does not necessarily require the system to preserve color information. Instead, the central requirement is to collect enough useful spatial information across the band for the scene to be reconstructed.

This distinction leads to a surprisingly simple design principle. We show that, across the large fractional bandwidth of the MWIR, an inverse-designed metalens can provide less useful imaging performance than a conventional hyperboloid metalens. Although the hyperboloid produces strong chromatic blur, it retains sufficient broadband spatial contrast and photon throughput for computational reconstruction. However, this poses a computational challenge wherein the raw measurements are blurred and must therefore be restored without relying on the thousands of paired training images commonly used to train neural networks.<sup>27,27,31,32</sup>

Here, we demonstrate broadband computational imaging across the full MWIR band using a 2 cm diameter, f/1 all-Silicon hyperboloid metalens designed for a nominal wavelength of 4.5 μm. We first compare the simulated broadband performance of the hyperboloid metalens with that of an inverse-designed metasurface at matched numerical aperture and across different spectral bandwidths. We find that as the bandwidth increases, the simple hyperboloid becomes the better optical front end for computational reconstruction. We then introduce an image restoration approach that conditions a pretrained restoration network on the experimentally measured MWIR PSF and adapts it to our imaging system using only four captured images (Fig. 1a). Our metalens itself uses simple meta-atoms and requires no dispersion engineering (Figs. 1b–d). Together, these results show that broadband MWIR imaging does not require a metalens that forms a sharp image at every wavelength. Instead, a simple chromatic metalens can capture sufficient spatial information for high-fidelity computational reconstruction across the entire MWIR band.

## Results

## Broadband information transfer favors a hyperboloid metalens

Unlike visible cameras, which typically use color filters to divide the spectrum into narrower RGB channels, MWIR cameras can collect light across the full 3–5 μm band in a single channel.

Therefore, we seek to investigate under what conditions a dispersion-engineered meta-optic preserve more spatial information than a simple hyperboloid metalens.

To this end, we calculated the integrated volume under the modulation transfer function (MTF) across varying fractional bandwidths $( \Delta \lambda / \lambda _ { 0 } )$ and report the results in Fig. 2. Because the MTF volume serves as a direct proxy for the spatial frequency information transferred to the sensor and available for computational restoration, this metric establishes the boundary conditions where engineering the chromatic dispersion yields a net advantage.

At an intermediate fractional bandwidth of ∼ 0.10 - 0.20 , the inverse-designed meta-optic achieves a larger integrated MTF volume than the hyperboloid metalens. This regime is relevant to previous computational imaging demonstrations in the visible, where RGB Bayer filters divide the spectrum into channels with fractional bandwidths of ∼ 0.18.<sup>25,27</sup> Within this bandwidth range, inverse design can balance the transfer of spatial frequencies across each spectral channel and improve the information available for reconstruction.

This behavior changes as the spectral bandwidth increases. Full-band MWIR imaging without spectral filters corresponds to a much larger fractional bandwidth of 0.50. Over this broad spectral range, the inverse-designed meta-optic must balance its response across a larger range of wavelengths. As a result, its integrated MTF volume decreases below that of the hyperboloid metalens. At this operating point (fractional bandwidth = 0.50), the hyperboloid retains a 9% higher integrated MTF volume compared to the inverse-designed counterpart. Consequently, for full MWIR imaging, a standard hyperboloid metalens provides superior photon throughput and contrast transfer, making it the preferred optical front-end when paired with a computational backend.

## MWIR metalens design

Guided by this analysis, we designed and fabricated a 2-cm-aperture hyperboloid metalens with a nominal focal length of 2 cm (f/# of 1) and a design wavelength of 4.5 μm. The metalens has a 10- fold reduction in lens thickness compared with a commercial Ge refractive lens. Importantly, the metalens is not designed to eliminate chromatic aberration across the 3–5 μm band. Following the analysis in Fig. 2, although the hyperboloid’s broadband output is blurred, it preserves sufficient spatial information across the MWIR band to make computational restoration possible. Additional details, including the phase response of the meta-atoms is provided in the Supplementary Note 1.

(a)  
![](images/a8dce91c1509dabeaf3d2ff8ad19479661599131bb3f76fc7ab929137f2d0bf2.jpg)

![](images/d5db9f2a0f762a8dc1a1ecbc50a0f1772ce21c87bfcc0af726e529eb03bc4418.jpg)

![](images/9f9aadcb29914623d8035d93e700c51f447020e3877da7b6fa564f1df4e51e68.jpg)

(c)  
![](images/681600b4eaa2359ef53f30e88e5db06d76273a746c2a648ff398327aa231d49a.jpg)

![](images/c8ec02774cfeba7ee15a8c7a9c3ac0d66b9bbe6390674fded5d69a90cc97d40b.jpg)  
Fig. 1. Overview of the proposed broadband MWIR imaging system with an all-Silicon hyperboloid metalens. (a) Schematic description of the broadband imaging pipeline. A hot-plate heated between $1 0 0 - 2 0 0 ^ { \circ } \mathrm { C }$ produces the MWIR signal and the images captured by a fabricated hyperboloid metalens is recorded by a monochromatic MWIR camera. The chromatic defocus present in the imaging measurements is restored using the computational restoration framework described in Supplementary Note 3. (b) Comparison of the bulk reduction from the Silicon-based metalens over a commercial off the shelf (COTS) Ge refractive lens. Side-perspective image of the ultra-thin form factor with all-Silicon fabricated metalens. (c) SEM of the fabricated all-Si meta-atoms. (d) Schematic of a meta-atom pillar. The height is 4 μm and the periodicity is 2 μm.

![](images/373e6b6f7949a44e69f75028585d1677f0af1f655c4a76986486ecfb6c026a20.jpg)  
Fig. 2. Comparison of simulated MTF performance of forward and inverse meta-optic designs. Simulated integrated modulation transfer function (MTF) area as a function of fractional bandwidth for the matched hyperboloid (forward design) and MTF-engineered (inverse design) meta-optics. The integrated MTF area provides a comparison metric of broadband imaging performance across increasing fractional bandwidth. Blue annotations indicate the relative difference of the integrated MTF area between the forward- and inverse-designed meta-optics at matched optimization bandwidths. Across the full MWIR range $( 3 - 5 \mu \mathrm { m } .$ , fractional bandwidth = 0.5), the hyperboloid metalens exhibits less than a $1 0 ^ { 3 }$ relative difference in MTF performance to the inverse design optic, suggesting that a forward design optic can achieve imaging performance comparable to an inverse design counterpart under identical aperture and focal length constraints.

## Few-shot computational restoration

The simulation results in Fig. 2 establish that the hyperboloid metalens preserves sufficient broadband spatial contrast at the full MWIR range that could enable successful computational image restoration. However, this information is not directly accessible from the raw image because of strong chromatic blur. We therefore developed a computational restoration pipeline to determine whether the information retained by the metalens can be recovered experimentally (Fig. 3).

Our approach adapts an all-in-one image restoration model<sup>33</sup> to the MWIR imaging system. The model is pretrained on 16 common image-degradation tasks, including deblurring, dehazing, and low-light enhancement (Fig. 3f). This pretrained restoration prior is particularly useful for MWIR imaging, where fabrication imperfections, incoherent illumination, and sensor readout errors can cause experimental measurements to differ from numerical simulations. Rather than learning image restoration entirely from MWIR data, the model begins with previously learned representations of common image degradations and is then adapted to our experimental system.

Our image recovery pipeline consists of two stages: Stage 1 uses a few-shot data collection procedure to circumvent the lack of high-quality paired MWIR training data. Stage 2 finetunes the all-in-one restoration model and applies a refinement step to better match the MWIR target images. This approach allows the pretrained network to be adapted to our MWIR system using only four captured images.

We evaluate the adapted network on a separate, previously unseen, set of heated stencil targets (Fig. 4). Additional measurements using a 1951 USAF resolution test chart and exposurebracketed captures are provided in the Supplementary Information. Across the broadband targets, the reconstructed images show substantially improved contrast and clearly recovered object structure. These results support the prediction from the optical analysis that the chromatically blurred measurements produced by the hyperboloid metalens still contain spatial information that can be recovered computationally, making it a competitive and practical approach to compact broadband MWIR imaging.

![](images/9b1bfe7e5726769c8b24c2a2b407a221feca847fdf7367c37e89ec8dfca2b611.jpg)  
1. MWIR Training Data Generation

![](images/cb50519578b8867f98104e1cfa082900da75bc9bf491493822fd43ce93a44be1.jpg)

![](images/7f66d514fb681e31942f35448822c766038b7acd820db8241e518bbb3d0f5f57.jpg)

2. MWIR Few-shot training adaptation  
![](images/f2fd6ab6ae5c18b88b760d23f8986460bfe7037d9e8d31f230dd0bd4ceb60792.jpg)

![](images/212b9a18c85174278b865d4ceb0d9936868d308c227b2090d9d5e833310d1301.jpg)  
Fig. 3. Restoration Model Pipeline. (a) Overview of the proposed restoration framework involving a generative all-in-one restoration model followed by color refinement and artifact removal. (b-e) Data preparation of the all-in-one restoration for the MWIR task with 4 image inputs. The restoration model is paired with a robust data augmentation pipeline. (f-i) Fine tuning of the restoration in a parameter efficient manner. Enhanced images are passed through a color thresholding stage to correct for generative artifacts and correct for luminance.

![](images/d5510004bf59479c1f68b4ade7db338f9ae3fa1b4210272a9be42e22c965fa01.jpg)  
Fig. 4. Imaging Results and Characterization. (a) Broadband MWIR images with all-Silicon hyperboloid metalens using a monochromatic detector (FLIR A6751). (b-c) Restored images following the successive restoration and color refinement stages described in Methods. (d) Comparison against Unet-based baseline all-in-one restoration pipeline<sup>34</sup> which is unable to recover large scene regions. (e) Ground truth target images for comparison.

## Experimental broadband MWIR imaging

Figure 4a shows raw broadband images captured through the hyperboloid metalens, designed to focus at 4.5 μm wavelength, using a commercial FLIR camera. As expected, integrating light across the MWIR band produces strong chromatic blur. This blur reduces image contrast and suppresses high-spatial-frequency detail, making object boundaries difficult to distinguish directly in the raw measurements.

Figures 4b–d show the corresponding images after computational restoration. Despite the severe blur in the raw image captures, the restoration model successfully recovers broadband contrast from target objects with various identifiable structures and boundaries. The corresponding ground truth target images are shown in Figure 4e for comparison. The restored images show close agreement with the target structures, including the recovery of higher-spatial-frequency features associated with object edges and boundaries.

## Discussion

In this work, we systematically compare a single-wavelength hyperboloid metalens with an inverse-designed meta-optic across different fractional bandwidths at matched aperture and numerical aperture. Our findings demonstrate that broadband computational imaging does not necessarily require complex dispersion engineering or inverse-designed phase profiles. Across the full 3--5 0m MWIR band, a standard forward-designed hyperboloid metalens preserves sufficient spatial frequency information and contrast to enable high-fidelity computational image restoration.

This behavior is quantitatively governed by the relationship between fractional bandwidth and modulation transfer function (MTF) volume. Our analysis shows that the inverse-designed metaoptics achieves a larger integrated MTF volume within an intermediate fractional bandwidth regime (∼ 0.10--0.20), matching the conditions in visible imaging where discrete color filters divide the spectrum into narrower sub-bands, i.e., RBG channels. Outside this regime, however, the hyperboloid retains higher overall MTF volume—exhibiting a 9% advantage across the full bandwidth $( \Delta \lambda = 2 . 0 \mu \mathrm { m } )$ and a 3% advantage in the narrowband limit $( \Delta \lambda = 0 . 3 \mu \mathrm { m } )$ . For filterless monochromatic MWIR detection, where a single channel collects light across a fractional bandwidth of 0.50, balancing the optical response over the entire spectral range no longer provides an advantage according to this metric. Instead, the simple hyperboloid preserves more of the spatial information available for computational reconstruction, making it a higher-performing optical front-end.

To recover this information experimentally, we adapt a pretrained image-restoration network to the measured response of our MWIR imaging system using only four captured images. This fewshot calibration avoids the large, paired datasets typically required to train image restoration networks from scratch. Despite the strong chromatic blur in the raw measurements, the reconstructed images show close agreement with the ground-truth targets. These experiments demonstrate that the information preserved by the hyperboloid metalens can be effectively recovered with limited system-specific training data.

More broadly, these results suggest that the optical front end need not eliminate chromatic aberrations if it preserves sufficient information for computational recovery across the broadband. This shifts the design objective away from producing an achromatic PSF and toward maximizing useful information transfer between the scene and sensor, thereby avoiding the fabrication complexity of multi-layer or multi-level metasurfaces. In our MWIR system, this principle enables broadband imaging using a simple, single-layer hyperboloid metalens without dispersion

engineering, paired with a calibration-efficient computational backend. Such an approach provides a promising path toward compact, lightweight broadband MWIR imaging systems for SWaPconstrained sensing and imaging applications.

## Methods

## Metalens fabrication

A $3 0 0 \mu \mathrm { m }$ thick double-sided polished Silicon wafer was coated with a layer of HMDS to promote resist adhesion. A layer of positive resist (AZ 1505) was deposited (4k rpm, \~500 nm thickness), followed by a 60s soft bake at $1 0 0 ^ { \circ } \mathrm { C }$ on a hot plate. The resist layer was then patterned with directwrite lithography (Heidelberg DWL66+), followed by a post-exposure bake for 60 s at $1 0 0 ^ { \circ } \mathrm { C }$ on a hot plate. The resist was developed in a commercial developer solution (AZ 726) for 60 s. Subsequently, the resist pattern was transferred into the bulk Silicon with deep reactive-ion etching (SPTS DRIE) to a target depth of 3μm to form the Silicon nanopillars. The residual photoresist was stripped using oxygen plasma.

## Measurements:

PSF Measurement: The refractive relay optics used to extend the imaging distance to the camera sensor plane were purchased from Edmund Optics. Specifically, a 50 mm diameter, 50 mm focal length Ge asphere (Edmund Optics #87-996) and 75 mm diameter, 150 mm focal length uncoated $\mathrm { C a F } _ { 2 }$ plano-convex lens (Edmund Optics, #24-821). To measure the PSF of the meta-optic, a tunable MWIR QCL is guided through a single mode fiber (Thorlabs IRFS32) to produce a pointsource illumination ${ \sim } 2 5$ cm away from the meta-optic. The PSF was then re-imaged with the refractive relay-optics and captured onto the camera sensor (FLIR MWIR A6751 InSb camera, 640 X 512 pixels, 15 μm pixel resolution).

Images: Broadband images were captured under blackbody radiation produced from a hot plate (Torrey Pines Scientific HP50) heated to $1 3 0 ^ { \circ } \mathrm { C } .$ . Laser-cut aluminum targets finished with matte black paint to prevent reflections were placed in front of the hot plate to pattern MWIR light and form the imaging object. Image captures were taken with the same refractive relay assembly and minimal post-processing routines, such as background subtraction, were performed before the reconstruction.

## MWIR Meta-optic characterization

We fabricate a 20 mm aperture hyperboloid metalens with a nominal focal length of 20 mm with a central focusing wavelength of 4.5 μm. We compare this result against a known indicator for broadband imaging performance to an inverse designed meta-optic with matched dimensions of numerical aperture and focal length. The simulation results demonstrate that inverse design optics with intermediate optimization bandwidths outperform a hyperboloid, yet the forward design optics retain an advantage at the extremes of the bandwidth, specifically at 9% at the full $\Delta \lambda = 2 . 0$ μm band and 3% at the narrowest $\Delta \lambda = 0 . 3$ μm band. These are bandwidth ranges of interest for unfiltered thermal imaging and for narrowband gas sensing, respectively. Raw image measurements are captured with a commercial FLIR sensor (A6751 InSb) and a restoration algorithm enhances the monochromatic images; further details are in the Supplementary Material. A set of image captures from matte-black aluminum stencils are used to pattern contrast and the target intensity is recorded on the camera sensor without additional filters to emulate ambient, realworld conditions of randomly emitting hot objects.

## Computational Reconstruction

(Stage 1) Data Collection: Due to practical limitations in collecting outdoors MWIR data, we build a robust data augmentation pipeline that can mimic outdoor conditions such as varying temperatures of scene objects, varying exposures, and varying object resolution scales based on distance from the sensor. Collecting paired data where the clean target is perfectly aligned to the corrupted MWIR input is also challenging due to the parallax between the regular and MWIR camera. We manually align lab-captured image pairs (Fig. 3.d) and use the best aligned 4-5 samples for the data augmentation pipeline (Fig. 3.e). The augmentation pipeline uses a library of augmentations ranging from simple geometric transformations such as rotation to complex image manipulations such as cutmix<sup>35</sup>. The resulting dataset is used for the all-in-one image restoration model finetuning phase described next.

(Stage 2) Diffusion Restoration Model: We build our restoration model upon a generative all-inone image restoration framework coupling a frozen latent-diffusion backbone with task-specific mixture-of-experts control adapters<sup>33</sup>. Because the backbone and main control pathway remain frozen, the model can be specialized to a new degradation by training only a small set of taskisolated parameters, making it well suited to the data-scarce regime of MWIR metalens imaging. We first extract low-level cues such as edges, textures, haze maps, etc. from the input image as part of the base model’s restoration pipeline. The restoration cue is used as conditioning additional to the source MWIR image. We introduce an additional lightweight branch (Fig. 3.g) for the MWIR restoration task which is finetuned on the augmented image dataset from Stage 1 with a small learning rate until performance saturates on a held-out frame, which restores the characteristic MWIR metalens aberrations (spatially varying blur and chromatic spread) while leaving the model's pre-trained restoration capabilities intact. This parameter-efficient scheme lets the generative prior of the base model carry most of the reconstruction burden, so that a very few samples (<5) suffice to recover a usable inverse for the metalens degradation.

Color Refinement Process: Our all-in-one diffusion restoration model was adapted to mid-wave infrared (MWIR) segmentation in a few-shot regime, fine-tuned on only four labelled samples. While the model generalized to unseen test samples at inference, outputs exhibit characteristic diffusion artifacts, spurious texture, halo regions around object boundaries, and intensity distributions that diverged from the target thermal scale (Fig. 3.h). A post-processing refinement pipeline (Fig. 3.i) was therefore developed to suppress these artifacts while preserving fine structural detail.

The color refinement pipeline applies three stages sequentially:

1. Background suppression: Pixels where the thermal input image pixels fall below a sensor floor (1) were labelled as noise and zeroed, isolating true foreground thermal objects from sensor background.

2. Sigmoid contrast remapping: A sigmoid transform was centered on the per-image Otsu threshold<sup>36</sup> computed within the thermal foreground mask, pushing sub-threshold $( \beta _ { 1 } )$ pixels toward black and super-threshold $( \beta _ { 2 } )$ pixels toward white while preserving fine structural gradients.

3. Edge-adaptive floor clipping: Sobel edge detection<sup>37</sup> was then used to identify highgradient regions in the sigmoid output. The $\gamma ^ { \prime } t h$ percentile of the strongest-edge pixels was used as a per-image floor, zeroing all values below it. This reduced spurious dark pixels with no loss of bright structure.

The hyper-parameters $\alpha , \beta , \gamma$ were tuned until the resulting distribution of restored images closely matched the target MWIR distribution.

## Data availability

All relevant data that support the findings are available within this Article and Supplementary Information. Source data are available from the corresponding authors upon request.

## Code availability

The diffusion model is available on:

## Acknowledgements

This work was partially supported by NSF award 2107454, and AFOSR. Device fabrication and characterization work was conducted in part at the Washington Nanofabrication Facility / Molecular Analysis Facility, a National Nanotechnology Coordinated Infrastructure (NNCI) site at the University of Washington.

Distribution A: Approved for public release - distribution unlimited. (AFRL/RASE-2026-1791)

## Author contributions

P.C. and A.M. conceived the idea. R.J. and Z.Z. designed the metasurfaces. D.M. and P.C. designed the simulation model and image restoration. R.J., A.W.S., and V.V. fabricated the devices. R.J. performed the experiments with the help from J.W. R.J., J.F., A.M., and P.C. interpreted the results together with all the authors. R.J. and D.M. wrote the manuscript with input from all the authors. A.M. and P.C. supervised the overall project.

## Competing interests

The authors declare no competing interests.

## References

1. Zhou, H., Chen, L., Shen, F., Guo, K. & Guo, Z. Broadband Achromatic Metalens in the Midinfrared Range. Phys. Rev. Appl. 11, 024066 (2019).

2. Li, X., Chen, S., Wang, D., Shi, X. & Fan, Z. Transmissive mid-infrared achromatic bifocal metalens with polarization sensitivity. Opt. Express 29, 17173–17182 (2021).

3. Yuan, Y., Yan, Z., Zhang, P., Chang, Z., Peng, F., Chen, R., Yang, Z., Chen, S., Zhao, Q. & Huang, X. A Broadband Achromatic Dielectric Planar Metalens in Mid-IR Range. Photonic Sens. 13, 230126 (2023).

4. Sandler, D. G., Hyman, H., Brinkley, T. J. & Leslie, D. H. MWIR adaptive optics system for imaging missile defense tests using the 3.7-m AEOS telescope. in 283–291 (San Diego, CA, 2000). doi:10.1117/12.405787.

5. Swami, S. S., Mohd, B., Schmitt, M. & Mundt, C. On-board detection of fast-moving targets using an MWIR sensor on a small satellite. in (SPIE, Edinburgh, United Kingdom, 2024). doi:10.1117/12.3031665.

6. Xiong, W., Sha, C. & Ding, J. Polarization-independent broadband achromatic metalens in the mid-infrared (3– 5 μm) region. Appl. Phys. Express 15, 022001 (2022).

7. Wang, J., Wirth-Singh, A., Saragadam, V., Johnson, R., Majumdar, A. & Veeraraghavan, A. A golay metalens for long-range, large aperture, thermal imaging via sparse aperture computational imaging. Nat. Commun. 16, 10281 (2025).

8. Khorasaninejad, M., Chen, W. T., Devlin, R. C., Oh, J., Zhu, A. Y. & Capasso, F. Metalenses at visible wavelengths: Diffraction-limited focusing and subwavelength resolution imaging. Science 352, 1190–1194 (2016).

9. Lin, D., Fan, P., Hasman, E. & Brongersma, M. L. Dielectric gradient metasurface optical elements. Science 345, 298–302 (2014).

10. Wirth-Singh, A., Martin Jimenez, A., Choi, M., Fröch, J. E., Johnson, R., Le Teichmann, T., Coppens, Z. & Majumdar, A. Meta-optics triplet for zoom imaging at mid-wave infrared. Appl. Phys. Lett. 125, 211705 (2024).

11. Xu, F., Chen, W., Li, M., Liu, P. & Chen, Y. Broadband achromatic and wide field-of-view single-layer metalenses in the mid-infrared. Opt. Express 31, 36439–36450 (2023).

12. Yue, S., Liu, Y., Wang, R., Hou, Y., Shi, H., Feng, Y., Wen, Z. & Zhang, Z. All-silicon polarizationindependent broadband achromatic metalens designed for the mid-wave and long-wave infrared. Opt. Express 31, 44340–44352 (2023).

13. Presutti, F. & Monticone, F. Focusing on bandwidth: achromatic metalens limits. Optica 7, 624 (2020).

14. Engelberg, J. & Levy, U. The advantages of metalenses over diffractive lenses. Nat. Commun. 11, 1991 (2020).

15. Chen, W. T., Zhu, A. Y., Sanjeev, V., Khorasaninejad, M., Shi, Z., Lee, E. & Capasso, F. A broadband achromatic metalens for focusing and imaging in the visible. Nat. Nanotechnol. 13, 220–226 (2018).

16. Shrestha, S., Overvig, A. C., Lu, M., Stein, A. & Yu, N. Broadband achromatic dielectric metalenses. Light Sci. Appl. 7, 85 (2018).

17. Ou, K., Yu, F., Li, G., Wang, W., Miroshnichenko, A. E., Huang, L., Wang, P., Li, T., Li, Z., Chen, X. & Lu, W. Mid-infrared polarization-controlled broadband achromatic metadevice. Sci. Adv. 6, eabc0711 (2020).

18. Shan, D., Xu, N., Gao, J., Song, N., Liu, H., Tang, Y., Feng, X., Wang, Y., Zhao, Y., Chen, X. & Sun, Q. Design of the all-silicon long-wavelength infrared achromatic metalens based on deep silicon etching. Opt. Express 30, 13616–13629 (2022).

19. Si, W., Hu, Z., Lan, D., Zhou, Y. & Li, W. Robust Achromatic All-Dielectric Metalens for Infrared Detection in Intelligent Inspection. Sensors 22, 6590 (2022).

20. Nalbant, H. C., Balli, F., Yelboğa, T., Eren, A. & Sözak, A. Transmission optimized LWIR metalens. Appl. Opt. 61, 9946–9950 (2022).

21. Meem, M., Majumder, A., Banerji, S., Garcia, J. C., Kigner, O. B., Hon, P. W. C., Sensale-Rodriguez, B. & Menon, R. Imaging from the visible to the longwave infrared wavelengths via an inverse-designed flat lens. Opt. Express 29, 20715–20723 (2021).

22. Huang, L., Colburn, S., Zhan, A. & Majumdar, A. Full-Color Metaoptical Imaging in Visible Light. Adv. Photonics Res. 3, 2100265 (2022).

23. Miller, O. D. & Monticone, F. Fundamental limits in photonics and electromagnetics: a tutorial. https://doi.org/10.48550/arXiv.2605.24738 (2026) doi:10.48550/arXiv.2605.24738.

24. Huang, L., Han, Z., Wirth-Singh, A., Saragadam, V., Mukherjee, S., Fröch, J. E., Tanguy, Q. A. A., Rollag, J., Gibson, R., Hendrickson, J. R., Hon, P. W. C., Kigner, O., Coppens, Z., Böhringer, K. F., Veeraraghavan, A. & Majumdar, A. Broadband thermal imaging using meta-optics. Nat. Commun. 15, 1662 (2024).

25. Tseng, E., Colburn, S., Whitehead, J., Huang, L., Baek, S.-H., Majumdar, A. & Heide, F. Neural nano-optics for high-quality thin lens imaging. Nat. Commun. 12, 6493 (2021).

26. Colburn, S., Zhan, A. & Majumdar, A. Full-color imaging with PSF-engineered metasurfaces and computational reconstruction. in FM4A.4 (Optica Publishing Group, 2017). doi:10.1364/FIO.2017.FM4A.4.

27. Fröch, J. E., Chakravarthula, P., Sun, J., Tseng, E., Colburn, S., Zhan, A., Miller, F., Wirth-Singh, A., Tanguy, Q. A. A., Han, Z., Böhringer, K. F., Heide, F. & Majumdar, A. Beating spectral bandwidth limits for large aperture broadband nano-optics. Nat. Commun. 16, 3025 (2025).

28. Engelberg, J. & Levy, U. Achromatic flat lens performance limits. Optica 8, 834–845 (2021).

29. Palum, R. Image Sampling with the Bayer Color Filter Array. in 239–245 (2001).

30. Green, P. Frontmatter. in Fundamentals and Applications of Colour Engineering (Wiley, 2023). doi:10.1002/9781119827214.fmatter.

31. Potlapalli, V., Zamir, S. W., Khan, S. & Khan, F. S. PromptIR: Prompting for All-in-One Blind Image Restoration. in (2023).

32. Li, B., Liu, X., Hu, P., Wu, Z., Lv, J. & Peng, X. All-in-One Image Restoration for Unknown Corruption. in 17452–17462 (2022). doi:10.1109/CVPR52688.2022.01693.

33. Mandal, D., Chattopadhyay, S., Tong, G. & Chakravarthula, P. UniCoRN: Latent Diffusion-based Unified Controllable Image Restoration Network across Multiple Degradations. in 1543–1553 (2026). doi:10.1109/WACV61042.2026.00156.

34. Chen, L., Chu, X., Zhang, X. & Sun, J. Simple Baselines for Image Restoration. in Computer Vision – ECCV 2022 17–33 (Springer Nature Switzerland, Cham, 2022). doi:10.1007/978-3-031-20071-7\_2.

35. Yun, S., Han, D., Oh, S. J., Chun, S., Choe, J. & Yoo, Y. CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features. in 6023–6032 (2019). doi:10.1109/ICCV.2019.00612.

36. Otsu, N. A threshold selection method from gray-level histograms. IEEE Trans. Syst. Man Cybern. 9, 62–66 (1979).

37. Sobel, I. & Feldman, G. A 3 × 3 isotropic gradient operator for image processing. (1968).

# Broadband Mid-Wave Infrared Imaging Enabled by All-Silicon Hyperboloid Metalens

Rose Johnson<sup>1,#</sup> , Debrabrata Mandal<sup>2,#</sup>, Zihao Zou<sup>2</sup>, Jessie Wei<sup>3</sup>, Anna Wirth-Singh<sup>3</sup>, Vladimir

Yarmolik<sup>1</sup>, Johannes Froech<sup>2</sup>, Arka Majumdar<sup>1,3,\*</sup> Praneeth Chakravarthula<sup>2,\*</sup>

<sup>1</sup> Department of Electrical and Computer Engineering, University of Washington, Seattle, WA 98195, USA.

<sup>2</sup> Department of Computer Science, University of North Carolina at Chapel Hill, Chapel Hill, NC 27517, USA.

<sup>3</sup> Department of Physics, University of Washington, Seattle, WA 98195, USA.

水 Corresponding authors. Email: arka@uw.edu, cpk@cs.unc.edu

\# These authors contributed equally

## Contents

Supplementary Notes . 3   
Note 1. Details on the Hyperboloid Metalens Design . 3   
Note 2. Integrated MTF Area Calculation . 3   
Note 3. Inverse Design Approach . . 4   
Note 4. Computational Restoration Algorithm . . 6   
Supplementary Figures . . 8   
Fig. S1. Additional MWIR broadband imaging results from varying targets. . 9   
Fig. S2. MWIR broadband imaging results from exposure bracketed captures . 10   
Fig. S3. MWIR broadband imaging results under varying temperatures 11

Note 1. Details on the Hyperboloid Metalens Design  
![](images/0efd38c246089aded9be24901f4c07c4a809d3316b0178c4060aa0c341803d16.jpg)

![](images/9e4474a4821b9814a2a718c95ec2fa176e588b1f34e8fb0ebff34445c6a64ea9.jpg)  
The scatterer (a) phase and (b) transmission responses are calculated using rigorous coupled wave analysis (RCWA) for pillar widths up to $2 \mu \mathrm { m } .$ . A discrete set of scatterers are selected with reasonable transmission to achieve the full 0 – 2π phase range. The dimensions are set to a height of 3 μm with a lattice periodicity of $2 \mu \mathrm { m }$ . The wavelength is 4.5 μm. The hyperboloid phase profile follows the equation,

$$
\varphi ( x , y ) = - \frac { 2 \pi } { \lambda } ( \sqrt { x ^ { 2 } + y ^ { 2 } + f ^ { 2 } } - f
$$

Where #, % are the position coordinates, . is the lens focal length, and + is the nominal wavelength.

To fabricate the hyperboloid metalens, the calculated phase and transmission data is used as a lookup table to map the selected phase to corresponding pillar width. The fabricated metalens is all-Silicon with a 2 cm diameter and 2 cm focal length.

## Note 2. Integrated MTF Area Calculation

The fractional bandwidth calculation includes an evaluation of imaging performance based on the simulated MTF area as function of spectral bandwidth. For a select number of bandwidths up to the full MWIR range, the wavelength range is centered around 4.5 μm and sampled every 10 nm. The fractional bandwidth considers the span of the wavelength range divided by the center frequency, and the monochromatic case is considered to be a bandwidth of 0.

To calculate the PSF, the angular spectrum method is used and the broadband PSF is obtained by integrating the non-normalized PSFs over the wavelengths across given bandwidth. The MTF is computed from the FFT of the PSF. Next, to find the MTF area, the MTF values are integrated between spatial frequency values between 0 to 100 cycles/mm across increasing bandwidths in the 10 nm intervals. Integration is performed using the numpy trapz function. The MTF area is normalized by the hyperboloid reference at zero bandwidth (monochromatic). This approach is completed for the hyperboloid phase profile and the four MTF-engineered phase profiles. A direct comparison is considered to find the difference between the MTF area between the MTFengineered and hyperboloid at those specific bandwidth ranges.

## Note 3. Inverse Design Approach

To perform the fractional bandwidth comparison, four inverse design profiles are simulated with the approach described in previous articles. The proxy model used a $8 ^ { \mathrm { t h } }$ order harmonics for phase-to-structure mapping and $5 ^ { \mathrm { t h } } \left( \mathbf { x } = 5 , \mathbf { y } = 5 \right)$ harmonic orders for the inverse structure-tophase mapping. This process achieves a differentiable mapping between structure (duty cycle) and phase to fit the phase data of the metasurfaces to a polynomial proxy function,

$$
d ( \boldsymbol { r } ) = \sum _ { i = 0 } ^ { N } b _ { i } \left( \frac { \phi ( \boldsymbol { r } ) } { 2 \pi } \right) ^ { 2 i }
$$

Where /(0) is the required duty cycle at a given position 0 from the optical axis of the metasurface, $\phi ( r )$ is the phase at the given position for the nominal wavelength + = 4.0 :;, and $b _ { i }$ are fittting parameters.

The inverse design metalenses include an all-Silicon design which considers a 2 μm periodicity and 3 μm height. The diameter is 1.2 cm and the focal length is 2.5 cm across the full $3 - 5$ μm wavelength range.

The proxy function is modeled by fitting the pre-computed transmission coefficients with a radially symmetric second-order polynomial function,

$$
\tilde { \phi } ( r , \lambda ) = \sum _ { n = 0 } ^ { 2 } \sum _ { m = 0 } ^ { 2 } b _ { n m } d ( r ) ^ { n } \lambda ^ { m } , n + m \leq 2
$$

Where + is a non-nominal wavelength and $b _ { n m }$ are fitting parameters. The transmission values are calculated using RCWA and fitted to the polynomial using linear least squares. We initialize the radial phase profile $\phi ( r )$ with a quadratic focusing phase and optimize its values at discrete radial positions. At each iteration, the proxy functions map $\phi ( r )$ to the dutycycle distribution $d ( r )$ and predict the phase response $\tilde { \phi } ( r , \lambda )$ . For wavelengths sampled uniformly across the optimization band, the transmitted fields are propagated to the focal plane using the angular spectrum method implemented with a zeroth-order Hankel transform. The resulting PSFs are used to calculate the normalized MTFs.

The phase profile is optimized by minimizing the loss function,

$$
\mathcal { L } [ \phi ] = - \frac { 1 } { N _ { \lambda } N _ { \nu } } { \sum _ { j = 1 } ^ { N _ { \lambda } } } { \sum _ { k = 1 } ^ { N _ { \nu } } } \big ( \nu _ { k } , \lambda _ { j } ; \phi \big ) ,
$$

where $N _ { \lambda }$ is the number of sampled wavelengths and $N _ { \nu }$ is the number of spatial-frequency samples below the optimization cutoff $\nu _ { c }$ . Gradients with respect to the sampled phase values are

computed by automatic differentiation through the proxy functions and propagation model. The Adam optimizer updates the phase profile, and the procedure is repeated independently for each of the four optimization bands.

## Note 4. Computational Restoration Algorithm

We implement a probabilistic image restoration framework for MWIR images, based on a generative all-in-one restoration model. Given a degraded MWIR image, the model first extracts low-level visual cues, such as edges, textures, and intensity information. The model has been pretrained on a large-scale dataset spanning multiple image degradation and restoration tasks, allowing it to recognize these common image features across different degradation types. This prior knowledge is particularly useful for MWIR restoration, where the captured images can differ substantially from the data encountered during pretraining. Although the MWIR images themselves may be out of distribution, their low-level features share common structure with those learned from other restoration tasks. Representing the input through these features therefore provides a more generalizable prior for restoring previously unseen MWIR degradations.

The extracted low-level cues are then used as conditioning information for a denoising diffusion based restoration model. Within its latent space, the model identifies the degradations present in the input and combines learned degradation-specific expert adaptors to guide the restoration process. This allows the model to draw on restoration strategies learned from multiple degradation types rather than relying on a model trained specifically for the MWIR imaging system. Finally, the restored output undergoes a refinement stage to reduce artifacts introduced by the domain gap between the pretraining and MWIR datasets. Together, these stages allow a pretrained restoration model to generalize to the chromatic degradation produced by our MWIR metalens while requiring only limited system-specific adaptation.

(d) Unet (Baseline)

(b) Restored

Supplementary Figures

(c ) Refined  
(e) Target  
![](images/afcf9c837cf4ae63adb7be56ef182b57c11d5abc26c87e88556e548f4388c1ff.jpg)

(a) Input

(b) Restored

(c ) Refined

(d) Unet (Baseline)

(e) Target

![](images/b3bcbda96f80c11280cab90f1ea3e08ba1143728523b73dd43dcbafb8d333e5d.jpg)  
Fig. S1. Additional MWIR broadband imaging results from varying targets.

Experimental imaging results from additional target stencils captured by the hyperboloid metalens (input) with the restored image results compared to the ground truth baseline. (a) Captured images using a monochromatic detector (FLIR A6751). (b-c) Restored images following the successive restoration and color refinement stages described in Methods. (d)

Comparison against Unet-based baseline restoration pipeline. (e) Ground truth target images for comparison.

While there are some artifacts that appear as a result from the computational imaging reconstruction pipeline, there is broadly an increase in the broadband intensity contrast produced between the input and refined image sets with additional restoration of fine feature details particularly at the target stencil boundaries.

(a) Input  
(b) Restored (c) Refined  
(d) Unet  
![](images/5330a9b8e9a65824c715b4ec46abbf1c546bda5da9bcd38d62705e0d766c544b.jpg)  
Fig. S2. MWIR broadband imaging results from exposure bracketed captures

Experimental imaging results from additional target stencils captured by the hyperboloid metalens (input) with the restored image results compared to the ground truth baseline at increasing exposure times. (a) Captured images using a monochromatic detector (FLIR A6751). (b-c) Restored images following the successive restoration and color refinement stages described in Methods. (d) Comparison against Unet-based baseline restoration pipeline.

(a) Input  
(b) Restored  
(c) Refined  
(d) Unet  
![](images/89ee36cf94ce88fb137939cd5e1a837c0e866f83402bda91de77660edd3f177e.jpg)  
Fig. S3. MWIR broadband imaging results under varying temperatures

Experimental imaging results from target stencils captured by the hyperboloid metalens (input) with the restored image results compared to the ground truth baseline at different stencil temperatures. Two stencils are imaged at different temperatures ranging from $1 0 0 { - } 2 5 0 ^ { \circ } \mathrm { C }$ and the resulting image is captured in the MWIR camera. The output is restored using both our method and compared against the Unet based baseline method.