# AntennaFlow: A Generative Flow Model for Offset Correction in Phaseless Antenna Testing

1<sup>st</sup> Yongzhi Li 1st Yongzhi Li

College of Computing and Data Science Nanyang Technological University Singapore YONGZHI001@e.ntu.edu.sg

2<sup>nd</sup> Chongting Shen

Beihang Sino-French Engineer School Beihang University Beijing, China schongting@buaa.edu.cn

3<sup>rd</sup> Menglin Chen

Beihang Sino-French Engineer School Beihang University Beijing, China elvin chen@buaa.edu.cn

4<sup>th</sup> Xun Jiang School of Computer Science and Engineering University of Electronic Science and Technology of China Chengdu, China xun jiang@std.uestc.edu.cn

Abstract—Near-field to far-field transformation is central to large-aperture antenna testing, yet two coupled challenges remain: costly phase acquisition at millimeter-wave bands and violations of the centering assumption under offset mounting. Existing methods address these issues separately, requiring either dense full-field data or offset vectors. We tackle both jointly by exploiting a key observation: amplitude fields under different offsets are coordinate-transformed views of the same near field. The challenge is to recover the center-aligned field from offset amplitudes without a phase or offset vector. We propose AntennaFlow, a three-stage framework: a contrastively learned encoder that maps offset views to an offset-invariant embedding, a deterministic flow-matching transport that maps offset amplitudes to center-aligned ones, and the Simplified Extrapolation Technique, whose Green-function Taylor expansion is valid only for centered fields. Experiments show that AntennaFlow enables fast, phaseless, offset-vector-free NF–FF reconstruction from sparse amplitude-only measurements, consistently outperforming existing baselines while preserving physical consistency.

Index Terms—Phaseless antenna testing, Offset Correction, Generative AI, Contrastive Learning, Conditional Flow Matching

## I. INTRODUCTION

With the rise of 6G [1], large-aperture arrays such as massive MIMO and phased arrays [2] are widely used in satellites and radar. Large electrical size leads to prohibitively long Rayleigh distances, making direct far-field testing impractical. Near-field to far-field (NF–FF) transformation [3] thus serves as the primary Over-The-Air (OTA) testing approach, enabling accurate measurements via near-field.

Despite its maturity, near-field measurement still faces two key challenges in practice: phase acquisition and offset placements [4]. At millimeter-wave frequencies, accurate phase measurement requires stable RF cables and precise synchronization, leading to high cost and long acquisition time. To address this, phaseless methods such as the Extrapolation Technique (ET) [5] and Simplified Extrapolation Technique

5<sup>th</sup> Zhengpeng Wang   
Electronic Information Engineering   
Beihang University   
Beijing, China   
wangzp@buaa.edu.cn

![](images/891c842b55024c1fcf61f182374fa7bc38c8dcf73949aac20df21fb0f2a1dfce.jpg)  
Fig. 1. Comparison of far-field reconstruction under offset mounting: prior methods suffer from offset sensitivity or reliance on offset vectors, while AntennaFlow achieves superior reconstruction quality.

(SET) [6] reconstruct the far field from amplitude-only data, avoiding the burden of phase measurements.

However, array antennas are often mounted as subsystems on complex platforms, including satellites, vehicles, and radar domes, making precise alignment with the measurement coordinate center difficult and degrading the measured amplitude and phase. TSWE [7] compensates for offset mounting, while SRM [8] recovers phase and can also mitigate offset. However, both suffer from low sampling efficiency and reliance on offset vectors, limiting practical applicability. Consequently, few methods can efficiently handle offset correction under phaseless settings, motivating offset-vector-free offset correction, as illustrated in Fig. 1.

We resolve this by noting that different offset placements yield only coordinate-transformed views of the same near field.

The challenge is to recover the centered field from offset amplitudes without phase and offset vector. Based on this, AntennaFlow uses contrastive learning to obtain an offsetinvariant embedding, a deterministic flow to map offset amplitudes to centered ones, and SET to extract the far field from the centered output. The three stages thus remove offset-induced errors in the NF-FF pipeline without ever observing the offset vector or any phase information. The main contributions of this work are summarized as follows:

• We unify phaseless NF-FF transformation and offset correction into an end-to-end amplitude calibration without phase or offset vectors.

• We realize this as AntennaFlow: offset-invariant amplitude embedding, deterministic flow-matching transport, and center-aligned SET extraction for efficient NF-FF reconstruction without phase or offset vectors.

• To our knowledge, this is the first application of generative AI to NF–FF transformation, achieving consistent gains in reconstruction quality, efficiency, and robustness under realistic measurement perturbations.

## II. BACKGROUND

## A. Antenna Testing

1) Phaseless Testing: Spherical near-field measurement, based on Hansen’s spherical wave expansion (SWE) [9], is the benchmark for high-accuracy antenna testing. However, costly and unstable phase measurements have driven phaseless approaches that reconstruct the far field from amplitudeonly data. SRM [8] estimates equivalent sources via iterative optimization, while ET [5] applies polynomial fitting for gain calibration. In our work, we adopt the Simplified Extrapolation Technique (SET) [6] as our core solver, which directly fits amplitude-decay curves to avoid iterative local minima and slow inference, and naturally aligns with our calibration through its dependence on array centering.

2) Offset Correction: Offset mounting enlarges the enclosing sphere, increasing Nyquist sampling demands, cost, and measurement degradation. TSWE [7] requires full-field data, SRM relies on costly phase retrieval, and SET, derived from Green’s formula with a local Taylor expansion, cannot handle offsets. Existing methods thus require either an offset vector and complete data or heavy computation. In contrast, we exploit shared structures across offset-induced distortions and learn a calibration map for correction under sparse, phaseless measurements without offset vector inputs. Table I summarizes the comparison.

## B. Generative AI

1) Image Generation with Diffusion and Flow Models: Recent diffusion-based generative models [10], [11] have achieved strong performance in image generation [12]. Our task, however, maps offset near-field amplitudes to physically meaningful center-aligned counterparts, rather than generating diverse images. SDE-based models introduce stochasticity that benefits natural image generation but leads to undesirable fluctuations for precise near-field calibration. In contrast, ODEbased flow-matching models provide deterministic mapping, making it better aligned with the requirements of this task.

TABLE I  
COMPARISON OF THE PREVIOUS METHODS AND THE PROPOSED METHOD
<table><tr><td rowspan="2">Capabilities</td><td colspan="6">Method</td></tr><tr><td>ET</td><td>SET</td><td>SWE</td><td>TSWE</td><td>SRM</td><td>Ours</td></tr><tr><td>Sparse Sampling</td><td>x</td><td>√</td><td>x</td><td>x</td><td>x</td><td>√</td></tr><tr><td>Fast Inference</td><td>x</td><td>√</td><td>x</td><td>x</td><td>x</td><td>√</td></tr><tr><td>Phaseless</td><td>√</td><td>√</td><td>x</td><td>x</td><td>√</td><td>√</td></tr><tr><td>Offset Correction</td><td>x</td><td>x</td><td>x</td><td>√</td><td>√</td><td>√</td></tr><tr><td>Offset-Vector Free</td><td>一</td><td>一</td><td>一</td><td>x</td><td>x</td><td>√</td></tr></table>

2) Contrastive Learning: Contrastive learning organizes representations by pulling positive pairs together and pushing negative pairs apart. In our setting, we use antenna identity as the supervision signal in a supervised contrastive framework [13], treating different offset views of the same antenna as positives, while views from different antennas are treated as negatives. This encourages the encoder to capture features that are invariant to spatial offsets, resulting in robust near-field representations that support offset-vector-free reconstruction.

## III. METHOD

## A. Design Overview

AntennaFlow is motivated by a simple measurement observation: offset placements do not alter the underlying physical near field, but only its coordinate representation. The core challenge is therefore to recover a canonical, center-aligned field from amplitude-only measurements without access to phase or offset vectors. To address this, we decompose the problem into three components. First, we learn an offsetinvariant representation f that captures antenna-specific characteristics shared across different offset measurements. Second, we model a deterministic conditional transport that maps an input amplitude field to its center-aligned counterpart, with f providing invariant structural guidance. Third, we ensure that the reconstructed field satisfies the assumptions required by the downstream phaseless solver (SET), whose formulation is valid only under center alignment. Together, these components enable an end-to-end pipeline for offset correction and phaseless NF–FF transformation, as illustrated in Fig. 2.

## B. AntennaFlow for Near-Field Calibration

AntennaFlow uses a contrastively pretrained ResNet encoder [14] and a U-Net flow-matching backbone [15] conditioned on its embedding to learn deterministic transport from offset amplitudes to center-aligned amplitudes.

1) Offset-Invariant Antenna Embedding: For a batch of labeled pretraining samples $\{ ( X _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { B } , X _ { i }$ denotes a nearfield amplitude and $Y _ { i }$ denotes the antenna identity label, which is available only during the training stage; the encoded representation is $\begin{array} { r } { \begin{array} { r c l } { h _ { i } } & { = } & { E ( X _ { i } ) } \end{array} } \end{array}$ , where $E ( \cdot )$ is the feature encoder. With this label assignment, the supervised contrastive paradigm [13] becomes a direct mechanism for compressing the set of offset views of each antenna into a tight cluster: for each anchor i, its positive set $P ( i ) = \{ j \ne i \mid Y _ { j } = Y _ { i } \}$ collects offset views of the same antenna, while the remaining $Y _ { j } ~ \neq ~ Y _ { i }$ are negatives. The cosine similarity scaled by temperature τ is computed as:

![](images/f902c19262787c166bce950c447e470e024a83de4f38a0a243665f0ac6c265b6.jpg)  
Fig. 2. The AntennaFlow pipeline for NF-FF transformation. (1) Data Collection: Near-field data are sampled from different antenna configurations. (2) A feature encoder is trained using a supervised contrastive learning framework to learn discriminative representations. (3) Flow Matching Transport: A U-Netbased architecture is used to learn the flow-matching velocity field for center near-field generation. (4) Far-Field Reconstruction via SET. Note that SET requires measurements from multiple distinct surfaces; for simplicity, only one surface is illustrated in the figure.

$$
s _ { i j } = \frac { h _ { i } ^ { \top } h _ { j } } { \tau } .\tag{1}
$$

The supervised InfoNCE loss is defined as:

$$
\mathcal { L } _ { \mathrm { S u p C o n } } = - \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \frac { 1 } { | P ( i ) | } \sum _ { p \in P ( i ) } \log \frac { \exp ( s _ { i p } ) } { \sum _ { j = 1 , j \neq i } ^ { B } \exp ( s _ { i j } ) } .\tag{2}
$$

Minimizing ${ \mathcal { L } } _ { \mathrm { { S u p C o n } } }$ encourages approximate invariance of the embedding across different offset near-fields. A learnable MLP $M ( \cdot )$ then produces the final conditional embedding:

$$
\mathbf { f } = M { \bigl ( } E ( X _ { i } ) { \bigr ) } ,\tag{3}
$$

which supports the next deterministic transport.

2) Deterministic Transport via Conditional Flow Matching: Given an offset amplitude x and its centered counterpart $x _ { 1 }$ , we learn a conditional transport rather than an unconditional generation. Specifically, we construct a displacement interpolation between a noise prior $x _ { 0 } \ \sim \ { \mathcal { N } } ( 0 , I )$ and $x _ { 1 }$ as $x _ { t } ~ = ~ ( 1 - t ) x _ { 0 } + t x _ { 1 }$ , whose time derivative yields a constant ground-truth velocity $\begin{array} { r } { v ^ { * } ( x _ { t } ) = \frac { d x _ { t } } { d t } = x _ { 1 } - x _ { 0 } } \end{array}$ . The velocity field $v _ { \theta } ( \cdot , \cdot , \mathbf { f } )$ is parameterized by a U-Net that takes the interpolated state $x _ { t } ,$ , the timestep $t ,$ and the conditional embedding f as inputs, and outputs a velocity vector of the same dimensionality as $x _ { t }$ . The model is trained via:

$$
\mathcal { L } _ { \mathrm { F M } } = \mathbb { E } _ { { x _ { 0 } } , { x _ { 1 } } , t } \left. v _ { \theta } ( { x _ { t } } , t , \mathbf { f } ) - v ^ { * } ( { x _ { t } } ) \right. _ { 2 } ^ { 2 } .\tag{4}
$$

Crucially, the conditioning $\mathbf { f } = M ( E ( x ) )$ is extracted from the input amplitude x, serving as an offset-invariant anchor of

the antenna identity. As a result, the transport is conditioned on x through f, rather than discarding input information.

At inference, we integrate from $x _ { 0 } \sim \mathcal { N } ( 0 , I )$ under f using the U-Net velocity field:

$$
x _ { t _ { k + 1 } } = x _ { t _ { k } } + \Delta t v _ { \theta } ( x _ { t _ { k } } , t _ { k } , \mathbf { f } ) ,\tag{5}
$$

and the final state $\hat { x } _ { 1 }$ is taken as the generated sample.

## C. Far-Field Transformation via SET

After offset removal, we apply SET [6] as the NF-FF transformation module. Specifically, AntennaFlow generates calibrated near-field amplitudes conditioned on the offset nearfield amplitudes of multiple measurement radii. In this work, we generate 3 calibrated near-field maps at distinct distances $r _ { 1 } , r _ { 2 }$ , and $r _ { 3 } .$ which serve as the required inputs for SET.

Let $\hat { x } _ { 1 } ( r , \theta , \phi )$ denote the calibrated near-field amplitude at radius r. Under the center-alignment property guaranteed by AntennaFlow, the spatial variation of the near-field is parameterized by angular coordinates $( \theta , \phi )$ . Following the SET formulation, we model the distance-dependent power decay by approximating the product of the power density $P ( r , \mathbf { \bar { \theta } } , \phi ) = \bar { | } \hat { x } _ { 1 } ( r , \theta , \phi ) \bar { | } ^ { 2 }$ and $r ^ { 2 }$ using a truncated polynomial expansion in $r ^ { - 2 k }$ , derived from the Taylor expansion of the Green’s function:

$$
P ( r , \theta , \phi ) \cdot r ^ { 2 } \approx \sum _ { k = 0 } ^ { K } A _ { 2 k } ^ { \prime } ( \theta , \phi ) \cdot r ^ { - 2 k } .\tag{6}
$$

where $A _ { 0 0 } ^ { \prime }$ is the zero-order intercept coefficient representing the far-field radiation characteristic, while higher-order terms account for the reactive and radiative near-field decay components. In practice, for sufficiently large measurement distances, higher-order terms decay rapidly with increasing powers of $r ^ { - 2 }$ , and prior work has shown that accurate approximation can be achieved using only the first three terms:

$$
P ( r , \theta , \phi ) \cdot r ^ { 2 } \approx A _ { 0 0 } ^ { \prime } ( \theta , \phi ) + { \frac { A _ { 0 2 } ^ { \prime } ( \theta , \phi ) } { r ^ { 2 } } } + { \frac { A _ { 0 4 } ^ { \prime } ( \theta , \phi ) } { r ^ { 4 } } } .
$$

To estimate the far-field coefficient $A _ { 0 0 } ^ { \prime }$ , AntennaFlow generates calibrated amplitude maps at three distinct mid-field distances, $r _ { 1 } , r _ { 2 } ,$ , and $r _ { 3 }$ . For each distance $r _ { m }$ , we define the processed observation as:

$$
y _ { m } ( \theta , \phi ) = | \hat { x } _ { 1 } ( r _ { m } , \theta , \phi ) | ^ { 2 } r _ { m } ^ { 2 } = P ( r _ { m } , \theta , \phi ) \cdot r _ { m } ^ { 2 } .\tag{7}
$$

The resulting observations provide a linear system for solving the polynomial coefficients:

$$
\left[ y _ { 1 } ( \theta , \phi ) \right] = \left[ 1 \quad r _ { 1 } ^ { - 2 } \quad r _ { 1 } ^ { - 4 } \right] \left[ A _ { 0 0 } ^ { \prime } ( \theta , \phi ) \right] .\tag{8}
$$

Solving this system independently for each angular pixel $( \theta , \phi )$ yields the far-field coefficient $A _ { 0 0 } ^ { \prime }$ . The far-field pattern $F ( \theta , \phi )$ is then obtained from this extracted intercept term:

$$
F ( \theta , \phi ) \propto \sqrt { A _ { 0 0 } ^ { \prime } ( \theta , \phi ) } .\tag{9}
$$

This pipeline enables phaseless, offset-free NF–FF transformation via learned center alignment consistent with SET.

## D. Training Data Collection

We construct a synthetic dataset to model near-field amplitude responses of antenna arrays under spatial misalignment. For each antenna, the near field is uniquely determined by its array structure and element parameters, and is computed as

$$
E ( \mathbf { r } ) = \sum _ { n = 1 } ^ { N } I _ { n } f _ { n } ( \theta _ { n } , \phi _ { n } ) { \frac { e ^ { - j k | \mathbf { r } - ( \mathbf { r } _ { n } + \mathbf { d } ) | } } { | \mathbf { r } - ( \mathbf { r } _ { n } + \mathbf { d } ) | } } ,\tag{10}
$$

where N denotes the number of elements, $I _ { n }$ the fixed element excitation, $f _ { n } ( \theta , \phi )$ the normalized element pattern, $\mathbf { r } _ { n }$ the element position with respect to the measurement center, and d the offset vector representing array misalignment.

According to prior work on near-field synthesis of nonuniformly spaced arrays [16], the near-field pattern is governed by the element locations, spacings, and element parameters; consequently, any change in array position or spacing leads to a different near-field distribution. Therefore, under the sampling conditions and parameter space defined in this work, each amplitude map sampled over the spherical domain is generated from a unique antenna configuration and can be treated as a single-solution supervision target for both the generative model and subsequent measurement validation.

## IV. EXPERIMENTS AND RESULTS

## A. Experiments setup

1) Test Dataset: To evaluate generalization to unseen antennas, we construct an independent test set of 1200 MATLABderived and 300 full-wave FEKO near-fields. The training and test antennas are strictly disjoint and differ in element type, spacing, and aperture size, with additional unseen offset configurations. The FEKO datasets introduce richer electromagnetic effects and a solver shift to assess cross-solver robustness. Together, these settings enable a comprehensive evaluation of reconstruction quality, physical fidelity, and robustness on antennas entirely unseen by the model.

![](images/368b3491758d57d8f6e68f51a54ce96a62efed7fab4491296f7c804465bce981.jpg)  
Fig. 3. Comparison of antenna far-field reconstruction using different phaseless reconstruction methods. The red solid curves denote the reconstructed far-field patterns, while the black dashed curves represent the theoretical farfield reference. The cyan curves indicate the corresponding ESS.

2) Evaluation Metrics: To quantitatively evaluate the reconstructed amplitude patterns in both the far field and the calibrated near-field produced by AntennaFlow, we adopt the Equivalent Stray Signal (ESS) [17], a standard metric in antenna measurements, which interprets reconstruction error as an equivalent stray radiation field superimposed on the target pattern. Let $E _ { r e c } ( \theta )$ and $E _ { r e f } ( \theta )$ denote the normalized amplitudes of the reconstructed and reference patterns, defined either at a fixed measurement radius in the near field or in the far field. Following [6], the ESS at angle θ is defined as:

$$
\begin{array} { r } { \mathrm { E S S } ( \theta ) = 2 0 \log _ { 1 0 } ( E _ { r e f } ( \theta ) ) \ ~ } \\ { + 2 0 \log _ { 1 0 } \left[ \frac { 1 - 1 0 ^ { - \Delta _ { \mathrm { d B } } ( \theta ) / 2 0 } } { 2 } \right] . } \end{array}\tag{11}
$$

where $\Delta _ { \mathrm { d B } } ( \theta ) = 2 0 \log _ { 1 0 } ( E _ { r e c } ( \theta ) ) - 2 0 \log _ { 1 0 } ( E _ { r e f } ( \theta ) )$ denotes the pattern difference in decibels. The ESS is evaluated using its Root Mean Square (RMS) and Peak values: a lower RMS ESS indicates better overall reconstruction fidelity, while Peak ESS reflects the maximum local deviation. By design, ESS is more sensitive to errors in high-energy regions and less sensitive to discrepancies in low-energy areas with limited physical relevance, thereby emphasizing deviations that impact radiation characteristics and better reflecting physical field behavior rather than local numerical artifacts. Consequently, it effectively characterizes the influence of reconstruction errors on sidelobe levels and other key features.

![](images/fc4b6349de262bf510e694904e22d953714f4f772ab1f356afc453bae448f7bc.jpg)  
Fig. 4. Visual comparison of reconstructed far-field amplitude patterns under offset conditions. From left to right: SET, TSWE, SRM, AntennaFlow, and the ground truth. AntennaFlow demonstrates the highest fidelity in restoring detailed pattern textures compared to the ground truth.

![](images/e3c39cab18f94ba2dc8c37676cffee24ebf84f1b25498508ec46b2dd10116027.jpg)  
Fig. 5. Reconstruction from offset near-field measurements. Different offset inputs (top) are mapped to consistent center-aligned fields (bottom).

3) Compared Methods: To rigorously validate the proposed framework, we compare it with SET, TSWE, and SRM, evaluating near-field and far-field reconstruction quality.

## B. Results

1) Quantitative Analysis: As shown in Fig. 3, existing methods degrade under offset conditions: SET produces pattern shifts, TSWE fails to recover the main lobe, and SRM shows sidelobe distortions. In contrast, AntennaFlow achieves precise alignment with the ground truth across the full angular spectrum with low ESS, demonstrating robustness to offsets in phaseless settings and preserving physically consistent radiation characteristics, including main lobes and sidelobes.

Table II further shows that AntennaFlow achieves the lowest RMS and Peak ESS on both datasets, indicating superior reconstruction quality with errors concentrated in less significant regions, while offering efficient inference much faster than TSWE and SRM and comparable to SET, demonstrating advantages in both fidelity and efficiency.

TABLE II  
QUANTITATIVE COMPARISON OF FAR-FIELD RECONSTRUCTION ACCURACY (RMS ESS AND PEAK ESS IN DB)
<table><tr><td>Dataset</td><td>Metric</td><td>SET</td><td>TSWE</td><td>SRM</td><td>Ours</td></tr><tr><td rowspan="2">MATLAB</td><td>RMS ESS (↓)</td><td>-18.97</td><td>-28.37</td><td>-34.44</td><td>-36.90</td></tr><tr><td>Peak ESS (↓)</td><td>-6.91</td><td>-18.48</td><td>-24.32</td><td>-27.06</td></tr><tr><td rowspan="2">FEKO</td><td>RMS ESS (↓)</td><td>-18.28</td><td>-25.25</td><td>-29.70</td><td>-33.81</td></tr><tr><td>Peak ESS (↓)</td><td>-8.42</td><td>-15.34</td><td>-18.33</td><td>-25.58</td></tr><tr><td></td><td>Inference Time</td><td>4s</td><td>10 min</td><td>30 min</td><td>9s</td></tr></table>

2) Visualization Results: To evaluate the reconstruction fidelity, we perform a comparative analysis of far-field (Fig. 4) and near-field (Fig. 5) results. In the far field, our method demonstrates superior quality compared to SET, TSWE, and SRM. It corrects severe misalignment and reconstructs highfidelity patterns from amplitude-only data while preserving main lobes and sidelobes. In the near field, Fig. 5 groups multiple offset placements of the same antenna with their reconstructions and the ground truth. Despite diverse offsets, the reconstructions converge to nearly identical center-aligned fields and closely match the ground truth. This consistency under heterogeneous offset inputs is direct visual evidence that AntennaFlow extracts the offset-invariant embedding of the antenna and reconstructs its near-field amplitude, which is the property the downstream SET extraction relies on.

TABLE III  
ABLATION STUDY ON THE EFFECT OF CONTRASTIVE PRE-TRAINING
<table><tr><td>CL Use</td><td>NFRMS</td><td>NF Peak</td><td>FF RMS</td><td>FF Peak</td></tr><tr><td>w/o CL</td><td>-25.74</td><td>-19.18</td><td>-24.43</td><td>-16.46</td></tr><tr><td>w/ CL</td><td>-34.39</td><td>-26.40</td><td>-33.81</td><td>-25.58</td></tr></table>

## C. Validation of the Learned Embedding and Reconstruction

The two points of AntennaFlow, an offset-invariant embedding and a deterministic flow-matching transport conditioned on it, together carry the burden of converting an offset amplitude into a center-aligned amplitude. We verify that each part behaves as intended through three complementary checks.

First, we verify that the encoder captures offset-invariant features by visualizing embeddings of offset samples from 24 antennas using t-SNE (Fig. 6). Embeddings from different antennas are well separated, while those from the same antenna under varying offsets form tight clusters, confirming that the encoder captures antenna-specific features invariant to offsets.

Second, we check whether this embedding suffices to drive the deterministic flow-matching transport back to the centeraligned field. As shown in Fig. 5, reconstructions from different offsets of the same antenna are nearly identical and closely match the ground truth. NF ESS and FF ESS evaluation further shows that the reconstructed fields preserve the key information while remaining physically consistent.

Third, we conduct an ablation study by removing contrastive pretraining and retraining the pipeline end-to-end. As shown in Table III, all metrics degrade, confirming its key role in extracting the offset-invariant features for the framework.

![](images/1d38c3e6144d2fdc240d32d568de80b2756b709b34e791ac53cc9c18c938da4f.jpg)  
Fig. 6. t-SNE projection of the encoder embeddings of 768 amplitude samples drawn from 24 distinct antenna configurations under random offsets. Colors denote antenna identity. Samples of the same antenna fall into tight clusters, while clusters of distinct antennas are clearly separated.

TABLE IV  
SIM-TO-REAL ROBUSTNESS EVALUATION.(RMS ESS AND PEAK ESS IN DB)
<table><tr><td>Perturbation</td><td>NF RMS</td><td>NF Peak</td><td>FF RMS</td><td>FF Peak</td></tr><tr><td>Default</td><td>-35.39</td><td>-26.40</td><td>-33.81</td><td>-25.58</td></tr><tr><td>Horn Probe</td><td>-33.94</td><td>-25.89</td><td>-32.67</td><td>-24.50</td></tr><tr><td>Dipole Probe</td><td>-34.28</td><td>-24.48</td><td>-32.91</td><td>-24.81</td></tr><tr><td>Additive noise</td><td>-33.49</td><td>-25.18</td><td>-31.07</td><td>-23.67</td></tr></table>

## D. Sim-to-Real Robustness Evaluation

To facilitate deployment in OTA antenna testing, we explicitly evaluate the sim-to-real gap arising from probe-dependent measurement responses and environmental noise. To approximate real measurement conditions, we introduce two representative perturbations: (1) Probe mismatch, using horn and dipole probes to simulate real-world sampling conditions and typically encountered in laboratory setups, and (2) introduced to model coupling effects and multipath interference arising from hardware imperfections and environmental uncertainty. These factors constitute the dominant sources of discrepancy between simulation and real OTA measurements. As shown in Table IV, AntennaFlow remains stable under these perturbations, with only minor degradation in reconstruction accuracy, indicating strong robustness to realistic measurement conditions and effective generalization beyond idealized simulations.

## V. CONCLUSION

In this paper, we present AntennaFlow, which uses the basic measurement-frame fact that offset-mounted antenna measurements of the same antenna are merely different coordinate views of the same physical near field. A feature encoder turns this fact into an offset-invariant embedding directly extractable from offset-mounted amplitude measurements; a deterministic flow-matching transport conditioned on this embedding reconstructs the center-aligned near field; and the recovered center alignment is precisely the precondition required by SET to complete a phaseless, offset-vector-free NF–FF reconstruction. AntennaFlow outperforms SRM, the strongest baseline in our comparison, offering an accurate, efficient, and lowcomplexity solution for modern OTA antenna testing.

## REFERENCES

[1] W. Jiang, B. Han, M. A. Habibi, and H. D. Schotten, “The road towards 6g: A comprehensive survey,” IEEE Open Journal of the Communications Society, vol. 2, pp. 334–366, 2021.

[2] E. Bjornson, L. Sanguinetti, H. Wymeersch, J. Hoydis, and T. L.¨ Marzetta, “Massive mimo is a reality–what is next?: Five promising research directions for antenna arrays,” Digital Signal Processing, vol. 94, pp. 3–20, 2019.

[3] O. M. Bucci, F. D’Agostino, C. Gennarelli, G. Riccio, and C. Savarese, “Near-field-far-field transformation with spherical spiral scanning,” IEEE Antennas and Wireless Propagation Letters, vol. 2, pp. 263–266, 2005.

[4] M. Sørensen, O. Franek, G. F. Pedersen, A. Radchenko, K. Kam, and D. Pommerenke, “Estimate on the uncertainty of predicting radiated emission from near-field scan caused by insufficient or inaccurate nearfield data: Evaluation of the needed step size, phase accuracy and the need for all surfaces in the huygens’ box,” in International Symposium on Electromagnetic Compatibility-EMC EUROPE. IEEE, 2012, pp. 1–6.

[5] A. Newell, R. Baird, and P. Wacker, “Accurate measurement of antenna gain and polarization at reduced distances by an extrapolation technique,” IEEE Transactions on Antennas and Propagation, vol. 21, no. 4, pp. 418–431, 1973.

[6] L. Yu, F. Zhang, Y. Zhang, X. Zhang, Z. Wang, Y. Jing, and W. Fan, “Antenna pattern reconstruction based on mid-field phaseless measured data using simplified extrapolation technique,” IEEE Transactions on Antennas and Propagation, 2025.

[7] R. Cornelius and D. Heberling, “Spherical wave expansion with arbitrary origin for near-field antenna measurements,” IEEE Transactions on Antennas and Propagation, vol. 65, no. 8, pp. 4385–4388, 2017.

[8] A. Paulus, J. Knapp, and T. F. Eibert, “Phaseless near-field far-field transformation utilizing combinations of probe signals,” IEEE Transactions on Antennas and Propagation, vol. 65, no. 10, pp. 5492–5502, 2017.

[9] J. E. Hansen, Spherical Near-Field Antenna Measurements. IET, 1988, vol. 26.

[10] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Advances in Neural Information Processing Systems, vol. 33, pp. 6840– 6851, 2020.

[11] Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le, “Flow matching for generative modeling,” arXiv preprint arXiv:2210.02747, 2022.

[12] Y. Li, S. Zhang, Y. Chen, B. Li, Y. Zhang, and X. Du, “Spotdiff: Spotting and disentangling interference in feature space for subject-preserving image generation,” arXiv preprint arXiv:2510.07340, 2025.

[13] P. Khosla, P. Teterwak, C. Wang, A. Sarna, Y. Tian, P. Isola, A. Maschinot, C. Liu, and D. Krishnan, “Supervised contrastive learning,” Advances in Neural Information Processing Systems, vol. 33, pp. 18 661–18 673, 2020.

[14] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016, pp. 770–778.

[15] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2015, pp. 234–241.

[16] M. Narasimhan and B. Philips, “Synthesis of near-field patterns of a nonuniformly spaced array,” IEEE Transactions on Antennas and Propagation, vol. 35, no. 11, pp. 1189–1198, 1987.

[17] D. W. Hess, “Historical background on the use of equivalent stray signal in comparison of antenna patterns,” in Proceedings of the 5th European Conference on Antennas and Propagation (EUCAP). IEEE, 2011, pp. 2522–2526.