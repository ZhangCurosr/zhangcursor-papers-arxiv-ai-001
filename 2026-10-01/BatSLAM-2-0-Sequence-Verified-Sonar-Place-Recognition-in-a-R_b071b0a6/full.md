# BatSLAM 2.0: Sequence-Verified Sonar Place Recognition in a Robust Pose Graph

Jan Steckel

Abstract—Echolocating bats can navigate dark and cluttered spaces using echolocation. Over a decade ago, BatSLAM showed that a robot with a biomimetic binaural sonar can build a topological map of the environment, by recognizing places from the received acoustic signals. Sonar place recognition, however, is ambiguous by nature: corridors produce nearly identical echo trains, and wrong loop closure can collapse the topological map. In this paper, we introduce BatSLAM 2.0, a novel sonar-only SLAM system built from three elements: an updated acoustic front-end, a sequence verifier that tracks and verifies loop closure candidates and a pose graph implemented on a high performance factor graph framework. The system was thoroughly evaluated both in simulated as well as real world recordings. In both cases, the BatSLAM2.0 algorithm shows the capability of robust topological map creation, countering map collapse, and robust scaling of map size.

Index Terms—Sonar, biomimetics, SLAM, place recognition, loop closure, pose graph.

## I. INTRODUCTION

Bats are the prime example of an animal that navigates using sound alone. Indeed, echolocating bats find their way through caves, forests and buildings using the echoes of their own calls, often in complete darkness and at high speeds [1], [2]. From an engineering perspective, this ability is more than a biological curiosity as acoustic sensing keeps working in exactly those conditions in which optical sensors such as cameras and lidar sensors struggle (darkness, smoke, dust, fog and transparent or reflective surfaces). Furthermore, ultrasonic sensors themselves are cheap, small and robust and can perform high resolution imaging on complex environments [3], [4].

Early work on sonar-based mapping modeled the environment as a set of geometric primitives (e.g. planes, edges and cilinders), which were tracked in an extended Kalman filter [5], [6]. This approach requires reliable extraction of such primitives from sparse and noisy range readings, which is not always possible in real-world scenarios.

In our earlier work we implemented a biologically inspired SLAM system called BatSLAM [7]. In this algorithm we combined a biomimetic binaural sonar (where the receivers replicated the outer ears of a bat), with a topological model called RatSLAM. This RatSLAM model is a biologically inspired SLAM system based on a model of the rodent hippocampus [8], [9] that creates topological maps of the environment that simultaneously has geometrically valid properties. Instead of extracting landmarks from the received signals, BatSLAM used the complete binaural cochleogram of each echo train as a local view, and recognized robot poses by comparing these views with a database of templates. Later work confirmed that bat-like sonar views indeed carry enough information to recognize real-world complex places [10]. The weak point of such an approach, however, is ambiguity. Many places produce similar local view templates (ie, similar echo trains received by the sonar sensor). A typical example is the two parallel corridors with plain walls that produce nearly constant echo trains irrespective of the position of the robot in that corridor. This can cause false place matches to be injected into the map with full confidence, that subsequently pulls geometrically distinct places onto each other. In the SLAM literature, this failure mode is called map collapse, and it is one of the main reasons why loop closure is typically treated as the most dangerous step in the SLAM pipeline [11], [12].

Typically, two families of techniques are used against false loop closures. On the one hand, the front-end can demand temporal consistency, and compare sequences of views rather than single views, as in SeqSLAM [13]. On the other hand, the back-end can treat loop closures as potentially being outliers (i.e. false loop closures), using robust cost functions, switchable constraints [14], dynamic covariance scaling [15], max-mixtures [16], clustering and consistency checks [17], [18] or graduated non-convexity methods [19]. Both families were developed mostly for SLAM systems relying on vision and lidar, but could fundamentally be applied to other modalities such as in-air sonar. The question of how well they carry over to the much more ambiguous sonar views has, to the best of our knowledge, not been answered so far.

In this paper, we introduce BatSLAM 2.0, a novel sonar-only SLAM system that revisits the original BatSLAM idea with a modern and robust SLAM back-end. The system replaces the pose cells and the experience map of BatSLAM with a factor-graph pose graph solved with iSAM2 [20] in GTSAM [21], and it keeps local view templates that are anchored to pose-graph nodes, so that each template has a position and an orientation that move with the optimized pose graph.

Around this core, we contribute four elements. Firstly, an acoustic front-end that adds an explicit direction cue to the cochleogram: the spectral shape of each echo, which carries the directivity of the ears and the emitter, is separated from the echo magnitude, to more explicitly encode the direction of origin of the reflection. Secondly, a sequence verifier that keeps competing recognition hypotheses optional, and commits to a decision only when it is the loop closure is unambiguously valid. The required evidence to accept a loop closure grows with the geometric correction such commit would force on the map, and plausibility is tested against the relative uncertainty between the matched places.

Thirdly, we propose a link management system in the SLAM back-end, in which the links of every committed hypothesis form a group that can be withdrawn, audited and re-admitted, which allows false loop closures to be pruned efficiently. Fourthly, a simulation study of which information in a sonar view makes a place recognizable, and of the capture regions of individual view templates. Finally, we validate the system on a real recording of an embedded 3D sonar [4], whose microphone array we turned into two virtual ears by beamforming, with a lidar-based ground truth. A video of BatSLAM 2.0 in operation is available online.<sup>1</sup>

The rest of the paper is structured as follows. First, we give an overview of the BatSLAM 2.0 system in section II. Next, we derive the acoustic front-end in section III, after which we describe the recognition and verification of places in section IV and the robust pose graph in section V. In section VI, we detail the simulation setup, and in section VII we present the results, including ablations and a robustness analysis. In section VIII, we apply the system to a real recording. Finally, we discuss the limitations of our work in section IX and conclude the paper in section X.

## II. OVERVIEW OF BATSLAM 2.0

The overall processing flow of BatSLAM 2.0 can be found in figure 1, and the accompanying video (footnote 1) shows it in operation. A broadband signal is emitted by an emitter, and the subsequent echoes are received and processed into a cochlear representation (see [7]. The robot’s odometry adds a new node to the pose graph for each measurement, connected to the previous node by an odometry factor. Each binaural echo cochleogram is converted into a local view descriptor by the acoustic front-end, which is subsequently compared with all stored templates. The best matches in this comparison become recognition candidates, which are handed to the sequence verifier. The sequence verifier checks whether subsequent measurements give rise to subsequent matches in the pose graph, which is a strong signal for recognizing correct loop closures, and then generates a loop closure hypothesis. When the verifier commits to a hypothesis, its matches become weak loop closure links in the pose graph, grouped per hypothesis, which can be withdrawn in a later stage by the link management subsystem in case of competing evidence becoming available.

## III. THE ACOUSTIC FRONT-END

In this section, we describe how a binaural echo train is converted into the local view that is used for place recognition. We start from a model of the received signals, after which we describe the cochlear processing and the two images that make up the local view. The processing steps are illustrated on a simulated pulse in figure 2.

## A. Echo formation

The robot emits a broadband call $s _ { e } ( t )$ (in our case a downward frequency-modulated sweep from 90 kHz to 30 kHz) and receives the echoes with two ears. During emission, the call is filtered by the directivity of the emitter transducer, modeled as an impulse response $h _ { e } ( t , \psi )$ for every direction ψ. Every reflector adds its own filtering $h _ { r , n } ( t )$ , and upon reception, the echo is filtered by the directivity of the left and right ear, $h _ { L } ( t , \psi )$ and $h _ { R } ( t , \psi )$ . For N reflectors, the signal received by the left ear can then be written as follows:

$$
s _ { L } ( t ) = \sum _ { n = 1 } ^ { N } a _ { n } \cdot h _ { L } ( t , \psi _ { n } ) * h _ { r , n } ( t ) * h _ { e } ( t , \psi _ { n } )\tag{1}
$$

with ∗ denoting the time-domain convolution, $\psi _ { n }$ the direction of the n-th reflector, $\tau _ { n } = 2 r _ { n } / c$ its round-trip delay (with $r _ { n }$ its distance and $c = 3 4 3 \mathrm { m s ^ { - 1 } } ) , a _ { n }$ the attenuation due to spherical spreading, and $w _ { L } ( t )$ additive sensor noise. The right ear follows by replacing L with R in these equations. The reflector filtering $h _ { r , n } ( t )$ also includes the frequency-dependent atmospheric absorption along the path [22]. In the frequency domain, the combined filtering of emitter and ear becomes a product, the Echolocation-Related Transfer Function (ERTF):

$$
\begin{array} { r } { H _ { L } ^ { E } ( f , \psi ) = H _ { e } ( f , \psi ) \cdot H _ { L } ( f , \psi ) , } \\ { H _ { R } ^ { E } ( f , \psi ) = H _ { e } ( f , \psi ) \cdot H _ { R } ( f , \psi ) , } \end{array}\tag{2}
$$

where $H _ { e } , \ H _ { L }$ and $H _ { R }$ are the Fourier transforms of the impulse responses above, i.e., the directivity of the emitter and the Head-Related Transfer Functions (HRTFs) of both ears.

## B. Cochlear processing

Following the original BatSLAM [7] and the spectrogram correlation and transformation receiver [23], we model the cochlea as a bank of bandpass filters followed by an envelope detector. Before that, we compress every echo into a short pulse with a matched filter, i.e., a correlation with the emitted call, which improves the range resolution. For frequency channel k of the left and the right ear, this yields:

$$
\begin{array} { r l r } & { } & { x _ { L , k } ( t ) = s _ { L } ( t ) * s _ { e } ( - t ) * g _ { k } ( t ) , } \\ & { } & { x _ { R , k } ( t ) = s _ { R } ( t ) * s _ { e } ( - t ) * g _ { k } ( t ) , } \end{array}\tag{3}
$$

where $g _ { k } ( t )$ is a bandpass filter with a Gaussian frequency response on a logarithmic frequency axis, centered at $f _ { k }$ and with a width of two channel spacings. We use $K \ = \ 6 4$ channels, logarithmically spaced between 30 kHz and 90 kHz. The envelope of every channel is the magnitude of its analytic signal, obtained with the Hilbert transform H [24]:

$$
\begin{array} { r } { e _ { L , k } ( t ) = \sqrt { x _ { L , k } ( t ) ^ { 2 } + \mathcal { H } \big ( x _ { L , k } ( t ) \big ) ^ { 2 } } , } \\ { e _ { R , k } ( t ) = \sqrt { x _ { R , k } ( t ) ^ { 2 } + \mathcal { H } \big ( x _ { R , k } ( t ) \big ) ^ { 2 } } . } \end{array}\tag{4}
$$

In practice, we compute the filtering and the Hilbert transform in one step, by keeping only the positive frequencies of each channel in the frequency domain. As every channel is narrow, its envelope can also be computed from a short inverse Fourier transform of only the frequency bins that the channel occupies, which yields the envelope at a decimated rate and reduces the cost of the front-end by a factor of about 15 (from roughly 300 ms to 25 ms of compute time per pulse in our Python implementation, at a correlation of 0.999 with the full-rate computation).

![](images/76c41c2db59daeb0e87c6688c4d366ab986bae9a392581d74c53399e3c363b19.jpg)  
Fig. 1. Overview of the processing flow of BatSLAM 2.0. For every pulse, the binaural echo is converted into a local view V with an energy image E and a spectral-shape image S (the direction cue), which is compared with all stored templates. The five best matches are candidates for the sequence verifier, which keeps several recognition hypotheses alive and commits to one only when it is long, strong, unambiguous and plausible given the current pose uncertainty. A committed hypothesis injects weak loop closure links into the pose graph, which is solved incrementally with iSAM2. The links of every hypothesis form a group that the link management can withdraw from the graph or re-admit later. New templates are anchored to pose-graph nodes, so that their poses follow every optimization, and no templates are laid while the robot is locked onto known territory.

Next, the envelopes are converted to range $r = c \cdot t / 2$ , and their power is averaged in range bins of 6 cm between 0.25 m and 10 m (162 bins), and over pairs of neighboring channels (32 channels). We write the result as $P _ { L } [ k , b ]$ and $P _ { R } [ k , b ]$ with k the channel and b the range bin.

Time-varying gain: Echoes from far away are weaker than echoes from nearby objects by tens of decibels (figure 2, panel b)). Bats compensate for this partially, by reducing their hearing sensitivity right after each call and restoring it over time. We apply a similar time-varying gain (TVG), which multiplies the amplitude of every range bin with its range $r _ { b } .$ , compensating one of the two spreading factors:

$$
\tilde { P } _ { L } [ k , b ] = r _ { b } ^ { 2 } \cdot P _ { L } [ k , b ] , \qquad \tilde { P } _ { R } [ k , b ] = r _ { b } ^ { 2 } \cdot P _ { R } [ k , b ] .\tag{5}
$$

## C. The local view

After these initial processing steps, we construct the local view which consists of two images per ear, each of $3 2 \times 1 6 2$ pixels (frequency channel, range bin). The first one, which we call the energy image, is the echo amplitude relative to the strongest bin of the view, compressed with a cube root:

$$
\begin{array} { r l r } & { } & { E _ { L } [ k , b ] = \left( \frac { \tilde { P } _ { L } [ k , b ] } { P _ { \mathrm { m a x } } } \right) ^ { 1 / 6 } , } \\ & { } & \\ & { } & { E _ { R } [ k , b ] = \left( \frac { \tilde { P } _ { R } [ k , b ] } { P _ { \mathrm { m a x } } } \right) ^ { 1 / 6 } , } \end{array}\tag{6}
$$

where $P _ { \mathrm { m a x } }$ is the maximum of $\tilde { P }$ over both ears, all channels and all range bins, and the power $1 / 6$ corresponds to the cube root of the amplitude (which as an additional square root term). In contrast to the logarithmic compression of the original BatSLAM, this compression has no floor parameter that needs to be adapted to the noise level of the sensor.

The second image, which we call the spectral-shape image, captures the direction cue of the echoes, which is dominated by the spectral content of the received echo. As shown in equation 1, every echo is filtered by the ERTF of the direction it arrives from, which gives it a characteristic spectrum. To separate this spectral shape from the level of the echo, we subtract, per range bin, the mean level over all channels:

$$
\begin{array} { r } { S _ { L } [ k , b ] = \frac { D _ { L } [ k , b ] - \bar { D } _ { L } [ b ] } { 1 5 \mathrm { d B } } , } \\ { S _ { R } [ k , b ] = \frac { D _ { R } [ k , b ] - \bar { D } _ { R } [ b ] } { 1 5 \mathrm { d B } } , } \end{array}\tag{7}
$$

where $D _ { L } [ k , b ] = 1 0 \log _ { 1 0 } \left( \tilde { P } _ { L } [ k , b ] / P _ { \operatorname* { m a x } } \right)$ is the level of a bin in decibels (and likewise $D _ { R } ) .$ , and $\bar { D } _ { L } [ b ]$ and $\bar { D } _ { R } [ b ]$ are the means over the 32 channels. We clip both images to the interval $[ - 1 , 1 ]$ , and set it to zero in bins that are more than 40 dB below $P _ { \mathrm { m a x } }$ , as these contain no echo energy. Both images are smoothed along range with a Gaussian kernel of one bin (6 cm), so that a template tolerates a few centimeters of displacement.

Finally, the energy images of both ears are concatenated into $\begin{array} { r } { E \ = \ \left( E _ { L } | E _ { R } \right) } \end{array}$ , and the spectral-shape images into $S = \left( S _ { L } | S _ { R } \right)$ , with | denoting concatenation. Each of both is normalized to zero mean and unit energy, and together they form the local view:

$$
V = \left( \frac { E - \mu _ { E } } { \left\| E - \mu _ { E } \right\| } \middle | \frac { S - \mu _ { S } } { \left\| S - \mu _ { S } \right\| } \right) ,\tag{8}
$$

with $\mu _ { E }$ and $\mu _ { S }$ the means of both images. This gives both images the same weight, independently of their dynamic range. The local view V has 20,736 elements (83 kB in single precision).

Figure 3 shows the local views of five places, together with the scene around the robot. Panels a) and b) show the same place in a corridor of the indoor floor, on two visits that are 641 m of travel apart. Both local views are nearly identical, with a similarity of 0.89 (equation 9 below). Panel c) shows the view of another place that is most similar to b), among all earlier pulses of the drive. At first sight, it looks alike, as both views are dominated by the echoes of the walls of a junction, but the echoes arrive at different ranges, and their spectral shapes differ, which yields a similarity of only 0.69. Panels d) and e) show a place in the pillar hall, where the local view consists of the isolated echoes of individual pillars, and a place in a street of the city world.

## IV. RECOGNIZING PLACES

In this section, we describe how BatSLAM 2.0 recognizes a place that the robot has visited before. We first describe the local view templates in more detail and how they are compared with the stored local view templates. After this description, we introduce the sequence verifier which turns individual, ambiguous matches into verified loop closures.

## A. Pose-anchored local view templates

A local view template is a local view $V _ { t }$ that is stored together with the index of the pose-graph node at which it was recorded (its anchor). The pose of a template is therefore always the current estimate $( x , y , \theta )$ of its anchor node, and it moves with every optimization of the full pose graph. As sonar views are strongly direction dependent (section VII-B), the heading needs to be part of the template’s pose. A new template is laid whenever the robot has traveled 0.45 m or turned $2 0 ^ { \circ }$ since the previous one, to overcome the buildup of templates/poses that are all geometrically located at the same place.

Similarity: We compare the current local view $V _ { q }$ with a template $V _ { t }$ using the correlation coefficient $\rho ,$ as in our earlier work on RadarSLAM [25], as we found that this was a more robust metric compared to the euclidean distance in the original BatSLAM work. As a revisit never passes through exactly the same position, we shift the range axis of $V _ { q }$ by up to three range bins (±18 cm) during comparison, and keep the best match:

$$
s ( q , t ) = \operatorname* { m a x } _ { | \delta | \leq 3 } \rho \bigl ( V _ { q } ^ { ( \delta ) } , V _ { t } \bigr ) ,\tag{9}
$$

with $s ( \boldsymbol { q } , t )$ the similarity between a query $q$ (the current pulse) and a template t, and $\bar { V _ { q } ^ { ( \delta ) } }$ the local view of the query, shifted by $\delta$ range binss $( q , t )$ . The winning shift $\delta ^ { * }$ also indicates where the robot is with respect to the anchor, and we use it as a forward offset $d ^ { * } = \delta ^ { * }$ · 6 cm in the loop closure links.

Candidates: For every pulse, the five most similar templates become candidates, provided that $s ( q , t ) ~ \geq ~ 0 . 6 0$ , that the template was laid more than 6 m of travel earlier, and that the headings of the robot and the anchor differ by less than $6 9 ^ { \circ }$ . No position gate is applied, i.e., the search over the templates is global.

## B. The sequence verifier

A single match in the sonar data is never trusted, as sonar data can be highly ambiguous. Instead, we keep a set of recognition hypotheses, and each of these hypotheses claims that the stretch of path that is being driven now was driven before. A hypothesis h is a chain of n pairs $( q _ { i } , t _ { i } )$ of query nodes and templates, in which every pair states that the anchor of $t _ { i }$ lies $d ^ { * }$ in front of $q _ { i } ,$ with the same heading.

Geometric consistency: A new pair can only extend a hypothesis if the motion between the queries agrees with the motion between the anchors. For two pairs a and $b ,$ let $\Delta p _ { q }$ and $\Delta \theta _ { q }$ be the displacement and heading change between both queries according to odometry, and $\Delta p _ { t }$ and $\Delta \theta _ { t }$ the same quantities between both anchors according to the map. Pair b is consistent with pair a when:

$$
\begin{array} { r l } & { \| \Delta p _ { t } - \Delta p _ { q } \| < 0 . 3 0 \mathrm { m } + 0 . 1 5 \cdot \| \Delta p _ { q } \| , } \\ & { ~ | \Delta \theta _ { t } - \Delta \theta _ { q } | < 0 . 2 5 \mathrm { r a d } , } \end{array}\tag{10}
$$

in which the position tolerance grows with the distance traveled, as the position error of odometry is unbounded and grows with every meter driven, both directly and through the accumulated heading error. The heading error, in contrast, stays well below 0.25 rad over the length of a hypothesis (a few meters), so that a fixed heading tolerance suffices. We test every new pair against both the last and the first pair of the hypothesis. Every pulse, each hypothesis is extended with its best consistent candidate, and every unused candidate seeds a new hypothesis. Several contradicting hypotheses can therefore be alive at the same time, and this set of hypotheses represents the ambiguity.

Evidence: Every hypothesis accumulates evidence as a sum of log-likelihood ratios of its similarities, minus a penalty for every pulse without a consistent candidate:

$$
\Lambda ( h ) = \sum _ { i = 1 } ^ { n } \ell \big ( s ( q _ { i } , t _ { i } ) \big ) - 0 . 5 \cdot m ,\tag{11}
$$

with m the number of missed pulses. The term $\ell ( s )$ is the evidence that a single pair with similarity s provides for a true revisit. Ideally, it is the log-likelihood ratio, which by Bayes rule equals the posterior minus the prior log-odds of a correct candidate:

$$
\begin{array} { l } { \displaystyle \ell ( s ) = \ln \frac { p ( s \mid \mathrm { c o r r e c t } ) } { p ( s \mid \mathrm { w r o n g } ) } } \\ { \displaystyle = \log \mathrm { i t } P ( \mathrm { c o r r e c t } \mid s ) - \log \mathrm { i t } P ( \mathrm { c o r r e c t } ) , } \end{array}\tag{12}
$$

with $p ( s \mid$ correct) and $p ( s \mid$ wrong) the distributions of the similarity of candidates whose anchor does and does not lie at the current place of the robot (within 0.6 m and $2 6 ^ { \circ }$ according to ground truth), and P(correct) the fraction of correct candidates. A positive ℓ therefore favors a revisit, and a negative ℓ favors a different place. We fitted P(correct | s) once, with a logistic regression logit P(correct $( ~ s ) = a \cdot s + b$ on the labeled candidates of the development drive. The loglikelihood ratio is then linear in $s ,$ and we refer to it, clipped, as the evidence curve:

$$
\ell ( s ) = \operatorname* { m i n } \Big ( 4 , \operatorname* { m a x } \big ( - 4 , ~ 4 0 \cdot ( s - 0 . 7 0 ) \big ) \Big ) .\tag{13}
$$

Here, the slope 40 is the fitted a, i.e., the evidence gained per unit of similarity, and $0 . 7 0 = ( \log \mathrm { i t } P ( \mathrm { c o r r e c t } ) - b ) / a$ is the similarity at which a candidate is as likely to be correct as any other candidate, so that it carries no evidence $( \ell = 0 )$ The clipping to [−4, 4], reached at $s = 0 . 7 0 \pm 4 / 4 0$ , prevents a single exceptionally good or bad match from dominating a hypothesis, as the linear model extrapolates without bound into the tails, where few candidates exist. A hypothesis dies after four consecutive missed pulses.

<sub>a)</sub> Received Binaural Echo (Normalized)  
![](images/750589f6046de594c815a863c90244741b1c6ce4e5cc310615521f00693b43bd.jpg)

<sub>c)</sub> Fine Cochleogram (64 Bands, 3 cm)  
![](images/000b28fda00989940b2d058e64dd277bb2a417cd4f7f4c04cd19a5d887e90e01.jpg)  
<sub>e)</sub> Energy Image E (32 Bands, 6 cm, $A ^ { 1 / 3 } )$

<sub>b)</sub> Matched-Filter Output (Envelope)  
![](images/383d4a439fea1ce61f2619f7b52ed6d580584e7805f3116765ab79b55ddd2965.jpg)  
<sub>d)</sub> After Time-Varying Gain (Amplitude ×r)

![](images/b49c33603058fbe75e687cc29f9fdfe648a7e79883bd4637604c63cbce2fd7f8.jpg)

![](images/73aad09be822428ddb33847d4adefc0ee67faa53fa626d66b3c398d81064fbba.jpg)  
<sub>f)</sub> Spectral-Shape Image S (Direction Cue)

![](images/376d02c79cd55acb5bbeb121af523c019e75e4189ca069f2626678a7ac93508f.jpg)  
Fig. 2. Processing steps of the acoustic front-end, illustrated on pulse 2600 of the development drive on the indoor floor. Panel a) shows the received binaural echo train, normalized per ear, as a function of range. Panel b) shows the output of the matched filter for the left ear in decibels, which reveals weak echoes up to 10 m that are invisible in the raw signal. Panel c) shows the fine cochleogram (64 bands between 30 kHz and 90 kHz, 3 cm range bins) as linear amplitude, for both ears (bands low to high within each ear), and panel d) shows the same cochleogram after the time-varying gain. Panel e) shows the energy image E of the descriptor (32 pooled bands, 6 cm range bins, cube-root compression), and panel f) the spectral-shape image S (red: bands above the mean level of the range bin, blue: below), which carries the direction-dependent filtering of the ears and the emitter. The speckle in panel f) beyond 5 m is noise that passes the level mask.

Commit rule: A hypothesis is committed (i.e., turned into loop closure links) when it has at least 8 pairs on at least 3 templates, an evidence $\Lambda \geq 1 0$ , and a margin of at least 4 over the best rival hypothesis that places the robot elsewhere (more than 1 m or 0.4 rad away). Furthermore, the correction it implies for the map must be plausible, which is estimated by a plausibility gate.

Plausibility gate: A hypothesis can be consistent and still be wrong, e.g., when a corridor has the same structure as another corridor a few meters away. Committing such an alias would force a jump of the current pose that is much larger than the drift the odometry could have accumulated. We therefore test whether the correction implied by a hypothesis can be explained by the uncertainty of the current pose estimate. From its last pair, the hypothesis predicts the current pose $x _ { h }$ , which we compare with the current estimate $\hat { x }$ of the pose graph through their Mahalanobis distance:

$$
d ( h ) = ( x _ { h } - \hat { x } ) ^ { T } \cdot \Sigma ^ { - 1 } \cdot ( x _ { h } - \hat { x } ) \leq 1 6 . 2 7 ,\tag{14}
$$

where Σ describes how far $x _ { h }$ and $\hat { x }$ may disagree when the hypothesis is correct. Two independent errors contribute: the drift of the estimate xˆ, given by its covariance in the pose graph, and the error of $x _ { h } .$ , which stems from a single sonar match and therefore has the covariance of one loop closure link. As both errors are independent, their covariances add. For a correct hypothesis, $d ( h )$ then follows a $\chi ^ { 2 }$ distribution with three degrees of freedom $( x , \ y , \ \theta )$ , and 16.27 is its 99.9th percentile: a larger $d ( h )$ means that the jump cannot be explained by drift and match error, and the hypothesis is rejected. For small corrections, we use the marginal covariance of the current node, which iSAM2 provides cheaply. This marginal, however, includes the drift that the current node shares with the anchor. For corrections larger than 1.5 m, we therefore use the covariance of the relative pose between the anchor and the current node, obtained from their joint marginal. Once loops have been closed, this relative covariance is much smaller, and corrections of several meters become implausible.

![](images/b7759d9863b9da49165b546f89a1f7d904e38fb4cd32bd0d99d9ebd269b62190.jpg)  
Fig. 3. Snapshots of five places and their local views. The top row shows the scene within 5 m of the robot (circle, with an arrow for its heading), with the frontal half-space of the sonar shaded in blue and the complete route in light grey. The middle row shows the energy image E and the bottom row the spectral-shape image S of the local view (red: above the mean level of the range bin, blue: below), each with the left ear above the right ear. Panels a) and b) show the same place in a corridor of the indoor floor, visited 641 m of travel apart; panel c) shows the view of another place, 7.7 m away, that is most similar to b) among all earlier pulses of the drive; panel d) shows the pillar hall of the indoor floor, and panel e) a street in the city. The similarity s is 0.89 between a) and b), and 0.69 between b) and c).

Risk-scaled commit: The damage of a wrong commit grows with the correction it forces on the map. When the implied correction exceeds 1.5 m, we therefore require at least 16 pairs (about 2.4 m of travel), an evidence $\Lambda \geq 4 0$ and a mean evidence of at least one per pair $( \Lambda \geq n )$ . Brief aliases die before they reach this length and strength, and the last condition stops the rare persistent alias along quasi-periodic structures (such as a lane between two rows of pillars), which stays consistent for many meters but matches only weakly (section VII-D).

Lock-on: Once a hypothesis is committed, all its pairs are released at once as loop closure links, which enter the pose graph as one link group (section V). Rival hypotheses that agree with it are discarded, as they describe the same revisit. The committed hypothesis itself stays alive, and the verifier keeps extending it with the same consistency test (equation 10). Every new pair is released immediately as an additional link of the same group, without passing the commit rule again: the pair continues a chain that has already been verified, and its consistency with both the first and the last pair ties it to that chain. At every pulse at which a committed hypothesis

Algorithm 1 Processing of one pulse in BatSLAM 2.0.

1: Add a node with odometry to the pose graph

2: Compute the local view from the echo

4: Commit a strong, unambiguous and plausible hypothesis

is extended, we say that the robot is locked on to a known stretch of path. Lock-on thus produces a dense sequence of links along the whole revisit, rather than a single burst at the moment of commitment, so that the drift is corrected along the entire revisited stretch. While locked on, no new templates are laid, as the robot drives through a place that is already represented by templates, and new templates of the same place would only compete with the existing ones as candidates at later revisits. Lock-on ends when the committed hypothesis dies, i.e., when the robot leaves the known path and no longer finds consistent candidates, or when the link management of the back-end withdraws its link group (section V). From then on, templates are laid again. The complete processing of a pulse is summarized in algorithm 1.

## V. THE ROBUST POSE GRAPH

In this section, we describe how odometry and verified recognitions are fused into a one consistent topological map.

In contrast to the pose cells and experience map of the original BatSLAM, a pose graph gives a principled estimate of every pose and its uncertainty, and it allows loop closures to be added and removed at any time, as was described in the previous section. In this section, we will detail the exact implementation of the pose graph subsystem with the GTSAM backend framework.

## A. Formulation

Every pulse i adds a node with the pose $x _ { i } = ( x , y , \theta )$ of the robot, connected to the previous node by the odometry increment $u _ { i }$ . A committed recognition adds a link between a query node $q$ and the anchor a of the matched template, with the measurement $z _ { q a } = ( d ^ { * } , 0 , 0 )$ . The map is then the solution of:

$$
\begin{array} { r l } & { \hat { X } = \arg \underset { X } { \operatorname* { m i n } } \sum _ { i } \left\| \Delta ( x _ { i - 1 } , x _ { i } ) - u _ { i } \right\| _ { \Sigma _ { o , i } } ^ { 2 } } \\ & { \quad \quad \quad \quad + \sum _ { ( q , a ) \in \mathcal { L } } \rho _ { H } \Big ( \left\| \Delta ( x _ { q } , x _ { a } ) - z _ { q a } \right\| _ { \Sigma _ { \ell } } \Big ) , } \end{array}\tag{15}
$$

where $\Delta ( x _ { a } , x _ { b } )$ is the pose of node b in the frame of node $^ { a , }$ $\| \cdot \| _ { \Sigma }$ is the Mahalanobis norm, $\mathcal { L }$ is the set of loop closure links currently in the graph, and $\rho _ { H }$ is the Huber loss. The odometry uncertainty $\Sigma _ { o , i }$ grows with the square root of the distance travelled (appendix A). The links are deliberately weak $( \sigma = 0 . 3 0 \mathrm { m }$ in position and 0.20 rad in heading), so that a single link can only nudge the map, whereas only a verified sequence of links (originating from the sequence verifier) closes a loop. We solve equation 15 incrementally with iSAM2 [20] in GTSAM [21], using Dogleg steps, as Gauss-Newton steps diverged in our system when many conflicting links were present.

## B. Link management

To maximally build safeguards into the system against map collapses, we assume that even a well-verified recognition can still be wrong. Therefore, we included a system to allow the pose graph to be able to forget these erroneous links. We therefore keep the links of every committed hypothesis together as a link group, which enters the graph as tentative, and can be removed from iSAM2 and re-admitted later. Three rules act on these groups:

• Length rule: a group whose hypothesis ends with fewer than 16 links is withdrawn permanently. This rule targets the same brief aliases as the risk-scaled commit, but for hypotheses with a small implied correction.

• Residual check: after every injection of links, the tentative group with the worst median Mahalanobis residual is removed if that median exceeds 12.

• GNC audit: every 400 nodes, we solve equation 15 in batch with Graduated Non-Convexity (GNC) [19], over the odometry and the links of all groups, including withdrawn ones. A group is kept (or re-admitted) when the mean GNC weight of its links is at least 0.5, and a group that survives two audits is confirmed.

The first two rules judge every group by itself, whereas the GNC audit operates globally on the graph, judging all groups jointly, similar to the RRR method [17] or the switchable constraints [14], but per group rather than per link. As we will show in section VII-D, the audit actually never changed a final map in our experiments, making it largely reduntant at this point. However, we opted to keep this mechanism present as a safeguard for more strongly aliased environments.

## VI. SIMULATION SETUP

In this section, we describe the approach used to simulate the sensor system, the worlds, the odometry model and the metrics used to evaluate BatSLAM 2.0.

## A. Sonar simulator

We simulated the binaural sonar by implementing equation 1 directly. The emitted call is a 3 ms linear FM downsweep from 90 kHz to 30 kHz, sampled at 500 kHz. For every reflector and ear, the simulator combines the directivity of the emitter and the ear (taken from the simulated HRTF of Phyllostomus discolor [26], shown in figure 4), spherical spreading, atmospheric absorption [22] and the reflectivity of the reflector. Walls and cylinders reflect specularly, and point reflectors isotropically. White Gaussian noise is added to both ears, and we express its standard deviation relative to the peak amplitude of the emitted call. Unless stated otherwise, we use the design sensor, with a noise level of −116 dB and a listening window of 10 m (section VII-A).

## B. Worlds and drives

We used three simulated worlds (figure 5). The main world is an indoor floor of 38 m by 18 m, consisting of a grid of rooms separated by corridors of 1.4 m to 1.8 m wide, and a hall with 40 pillars. The walls carry 170 small reflectors (e.g., pipes and radiators), and to make aliasing explicit, we placed an identical pattern of three objects on two different corridor walls. The second world is a city of 46 m by 26 m with the same topology but with streets of 3 m to 4 m, and the third is a small hall used during development.

In the indoor floor and the city, the robot drives a random tour of about 1 km, in which every corridor is driven at least twice in both directions, with a small lateral along the corridor wobble so that revisits are never exactly identical in position. A pulse is emitted every 0.15 m of travel, yielding about 6400 pulses per drive. We developed and tuned the system on one route of the indoor floor, and used other random routes as unseen test cases.

## C. Odometry model

The odometry was generated from the ground-truth increments with random noise (3 % forward, 1 % lateral and $1 ^ { \circ }$ per $\sqrt { \mathrm { m } }$ in heading) and a systematic error (a heading bias of $0 . 6 ^ { \circ } \mathrm { m } ^ { - 1 }$ and a scale error of 2 %). We scaled the systematic part with a bias factor b: $b = 1$ represents uncalibrated wheel odometry, $b = 0 . 2 5$ a roughly calibrated robot and $b = 0$ a calibrated robot. Over 1 km, dead reckoning drifts by 4– 36 m, depending on b and the random seed. Typical robots in indoor environments can be assumed to have decently calibrated odometry, whereas in outdoor environments these calibrations are less accurate due to a higher prevalence of wheel slippage.

![](images/370717d7fdc8f86378463e0c1425345ae3319df547ec8a2a597f36db4277b9b2.jpg)  
Fig. 4. Directivity of the simulated sonar (Lambert azimuthal equal-area projection of the frontal hemisphere, seen from behind the head; grid lines every 30<sup>◦</sup>). Rows: emitter, left ear, right ear and the ERTFs of both ears, at five frequencies between 30 kHz and 90 kHz, each normalized to its maximum.

## D. Metrics

As BatSLAM 2.0 builds a topological map, our primary criterion is the absence of wrong loop closures and map collapse rather than metric accuracy. The link precision is the fraction of links for which query and anchor are truly within 0.6 m and 26<sup>◦</sup> of each other. The number of wrong commits counts committed hypotheses of which the majority of links are wrong. The revisit coverage is the fraction of true same-direction revisits that received a correct link. The collapsed pairs are the fraction of node pairs that are at least 4 m apart in reality, but less than half that distance apart in the map. Finally, we report the absolute trajectory error (ATE) without alignment and after a rigid alignment in the plane.

The pose-graph part of BatSLAM 2.0 is implemented in Python using GTSAM 4.2 with Python bindings, and the simulator is implemented using custom MATLAB code. All experiments ran on a desktop computer with an AMD Ryzen 9 7900X.

## VII. RESULTS

In this section, we present the results in the order of the processing chain. We first analyze the recognition of a single sonar view, and how far a robot may deviate from a template for it still to be recognized. Next, we evaluate the complete system, followed by ablations, a robustness analysis and the computational cost.

## A. Sonar Template Recognition

To evaluate the front-end on its own, we tested how well a single local view recognizes the place where it was taken, without the sequence verifier. For every third pulse, we compared its local view with those of all earlier pulses at least 6 m of travel back, using the similarity of equation 9. Only pulses with a true revisit, i.e., an earlier pulse within 0.6 m and 26<sup>◦</sup>, serve as queries, so that only same-direction revisits are counted. We report two metrics. The top-1 accuracy is the fraction of queries for which the most similar earlier view lies within 1 m and 30 of the query. The AUC is the area under the ROC curve that separates same-place pairs (within 0.6 m and 26<sup>◦</sup>) from different-place pairs (more than 3 m or 57<sup>◦</sup> apart), i.e., the probability that a random same-place pair is more similar than a random different-place pair. We estimate it from 20 000 random pairs of each class, so that differences in AUC below about 0.005 lie within its sampling noise. Figure 6 reports both metrics for different sensors, and table II in the appendix for the design choices of the descriptor.

![](images/b0ebffe73e92d3e9883158288c006c4705614189721e441e09de58b6679ed301.jpg)

![](images/8521766c6a27c1abbfd900ff6472c4dbb13c078d765773eb5305a415f5d748c7.jpg)  
Fig. 5. The simulated worlds and exemplary results of BatSLAM 2.0. Top row: the ground-truth drive (colored from start to end) in a) the indoor floor (38 m by 18 m, development drive on route 3, 964 m), b) the city (46 m by 26 m) and c) the small development hall (16 m by 10 m). Black crosses: the two copies of the aliasing pattern. Bottom row, d)–f): the final map estimated by BatSLAM 2.0 for the same drives with calibrated odometry (blue), with its trajector error (ATE); wrong links in the final graph are drawn in red.

Sensor quality: Figure 6 compares four sensors on the same drive through the city map, combining two maximal ranges (5 m and 10 m) with two noise levels (−96 dB and −116 dB, i.e., the standard deviation of the noise relative to the peak amplitude of the emitted call). For every sensor, we evaluated three descriptors: the descriptor of the original BatSLAM (8 frequency channels, 3 cm range bins and a logarithmic compression with a floor at −30 dB), the same with the timevarying gain (TVG), and our final descriptor (section III). The noise level clearly dominates. At a noise level of −96 dB, no descriptor reaches a top-1 accuracy of 0.5, and a longer range does not help, as the echoes beyond 5 m are buried in noise. The TVG then even hurts, as it amplifies this noise, and the final descriptor offers no clear advantage over the original one. With 20 dB less noise (−116 dB), every descriptor improves substantially, and both the TVG and the final descriptor pay off. Only then does a longer range pay off as well, up to a top-1 accuracy of 0.89 for the final descriptor at 10 m. This sensor (10 m, −116 dB) is the design sensor that we use in the remainder of this paper. To test the robustness, we also use three degraded sensors: the design sensor with 10 dB and 20 dB more noise (the −10 dB and −20 dB sensors, at −106 dB and −96 dB), and the sensor of the original configuration (5 m, −96 dB).

Descriptor design: Table II in the appendix compares the design choices of the descriptor for the design sensor on both worlds. All variants share the spatiospectral resolution (32 frequency channels, 6 cm range bins) and the TVG of our front-end, and differ only in the listed aspect. Adding the spectral-shape image to the energy image raises the top-1 accuracy on the indoor floor from 0.88 to 0.91, and the AUC on both worlds. Adding the interaural level difference (ILD) as a third image raises the AUC further, but lowers the top-1 accuracy on both worlds, as the ILD changes quickly with small pose changes, and we therefore left it out. The compression law matters little: with the spectral-shape image, every compressive law reaches a top-1 accuracy between 0.88 and 0.93 on the indoor floor and of 0.89 in the city, and only a linear amplitude is clearly worse. We use the cube root, as it performs on par with a logarithm without a floor that depends on the noise level (section III). The matching method, in contrast, has shown to matter a lot. With the matching of the original BatSLAM (the Euclidean distance between unit-energy views, without range shift), the top-1 accuracy of our descriptor drops to about 0.80 on both worlds, and that of a logarithmic energy image alone to 0.40 and 0.60. Both the new descriptor and the new matching thus contribute to the improvement over the original BatSLAM. Note that these last rows use the resolution and TVG of our front-end, and therefore differ from the original descriptor in figure 6.

## B. Capture region of a view template

To measure how far a revisit may deviate from a template, we rendered views at controlled offsets from 16 template poses along the development drive. An offset view counts as recognized when its similarity to the template exceeds 0.70 and no template of another place is more similar. Figure 7 shows that the capture region is narrow: about ±0.20 m along track, ±0.15 m laterally and $\pm 2 5 ^ { \circ }$ in heading.

Lateral Offset (m)  
![](images/1ed88c5a87597b4420d16afbb33f964adbe39bc0168b3dbe91ee241dde668b5a.jpg)

![](images/d4d08dbbc590eda9a07cdeb7744b9a67adcc14c5a4aaf76bfc688e275d4a9bf1.jpg)  
Fig. 6. Single-view place recognition for four sensors on the city drive: a) top-1 accuracy, b) AUC. Grey: descriptor of the original BatSLAM; orange: the same with TVG; blue: the final descriptor.

![](images/b89a2a6648eed2b679be5c7ef467308e839fc7f6298324cf11fefb927fcd9c5c.jpg)

b)  
![](images/1c698e957aad703f044169dc687830b868207483525010be68dd9741e7b5ff0d.jpg)  
Fig. 7. Capture region of local view templates on the indoor floor, averaged over 16 template poses: fraction of recognized offset views for combined a) along-track and heading offsets, and b) lateral and heading offsets.

## C. The complete system

We ran the complete system on three odometry seeds of the development drive, the two unseen routes and the city environment (which was unseen during development), each with calibrated odometry and with a residual heading bias, on three degraded sensors, and with uncalibrated odometry. Table III in the appendix summarizes the results per condition, figure 5d–f shows the maps of one run per world, figure 8 shows the links of one run, and table VI and figure 12 in the appendix list all runs.

With the design sensor and calibrated or roughly calibrated odometry, all 12 runs produced a consistent map: none of the 457 committed hypotheses was wrong, and no distant places were pulled together. The link precision was 0.994–1.000, and 0.79–0.84 of the true revisits were linked. The trajectory error was 0.47–1.91 m (0.18–0.53 m after alignment), against 4.2– 19.0 m for odometry alone.

The degraded sensors also produced consistent maps (0.25– 0.81 m after alignment), but on the −20 dB sensor, one wrong hypothesis of 17 links survived. It was committed early in the drive, when the map was still so uncertain that a correction of 8 m was plausible, and it distorts the map locally without collapsing it.

The difference between the aligned and unaligned errors is caused by the heading drift that accumulates before the first loop closure, which can only be removed by a later loop closure with that first stretch of the route. If there is none, the whole map remains rotated (by about $5 ^ { \circ }$ on route 7). The unaligned error therefore measures how the map is anchored, and the aligned error measures its shape. With uncalibrated odometry, the links remain almost all correct, but the metric accuracy degrades (section VII-E).

## D. Ablations

To assess the contribution of every component, we removed or replaced several components of the full system, one at a time, and evaluated all variants on four standard cases (tables IV and IX in the appendix). Without sequence verification, the system made 215 wrong commits, of which 10 survived the link management, and the mean trajectory error grew to 4.85 m. Without link management as well (the naive variant), the map collapsed completely (rendering ATE a useless metric). This confirms that a single sonar view is too ambiguous to be trusted to perform loop closure. The front-end matters has an equally large contribution to the performance of the system, as removing the time-varying gain or the spectral-shape image, wrong hypotheses survive often. Without the range-shift search, the coverage of recognition drops to 0.60. The original front-end and the hand-set evidence constants of our first implementation made no wrong commits, but linked fewer revisits (0.73 and 0.68).

![](images/0c212a55bd214d38767673cbd5b4717596038348ae4e57c2bc4aab6d4b04a67e.jpg)  
Fig. 8. Loop closure matrix of the development drive (route 3, calibrated odometry). Every dot is a link in the final graph between a query pulse and the anchor of the matched template. Grey: true same-direction revisits; blue: correct links; red: wrong links.

The safeguards against aliases, in contrast, hardly come into play on these cases, as they contain almost no aliases that survive the sequence verification. Removing the complete link management even lowered the error (0.55 m against 0.87 m), as the length rule also withdraws short correct groups. Requiring 24 pairs for a large correction, without a condition on the evidence, raised the error on one particular run to 2.98 m, as the true revisit of the start of the route was split into two hypotheses that never reached 24 pairs, which left the map rotated.

Safeguards against aliases: To study the safeguards, we repeated the ablations on the four hardest cases: the three degraded sensors and a development drive with a recurring alias (table VIII in the appendix). Without the risk-scaled commit, 12 of 346 commits were wrong. Of these, 8 implied a correction of more than 1.5 m (against only 13 % of the correct commits), and 11 ended with 16 links or fewer, as an alias ends where the similarity between both corridors does. These observations motivate the risk-scaled commit and the length rule. With all safeguards in place, 6 wrong hypotheses were committed, and the link management withdrew all but the early alias on the −20 dB sensor. Without link management, 7 survived, and removing only the length rule gave identical results: the residual check and GNC audits never changed a final map, as a single wrong hypothesis that passed the gates of the verifier cannot be detected by a joint consistency test. Without any plausibility gate, 3 wrong hypotheses survived. Requiring 24 pairs regardless of the evidence is the only variant without a surviving wrong hypothesis, but as shown above, it also withholds the loop closures that anchor the map (section IX).

## E. Robustness

Next, we varied the systematic odometry error with the bias factor $b ,$ tripled the random odometry noise, and combined the degraded sensors with a residual bias (figure ${ 9 } ;$ table VII in the appendix). Up to $b = 0 . 5 ~ ( 0 . 3 ^ { \circ } \mathrm { m } ^ { - 1 } )$ , every run produced a consistent map with the same coverage as with calibrated odometry, while the error of dead reckoning grew beyond 20 m. The same holds for three times the random noise (0.48 m after alignment). With uncalibrated odometry $( b = 1 )$ , the heading has drifted by about $4 5 ^ { \circ }$ at the first revisit. In three of six runs, later loop closures removed this rotation (1.4 m to 1.8 m). In the other three, the links were still correct, but as the pose graph treats odometry errors as random, the stretches between loop closures keep the curvature of the biased odometry, and the map remains distorted (9.3 m to 14.5 m). With $b = 2 .$ , fewer true revisits pass the consistency test and plausibility gate (coverage 0.32 to 0.41), and the map collapses about as much as dead reckoning does, but not a single wrong hypothesis remained. In summary, BatSLAM 2.0 needs odometry calibrated to within about $0 . 3 ^ { \circ } \mathrm { m } ^ { - 1 }$ , and beyond that, it degrades in metric accuracy rather than in topology.

## F. Computational cost

On a single core, a pulse took 69 ms on average, of which 41 ms went to the pose graph. As the template comparison and the pose graph grow with the map, our (unoptimized) Python implementation meets the 100 ms budget of a 10 Hz pulse rate on average, but not towards the end of a 1 km drive.

## VIII. REAL-WORLD EXPERIMENT

In this section, we apply BatSLAM 2.0 to a real recording. As no binaural bat-head sonar recording of a suitable drive was available, we used an embedded 3D sonar sensor (eRTIS [4]), and formed two virtual ears from its microphone array. Apart from the virtual ears and the frequency band, the system is identical to the one of the previous sections.

## A. Recording, virtual ears and ground truth

Recording: The eRTIS sensor was mounted on a mobile platform together with a 3D lidar (Ouster OS0) and a camera, and driven through a hall of 22 m by 8 m in a university building (figure 10). The drive is a figure-eight of 89 m, during which the sensor emitted 1186 calls at about 7.1 Hz. The call is a 2.5 ms FM downsweep from 50 kHz to 25 kHz, recorded by 32 MEMS microphones in a pseudo-random planar array of about 7 cm by 4 cm. We used the ranges between 0.6 m and 5.5 m.

Virtual ears: We split the array into a left and a right half of 16 microphones each, and steered each half with a delayand-sum beamformer towards an azimuth of $3 0 ^ { \circ }$ to its own side:

$$
s _ { L } ( t ) = \sum _ { i \in \mathcal { L } } w _ { i } x _ { i } \biggl ( t + \frac { y _ { i } \sin 3 0 ^ { \circ } } { c } \biggr ) ,
$$

$$
s _ { R } ( t ) = \sum _ { i \in \mathcal { R } } w _ { i } x _ { i } \biggl ( t - \frac { y _ { i } \sin 3 0 ^ { \circ } } { c } \biggr ) ,\tag{16}
$$

![](images/02987482f7592bf9be71e83f98706d976821c9e9078ab82c243eb340df61b49b.jpg)

![](images/7f0f59f4f4db30cbef1c62c3286960a56528b0c0fe38e7c5b17e96c6babe1d28.jpg)  
Fig. 9. Robustness against systematic odometry errors on the development drive. a) Trajectory error of odometry (orange), BatSLAM 2.0 (blue) and BatSLAM 2.0 after alignment (green) versus the bias factor b; b) wrong hypotheses in the final graph and percentage of collapsed pairs.

with $x _ { i } ( t )$ the signal of microphone i, y<sub>i</sub> its lateral position, L and R the two halves, and $w _ { i }$ a Gaussian taper. As the aperture of each half spans between two and a half and five wavelengths over the band of the call, the beams are wide at 25 kHz and narrow at 50 kHz, which gives the virtual ears a frequencydependent directivity, similar to the HRTF characteristics of the real bats used in simulation. As the call covers only one octave, we used 16 pooled channels between 25 kHz and 50 kHz.

Ground truth and odometry: The ground truth was computed from the lidar scans with PIN-SLAM [27], and the clock offset between sonar and lidar was estimated by comparing the acoustic images with images predicted from the lidar map. As the platform recorded no wheel odometry, we derived it from the ground truth with a scale error of −5 % on distance and 10 % on rotation, plus the random noise of the simulations. On a figure-eight, the rotation error does not cancel, and at the first revisit, the heading is off by 36<sup>◦</sup>. Dead reckoning has an error of 3.62 (0.46) m, and all results are the mean (standard deviation) over ten noise seeds.

## B. Results

The evidence curve: The real similarities s (equation 9) are lower than the simulated ones. The correct candidates had a median similarity of 0.65, and 99 % of the wrong ones stayed below 0.645 (figure 11b). Both classes are well separated, but the evidence curve of the simulations (equation 13, with its zero crossing at 0.70) lies above most correct matches. A correct pair with the median similarity then contributes $\ell = 4 0 \cdot ( 0 . 6 5 - 0 . 7 0 ) = - 2$ to the evidence $\Lambda ( h )$ of equation 11, so that a hypothesis of typical correct pairs loses evidence with every pulse instead of gaining it, and only stretches with similarities above 0.70 can reach the commit threshold. We therefore refitted the two constants of equation 13 with the same logistic regression as in the simulations (equation 12), on the candidates of one half of the drive (alternating stretches of 5 m), labeled with the lidar ground truth, and ran the complete drive with it. Both folds gave similar curves (a zero crossing at

0.576 and 0.546). This curve is a property of the sensor, and has to be determined once per sensor rather than per deployment.

The map: Table V in the appendix and figure 11a summarize the results. With either evidence curve, none of the 30 runs committed a wrong hypothesis or collapsed. With the curve of the simulations, the system linked only half of the revisits, and reduced the trajectory error from 3.62 (0.46) m to 0.77 (0.07) m. With the refitted curve, it linked 0.64 (0.10) of the revisits and reached 0.67 (0.13) m (0.52 (0.11) m after alignment). The remaining error stems mostly from the first 46 m of the drive, before the first same-direction revisit. Places in the right loop were only revisited in the opposite direction, and were, as expected, never linked. With the refitted curve, a few links (3.3 (2.9) per run on average) at the tail of one correct hypothesis were 0.6 m to 1.2 m off, at a place where the platform turned off a path that it had previously driven straight on. The long-baseline tolerance of the verifier allows such a slow divergence.

Odometry errors: With a rotation error of 15 % instead of 10 %, the heading error at the first revisit grows to $5 3 ^ { \circ }$ , which the plausibility gate considers implausible given the heading noise assumed by the back-end. The correct loop closures are then rejected (3.18 (0.21) m), but no wrong loop closure is committed either, and with twice the assumed heading noise, the system recovers (0.66 (0.07) m). Finally, the system runs in real time on this drive, with 4.8 ms for the front-end and 6.2 ms for the rest of the system per pulse on average, against 141 ms between two calls.

## IX. DISCUSSION AND LIMITATIONS

In contrast to SeqSLAM [13], temporal consistency alone is insufficient for sonar: a short segment of one corridor can genuinely produce an echo sequence that closely resembles a segment elsewhere. The main remaining vulnerability therefore occurs early in a drive, when the relative pose uncertainty between two places may still span several meters and a strong aliased match remains geometrically plausible. Postponing all large corrections would reduce this risk, but would also delay the loop closures needed to constrain the map. Our ablations expose this trade-off directly: requiring 24 supporting pairs eliminated the remaining alias for the −20 dB sensor, but left the map of route 7 rotated. For a purely topological representation, the stricter criterion is therefore preferable, as the global rotation of the map is not that important for practical applications. For metric SLAM, however, timely anchoring of the map is also important, and a tradeoff can be made here.

a) Hall of a University Building: Occupancy Grid and Trajectory from the 3D Lidar SLAM  
![](images/1e9c2cf3ceab4dc89753e0f7046f3a49806823e12682036b7d1a91a854fc2997.jpg)

b)  
A, Timestamp: 50 s  
![](images/6866de8cf6a60072720ba8d9c9ff52def2559e3600ec60fa67d634b8945c9736.jpg)  
A, Pulse 357 (25−50 kHz)

B, Timestamp: 73 s  
![](images/8caf508b922f9259a4daa1c7209eb116650c54dd285272c969202008e77b1cf5.jpg)

C, Timestamp: 114 s  
![](images/482c06fbdd4c0f5e338184951635693df45f9654e7eb44fed297a61ba1a7f821.jpg)  
C, Pulse 810 (25−50 kHz)

c)  
![](images/abbb32bd6e51bb1c6d889572f48d98e4517d2a8e0ea9f0dcbd4eddc9e60d6c01.jpg)  
Range (m)

B, Pulse 517 (25−50 kHz)  
![](images/0ed5ad3096aeb91d424b5c791194c89fa997f700478855b8034f00e7f3876e1a.jpg)  
Range (m)

![](images/720d270926a17584713fcd712d4c40d07ff90f47c0583e5e708ea2484fecfe3c.jpg)  
Range (m)  
Fig. 10. The real-world recording. a) Occupancy grid of the hall from the lidar scans, with the ground-truth trajectory (colored by time); the left loop was driven twice in the same direction, the right loop once in each direction. b) Camera images at places A, B and C. c) Binaural cochleograms of the closest sonar pulse, formed with the virtual ears of equation 16.

A second limitation is that most experiments were conducted in simulation. The simulator models sensor directivity, geometric spreading, atmospheric absorption, and reflectivity, but does not include weak higher-order reflections. The realworld experiment demonstrates that the approach transfers to an actual acoustic sensor, requiring just the evidence curve to be recalibrated.

The current pose graph assumes that odometry errors are predominantly random. Under a systematic heading bias, the inferred topology remains correct, but metric accuracy progressively deteriorates as expected. Introducing an explicit heading-bias state into the factor graph would provide a natural way to estimate and compensate for such systematic errors. Similarly, locations revisited only in the opposite direction are currently never associated. This limitation could be addressed by explicitly modeling how a place is expected to appear acoustically when traversed from the reverse direction, but requires complex learned world models, which is part of our future work.

Finally, although the simulated environments were deliberately designed to contain substantial self-similarity, the sequence verifier rejected nearly all incorrect hypotheses before they reached the back-end. Consequently, the GNC audit did not alter any final map in the experiments reported here. Its role is likely to become more important in environments with stronger structural aliasing, such as office floors containing repeated rooms or parking garages with highly repetitive geometry, where several individually plausible but mutually inconsistent loop-closure hypotheses may coexist. However, the target application for an algorithm like BatSLAM is in environments where optical techniques tend to fail, such as in heavy industry applications (mining, agriculture, etc). In these environments, the type of degenerative aliassing as encountered in long office corridors is typically not encountered, making this issue less pressing. In addition, all parameters were fixed using a single development trajectory, whereas the real sensor required different evidence constants. These constants primarily determine loop-closure coverage rather than map safety, but currently need to be calibrated once for each sensor configuration. Self-learning using active inference of these parameters might be an interesting avenue for further research into this architecture.

![](images/4f0997637f406de8bc1ada19ffed21f029dad98355a608a7cccaedcb017e5601.jpg)

![](images/cf0c6ed1947d07acbcd65a922f3e37a3762acfe5f6e2d66ca0de3d66eb1fe2ff.jpg)  
Fig. 11. Real-world results. a) Map of one run (refitted evidence curve) with odometry, ground truth, BatSLAM 2.0 estimate and links. b) Similarity of correct and wrong candidates, with the simulated (dashed) and refitted (solid) evidence curves.

## X. CONCLUSION

In this paper, we presented BatSLAM 2.0, a sonar-only SLAM system that combines a biomimetic acoustic front-end with sequence-verified place recognition and a robust posegraph back-end. The central challenge is the strong perceptual aliasing inherent to sonar. BatSLAM 2.0 addresses this ambiguity through an explicit direction cue in the place descriptor, a sequence verifier that accepts only long, strong, unambiguous, and geometrically plausible hypotheses, a commit rule that increases the required evidence with the magnitude of the implied correction, and a back-end in which each accepted loop closure remains represented as a removable group of weak constraints.

Across all 12 simulated runs with calibrated or approximately calibrated odometry, the system produced consistent maps without incorrect loop closures, achieving 0.18–0.53 m after alignment. When systematic odometry bias was increased beyond $0 . 3 ^ { \circ } \mathrm { m } ^ { - 1 }$ , performance degraded primarily in metric accuracy rather than in topological correctness (ie, no map collapses occured). Our analysis further showed that sensor signal-to-noise ratio and spectral-shape information are central to reliable place recognition, and that the effective capture region of a stored template is limited to approximately ±0.15 m laterally and $\pm 2 5 ^ { \circ }$ in heading. On a real acoustic recording, BatSLAM 2.0 likewise produced a consistent map without incorrect loop closures while operating in real time, reducing the trajectory error from 3.62 (0.46) m to 0.67 (0.13) m.

Future work will focus on substantially longer real-world trajectories using physical odometry and a bat-like binaural sensing head, explicit estimation of systematic odometry errors, and place recognition across opposite traversal directions.

## DATA AND CODE AVAILABILITY

The BatSLAM 2.0 source code, simulator, and scripts used to generate all experimental results, tables, and figures in this paper are available at https://github.com/Cosys-Lab/BatSLAM2.

## ACKNOWLEDGMENT

Claude Opus 5.5 (Anthropic) was used to assist with writing the code, generating the test harness, and drafting the text of this paper. The authors substantially revised the text and take full responsibility for the content of this paper.

## REFERENCES

[1] D. R. Griffin, Listening in the Dark: The Acoustic Orientation of Bats and Men. New Haven: Yale University Press, 1958.

[2] N. Ulanovsky and C. F. Moss, “What the bat’s voice tells the bat’s brain,” Proceedings of the National Academy of Sciences, vol. 105, no. 25, pp. 8491–8498, 2008.

[3] J. Steckel, A. Boen, and H. Peremans, “Broadband 3-D sonar system using a sparse array for indoor navigation,” IEEE Transactions on Robotics, vol. 29, no. 1, pp. 161–171, 2013.

[4] R. Kerstens, D. Laurijssen, and J. Steckel, “eRTIS: A fully embedded real time 3D imaging sonar sensor for robotic applications,” in Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), 2019, pp. 1438–1443.

[5] J. J. Leonard and H. F. Durrant-Whyte, “Mobile robot localization by tracking geometric beacons,” IEEE Transactions on Robotics and Automation, vol. 7, no. 3, pp. 376–382, 1991.

[6] J. D. Tardós, J. Neira, P. M. Newman, and J. J. Leonard, “Robust mapping and localization in indoor environments using sonar data,” The International Journal of Robotics Research, vol. 21, no. 4, pp. 311–330, 2002.

[7] J. Steckel and H. Peremans, “BatSLAM: Simultaneous localization and mapping using biomimetic sonar,” PLoS ONE, vol. 8, no. 1, p. e54076, 2013.

[8] M. J. Milford, G. F. Wyeth, and D. Prasser, “RatSLAM: A hippocampal model for simultaneous localization and mapping,” in Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), 2004, pp. 403–408.

[9] M. J. Milford and G. F. Wyeth, “Mapping a suburb with a single camera using a biologically inspired SLAM system,” IEEE Transactions on Robotics, vol. 24, no. 5, pp. 1038–1053, 2008.

[10] D. Vanderelst, J. Steckel, A. Boen, H. Peremans, and M. W. Holderied, “Place recognition using batlike sonar,” eLife, vol. 5, p. e14188, 2016.

[11] C. Cadena, L. Carlone, H. Carrillo, Y. Latif, D. Scaramuzza, J. Neira, I. Reid, and J. J. Leonard, “Past, present, and future of simultaneous localization and mapping: Toward the robust-perception age,” IEEE Transactions on Robotics, vol. 32, no. 6, pp. 1309–1332, 2016.

[12] S. Lowry, N. Sünderhauf, P. Newman, J. J. Leonard, D. Cox, P. Corke, and M. J. Milford, “Visual place recognition: A survey,” IEEE Transactions on Robotics, vol. 32, no. 1, pp. 1–19, 2016.

[13] M. J. Milford and G. F. Wyeth, “SeqSLAM: Visual route-based navigation for sunny summer days and stormy winter nights,” in Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), 2012, pp. 1643–1649.

[14] N. Sünderhauf and P. Protzel, “Switchable constraints for robust pose graph SLAM,” in Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2012, pp. 1879–1884.

[15] P. Agarwal, G. D. Tipaldi, L. Spinello, C. Stachniss, and W. Burgard, “Robust map optimization using dynamic covariance scaling,” in Proceed ings of the IEEE International Conference on Robotics and Automation (ICRA), 2013, pp. 62–69.

[16] E. Olson and P. Agarwal, “Inference on networks of mixtures for robust robot mapping,” The International Journal of Robotics Research, vol. 32, no. 7, pp. 826–840, 2013.

[17] Y. Latif, C. Cadena, and J. Neira, “Robust loop closing over time for pose graph SLAM,” The International Journal of Robotics Research, vol. 32, no. 14, pp. 1611–1626, 2013.

[18] J. G. Mangelson, D. Dominic, R. M. Eustice, and R. Vasudevan, “Pairwise consistent measurement set maximization for robust multi-robot map merging,” in Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), 2018, pp. 2916–2923.

[19] H. Yang, P. Antonante, V. Tzoumas, and L. Carlone, “Graduated nonconvexity for robust spatial perception: From non-minimal solvers to global outlier rejection,” IEEE Robotics and Automation Letters, vol. 5, no. 2, pp. 1127–1134, 2020.

[20] M. Kaess, H. Johannsson, R. Roberts, V. Ila, J. J. Leonard, and F. Dellaert, “iSAM2: Incremental smoothing and mapping using the Bayes tree,” The International Journal of Robotics Research, vol. 31, no. 2, pp. 216–235, 2012.

[21] F. Dellaert, “Factor graphs and GTSAM: A hands-on introduction,” Georgia Institute of Technology, Tech. Rep. GT-RIM-CP&R-2012-002, 2012.

[22] “Acoustics – attenuation of sound during propagation outdoors – part 1: Calculation of the absorption of sound by the atmosphere,” International Organization for Standardization, Tech. Rep. ISO 9613-1:1993, 1993.

[23] H. Peremans and J. Hallam, “The spectrogram correlation and transformation receiver, revisited,” The Journal of the Acoustical Society of America, vol. 104, no. 2, pp. 1101–1110, 1998.

[24] S. L. Marple, “Computing the discrete-time "analytic" signal via FFT,” IEEE Transactions on Signal Processing, vol. 47, no. 9, pp. 2600–2603, 1999.

[25] G. Schouten and J. Steckel, “A biomimetic radar system for autonomous navigation,” IEEE Transactions on Robotics, vol. 35, no. 3, pp. 539–548, 2019.

[26] F. De Mey, J. Reijniers, H. Peremans, M. Otani, and U. Firzlaff, “Simulated head related transfer function of the phyllostomid bat Phyllostomus discolor,” The Journal of the Acoustical Society ofAmerica, vol. 124, no. 4, pp. 2123–2132, 2008.

[27] Y. Pan, X. Zhong, L. Wiesmann, T. Posewsky, J. Behley, and C. Stachniss, “PIN-SLAM: LiDAR SLAM using a point-based implicit neural representation for achieving global map consistency,” IEEE Transactions on Robotics, vol. 40, pp. 4045–4064, 2024.

TABLE I  
PARAMETERS OF BATSLAM 2.0. THE SAME VALUES WERE USED FOR EVERY EXPERIMENT IN THIS PAPER.
<table><tr><td>Stage</td><td>Parameter</td><td>Value</td></tr><tr><td>Front-end</td><td>Filterbank</td><td>64 Gaussian bands, 30 kHz to 90 kHz, log-spaced, width of two band spacings</td></tr><tr><td></td><td>Band pooling</td><td>Pairs of bands (32 bands) 6 cm, from 0.25 m to 10 m (162 bins)</td></tr><tr><td></td><td>Range bins</td><td></td></tr><tr><td></td><td>Time-varying gain</td><td>Amplitude ×r</td></tr><tr><td></td><td>Energy image</td><td>Cube root of the amplitude, relative to the peak of the view</td></tr><tr><td></td><td>Spectral-shape image</td><td>±15 dB scaled to ±1, only above −40 dB</td></tr><tr><td></td><td>Range smoothing</td><td>Gaussian, one bin (6 cm)</td></tr><tr><td>Templates</td><td>Spacing</td><td>0.45 m travelled or 20° turned</td></tr><tr><td></td><td>Range-shift search</td><td>±3 bins (±18 cm)</td></tr><tr><td></td><td>Candidates</td><td>5 per pulse, similarity ≥ 0.60</td></tr><tr><td></td><td>Exclusion of recent templates</td><td>6 m of travel</td></tr><tr><td>Verifier</td><td>Heading gate</td><td>69° (1.2 rad)</td></tr><tr><td></td><td>Evidence per match</td><td>40 · (s − 0.70), clipped to ±4</td></tr><tr><td></td><td>Missed pulse</td><td>Penalty 0.5, dies after 4 missed pulses</td></tr><tr><td></td><td>Consistency</td><td>0.30 m +0.15× distance, 0.25 rad</td></tr><tr><td></td><td>Commit</td><td>8 pairs, 3 templates, evidence 10, margin 4 over rivals (&gt;1 m or &gt;0.4 rad apart)</td></tr><tr><td></td><td>Plausibility gate</td><td>χ at 99.9 % (16.27); relative covariance for corrections &gt;1.5 m</td></tr><tr><td>Pose graph</td><td>Risk-scaled commit</td><td>Correction &gt;1.5 m: 16 pairs, evidence 40, evidence ≥ number of pairs</td></tr><tr><td></td><td>Prior</td><td> $\sigma = ( 0 . 0 1 , 0 . 0 1 , 0 . 0 0 5 ) ^ { - }$ </td></tr><tr><td></td><td>Odometry</td><td> $\sigma = ( 0 . 0 4 , 0 . 0 2 , 0 . 0 3 5 ) \cdot \sqrt { d } + ( 0 . 0 0 5 , 0 . 0 0 5 , 0 . 0 0 3 )$ </td></tr><tr><td></td><td>Loop closure link</td><td> $\sigma = ( 0 . 3 0 , 0 . 3 0 , 0 . 2 0 )$  , Huber loss with threshold 1.345</td></tr><tr><td></td><td>Solver</td><td>iSAM2, Dogleg, relinearization threshold 0.01</td></tr><tr><td>Link management</td><td>Length rule</td><td>Fewer than 16 links after 10 pulses without a new link</td></tr><tr><td></td><td>Residual check</td><td>Median  $\chi ^ { 2 } > 1 2$  (tentative groups)</td></tr><tr><td></td><td>GNC audit</td><td>Every 400 nodes, Geman-McClure loss, keep if mean weight ≥ 0.5</td></tr></table>

## APPENDIX A

## PARAMETERS

Table I lists all parameters of BatSLAM 2.0. They are the default values of batslam/config.py in the accompanying code, and were used for all worlds, routes and sensors; none of them was tuned per test case.

## APPENDIX B

## REPRODUCIBILITY

All code, scene definitions and experiment definitions are available at https://github.com/Cosys-Lab/BatSLAM2. The simulator (section VI) is included as a copy under sim/simulator. An important note concerns the HRTF data: the HRTF file that we started from stores its directivity arrays in (elevation, azimuth, frequency) order, while the interpolation routine of the simulator assumed (azimuth, elevation, frequency). As both angle grids are identical (−90<sup>◦</sup> to 90<sup>◦</sup> in steps of 2.5<sup>◦</sup>), this went unnoticed at first, and it caused the simulated ears to point up and down instead of left and right (a reflector 45 to the side produced an interaural level difference of only 0.2 dB, and a reflector 30<sup>◦</sup> above the head one of 14 dB). All results in this paper were produced with the corrected (transposed) HRTF, for which the same test gives 18 dB and 0 dB, respectively. The check is included as sim/check\_hrtf\_orientation.m.

The datasets are generated with sim/generate\_dataset.m (worlds, drives, echoes and odometry) and sim/generate\_capture\_dataset.m (the capture study). The experiments are defined in suites/hc\_main.json, suites/hc\_ablation.json and suites/hc\_robustness.json, and are run with scripts/run\_suite.py. The front-end studies are run with scripts/bench\_descriptor.py, scripts/bench\_sensor.py and scripts/capture\_analysis.py. All numbers and tables in this paper are generated from the results by scripts/build\_paper\_generated.py, and all figures by scripts/paper\_figures.py.

## APPENDIX C

## RESULT TABLES

This appendix lists the numerical results that are discussed in the main text: single-view place recognition for the design choices of the descriptor (table II), the results of the complete system per condition (table III), the ablations (table IV) and the real-world results (table V).

SINGLE-VIEW PLACE RECOGNITION FOR THE DESIGN CHOICES OF THE DESCRIPTOR, WITH THE DESIGN SENSOR. ALL VARIANTS USE 32 CHANNELS, 6 cm RANGE BINS AND THE TVG. OUR DESCRIPTOR USES A CUBE-ROOT COMPRESSION. BEST VALUE PER COLUMN IN BOLD. ILD: INTERAURAL LEVEL DIFFERENCE.  
TABLE II
<table><tr><td></td><td colspan="2">Indoor floor</td><td colspan="2">City</td></tr><tr><td>Variant</td><td>Top-1</td><td>AUC</td><td>Top-1</td><td>AUC</td></tr><tr><td>Cues</td><td></td><td></td><td></td><td></td></tr><tr><td>Energy</td><td>0.875</td><td>0.915</td><td>0.878</td><td>0.918</td></tr><tr><td>Energy + shape (ours)</td><td>0.913</td><td>0.945</td><td>0.888</td><td>0.938</td></tr><tr><td>Energy + shape + ILD</td><td>0.892</td><td>0.968</td><td>0.878</td><td>0.960</td></tr><tr><td colspan="5">Compression of the energy image (energy + shape)</td></tr><tr><td>Logarithm, floor —30 dB</td><td>0.884</td><td>0.930</td><td>0.888</td><td>0.930</td></tr><tr><td>Logarithm, floor —50 dB</td><td>0.929</td><td>0.947</td><td>0.886</td><td>0.925</td></tr><tr><td>Square root</td><td>0.900</td><td>0.940</td><td>0.889</td><td>0.937</td></tr><tr><td>Linear</td><td>0.858</td><td>0.925</td><td>0.857</td><td>0.921</td></tr><tr><td>Matching of the original BatSLAM</td><td></td><td></td><td></td><td></td></tr><tr><td>Logarithmic energy, floor —30 dB</td><td>0.401</td><td>0.801</td><td>0.599</td><td>0.814</td></tr><tr><td>Energy + shape</td><td>0.793</td><td>0.901</td><td>0.802</td><td>0.889</td></tr></table>

TABLE III

RESULTS OF BATSLAM 2.0 PER CONDITION (RANGES OVER THE RUNS). ATE: TRAJECTORY ERROR OF ODOMETRY AND OF BATSLAM 2.0, WITHOUT AND AFTER RIGID ALIGNMENT; WRONG: WRONG COMMITTED HYPOTHESES IN THE FINAL GRAPH, OUT OF ALL COMMITTED HYPOTHESES; COVERAGE: FRACTION OE TRUE REVISITS WITH A CORRECT LINK
<table><tr><td></td><td colspan="3">ATE [m]</td><td></td><td></td></tr><tr><td>Condition (runs)</td><td>Odometry</td><td>Ours</td><td>Aligned</td><td>Wrong</td><td>Coverage</td></tr><tr><td>Calibrated (6)</td><td>4.2-11.1</td><td>0.47-1.91</td><td>0.18–0.46</td><td>0/230</td><td>0.79–0.84</td></tr><tr><td>Residual bias (6)</td><td>11.0-19.0</td><td>0.62-1.65</td><td>0.23-0.53</td><td>0/227</td><td>0.79–0.84</td></tr><tr><td>Degraded sensors (3)</td><td>7.5</td><td>0.52–-1.79</td><td>0.25-0.81</td><td>1/106</td><td>0.67-0.74</td></tr><tr><td>Uncalibrated (4)</td><td>19.1–22.3</td><td>1.38-12.68</td><td>0.45–2.96</td><td>0/149</td><td>0.62–0.81</td></tr></table>

TABLE IV

ABLATIONS OVER FOUR STANDARD CASES (ROUTES 3 AND 5 WITH CALIBRATED ODOMETRY, ROUTE 7 WITH A RESIDUAL BIAS, AND THE CITY). ATE: MEAN OVER THE CASES; WRONG: WRONG COMMITTED HYPOTHESES IN THE FINAL GRAPHS, SUMMED OVER THE CASES. NAIVE: NO SEQUENCE VERIFICATION AND NO LINK MANAGEMENT.
<table><tr><td>Variant</td><td>ATE [m]</td><td>Wrong</td><td>Coverage</td></tr><tr><td>Full system (ours)</td><td>0.87</td><td>0</td><td>0.81</td></tr><tr><td rowspan="2">Naive No sequence verification</td><td>18.90</td><td>364</td><td>0.72</td></tr><tr><td>4.85</td><td>10</td><td>0.80</td></tr><tr><td>Original front-end (8 bands) No time-varying gain</td><td>1.54</td><td>0</td><td>0.73</td></tr><tr><td></td><td>2.53</td><td>3</td><td>0.72</td></tr><tr><td>Energy image only</td><td>1.38</td><td>8</td><td>0.74</td></tr><tr><td>No range-shift search</td><td>1.47</td><td>0</td><td>0.60</td></tr><tr><td>Hand-set evidence constants</td><td>1.51</td><td>0</td><td>0.68</td></tr><tr><td>No risk-scaled commit</td><td>0.93</td><td>1</td><td>0.81</td></tr><tr><td>Length only (24 pairs)</td><td>1.20</td><td>0</td><td>0.81</td></tr><tr><td>No link management</td><td>0.55</td><td>0</td><td>0.82</td></tr></table>

TABLE V  
REAL-WORLD RESULTS, MEAN (STANDARD DEVIATION) OVER TEN ODOMETRY SEEDS (DISTANCE ×0.95, ROTATION ×1.10).
<table><tr><td>Evidence curve</td><td>ATE (m)</td><td>Aligned (m)</td><td>Links</td><td>Precision</td><td>Wrong Hyp.</td><td>Revisits Linked</td><td>Collapsed Pairs</td></tr><tr><td>Odometry only</td><td>3.62 (0.46)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Simulation evidence curve</td><td>0.77 (0.07)</td><td>0.60 (0.05)</td><td>108</td><td>1.000 (0.000)</td><td>0</td><td>0.50 (0.01)</td><td>0.000</td></tr><tr><td>Refitted, even stretches</td><td>0.67 (0.13)</td><td>0.52 (0.11)</td><td>141</td><td>0.980 (0.018)</td><td>0</td><td>0.64 (0.10)</td><td>0.000</td></tr><tr><td>Refitted, odd stretches</td><td>0.66 (0.12)</td><td>0.52 (0.11)</td><td>142</td><td>0.975 (0.021)</td><td>0</td><td>0.64 (0.10)</td><td>0.000</td></tr></table>

## APPENDIX D

## ADDITIONAL RESULTS

In this appendix, we show the trajectories and the detailed results of all runs of the main and robustness experiments, and of the ablations. In every map, the grey line is the ground truth, the orange line the dead-reckoning estimate, and the blue line the trajectory estimated by BatSLAM 2.0 (without alignment, as it was estimated online). Links of wrong committed hypotheses that remain in the final graph would be drawn in red. The title of every panel lists the trajectory error of BatSLAM 2.0 and of odometry, and the number of wrong committed hypotheses in the final graph.

TABLE VI  
RESULTS OF BATSLAM 2.0 ON ALL RUNS OF THE MAIN EXPERIMENT (DRIVES OF ABOUT 1 km). ATE ODO: DEAD RECKONING; ATE AND ALIGNED: BATSLAM 2.0 WITHOUT AND AFTER RIGID ALIGNMENT; PRECISION: FRACTION OF CORRECT LINKS; WRONG (FINAL / ALL): WRONG COMMITTED HYPOTHESES IN THE FINAL GRAPH AND OVER ALL COMMITS; COVERAGE: FRACTION OF TRUE REVISITS WITH A CORRECT LINK; COLLAPSED: FRACTION OF COLLAPSED PAIRS.
<table><tr><td>Case</td><td>Odometry</td><td>ATE odo [m] ↓</td><td>ATE [m] ↓</td><td>Aligned [m] ↓</td><td>Precision ↑</td><td>Wrong (final) ↓</td><td>Wrong (all) ↓</td><td>Coverage ↑</td><td>Collapsed [%] ↓</td></tr><tr><td>Indoor, route 3, seed 1</td><td>Calibrated</td><td>7.5</td><td>0.78</td><td>0.25</td><td>0.999</td><td>0/39</td><td>0</td><td>0.84</td><td>0.0</td></tr><tr><td>Indoor, route 3, seed 2</td><td>Calibrated</td><td>4.2</td><td>0.89</td><td>0.24</td><td>0.999</td><td>0/40</td><td>0</td><td>0.84</td><td>0.0</td></tr><tr><td>Indoor, route 3, seed 3</td><td>Calibrated</td><td>7.9</td><td>1.02</td><td>0.19</td><td>1.000</td><td>0/36</td><td>1</td><td>0.84</td><td>0.0</td></tr><tr><td>Indoor, route 5</td><td>Calibrated</td><td>10.1</td><td>0.47</td><td>0.23</td><td>0.998</td><td>0/45</td><td>0</td><td>0.82</td><td>0.0</td></tr><tr><td>Indoor, route 7</td><td>Calibrated</td><td>7.6</td><td>1.91</td><td>0.18</td><td>1.000</td><td>0/46</td><td>0</td><td>0.80</td><td>0.0</td></tr><tr><td>City</td><td>Calibrated</td><td>11.1</td><td>0.60</td><td>0.46</td><td>0.996</td><td>0/24</td><td>0</td><td>0.79</td><td>0.0</td></tr><tr><td>Indoor, route 3, sensor —10 dB</td><td>Calibrated</td><td>7.5</td><td>0.52</td><td>0.26</td><td>0.997</td><td>0/34</td><td>0</td><td>0.72</td><td>0.0</td></tr><tr><td>Indoor, route 3, sensor —20 dB</td><td>Calibrated</td><td>7.5</td><td>1.29</td><td>0.81</td><td>0.989</td><td>1/32</td><td>23</td><td>0.67</td><td>0.1</td></tr><tr><td>Indoor, route 3, original sensor</td><td>Calibrated</td><td>7.5</td><td>1.79</td><td>0.25</td><td>0.995</td><td>0/40</td><td></td><td>0.74</td><td>0.0</td></tr><tr><td>Indoor, route 3, seed 1</td><td>Residual bias</td><td>11.0</td><td>0.73</td><td>0.29</td><td>1.000</td><td>0/38</td><td>1</td><td>0.83</td><td>0.0</td></tr><tr><td>Indoor, route 3, seed 2</td><td>Residual bias</td><td>14.0</td><td>1.34</td><td>0.29</td><td>1.000</td><td>0/38</td><td>0</td><td>0.84</td><td>0.0</td></tr><tr><td>Indoor, route 3, seed 3</td><td>Residual bias</td><td>19.0</td><td>0.96</td><td>0.24</td><td>0.999</td><td>0/40</td><td>1</td><td>0.84</td><td>0.0</td></tr><tr><td>Indoor, route 5</td><td>Residual bias</td><td>14.6</td><td>1.20</td><td>0.27</td><td>0.999</td><td>0/41</td><td>0</td><td>0.80</td><td>0.0</td></tr><tr><td>Indoor, route 7</td><td>Residual bias</td><td>11.0</td><td>1.65</td><td>0.23</td><td>1.000</td><td>0/42</td><td>0</td><td>0.79</td><td>0.0</td></tr><tr><td>City</td><td>Residual bias</td><td>17.5</td><td>0.62</td><td>0.53</td><td>0.994</td><td>0/28</td><td>0</td><td>0.82</td><td>0.0</td></tr><tr><td>Indoor, route 3, seed 1</td><td>Uncalibrated</td><td>22.1</td><td>1.38</td><td>0.45</td><td>1.000</td><td>0/37</td><td>1</td><td>0.81</td><td>0.0</td></tr><tr><td>Indoor, route 3, seed 2</td><td>Uncalibrated</td><td>19.1</td><td>9.31</td><td>2.22</td><td>0.998</td><td>0/36</td><td>1</td><td>0.62</td><td>0.6</td></tr><tr><td>Indoor, route 3, seed 3</td><td>Uncalibrated</td><td>22.3</td><td>12.68</td><td>2.96</td><td>0.999</td><td>0/33</td><td>1</td><td>0.62</td><td>0.8</td></tr><tr><td>Indoor, route 5</td><td>Uncalibrated</td><td>21.9</td><td>1.80</td><td>0.45</td><td>1.000</td><td>0/43</td><td>0</td><td>0.75</td><td>0.0</td></tr></table>

TABLE VII  
ROBUSTNESS RUNS: ODOMETRY BIAS FACTOR b, THREE TIMES THE RANDOM ODOMETRY NOISE, AND DEGRADED SENSORS WITH A RESIDUAL BIAS. COLUMNS AS IN TABLE VI.
<table><tr><td>Case</td><td>Bias</td><td>ATE odo [m] ↓</td><td>ATE [m] ↓</td><td>Aligned [m] ↓</td><td>Precision ↑</td><td>Wrong (final) ↓</td><td>Wrong (all) ↓</td><td>Coverage ↑</td><td>Collapsed [%] ↓</td></tr><tr><td>Indoor, route 3, seed 4</td><td>b = 0</td><td>8.0</td><td>1.22</td><td>0.30</td><td>1.000</td><td>0/38</td><td>1</td><td>0.82</td><td>0.0</td></tr><tr><td>Indoor, route 3, seed 5</td><td>b= 0</td><td>7.6</td><td>0.34</td><td>0.20</td><td>0.999</td><td>0/40</td><td>1</td><td>0.83</td><td>0.0</td></tr><tr><td>Indoor, route 3, seed 4</td><td>b = 0.25</td><td>15.9</td><td>1.98</td><td>0.33</td><td>0.999</td><td>0/38</td><td>1</td><td>0.83</td><td>0.0</td></tr><tr><td>Indoor, route 3, seed 5</td><td>b = 0.25</td><td>20.7</td><td>0.45</td><td>0.24</td><td>0.999</td><td>0/37</td><td>2</td><td>0.83</td><td>0.0</td></tr><tr><td>Indoor, route 3, sensor -20 dB</td><td>b = 0.25</td><td>11.0</td><td>1.98</td><td>1.43</td><td>0.988</td><td>1/33</td><td>3</td><td>0.65</td><td>0.2</td></tr><tr><td>Indoor, route 3, sensor -10 dB</td><td>b = 0.25</td><td>11.0</td><td>0.94</td><td>0.28</td><td>1.000</td><td>0/37</td><td>21</td><td>0.76</td><td>0.0</td></tr><tr><td>Indoor, route 3, seed 1, odometry noise ×3</td><td>b = 0.25</td><td>11.7</td><td>2.37</td><td>0.48</td><td>1.000</td><td>0/50</td><td></td><td>0.81</td><td>0.0</td></tr><tr><td>Indoor, route 3, original sensor</td><td>b = 0.25</td><td>11.0</td><td>1.48</td><td>0.28</td><td>0.998</td><td>0/38</td><td>2</td><td>0.73</td><td>0.0</td></tr><tr><td>Indoor, route 3, seed 4</td><td>b = 0.5</td><td>20.3</td><td>1.36</td><td>0.40</td><td>1.000</td><td>0/38</td><td>1</td><td>0.82</td><td>0.0</td></tr><tr><td>Indoor, route 3, seed 5</td><td>b = 0.5</td><td>22.9</td><td>0.75</td><td>0.30</td><td>1.000</td><td>0/40</td><td>1</td><td>0.82</td><td>0.0</td></tr><tr><td>Indoor, route 3, seed 4</td><td>b = 1</td><td>20.1</td><td>1.60</td><td>0.48</td><td>1.000</td><td>0/40</td><td>1</td><td>0.83</td><td>0.0</td></tr><tr><td>Indoor, route 3, seed 5</td><td>b = 1 b = 2</td><td>20.3</td><td>14.54 19.08</td><td>2.87 18.17</td><td>1.000 1.000</td><td>0/32</td><td>1 0</td><td>0.62</td><td>0.8</td></tr><tr><td>Indoor, route 3, seed 4</td><td></td><td>33.5</td><td></td><td></td><td>1.000</td><td>0/23</td><td>0</td><td>0.41</td><td>4.8</td></tr><tr><td>Indoor, route 3, seed 5</td><td>b = 2</td><td>35.7</td><td>20.47</td><td>19.71</td><td></td><td>0/11</td><td></td><td>0.32</td><td>7.4</td></tr></table>

r7\_b025 ATE 1.65 m (aligned 0.23, odo 11.0), 0 wrong city\_b0 ATE 0.60 m (aligned 0.46, odo 11.1), 0 wrong noise1.5e-5\_b0 ATE 1.29 m (aligned 0.81, odo 7.5), 1 wrong noise5e-6\_b0 ATE 0.52 m (aligned 0.26, odo 7.5), 0 wrong oldsensor\_b0 ATE 1.79 m (aligned 0.25, odo 7.5), 0 wrong

![](images/b3b6f009710d77a04506f4c26572870cf3452f42eb4c3652febc05b5f0390138.jpg)  
r3\_b0\_s1 ATE 0.78 m (aligned 0.25, odo 7.5), 0 wrong

![](images/67a15e9a64099312e3a09397837bd8f2ef495420f7251c721b348e0255c417ff.jpg)  
r3\_b0\_s2 ATE 0.89 m (aligned 0.24, odo 4.2), 0 wrong

![](images/e2195e5c87f6955ab2c938ce5d849790afe206645d2f530bfbfece22aa36bd3a.jpg)  
r3\_b0\_s3 ATE 1.02 m (aligned 0.19, odo 7.9), 0 wrong

![](images/da0525b4ba4db1ca5827cb8c77ecd56cbcd0ecd906a99e56e731e95209d705e9.jpg)  
r5\_b0 ATE 0.47 m (aligned 0.23, odo 10.1), 0 wrong

![](images/eedc81202e037769189e762812852f4a1c309511484e7d821abfdaba6f819f80.jpg)  
r7\_b0 ATE 1.91 m (aligned 0.18, odo 7.6), 0 wrong

![](images/f2e93ee11b813e8d1e9687cb8f4f36b09fec7e5a9875d6118f9cef3790937909.jpg)  
city\_b025 ATE 0.62 m (aligned 0.53, odo 17.5), 0 wrong

![](images/1ad32667b6f0dbec62c6880aac8e1b38b82402343e7dbe1727aec3be713605df.jpg)

![](images/afa081abe5315792fcb696fbcf31075f4f38455839659b36751b332ce6cadcba.jpg)  
r3\_b025\_s2 ATE 1.34 m (aligned 0.29, odo 14.0), 0 wrong

![](images/f023d54ab45fa22ec9c8b2f924adaafd2f02518982a04399fb2cd35fd7d16e1d.jpg)  
r3\_b025\_s3 ATE 0.96 m (aligned 0.24, odo 19.0), 0 wrong

![](images/9460c15177107821a39265772910062cbcb32961ef67c09426c5f5ef85df0532.jpg)  
r5\_b025 ATE 1.20 m (aligned 0.27, odo 14.6), 0 wrong

![](images/3a64e1ffe3f2894709dda55cf594924da6f7aa1759f6deed644d715eb3e949e7.jpg)

![](images/31a14998af4165d7aad070bf7d0b15944002832b6d355384eb5332d9614b2864.jpg)  
r3\_b1\_s1 ATE 1.38 m (aligned 0.45, odo 22.1), 0 wrong

![](images/ef79e23bcbe4657512c71e5304748adafd1cebc4abc7dc23958855fcb71e84a1.jpg)  
r3\_b1\_s2 ATE 9.31 m (aligned 2.22, odo 19.1), 0 wrong

![](images/1c38777dd339f5fea6a2108e406514ed5b199bbe3d9aff782987d23c71f5cef6.jpg)

![](images/efb24af27f9e11d68e8c4eb33b5f46c15b2e5c04191b34aac2f55fd76729d703.jpg)

![](images/ccef918edae3acc90520185db27ef60565ec128096f264d4e892b8c4a717190e.jpg)

![](images/fc690f4a4fe6fe74f5be64133f686363c63c7a772752f93cb1971fc24fb0db93.jpg)

![](images/0d8ce1e298d9f4a1d329ee03810331511c6d0e6ce23bbd4b2c5b79cc0faf441f.jpg)

![](images/9fb196a5bac0d63fd3fdcbd91cd07d6afb1dca2ac38e73750c8026e590375ba1.jpg)  
Fig. 12. Trajectories of all runs of the main experiment (table VI), sorted by the odometry condition. Grey: ground truth; orange: odometry; blue: BatSLAM 2.0. Run names: r3, r5, r7: routes 3, 5 and 7 on the indoor floor; city: the city world; b0, b025, b1: bias factor b = 0, 0.25 and 1; s1–s3: odometry noise realization; noise5e-6, noise1.5e-5, oldsensor: degraded sensors.

TABLE VIII  
ABLATION OF THE SAFEGUARDS ON THE FOUR HARDEST CASES (THREE DEGRADED SENSORS AND ROUTE 3 WITH ODOMETRY SEED 3). EVERY CELL LISTS THE ALIGNED TRAJECTORY ERROR (M) AND, IN PARENTHESES, THE NUMBER OF WRONG HYPOTHESES IN THE FINAL GRAPH; THE LAST TWO COLUMNS SUM THE WRONG HYPOTHESES IN THE FINAL GRAPHS AND OVER ALL COMMITS.
<table><tr><td>Variant</td><td>-10 dB sensor</td><td>-20 dB sensor</td><td>Original sensor</td><td>Route 3, seed 3</td><td>Wrong (final)</td><td>Wrong (all)</td></tr><tr><td>No risk-scaled commit</td><td>0.26 (0)</td><td>0.24 (0)</td><td>1.06 (1)</td><td>0.19 (0)</td><td>1</td><td>10</td></tr><tr><td>Risk-scaled commit, length only (24 pairs)</td><td>0.26 (0)</td><td>0.24 (0)</td><td>0.25 (0)</td><td>0.19 (0)</td><td>0</td><td>5</td></tr><tr><td>Plausibility gate on the absolute marginal</td><td>0.26 (0)</td><td>0.23 (0)</td><td>0.69 (1)</td><td>0.19 (0)</td><td>1</td><td>12</td></tr><tr><td>No plausibility gate</td><td>0.26 (0)</td><td>1.37 (2)</td><td>0.60 (1)</td><td>0.19 (0)</td><td>3</td><td>19</td></tr><tr><td>No length rule</td><td>0.26 (0)</td><td>0.81 (2)</td><td>1.07 (3)</td><td>0.20 (2)</td><td>7</td><td>7</td></tr><tr><td>No link management</td><td>0.26 (0)</td><td>0.81 (2)</td><td>1.07 (3)</td><td>0.20 (2)</td><td>7</td><td>7</td></tr><tr><td>Full system (ours)</td><td>0.26 (0)</td><td>0.81 (1)</td><td>0.25 (0)</td><td>0.19 (0)</td><td>1</td><td>6</td></tr></table>

![](images/0f1b1a23ed29967b468573eaeed01452fb3afeef0a0ac4b8bdc038cebf1921ff.jpg)  
Fig. 13. Trajectories of all runs of the robustness experiment (table VII). biasb\_sk: bias factor b with odometry realization k; odomnoise\_x3: three times the random odometry noise; the last three runs use degraded sensors with a residual bias. Colors as in figure 12.

TABLE IX  
ABLATIONS PER CASE: TRAJECTORY ERROR (ATE, IN M) AND THE NUMBER OF WRONG COMMITTED HYPOTHESES IN THE FINAL GRAPH (IN PARENTHESES), FOR ROUTE 3 AND ROUTE 5 WITH CALIBRATED ODOMETRY, ROUTE 7 WITH A RESIDUAL BIAS, AND THE CITY WITH CALIBRATED ODOMETRY.
<table><tr><td>Variant</td><td>Route 3</td><td>Route 5</td><td>Route 7, residual bias</td><td>City</td></tr><tr><td>Naive (single views, no gates, no link mgmt.)</td><td>8.00 (63)</td><td>26.70 (68)</td><td>8.22 (72)</td><td>32.68 (161)</td></tr><tr><td>No sequence verification (single views)</td><td>5.08 (1)</td><td>1.95 (1)</td><td>8.17 (6)</td><td>4.21 (2)</td></tr><tr><td>Original front-end (8 wide bands)</td><td>1.97 (0)</td><td>0.44 (0)</td><td>3.15 (0)</td><td>0.59 (0)</td></tr><tr><td>No time-varying gain</td><td>0.85 (1)</td><td>6.82 (2)</td><td>1.84 (0)</td><td>0.62 (0)</td></tr><tr><td>Energy image only (no shape)</td><td>0.63 (0)</td><td>1.70 (3)</td><td>1.26 (4)</td><td>1.92 (1)</td></tr><tr><td>No range-shift search</td><td>0.45 (0)</td><td>1.57 (0)</td><td>3.28 (0)</td><td>0.56 (0)</td></tr><tr><td>No risk-scaled commit</td><td>0.88 (0)</td><td>0.46 (0)</td><td>1.79 (1)</td><td>0.57 (0)</td></tr><tr><td>Risk-scaled commit, length only (24 pairs)</td><td>0.74 (0)</td><td>0.47 (0)</td><td>2.98 (0)</td><td>0.60 (0)</td></tr><tr><td>Risk-scaled commit, length only (16 pairs)</td><td>0.78 (0)</td><td>0.47 (0)</td><td>1.78 (1)</td><td>0.52 (0)</td></tr><tr><td>No length rule</td><td>0.63 (0)</td><td>0.47 (0)</td><td>0.49 (0)</td><td>0.63 (0)</td></tr><tr><td>No link management</td><td>0.63 (0)</td><td>0.47 (0)</td><td>0.49 (0)</td><td>0.63 (0)</td></tr><tr><td>No plausibility gate</td><td>0.78 (0)</td><td>0.47 (0)</td><td>1.65 (0)</td><td>0.60 (0)</td></tr><tr><td>Plausibility gate on the absolute marginal</td><td>0.78 (0)</td><td>0.47 (0)</td><td>1.65 (0)</td><td>0.60 (0)</td></tr><tr><td>No heading gate</td><td>0.64 (0)</td><td>0.42 (0)</td><td>1.55 (0)</td><td>0.50 (0)</td></tr><tr><td>Hand-set evidence constants</td><td>0.68 (0)</td><td>0.63 (0)</td><td>3.29 (0)</td><td>1.45 (0)</td></tr><tr><td>Full system (ours)</td><td>0.78 (0)</td><td>0.47 (0)</td><td>1.65 (0)</td><td>0.60 (0)</td></tr></table>