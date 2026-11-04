# Goal-Driven Autonomous Optical Experimentation via a Self-Recursive Hardware–Software Harness

Tianxing Zhu<sup>\*</sup>, Huacheng Lu<sup>\*</sup>, Ming Liu<sup>\*\*</sup>, and Malte Hillebrand<sup>\*</sup> Konica Minolta (China) Investment Ltd., Shanghai, China Giant Testing Equipment Ltd., Shanghai, China

## Abstract

We propose goal-driven self-recursive optical research in which agents use physical evidence to revise the hardware, software, models, andprotocols used topursue a scientific goal. Theframework can be instantiated with diferent stimulus systems and measurement devices. Weformalize its updates and derive a relation between physical attainability, command access, and model approximation error. We demonstrate the concept in an optical seethrough color-matching case study using an iPad, RayNeo X3 glasses, and a TOP 300 probe coupled to a CAS140D spectroradiometer, both from Instrument Systems. A revised glasses application adds USB-controlled RGB holds and frame acknowledgements, keeping model-generated colors displayed during spectral acquisition. The interface passes 12 live control checks. Spectral feedback reduces one target color diferencefrom 19.756 to 1.211 $\Delta E _ { 7 6 } ^ { * } ;$ all five targets improve in a separate comparison. These measurements demonstrate the concept in one configuration. The resulting software and protocols remain availablefor subsequent investigations.

## Author Keywords

self-recursive research; autonomous experimentation; scientific agents; optical see-through displays

## 1. Introduction

An optical investigation often requires a new stimulus application, coordinated interfaces for measurement devices, or a calibration method suited to the phenomenon being measured. Developing these capabilities is part of the research. An autonomous system must be able to implement scientific decisions on physical equipment and use the resulting evidence to revise both the experiment and its supporting applications.

Self-driving laboratories combine experimental design, physical execution, and inference in closed loops [1]. A mobile robotic chemist demonstrated autonomous exploration of photocatalytic formulations [2], while scientific agents use language-based reasoning to operate computational and laboratory tools [3, 4]. We consider how such a system can establish and develop the hardware and software environment needed for a scientific goal.

We introduce goal-driven self-recursive optical research through an integrated hardware and software harness comprising the apparatus, device applications, measurement interfaces, and analysis tools available to agents. The concept can be instantiated in diferent experimental configurations. To demonstrate it, we develop a case study in which agents create applications for iPad background generation, RayNeo AR glasses rendering, and optical measurement. Scientific planning determines the experiments, and progress monitoring evaluates the evidence against the goal.

Once the harness is established, a result can prompt a diferent test or a revision to the software, measurement protocol, or analytical method. Agents implement the revision and resume physical testing.

We call this process self-recursive because evidence changes the means used to obtain further evidence. The scientific goal persists across these changes, including the joint adaptation of experiments and methods.

As a study and demonstration of the concept, we use optical seethrough color compensation, in which display emission combines with transmitted background light at the viewing pupil [5, 6]. A question about temporal response or spatial variation may require new capabilities in the stimulus and measurement applications. A prediction discrepancy may require a diferent response model. This case study examines these revisions within one research environment and illustrates how physical attainability, available commands, and model accuracy afect the optical outcome.

## 2. Related Work

Autonomous laboratories and scientific agents: Selfdriving laboratories combine learning algorithms with automation and instruments in chemistry and materials science [1]. ChemOS 2.0 orchestrates heterogeneous experimental and computational components [7], and distributed knowledge graphs represent laboratory resources and experimental knowledge [8].

Scientific agents extend the reasoning and development work per formed in these environments. Coscientist combines language models with search, computation, and laboratory capabilities [3]. Chem-Crow gives language models access to specialized chemistry tools [4]. Work on scientific facilities examines retrieval and tool augmentation [9], while agents that learn instrument operation retain procedural knowledge from experience [10]. These capabilities support the development of applications and protocols through which agents operate equipment.

A stimulus application determines which commands the system can execute. An acquisition interface determines which physical distinctions it can measure, and a protocol defines the experimental comparison. We treat these as revisable components of the harness, with development directed by the scientific goal and the evidence already acquired.

Optical discovery and programmable experimental environments: AutoSciLab combines autonomous experimentation with interpretable scientific modeling in nanophotonics [11]. Optical see-through displays ofer another setting for optical autonomy: programmable display emission combines with background light in a defined geometry [5, 6]. The response depends on the stimulus, background, and viewing position, so the investigation requires coordinated control and observation.

The required capabilities depend on the question. Spectral measurements require radiometric observation; spatial measurements require image formation and registration. Settling studies require stimulus sequences and observations that resolve the changing response. Goal-driven self-recursion includes developing the applications, measurement strategies, and models needed as these questions arise.

![](images/98ade53021f0a995a534c5eec7e87fa6d93a13f69ae68088b9b186052c6d11ad.jpg)  
Figure 1. Goal-driven self-recursion with the notation of Eqs. (1) and (2). The goal G directs establishment of $\mathbf { H _ { 0 } } .$ , experimental planning, and execution. Optical observations and execution evidence inform the recursive update of the harness, plan, and method. The revised environment enables subsequent tests, while accumulated evidence is evaluated against the same scientific goal.

## 3. Goal-driven self-recursion

A persistent goal and an evolving research environment: The scientific goal specifies the intended outcome, its application domain, and the evidence used to assess progress. In color compensation, it may be to reproduce a family of target appearances across specified backgrounds and viewing conditions. Individual target matches test parts of that objective. The goal determines which measurements and capability revisions are useful and when the evidence supports the intended outcome.

At research stage k, the harness includes the physical apparatus, device and observation applications, and their connection to scientific computation. The plan organizes the questions to be tested. The method specifies the response models and inferential procedures. Together with the goal and accumulated evidence, they determine a physical action:

$$
\begin{array} { r l } & { a _ { t } = \pi ( G , D _ { t } ; H _ { k } , \Pi _ { k } , m _ { k } ) , \qquad a _ { t } \in \mathcal { A } ( H _ { k } ) , } \\ & { \qquad y _ { t } = \mathcal { O } _ { H _ { k } } [ F _ { p h y s } ( a _ { t } ) ] + \varepsilon _ { t } . } \end{array}\tag{1}
$$

Here G is the scientific goal, $D _ { t }$ is the accumulated evidence, and $H _ { k }$ is the integrated harness. The plan is $\Pi _ { k }$ and the method is $m _ { k }$ . The action set $\boldsymbol { \mathcal { A } } ( \boldsymbol { H _ { k } } )$ contains the tests that the harness can execute. An action can specify a display command, a background condition, and an acquisition configuration. The observation operator describes how the instruments and acquisition applications convert the physical response into evidence.

The apparatus supplies the physical interaction, and the software gives the agents access to it. Adding an application feature or acquisition strategy therefore changes which experiments the system can perform.

Evidence-driven revision of the means of investigation: Experimental evidence updates the research environment and scientific method:

$$
\begin{array} { r l } & { ( H _ { k + 1 } , \Pi _ { k + 1 } , m _ { k + 1 } ) } \\ & { = \mathcal { R } ( G , D _ { t } , E _ { t } ; H _ { k } , \Pi _ { k } , m _ { k } ) . } \end{array}\tag{2}
$$

The evidence $E _ { t }$ includes execution feedback and discrepancies that help explain the observation. The revision operator can change the test plan, extend stimulus-control applications, refine acquisition through the available measurement devices, replace a model, or revise the measurement protocol. The updated harness and plan implement the change, and subsequent tests measure its consequences.

The goal determines which revisions to pursue. A new capability may enable an informative test, a model revision may improve the measured optical response, and a plan revision may resolve a remaining scientific question.

Goal-relative progress across experimental contexts: Improving a display setting and revising the method that selects settings are diferent research actions. For an optical context defined by a target, background, and geometry, the method selects a command available through the harness:

$$
\begin{array} { c } { { u _ { H , m } ( c ) = a r g m i n \updownarrow _ { m } ( \widehat { F } _ { m } ( u , b , g ) , q ) , } } \\ { { u \in \mathcal { U } ( H ) } } \\ { { K ( H , m ; c ) = d ( F _ { p h y s } ( u _ { H , m } ( c ) , b , g ) , q ) , } } \\ { { c = ( q , b , g ) . } } \end{array}\tag{3}
$$

The physical outcome $K ( H , m ; c )$ measures performance in that context. Outcomes across contexts describe how model and capability revisions afect the goal. A distribution $\mu _ { G }$ specifies the target and context domain:

$$
\begin{array} { r } { R _ { G } ( H , m ) = \mathbb { E } _ { c \sim \mu _ { G } } [ K ( H , m ; c ) ] , } \\ { \Gamma _ { G } = R _ { G } ( H , m ) - R _ { G } ( H ^ { \prime } , m ^ { \prime } ) . } \end{array}\tag{4}
$$

Positive gain means that the revised harness and method reduce expected mismatch over the goal domain. This measure covers changes to the response method and to available commands. New observation capabilities can also advance the investigation by resolving a scientific question. Progress is assessed using the outcome specified by the goal [12].

## 4. Demonstration platform and harness development

Hardware and software integration: In the developed case study, the harness combines the apparatus and optical geometry with device applications, measurement interfaces, analysis tools, and the agents that develop and use them. It carries a research question through physical execution to a measured response. The stim ulus, background, viewing condition, and acquisition settings give that response its experimental context.

In this demonstration platform, the iPad generates the background and the RayNeo X3 supplies display emission. A TOP 300 optical probe and a CAS140D spectroradiometer, both from Instrument Systems, provide optical measurement [13]. The probe includes an alignment viewfinder and couples the collected light to the spectroradiometer. Device applications generate stimuli, acquisition software records spectral and spatial observations, and analysis tools fit response models and select further tests.

Establishing the harness starts with the stimuli and observations required by the goal. Development agents create or extend the applica-

Table 1. Agent responsibilities in the demonstration platform and the capabilities they establish and revise.
<table><tr><td rowspan=1 colspan=1>Agent role</td><td rowspan=1 colspan=1>Harness contribution</td><td rowspan=1 colspan=1>Goal-directed development</td></tr><tr><td rowspan=1 colspan=1>Scientific planning</td><td rowspan=1 colspan=1>Experimental questions, models, andtest plans</td><td rowspan=1 colspan=1>Select informative tests and revise the investigation using newevidence</td></tr><tr><td rowspan=1 colspan=1>iPad applicationdevelopment</td><td rowspan=1 colspan=1>Background generation andstimulus-state observation</td><td rowspan=1 colspan=1>Add background patterns, sequences, or control features required bynew questions</td></tr><tr><td rowspan=1 colspan=1>RayNeo applicationdevelopment</td><td rowspan=1 colspan=1>Glasses rendering and display control</td><td rowspan=1 colspan=1>Extend stimulus representation and rendering behavior forsubsequent tests</td></tr><tr><td rowspan=1 colspan=1>Observation applicationdevelopment</td><td rowspan=1 colspan=1>Camera-spectrometer acquisition andanalysis interfaces</td><td rowspan=1 colspan=1>Relate spatial and spectral observations and develop newacquisition protocols</td></tr><tr><td rowspan=1 colspan=1>Progress monitoring</td><td rowspan=1 colspan=1>Interpretation of progress against thepersistent goal</td><td rowspan=1 colspan=1>Identify unresolved objectives and direct testing, analysis, orcapability development</td></tr></table>

![](images/66406e93eb31b37c24b59dc53fb2b7ef77491e88de715ae8388e125ed663a6b0.jpg)

Figure 2. Agent responsibilities and implemented   
interfaces in the case study. The iPad viewer uses HTTP   
over a USB SSH connection; the glasses application   
receives USB/ADB commands; spectral acquisition   
uses the CAS4x64 SDK over USB. The TOP 300   
integrates a viewfinder with the fiber-coupled optical   
probe [13]. Development examples identify source-level   
changes such as adding a HOLD command entry point,   
extending a viewer endpoint, or revising the SDK   
worker.

tions, while optical characterization relates the controllable device states to measured responses.

Agent responsibilities: In this case study, scientific planning, application development, and progress monitoring share a goal and the physical evidence from the investigation (Fig. 2). Table 1 assigns their responsibilities across the device and instrument applications.

The iPad development agent builds background-generation features, and the RayNeo development agent builds rendering and stimulus control. The observation development agent builds the cameraspectrometer application and its acquisition interfaces.

The progress-monitoring agent evaluates results against the goal and test plan. It identifies unresolved objectives and directs the next measurement, analysis, or application revision. This role includes deciding when another measurement is useful and when the investigation first needs a new capability.

Illustration of recursive development: As a possible extension of the case study, consider an investigation of target appearance across changing backgrounds. An initial harness generates static backgrounds, renders virtual colors, and acquires spectra for a response model. Suppose new observations depend on the preceding stimulus. The next comparison must distinguish optical memory from a static response discrepancy and may require new background and rendering sequences with time-resolved acquisition.

The planning agent specifies that comparison. Device development agents extend the iPad and glasses applications to generate the sequences, and the observation development agent extends acquisition to resolve the response. These tests inform the response model and measurement protocol. Progress monitoring evaluates the result against the original appearance objective and selects the next investigation.

The progress-monitoring agent checks revisions against the measurement relationships they must preserve. A measurement transaction records the requested stimulus, device acknowledgement, acquisition settings, and spectrum. Invariant checks cover stimulus identity, calibration and units, display stability, and commandobservation correspondence. A revision capsule records the scientific motivation, source or build fingerprint, and reference checks. Known stimuli and retained reference observations allow comparisons across application versions. In Section 6.2, command validation and framebufer checks precede spectral acquisition with the revised interface.

Physical optical arrangement: The iPad background and Ray-Neo X3 emission enter a common viewing pupil (Fig. 3). The TOP 300 probe collects the light and guides it through a fiber to the CAS140D spectroradiometer. Its integrated viewfinder camera supports alignment [13]. Device applications control the stimuli, and acquisition software records the spectral response used for the colorimetric tests in Section 6.

## 5. Optical formulation

Additive light formation and the attainable set: For an optical see-through display, a first-order spectral description at a fixed viewing geometry is

$$
\begin{array} { l } { { S _ { o b s } ( \lambda ; u , b , g ) = S _ { A R } ( \lambda ; u , g ) } } \\ { { \ + \tau ( \lambda , g ) S _ { b g } ( \lambda ; b , g ) + \epsilon ( \lambda ) . } } \end{array}\tag{5}
$$

Display emission and transmitted background light combine at the observation plane, with spectral transmittance weighting the background contribution. Integration against the CIE colormatching functions gives tristimulus values, which are transformed to CIELAB using a specified reference white [14]. We denote the physical color response by $F _ { p h y s } ( u , b , g )$

The physical command domain defines the attainable color set for

![](images/99af9e85dd9e35de577a1d9f2a3e66632f6a89e077ecb9155be4c9aabafc7dab.jpg)  
Figure 3. Photograph of the demonstration platform. (1) TOP 300 optical probe, (2) mounted RayNeo X3 glasses, and (3) iPad displaying an alignment pattern. Adjustable translation stages position the glasses between the background display and the probe viewing pupil. The probe is fiber-coupled to the CAS140D spectroradiometer.

## ① TOP 300

Optical probe with integrated viewfinder

Optical see-through glasses

Controlled background and alignment pattern

a background and geometry. The harness exposes a subset of those commands, giving two target floors:

$$
\begin{array} { r l } & { \mathcal { G } _ { b , g } = \{ F _ { p h y s } ( u , b , g ) : u \in \mathcal { U } _ { p h y s } \} , } \\ & { \qquad \mathcal { U } ( H ) \subseteq \mathcal { U } _ { p h y s } , } \\ & { L _ { p h y s } ^ { * } ( c ) = \underset { u \in \mathcal { U } _ { p h y s } } { \operatorname* { m i n } } \ \mathring { d } ( F _ { p h y s } ( u , b , g ) , q ) , } \\ & { \qquad L _ { H } ^ { * } ( c ) = \underset { u \in \mathcal { U } ( H ) } { \operatorname* { m i n } } \ \mathring { d } ( F _ { p h y s } ( u , b , g ) , q ) . } \end{array}\tag{6}
$$

Display commands add light to a background contribution set by optical transmission. The attainable set therefore depends on the background and geometry. An idealized linear additive system translates the response set in XYZ. Nonlinear device response, wavelength-dependent transmission, and conversion to CIELAB produce a more general geometry.

Extending a stimulus application can expose additional physically supported commands and lower the executable target floor. Extending acquisition can make a response observable. The apparatus defines the physical interactions; the harness determines which of them agents can use and measure.

Error bound for command access and model approximation: Consider a method that globally minimizes predicted target distance over the command domain exposed by the harness. At a fixed context, suppose its uniform approximation error satisfies

$$
\begin{array} { r l } & { \underset { u \in \mathcal { U } ( H ) } { \operatorname* { s u p } } d ( \widehat { F } _ { m } ( u , b , g ) , F _ { p h y s } ( u , b , g ) ) } \\ & { } \\ & { \qquad \leq \epsilon _ { H , m } ( c ) . } \end{array}\tag{7}
$$

The harness access gap is the diference between the executable and physical target floors. The selected command then satisfies

$$
\begin{array} { c } { { \delta _ { H } ( c ) = L _ { H } ^ { * } ( c ) - L _ { p h y s } ^ { * } ( c ) \geq 0 , } } \\ { { L _ { p h y s } ^ { * } ( c ) \leq K ( H , m ; c ) } } \\ { { \leq L _ { p h y s } ^ { * } ( c ) + \delta _ { H } ( c ) + 2 \epsilon _ { H , m } ( c ) . } } \end{array}\tag{8}
$$

Apply the triangle inequality to the selected physical response, its prediction, and the target. Optimality in the predicted problem bounds the predicted distance by that of an optimal executable command. A second application of the triangle inequality gives the executable target floor plus twice the model approximation error. Substitution of the access-gap definition gives the upper bound in Eq. (8). Physical optimality over the full command domain gives the lower bound.

Equation (8) separates the efects of the physical configuration, command access, and response model. Changing the apparatus or geometry can change the attainable set. Extending command access can reduce the harness gap, and improving the model can reduce approximation error. The gap is zero when the application exposes the full physical command set. Observation capabilities afect the evidence used to learn and apply the response model.

CIELAB Euclidean distance supplies the metric structure for this derivation. Equation (7) defines a uniform error over the executable command domain; empirical prediction scores describe sampled responses.

Residuals and experimental design: For an observed optical response, the target error admits the vector decomposition

$$
\begin{array} { l } { r _ { t a r g e t } = \widehat { F } _ { m } ( u , b , g ) - q , } \\ { r _ { m o d e l } = y - \widehat { F } _ { m } ( u , b , g ) , } \\ { y - q = r _ { t a r g e t } + r _ { m o d e l } . } \end{array}\tag{9}
$$

The first residual is the predicted displacement from the target. The second is the diference between the measured and predicted response. A small predicted target residual with a large measured mismatch directs attention to the response model. Dependence on command, background, or target region suggests variables and representations to test.

Repeated references under fixed geometry characterize variation around a stimulus. Changing the background while holding the target probes background dependence; neighboring commands near a boundary probe local response and control authority. Comparisons across model families test which representation predicts the relevant optical behavior. When a comparison requires an unavailable stimulus or observation, the agents must revise the application or acquisition protocol before performing it.

b Harness capabilities  
aPhysical attainability  
![](images/7bc1f29664f85f99e4be8e57680464c1a9d3866b9f9ec9a022d3f9bf0adc9a04.jpg)

![](images/020a738ad5ec91e3855287126a5e3dd16b564e19a16ad312be4a11d309836cdf.jpg)

![](images/8319ab78c1eb97f7dec9f0d8e70ca39fe3f65f5c47d1f5554c2fad266890a9e2.jpg)  
Figure 4. Geometric interpretation of Eqs. (6)-(9) in schematic CIELAB slices, with $\mathbf { a } ^ { * }$ and $\mathbf { b } ^ { * }$ axes at fixed $\mathbf { L } ^ { \ast }$ . (a) The minimum target distance is the physical target floor, $\bf { L _ { p h y s } ^ { * } ( c ) }$ . (b) The restricted command set introduces the harness access gap, $\delta _ { \mathbf { H } } ( \mathbf { c } )$ . (c) The two residual vectors are $\mathbf { r _ { t a r g e t } }$ from target to prediction and $\mathbf { r _ { m o d e l } }$ from prediction to measurement. Equation (7) defines the uniform model approximation quantity, $\epsilon _ { \mathbf { H } , \mathbf { m } } ( \mathbf { c } )$ , over the executable domain.

## 6. Optical color-matching case study

Color matching and characterization: The case study seeks to match a desired color through the glasses while background light enters the viewing path. It uses measured spectra to update the display setting and to revise the response model that selects settings for later tests.

We calculate color from the measured spectrum and report Euclidean distance in CIELAB as $\Delta E _ { 7 6 } ^ { * }$ , using reference white XYZ = (95.047, 100, 108.883) [14]. Smaller values indicate closer colorimetric matches. Presentation and viewing conditions determine the visual interpretation of these diferences [15]. The characterization data contain 2,170 stimulus conditions and 356 additional control measurements for learning the response to display settings and backgrounds.

The corpus combines 320 direct iPad calibrations (64 patterns at five brightness settings) with 1,850 combined-platform conditions: 320 transmission, 960 white-background/display, 378 coloredbackground/display, and 192 background-occluded display-only measurements. The host acquired the combined conditions and all 356 controls through $H _ { 1 } { \mathrm { : } }$ ’s acknowledged HOLD/BLACK interface; the 960-condition block, for example, crosses five iPad brightness settings, three glasses brightness settings, and 64 patterns.

Harness establishment and a revised stimulus protocol: The color-matching loop requires model-generated RGB settings to remain displayed throughout an exposure-dependent spectrometer acquisition. The glasses application accepts arbitrary RGB HOLD commands through a USB/ADB entry point. It retains the requested color and reports the command token, session, eye selection, and draw acknowledgements. The host coordinates acquisition with this held state and stores its readback with the spectrum.

The transition from $H _ { 0 }$ to $H _ { 1 }$ adds a command receiver that is absent from the baseline binary and present in the updated application. Validation covers all 64 named stimulus commands and passes 12 live device-control tests. Three framebufer checks each sample 384 pixels and find zero RGB mismatches. Each spectrum in the feedback example retains a custom RGB HOLD state with both software views acknowledged.

An acknowledged hold lets a fitted model request a color and measure its physical response over the acquisition interval. The retained revision record includes the added receiver, source fingerprints, device-check results, and command states stored with the measurements (Fig. 5c).

At fixed eye, patch geometry, and brightness, the revised interface expands catalog-based playback to the full digital RGB domain, $\mathcal { U } ( H _ { 0 } ) \subset \mathcal { U } ( H _ { 1 } ) = \{ 0 , \dotsc , 2 5 5 \} ^ { 3 } = \mathcal { U } _ { p h y s }$ . Equation (8) then gives $\delta _ { H _ { 1 } } ( c ) = 0$ for RGB command access. Inverse-model outputs are rounded to the device grid and held for measurement.

Feedback-guided color matching: For one fixed target, the initial display setting gives a color diference of 19.756 $\Delta E _ { 7 6 } ^ { * }$ . Two feedback-guided updates reduce it to 5.315 and 1.211 $\Delta E _ { 7 6 } ^ { * }$ , a reduction of 93.9% (Fig. 5a,b).

The reference and updated spectra have blue peaks near 449 and 460-461 nm, consistent with diferent background and virtualdisplay primaries. Their combination gives a near-colorimetric match of 1.211 $\Delta E _ { 7 6 } ^ { * }$ through metameric compensation in the specified colorimetric system [16].

In a separate five-target comparison, the median color diference falls from 9.377 to 1.882 $\Delta E _ { 7 6 } ^ { * } .$ . All five targets improve.

Response-model revision: The response model predicts color from the display setting and background, then supports inverse selection of a setting for a desired color. Revising this model changes the rule used to select settings in subsequent tests.

We compare three response families on the expanded calibration set: a diagonal XYZ afine model fitted in XYZ, the same parameterization fitted by minimizing fixed-reference CIELAB residuals, and an extension with bounded channel-wise quadratic terms fitted with the CIELAB objective. Each family produces 68 predictions with whole target-color groups excluded from its fit. Prediction RMSE values are 1.842, 1.363, and 1.444 $\Delta E _ { 7 6 } ^ { * } ,$ , respectively. We select the CIELAB-objective afine model, refit it on the expanded evidence, and acquire new spectra for the settings it generates.

(c) Documented software and protocol revision  
![](images/02a47f82126eed81f1fc5aef5887a5f987d5d2ccd0dd43b28d06a1560421b559.jpg)

![](images/37ffc45edea016785264b4a79a26e8ea2074659be4a4556deb4b06de387dc2ac.jpg)

![](images/19e1b5ddab41e88934366ac310d50947ac9003aa7ece306db4b7ea89e7ae3d67.jpg)

![](images/467873d5ec9b6f7f2fbb9a23204f4bb6afddb60cc4642f45da0dbefa6ff5fc3b.jpg)  
Update 0.2.0 adds a command entry point; checks and measurement states document its use.  
Figure 5. Physical measurements and a documented harness revision. (a) Four calibrated combined-light spectra: the target reference under the neutral background and the initial, first-updated, and second-updated settings under the changed background. The original W/(cm<sup>2</sup> sr nm) calibration is converted to mW/(m<sup>2</sup> sr nm). (b) Left: the measured three-setting trajectory for one target; right: independent before-and-after measurements for five desired colors. Smaller $\pmb { \Delta } \mathbf { E } _ { 7 6 } ^ { \ast }$ indicates a closer color match. (c) The command-interface revision, its device checks, and its subsequent use for spectral acquisition.

The previous and updated models each select settings for three target colors under three backgrounds, giving nine matched tests and 18 spectra. The updated model gives closer matches in seven tests and larger diferences in two. The previous model has a median physical diference of 1.342 $\Delta E _ { 7 6 } ^ { * }$ . The median decrease across paired tests is 0.197 $\Delta E _ { 7 6 } ^ { * }$ . This comparison measures calibration refinement across the matched contexts; Section 6.3 measures feedback adjustment of the display command.

## 7. Discussion

Stimulus and acquisition applications are research outputs that determine which comparisons agents can execute and which responses they can observe. Equation (8) separates their optical efects: apparatus changes afect physical attainability, application changes affect command access, and model revisions afect prediction error. A measured discrepancy can therefore direct attention to a specific part of the research environment.

A persistent goal allows the investigation to revise its plan, applications, and response models without changing its intended outcome. Progress monitoring evaluates those revisions against unresolved parts of the objective and the evidence needed to resolve them.

Discrepancies can expose assumptions in the optical model or the harness. A structured residual may reveal background dependence. A mismatch between requested and observed stimuli can identify a device state that needs readback. Dependence on acquisition conditions can require a diferent measurement or analysis protocol. Agents turn these observations into hypotheses and revisions, then acquire the physical comparisons needed to assess them.

Further work can examine transfer between goals, coordination among development agents, and the choice between acquiring another measurement and developing a new capability.

## 8. Conclusion

Goal-driven self-recursive optical research uses physical evidence to revise a goal-specific hardware and software harness. The reported optical see-through color-matching case study demonstrates expanded command access, spectral feedback, and response-model refinement. The optical formulation distinguishes physical attainability, command access, and model approximation error. This configuration illustrates a research structure that can be instantiated with other stimulus systems and measurement devices. The resulting applications and protocols remain available for subsequent investigations.

Table 2. Optical evidence and its role in goal-driven recursion. Color diference is measured in $\mathbf { \Delta } \Delta \mathbf { E } _ { 7 6 } ^ { * } ;$ smaller values indicate a closer match to the desired color.
<table><tr><td rowspan=1 colspan=1>Scientific question</td><td rowspan=1 colspan=1>Measured outcome</td><td rowspan=1 colspan=1>Role in the research process</td></tr><tr><td rowspan=1 colspan=1>Can a new commandinterface enable the opticaltest?</td><td rowspan=1 colspan=1>Arbitrary RGB holds, 64 checkedcommands, and 12 passed livecontrol tests</td><td rowspan=1 colspan=1>Software and protocol development make model-generated settingsexecutable and observable.</td></tr><tr><td rowspan=1 colspan=1>Can feedback improve onedisplayed color?</td><td rowspan=1 colspan=1>Color difference 19.756 → 1.211after two updates; 93.9% reduction</td><td rowspan=1 colspan=1>An observation changes the next physical action while the targetremains fixed.</td></tr><tr><td rowspan=1 colspan=1>Can the approach supportseveral desired colors?</td><td rowspan=1 colspan=1>For five colors, median difference9.377 → 1.882; all five improve</td><td rowspan=1 colspan=1>A shared control and measurement environment supports a familyof optical targets.</td></tr><tr><td rowspan=1 colspan=1>Can a revised modelimprove later tests?</td><td rowspan=1 colspan=1>The updated model gives closermatches in seven of nine newphysical tests.</td><td rowspan=1 colspan=1>Evidence changes the reusable relationship that generatessubsequent display settings.</td></tr></table>

## 9. Code availability

The case-study software is available in the SpectralColorLab repository: https://github.com/TZHU64/SpectralColorLab.

## 10. References

1. Tom G, Schmid SP, Baird SG, Cao Y, Darvish K, Hao H, et al. Self-driving laboratories for chemistry and materials science. Chem Rev. 2024;124(16):9633-732. doi:10.1021/acs.chemrev.4c00055.

2. Burger B, Mafettone PM, Gusev VV, Aitchison CM, Bai Y, Wang X, et al. A mobile robotic chemist. Nature. 2020;583(7815):237-41. doi:10.1038/s41586-020-2442-2.

3. Boiko DA, MacKnight R, Kline B, Gomes G. Autonomous chemical research with large language models. Nature. 2023;624(7992):570-78. doi:10.1038/s41586-023-06792-0.

4. Bran AM, Cox S, Schilter O, Baldassari C, White AD, Schwaller P. Augmenting large language models with chemistry tools. Nat Mach Intell. 2024;6(5):525-35. doi:10.1038/s42256-024-00832-8.

5. Xiong J, Hsiang EL, He Z, Zhan T, Wu ST. Augmented reality and virtual reality displays: emerging technologies and future perspectives. Light Sci Appl. 2021;10(1):216. doi:10.1038/s41377-021-00658-8.

6. Gabbard JL, Swan JE II, Zedlitz J, Winchester WW III. More than meets the eye: an engineering study to empirically examine the blending of real and virtual color spaces. In: 2010 IEEE Virtual Reality Conference; 2010 Mar 20-24; Waltham, MA, USA. Piscataway (NJ): IEEE; 2010. p. 79-86. doi:10.1109/VR.2010.5444808.

7. Sim M, Vakili MG, Strieth-Kalthof F, Hao H, Hickman RJ, Miret S, et al. ChemOS 2.0: an orchestration architecture for chemical self-driving laboratories. Matter. 2024;7(9):2959-77.

doi:10.1016/j.matt.2024.04.022.

8. Bai J, Mosbach S, Taylor CJ, Karan D, Lee KF, Rihm SD, et al. A dynamic knowledge graph approach to distributed self-driving laboratories. Nat Commun. 2024;15(1):462. doi:10.1038/s41467-023-44599-9.

9. Prince MH, Chan H, Vriza A, Zhou T, Sastry VK, Luo Y, et al. Opportunities for retrieval and tool augmented large language models in scientific facilities. NPJ Comput Mater. 2024;10(1):251. doi:10.1038/s41524-024-01423-2.

10. Vriza A, Prince MH, Zhou T, Chan H, Cherukara MJ. Operating advanced scientific instruments with AI agents that learn on the job. NPJ Comput Mater. 2026;12(1):160. doi:10.1038/s41524-026-02005-0.

11. Desai S, Addamane S, Tsao JY, Brener I, Swiler LP, Dingreville R, et al. AutoSciLab: a self-driving laboratory for interpretable scientific discovery. Proc AAAI Conf Artif Intell. 2025;39(1):146-54. doi:10.1609/aaai.v39i1.31990.

12. Volk AA, Abolhasani M. Performance metrics to unleash the power of self-driving labs in chemistry and materials science. Nat Commun. 2024;15(1):1378. doi:10.1038/s41467-024-45569-5.

13. Instrument Systems. TOP 300 AR/VR optical probe [Internet]. Version 1.2. Munich: Instrument Systems; [date unknown] [cited 2026 Oct 23]. Available from: https://www.instrumentsystems.com/fileadmin/Downloads/ Brochures\_Datasheets/TOP300\_EN.pdf.

14. International Commission on Illumination. Colorimetry. 4th ed. Vienna: CIE; 2018. Report No.: CIE 015:2018. doi:10.25039/TR.015.2018.

15. International Commission on Illumination. Methods for evaluating colour diferences in images. Vienna: CIE; 2011. Report No.: CIE 199:2011.

16. International Commission on Illumination. International lighting vocabulary. 2nd ed. Vienna: CIE; 2020. Standard No.: CIE S 017:2020. Entry 17-23-006, Metamerism.