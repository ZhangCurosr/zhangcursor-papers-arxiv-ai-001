# AI for Science with GPT-6 Astra: Thermal Design and Electrothermal Analysis of 2D CFET

Min-Hui Kim<sup>1</sup>, Khushi Sharma<sup>2</sup>, Sarah Zhang<sup>3</sup>, Ye Wang<sup>4\*</sup>

<sup>1</sup>Graduate School of Semiconductor Materials and Devices,

Ulsan National Institute of Science and Technology (UNIST), Ulsan 44919, Republic of Korea

<sup>2</sup>Materials Science and Engineering, National University of Singapore, Singapore

<sup>3</sup>Materials Science and Engineering, Cornell University, Ithaca, NY, USA

<sup>4</sup>Department of Applied Physics and Science Education,

Technische Universiteit Eindhoven, Eindhoven, The Netherlands

Corresponding author: y.wang19@tue.nl

The first three authors are listed alphabetically by surname.

This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible.

Abstract—Thermal optimization of 2D CFET inverters requires testing structural proposals against their electrical costs. We examine these research tasks using an AI agent workflow within a supplied electrothermal model. At 12 nm, Astra selects a redistributed source-interconnect geometry, while a coordinating agent proposes a substrate-directed heat-removal path. The combined design reduces peak temperature rise by 1.67 K at fixed metal volume and 20 µW. A subsequent metal-resistance sensitivity gives about 0.6-K inverter cooling alongside a 2% nFET on-current loss. Effective contact-length scaling further shows that lower temperature can accompany higher thermal resistance when current falls. Reproduction identifies agreeing implementations and retains a 104.95-K failure for diagnosis. These results show that an AI scientist workflow can propose thermal structures, test them under common constraints, and quantify their electrical cost.

Index Terms—GPT-6, AI for science, two-dimensional materials, 2D CFET, electrothermal modeling, thermal management.<sup>[</sup>

## I. INTRODUCTION

Two-dimensional channels support strong electrostatic control and dense complementary field-effect transistor (CFET) integration [1]. Stacking n- and p-channel devices concentrates heat within one footprint, where dielectrics and thermal boundary resistances hinder its removal [2]–[4]. In an inverter, temperature also changes transistor currents and redistributes power. Thermal designs must therefore be evaluated under circuit operation as well as a prescribed heat load.

Finding that design requires deciding which geometry to test, how to compare candidates fairly, and what to investigate next. Researchers ordinarily make these decisions when using TCAD solvers [5]. AI-scientist frameworks that connect hypothesis generation to code execution [6] motivate testing whether agents can undertake such tasks in CMOS logic research.

Within a supplied Python model and fixed metal budget, ${ \mathrm { A s } } -$ tra searches interconnect geometries and a coordinating agent proposes a via extension. We evaluate the resulting inverter, distinguish the agents’ contributions through search records, and check implementation consistency through reproduction.

![](images/3cb79e59a2bb93a0328f62630788ac0755e10e18675aee38153622f034be0b7f.jpg)  
Fig. 1. Agent workflow within a supplied model. Astra plans the interconnect search; the coordinating agent interprets the trend and proposes the via extension. Tool calls execute the tests; separate reruns check code consistency.

## II. METHODOLOGY AND DEVICE MODEL

## A. Thermal Model and Electrical Loading

The search uses four $\mathbf { M o S } _ { 2 }$ sheets above four ${ \mathrm { { W S e } _ { 2 } } }$ sheets $( L _ { g } = 1 2$ nm, width 40 nm, heated length 16 nm; Table III). Source local interconnects join each tier’s sheet contacts; their widths vary while contact length stays fixed. Channel crossplane, metal and dielectric conductivities are 2, 25 and 1.2 $\mathrm { { \dot { W } m ^ { - 1 } K ^ { - 1 } } }$

We solve anisotropic steady-state heat conduction by finite volumes, including thermal boundary resistance. The bottom is fixed at 300 K; the top exchanges heat with a 300-K bath (Table III). Fixed-power tests use 20 $\mu \mathrm { W } ,$ , 30% contact dissipation and $R _ { c m } = 5 ~ \mathrm { m ^ { 2 } K / G W }$ , with constant conductivities and no interconnect Joule heating. Energy balance, source normalization and reciprocity are checked. Refining the original fixed-power comparison changes its improvement by 0.004 $\mathrm { K } ;$ monolayers remain one cell thick.

To test circuit operation, we instead determine power from the inverter currents at $V _ { D D } = 0 . 7 ~ \mathrm { V } .$ Gate lengths are 8, 12 and 32 nm; follow-up width/contact scans cover all three. Both gates share an input, and their interconnected drains form the output. We solve

(a) Selected device temperature  
![](images/160d67ebe9bf4dcf989a820827fdca1919e8c1da09a744de875c7ab11fd26afc.jpg)  
1 n channel (MoS<sub>2</sub>)

(b) Width at fixed position  
![](images/135de489529a2696ad1ae30e8a614392ef693c57c146490f7d72059098331581.jpg)  
L<sub>g</sub> = 12 nm; L<sub>c</sub> = 20 nm  
2 p channel (WSe<sub>2</sub>)  
6 Gate dielectric  
7 Inter-tier dielectric  
3 Source contact

(c) Contact-length sensitivity  
![](images/78dbfdb6b2f007eaec1bdadbb9ed257c057443759bf10c1c74f34a093b5ff1c0.jpg)  
8 Source interconnect  
4 Drain contact  
5 Gate metal  
9 Source via  
10 Substrate  
Fig. 2. Thermal design and electrical cost. (a) Selected device at 20 µW, $P _ { n } / P _ { p } = 2$ $R _ { c m } = 5 $ m<sup>2</sup>K/GW. At $L _ { q } = 1 2 \ \mathrm { n m } , ( \mathrm { b } )$ upper width varies at fixed metal volume or fixed lower width; (c) effective contact length varies. Changes are relative to $w _ { n } = w _ { p } = 8$ nm, $L _ { c } = 2 0$ nm. $R _ { t h , p e a k }$ uses fixed 20 $\mu \mathrm { W } ;$ self-heated $I _ { o n , n }$ includes metal resistance at $V _ { G S } = V _ { D S } = 0 . 7 \mathrm { ~ V } .$

$$
I _ { n } ( V _ { i n } , V _ { o u t } , T _ { n } ) = I _ { p } ( V _ { D D } - V _ { i n } , V _ { D D } - V _ { o u t } , T _ { p } )\tag{1}
$$

together with $\textbf { T } = ~ 3 0 0 ~ \mathrm { K } + \mathbf { H } \mathbf { p }$ , where H maps channel/contact powers to channel-average temperatures. The charge-sheet current model accounts for finite oxide thickness [10], [11], mobility proportional to $T ^ { - 1 . 3 5 }$ and saturation velocity proportional to $T ^ { - 0 . 5 }$ . The n/p width-normalized contact resistances are 500/800 Ω µm per contact. Their voltage drops determine contact heating, replacing the prescribed 30% fraction; current balance gives total power $P _ { D C } = V _ { D D } I _ { \cdot }$

A follow-up sensitivity adds finite-volume metal conduction, voltage drops and Joule heating, assuming $\rho _ { m }$ = $2 . 9 3 ~ \times ~ 1 0 ^ { - 7 }$ Ωm. Effective contact-length scaling uses $R _ { c } ( L _ { c } ) / R _ { c } ( L _ { 0 } ) \ = \ \operatorname { t a n h } ( L _ { 0 } / L _ { T } ) / \operatorname { t a n h } ( L _ { c } / L _ { T } )$ [17], with $L _ { 0 } ~ = ~ L _ { T } ~ = ~ 2 0$ nm and the above reference resistances. Injection beneath a top contact is not resolved. Width scans compare fixed metal volume with fixed lower width; contactlength scans change footprint and metal volume.

## B. Agent Roles and Structural Search

Astra tests whether widening the hotter upper source interconnect and moving it inward improves cooling, using the requested Ultra setting (Fig. 1). The objective $J ( d ) =$ $\operatorname* { m a x } _ { r \in \{ 0 . 5 , 1 , 2 \} } \Delta T _ { \operatorname* { m a x } } ( d , r )$ minimizes the worst peak rise above 300 K over power ratios $r = P _ { n } / P _ { p }$ . All candidates preserve the same metal volume, top metal area, heat sources and thermal boundaries.

The coordinating agent identifies the reversed width trend and proposes substrate-directed vias. No equivalent Astra acknowledgement is documented, so its sweep is distinguished from this interpretation. Design families were specified before evaluation and candidate results retained; the search was adaptive, not blind.

TABLE I  
THERMAL PERFORMANCE OF DESIGN STAGES ON A COMMON MESH
<table><tr><td>Structure</td><td> $\overline { { \Delta T _ { \mathrm { m a x } } } }$  (K)</td><td>Reduction (K)</td></tr><tr><td>Baseline</td><td>141.63</td><td>0.00</td></tr><tr><td>Redistributed interconnect</td><td>140.75</td><td>0.88</td></tr><tr><td>Substrate-directed path</td><td>140.83</td><td>0.80</td></tr><tr><td>Combined</td><td>139.96</td><td>1.67</td></tr></table>

n-on-p; $P = 2 0 ~ \mu \mathrm { W } , \ : R _ { c m } = 5 ~ \mathrm { m ^ { 2 } K / G W } ;$ 43,493 cells. Metal volume, top area, heat sources and boundary conditions are fixed. All four peak at $r = 2$ over $r = 0 . 5 , 1 , 2$

## III. RESULTS AND DISCUSSION

## A. Source Interconnects and Heat Flow

The selected design narrows and repositions the upper source interconnect while reallocating the saved metal to the lower tier. Peak rise falls by 0.88 K (Table I), contradicting the initial widening hypothesis. Width and position change together, while semiconductor contacts remain fixed.

To improve downward heat flow, the coordinating agent adds a narrow via directed toward the substrate [Fig. 2(a)]. With the interconnect change and the same metal budget, peak rise falls from 141.63 to 139.96 K. The 1.67-K (1.18%) reduction exceeds either separate change and persists under refinement.

The fixed-position control distinguishes narrowing from redistribution [Fig. 2(b)]. Changing an 8/8-nm upper/lower pair to 4/12 nm reduces thermal resistance by 0.26%, but narrowing only the upper interconnect raises it by 0.33%. The benefit therefore depends on metal allocation. Contact scaling further separates temperature from performance [Fig. 2(c)]: shortening $L _ { c }$ from 20 to 8 nm raises thermal resistance by 4.2%, yet lowers inverter peak temperature by 6.2 K as nFET on-current falls by 17%. Lower temperature alone is not evidence of better heat removal.

(a) Voltage transfer characteristic  
![](images/a687d004610bb93967da2eb50c3dc25417574949bad03ee84f7cb1bd64b38021.jpg)

(c) Peak lattice temperature  
![](images/ae031258295eb627db142d4fa24a43c7f0db865d09eff49c991b78a89aec1821.jpg)

![](images/773b3d205748ec6ba1ad42f1d628e44aca12de363c8cb8ff24ab9a43b53b1b27.jpg)

(d) Bias-dependent hotspot, 12 nm  
![](images/c83adff28d49d6a92f71521dfd88332af951739d7b51ad373c56cc99e1e8f50e.jpg)  
S / D: source / drain contacts  
Fig. 3. DC inverter response without metal resistance at $V _ { D D } = 0 . 7 \mathrm { ~ V } .$ (a) Voltage transfer characteristics, (b) DC power and (c) peak lattice temperature for baseline interconnects. Colors denote gate length; solid curves include self-heating, dashed curves are isothermal at 300 K. (d) Selected 12-nm device sections at $y = 0$ under two static biases, with a common temperature scale and emphasized channels.

## B. CFET Inverter Electrothermal Operation

The fixed-power result establishes improved heat removal. To test whether it persists in an inverter, we let the transistor currents and temperatures determine the heat load together. At $L _ { g } = 8$ nm and $V _ { g s } = V _ { d s } = 0 . 7 ~ \mathrm { V } ,$ self-heating lowers nFET on-current from 245.5 to 208.6 $\mu \mathrm { A } / \mu \mathrm { m }$ (15.0%); the pFET loses 11.5% under corresponding bias. Currents use the summed sheet width of 4 × 40 nm. These unequal responses shift the inverter switching voltage $( V _ { i n } = V _ { o u t } )$ from 0.3211 to 0.3190 V and reduce the voltage-gain magnitude there from 83.4 to 52.1 [Fig. 3(a)].

Near the transition, however, thermal broadening of carrier occupation competes with mobility loss, raising peak DC power from 3.62 to $3 . 7 3 ~ \mu \mathrm { W }$ at 8 nm [Fig. 3(b)]. Temperature peaks at a different input voltage: $V _ { i n } = 0 . 3 1 0 2 \mathrm { ~ V } ,$ versus 0.3163 V for power. There, $V _ { o u t } = 0$ .561 V places most of the voltage drop across the upper nFET, whose mean temperature reaches 340.7 K while the pFET remains at 309.1 K. Thus total power alone does not determine the hotspot. Maximum temperature decreases from 341.4 to 312.1 K between 8 and 32 nm [Fig. 3(c)].

At 12 nm, the combined design lowers maximum inverter temperature from 330.75 to 330.41 K. This smaller 0.335-K reduction reflects the baseline’s $2 . 9 7 – \mu \mathrm { W }$ load at its temperature maximum. Rescaling the fixed-power source distribution gives 0.248-K cooling; using the inverter’s channel/contact fractions gives 0.329 K. Self-consistent feedback and the shift in peak bias bring this to 0.335 K. Relative to temperature rise, the benefit is 1.09%, close to the fixed-power 1.18%.

TABLE II  
CROSS-AGENT REPRODUCTION OF THE FIXED THERMAL REFERENCE CASES
<table><tr><td>Model / effort label</td><td>eT (K)</td><td>eC (pp)</td><td>Status</td></tr><tr><td>GPT-6 / Lighta GPT-6  $\mathbf { A s t r a } / \mathbf { U l t r a } ^ { b }$ </td><td>0 (ref.) 1.34e-12</td><td>0 (ref.) 3.20e-13</td><td>Reference Pass</td></tr><tr><td>Claude Fable 5.1 / high</td><td>4.26e-13</td><td>2.42e-13</td><td>Pass</td></tr><tr><td>Claude Opus 5 / high (A)</td><td>4.98e-10</td><td>2.88e-10</td><td>Pass</td></tr><tr><td>Claude Opus 5 (B)</td><td>4.26e-13</td><td>2.42e-13</td><td>Pass</td></tr><tr><td>GPT-5.6 Sol / max</td><td>4.98e-10</td><td>2.88e-10</td><td>Pass</td></tr><tr><td>GPT-5.6 Terra / max†</td><td>1.39e-12</td><td>4.12e-13</td><td></td></tr><tr><td>GPT-5 / Codex</td><td>1.08e-12</td><td></td><td>Pass</td></tr><tr><td>Claude Haiku 4.5</td><td></td><td>8.74e-13</td><td>Pass</td></tr><tr><td></td><td>104.95</td><td>15.16</td><td>Fail</td></tr><tr><td>Claude Opus 4.8 / middle*</td><td>4.26e-13</td><td>2.42e-13</td><td>Pass</td></tr><tr><td>Claude Sonnet 5 / middle*</td><td>1.08e-12</td><td>8.74e-13</td><td>Pass</td></tr><tr><td>GPT-5.6 Terra / max*</td><td>1.08e-12</td><td>8.74e-13</td><td>Pass</td></tr><tr><td>GPT-5.6 Luna / max*</td><td>7.96e-13</td><td>4.12e-13</td><td>Pass</td></tr><tr><td>Kimi / basic high*</td><td>7.96e-13</td><td>4.12e-13</td><td>Pass</td></tr><tr><td>Perplexity [Grok 4.6]*</td><td>1.08e-12</td><td>8.74e-13</td><td>Pass</td></tr><tr><td>Grok-labelled table</td><td>4.59e-04</td><td>1.09e-04</td><td>Table only</td></tr></table>

e<sub>T</sub>: maximum peak-rise error (four cases); e<sub>C</sub>: maximum reduction error (two resistances). Pass requires $e _ { T } \leq 0 . 0 1 $ K, $e _ { C } \leq 0 . 0 1 $ pp, residual/energy error < $1 0 ^ { - 7 }$ and rerun agreement. <sup>a</sup>Reported setting. <sup>b</sup>Requested setting; reference hidden before freezing. <sup>∗</sup>Unverified archive label. <sup>†</sup>Prior Sol summary visible. <sup>‡</sup>No code evidence. A/B: separate submissions. Values include rounding; model/effort comparisons are uncontrolled.

The hotspot moves between tiers with bias [Fig. 3(d)]. Including metal resistance gives 0.6-K cooling and 2.2% lower nFET on-current for the combined 12-nm design. The accompanying width/contact scans show the same trade-off at 8 and 32 nm. Assumed $L _ { T }$ values of 5–50 nm give shortcontact current penalties of 2–23%.

Separate 1–5 nm Green-function calculations [12]–[16], discussed alongside sub-nanometre gate demonstrations [7]– [9] in the additional analysis, are not coupled to this inverter model.

## C. Code Reproduction and Alternative Designs

Four fixed thermal cases test code consistency (Table II). Passing submissions agree within $5 \times 1 0 ^ { - 1 0 } ~ \mathrm { K } ;$ fresh Ultra code agrees within $1 . 3 4 \times 1 0 ^ { - 1 2 } \mathrm { ~ K ~ }$ . Haiku-labelled code fails by 104.95 K and 15.16 percentage points owing to indexing and resistance-unit errors. Corrections restore agreement; we retain the original failure. Supplied model/effort labels do not establish a controlled effort comparison.

The separate Fable 5.1 C4 design uses one sheet per tier, doubles interface conductances, introduces AlN, lengthens contacts and adds gate metal (Table III). Corrected, refined mean thermal resistance falls by 45.19%/40.37% in the top/bottom tiers with 26.12% more metal. These material/interface changes extend beyond our search; different loads and metrics prevent ranking the agents.

Cross-implementation checks recover the 1.67255-K reduction with the other study’s matrix assembly, despite shared discretization and backend. Neither search has been independently repeated or experimentally validated.

## IV. CONCLUSION

We used an Astra-based AI scientist workflow to design and evaluate 2D CFET thermal structures. At fixed metal volume $( L _ { g } ~ = ~ 1 2 ~ \mathrm { { \ n m } }$ $L _ { c } ~ = ~ 2 0 ~ \mathrm { \ n m } )$ , the best tested design redistributes source-interconnect metal between the tiers and introduces a substrate-directed heat-removal path. It lowers peak temperature by 1.67 K at 20 µW. Including metal resistance gives 0.6-K inverter cooling with 2.2% lower nFET on-current, revealing the electrical cost of improved heat removal. The workflow therefore produced a testable structural hypothesis and evaluated its circuit-level trade-off; experimental validation remains necessary.

## ACKNOWLEDGMENT

We acknowledge Vina Faramarzi (ASML Netherlands B.V.), who conceived the research idea together with Ye Wang. Part of this project originated in the Eindhoven Semiconductor Summer School 2026, with participation by Min-Hui Kim, Prasanna Prasad Mahajan, Aleksander Ogonowski, Michael Scholl, Khushi Sharma, Viren Sharma, Sarah Zhang and Yichen Zou. We thank Prof. Shihab Al-Daffaie of TU/e for coordinating the summer school and project work. This work was supported by the Technische Universiteit Eindhoven startup grant RF204648 and the Dutch Research Council (NWO), grant no. EINF-19934.

## REFERENCES

[1] M. M. Islam et al., “Challenges and prospects of 2D electronics for future monolithic complementary field-effect transistors,” Nat. Commun., vol. 17, Art. no. 3586, 2026. doi:10.1038/s41467-026-71986-9.

TABLE III  
DEVICE PARAMETERS FOR THE TWO DESIGN STUDIES
<table><tr><td>Parameter</td><td>GPT study</td><td>Claude baseline → C4</td></tr><tr><td>Sheets per tier (total)</td><td>4 (8)</td><td>1 (2)</td></tr><tr><td> $L _ { g } , L _ { h } , W \ ( \mathrm { n m } )$ </td><td>12, 16, 40</td><td>20, 30, 50</td></tr><tr><td> $t _ { n } , t _ { p } \ ( \mathrm { n m } )$ </td><td>0.65, 0.70</td><td>0.65, 0.65</td></tr><tr><td>Gate dielectric (nm)</td><td>1</td><td> $3 \ \mathrm { ( H f O _ { 2 } ) }$ </td></tr><tr><td>Within-tier gap (nm)</td><td>6</td><td>Not applicable</td></tr><tr><td>Inter-tier gap (nm)</td><td>20</td><td>26</td></tr><tr><td>Contact length (nm)</td><td>20</td><td>20 → 30</td></tr><tr><td>Substrate (nm)</td><td>40 (effective)</td><td>1000 (Si)</td></tr><tr><td>Domain (nm³)</td><td>96×80×125.4</td><td>300×200×1291.3</td></tr><tr><td>Sheet  $k _ { | | } ^ { n } , k _ { | | } ^ { p }$ </td><td>35,25</td><td>35, 10</td></tr><tr><td>Substrate  $k _ { \parallel } , k _ { \perp }$ </td><td>35,30</td><td>120, 120</td></tr><tr><td> $R _ { c m } ^ { n } , R _ { c m } ^ { p }$ </td><td>5,5</td><td> $( 4 0 , 5 0 )  ( 2 0 , 2 5 )$ </td></tr><tr><td>Channel-oxide  $\mathrm { T B R } ^ { n , p }$ </td><td>71, 71</td><td>(50, 55.56) → (25, 27.78)</td></tr><tr><td>Top h  $( \mathrm { M W \ m ^ { - 2 } K ^ { - 1 } } )$ </td><td>2</td><td>0 (adiabatic)</td></tr></table>

Arrows: baseline to candidate. Entries without arrows stay fixed within each study. $L _ { h } \colon$ channel plus access/spacer length. Gaps: sheet surface to surface. k: W/mK; TBR: thermal boundary resistance in m<sup>2</sup>K/GW. Both use n-on-p stacking, a 300-K bottom and adiabatic sides.

[2] T. Miao et al., “A novel thermal network model and electrothermal coupling study for NSFETs and CFETs considering thermal crosstalk,” Micro Nanostruct., vol. 208, Art. no. 208322, 2025. doi:10.1016/j.micrna.2025.208322.

[3] S. Shahin et al., “Self-heating and parasitic effects in multi-tier CFET design,” IEEE J. Explor. Solid-State Comput. Devices Circuits, vol. 12, pp. 153–158, 2026. doi:10.1109/JXCDC.2026.3704373.

[4] E. Yalon et al., “Energy dissipation in monolayer MoS<sub>2</sub> electronics,” Nano Lett., vol. 17, pp. 3429–3433, 2017. doi:10.1021/acs.nanolett.7b00252.

[5] Synopsys, “Sentaurus Device: Multidimensional (1D/2D/3D) device simulator.” Accessed Sep. 12, 2026. [Online]. Available: Synopsys TCAD.

[6] C. Lu et al., “The AI Scientist: Towards fully automated open-ended scientific discovery,” arXiv:2408.06292, 2024. doi:10.48550/arXiv.2408.06292.

[7] S. B. Desai et al., “MoS<sub>2</sub> transistors with 1-nanometer gate lengths,” Science, vol. 354, pp. 99–102, 2016. doi:10.1126/science.aah4698.

[8] F. Wu et al., “Vertical MoS<sub>2</sub> transistors with sub-1-nm gate lengths,” Nature, vol. 603, pp. 259–264, 2022. doi:10.1038/s41586-021-04323-3.

[9] M. Perucchini et al., “Physical insights into the operation of a 1-nm gate length transistor based on MoS with metallic carbon nanotube gate,” Appl. Phys. Lett., vol. 113, 183507, 2018. doi:10.1063/1.5054281.

[10] D. J. Frank, Y. Taur, and H.-S. P. Wong, “Generalized scale length for two-dimensional effects in MOSFETs,” IEEE Electron Device Lett., vol. 19, pp. 385–387, 1998. doi:10.1109/55.720194.

[11] C. Gilardi et al., “Extended Scale Length Theory for Low-Dimensional Field-Effect Transistors,” IEEE Trans. Electron Devices, vol. 69, pp. 5302–5309, 2022. doi:10.1109/TED.2022.3190464.

[12] M. P. Anantram, M. S. Lundstrom, and D. E. Nikonov, “Modeling of Nanoscale Devices,” Proc. IEEE, vol. 96, pp. 1511–1550, 2008. doi:10.1109/JPROC.2008.927355.

[13] D. Sanchez and L. Serra, “Thermoelectric transport of mesoscopic conductors coupled to voltage and thermal probes,” Phys. Rev. B, vol. 84, 201307(R), 2011. doi:10.1103/PhysRevB.84.201307.

[14] W. Lee et al., “Heat dissipation in atomic-scale junctions,” Nature, vol. 498, pp. 209–212, 2013. doi:10.1038/nature12183.

[15] A. Szabo, R. Rhyner, and M. Luisier, “Ab initio simulation of singleand few-layer MoS<sub>2</sub> transistors: Effect of electron-phonon scattering,” Phys. Rev. B, vol. 92, 035435, 2015. doi:10.1103/PhysRevB.92.035435.

[16] C. Stieger, A. Szabo, T. Bunjaku, and M. Luisier, “Ab-initio quantum transport simulation of self-heating in single-layer 2-D materials,” J. Appl. Phys., vol. 122, 045708, 2017. doi:10.1063/1.4990384.

[17] L. Hoang et al., “Understanding the impact of contact-induced strain on the electrical performance of monolayer WS<sub>2</sub> transistors,” Nano Lett., vol. 24, pp. 12768–12774, 2024. doi:10.1021/acs.nanolett.4c02616.