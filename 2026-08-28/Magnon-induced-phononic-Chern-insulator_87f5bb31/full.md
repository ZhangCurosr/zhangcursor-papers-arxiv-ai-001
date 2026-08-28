# Magnon-induced phononic Chern insulator

Rui-Chang Shen,<sup>1</sup> Yihao Yang,<sup>2,</sup> <sup>3,</sup> <sup>4,</sup> <sup>∗</sup> and Haoran Xue<sup>1,</sup> <sup>5,</sup> <sup>†</sup>

<sup>1</sup>Department of Physics, The Chinese University of Hong Kong, Shatin, Hong Kong SAR, China

<sup>2</sup>State Key Laboratory of Extreme Photonics and Instrumentation,

ZJU-Hangzhou Global Scientific and Technological Innovation Center, Zhejiang University, Hangzhou 310027, China. <sup>3</sup>The Electromagnetics Academy at Zhejiang University,

College of Information Science and Electronic Engineering, Zhejiang University, Hangzhou 310027, China. <sup>4</sup>Zhejiang Key Lab. of Intelligent Electromagnetic Control and Advanced Electronic Integration, Jinhua Institute of Zhejiang University, Zhejiang University, Jinhua 321099, China. <sup>5</sup>State Key Laboratory of Quantum Information Technologies and Materials, The Chinese University of Hong Kong, Shatin, Hong Kong SAR, China

High-frequency artificial phononic crystals ofer a low-loss platform compatible with on-chip integration, yet realizing Chern phononic phases at GHz frequencies remains challenging. Here, we propose a magnon-induced phononic Chern insulator in a honeycomb phononic crystal hybridized with ferromagnetic islands at the hexagon centers. A circularly polarized Kittel mode couples to the surrounding phonons with a phase winding, which breaks time-reversal symmetry and opens a full Chern gap. In the large-detuning regime, this mechanism leads to an efective Haldane-type phononic model with magnon-induced complex hopping. By tuning the magnon-phonon interaction, the full hybrid system accesses Chern phases with tunable Chern numbers |C| = 1 and |C| = 2. The predicted gaps can exceed realistic phonon and magnon linewidths, enabling their observation in GHz acoustic devices. Our work establishes chiral magnon–phonon hybridization as a route to magnetically reconfigurable topological phononics.

Introduction— Phonons, the quantized excitations of lattice vibrations, are fundamental carriers of thermal energy, mechanical signals, and coherent information [1]. The introduction of topological band theory into acoustic and elastic systems has given rise to topological phononics, enabling robust boundary transport and unconventional control of sound and mechanical waves [2– 14]. Most experimental realizations, however, operate at kilohertz-to-megahertz frequencies, where the comparatively long acoustic wavelengths limit miniaturization and integration. Gigahertz (GHz) phonons, by contrast, combine compact wavelengths and low dissipation with compatibility with integrated electromechanical architectures [15, 16]. They can furthermore interface coherently with superconducting qubits, spin excitations, microwave-frequency resonators, and ferromagnetic dynamics [17–21], making them promising information carriers for hybrid quantum and classical devices. To date, GHz topological-phononics experiments have primarily realized valley-Hall, crystalline, or pumpingbased phases [22–24]. Although such phases can suppress scattering under appropriate symmetry conditions, they cannot provide the strictly one-way boundary transport of a Chern insulator and thus remain vulnerable to symmetry-breaking disorders [25].

Realizing a phononic Chern phase requires breaking the time-reversal symmetry and opening a complete bulk gap. Existing approaches have employed gyroscopic motion [2, 8], circulating fluids [4–6, 26, 27], Floquet modulation [28, 29], and charge-doped piezomagnetic platforms [30]. These schemes have established the basic principles of acoustic Chern transport, but their reliance on moving media, continuous external driving, or material-specific responses presents substantial chal lenges for miniaturized high-frequency devices.

Magnons, the collective excitations of spin precession in ordered magnets, provide a natural microscopic source of time-reversal symmetry breaking. Their intrinsic GHzfrequency dynamics and magnetic tunability have made them important carriers of coherent information [31– 33]. Magnons couple to lattice vibrations through effects like magnetoelastic and magneto-rotation interactions [34, 35], and suitable symmetry breaking can convert these interactions into nonreciprocal acoustic responses [35–38].

Here, we propose a magnon-induced phononic Chern insulator in a honeycomb phononic crystal containing ferromagnetic islands. Unlike previous studies of topological magnon-phonon polaritons [39–43], our scheme incorporates magnon–phonon coupling into an artificial phononic-crystal platform, where magnons not merely serve as a constituent of a hybrid excitation, but as virtual mediators that imprint a synthetic gauge flux on a lattice and thereby produce a globally gapped phononic Chern phase. Specifically, due to the discrete rotational symmetry, the magnon couples to the surrounding phonons with a phase winding. This induces virtual complex phonon hoppings and an efective Haldane-type flux, opening a complete Chern gap with a tunable Chern number up to two. We also present the phase diagram of the model and analyse the gap size using realistic parameters, further paving the way for possible experiments.

Human–AI discovery workflow— The theoretical framework was developed through an AI scientificdiscovery workflow centered on the Qiushi Discovery Engine [44], as illustrated in Fig. 1(a). Starting from the open-ended objective of using magnons to realize a phononic Chern insulator, the engine autonomously constructed the initial lattice model, derived its efective Hamiltonian, explored candidate topological mechanisms, and performed closed-loop consistency checks and self-verification. Human researchers subsequently examined the physical assumptions, independently verified the analytical and numerical results, and formulated the final interpretation. This human–AI workflow therefore combined autonomous theoretical exploration with human scientific validation.

![](images/1844f679105e3f05e65e74cd3dd6f53dd77d89f42e08c7344b5a148617ca05e9.jpg)  
FIG. 1. (a) Schematic of Human-Agentic hybrid workflow. (b) A honeycomb phononic crystal with one phonon mode on each A and B sublattice site. A magnetic island at the plaquette center supports a right-handed circular Kittel mode. The six surrounding phononic sites are labeled in their physical counterclockwise order as $\mathrm { A _ { 1 } , B _ { 1 } , A _ { 2 } , B _ { 2 } , A _ { 3 } , B _ { 3 } }$ . Their radial vectors are $\begin{array} { r } { \pmb { d } _ { \alpha , i } = a _ { 0 } ( \cos \theta _ { \alpha , i } , } \end{array}$ sin $\theta _ { \alpha , i } )$ , with $\theta _ { \mathrm { A } , i } = 2 \pi ( i - 1 ) / 3 , \theta _ { \mathrm { B } , i } = ( 2 i - 1 ) \pi / 3 .$ and $a _ { 0 }$ is the distance from the plaquette center to each surrounding phononic site. (c) Momentum-space intensity of the magnon–phonon terms $| G _ { \mathrm { A } } ( \pmb { k } ) | ^ { 2 }$ and $| \bar { G _ { \mathrm { B } } } ( \pmb { k } ) | ^ { 2 }$ , showing that the magnon couples selectively to sublattice A at K and to sublattice B at $\mathrm { K } ^ { \prime }$ . (d) Schematic of the magnon-induced efective phonon hopping, showing $H \mathrm { { e f f } \mathrm { { } \mathrm { ~ - ~ } } } H \mathrm { { p h } }$ , where only the magnon-induced coupling between phonons is depicted. The coupling phase is illustrated in the right panel. Each NN bond receives virtual-magnon contributions from the center magnons of its two adjacent plaquettes.

Phonon-magnon model— We consider a honeycomb phononic crystal with one localized phonon mode on each A and B sublattice [Fig. 1(b)]. A ferromagnetic island at each plaquette center supports a uniform Kittel mode [45–47]. The system is governed by the Hamiltonian

$$
H = H _ { \mathrm { p h } } + H _ { \mathrm { m } } + H _ { \mathrm { i n t } } .\tag{1}
$$

Here, $H _ { \mathrm { p h } }$ is the phonon Hamiltonian given by

$$
\begin{array} { l } { { \displaystyle H _ { \mathrm { p h } } = \sum _ { r _ { \mathrm { A } } } \omega _ { \mathrm { p h } } a _ { r _ { \mathrm { A } } } ^ { \dagger } a _ { r _ { \mathrm { A } } } + \sum _ { r _ { \mathrm { B } } } \omega _ { \mathrm { p h } } b _ { r _ { \mathrm { B } } } ^ { \dagger } b _ { r _ { \mathrm { B } } } } } \\ { { ~ + ~ \displaystyle \sum _ { \left. r _ { \mathrm { A } } , r _ { \mathrm { B } } \right. } t \left( a _ { r _ { \mathrm { A } } } ^ { \dagger } b _ { r _ { \mathrm { B } } } + \mathrm { H . c . } \right) , } } \end{array}\tag{2}
$$

where $a _ { r _ { \mathrm { A / B } } } ^ { \dagger } \left( a _ { r _ { \mathrm { A / B } } } \right)$ and $b _ { r _ { \mathrm { A / B } } } ^ { \dagger } \left( b _ { r _ { \mathrm { A / B } } } \right)$ create (annihilate) a phonon mode at position $r _ { \mathrm { A } }$ and r<sub>B</sub>, respectively, $\omega _ { \mathrm { p h } }$ is the phonon frequency, t is the nearest-neighbor phonon hopping, and H.c. stands for the Hermitian conjugate. Due to the large spatial separation between the ferromagnetic islands, the magnon Hamiltonian $H _ { \mathrm { m } }$ only consists of on-site resonances:

$$
H _ { \mathrm { m } } = \sum _ { R } { \omega _ { \mathrm { m } } m _ { R } ^ { \dagger } m _ { R } } ,\tag{3}
$$

where $m _ { R } ^ { \dag } \ ( m _ { R } )$ are magnon creation (annihilation) operators at position R and $\omega _ { \mathrm { m } }$ is the magnon frequency. Without phonon-magnon interaction, the phononic crystal behaves like a phononic graphene with two Dirac points at the Brillouin corners (i.e., K and K<sup>′</sup> valleys), whereas the magnon dispersion is a flatband. This setup has the advantage that, once the Dirac points are properly gapped, a clean and full Chern gap can be easily obtained without further parameter optimization, and a low-energy description can directly capture the underlying mechanism.

The interaction term takes the form:

$$
\begin{array} { r l } {  { H _ { \mathrm { i n t } } = \sum _ { R } \sum _ { i = 1 } ^ { 3 } [ g \xi ^ { 2 ( i - 1 ) } a _ { R + d _ { \mathrm { A } , i } } ^ { \dagger } m _ { R }  } } \\ & {  + g \xi ^ { 2 i - 1 } b _ { R + d _ { \mathrm { B } , i } } ^ { \dagger } m _ { R } + \mathrm { H . c . } ] . } \end{array}\tag{4}
$$

Here, g is the magnitude of the magnon–phonon coupling and $\xi = e ^ { - i \pi / 3 }$ is the phase increment between adjacent radial directions. This phase factor originates from projecting the right-circular Kittel mode onto the radial efective fields associated with the six surrounding phononic sites (see Supplemental Material (SM), Sec. S1 for more details on the model construction [48]). The A and B sites are labeled in their physical counterclockwise order, with $\theta _ { \mathrm { A } , i } = 2 \pi ( i - 1 ) / 3$ and $\theta _ { \mathrm { B } , i } = ( 2 i - 1 ) \pi / 3$ For a fixed magnetization direction, these complex coupling phases generate the synthetic flux responsible for the Chern gap.

![](images/8fc4f834d9fb479f9a8f74b04746f2c2ad4273448c07749ab2e1974799f73ee8.jpg)  
FIG. 2. (a) Bulk band structures in the absence (gray dashed curves) and presence (colored solid curves) of the magnonphonon coupling, which explicitly opens a gap at the Dirac points. The lower, middle and upper bands have the Chern numbers $C = ( - 1 , ~ + 1 , ~ 0 )$ , respectively. (b) The corresponding strip spectrum for $g = 0 . 4$ , with periodic boundary conditions applied along ${ \pmb a } _ { 1 } = a _ { 0 } ( 3 , \sqrt { 3 } ) / 2$ . (c) Spatial distribution of the edge states, calculated as the sum of the probability of all localized modes within the bandgap. Parameters are set as $t = 1$ and $\delta _ { \omega } = 3 . 2 $

In the following, we consider a generic large-detuning scenario where $| \delta _ { \omega } | \equiv | \omega _ { \mathrm { m } } - \omega _ { \mathrm { p h } } | \gg g , t$ . In such a case, the magnon serves as an auxiliary degree of freedom to manipulate the phonon band topology, and the resulting topological transport remains phonon-dominated, as desired for phononic applications. In the Bloch basis $\Psi _ { \pmb { k } } = ( a _ { \pmb { k } \mathrm { A } } , a _ { \pmb { k } \mathrm { B } } , m _ { \pmb { k } } ) ^ { T }$ , the model Hamiltonian is [48]

$$
H ( \pmb { k } ) = \left( \begin{array} { c c c } { 0 } & { t f ( \pmb { k } ) } & { G _ { \mathrm { A } } ( \pmb { k } ) } \\ { t f ^ { \ast } ( \pmb { k } ) } & { 0 } & { G _ { \mathrm { B } } ( \pmb { k } ) } \\ { G _ { \mathrm { A } } ^ { \ast } ( \pmb { k } ) } & { G _ { \mathrm { B } } ^ { \ast } ( \pmb { k } ) } & { \delta _ { \omega } } \end{array} \right) ,\tag{5}
$$

where $\textbf { \textit { k } } = \mathbf { \beta } ( k _ { x } , k _ { y } )$ is the momentum and $\begin{array} { r l } { f ( k ) } & { { } = } \end{array}$ $\textstyle \sum _ { i = 1 } ^ { 3 } e ^ { i \pmb { k } \cdot \pmb { d } _ { \mathrm { A } , i } }$ . The magnon–phonon coupling terms are given by

$$
\begin{array} { l } { { \displaystyle { G _ { \mathrm { A } } ( \pmb k ) } = g \sum _ { i = 1 } ^ { 3 } \xi ^ { 2 ( i - 1 ) } e ^ { - i { \pmb k \cdot d } _ { \mathrm { A } , i } } , } } \\ { { \displaystyle { G _ { \mathrm { B } } ( \pmb k ) } = g \sum _ { i = 1 } ^ { 3 } \xi ^ { 2 i - 1 } e ^ { - i { \pmb k \cdot d } _ { \mathrm { B } , i } } . } } \end{array}\tag{6}
$$

At the two Dirac points, the phases sum constructively on one sublattice and destructively on the other: $| G _ { \mathrm { A } } ( { \bf K } ) | ^ { 2 } ~ = ~ 9 g ^ { 2 } , ~ | G _ { \mathrm { B } } ( { \bf K } ) | ^ { 2 } ~ = ~ 0 ;$ and $\begin{array} { r l } { | G _ { \mathrm { A } } ( \mathbf { K } ^ { \prime } ) | ^ { 2 } } & { = } \end{array}$ 0, $| G _ { \mathrm { B } } ( \mathbf { K } ^ { \prime } ) | ^ { 2 } = 9 g ^ { 2 }$ . Consequently, the magnon couples exclusively to sublattice A at K and to sublattice B at $\mathrm { K } ^ { \prime } .$ as plotted in Fig. 1(c). This valley–sublattice locking generates opposite Dirac masses at the two valleys $\left( m _ { \mathrm { K / K ^ { \prime } } } = \mp 9 g ^ { 2 } / 2 \delta _ { \omega } \right)$ , ensuring that their Berrycurvature contributions constructively add up to open a topological Chern gap (see SM Sec. S2 [48]). Interestingly, this mechanism closely mimics the gap-opening process in of-resonantly driven graphene, where photonassisted transitions similarly generate a valley-opposite Dirac mass [49–52].

Efective Haldane-like model— To gain a deeper insight into the role of the central magnon on the phononic crystal, we apply the Schriefer-Wolf transformation to derive the efective phonon Hamiltonian as [53, 54]:

$$
H _ { \mathrm { e f f } } = H _ { \mathrm { p h } } - \sum _ { R } \frac { V _ { R } ^ { \dagger } V _ { R } } { \delta _ { \omega } } ,\tag{7}
$$

where $\begin{array} { r } { V _ { R } = \sum _ { i = 1 } ^ { 3 } \left[ g \xi ^ { - 2 ( i - 1 ) } a _ { R + d _ { \mathrm { A } , i } } + g \xi ^ { - ( 2 i - 1 ) } b _ { R + d _ { \mathrm { B } , i } } \right] } \end{array}$ The magnon-phonon interaction generates four types of efective terms [see Fig. 1(d)]: an on-site energy renormalization, nearest-neighbor (NN) hoppings, next-nearest-neighbor (NNN) hoppings, and thirdnearest-neighbor (NNNN) hoppings. Specifically, the efective hopping amplitudes between two sites of the same sublattice around the same magnon center are

$$
t _ { \mathrm { A } _ { i }  \mathrm { A } _ { j } } ^ { \mathrm { e f f } } = - \frac { g ^ { 2 } } { \delta _ { \omega } } \xi ^ { 2 ( i - j ) } , \mathrm { ~ a n d ~ } t _ { \mathrm { B } _ { i }  \mathrm { B } _ { j } } ^ { \mathrm { e f f } } = - \frac { g ^ { 2 } } { \delta _ { \omega } } \xi ^ { 2 ( i - j ) } ,\tag{8}
$$

where $i , j$ label phonon sites surrounding the plaquette. For $i \ = \ j .$ , these expressions reduce to real onsite energy shifts $- g ^ { 2 } / \delta _ { \omega } ;$ for $i \neq j .$ , they yield complex NNN hoppings on the two sublattices. This is precisely the Haldane-type flux pattern that generates a nonzero Chern number without an external magnetic field [55].

In addition, a magnon-mediated hopping from a $\mathrm { B } _ { j }$ site to an $\mathrm { A } _ { i }$ site carries the phase associated with the diference between their physical radial angles. It is given by

$$
t _ { \mathrm { A } _ { i }  \mathrm { B } _ { j } } ^ { \mathrm { e f f } } = - \frac { g ^ { 2 } } { \delta _ { \omega } } \xi ^ { 2 ( i - j ) - 1 } .\tag{9}
$$

![](images/718de39adfa297fbd0527c9ba8d903ea7d42075187905130bbd8fd4959818bb8.jpg)

![](images/9de62edfa013f4a63a33dc119ab5698bf3f74662f5e731ebcf5c79a24a86ce14.jpg)

![](images/766fc8cd472c82d121a30e3e4cd1650272e90ce76039a020f331126e459055ff.jpg)  
FIG. 3. (a) and (b) Chern-number phase diagrams of the lowest band and the middle band, respectively, as functions of the magnon detuning $\delta _ { \omega }$ and the magnon–phonon coupling strength $^ { g , }$ with $t = 1$ . The dashed curve $4 g ^ { 2 } = t ^ { 2 } + t \delta _ { \omega }$ , marks the gap-closing condition at the M point, while the vertical dashed line in (b) indicates the Γ-point transition at $\delta _ { \omega } = 3 t$ . (c) and (d) Plots of bulk bandgaps $\Delta _ { 1 }$ (gap between the lower and middle bands) and $\Delta _ { 2 }$ (gap between the middle and upper bands), respectively, as functions of $\delta _ { \omega }$ and $g .$

This cross-sublattice term includes both NN and NNNN contributions, depending on the relative positions of the two sites.

Topological bandgap and boundary modes— To elucidate the emergence of the topological phase, we investigate the bulk band structure of the full coupled system. At $g \ : = \ : 0$ the two phononic bands host gapless Dirac points at the K and $\mathrm { K } ^ { \prime }$ valleys, while the magnon band is decoupled [Fig. 2(a), dashed curves]. The chiral magnon–phonon coupling hybridizes these modes and opens a bandgap at the phonon Dirac points. Through a numerical calculation of the Chern number [56], we find $C = \mp 1$ for the two phonon-dominated bands and $C = 0$ for the magnon-dominated bands [Fig. 2(a), solid curves], consistent with previous analysis.

The nontrivial bulk topology implies the existence of chiral edge states via the bulk-boundary correspondence [57]. For a strip geometry, Fig. 2(b) shows the edge modes traversing the bulk gap, strictly localized at opposite boundaries and propagating unidirectionally. Figure 2(c) further visualizes the boundary localization through the spatial intensity obtained by summing the probability of all in-gap states on a finite lattice. Reversing the magnetization direction $( + z  - z )$ complexconjugates the local coupling phases $( \xi \to \xi ^ { * } )$ , reversing the efective synthetic flux and changing the lowest-band Chern number from −1 to +1. Accordingly, the chiral edge states reverse their propagation direction.

Phase diagram and gap visibility— Finally, we work out the full phase diagram of the model by going beyond the large detuning limit and also allowing for strong magnon-phonon interaction. Figures 3(a) and (b) show the evolution of the lower- and middle-band Chern numbers as functions of $\delta _ { \omega }$ and $^ { g , }$ with the dashed line denoting analytic phase boundaries. For the lower band, the Chern number changes from −1 to +2 at $4 g ^ { 2 } = t ^ { 2 } + t \delta _ { \omega }$ suggesting a large Chern number phase with multiple chiral edge states can be obtained via increasing the magnon-phonon hopping. In the large detuning limit, this phase transition can be inferred from the efective NNNN hopping [Fig. 1(d)]. For the middle band, an additional phase transition occurs at $\delta _ { \omega } = 3 t$ , making the phase diagram much richer than the lowest band. Notably, when the sum of the Chern numbers of the lowest two bands is nonzero, the second bandgap will also host a chiral edge state [48].

Apart from the Chern numbers, gap size is also a crucial quantity that determines the performance of the system. Under strong detuning and weak coupling, the first bandgap $\Delta _ { 1 }$ can be enlarged by reducing the frequency detuning $\delta _ { \omega }$ or increasing the coupling $g \ [ \mathrm { F i g . \ 3 ( c ) } ]$ , consistent with the low-energy theory $\begin{array} { r } { ( \mathrm { i . e . , ~ } \Delta _ { 1 } \propto g ^ { 2 } / \delta _ { \omega } ) } \end{array}$ Away from this regime, the gap size is jointly determined by valley and M point modes, thus showing a nonmonotonic behavior. For the second bandgap, its size $\Delta _ { 2 }$ vanishes along $\delta _ { \omega } = 3 t$ , which corresponds to the band touching between the middle and upper bands. The gap size generally grows away from the closing points.

A realistic GHz phononic-crystal platform can be implemented using SAW devices or suspended elastic crystals on LiNbO , AlN, or Si, with nearest-neighbor hopping rates of $t / 2 \pi \sim 1 0$ –100 MHz and phonon linewidths approaching $\gamma _ { \mathrm { p h } } / 2 \pi \sim 1$ MHz [22, 58–62]. A CoFeB or YIG island placed at each plaquette center provides a Kittel mode whose frequency can be tuned by an external magnetic field, thereby enabling control of the magnon– phonon detuning $\delta _ { \omega }$ , while magnetoelastic and magnetorotational interactions generate the required coupling [34, 35, 37, 63]. CoFeB can ofer relatively strong coupling, $g / 2 \pi$ ∼ 10–30 MHz [64], but its larger Gilbert damping can lead to magnon linewidths of $\gamma _ { \mathrm { m } } / 2 \pi \sim 1 0 ^ { - }$ 100 MHz [65]. YIG provides a favorable low-loss alternative, with $g / 2 \pi \sim 5 $ –10 MHz and magnon linewidths near $\gamma _ { \mathrm { m } } / 2 \pi \sim 1$ MHz [66]. For representative parameters $t / 2 \pi = 3 0$ MHz, $\delta _ { \omega } = 3 . 2 t$ , and $g / 2 \pi = 7$ MHz, the estimated topological gap is approximately 5 MHz, exceeding the expected phonon and YIG magnon linewidths and therefore allowing clear spectral resolution of the gap, which can be further enhanced by reducing the frequency detuning.

Conclusion— We have proposed a magnon-induced phononic Chern insulator in a graphene-type phononic crystal system. The key to the design is the placement of the phonon and magnon sites, which generates favored time-reversal-breaking hoppings and opens a full topological gap. The model is studied analytically by analyzing the full and reduced phonon Hamiltonians, which clearly reveal the core function of the magnon– phonon interaction. We also investigate the model numerically, showing the existence of the chiral edge states and the fruitful phase diagram. More broadly, this work exemplifies an emerging human–AI discovery workflow in which the Qiushi Discovery Engine constructed the initial model and carried out the subsequent theoretical exploration and numerical checks, the authors then examined its physical assumptions, verified the results, and formulated the final interpretation. These results demonstrate the power of using discrete lattices to engineer magnon-phonon interactions and establish magnons as tunable mediators of synthetic gauge fields for reconfigurable GHz phonon transport.

Acknowledgements— This work was supported by the National Natural Science Foundation of China un der Grant Nos. 62401491 (H.X.) 62175215 (Y.Y.), and U25D8017 (Y.Y.), the National Key Research and Development Program of China under Grant Nos. 2022YFA1405200 (Y.Y.), and 2022YFA1404900 (Y.Y.), the Class D for Young Scientific Research Peak Creation No. K20250230 (Y.Y.), the Fundamental Research Funds for the Zhejiang Provincial Universities No. 226- 2025-00231 (Y.Y.), the Science Challenge Project N0. TZ2025015 (Y.Y.), the Research Grants Council of the Hong Kong SAR, China, under Grant No. 24304825 (H.X.), the Guangdong Provincial Quantum Science Strategic Initiative under Grant No. GDZX2501012 (H.X.), and the Chinese University of Hong Kong under Grant Nos. 4053729 (H.X.) and 4053794 (H.X.).

∗ yangyihao@zju.edu.cn † haoranxue@cuhk.edu.hk

[1] M. Maldovan, Sound and heat revolutions in phononics, Nature 503, 209 (2013).

[2] P. Wang, L. Lu, and K. Bertoldi, Topological phononic crystals with one-way elastic edge waves, Phys. Rev. Lett. 115, 104302 (2015).

[3] S. H. Mousavi, A. B. Khanikaev, and Z. Wang, Topologically protected elastic waves in phononic metamaterials, Nat. Commun. 6, 8682 (2015).

[4] Z. Yang, F. Gao, X. Shi, X. Lin, Z. Gao, Y. Chong, and B. Zhang, Topological acoustics, Phys. Rev. Lett. 114, 114301 (2015).

[5] A. B. Khanikaev, R. Fleury, S. H. Mousavi, and A. Alu, Topologically robust sound propagation in an angularmomentum-biased graphene-like resonator lattice, Nat. Commun. 6, 8260 (2015).

[6] X. Ni, C. He, X.-C. Sun, X.-p. Liu, M.-H. Lu, L. Feng, and Y.-F. Chen, Topologically protected one-way edge mode in networks of acoustic resonators with circulating air flow, New J. Phys. 17, 053016 (2015).

[7] R. S¨usstrunk and S. D. Huber, Observation of phononic helical edge states in a mechanical topological insulator, Science 349, 47 (2015).

[8] L. M. Nash, D. Kleckner, A. Read, V. Vitelli, A. M. Turner, and W. T. Irvine, Topological mechanics of gyroscopic metamaterials, Proc. Natl. Acad. Sci. USA 112, 14495 (2015).

[9] C. L. Kane and T. C. Lubensky, Topological boundary modes in isostatic lattices, Nat. Phys. 10, 39 (2014).

[10] M. Xiao, G. Ma, Z. Yang, P. Sheng, Z. Zhang, and C. T. Chan, Geometric phase and band inversion in periodic acoustic systems, Nat. Phys. 11, 240 (2015).

[11] C. He, X. Ni, H. Ge, X.-C. Sun, Y.-B. Chen, M.-H. Lu, X.-P. Liu, and Y.-F. Chen, Acoustic topological insulator and robust one-way sound transport, Nat. Phys. 12, 1124 (2016).

[12] M. Yan, J. Lu, F. Li, W. Deng, X. Huang, J. Ma, and Z. Liu, On-chip valley topological materials for elastic wave manipulation, Nat. Mater. 17, 993 (2018).

[13] G. Ma, M. Xiao, and C. T. Chan, Topological phases in acoustic and mechanical systems, Nat. Rev. Phys. 1, 281 (2019).

[14] H. Xue, Y. Yang, and B. Zhang, Topological acoustics, Nat. Rev. Mater. 7, 974 (2022).

[15] D. Hatanaka, I. Mahboob, K. Onomitsu, and H. Yamaguchi, Phonon waveguides for electromechanical circuits, Nat. Nanotechnol. 9, 520 (2014).

[16] P. Delsing, A. N. Cleland, M. J. Schuetz, J. Kn¨orzer, G. Giedke, J. I. Cirac, K. Srinivasan, M. Wu, K. C. Balram, C. B¨auerle, et al., The 2019 surface acoustic waves roadmap, J. Phys. D: Appl. Phys. 52, 353001 (2019).

[17] M. V. Gustafsson, T. Aref, A. F. Kockum, M. K. Ekstr¨om, G. Johansson, and P. Delsing, Propagating phonons coupled to an artificial atom, Science 346, 207 (2014).

[18] Y. Chu, P. Kharel, W. H. Renninger, L. D. Burkhart, L. Frunzio, and R. J. Schoelkopf, Quantum acoustics with superconducting qubits, Science 358, 199 (2017).

[19] K. J. Satzinger, Y. Zhong, H.-S. Chang, G. A. Peairs, A. Bienfait, M.-H. Chou, A. Y. Cleland, C. R. Conner, E. Dumur, J. Grebel, I. Gutierrez, B. H. November, R. G.<sup>´</sup> Povey, S. J. Whiteley, D. D. Awschalom, D. I. Schuster, and A. N. Cleland, Quantum control of surface acousticwave phonons, Nature 563, 661 (2018).

[20] A. H. Safavi-Naeini, D. Van Thourhout, R. Baets, and R. Van Laer, Controlling phonons and photons at the wavelength scale: integrated photonics meets integrated phononics, Optica 6, 213 (2019).

[21] M. Weiler, H. Huebl, F. S. Goerg, F. D. Czeschka, R. Gross, and S. T. B. Goennenwein, Elastically driven ferromagnetic resonance in nickel thin films, Phys. Rev. Lett. 106, 117601 (2011).

[22] Q. Zhang, D. Lee, L. Zheng, X. Ma, S. I. Meyer, L. He, H. Ye, Z. Gong, B. Zhen, K. Lai, and A. T. C. Johnson, Gigahertz topological valley hall efect in nanoelectromechanical phononic crystals, Nat. Electron. 5, 157 (2022).

[23] Z.-D. Zhang, S.-Y. Yu, M.-H. Lu, and Y.-F. Chen, Near ghz lithium niobate higher-order topological nanomechanical metamaterials, Nano Lett. 24, 15421 (2024).

[24] X.-B. Xu, M. Oudich, Y. Zeng, J.-Z. Zhang, Y.-H. Yang, J.-Q. Wang, W. Wang, L. Sun, G.-C. Guo, Y. Jing, and C.-L. Zou, Gigahertz topological phononic circuits based on micrometre-scale unsuspended waveguide arrays, Nat. Electron. 8, 689 (2025).

[25] D. Leykam, H. Xue, B. Zhang, and Y. Chong, Limitations and possibilities of topological photonics, Nat. Rev. Phys. 8, 55 (2026).

[26] A. Souslov, B. C. Van Zuiden, D. Bartolo, and V. Vitelli, Topological sound in active-liquid metamaterials, Nat. Phys. 13, 1091 (2017).

[27] Y. Ding, Y. Peng, Y. Zhu, X. Fan, J. Yang, B. Liang, X. Zhu, X. Wan, and J. Cheng, Experimental demonstration of acoustic chern insulators, Phys. Rev. Lett. 122, 014302 (2019).

[28] R. Fleury, A. B. Khanikaev, and A. Alu, Floquet topological insulators for sound, Nat. Commun. 7, 11744 (2016).

[29] A. Darabi, X. Ni, M. Leamy, and A. Al\`u, Reconfigurable floquet elastodynamic topological insulator based on synthetic angular momentum bias, Sci. Adv. 6, eaba8656 (2020).

[30] Q. Zhang, L. He, E. J. Mele, B. Zhen, and A. C. Johnson, General duality and magnet-free passive phononic chern insulators, Nat. Commun. 14, 916 (2023).

[31] P. Pirro, V. I. Vasyuchka, A. A. Serga, and B. Hillebrands, Advances in coherent magnonics, Nat. Rev. Mater. 6, 1114 (2021).

[32] B. Z. Rameshti, S. V. Kusminskiy, J. A. Haigh, K. Usami, D. Lachance-Quirion, Y. Nakamura, C.-M. Hu, H. X. Tang, G. E. Bauer, and Y. M. Blanter, Cavity magnonics, Phys. Rep. 979, 1 (2022).

[33] H. Yuan, Y. Cao, A. Kamra, R. A. Duine, and P. Yan, Quantum magnonics: When magnon spintronics meets quantum information science, Phys. Rep. 965, 1 (2022).

[34] C. Kittel, Interaction of spin waves and ultrasonic waves in ferromagnetic crystals, Phys. Rev. 110, 836 (1958).

[35] M. Xu, K. Yamamoto, J. Puebla, K. Baumgaertl, B. Rana, K. Miura, H. Takahashi, D. Grundler, S. Maekawa, and Y. Otani, Nonreciprocal surface acoustic wave propagation via magneto-rotation coupling, Sci. Adv. 6, eabb1724 (2020).

[36] P. J. Shah, D. A. Bas, I. Lisenkov, A. Matyushov, N. X. Sun, and M. R. Page, Giant nonreciprocity of surface acoustic waves enabled by the magnetoelastic interaction, Sci. Adv. 6, eabc5648 (2020).

[37] L. Liao, J. Puebla, K. Yamamoto, J. Kim, S. Maekawa, Y. Hwang, Y. Ba, and Y. Otani, Valley-selective phononmagnon scattering in magnetoelastic superlattices, Phys. Rev. Lett. 131, 176701 (2023).

[38] L. Liao, J. Liu, J. Puebla, Q. Shao, and Y. Otani, Hybrid magnon-phonon crystals, npj Spintronics 2, 47 (2024).

[39] R. Takahashi and N. Nagaosa, Berry curvature in magnon-phonon hybrid systems, Phys. Rev. Lett. 117, 217205 (2016).

[40] G. Go, S. K. Kim, and K.-J. Lee, Topological magnonphonon hybrid excitations in two-dimensional ferromagnets with tunable chern numbers, Phys. Rev. Lett. 123, 237207 (2019).

[41] E. Thingstad, A. Kamra, A. Brataas, and A. Sudbø, Chiral phonon transport induced by topological magnons, Phys. Rev. Lett. 122, 107201 (2019).

[42] X. Zhang, Y. Zhang, S. Okamoto, and D. Xiao, Thermal hall efect induced by magnon-phonon interactions, Phys. Rev. Lett. 123, 167202 (2019).

[43] S. Zhang, G. Go, K.-J. Lee, and S. K. Kim, Su (3) topology of magnon-phonon hybridization in 2d antiferromagnets, Phys. Rev. Lett. 124, 147204 (2020).

[44] S. Yang, F. Chen, R. Zhao, J. Wu, Y. Wang, H. Luo, N. Han, Q. Chen, Y. Hu, W. Li, M. Li, H. Chen, and Y. Yang, End-to-end autonomous scientific discovery on a real optical platform, arXiv:2604.27092 10.48550/arXiv.2604.27092 (2026).

[45] C. Kittel, On the theory of ferromagnetic resonance absorption, Phys. Rev. 73, 155 (1948).

[46] T. Holstein and H. Primakof, Field dependence of the intrinsic domain magnetization of a ferromagnet, Phys. Rev. 58, 1098 (1940).

[47] X. Zhang, C.-L. Zou, L. Jiang, and H. X. Tang, Cavity magnomechanics, Sci. Adv. 2, e1501286 (2016).

[48] See Supplemental Material for additional details on the theoretical model, Dirac mass and topological phase transitions.

[49] T. Oka and H. Aoki, Photovoltaic hall efect in graphene, Phys. Rev. B 79, 081406(R) (2009).

[50] T. Kitagawa, T. Oka, A. Brataas, L. Fu, and E. Demler, Transport properties of nonequilibrium systems under the application of light: Photoinduced quantum hall insulators without landau levels, Phys. Rev. B 84, 235108 (2011).

[51] G. Usaj, P. M. Perez-Piskunow, L. E. F. Foa Torres, and C. A. Balseiro, Irradiated graphene as a tunable floquet topological insulator, Phys. Rev. B 90, 115423 (2014).

[52] M. S. Rudner and N. H. Lindner, Band structure engineering and non-equilibrium dynamics in floquet topological insulators, Nat. Rev. Phys. 2, 229 (2020).

[53] J. R. Schriefer and P. A. Wolf, Relation between the Anderson model and the Kondo problem, Phys. Rev. 149, 491 (1966).

[54] S. Bravyi, D. P. DiVincenzo, and D. Loss, Schriefer– wolf transformation for quantum many-body systems, Ann. Phys. 326, 2793 (2011).

[55] F. D. M. Haldane, Model for a quantum hall efect without landau levels: Condensed-matter realization of the “parity anomaly”, Phys. Rev. Lett. 61, 2015 (1988).

[56] T. Fukui, Y. Hatsugai, and H. Suzuki, Chern numbers in discretized Brillouin zone: Eficient method of computing (spin) Hall conductances, J. Phys. Soc. Jpn. 74, 1674 (2005).

[57] D. J. Thouless, M. Kohmoto, M. P. Nightingale, and M. den Nijs, Quantized Hall conductance in a twodimensional periodic potential, Phys. Rev. Lett. 49, 405 (1982).

[58] L. Shao, S. Maity, L. Zheng, L. Wu, A. Shams-Ansari, Y.-I. Sohn, E. Puma, M. N. Gadalla, M. Zhang, C. Wang, E. L. Hu, K. Lai, and M. Loncar, Phononic band structure engineering for high-q gigahertz surface acoustic wave resonators on lithium niobate, Phys. Rev. Appl. 12, 014022 (2019).

[59] D. Hatanaka, M. Asano, H. Okamoto, and H. Yamaguchi, Phononic crystal cavity magnomechanics, Phys. Rev. Appl. 19, 054071 (2023).

[60] Z.-D. Zhang, S.-Y. Yu, M.-H. Lu, and Y.-F. Chen, Gigahertz surface acoustic wave topological rainbow in nanoscale phononic crystals, Phys. Rev. Lett. 133, 267001 (2024).

[61] M. Shen, J. Xie, C.-L. Zou, Y. Xu, W. Fu, and H. X. Tang, High frequency lithium niobate film-thicknessmode optomechanical resonator, Appl. Phys. Lett. 117, 131104 (2020).

[62] Z. Zheng, H. Feng, A. T. I¸sık, P. J. M. van der Slot, C. Wang, and D. Marpaung, Gigahertz thermoelastic acousto-optic modulation in lithium niobate integrated photonic device, Nanophotonics 14, 4683 (2025).

[63] L. Dreher, M. Weiler, M. Pernpeintner, H. Huebl, R. Gross, M. S. Brandt, and S. T. B. Goennenwein, Surface acoustic wave driven ferromagnetic resonance in nickel thin films: Theory and experiment, Phys. Rev. B 86, 134415 (2012).

[64] Y. Hwang, J. Puebla, K. Kondou, C. Gonzalez-

Ballestero, H. Isshiki, C. S´anchez Mu˜noz, L. Liao, F. Chen, W. Luo, S. Maekawa, and Y. Otani, Strongly coupled spin waves and surface acoustic waves at room temperature, Phys. Rev. Lett. 132, 056704 (2024).

[65] H. Matsumoto, I. Yasuda, M. Asano, Y. Todaka, T. Kawada, M. Kawaguchi, D. Hatanaka, and M. Hayashi, Magnon-phonon coupling of synthetic antiferromagnets in a surface acoustic wave cavity resonator, Nano Lett. 24, 5683 (2024).

[66] K. K¨unstle, Y. Kunz, T. Moussa, K. Lasinger, K. Yamamoto, P. Pirro, J. F. Gregg, A. Kamra, and M. Weiler, Magnon-polaron control in a surface magnetoacoustic wave resonator, Nat. Commun. 16, 10116 (2025).