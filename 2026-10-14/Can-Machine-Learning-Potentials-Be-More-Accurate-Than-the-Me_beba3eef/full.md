# Can Machine Learning Potentials Be More Accurate Than the Methods They Are Trained On?

Pavlo O. Dral,<sup>∗,†,‡,¶,§</sup>

Yi-Fan Hou,<sup>†</sup> Yuxinxin Chen,<sup>†</sup> Hassan Nawaz,<sup>†</sup> and Mateusz Witkowski,<sup>‡,¶</sup>

Szymon Smiga, <sup>´</sup> <sup>‡,¶</sup> and Ireneusz Grabowski<sup>‡,¶</sup>

†State Key Laboratory of Physical Chemistry of Solid Surfaces, College of Chemistry and

Chemical Engineering, and Fujian Provincial Key Laboratory of Theoretical and

Computational Chemistry, Xiamen University, Xiamen, Fujian 361005, China

‡Institute of Physics, Faculty of Physics, Astronomy, and Informatics, Nicolaus Copernicus

University in Toru´n, ul. Grudziadzka 5, 87-100 Toru´n, Poland

¶Institute of Advanced Studies, Nicolaus Copernicus University in Toru´n, ul. Wile´nska 4,

87-100 Toru´n, Poland

§Aitomistic, Shenzhen 518000, China

E-mail: dral@xmu.edu.cn

![](images/6900fb6123ee48549ffa43b20a45484fc3d2cdc59ac7be2cc1e5e3af522d1afb.jpg)

Aitomistic Shenzhen

![](images/5678ae4ae0c6eccb6de4959b283a6375536c9a0daae3a0e1cdf68acfdea54e44.jpg)

![](images/7e239c5e5049658307fe6da6b10d8c8088e18ee8191c12fe12693c096d2f384e.jpg)

## Abstract

Although machine learning potentials (MLPs) generalize across configurational and chemical compound spaces, their accuracy is commonly believed to be limited by their training methods, with improvements beyond this limit achieved so far only by refitting to experimental observables. In this Perspective, we argue that MLPs can also be made more accurate than their training methods non-empirically by generalizing across a third space, that of quantum-chemical methods. We propose routes to methodtransferable MLPs that predict at unseen methods, with working implementations in both density-functional and wavefunction-based theories showing that they can be more accurate than any of their training methods. Aiming beyond the accuracy of the training methods could also make universal MLPs more accurate than single-method models, even when the method-transferable MLPs remain less accurate than those QM methods. Method-transferable MLPs will shape the construction of training datasets and the evaluation of models, pushing the accuracy limits of chemical simulations.

## Introduction

Machine learning potentials (MLPs) are actively reshaping how atomistic simulations are performed.<sup>1–5</sup> MLPs, with their superior computational speed compared to quantum mechanical (QM) methods, enable simulations at unprecedented time scales and system sizes, allowing for the gain of previously unattainable insight into processes in molecules and materials.<sup>6–13</sup> Their usefulness rests on generalization across two spaces. In configurational space, a model learns the mapping from nuclear geometry to potential energy and predicts energies for geometries it has never seen. <sup>14</sup> In chemical compound space, once the total energy is split into atomic contributions, a model trained on some molecules predicts energies for unseen compounds (Figure 1a). <sup>15,16</sup>

## a Extending MLP generalization to QM-method space

![](images/d0419c67e07e0480ce49062bf1acf1b5b6e620776a81312a077cafe4ddc86d38.jpg)  
b Targeting higher accuracy with method-transferable MLPs

![](images/0c8427979c49f1bc164c801ca7c477847efb76eafe9ada8095f9c6b9acc6e216.jpg)  
Number of training geometries

c Improving MLP accuracy through empirical refinement  
![](images/0101cc36bb673fd56f901acfbc86432ca9392c8bfbcf4425cea3eb01e2fadfc6.jpg)

d Method-transferable MLPs from QM data alone (this work)  
![](images/ef200af1c741aecc39f44299e96292d25d7ea204b883545303f7dbf4f81411c8.jpg)  
Figure 1: Exceeding the accuracy of the training methods with MLPs. (a) Generalization across configurations, compounds and QM methods. FCI includes all determinants (all excitation ranks) in the chosen basis. Throughout this Perspective, filled/open markers denote training/unseen cases; noncircular method symbols distinguish them from atoms and neurons. (b) Schematic MLP error against the same more accurate reference as the number of training geometries grows. The dashed levels are limits of exact reproduction of the training methods. The orange arrow marks the accuracy gain; green shading indicates the target: accuracy beyond every training QM method. Even when this target is not reached, method-transferable MLPs can improve on single-method MLPs. (c) Empirical refinement compares simulated observables, such as IR spectra, with experiment to refine a QM-pretrained MLP. <sup>17–19</sup> (d) Known relationships between QM methods, represented by Pople’s diagram and Jacob’s ladder, could guide method-transferable MLPs.

No such generalization exists for the method space, where MLP would make predictions for unseen QM methods, fundamentally limiting the accuracy of an MLP to that of the training method used to generate the training data (Figure 1b). No surprise, then, that the belief that an MLP cannot surpass the accuracy of its training method is widely held and expressed in authoritative reviews and books, <sup>20–23</sup> including ours.<sup>5,24,25</sup>

Inspiringly, MLPs have been made more accurate than their training methods, but, unfortunately, so far only empirically. MLPs empirically refined on experimental observables predict those observables more accurately than the QM methods they were initially trained on (Figure 1c).<sup>17–19</sup> Spectroscopic refinement of potential-energy surfaces against measured rovibrational levels has done this for decades. <sup>26</sup> The limits of this route are structural: it requires experiments, which are often unavailable and carry their own errors. Most importantly, it ofers no systematic way to improve an MLP from theory alone. Quantum chemistry has systematic orderings of its own methods, such as Jacob’s ladder for density functionals<sup>27</sup> and the Pople diagram for the treatment of correlation and the basis set. <sup>28</sup> Methods therefore stand in known relations to one another, and a model that learns those relations could be made transferable across them.

In this Perspective, we argue that MLPs can exceed the accuracy of their training methods non-empirically, from QM data alone (Figure 1d). Just as an MLP can be made transferable in configurational and chemical compound space, it can be made transferable in a third space, that of QM methods. We envision and demonstrate possible constructions of such method-transferable MLPs, predicting at unseen methods. The constructions we propose below extend existing approaches to learning across methods, which predict only at their training methods (reviewed in the next section).

## Existing MLPs learning across methods

Existing approaches to learn across methods were designed with the motivation to exploit the heterogeneous data obtained with diferent levels, with the major goal of reducing the required number of expensive high-level QM calculations, rather than improving upon the training QM methods. Plenty of such methods were developed over time. <sup>29</sup> Because these methods are central to the development of method-transferable MLPs, here, we give a brief overview of such methods.

Two of the most common approaches—∆-learning <sup>30</sup> (Figure 2a) and transfer learning <sup>31</sup> (Figure 2b)—require a low- and high-level set of training QM methods to generate the reference data. ∆-learning is simply learning the diference between them, and requires the low-level QM method to perform the baseline calculations at inference time. Transfer learning first trains an MLP on the source, low-level QM data and then fine-tunes it to the target, high-level QM data to obtain two sets of MLPs (with the intention to only use the one trained on a high level) as, e.g., has been done in ANI-1ccx<sup>15</sup> and other potentials.<sup>32</sup> None of the transfer-learned models requires baseline QM calculations at inference time, but without a physical baseline, the calculations might produce unphysical results. The two approaches can be eficiently combined such that the ∆-learning correction is trained in two stages like in transfer learning: this was utilized, e.g., in our AIQM methods<sup>33,34</sup> which correct the semi-empirical QM baseline with an MLP-like correction to the coupled-cluster level. The MLP-like correction was pretrained on DFT-level corrections before it was fine-tuned to yield the coupled-cluster-level corrections.

Variants and extensions of these approaches exist. Low fidelity as a feature (Figure 2c) utilizes the baseline QM properties as an input, and has been used on dopant formation energies in hafnia<sup>35</sup>—a similar approach combined with ∆-learning has been adopted in molecular-orbital-based<sup>36</sup> potentials such as OrbNet Denali, <sup>37,38</sup> etc. Hierarchical machine learning stacks ∆-models to systematically improve across the hierarchy of QM methods (Figure 2d), which enabled very data-eficient training reducing the cost of generating highlevel reference data by 99% and enabled obtaining spectroscopic accuracy (an error of ca. 1 cm<sup>−1</sup>),<sup>39</sup> and was later adopted in other applications.<sup>40,41</sup>

a -learning  
![](images/d3eb620c2829b3c8834639920fbe227ae1617efec13ce0d8eeb984d4ab4bcb51.jpg)  
High-level prediction

b Transfer learning  
![](images/a80338aa865b2994ff5d5c018b05fc86c9a18ab9df2510f4505c34049aeb57aa.jpg)  
Reuse learned parameters

c Low fidelity as a feature  
![](images/bea61553e56c6c84586d2f072eed2bf3946548579df2eff0f685a85e81a8774d.jpg)  
High-level prediction

![](images/51cd195a917674c6168e47eb5f739ab646b3ff05f8585c9c06a380872e123f91.jpg)

![](images/e0c7ad9e15da7b1defef520124a6bfc965f849c70c0758769ac658d53cd798d9.jpg)  
Covariance across fidelities

![](images/11c00786d4b3abdd67ece18b1510ca20fa393e1c5425fc0e86f8049435fa41d9.jpg)  
Combine grid contributions

![](images/43df6fc4eb6a2872fd1def3e20f355376ef072acf8aa1c9ea3e75f12db7751e9.jpg)

![](images/bff0ea7fc6181a8dfc837ea0cd1ff5a9fde222fd28f05dabd91e4b644f4d60d9.jpg)

![](images/8e2a46890a77351af5b04db5c688fd50096027fad327fc97d7a8cccc97a85281.jpg)  
Figure 2: Existing approaches for learning in method space. E denotes energy; $L ,$ M and H denote low, intermediate and high levels. $\Delta E _ { H - L }$ is a learned high-minus-low energy correction; numbered subscripts and m identify individual QM methods. (a) ∆-learning; (b) transfer learning; (c) low-fidelity properties as features; (d) hierarchical machine learning; (e) co-kriging, where $K _ { L L }$ and $K _ { H H }$ are lowand high-fidelity covariance blocks and the of-diagonal blocks couple them; (f) the combination technique CQML, whose third axis is training-set size; (g) multi-task outputs; (h) fixed or learned method descriptors as input to MLP; and (i) meta-learning followed by adaptation using new-method training data.

Embedding correlation between methods is done with methods such as co-kriging (Figure 2e) that exploits the covariance between fidelities in a kernel, and it extends, in principle, to more than two levels. <sup>35,42</sup> The combination technique CQML (Figure 2f) places the levels on a grid over correlation treatment, basis set, and training set size, its third axis being the number of training molecules, and this line of work has since been extended further and optimized.<sup>43–47</sup> All of them are built to deliver the top of the hierarchy they were given.

A new route to learn heterogeneous data, which we argue is poised to be the new standard one, is learning all training methods with MLPs in a single model. Then the QM method can be chosen at inference, so that predictions targeting any training QM method stay available. One way to achieve this is a separate output head in a neural network (NN) per method with multi-task learning (Figure 2g). <sup>48–51</sup> Another way is, arguably, more elegant: providing a (QM) method descriptor as an input to the network (Figure 2h) such as in PFP,<sup>16</sup> SevenNet-MF,<sup>52</sup> and our own all-in-one model OMNI-P1. <sup>53</sup> It is the method we now advocate and extend later in this Perspective. DPA-3 uses a related approach with a dataset descriptor as input. <sup>54</sup>

The input can also be learned rather than fixed. Oerder et al. train one embedding for each of ten reference methods drawn from MultiXC-QM9,<sup>55,56</sup> and Chen et al. embed the fidelity as a trainable vector alongside the elemental embeddings and let measured band gaps enter as one more fidelity,<sup>57</sup> which puts experiment in as one more trained label rather than as a first-principles, systematic approach to improve past the training methods.

Meta-learning is another relevant approach (Figure 2i), where it is used to obtain a representation that refits cheaply to a new method. <sup>58</sup> The model still has to be fine-tuned before it predicts at any method at all.

None of these approaches predicts at a method that was not in its training set, and we pinpoint this as the major limitation that caps the accuracy of MLPs.

## Exceeding the accuracy of the training methods

Given the huge body of studies around MLPs and learning across method space, it is surprising that the field seems to concede to the ‘impossibility’ of training MLPs more accurate than the training QM methods. Hence, here we conceive several ideas to challenge this assumption and inspire a new era in MLP development informed by the systematic relationships known in QM theory. Beyond ideas, we provide architectural recipes and numerical demonstrations of the feasibility of such an approach.

## Taking a page from the quantum chemistry book: repurposing existing extrapolation schemes for MLPs

Think about established extrapolation schemes in QM theory such as Helgaker’s complete basis set (CBS) extrapolation scheme <sup>59,60</sup> or focal-point analysis. <sup>61</sup> While these methods enable predictions beyond the seen methods, their practical disadvantage is the need to perform additional QM calculations with non-negligible cost.

The reader can now easily connect the dots, and see that many of the existing MLPs learning across method space, such as variants based on multi-task learning or on using a method descriptor as input, can readily be repurposed to produce outputs at the same (training) QM methods that enter established extrapolation schemes but at negligible cost. This is the extrapolation route. Until now, the studies extrapolated first and only then trained MLP on already extrapolated values, by transfer learning as ANI-1ccx<sup>15</sup> or by stacking ∆- models up to the extrapolated level as in hierarchical ML, <sup>39</sup> so that the extrapolation scheme is fixed before the training data are generated, and the extrapolated values exist only for the geometries at which every QM method of the scheme was calculated. We propose to reverse this order and decouple training and extrapolation, which brings practical advantages: one MLP returns every training QM method, which can be used in various extrapolation schemes as a simple post-processing step (i.e., extrapolation scheme does not need to be predefined before training MLP), diferent geometries can be labeled with diferent methods, e.g., potentially improving such workflows as active learning, where progressively fewer geometries can be labeled with the more expensive QM methods.

As a demonstration, we construct an MLP that describes the umbrella inversion of ammonia (Figure 3a) more accurately than any of its training QM methods. Ammonia inversion was one of the conformational energy prototypes in the original focal-point analysis studies,<sup>61</sup> and we use this scheme to target CCSD(T) at the CBS limit [CCSD(T)/CBS] from calculations with two basis sets with the cardinal numbers $X > Y$ . It extrapolates the HF energy and the correlation energy of the much cheaper MP2 and adds the diference between CCSD(T) and MP2 calculated with the smaller basis set:

$$
E _ { \mathrm { C B S } } ^ { \mathrm { f o c a l } } = E _ { X Y } ^ { \infty } [ \mathrm { H F } ] + E _ { X Y } ^ { \mathrm { c o r r } , \infty } [ \mathrm { M P 2 } ] + E ( \mathrm { C C S D } ( \mathrm { T } ) / Y ) - E ( \mathrm { M P 2 } / Y ) ,\tag{1}
$$

where $E _ { X Y } ^ { \infty }$ [HF] and $E _ { X Y } ^ { \mathrm { c o r r , \infty } } [ \mathrm { M P 2 } ]$ are the established two-point extrapolations of the HF energy and MP2 correlation energy, respectively <sup>60,62</sup> (SI, p. S13). The last two terms are total energies evaluated with basis set Y . For comparison, we also apply the direct extrapolation of the CCSD(T) energy from the same two basis sets (Figure 3b; SI, Eq. S3).

We train one MLP on HF, MP2, CCSD, and CCSD(T) energies with the aug-cc-pVTZ and aug-cc-pVQZ basis sets and on HF, MP2, and CCSD forces with aug-cc-pVTZ, specifying the QM method through a descriptor as in our OMNI-P models <sup>25,53</sup> (Figure 3b). Progressively fewer geometries supply training labels for the more expensive methods, so training does not require every QM method at every geometry. Neither extrapolated energies nor the schemes enter training; the reported predictions average five models. We assess the resulting extrapolated predictions against a theoretical best estimate (TBE) from true QM calculations with the larger aug-cc-pVQZ and aug-cc-pV5Z basis sets (SI, p. S13).

On the held-out test set, the focal-point MLP gives a mean absolute error of 0.017 kcal/mo in relative energies against the TBE, five times smaller than the 0.085 kcal/mol of the most accurate training QM method, CCSD(T)/aug-cc-pVQZ (Figure 3c). The same scheme applied to true QM energies gives 0.010 kcal/mol, showing that the MLP retains most of the accuracy gained by extrapolation. Thus, an MLP trained without extrapolated targets can exceed the accuracy of every method supplying its training data for the right reason (agreement for a barrier alone can be misleading: CCSD/aug-cc-pVQZ reproduces the ammonia barrier through cancellation of basis-set and correlation errors, despite larger errors over the surrounding PES, see SI, p. S19). This advantage persists along the relaxed inversion profile up to about 12 kcal/mol above the minimum, close to the edge of the training geometries (dotted lines in Figure 3e).

(a)  
(b)  
![](images/a6893b6f58cad7dc0e7127cd8b071e8c13e12859d62fead62fc32cbad907b96f.jpg)

![](images/9dce8286a708b38caab70b5845ecbf95d47b1ca8f98ca8ca653951d404771343.jpg)  
(e)  
(d)

![](images/01d2d719b969dec6a4737eff12758bea9561d7c55c610529ace05b0402f2e18a.jpg)

![](images/30c525e552abdbf803f8518f5998af6b678e40394a1f6daf2ca2d6a9276cdeb7.jpg)  
Figure 3: One MLP, extrapolation after training: the inversion of ammonia. (a) Inversion through the planar saddle point. (b) One MLP predicts energies at geometry R for every training QM method m (method and basis set). Numbers count labelled training geometries used for training; purple squares and green diamonds identify energies entering the focal-point $\left( \operatorname { E q . 1 } \right)$ and direct-extrapolation (SI, Eq. S3) schemes, respectively; $Y = \mathrm { a u g - c c - p V T Z }$ $X = \operatorname { a u g - c c - p V Q Z } .$ (c) Mean absolute error (MAE) of relative energies against the theoretical best estimate (TBE) on the held-out test set, for all training methods and both schemes. Bars: true QM; symbols: corresponding MLP predictions (filled: training methods; open: unseen schemes); dashed line: best training QM method. (d) One-dimensional inversion splitting on each relaxed surface, divided by the TBE-surface splitting; bars and symbols as in (c). (e) Relaxed inversionprofile errors against the TBE for both MLP schemes and the best training QM method in true QM. The umbrella angle $\varphi$ measures N–H tilt out of the hydrogen plane; N–H distances are relaxed on each surface. Dotted lines bound the training geometries.

The gain in accuracy also improves the tunneling splitting of ammonia, illustrating how extrapolation after training can benefit simulations at MLP cost: on the relaxed inversion profile, the focal-point MLP gives 0.97 times the splitting on the TBE surface, against 0.89 for the MLP at CCSD(T)/aug-cc-pVQZ (Figure 3d). A single splitting requires about 5 million bead energy-and-force evaluations during instanton optimization, yet the full MLP calculation takes about 40 core-hours on Intel Xeon Gold 6240 CPUs, avoiding repeated QM calculations for the components of the focal-point scheme (SI, p. S19). In full-dimensional ring-polymer instanton calculations, <sup>63</sup> the focal-point MLP gives a splitting of 0.754 cm<sup>−1</sup> and the MLP at CCSD(T)/aug-cc-pVQZ 0.669 cm<sup>−1</sup>, against the experimental 0.793 cm<sup>−1</sup> <sup>64</sup> (SI, Table S13). The approximations in the electronic surfaces and tunneling treatment are discussed in the SI (p. S19). Beyond ammonia, including full CI among the training methods allows the same construction to target the full CI/CBS limit for the $\mathrm { H } ^ { \cdot } + \mathrm { H } _ { 2 }$ reaction (SI, p. S24).

Because the extrapolation schemes are applied to the $\mathrm { M L P ^ { \prime } s }$ predictions only after training, the same MLP can supply predictions from diferent schemes without further QM calculations, and these schemes can difer in accuracy: on the held-out $\mathrm { H } ^ { \cdot } + \mathrm { H } _ { 2 }$ geometries, direct extrapolation gives 0.022 and focal point 0.059 kcal/mol, while for ammonia the two are almost the same (0.018 and 0.017; SI, Tables S14 and S10). Their relative accuracy can be assessed against reference calculations chosen for the property of interest, e.g., at the minimum and saddle point for a barrier (SI, p. S19). Hence, the training dataset need not be constructed for one predetermined extrapolation scheme: a method-transferable MLP can reuse its learned QM methods in diferent schemes as the target property changes. However, when candidate schemes are known before training, we can readily include their prediction errors relative to the same schemes applied to the training QM energies in the loss, which brings the MLP to the accuracy of the same schemes on true QM energies (SI, p. S26).

## Putting QM methods on continuous coordinates in the method space

MLPs are capable of generalizing in configurational space by adopting continuous coordinates: molecular descriptors as a function of nuclear coordinates. We propose to extend this generalization to method space by giving MLPs continuous method descriptors informed by QM theory. The MLP can then learn relationships between QM methods and predict at methods absent from training. As a realization of our proposal, we can use parameters of common Kohn–Sham DFT functionals.

As a proof of concept, we use the fraction of exact (Hartree–Fock) exchange a as a physically informed continuous method descriptor for a space of hybrid functionals. In particular, we use the functional family of Adamo and Barone, <sup>65</sup> in which the remaining exchange is Becke’s 1988 (B88) functional,<sup>66</sup> and the correlation is the Lee–Yang–Parr (LYP) functional:<sup>67</sup>

$$
E _ { x c } ( a ) = a E _ { x } ^ { \mathrm { H F } } + \left( 1 - a \right) E _ { x } ^ { \mathrm { B 8 8 } } + E _ { c } ^ { \mathrm { L Y P } } .\tag{2}
$$

Several named functionals, including our recently proposed UMK26 $( a = 0 . 7 5 ) , ^ { 6 8 } \ \mathrm { o c } -$ cupy points in this space (Figure 4c). We use BLYP (a = 0), B1LYP (a = 0.25) <sup>65</sup> and HFLYP (a = 1) for training, leaving BLYP35 $( a = 0 . 3 5 )$ and BHandHLYP $( a = 0 . 5 ) ^ { 6 9 }$ as unseen functionals to test the predictions. The same construction can accommodate several

parameters, as in B3LYP-type hybrids. <sup>70</sup>

As a demonstration, we construct an MLP for intramolecular proton transfer in malonaldehyde (Figure 4a), whose barrier is sensitive to the fraction of exact exchange. We repurpose the OMNI-P1 architecture <sup>53</sup> by taking a as the method descriptor alongside the molecular descriptor and train on energies and forces from the three functionals with the cc-pVDZ basis set. One MLP then predicts the energy as a function of both the geometry R and the QM method descriptor a (Figure 4b):

$$
E _ { \mathrm { M L P } } ( { \mathbf { R } } ; a ) = \sum _ { i = 1 } ^ { N } E _ { i } ( { \mathbf { R } } ; a ) ,\tag{3}
$$

where N is the number of atoms and $E _ { i }$ is the atomic energy predicted from the chemical environment and a by a network shared by atoms of the same element. Changing a queries a diferent functional without retraining the MLP (SI, p. S4).

To choose a method more accurate than the training functionals, we find the optimal value $a _ { \mathrm { o p t } } = 0 . 4 5 5$ at which the MLP reproduces the CCSD(T)/cc-pVDZ reference barrier between the B3LYP/cc-pVDZ minimum and transition state (Figure 4d). This requires two CCSD(T) single-point energies and no further DFT calculations. These energies select the method descriptor, while the geometry dependence of the surface is learned from the DFT training data. On relaxing the minimum and transition state with the selected MLP, the barrier error remains 0.05 kcal/mol, compared with 1.23 kcal/mol for the best training functional (SI, Table S17).

The optimal method descriptor $a _ { \mathrm { o p t } }$ defines an actual DFT functional, so we can check whether the MLP reproduces it. Along the B3LYP reaction path, the MLP difers from this functional by an average of 0.010 kcal/mol, and it also predicts the withheld BLYP35 and BHandHLYP functionals with errors comparable to those at the training functionals (SI, Table S20). The method descriptor thus retains its physical meaning at values not included in training. Away from the two geometries used to choose $a _ { \mathrm { o p t } }$ , the MLP is also more accurate

(a)  
![](images/ae738afc0133d4a70551713b3e496b3c8512aba7b86b6e14f38b6c3438795a73.jpg)

(b)  
![](images/d5f7296bba2ffc9c0adff0f3f6e290a41662bea15f09b5f72fa4ff3b4094241d.jpg)  
(c)

$$
E _ { x c } ( a ) = a E _ { x } ^ { \mathrm { H F } } + ( 1 - a ) E _ { x } ^ { \mathrm { B 8 8 } } + E _ { c } ^ { \mathrm { L Y P } }
$$

(d)  
![](images/cb81038d88fef683f699a5dcb24c88b5133fe4a4e817aaef294e0ac3fd4b5a86.jpg)

![](images/cfc6cf19da5f48dc8314d088642d776ec3576db8a181c4ffc812550db601d1fb.jpg)

(e)  
![](images/f5212dc4278957dbe5f8536a2f039c05c7b6ab0d07c29e8e2fa6f5c4416654e0.jpg)

(f)  
![](images/abaa35f978d4868805212be31f48df6dc688d30aa77b9bb0b82cf1c4532d68fd.jpg)  
Figure 4: The fraction of exact exchange as the method descriptor. (a) The proton transfer in malonaldehyde: the B3LYP minimum, transition state and equivalent minimum as 2D and 3D structures. (b) The MLP of Eq. 3, trained at three values of a and queried at any a. (c) The space of Eq. 2 with its five named functionals, B3LYP beside it, and $a _ { \mathrm { o p t } }$ . (d) The barrier at the two B3LYP stationary geometries against a: the real functionals (DFT with Eq. 2), the MLP and CCSD(T), and where each meets CCSD(T). (e) The energy along the B3LYP reaction path, with the mean absolute error (MAE) against CCSD(T) over its 12 symmetry-unique geometries. (f) The tunneling splitting of malonaldehyde (H) and of its deuterated isotopologue (D) from ring-polymer instanton calculations on the MLP against $^ { a , }$ the range over the ensemble’s five MLPs at $a _ { \mathrm { o p t } }$ , and experiment $( \mathrm { H } ^ { 7 1 }$ and $\mathrm { D } ^ { 7 2 } )$ ).

than every functional in its training set: its mean absolute error against CCSD(T) along the B3LYP reaction path is 0.14 kcal/mol, compared with 0.46–0.84 kcal/mol for the training functionals (Figure 4e; SI, Table S18).

The MLP is most useful in simulations that require many energy evaluations, such as tunneling splittings. Ring-polymer instanton calculations of one splitting require about 2.6 million bead energy-and-force evaluations, supplied by our MLP in about 13 core-hours on an Intel Xeon Gold 6240 CPU, compared with an estimated 25,000–32,000 core-hours using cc-pVDZ DFT. At $a _ { \mathrm { o p t } }$ , our MLP trained on relatively inexpensive DFT data gives tunneling splittings of 29.8 and $4 . 3 9 ~ \mathrm { c m } ^ { - 1 }$ for H and D (Figure 4f), compared with the experimental values of 21.6 and $2 . 9 1 5 ~ \mathrm { c m } ^ { - 1 }$ , respectively.<sup>71,72</sup> The same instanton theory applied to an MLP transfer-learned from MP2 to the more expensive CCSD(T) reference with the larger aug-cc-pVTZ basis set gives 25.3 and $3 . 7 ~ \mathrm { c m } ^ { - 1 }$ .<sup>73</sup> Our method-transferable MLP thus gives splittings within 20 % of those of the dedicated single-method MLP.

We envision extending continuous method descriptors to wavefunction-based methods, so that an MLP can learn how electron correlation and basis-set completeness change together. The discrete correlation approximations and basis sets of the Pople diagram<sup>28</sup> would need physically justified coordinates with a meaningful joint limit, such as full CI at the complete basis set limit. The MLP could then learn the relations between methods and predict at coordinates not included in training. We are working toward this extension by developing such coordinates and testing predictions at unseen methods and will report these developments separately.

## Outlook: toward universal method-transferable MLPs

Our demonstrations concern individual molecules, for which dense sampling of the configurational space allows the MLP to resolve accuracy gains beyond the training QM methods. Universal MLPs sample the potential energy surfaces of many compounds much more sparsely.

Extending our constructions to them requires learning the relationships between methods across chemical compound space as well. Whether these relationships can be learned accurately enough to exceed the accuracy of the training QM methods remains an open question.

Recent universal MLPs make this question increasingly relevant. On selected ligandstrain and conformer benchmarks, UMA-M approaches direct DFT accuracy. <sup>74</sup> For such applications, reducing the fitting error alone cannot systematically correct errors in the training QM method. Learning to predict at more accurate, unseen methods would give universal MLPs a way to address this limitation.

The constructions demonstrated here suggest how to pursue this goal. Training a universal MLP on the components of extrapolation schemes would allow the scheme to be chosen after training, as demonstrated for ammonia. Continuous method descriptors ofer predictions between and beyond the training methods, with interpolation demonstrated here for malonaldehyde. Both constructions can use existing architectures with a method input or multiple outputs.

Method transfer can also improve universal MLPs when fitting errors remain substantial. An MLP aimed at a more accurate QM method can be more accurate than one aimed at a less accurate method even if neither reaches its target (Figure 1b). Our universal MLP OMNI-P1 illustrates this within one model: changing its method descriptor from DFT to CCSD(T)\*/CBS improves its accuracy, although both predictions remain less accurate than direct DFT calculations. <sup>53</sup> This motivates pursuing more accurate, unseen methods even for inexpensive models with substantial fitting errors. The outputs could also correct HF or MP2 toward CCSD(T)\*/CBS through ∆-learning, extending the DFT-based construction demonstrated with OMNI-P1. <sup>53</sup>

These aims should guide dataset construction and model evaluation. Labels from multiple QM methods, as in ANI-1ccx <sup>75</sup> and MultiXC-QM9,<sup>56</sup> provide information about relationships between methods. <sup>76</sup> Expensive high-level labels remain a constraint for methodtransferable MLPs too, but complete method coverage at every geometry is unnecessary:

cheaper calculations can cover many geometries, supplemented by high-level calculations at selected ones.<sup>77</sup> Method descriptors informed by systematic relationships, such as those represented in Jacob’s ladder and the Pople diagram, are another key research direction. Evaluation must distinguish reproduction of the training QM methods from accuracy against references more accurate than every training method, so that improvements beyond those methods can be established.

In this Perspective, we argued that MLPs can exceed the accuracy of their training QM methods by learning across method space and predicting at unseen methods. We demonstrated this for individual molecules by repurposing established extrapolation schemes and introducing continuous method descriptors. For universal MLPs, these routes could improve on single-method models even before exceeding the accuracy of their training QM methods. Learning across method space is poised to become as routine as learning across configurational and chemical compound spaces.

## Code and data availability

The machine learning potentials and simulations in this work were implemented in the opensource MLatom package, available under the Apache 2.0 license at https://github.com/ dralgroup/mlatom. The paper-specific scripts, MLatom source needed to use the reported models, input files, trained models, training and test data, model predictions, and selected raw simulation logs and outputs are available together on Zenodo at https://doi.org/10. 5281/zenodo.23329249. Original code is provided under the Apache 2.0 license, and original data, model parameters, and results under the CC BY 4.0 license; third-party components retain their original licenses.

## Authors contributions

P.O.D. conceived and directed the study, carried out the research and wrote the manuscript. Y.-F.H. proposed learning across the continuous space of DFT functionals and developed the active-learning protocol used in this work. Y.C. developed the extension of the OMNI-P architecture on which all MLPs in this work are based. H.N. and M.W. carried out a preliminary study of continuous descriptors of wavefunction-based methods, demonstrating the feasibility of constructing MLPs more accurate than the QM methods used to train them. S.S. and I.G. contributed through discussions on extrapolation schemes and on machine <sup>´</sup> learning across the space of DFT functionals. P.O.D. takes responsibility for all content. The calculations, analysis and drafting were carried out by the AI Dral Group, the group’s collective of specialised AI research agents, under the authors’ direction and verification, with every result reproducible from the repository by a single command.

## Declaration of generative AI use

Generative AI was used substantively in this work. The study was executed by the AI Dral Group, an AI research-agent collective that is independent of the underlying agentic runtime; in this work it operated on the Claude Code agent with the Claude Opus 5.5 and Claude Fable 5.1 models (Anthropic), the Codex agent with the GPT-6 Astra and GPT-6 Sol models (OpenAI), and the Protomia agent (Aitomistic) with the Qwen3.8-27B model (Qwen team, Alibaba) run locally. The human authors directed and verified every step and take full responsibility for the content, which is machine-verifiable and reproducible from the committed repository.

## Acknowledgement

P.O.D. acknowledges funding by the projects for International Senior Scientists (Project No.: W2531013) and for Outstanding Youth Scholars (Overseas, 2021) of the National Natural Science Foundation of China, via the Lab project of the State Key Laboratory of Physical Chemistry of Solid Surfaces. S.S. thanks the Polish National Science Center for the partial <sup>´</sup> financial support under Grant No. 2021/42/E/ST4/00096. We acknowledge Polish highperformance computing infrastructure PLGrid for awarding this project access to the LUMI supercomputer, owned by the EuroHPC Joint Undertaking, hosted by CSC (Finland) and the LUMI consortium through PLL/2026/10/019484.

This work was supported by the PIAST-AI Factory project – PIAST AIF, co-financed by the EuroHPC Joint Undertaking through the European Union’s Digital Europe Programme, and by Poland as the Participating Country, under Grant Agreement No. 101250730. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union, EuroHPC Joint Undertaking or Poland. Neither the European Union, the EuroHPC Join Undertaking nor Poland can be held responsible for them.

## References

(1) Xia, J.; Zhang, Y.; Jiang, B. The Evolution of Machine Learning Potentials for Molecules, Reactions and Materials. Chem. Soc. Rev. 2025, 54, 4790–4821.

(2) Jinnouchi, R.; Minami, S. Machine Learning Force Fields in Electrochemistry: From Fundamentals to Applications. ACS Nano 2025, 19, 22600–22644.

(3) Kulichenko, M.; Nebgen, B.; Lubbers, N.; Smith, J. S.; Barros, K.; Allen, A. E. A.; Habib, A.; Shinkle, E.; Fedik, N.; Li, Y. W.; Messerly, R. A.; Tretiak, S. Data Generation for Machine Learning Interatomic Potentials and Beyond. Chem. Rev. 2024, 124, 13681–13714.

(4) Behler, J. Four Generations of High-Dimensional Neural Network Potentials. Chem. Rev. 2021, 121, 10037–10072.

(5) Dral, P. O. Quantum Chemistry in the Age of Machine Learning. J. Phys. Chem. Lett. 2020, 11, 2336–2347.

(6) Bocus, M.; Vandenhaute, S.; Van Speybroeck, V. The Operando Nature of Isobutene Adsorbed in Zeolite H-SSZ-13 Unraveled by Machine Learning Potentials Beyond DFT Accuracy. Angew. Chem. Int. Ed. 2025, 64, e202413637.

(7) Shi, Y. F.; Kang, P. L.; Shang, C.; Liu, Z. P. Methanol Synthesis from CO2/CO Mixture on Cu–Zn Catalysts from Microkinetics-Guided Machine Learning Pathway Search. J. Am. Chem. Soc. 2022, 144, 13401–13414.

(8) Yan, Z.; Wei, D.; Li, X.; Chung, L. W. Accelerating Reliable Multiscale Quantum Refinement of Protein–Drug Systems Enabled by Machine Learning. Nat. Commun. 2024, 15, 4181.

(9) Nandi, A.; Zhang, P.; Chen, J.; Guo, H.; Bowman, J. M. Quasiclassical Simulations Based on Cluster Models Reveal Vibration-Facilitated Roaming in the Isomerization of CO Adsorbed on NaCl. Nat. Chem. 2021, 13, 249–254.

(10) Friederich, P.; Hase, F.; Proppe, J.; Aspuru-Guzik, A. Machine-Learned Potentials for Next-Generation Matter Simulations. Nat. Mater. 2021, 20, 750–761.

(11) Deringer, V. L.; Bernstein, N.; Csanyi, G.; Ben Mahmoud, C.; Ceriotti, M.; Wilson, M.; Drabold, D. A.; Elliott, S. R. Origins of Structural and Electronic Transitions in Disordered Silicon. Nature 2021, 589, 59–64.

(12) Westermayr, J.; Gastegger, M.; Voros, D.; Panzenboeck, L.; Joerg, F.; Gonzalez, L.; Marquetand, P. Deep Learning Study of Tyrosine Reveals That Roaming Can Lead to Photodamage. Nat. Chem. 2022, 14, 914–919.

(13) Wang, L.; Salguero, C.; Lopez, S. A.; Li, J. Machine Learning Photodynamics Uncover Blocked Non-Radiative Mechanisms in Aggregation-Induced Emission. Chem 2024, 10, 2295–2310.

(14) Behler, J.; Parrinello, M. Generalized Neural-Network Representation of High-Dimensional Potential-Energy Surfaces. Phys. Rev. Lett. 2007, 98, 146401.

(15) Smith, J. S.; Nebgen, B. T.; Zubatyuk, R.; Lubbers, N.; Devereux, C.; Barros, K.; Tretiak, S.; Isayev, O.; Roitberg, A. E. Approaching Coupled Cluster Accuracy with a General-Purpose Neural Network Potential Through Transfer Learning. Nat. Commun. 2019, 10, 2903.

(16) Takamoto, S.; Shinagawa, C.; Motoki, D.; Nakago, K.; Li, W.; Kurata, I.; Watanabe, T.; Yayama, Y.; Iriguchi, H.; Asano, Y.; Onodera, T.; Ishii, T.; Kudo, T.; Ono, H.; Sawada, R.; Ishitani, R.; Ong, M.; Yamaguchi, T.; Kataoka, T.; Hayashi, A.; Charoenphakdee, N.; Ibuka, T. Towards Universal Neural Network Potential for Material Discovery Applicable to Arbitrary Combination of 45 Elements. Nat. Commun. 2022, 13, 2991.

(17) R¨ocken, S.; Zavadlav, J. Accurate Machine Learning Force Fields via Experimental and Simulation Data Fusion. npj Comput. Mater. 2024, 10, 69.

(18) Gumber, S.; Alzate-Vargas, L.; Nebgen, B. T.; van Veelen, A.; Kadvani, S.; Gibson, T.; Messerly, R. Going Beyond Density Functional Theory Accuracy: Leveraging Experimental Data to Refine Pre-Trained Machine Learning Interatomic Potentials. arXiv preprint arXiv:2506.10211 2025,

(19) Han, B.; Yu, K. Refining Potential Energy Surface Through Dynamical Properties via Diferentiable Molecular Simulation. Nat. Commun. 2025, 16, 816.

(20) Behler, J. Constructing High-Dimensional Neural Network Potentials: A Tutorial Review. Int. J. Quantum Chem. 2015, 115, 1032–1050.

(21) Behler, J. Perspective: Machine Learning Potentials for Atomistic Simulations. J. Chem. Phys. 2016, 145, 170901.

(22) Hellstr¨om, M.; Behler, J. Machine Learning Meets Quantum Chemistry; 2020; pp 253– 275.

(23) Omranpour, A.; Montero De Hijes, P.; Behler, J.; Dellago, C. Perspective: Atomistic Simulations of Water and Aqueous Systems with Machine Learning Potentials. J. Chem. Phys. 2024, 160, 170901.

(24) Dral, P. O.; Barbatti, M. Molecular Excited States Through a Machine Learning Lens. Nat. Rev. Chem. 2021, 5, 388–405.

(25) Martyka, M.; Tong, X.-Y.; Jankowska, J.; Dral, P. O. OMNI-P2x Universal Neural Network Potential for Excited-State Simulations. Nat. Commun. 2026, 17, 4949.

(26) Huang, X.; Schwenke, D. W.; Tashkun, S. A.; Lee, T. J. An Isotopic-Independent Highly Accurate Potential Energy Surface for CO<sub>2</sub> Isotopologues and an Initial $^ { 1 2 } \mathrm { C } ^ { 1 6 } \mathrm { O } _ { 2 }$ Infrared Line List. J. Chem. Phys. 2012, 136, 124311.

(27) Perdew, J. P.; Schmidt, K. Jacob’s Ladder of Density Functional Approximations for the Exchange-Correlation Energy. AIP Conf. Proc. 2001, 577, 1–20.

(28) Pople, J. A. Two-Dimensional Chart of Quantum Chemistry. J. Chem. Phys. 1965, 43, S229–S230.

(29) Dral, P. O.; Zubatiuk, T.; Xue, B.-X. In Quantum Chemistry in the Age of Machine Learning; Dral, P. O., Ed.; Elsevier: Amsterdam, Netherlands, 2023; Chapter 21, pp 491–507.

(30) Ramakrishnan, R.; Dral, P. O.; Rupp, M.; von Lilienfeld, O. A. Big Data Meets Quantum Chemistry Approximations: The ∆-Machine Learning Approach. J. Chem. Theory Comput. 2015, 11, 2087–2096.

(31) Pan, S. J.; Yang, Q. A Survey on Transfer Learning. IEEE Trans. Knowl. Data Eng. 2010, 22, 1345–1359.

(32) K¨aser, S.; Boittier, E. D.; Upadhyay, M.; Meuwly, M. Transfer Learning to CCSD(T): Accurate Anharmonic Frequencies from Machine Learning Models. J. Chem. Theory Comput. 2021, 17, 3687–3699.

(33) Zheng, P.; Zubatyuk, R.; Wu, W.; Isayev, O.; Dral, P. O. Artificial Intelligence-Enhanced Quantum Chemical Method with Broad Applicability. Nat. Commun. 2021, 12, 7022.

(34) Chen, Y.; Dral, P. O. AIQM2: Organic Reaction Simulations Beyond DFT. Chem. Sci. 2025, 16, 15901–15912.

(35) Batra, R.; Pilania, G.; Uberuaga, B. P.; Ramprasad, R. Multifidelity Information Fusion with Machine Learning: A Case Study of Dopant Formation Energies in Hafnia. ACS Appl. Mater. Interfaces 2019, 11, 24906–24918.

(36) Welborn, M.; Cheng, L.; Miller, T. F., III Transferability in Machine Learning for Electronic Structure via the Molecular Orbital Basis. J. Chem. Theory Comput. 2018, 14, 4772–4779.

(37) Qiao, Z.; Welborn, M.; Anandkumar, A.; Manby, F. R.; Miller, T. F., III OrbNet: Deep Learning for Quantum Chemistry Using Symmetry-Adapted Atomic-Orbital Features. J. Chem. Phys. 2020, 153, 124111.

(38) Christensen, A. S.; Sirumalla, S. K.; Qiao, Z.; O’Connor, M. B.; Smith, D. G. A.; Ding, F.; Bygrave, P. J.; Anandkumar, A.; Welborn, M.; Manby, F. R.; Miller, T. F., III OrbNet Denali: A Machine Learning Potential for Biological and Organic Chemistry with Semi-Empirical Cost and DFT Accuracy. J. Chem. Phys. 2021, 155, 204103.

(39) Dral, P. O.; Owens, A.; Dral, A.; Cs´anyi, G. Hierarchical Machine Learning of Potential Energy Surfaces. J. Chem. Phys. 2020, 152, 204110.

(40) Nandi, A.; Qu, C.; Houston, P. L.; Conte, R.; Bowman, J. M. ∆-Machine Learning for Potential Energy Surfaces: A PIP Approach to Bring a DFT-Based PES to CCSD(T) Level of Theory. J. Chem. Phys. 2021, 154, 051102.

(41) Bowman, J. M.; Qu, C.; Conte, R.; Nandi, A.; Houston, P. L.; Yu, Q. ∆-Machine Learned Potential Energy Surfaces and Force Fields. J. Chem. Theory Comput. 2023, 19, 1–17.

(42) Pilania, G.; Gubernatis, J. E.; Lookman, T. Multi-Fidelity Machine Learning Models for Accurate Bandgap Predictions of Solids. Comput. Mater. Sci. 2017, 129, 156–163.

(43) Zaspel, P.; Huang, B.; Harbrecht, H.; von Lilienfeld, O. A. Boosting Quantum Machine Learning Models with a Multilevel Combination Technique: POple Diagrams Revisited. J. Chem. Theory Comput. 2019, 15, 1546–1559.

(44) Vinod, V.; Maity, S.; Zaspel, P.; Kleinekath¨ofer, U. Multifidelity Machine Learning for Molecular Excitation Energies. J. Chem. Theory Comput. 2023, 19, 7658–7670.

(45) Heinen, S.; Khan, D.; von Rudorf, G. F.; Karandashev, K.; Arismendi Arrieta, D. J.; Price, A. J. A.; Nandi, S.; Bhowmik, A.; Hermansson, K.; von Lilienfeld, O. A. Reducing Training Data Needs with Minimal Multilevel Machine Learning (M3L). Mach. Learn.: Sci. Technol. 2024, 5, 025058.

(46) Vinod, V.; Kleinekath¨ofer, U.; Zaspel, P. Optimized Multifidelity Machine Learning for Quantum Chemistry. Mach. Learn.: Sci. Technol. 2024, 5, 015054.

(47) Vinod, V.; Zaspel, P. Investigating Data Hierarchies in Multifidelity Machine Learning for Excitation Energies. J. Chem. Theory Comput. 2025, 21, 3077–3091.

(48) Jacobson, L.; Stevenson, J.; Ramezanghorbani, F.; Dajnowicz, S.; Leswing, K. Leveraging Multitask Learning to Improve the Transferability of Machine Learned Force Fields. ChemRxiv 2023,

(49) Messerly, M.; Matin, S.; Allen, A. E. A.; Nebgen, B.; Barros, K.; Smith, J. S.; Lubbers, N.; Messerly, R. Multi-Fidelity Learning for Interatomic Potentials: Low-Level Forces and High-Level Energies Are All You Need. Mach. Learn.: Sci. Technol. 2025, 6, 035066.

(50) Zhang, D.; Liu, X.; Zhang, X.; Zhang, C.; Cai, C.; Bi, H.; Du, Y.; Qin, X.; Peng, A.; Huang, J.; Li, B.; Shan, Y.; Zeng, J.; Zhang, Y.; Liu, S.; Li, Y.; Chang, J.; Wang, X.; Zhou, S.; Liu, J.; Luo, X.; Wang, Z.; Jiang, W.; Wu, J.; Yang, Y.; Yang, J.; Yang, M.; Gong, F.-Q.; Zhang, L.; Shi, M.; Dai, F.-Z.; York, D. M.; Liu, S.; Zhu, T.; Zhong, Z.; Lv, J.; Cheng, J.; Jia, W.; Chen, M.; Ke, G.; E, W.; Zhang, L.; Wang, H. DPA-2: A Large Atomic Model as a Multi-Task Learner. npj Comput. Mater. 2024, 10, 293.

(51) Batatia, I.; Lin, C.; Hart, J.; Kasoar, E.; Elena, A. M.; Norwood, S. W.; Wolf, T.; Cs´anyi, G. Cross Learning Between Electronic Structure Theories for Unifying Molecular, Surface, and Inorganic Crystal Foundation Force Fields. arXiv:2510.25380v1 [physics.chem-ph] 2025,

(52) Kim, J.; Kim, J.; Kim, J.; Lee, J.; Park, Y.; Kang, Y.; Han, S. Data-Eficient Multifidelity Training for High-Fidelity Machine Learning Interatomic Potentials. J. Am. Chem. Soc. 2025, 147, 1042–1054.

(53) Chen, Y.; Dral, P. O. One to Rule Them All: A Universal Interatomic Potential Learning Across Quantum Chemical Levels. J. Chem. Theory Comput. 2025, 21, 8762–8772.

(54) Zhang, D.; Peng, A.; Cai, C.; Li, W.; Zhou, Y.; Zeng, J.; Guo, M.; Zhang, C.; Li, B.; Jiang, H.; Zhu, T.; Jia, W.; Zhang, L.; Wang, H. A Graph Neural Network for the Era of Large Atomistic Models. npj Comput. Mater. 2026, 12, 276.

(55) Oerder, R.; Schmieden, G.; Hamaekers, J. Multi-Fidelity Learning for Atomistic Models via Trainable Data Embeddings. Mach. Learn.: Sci. Technol. 2025, 6, 045004.

(56) Nandi, S.; Vegge, T.; Bhowmik, A. MultiXC-QM9: Large Dataset of Molecular and Reaction Energies from Multi-Level Quantum Chemical Methods. Sci. Data 2023, 10, 783.

(57) Chen, C.; Zuo, Y.; Ye, W.; Li, X.; Ong, S. P. Learning Properties of Ordered and Disordered Materials from Multi-Fidelity Data. Nat. Comput. Sci. 2021, 1, 46–53.

(58) Allen, A. E. A.; Lubbers, N.; Matin, S.; Smith, J.; Messerly, R.; Tretiak, S.; Barros, K. Learning Together: Towards Foundation Models for Machine Learning Interatomic Potentials with Meta-Learning. npj Comput. Mater. 2024, 10, 154.

(59) Helgaker, T.; Klopper, W.; Koch, H.; Noga, J. Basis-Set Convergence of Correlated Calculations on Water. J. Chem. Phys. 1997, 106, 9639–9646.

(60) Halkier, A.; Helgaker, T.; Jorgensen, P.; Klopper, W.; Koch, H.; Olsen, J.; Wilson, A. K. Basis-Set Convergence in Correlated Calculations on Ne, N2, and H2O. Chem. Phys. Lett. 1998, 286, 243–252.

(61) Cs´asz´ar, A. G.; Allen, W. D.; Schaefer, H. F. In Pursuit of the ab initio Limit for Conformational Energy Prototypes. J. Chem. Phys. 1998, 108, 9751–9764.

(62) Halkier, A.; Helgaker, T.; Jorgensen, P.; Klopper, W.; Olsen, J. Basis-Set Convergence of the Energy in Molecular Hartree-Fock Calculations. Chem. Phys. Lett. 1999, 302, 437–446.

(63) Richardson, J. O.; Althorpe, S. C. Ring-Polymer Instanton Method for Calculating Tunneling Splittings. J. Chem. Phys. 2011, 134, 054109.

(64) Urban, S.; D’Cunha, R.; Rao, K. N.; Papouˇsek, D. The ∆ <sup>ˇ</sup> k = ±2 “Forbidden Band” and Inversion–Rotation Energy Levels of Ammonia. Can. J. Phys. 1984, 62, 1775–1791.

(65) Adamo, C.; Barone, V. Toward Reliable Adiabatic Connection Models Free from Adjustable Parameters. Chem. Phys. Lett. 1997, 274, 242–250.

(66) Becke, A. D. Density-Functional Exchange-Energy Approximation with Correct Asymptotic Behavior. Phys. Rev. A 1988, 38, 3098–3100.

(67) Lee, C.; Yang, W.; Parr, R. G. Development of the Colle–Salvetti Correlation-Energy Formula into a Functional of the Electron Density. Phys. Rev. B 1988, 37, 785–789.

(68) Kumar, V. B.; Smiga, S.; Ara´ujo Mendes, R.; Jezierska, B.; Bartlett, R.; Grabowski, I.<sup>´</sup> Primary Role of Semi-Local and Exact-Exchange Diference in the Description of Correlation Efects in Hybrid Functionals. J. Chem. Theory Comput. 2026, DOI: 10.1021/acs.jctc.6c00843.

(69) Renz, M.; Theilacker, K.; Lambert, C.; Kaupp, M. A Reliable Quantum-Chemical Protocol for the Characterization of Organic Mixed-Valence Compounds. J. Am. Chem. Soc. 2009, 131, 16292–16302.

(70) Stephens, P. J.; Devlin, F. J.; Chabalowski, C. F.; Frisch, M. J. Ab Initio Calculation of Vibrational Absorption and Circular Dichroism Spectra Using Density Functional Force Fields. J. Phys. Chem. 1994, 98, 11623–11627.

(71) Baba, T.; Tanaka, T.; Morino, I.; Yamada, K. M. T.; Tanaka, K. Detection of the Tunneling-Rotation Transitions of Malonaldehyde in the Submillimeter-Wave Region. J. Chem. Phys. 1999, 110, 4131–4133.

(72) Baughcum, S. L.; Smith, Z.; Wilson, E. B.; Duerst, R. W. Microwave Spectroscopic Study of Malonaldehyde. 3. Vibration-Rotation Interaction and One-Dimensional Model for Proton Tunneling. J. Am. Chem. Soc. 1984, 106, 2260–2265.

(73) K¨aser, S.; Richardson, J. O.; Meuwly, M. Transfer Learning for Afordable and High-

Quality Tunneling Splittings from Instanton Calculations. J. Chem. Theory Comput.   
2022, 18, 6840–6850.

(74) Wood, B. M.; Dzamba, M.; Fu, X.; Gao, M.; Shuaibi, M.; Barroso-Luque, L.; Abdelmaqsoud, K.; Gharakhanyan, V.; Kitchin, J. R.; Levine, D. S.; Michel, K.; Sriram, A.; Cohen, T.; Das, A.; Sahoo, S. J.; Rizvi, A.; Ulissi, Z. W.; Zitnick, C. L. UMA: A Family of Universal Models for Atoms. Advances in Neural Information Processing Systems 38. 2025; pp 143528–143564, arXiv:2506.23971.

(75) Smith, J. S.; Zubatyuk, R.; Nebgen, B.; Lubbers, N.; Barros, K.; Roitberg, A. E.; Isayev, O.; Tretiak, S. The ANI-1ccx and ANI-1x Data Sets, Coupled-Cluster and Density Functional Theory Properties for Molecules. Sci. Data 2020, 7, 134.

(76) Ullah, A.; Chen, Y.; Dral, P. O. Molecular Quantum Chemical Data Sets and Databases for Machine Learning Potentials. Mach. Learn.: Sci. Technol. 2024, 5, 041001.

(77) Gardner, J. L. A.; Schulz, H.; H´elie, J.; Sun, L.; Simm, G. N. C. Understanding Multi-Fidelity Training of Machine-Learned Force-Fields. Mach. Learn.: Sci. Technol. 2026, 7, 045057.

# Can Machine Learning Potentials Be More Accurate Than the Methods They Are Trained On?

Pavlo O. Dral<sup>1,2,3,4,\*</sup>

Yi-Fan Hou,<sup>1</sup> Yuxinxin Chen,<sup>1</sup> Hassan Nawaz<sup>1</sup> and Mateusz Witkowski<sup>2,3</sup>

Szymon Smiga<sup>´</sup> <sup>2,3</sup> and Ireneusz Grabowski<sup>2,3</sup>

<sup>1</sup>State Key Laboratory of Physical Chemistry of Solid Surfaces, College of Chemistry and Chemical

Engineering, and Fujian Provincial Key Laboratory of Theoretical and Computational Chemistry,

Xiamen University, Xiamen, Fujian 361005, China

<sup>2</sup>Institute of Physics, Faculty of Physics, Astronomy, and Informatics, Nicolaus Copernicus Uni-

versity in Toru´n, ul. Grudziadzka 5, 87-100 Toru´n, Poland

<sup>3</sup>Institute of Advanced Studies, Nicolaus Copernicus University in Toru´n, ul. Wile´nska 4, 87-100 Toru´n, Poland

<sup>4</sup>Aitomistic, Shenzhen 518000, China

\* E-Mail: dral@xmu.edu.cn

## Table of Contents

Computational details S2   
Programs and reference QM calculations S2   
The method-transferable MLP S4   
Active learning S8   
Tunneling splittings by ring-polymer instanton theory S9   
Rate constants S10   
Extrapolation after training S13   
The schemes and the TBE S13   
Training data S15   
Ammonia S19   
Hydrogen exchange: extrapolation to full CI/CBS S24   
The errors of the schemes in the training loss S26   
Interpolation along exact exchange S29   
Training data and active learning S29   
The MLP and the choice of $a _ { \mathrm { o p t } }$ S30   
Checks along the reaction path and across the method space . S31   
Tunneling splittings of malonaldehyde S36   
References S39

## Computational details

## Programs and reference QM calculations

Programs. All calculations, unless stated otherwise, run through MLatom, <sup>S1</sup> in a development version that adds the MLP with the method descriptor and the active learning with several reference methods of this work, with Python 3.11, PyTorch 2.1.2 and TorchANI $2 . 2 . 3 ^ { \mathrm { S 2 } }$ for the MLP, PySCF $2 . 4 . 0 ^ { \mathrm { S 3 } }$ for DFT and CCSD(T), geomeTRIC $1 . 0 . 2 ^ { \mathrm { S 4 } }$ for geometry optimizations, and i-PI $3 . 2 . 0 ^ { \mathrm { S 5 } }$ for the ring-polymer instanton, with our own interface to the MLPs of MLatom and our own corrections to its tunneling-splitting postprocessing. MLatom provides the IRC, the molecular dynamics (velocity Verlet), the Wigner and harmonic quantum Boltzmann sampling of initial conditions, the analytic Hessians of the MLP, and the averaging of the ensemble. In the extrapolation route, the QM labels are calculated with PySCF 2.4.0 directly, without MLatom.

Reference calculations of the extrapolation route. The QM calculations are done with PySCF $2 . 4 ^ { \mathrm { S 3 } }$ and the $\mathrm { c c - p V X Z ^ { S 6 } }$ and $\mathrm { \ a u g { - c c { - p V } X Z ^ { S 7 } } }$ basis sets, from the unrestricted HF solution for $\mathrm { H } ^ { \cdot } + \mathrm { H } _ { 2 }$ and the restricted one for ammonia, with the N 1s orbital frozen in all correlated calculations. CCSD and CCSD(T) energies come from one CCSD solution per geometry and basis set, and full CI for $\mathrm { H } ^ { \cdot } + \mathrm { H } _ { 2 }$ is a complete-active-space configuration interaction with all electrons in all orbitals. SCF energies are converged to 10<sup>−12</sup> and CCSD energies to $1 0 ^ { - 1 0 }$ Hartree. The analytic gradients of HF, MP2, and CCSD agree with central finite diferences to $2 \times 1 0 ^ { - 6 }$ H $\mathrm { a r t r e e } / \AA$ for both systems, while the analytic CCSD(T) gradients of this version of PySCF deviate by $1 0 ^ { - 4 }$ to $3 \times 1 0 ^ { - 3 }$ H $\mathrm { a r t r e e / \AA }$ , too much to be used as training labels; the MLP therefore learns the forces of HF, MP2, and CCSD and only the energies of CCSD(T) and full CI, and validation and test geometries are calculated with energies only. For ammonia, the aug-cc-pVQZ calculations, which are the most expensive of the training set, are done for half of the training geometries and with energies only, so that all its forces come from aug-cc-pVTZ.

Every QM method is also calculated at one reference geometry: $\mathrm { H _ { 2 } }$ at $0 . 7 4 1 4 \mathrm { ~ \AA ~ }$ , the equilibrium distance of the London–Eyring–Polanyi–Sato surface used to generate the geometries (p. S16), with the third atom 8 $\textup { \AA }$ away for $\mathrm { H } ^ { \cdot } + \mathrm { H } _ { 2 }$ , and the pyramidal stationary point of the focal-point analysis<sup>S8</sup> used to generate the geometries $\mathrm { ( N { - } H ~ 1 . 0 1 2 ~ \AA }$ , H–N–H 106.7<sup>◦</sup>) for ammonia. The reference geometry is the same for every QM method and is not optimized with any of them, so some training geometries lie below it: on each surface, 27–53 of the $\mathrm { H } { \cdot } + \mathrm { H } _ { 2 }$ geometries, by at most 0.03 kcal/mol, and up to six of the ammonia geometries, by at most 0.26 kcal/mol with HF and 0.004 kcal/mol with the correlated methods. Every relative energy reported is an energy diference on one surface: the energy at a geometry minus the energy of the same surface at the reference geometry, for the MLP (its own prediction there, which is not exactly zero) as for every QM method.

Reference calculations of the interpolation route. DFT energies and forces in the space of Eq. 2 are calculated with $\mathrm { P y S C F ^ { S 3 } }$ through MLatom,<sup>S1</sup> each functional given as a custom exchange–correlation functional, with the cc-pVDZ basis set<sup>S6</sup> and PySCF’s default integration grid and convergence thresholds. CCSD(T) energies are calculated with PySCF from the restricted Hartree–Fock reference, with the 1s orbitals of the carbon and oxygen atoms frozen; where the CCSD iterations stalled on a nearly singular DIIS subspace, they were restarted with a larger subspace and then without DIIS. Relative energies in the comparisons are formed by subtracting the energy of the same surface at the B3LYP/cc-pVDZ minimum, one fixed geometry for every functional and for CCSD(T). The B3LYP/cc-pVDZ intrinsic reaction coordinate (IRC) used in the tests is calculated with MLatom from the B3LYP transition state in both directions, with the exact Hessian at every step and steps of 0.05 $\mathrm { { \AA } } ;$ it has 27 points, the transition state and 13 on each side.

## The method-transferable MLP

Architecture and method descriptor. The MLP is an ANI-type neural network<sup>S9</sup> with the atomic environment vectors of ANI-1x and atomic networks of 160, 128, and 96 neurons, as implemented in the TorchANI<sup>S2</sup> interface of MLatom,<sup>S1</sup> with the method descriptor appended to the atomic environment vector of every atom, as in our OMNI-P series of models.<sup>S10,S11</sup> In the extrapolation route, the descriptor is a one-hot vector over the nine $\left( \mathrm { H } ^ { \cdot } + \mathrm { H } _ { 2 } \right)$ or eight (ammonia) training QM methods (Table S1). CCSD is among them although neither scheme uses it, because it comes out of the same coupled-cluster calculation as CCSD(T) at no extra cost. Each training geometry contributes energy labels for its selected QM methods (Table S1). Forces are also used as labels for HF, MP2 and CCSD, whose analytic gradients in PySCF 2.4.0 agree with finite diferences of the energies (p. S2). CCSD(T) and full CI are used with energies only: the analytic CCSD(T) gradients of this PySCF version deviate from finite diferences by up to $3 \times 1 0 ^ { - 3 }$ Hartree/<sup>˚</sup>A, and full CI has no analytic gradient in it. For ammonia, the aug-cc-pVQZ calculations, which take minutes per geometry, are done with energies only, so all forces of ammonia come from aug-cc-pVTZ. Each label is used together with the method descriptor of its QM method. In the interpolation route, all three training functionals are used with energies and forces.

In the interpolation route, the descriptor is the exact-exchange fraction a, and each atomic network returns one energy as in Eq. 3. Both routes use GELU activations, <sup>S12</sup> whose smooth second derivatives are needed for frequencies and tunneling splittings. <sup>S13</sup>

Table S1: Training QM methods, method descriptors and labels of the MLPs of both routes. $\mathbf { e } _ { k }$ is the one-hot vector with 1 at position k of 9 $\left( \mathrm { H } ^ { \cdot } + \mathrm { H } _ { 2 } \right)$ or 8 (ammonia); the share is that of the training geometries used for training with the QM method (percentages rounded to integers) (p. S16 and p. S29).
<table><tr><td>system  $\mathrm { H } ^ { \cdot } + \mathrm { H } _ { 2 }$ </td><td>QM method</td><td>descriptor</td><td>labels</td><td>share</td></tr><tr><td rowspan="11"></td><td> $\overline { { \mathrm { H F / c c } } } \mathrm { - p V T Z }$ </td><td>e1</td><td>energies, forces</td><td>100%</td></tr><tr><td> $\mathrm { M P 2 / c c { \mathrm { - } } p V T Z }$ </td><td> $\mathbf { e } _ { 2 }$ </td><td>energies, forces</td><td>100%</td></tr><tr><td> $\mathrm { C C S D / c c } { \mathrm { - p V T Z } }$ </td><td> $\mathbf { e } _ { 3 }$ </td><td>energies, forces</td><td>60%</td></tr><tr><td> $\mathrm { C C S D ( T ) / c c { \cdot } p V T Z }$ </td><td> $\mathbf { e } _ { 4 }$ </td><td>energies</td><td>60%</td></tr><tr><td>full  $\mathrm { C I / c c { \mathrm { - } } p V T Z }$ </td><td> $\mathbf { e } _ { 5 }$ </td><td>energies</td><td>25%</td></tr><tr><td> $\mathrm { H F / c c { \mathrm { - p V } } Q Z }$ </td><td> $\mathbf { e } _ { 6 }$ </td><td>energies, forces</td><td>63%</td></tr><tr><td> $\mathrm { M P 2 / c c { \mathrm { - } } p V Q Z }$ </td><td> $\mathbf { e } _ { 7 }$ </td><td>energies, forces</td><td>63%</td></tr><tr><td> $\mathrm { C C S D / c c } { \mathrm { - p V Q Z } }$ </td><td> $\mathbf { e } _ { 8 }$ </td><td>energies, forces</td><td>38%</td></tr><tr><td> $\mathrm { C C S D ( T ) / c c { \cdot } p V Q Z }$ </td><td> $\mathbf { e } _ { 9 }$ </td><td>energies</td><td>38%</td></tr><tr><td>ammonia  $\mathrm { H F / a u g - c c - p V T Z }$ </td><td> $\mathbf { e } _ { 1 }$ </td><td>energies, forces</td><td>100%</td></tr><tr><td> $\mathrm { M P 2 / a u g  – c c – p V T Z }$ </td><td> $\mathbf { e } _ { 2 }$ </td><td>energies, forces</td><td>100%</td></tr><tr><td></td><td> $\mathrm { C C S D / a u g - c c - p V T Z }$ </td><td> $\mathbf { e } _ { 3 }$ </td><td>energies, forces</td><td>60%</td></tr><tr><td></td><td> $\mathrm { C C S D ( T ) / a u g \mathrm { - c c \mathrm { - } p V T Z } }$ </td><td> $\mathbf { e } _ { 4 }$ </td><td>energies</td><td>60%</td></tr><tr><td></td><td> $\scriptstyle { \mathrm { H F / a u g - c c - p V Q Z } }$ </td><td> $\mathbf { e } _ { 5 }$ </td><td>energies</td><td>50%</td></tr><tr><td></td><td> $\mathrm { M P 2 / a u g - c c - p V Q Z }$ </td><td> $\mathbf { e } _ { 6 }$ </td><td>energies</td><td>50%</td></tr><tr><td></td><td> $\mathrm { C C S D / a u g \mathrm { - c c \mathrm { - } p V Q Z } }$ </td><td> $\mathbf { e } _ { 7 }$ </td><td>energies</td><td>30%</td></tr><tr><td></td><td> $\mathrm { C C S D ( T ) / a u g \mathrm { - c c \mathrm { - } p V Q Z } }$ </td><td> $\mathbf { e } _ { 8 }$ </td><td>energies</td><td>30%</td></tr><tr><td rowspan="3">malonaldehyde</td><td> $\mathrm { B L Y P / c c - p V D Z }$ </td><td> $a = 0$ </td><td>energies, forces</td><td>100%</td></tr><tr><td> $\mathrm { B 1 L Y P / c c - p V D Z }$ </td><td> $a = 0 . 2 5$ </td><td>energies, forces</td><td>100%</td></tr><tr><td> $\mathrm { H F L Y P / c c - p V D Z }$ </td><td> $a = 1$ </td><td>energies, forces</td><td>100%</td></tr></table>

Training. The MLP is trained with the defaults of MLatom’s TorchANI interface: force weight 0.1, and AdamW for network weights and SGD for biases, both with an initial learning rate of $1 0 ^ { - 3 }$ halved after 64 epochs without improvement of the validation error until the AdamW learning rate falls below $1 0 ^ { - 5 }$ . In the extrapolation route, we use batches of 128 instead of 8 geometries and disable per-batch logging. The ammonia MLP of the main text is trained on the energies and forces of the training QM methods only.

In the interpolation route, the MLP is trained on energies and forces with the settings of our active-learning protocol:<sup>S14</sup> the defaults of MLatom’s TorchANI interface above, including batches of 8 geometries, with 10% of the training set for validation instead of the interface’s 20%.

Energy zero and atomic energies. In both routes, the MLP gives the energy of QM method $m .$ , with method descriptor $d _ { m } \ ( \mathbf { e } _ { m } \ \mathrm { o r } \ a )$ , at the geometry R as

$$
E _ { m } ^ { \mathrm { M L P } } ( { \bf R } ) = \sum _ { i } E _ { i } ( { \bf R } , d _ { m } ) + \sum _ { i } \varepsilon _ { Z _ { i } } ,\tag{S1}
$$

where $E _ { i }$ is the output of the atomic network of atom i and $\varepsilon _ { Z }$ is one atomic energy per element, the same for all QM methods. The two routes difer only in the energy label $y _ { m } ( \mathbf { R } )$ that Eq. S1 is trained to reproduce:

$$
y _ { m } ( \mathbf { R } ) = E _ { m } ( \mathbf { R } ) - E _ { m } ( \mathbf { R } _ { \mathrm { r e f } } ) \quad ( \mathrm { e x t r a p o l a t i o n } ) , \qquad y _ { m } ( \mathbf { R } ) = E _ { m } ( \mathbf { R } ) \quad ( \mathrm { i n t e r p o l a t i o n } ) ,\tag{S2}
$$

with $E _ { m }$ the energy of QM method m and $\mathbf { R } _ { \mathrm { r e f } }$ the reference geometry (p. S2). Before training, MLatom’s TorchANI interface fits the $\varepsilon _ { Z }$ by least squares to all labels of all QM methods at once, so that the networks learn only the much smaller remainder: they are trained on $y _ { m } ( \mathbf { R } ) - \textstyle \sum _ { i } \varepsilon _ { Z _ { i } } ;$ in prediction, $\sum _ { i } \varepsilon _ { Z _ { i } }$ is added back as in Eq. S1, and nothing else is: the extrapolation MLP predicts energies relative to the reference geometry and the interpolation MLP total energies. Since all geometries of a route are the same molecule, $\sum _ { i } \varepsilon _ { Z _ { i } }$ is one constant, and every energy reported is a diference between two geometries on one surface, from which this constant and $E _ { m } ( { \bf R } _ { \mathrm { r e f } } )$ both cancel. The schemes act on these diferences unchanged because the coeficients with which each scheme combines the energies of the QM methods (Eq. S6) sum to one. The subtraction of $E _ { m } ( { \bf R } _ { \mathrm { r e f } } )$ in the extrapolation route was an early design choice, which, in principle, is not required.

Angular cutof and ensemble. In the extrapolation route, the angular cutof of the atomic environment vectors is chosen by five-fold cross-validation among 3.3, 3.5, and 4.0 <sup>˚</sup>A with the four angular shifts spread evenly from 0.9 <sup>˚</sup>A to the cutof: the five folds are fixed once for all training geometries, and the five MLPs of the chosen cutof, each validated on a diferent fold, form the ensemble whose averaged predictions are reported. The mean validation errors of the three cutofs are 0.026–0.030 kcal/mol for the $\mathrm { H } ^ { \cdot } + \mathrm { H } _ { 2 }$ reaction and 0.006–0.007 kcal/mol for ammonia (Tables S2 and S3). These diferences are smaller than the standard deviations of the errors over the folds, 0.003–0.011 and 0.001–0.002 kcal/mol, so the choice among them does not change the validation error measurably. We use the cutof with the lowest mean error, $3 . 5 \mathrm { ~ \AA ~ }$ for the $\mathrm { H } ^ { \cdot } + \mathrm { H } _ { 2 }$ reaction and 3.3 <sup>˚</sup>A for ammonia. In the interpolation route, the angular cutof of the atomic environment vectors is 4.0 <sup>˚</sup>A, with the four angular shifts spread evenly from 0.9 <sup>˚</sup>A to the cutof. It was chosen once, by five-fold cross-validation among 3.3, 3.5 and 4.0 <sup>˚</sup>A on 450 geometries of an early design study of the initial data set, and was not chosen again for the reported data set (Table S4). The three cutofs give energy errors of 0.165–0.192 kcal/mol, diferences smaller than the standard deviations over the folds, 0.030–0.055 kcal/mol; 4.0 <sup>˚</sup>A, which has the lowest error, is also the value of our transferable models.<sup>S15,S16</sup>

Table S2: Choice of the angular cutof of the MLP for the $\mathrm { H } ^ { \cdot } + \mathrm { H } _ { 2 }$ reaction by five-fold cross-validation: MAE of relative energies on the validation fold in kcal/mol, averaged over the folds, for each training QM method and their mean (standard deviation over the folds in parentheses). The five MLPs of the chosen cutof (3.5 <sup>˚</sup>A) form the ensemble.
<table><tr><td>training QM method</td><td>3.3 Å</td><td>3.5 Å</td><td>4.0 Å</td></tr><tr><td> $\overline { { \mathrm { H F / c c } } } \mathrm { - p V T Z }$ </td><td>0.034</td><td>0.026</td><td>0.031</td></tr><tr><td> $\mathrm { M P 2 / c c { \mathrm { - } } p V T Z }$ </td><td>0.022</td><td>0.022</td><td>0.023</td></tr><tr><td> $\mathrm { C C S D / c c } { \mathrm { - p V T Z } }$ </td><td>0.024</td><td>0.022</td><td>0.027</td></tr><tr><td> $\mathrm { C C S D ( T ) / c c { \cdot } p V T Z }$  full  $\mathrm { C I / c c { \mathrm { - } } p V T Z }$ </td><td>0.028</td><td>0.029</td><td>0.031</td></tr><tr><td></td><td>0.033</td><td>0.031</td><td>0.040</td></tr><tr><td> $\mathrm { H F / c c { \mathrm { - p V } } Q Z }$ </td><td>0.027</td><td>0.029</td><td>0.026</td></tr><tr><td> $\mathrm { M P 2 / c c { \mathrm { - } } p V Q Z }$ </td><td>0.024</td><td>0.021</td><td>0.025</td></tr><tr><td> $\mathrm { C C S D / c c } { \mathrm { - p V Q Z } }$ </td><td>0.032</td><td>0.032</td><td>0.034</td></tr><tr><td> $\mathrm { C C S D ( T ) / c c { \cdot } p V Q Z }$ </td><td>0.030</td><td>0.028</td><td>0.032</td></tr><tr><td>mean</td><td>0.028 (0.011)</td><td>0.026 (0.009)</td><td>0.030 (0.003)</td></tr><tr><td>folds</td><td>5</td><td>5</td><td>5</td></tr></table>

Table S3: Choice of the angular cutof of the MLP for the ammonia by five-fold crossvalidation: MAE of relative energies on the validation fold in kcal/mol, averaged over the folds, for each training QM method and their mean (standard deviation over the folds in parentheses). The five MLPs of the chosen cutof (3.3 <sup>˚</sup>A) form the ensemble.
<table><tr><td>training QM method</td><td>3.3 Å</td><td>3.5 Å</td><td>4.0 Å</td></tr><tr><td>HF/aug-cc-pVTZ</td><td>0.005</td><td>0.005</td><td>0.006</td></tr><tr><td>MP2/aug-cc-pVTZ</td><td>0.004</td><td>0.005</td><td>0.005</td></tr><tr><td>CCSD/aug-cc-pVTZ</td><td>0.005</td><td>0.005</td><td>0.004</td></tr><tr><td>CCSD(T)/aug-cc-pVTZ</td><td>0.008</td><td>0.009</td><td>0.008</td></tr><tr><td>HF/aug-cc-pVQZ</td><td>0.009</td><td>0.012</td><td>0.010</td></tr><tr><td>MP2/aug-cc-pVQZ</td><td>0.006</td><td>0.007</td><td>0.008</td></tr><tr><td>CCSD/aug-cc-pVQZ</td><td>0.007</td><td>0.008</td><td>0.006</td></tr><tr><td>CCSD(T)/aug-cc-pVQZ</td><td>0.007</td><td>0.007</td><td>0.006</td></tr><tr><td>mean</td><td>0.006 (0.001) 5</td><td>0.007 (0.002)</td><td>0.007 (0.001)</td></tr><tr><td colspan="2">folds</td><td colspan="2">5</td></tr></table>

Table S4: Choice of the angular cutof of the atomic environment vectors of the malonaldehyde MLP: root-mean-square error (RMSE) of the energies in five-fold cross-validation and its standard deviation over the five folds, in kcal/mol, on 450 geometries of an early design study of the initial data set, labelled with the three training functionals. These geometries are not the training set of the reported MLP. The four angular shifts are spread evenly from 0.9 <sup>˚</sup>A to the cutof.
<table><tr><td>cutoff (Å)</td><td>RMSE</td><td>standard deviation over the folds</td></tr><tr><td>3.3</td><td>0.179</td><td>0.030</td></tr><tr><td>3.5</td><td>0.192</td><td>0.038</td></tr><tr><td>4.0</td><td>0.165</td><td>0.055</td></tr></table>

## Active learning

Both routes select training geometries by our physics-informed active learning<sup>S14</sup> in MLatom.<sup>S1</sup> In each iteration, the MLP runs a set of simulations; each molecular-dynamics trajectory is stopped at the first geometry at which the MLP is uncertain, and only this geometry is calculated with QM and added to the training set. A geometry optimization likewise contributes at most the first uncertain geometry along its path. The MLP is then retrained and the next iteration starts. The uncertainty is the diference between the energies of a main MLP, trained on energies and forces, and an auxiliary MLP trained on energies only; a geometry is uncertain when the diference exceeds a threshold, by default the median plus three median absolute deviations of that diference over the validation set. The threshold, the simulations, the initial data set and the end of the loop are specific to each system (pp. S16 and S29).

## Tunneling splittings by ring-polymer instanton theory

Frequencies and tunneling splittings. Except for the rate constants of the $\mathrm { H } ^ { \cdot } + \mathrm { H } _ { 2 }$ reaction (p. S10), harmonic frequencies are obtained from the analytic Hessian of the ensemble, and zero-point energies as half the sum of the real frequencies. Tunneling splittings of malonaldehyde and ammonia are calculated with ring-polymer instanton theory <sup>S17</sup> in i-PI 3.2,<sup>S5</sup> following the protocol of K¨aser, Richardson and Meuwly for malonaldehyde, with the same scripts for both molecules.<sup>S18</sup> i-PI calls the MLP directly in its own process, through its Python force-field interface, without a socket or an external driver. The reactant is the harmonic analysis of one minimum of the MLP from its analytic Hessian, with the masses and conventions of i-PI (i-PI’s own harmonic analysis by finite diferences of the forces puts the splitting 0.6% higher), and the splitting is twice the tunneling matrix element of the instanton in the splitting mode of i-PI.

The half ring polymer between the two minima of the MLP is optimized with the L-BFGS algorithm and i-PI’s convergence thresholds tightened tenfold, first with 64 beads at 100 K from the straight line between the minima, then with 256 beads at 50 K, and finally with 1024 beads at 25 K, each stage starting from the previous instanton interpolated to the new number of beads. The atoms carry the masses of their most abundant isotopes, except where an isotopologue is named. The two molecules difer in the surface, in the optimization of the minima and in the Hessians of the beads, which are given for malonaldehyde on p. S36 and for ammonia below.

Fluctuation factor. The fluctuation factor needs the determinant of the Hessian of the whole half ring polymer, a matrix of $3 n \times 1 0 2 4$ rows for a molecule of n atoms; we compute it by a block LU factorization of this block-tridiagonal matrix instead of i-PI’s dense diagonalization, which on three test runs agrees with a dense calculation to $3 \times 1 0 ^ { - 1 0 }$ in the

logarithm of the determinant (Table S23).

Inversion splittings of ammonia. On each surface, in true QM and from the MLP alike, the relaxed inversion path consists of 34 values of the H–N–H angle from $7 8 ^ { \circ }$ to the planar structure (120<sup>◦</sup>), 2<sup>◦</sup> apart below $9 6 ^ { \circ }$ and 1<sup>◦</sup> above, with the N–H distance relaxed at each by a parabola through three points; the path is parametrized by the mass-weighted arc length with the motion of the centre of mass removed and mirrored at the planar structure, and the one-dimensional Schr¨odinger equation of this symmetric double well is solved by finite diferences on 1201 grid points; the splitting is the diference between the two lowest levels, and it changes by less than 0.1% when the path is shortened by $4 \%$ at each end. The same treatment is applied to every surface. It overestimates the experimental splitting of $\mathrm { N H _ { 3 } }$ $( 0 . 7 9 \mathrm { c m } ^ { - 1 } )$ by a factor of 1.8 on the TBE surface, so the surfaces are compared through the ratio of their splittings to that of the TBE surface.

The full-dimensional splittings are calculated with ring-polymer instanton theory <sup>S17</sup> with the same driver as for malonaldehyde (i-PI, above): a half ring polymer of 1024 beads at 25 K, reached through 64 beads at 100 K and 256 at 50 K, the ensemble evaluated in double precision, the bead Hessians analytic, the zero mode removed by projecting out the tangent of the path, the minimum optimized by Newton steps with the analytic Hessian of the surface, and the isotope masses of the one-dimensional treatment. Doubling the number of beads to 2048 changes the splittings by −0.4 % $\mathrm { ( N H _ { 3 } ) }$ and $- 0 . 2 \% \left( \mathrm { N D _ { 3 } } \right)$ , and halving the temperature to 12.5 K at the same bead spacing (2048 beads) changes them by +0.4 % and $+ 0 . 3 \%$ ; the 1024-bead values are thus within 0.6 % of the bead limit.

## Rate constants

The $\mathrm { H } ^ { \cdot } + \mathrm { H } _ { 2 } \longrightarrow \mathrm { H } _ { 2 } + \mathrm { H } ^ { \cdot }$ reaction is degenerate and has no measurable rate constant; the rate constants are therefore those of the $\mathrm { D } ^ { \cdot } + \mathrm { H _ { 2 } } \longrightarrow \mathrm { H D } + \mathrm { H } ^ { \cdot }$ reaction, which runs on the same potential energy surface with diferent nuclear masses and for which experiment and accurate quantum dynamics on the basis-set-limit surface of the same reaction<sup>S19</sup> agree over 167–2112 K.<sup>S20</sup> On every surface, the collinear symmetric saddle point and its Hessian are obtained from the energies at the same 22 geometries. The surfaces are each training QM method and each scheme, in true QM and from the MLP (the averaged energies of the ensemble) alike, and the TBE in true QM.

The 22 geometries are five points along the symmetric stretch (H–H distances from 0.91 to 0.95 <sup>˚</sup>A), the middle atom displaced along the axis and perpendicular to it at three of those five, and $\mathrm { H _ { 2 } }$ at five bond lengths with the third atom 8 $\textup { \AA }$ away. The saddle point is the minimum of a quartic polynomial fitted along the symmetric stretch, and the three curvatures that fix the Cartesian Hessian of the linear symmetric H−H−H are obtained from the displaced energies by Richardson-extrapolated finite diferences and interpolated to the saddle point. The barriers quoted in the text are those between each surface’s own stationary points, which difer by up to 0.05 kcal/mol from the barriers at the fixed benchmark geometry of SI, Figure S2 (SI, Tables S16 and S12).

The harmonic frequencies of D−H−H at the saddle point and of $\mathrm { H _ { 2 } }$ are obtained with the harmonic analysis of PySCF and the isotope masses. The rate constants of $\mathrm { D } ^ { \cdot } + \mathrm { H _ { 2 } } \longrightarrow$ HD + H<sup>·</sup> are calculated with conventional transition-state theory <sup>S21</sup> with rigid-rotor and harmonic-oscillator partition functions and the tunneling correction of a symmetric Eckart barrier<sup>S22</sup> with the zero-point-corrected barrier height and the imaginary frequency of the saddle point. The absolute rates on a model surface with the exact barrier difer from the published accurate quantum-dynamics rates<sup>S23</sup> by factors of 1.2, 1.4, 2.8, and 13 at 1000, 500, 300, and 200 K. Those published rates use a Born–Huang surface, whereas our calculations omit the diagonal Born–Oppenheimer correction, which raises the barrier of this reaction by 0.14 kcal/mol;<sup>S24</sup> the factors therefore mix diferences in the electronic surface with those in the kinetic treatment. We compare surfaces through rates calculated under the same treatment, relative to the TBE surface.

How much of the treatment cancels in such a ratio is itself measurable: replacing the

Eckart correction by the Wigner one changes the ratios quoted here by 0.2% at 1000 K, 1.2% at 500 K, 5.5% at 300 K and 17% at 200 K, so the cancellation is good where tunneling is a correction and only partial at the lowest temperature, where it is not.

## Extrapolation after training

## The schemes and the TBE

The schemes. The direct-extrapolation scheme extrapolates the HF energy and the CCSD(T) correlation energy:

$$
E _ { \mathrm { C B S } } ^ { \mathrm { d i r e c t } } = E _ { X Y } ^ { \infty } [ \mathrm { H F } ] + E _ { X Y } ^ { \mathrm { c o r r , \infty } } [ \mathrm { C C S D ( T ) } ] .\tag{S3}
$$

In both extrapolation schemes (Eqs. 1 and S3), the correlation energy of a QM method M calculated with two basis sets with the cardinal numbers $X > Y$ is extrapolated with the two-point formula of Halkier et al.:<sup>S25</sup>

$$
E _ { X Y } ^ { \mathrm { c o r r } , \infty } [ M ] = \frac { X ^ { 3 } E _ { X } ^ { \mathrm { c o r r } } [ M ] - Y ^ { 3 } E _ { Y } ^ { \mathrm { c o r r } } [ M ] } { X ^ { 3 } - Y ^ { 3 } } ,\tag{S4}
$$

where $E _ { X } ^ { \mathrm { c o r r } } [ M ] = E ( M / X ) - E ( \mathrm { H F } / X )$ , and the HF energy with the two-point formula developed for it:<sup>S26</sup>

$$
E _ { X Y } ^ { \infty } [ \mathrm { H F } ] = \frac { e ^ { - \alpha Y } E ( \mathrm { H F } / X ) - e ^ { - \alpha X } E ( \mathrm { H F } / Y ) } { e ^ { - \alpha Y } - e ^ { - \alpha X } } ,\tag{S5}
$$

with $\alpha = 1 . 6 3$ . For X = 4 and Y = 3, both schemes reduce to the same linear combination,

$$
E _ { \mathrm { C B S } } \approx E ( { \mathrm { C C S D } ( \mathrm { T } ) } / { Y } ) \ [ + \Delta E _ { \mathrm { F C I } } ] + 1 . 7 3 \Delta _ { Y X } [ M ] - 0 . 4 9 \Delta _ { Y X } [ \mathrm { H F } ] ,\tag{S6}
$$

where the optional $\Delta E _ { \mathrm { F C I } } = E ( \mathrm { f u l l ~ C I } / Y ) - E ( \mathrm { C C S D ( T ) } / Y )$ is added to both schemes for the $\mathrm { H } ^ { \cdot } + \mathrm { H } _ { 2 }$ reaction (p. S24), for which full CI is afordable, so that both target full $\mathrm { C I } / \mathrm { C B S } ; \Delta _ { Y X } [ M ] = E ( M / X ) - E ( M / Y )$ is the change of the energy of a QM method M from the smaller to the larger basis set, and M is CCSD(T) in the direct-extrapolation scheme and MP2 in the focal-point scheme; the coeficients are rounded from $6 4 / 3 7 = 1 . 7 2 9 7$ and $6 4 / 3 7 - e ^ { \alpha } / ( e ^ { \alpha } - 1 ) = 0 . 4 8 6 1$ . Only these changes are amplified: an error of the MLP that is the same for the two basis sets of a QM method cancels in them (SI, Tables S5 and S6), and what remains is the error of the first term, which enters once.

Table S5: Errors of the MLP (the ensemble’s averaged prediction) against true QM on the validation set of the $\mathrm { H } ^ { \cdot } + \mathrm { H } _ { 2 }$ reaction for pairs of training QM methods: the MAE of the first and of the second QM method, the MAE of their diference, and the correlation coeficient of the two errors; MAEs in kcal/mol. A diference whose error is smaller than that of either QM method means an error shared by both, which the schemes do not amplify.
<table><tr><td>pair of QM methods</td><td>first</td><td>second</td><td>difference</td><td>correlation</td></tr><tr><td>full  $\overline { { \mathrm { C I / c c - p V T Z - C C S D ( T ) / c c - p V T Z } } }$ </td><td>0.010</td><td>0.010</td><td>0.006</td><td>0.95</td></tr><tr><td> $\mathrm { C C S D ( T ) / c c \mathrm { - } p V T Z \mathrm { ~ - ~ } C C S D / c c \mathrm { - } p V T Z }$ </td><td>0.010</td><td>0.014</td><td>0.015</td><td>0.67</td></tr><tr><td> $\mathrm { C C S D ( T ) / c c \mathrm { - } p V Q Z \mathrm { ~ - ~ } C C S D / c c \mathrm { - } p V Q Z }$ </td><td>0.011</td><td>0.013</td><td>0.013</td><td>0.56</td></tr><tr><td> $\mathrm { M P 2 / c c { \mathrm { - } } p V Q Z \mathrm { ~ - ~ } M P 2 / c c { \mathrm { - } } p V T Z }$ </td><td>0.012</td><td>0.010</td><td>0.005</td><td>0.95</td></tr><tr><td> $\mathrm { C C S D ( T ) / c c \mathrm { - } p V Q Z \mathrm { ~ - ~ } C C S D ( T ) / c c \mathrm { - } p V T Z }$ </td><td>0.011</td><td>0.010</td><td>0.004</td><td>0.98</td></tr><tr><td> $\mathrm { H F / c c - p V Q Z - H F / c c - p V T Z }$ </td><td>0.014</td><td>0.010</td><td>0.008</td><td>0.88</td></tr></table>

Table S6: Errors of the MLP (the ensemble’s averaged prediction) against true QM on the validation set of the ammonia inversion for pairs of training QM methods: the MAE of the first and of the second QM method, the MAE of their diference, and the correlation coeficient of the two errors; MAEs in kcal/mol. A diference whose error is smaller than that of either QM method means an error shared by both, which the schemes do not amplify.
<table><tr><td>pair of QM methods</td><td>first</td><td>second</td><td>difference</td><td>correlation</td></tr><tr><td> $\mathrm { C C S D ( T ) / a u g - c c \mathrm { - } p V T Z \mathrm { ~ - ~ } C C S D / a u g \mathrm { - } c c \mathrm { - } p V T Z }$ </td><td>0.008</td><td>0.003</td><td>0.007</td><td>0.61</td></tr><tr><td> $\mathrm { C C S D ( T ) / a u g - c c \mathrm { - } p V Q Z \mathrm { ~ - ~ } C C S D / a u g \mathrm { - } c c \mathrm { - } p V Q Z }$ </td><td>0.009</td><td>0.007</td><td>0.002</td><td>0.97</td></tr><tr><td> $\mathrm { M P 2 / a u g - c c - p V Q Z - M P 2 / a u g - c c - p V T Z }$ </td><td>0.004</td><td>0.002</td><td>0.004</td><td>0.50</td></tr><tr><td> $\mathrm { C C S D ( T ) / a u g - c c - p V Q Z - C C S D ( T ) / a u g - c c - p V T Z }$ </td><td>0.009</td><td>0.008</td><td>0.004</td><td>0.88</td></tr><tr><td> $\mathrm { H F / a u g  – c c – p V Q Z - H F / a u g – c c – p V T Z }$ </td><td>0.008</td><td>0.003</td><td>0.008</td><td>0.34</td></tr></table>

The TBE. The TBE is the direct-extrapolation scheme with the next larger pair of basis sets $( X = 5 , Y = 4 )$ , so it shares its rule with one of the two schemes it scores; its focalpoint form difers from it by 0.012 kcal/mol on average over the validation geometries of the $\mathrm { H } ^ { \cdot } + \mathrm { H } _ { 2 }$ reaction, enough to unsettle the ordering of the two schemes but not their distance from the training QM methods. For the $\mathrm { H } ^ { \cdot } + \mathrm { H } _ { 2 }$ reaction, $\Delta E _ { \mathrm { F C I } }$ enters the schemes and the TBE alike from cc-pVTZ; with cc-pVQZ instead, its contribution to the barrier difers by 0.001 kcal/mol.

The schemes applied to true QM energies. Tables S7 and S8 give the barrier with every QM method and basis set, and Table S9 the barriers from the two schemes and the

TBE. All values in these tables are calculated with true QM methods (PySCF; frozen core for ammonia).  
Table S7: Barrier of the $\mathrm { H } ^ { \cdot } + \mathrm { H } _ { 2 } \longrightarrow \mathrm { H } _ { 2 } + \mathrm { H } ^ { \cdot }$ reaction in kcal/mol calculated with true QM methods at one saddle-point geometry $( r _ { \mathrm { H H } } = 0 . 9 3 0 \mathrm { ~ \AA } )$ relative to $\mathrm { H } ^ { \cdot } + \mathrm { H } _ { 2 } \left( r _ { \mathrm { H H } } = 0 . 7 4 1 4 \ : \mathring { \mathrm { A } } \right)$ ; unrestricted HF reference for the open-shell species. The exact barrier is 9.61 kcal/mol. Full CI is exact for a given basis set, i.e., its error is only due to the incompleteness of the basis set. Dashes: not calculated.
<table><tr><td colspan="5">cc-pVXZ</td><td colspan="4">aug-cc-pVXZ</td></tr><tr><td>QM method</td><td>DZ</td><td>TZ</td><td>QZ</td><td>5Z</td><td>DZ</td><td>TZ</td><td>QZ</td><td>5Z</td></tr><tr><td>HF</td><td>16.92</td><td>17.56</td><td>17.61</td><td>17.61</td><td>16.81</td><td>17.57</td><td>17.61</td><td>17.61</td></tr><tr><td>MP2</td><td>13.22</td><td>13.27</td><td>13.09</td><td>13.03</td><td>12.56</td><td>13.10</td><td>13.01</td><td>12.99</td></tr><tr><td>CCSD</td><td>10.41</td><td>10.30</td><td>10.10</td><td>10.05</td><td>9.49</td><td>10.07</td><td>10.02</td><td>10.02</td></tr><tr><td>CCSD(T)</td><td>10.22</td><td>10.01</td><td>9.79</td><td>9.73</td><td>9.26</td><td>9.77</td><td>9.70</td><td>9.70</td></tr><tr><td>full CI</td><td>10.15</td><td>9.94</td><td>9.71</td><td></td><td>9.18</td><td>9.69</td><td>9.63</td><td></td></tr></table>

Table S8: Inversion barrier of ammonia in kcal/mol calculated with true QM methods (frozen core) at the reference geometries of the original focal-point analysis.
<table><tr><td colspan="4">cc-pVXZ</td><td colspan="5">aug-cc-pVXZ</td></tr><tr><td>QM method</td><td>DZ</td><td>TZ</td><td>QZ</td><td>5Z</td><td>DZ</td><td>TZ</td><td>QZ</td><td>5Z</td></tr><tr><td>HF</td><td>7.16</td><td>5.30</td><td>4.97</td><td>4.72</td><td>5.06</td><td>4.71</td><td>4.66</td><td>4.65</td></tr><tr><td>MP2</td><td>7.98</td><td>5.92</td><td>5.37</td><td>4.96</td><td>5.31</td><td>5.06</td><td>4.88</td><td>4.81</td></tr><tr><td>CCSD</td><td>8.27</td><td>6.18</td><td>5.63</td><td>5.29</td><td>5.67</td><td>5.42</td><td>5.23</td><td>5.17</td></tr><tr><td>CCSD(T)</td><td>8.51</td><td>6.36</td><td>5.78</td><td>5.41</td><td>5.80</td><td>5.53</td><td>5.34</td><td>5.28</td></tr></table>

Table S9: Barriers from the two extrapolation schemes (Eqs. S3 and 1) applied to the true QM energies with the TZ and QZ basis sets, and the theoretical best estimate (TBE), Eq. S3 with the QZ and 5Z basis sets, all in kcal/mol. For the $\mathrm { H } ^ { \cdot } + \mathrm { H } _ { 2 }$ reaction, all three include the diference between full CI and CCSD(T) calculated with the TZ basis set (Eq. S6), so that they approximate full CI/CBS; with the QZ basis set, this diference changes the barrier by 0.001 kcal/mol. The exact barrier of this reaction is 9.6089 kcal/mol: the TBE is 0.02 kcal/mol from it with cc-pVXZ, and the focal-point scheme happens to be closer at this one geometry (0.01 kcal/mol). The TBE is the reference of the SI because it can be calculated at every geometry; the exact value is known only for the barrier. Ammonia: at the reference geometries of Table S8.
<table><tr><td rowspan="2"></td><td rowspan="2">H +  $\overline { { \mathrm { { H } _ { 2 } } } }$  cc-pVXZ</td><td rowspan="2">(with full CI − CCSD(T)) aug-cc-pVXZ</td><td colspan="2">ammonia</td></tr><tr><td>cc-pVXZ</td><td>aug-cc-pVXZ</td></tr><tr><td>direct extrapolation</td><td>9.53</td><td>9.56</td><td>5.52</td><td>5.22</td></tr><tr><td>focal point</td><td>9.60</td><td>9.52</td><td>5.56</td><td>5.25</td></tr><tr><td>TBE</td><td>9.59</td><td>9.61</td><td>5.23</td><td>5.23</td></tr></table>

## Training data

The training set of each system has two parts: grid and random geometries, both generated before any QM calculation, and geometries added by active learning. The validation set and the held-out test set are generated in the same way as the first part.

Grid and random geometries. For the $\mathrm { H } ^ { \cdot } + \mathrm { H } _ { 2 }$ reaction, the geometries are generated from the three H–H distances without any QM calculation: the collinear minimum-energy path of a London–Eyring–Polanyi–Sato surface<sup>S27</sup> through the saddle point, the antisymmetric and symmetric stretches and the bending at the saddle point, the $\mathrm { H _ { 2 } }$ potential curve, $\mathrm { H _ { 2 } }$ at several bond lengths with the third atom beyond the 5.2 <sup>˚</sup>A radial cutof of the MLP (the asymptote from which the barrier is measured), and random geometries with two H–H distances between 0.45 and 4.5 $\textup { \AA }$ and the angle between them between 40 and 180<sup>◦</sup>. Geometries more than 1.5 eV (34.6 kcal/mol) above $\mathrm { H } _ { 2 } + \mathrm { H } ^ { \cdot }$ on this surface or without any H–H distance below 1.35 $\textup { \AA }$ are discarded, which keeps the region visited by the reaction at every temperature considered here (the barrier is 0.42 eV and the thermal energy at 2000 K 0.17 eV) and in which single-reference methods, and hence the schemes, are defined.

For ammonia, the geometries are the inversion path, with the N–H distance interpolated linearly between the pyramidal and planar stationary points of the original focal-point analysis,<sup>S8</sup> the symmetric stretch at both stationary points, the umbrella angle at the equilibrium N–H distance, and random displacements of geometries along the path, with every N–H distance between 0.80 and 1.35 $\textup { \AA }$ . Each system has 4000 training geometries. For the $\mathrm { H } ^ { \cdot } + \mathrm { H } _ { 2 }$ reaction, 300 further geometries are sampled around the CCSD(T)/cc-pVTZ saddle point from the harmonic quantum Boltzmann distribution at 300 and 1000 K, with the sign of the imaginary frequency flipped, so that the training set describes the curvature of the potential around the saddle point.

Active learning. The second part of each training set is selected by active learning with MLatom’s standard procedure<sup>S1</sup> (p. S8). The $\mathrm { M L P }$ of this active learning is trained on one QM method, MP2 with the smaller basis set: MP2/cc-pVTZ for the $\mathrm { H } ^ { \cdot } + \mathrm { H } _ { 2 }$ reaction and MP2/aug-cc-pVTZ for ammonia. The initial data set has 250 geometries, from Wigner sampling at the minimum, Wigner sampling at the saddle point with the sign of the imaginary frequency flipped, and harmonic quantum Boltzmann sampling at the saddle point. The uncertainty threshold is MLatom’s default for the $\mathrm { H } ^ { \cdot } + \mathrm { H } _ { 2 }$ reaction; for ammonia, it is the median of the default thresholds of five MLPs trained on the initial data set with diferent validation sets. For the $\mathrm { H } ^ { \cdot } + \mathrm { H } _ { 2 }$ reaction, the loop is followed by rounds in which the 50 most uncertain geometries of 2.5-ps molecular-dynamics runs are added, until the error on a fixed test set of such geometries stops decreasing; this gives 864 geometries. For ammonia, we use the data set after 11 iterations, 830 geometries, 9 of them added in the last iteration. All these geometries are then calculated with the training QM methods in the same way as the grid and random ones.

Validation set. The validation sets, 274 geometries for the $\mathrm { H } ^ { \cdot } + \mathrm { H } _ { 2 }$ reaction and 296 for ammonia, were not used in training, but they were used during development: to choose between variants of the training (the extra loss terms for $\mathrm { H } ^ { \cdot } + \mathrm { H } _ { 2 }$ , the scheme term and the final ensemble for ammonia) and to accept the final ensembles. The errors on them are therefore validation errors, not test errors. They are generated in the same way as the grid and random training geometries: the points of the reaction or inversion path and of the stretches and bends lie halfway between the training points, and the random geometries are drawn with a diferent random seed. None of them lies within 0.001 $\textup { \AA }$ (root-mean-square displacement per atom in their common generated orientation, without rigid alignment) of a grid/random training geometry, and their median distance to the nearest one is $0 . 0 5 \left( \mathrm { H } ^ { \cdot } + \mathrm { H } _ { 2 } \right)$ and 0.04 $\textup { \AA }$ (ammonia).

Held-out test set. For each system, a held-out test set was drawn after the development had finished, and it has not been used for any decision. It is drawn with the random part of the same procedure as the validation set, with a new random seed, and every geometry is at least 0.01 <sup>˚</sup>A (root-mean-square displacement per atom after alignment) from every geometry used before: training, validation, active-learning and other geometries calculated during development. This leaves 204 geometries for the $\mathrm { H } ^ { \cdot } + \mathrm { H } _ { 2 }$ reaction and 297 for ammonia. They are calculated with the training QM methods and the TBE, with energies only. For the $\mathrm { H } ^ { \cdot } + \mathrm { H } _ { 2 }$ reaction, four geometries lack an energy of the TBE (calculations not converged) and 11 lie above the energy cap of the training set, so 189 are used for the test.

Geometries more than 34.6 kcal/mol $\left( \mathrm { H } ^ { \cdot } + \mathrm { H } _ { 2 } \right)$ or 42.9 kcal/mol (ammonia) above the reference geometry (p. S2) with CCSD(T)/cc-pVTZ or CCSD(T)/aug-cc-pVTZ are left out of training and testing: 178 of the 4300 geometries of the $\mathrm { H } ^ { \cdot } + \mathrm { H } _ { 2 }$ reaction, and 8 of its 282 validation geometries, leaving 274; with the active-learning geometries (2 of the 864 above the cap for the reaction, 3 of the 830 for ammonia), the MLPs are trained on 4970 $\left( \mathrm { H } ^ { \cdot } + \mathrm { H } _ { 2 } \right)$ and 4827 (ammonia) geometries. The 42.9 kcal/mol ammonia cutof spans energies well above the inversion barrier.

QM methods of each geometry. Training uses diferent numbers of labels for each QM method (Table S1). We put all geometries in one random order and select a first part of the available labels for each method. Within each basis set, every geometry used to train on a more expensive method is also used to train on the cheaper methods; the subsets need not be nested across basis sets. For the $\mathrm { H } ^ { \cdot } + \mathrm { H } _ { 2 }$ reaction, 4967 training geometries are used for training with HF and MP2/cc-pVTZ, 60% with CCSD and CCSD(T)/cc-pVTZ, 25% with full CI/cc-pVTZ, 63% with HF and MP2/cc-pVQZ, and 38% with CCSD and $\mathrm { C C S D ( T ) / c c { \cdot } p V Q Z }$ . For ammonia, all 4827 training geometries are used for training with HF and MP2/aug-cc-pVTZ, 60% with CCSD and CCSD(T)/aug-cc-pVTZ, 50% with HF and MP2/aug-cc-pVQZ, and 30% with CCSD and CCSD(T)/aug-cc-pVQZ. The reference geometry (p. S2) is in the training set of every QM method.

## Ammonia

Validation set. Figure S1 shows the energy errors of the individual validation geometries. On the 179 of the 296 validation ammonia geometries that are farther than 0.01 <sup>˚</sup>A (root-mean-square displacement per atom after alignment) from every training geometry, the mean absolute errors of the focal-point and direct MLPs against the TBE are 0.016 and 0.018 kcal/mol, against 0.010 and 0.012 for the same schemes on true QM and 0.084 for CCSD(T)/aug-cc-pVQZ. On the full set of 296 (Table S11), focal point has the smaller mean absolute error (0.015 against 0.017 kcal/mol), whereas direct extrapolation is closer to the TBE for the barrier (5.241 against 5.274 kcal/mol; TBE 5.225) and the one-dimensional splitting (1.00 against 0.97 times the TBE). The TBE is the direct scheme with aug-cc-pVQZ and aug-cc-pV5Z, so it shares the aug-cc-pVQZ energies and the form of the extrapolation with the direct scheme; this does not afect the comparison with the training QM methods, but it favours the direct scheme in a comparison of the two schemes with each other. Their ordering also depends on whether a barrier or the surrounding surface is compared (SI, Tables S12 and S11). Held-out test set. The errors on the held-out test set of ammonia (p. S16) are 0.017 (focal point) and 0.018 kcal/mol (direct) against the TBE, against 0.010 and 0.013 for the same schemes on true QM and 0.085 for CCSD(T)/aug-cc-pVQZ, the most accurate training QM method (Table S10). Barrier and surface. Figure S2 gives the complete method/basis barrier comparison. The barrier alone can conceal errors in a training method. In true QM, CCSD/aug-cc-pVQZ gives 5.226 kcal/mol, close to the 5.225 kcal/mol TBE, because the perturbative-triples correction raises the barrier by 0.105 kcal/mol while the subsequent basis-set correction lowers it by 0.106 kcal/mol. The CCSD/aug-cc-pVQZ profile nevertheless difers from the

Table S10: Held-out test set of the ammonia inversion: 297 geometries never used during development. Relative energies against the TBE, on every surface measured from that surface’s own energy at the reference geometry: MAE and maximum absolute error in kcal/mol of the true QM calculations and of the MLP (the ensemble’s averaged prediction), and the MAE of the MLP against the same QM method or scheme in true QM.
<table><tr><td rowspan="3"></td><td colspan="2">true QM</td><td colspan="2">the MLP</td><td rowspan="3">MLP vs true QM</td></tr><tr><td>MAE</td><td>max</td><td>MAE</td><td>max</td></tr><tr><td>0.010</td><td>0.035</td><td>0.017</td><td>0.103</td></tr><tr><td>focal-point scheme</td><td></td><td></td><td></td><td></td><td>0.017</td></tr><tr><td>direct-extrapolation scheme</td><td>0.013</td><td>0.071 0.275</td><td>0.018</td><td>0.070 0.271</td><td>0.015</td></tr><tr><td> $\mathrm { C C S D ( T ) / a u g { \mathrm { - } } c c { \mathrm { - } } p V Q Z }$   $\mathrm { M P 2 / a u g – c c – p V T Z }$ </td><td>0.085 0.092</td><td>0.400</td><td>0.083</td><td>0.399</td><td>0.010</td></tr><tr><td> $\mathrm { C C S D / a u g \mathrm { - } c c \mathrm { - } p V T Z }$ </td><td>0.101</td><td>0.363</td><td>0.092 0.101</td><td></td><td>0.003</td></tr><tr><td> $\mathrm { C C S D / a u g \mathrm { - } c c \mathrm { - } p V Q Z }$ </td><td>0.133</td><td>0.730</td><td>0.133</td><td>0.375 0.706</td><td>0.003</td></tr><tr><td> $\mathrm { M P 2 / a u g - c c - p V Q Z }$ </td><td>0.171</td><td>0.723</td><td></td><td></td><td>0.008</td></tr><tr><td></td><td></td><td>0.816</td><td>0.171</td><td>0.720</td><td>0.005</td></tr><tr><td> $\mathrm { C C S D ( T ) / a u g \mathrm { - c c \mathrm { - } p V T Z } }$ </td><td>0.266 1.227</td><td>6.068</td><td>0.264 1.229</td><td>0.818</td><td>0.009</td></tr><tr><td> $\mathrm { H F / a u g - c c - p V T Z }$ </td><td></td><td></td><td></td><td>6.070</td><td>0.003</td></tr><tr><td> $\scriptstyle { \mathrm { H F / a u g - c c - p V Q Z } }$ </td><td>1.291</td><td>6.231</td><td>1.295</td><td>6.243</td><td>0.010</td></tr></table>

Table S11: Relative energies of the 296 geometries of the validation set (used during development, not a test set) of the ammonia inversion against the TBE, on every surface measured from that surface’s own energy at the reference geometry, for the MLP as for the QM methods: MAE and maximum absolute error in kcal/mol of the true QM calculations and of the MLP (the ensemble’s averaged prediction), and the MAE of the MLP against the same QM method or scheme in true QM.
<table><tr><td rowspan="3"></td><td colspan="2">true QM</td><td colspan="2">the MLP</td><td rowspan="3">MLP vs true QM</td></tr><tr><td>MAE</td><td>max</td><td>MAE</td><td>max</td></tr><tr><td>0.010</td><td>0.040</td><td>0.015</td><td>0.123</td></tr><tr><td>focal-point scheme</td><td></td><td>0.069</td><td></td><td></td><td>0.015</td></tr><tr><td>direct-extrapolation scheme</td><td>0.011 0.077</td><td>0.305</td><td>0.017</td><td>0.089</td><td>0.013</td></tr><tr><td> $\mathrm { C C S D ( T ) / a u g { \mathrm { - } } c c { \mathrm { - } } p V Q Z }$   $\mathrm { M P 2 / a u g – c c – p V T Z }$ </td><td>0.092</td><td>0.323</td><td>0.076 0.092</td><td>0.294 0.319</td><td>0.009</td></tr><tr><td> $\mathrm { C C S D / a u g \mathrm { - } c c \mathrm { - } p V T Z }$ </td><td>0.097</td><td>0.381</td><td>0.096</td><td>0.375</td><td>0.002</td></tr><tr><td> $\mathrm { C C S D / a u g \mathrm { - } c c \mathrm { - } p V Q Z }$ </td><td>0.118</td><td>0.639</td><td>0.118</td><td></td><td>0.003</td></tr><tr><td> $\mathrm { M P 2 / a u g - c c - p V Q Z }$ </td><td>0.154</td><td>0.630</td><td>0.154</td><td>0.622</td><td>0.007</td></tr><tr><td> $\mathrm { C C S D ( T ) / a u g \mathrm { - c c \mathrm { - } p V T Z } }$ </td><td>0.239</td><td>0.896</td><td>0.237</td><td>0.652 0.892</td><td>0.004</td></tr><tr><td> $\mathrm { H F / a u g - c c - p V T Z }$ </td><td>1.091</td><td>5.346</td><td>1.092</td><td>5.351</td><td>0.008</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>0.003</td></tr><tr><td> $\scriptstyle { \mathrm { H F / a u g - c c - p V Q Z } }$ </td><td>1.150</td><td>5.540</td><td>1.152</td><td>5.532</td><td>0.008</td></tr></table>

TBE by 0.505 kcal/mol at $8 6 ^ { \circ }$ , and its mean absolute error over the 297 held-out test geometries is 0.133 kcal/mol (Figure 3c; 0.118 on the 296 validation geometries). Agreement for the barrier alone therefore does not establish accuracy over the surrounding surface.

symbols: every validation geometry MAE, all 274   
CCSD(T)/cc-pVQZ (true QM) 0.120   
direct, the MLP 0.025   
— focal, the MLP 0.042

![](images/434e4565a7d1a3c5e2567bdcb3d5d84fd5f46da514e18380d07fd9fd92478d40.jpg)

![](images/c586299ed64d07f5dbedc15b1e5b374da5022ec38674c62efdf25cf10d8e763d.jpg)  
symbols: every validation geometry MAE, all 296 CCSD(T)/aug-cc-pVQZ (true QM) 0.077 direct, the MLP 0.017 — focal, the MLP 0.015

Figure S1: The MLP over the validation geometries. Errors against the TBE of the relative energies of the validation geometries (the reaction path, the saddle region, the reactants and thermal distortions around them; lines for the series along the two paths, symbols for every geometry) of the $\mathrm { H } ^ { \cdot } + \mathrm { H } _ { 2 }$ reaction (a; against the antisymmetric stretch, zero at the saddle point) and of the inversion of ammonia (b; against the H–N–H angle), for the MLP with both schemes and for the most accurate training QM method, CCSD(T) with the larger basis set, in true QM (filled symbols: training QM method; open symbols: unseen schemes); below each panel, the MAE over all validation geometries. The dip of the direct-extrapolation scheme at the saddle point in (a) is that of the scheme itself, which the MLP reproduces (−0.06 kcal/mol in true QM).

Inversion profile. The focal-point MLP follows the relaxed inversion profile more closely than the best training QM method, CCSD(T)/aug-cc-pVQZ, up to about 12 kcal/mol above the minimum. This comparison is limited by geometry: at $8 8 ^ { \circ }$ H–N–H (11.7 kcal/mol on the TBE profile), its error is −0.085 against −0.092 kcal/mol for CCSD(T)/aug-cc-pVQZ, but at $8 6 ^ { \circ }$ (14.4 kcal/mol) it is −0.134 against −0.104 kcal/mol (Figure 3e); by $7 8 ^ { \circ }$ , outside the panel’s range, it reaches −0.52 kcal/mol. Only 82 active-learning training geometries lie below $9 3 ^ { \circ }$ , down to 86<sup>◦</sup>, whereas the same focal-point scheme on true QM energies stays within 0.016 kcal/mol of the TBE over this inner-wall range (p. S10). Extrapolation in the QM-method space therefore does not establish accuracy at arbitrary geometries.

Inversion splittings. On the relaxed inversion path, the one-dimensional splittings of $\mathrm { N H _ { 3 } }$ on the focal-point and direct MLP surfaces are 0.97 and 1.00 times that on the TBE surface, against 0.98 and 1.01 for the schemes on true QM and 0.89 for the MLP at $\mathrm { C C S D ( T ) / a u g \mathrm { - } }$ cc-pVQZ (0.90 in true QM; SI, Table S12). The one-dimensional treatment gives $1 . 4 4 ~ \mathrm { { c m } ^ { - 1 } }$ on the TBE surface (p. S10).

![](images/06449bbff98246ff7cbcd5010b6ec57e43245baffed41f208c6580974db61d2a.jpg)  
Figure S2: Two extrapolation schemes applied to the same training QM methods, in true QM and on the MLP. (a) The barrier of the $\mathrm { H } ^ { \cdot } + \mathrm { H } _ { 2 }$ reaction in kcal/mol with every QM method and the cc-pVXZ basis sets at the benchmark saddle geometry $( r _ { \mathrm { H H } } = 0 . 9 3 0 \ \mathrm { \AA } )$ ,<sup>S28</sup> in parentheses and in color the error against the exact barrier of 9.6089 kcal/mol; the training QM methods are framed (full CI with cc-pV5Z was not calculated). Below the table: the TBE and both schemes applied to the true QM energies and to the ${ \mathrm { M L P } } { \mathrm { } } { \mathrm { } } { \mathrm { s } } .$ all including $\Delta E _ { \mathrm { F C I } }$ . (b) The same for the inversion barrier of ammonia with the augmented basis sets at the geometries of the original focal-point analysis, <sup>S8</sup> with the errors against the TBE.

For comparison with experiment, the splittings are calculated in full dimension with ringpolymer instanton theory<sup>S17</sup> (p. S9; SI, Table S13) for $\mathrm { N H _ { 3 } , \ N D _ { 3 } , }$ and $^ { 1 5 } \mathrm { N } \mathrm { H } _ { 3 } ,$ against the measured 0.793,<sup>S29</sup> 0.053,<sup>S30</sup> and $0 . 7 5 8 \mathrm { c m } ^ { - 1 }$ .<sup>S31</sup> On the focal-point, direct and CCSD(T)/augcc-pVQZ surfaces of the MLP, they are 0.754, 0.764 and $0 . 6 6 9 \mathrm { c m } ^ { - 1 }$ for $\mathrm { N H _ { 3 } , 0 . 0 4 7 3 , 0 . 0 4 8 7 }$ and $0 . 0 4 1 1 \mathrm { c m } ^ { - 1 }$ for $\mathrm { N D } _ { 3 } ,$ and 0.719, 0.730 and 0.638 cm<sup>−1</sup> for $^ { 1 5 } \mathrm { N } \mathrm { H _ { 3 } } ;$ both schemes raise the splitting of the best training QM method by 13–19 %. The remaining diference from experiment is not a test of the surfaces: they are valence-only Born–Oppenheimer surfaces without core–valence and relativistic corrections, and on one-dimensional ammonia double wells the instanton approximation itself is 7–10 % above the exact splitting, while its full-dimensional error for ammonia has not been benchmarked.

Table S12: Inversion barrier (kcal/mol) and splittings $\left( \mathrm { c m } ^ { - 1 } \right)$ of ammonia and its isotopologues on every surface, in true QM and from the MLP (the ensemble), with the same one-dimensional treatment of the relaxed inversion path; in parentheses the ratio to the splitting of the true QM TBE surface, so that every surface is compared under the same treatment. The measured splitting of $\mathrm { N H _ { 3 } }$ is 0.79 cm<sup>−1</sup>.
<table><tr><td></td><td colspan="4">true QM</td><td colspan="7">the MLP</td></tr><tr><td></td><td>barrier</td><td>NH3</td><td></td><td> $\mathrm { N D _ { 3 } }$ </td><td> $\mathrm { ^ { 1 5 } N H _ { 3 } }$ </td><td>barrier</td><td></td><td> $\mathrm { N H _ { 3 } }$ </td><td></td><td> $\mathrm { N D _ { 3 } }$ </td><td> $\mathrm { ^ { 1 5 } N H _ { 3 } }$ </td></tr><tr><td> $\overline { { \mathrm { H F / a u g - c c - p V T Z } } }$ </td><td>4.828</td><td>3.530 (2.45)</td><td></td><td>0.322 (3.29)</td><td>3.377 (2.47)</td><td>4.816</td><td>3.551 (2.47)</td><td></td><td>0.324 (3.31)</td><td></td><td>3.397 (2.48)</td></tr><tr><td> $\mathrm { M P 2 / a u g - c c - p V T Z }$ </td><td>5.053</td><td>1.701 (1.18)</td><td>0.123 (1.26)</td><td></td><td>1.620 (1.18)</td><td>5.048</td><td>1.701 (1.18)</td><td></td><td>0.123 (1.26)</td><td></td><td>1.621 (1.18)</td></tr><tr><td> $\mathrm { C C S D / a u g \mathrm { - c c \mathrm { - } p V T Z } }$ </td><td>5.419</td><td>1.287 (0.89)</td><td></td><td>0.084 (0.86)</td><td>1.223 (0.89)</td><td>5.410</td><td>1.289 (0.89)</td><td></td><td>0.084 (0.86)</td><td></td><td>1.225 (0.89)</td></tr><tr><td>CCSD(T)/aug-cc-pVTZ</td><td>5.532</td><td>1.068 (0.74)</td><td>0.066 (0.67)</td><td></td><td>1.014 (0.74)</td><td>5.533</td><td>1.062 (0.74)</td><td></td><td>0.065 (0.67)</td><td></td><td>1.009 (0.74)</td></tr><tr><td>HF/aug-cc-pVQZ</td><td>4.785</td><td>3.687 (2.56)</td><td></td><td>0.341 (3.49)</td><td>3.528 (2.58)</td><td>4.778</td><td>3.688 (2.56)</td><td></td><td>0.341 (3.48)</td><td>3.529</td><td>(2.58)</td></tr><tr><td> $\mathrm { M P 2 / a u g - c c - p V Q Z }$ </td><td>4.884</td><td>2.019 (1.40)</td><td></td><td>0.155 (1.58)</td><td>1.925 (1.41)</td><td>4.892</td><td>2.008 (1.39)</td><td></td><td>0.154 (1.57)</td><td></td><td>1.915 (1.40)</td></tr><tr><td> $\mathrm { C C S D / a u g \mathrm { - c c \mathrm { - } p V Q Z } }$ </td><td>5.226</td><td>1.562 (1.08)</td><td></td><td>0.109 (1.11)</td><td>1.487 (1.09)</td><td>5.237</td><td>1.553 (1.08)</td><td></td><td>0.108 (1.11)</td><td></td><td>1.479 (1.08)</td></tr><tr><td>CCSD(T)/aug-cc-pVQZ</td><td>5.331</td><td>1.298 (0.90)</td><td></td><td>0.085 (0.87)</td><td>1.234 (0.90)</td><td>5.348</td><td>1.288 (0.89)</td><td></td><td>0.084 (0.86)</td><td></td><td>1.225 (0.89)</td></tr><tr><td>direct-extrapolation scheme</td><td>5.215</td><td>1.456 (1.01)</td><td></td><td>0.099 (1.01)</td><td>1.385 (1.01)</td><td>5.241</td><td>1.443 (1.00)</td><td></td><td>0.098 (1.01)</td><td></td><td>1.373 (1.00)</td></tr><tr><td>focal-point scheme</td><td>5.248</td><td>1.409 (0.98)</td><td></td><td>0.095 (0.97)</td><td>1.340 (0.98)</td><td>5.274</td><td>1.390 (0.97)</td><td></td><td>0.093 (0.95)</td><td></td><td>1.322 (0.96)</td></tr><tr><td>TBE</td><td>5.225</td><td>1.440 (1.00)</td><td></td><td>0.098 (1.00)</td><td>1.370 (1.00)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table S13: Full-dimensional inversion splittings $\left( \mathrm { c m } ^ { - 1 } \right)$ of ammonia and its isotopologues from ring-polymer instanton theory (half ring polymer of N beads at 25 K, networks in double precision) on the final $\mathrm { N H _ { 3 } }$ MLP (five-model ensemble trained on the archived and the active-learning geometries, without the errors of the schemes in the training loss) with the direct and focal-point schemes and at its highest training method, CCSD(T)/aug-cc-pVQZ; beside them, experiment. Change with N: splitting at $N = 5 1 2$ relative to $N = 1 0 2 4$ . On one-dimensional ammonia double wells, the same instanton calculation is 7–10 % above the exact splitting.
<table><tr><td colspan="2">surface</td><td>N = 1024 N = 512</td><td>(%)</td><td>experiment</td><td>ratio</td></tr><tr><td rowspan="3">direct extrapolation</td><td> $\mathrm { N H _ { 3 } }$ </td><td>0.7645</td><td>+1.7</td><td>0.7934</td><td>0.96</td></tr><tr><td> $\mathrm { N D _ { 3 } }$ </td><td>0.04867</td><td>+0.9</td><td>0.053</td><td>0.92</td></tr><tr><td> $^ { 1 5 } \mathrm { N } \mathrm { H _ { 3 } }$ </td><td>0.7295</td><td>+1.7</td><td>0.7577</td><td>0.96</td></tr><tr><td rowspan="3">focal point</td><td> $\mathrm { N H _ { 3 } }$ </td><td>0.7539</td><td>+1.7</td><td>0.7934</td><td>0.95</td></tr><tr><td> $\mathrm { N D _ { 3 } }$ </td><td>0.04730</td><td>+0.9</td><td>0.053</td><td>0.89</td></tr><tr><td> $^ { 1 5 } \mathrm { N } \mathrm { H _ { 3 } }$ </td><td>0.7192</td><td>+1.7</td><td>0.7577</td><td>0.95</td></tr><tr><td rowspan="3">CCSD(T)/aug-cc-pVQZ</td><td> $\mathrm { N H _ { 3 } }$ </td><td>0.6694</td><td>+1.7</td><td>0.7934</td><td>0.84</td></tr><tr><td> $\mathrm { N D _ { 3 } }$ </td><td>0.04105</td><td>+0.9</td><td>0.053</td><td>0.77</td></tr><tr><td> $^ { 1 5 } \mathrm { N } \mathrm { H _ { 3 } }$ </td><td>0.6383</td><td>+1.7</td><td>0.7577</td><td>0.84</td></tr></table>

Cost of the instanton calculation. For the reported focal-point $\mathrm { N H _ { 3 } }$ splitting, optimization of the five-model MLP surface at 64, 256, and 1024 beads, including a continuation of the final stage, recorded 303, 1319, and 4704 steps, respectively. Counting the bead geometries at each step gives about 5.17 million bead energy-and-force evaluations during optimization; these are repeated evaluations, not distinct training geometries. The single run was allocated four CPU cores (Intel Xeon Gold 6240) for 9.83 h (39.3 allocated core-hours). Its quoted splitting uses analytic Hessians at the 1024 final beads; an i-PI finite-diference bead Hessian was also evaluated as a check. A matched direct-QM runtime is unavailable: evaluating the focal-point combination at every bead would require its energy and gradient components, including CCSD(T)/aug-cc-pVTZ, whose analytic gradient in the PySCF version used here failed the finite-diference check (p. S2).

## Hydrogen exchange: extrapolation to full CI/CBS

The same construction reaches the exact solution when full CI is afordable. For the $\mathrm { H } ^ { \cdot } +$ $\mathrm { H } _ { 2 } \longrightarrow \mathrm { H } _ { 2 } + \mathrm { H }$ <sup>·</sup> reaction, whose exact Born–Oppenheimer barrier is known from quantum Monte Carlo calculations, $9 . 6 0 8 9 \pm 0 . 0 0 0 1 \ \mathrm { k c a l / m o l }$ , <sup>S24,S28,S32,S33</sup> we train one MLP on HF, MP2, CCSD, and CCSD(T) with cc-pVTZ and cc-pVQZ and on full CI/cc-pVTZ, and add $\Delta E _ { \mathrm { F C I } }$ (Eq. S6) to both schemes, so that both target full CI/CBS (p. S13). Even full CI, exact for a given basis set, overestimates the barrier by 0.1 kcal/mol with cc-pVQZ and by 0.3 with cc-pVTZ (SI, Figure S2a). The TBE is the direct scheme with cc-pVQZ and cc-pV5Z plus $\Delta E _ { \mathrm { F C I } } ;$ its barrier, 9.59 kcal/mol, is within 0.02 kcal/mol of the exact one.

The direct and focal-point MLP barriers are 9.54 and 9.61 kcal/mol, against 9.53 and 9.60 kcal/mol for the same schemes on true QM and 9.79 kcal/mol for the best training QM method, CCSD(T)/cc-pVQZ (SI, Table S16). On the 274 validation geometries, the mean absolute errors against the TBE are 0.025 and 0.042 kcal/mol, against 0.023 and 0.037 for the schemes on true QM and 0.120 for CCSD(T)/cc-pVQZ (Table S15); on the 237 of them farther than 0.01 <sup>˚</sup>A from every training geometry, 0.023 and 0.044, against 0.021, 0.039 and 0.114. The errors on the held-out test set of this reaction (p. S16) are 0.022 (direct) and 0.059 kcal/mol (focal point) against the TBE, against 0.019 and 0.047 for the same schemes on true QM and 0.118 for CCSD(T)/cc-pVQZ (Table S14): the focal-point MLP is two times more accurate than the most accurate training QM method, short of the margin of 0.05 kcal/mol at the upper 95 % confidence bound that we set before the calculation, as on the 132 random geometries of the validation set that are at least 0.01 <sup>˚</sup>A (root-meansquare displacement per atom after alignment) from every training geometry (0.052 against 0.109 kcal/mol).

Table S14: Held-out test set of the $\mathrm { H } ^ { \cdot } + \mathrm { H } _ { 2 }$ reaction: 189 geometries never used during development. Relative energies against the TBE, on every surface measured from that surface’s own energy at the reference geometry: MAE and maximum absolute error in kcal/mol of the true QM calculations and of the MLP (the ensemble’s averaged prediction), and the MAE of the MLP against the same QM method or scheme in true QM.
<table><tr><td rowspan="2"></td><td colspan="2">true QM</td><td colspan="2">the MLP</td><td rowspan="2">MLP vs true QM</td></tr><tr><td>MAE</td><td>max</td><td>MAE</td><td>max</td></tr><tr><td>direct-extrapolation scheme</td><td>0.019</td><td>0.070</td><td>0.022</td><td>0.410</td><td>0.014</td></tr><tr><td>focal-point scheme</td><td>0.047</td><td>0.198</td><td>0.059</td><td>0.292</td><td>0.018</td></tr><tr><td> $\mathrm { C C S D ( T ) / c c { \cdot } p V Q Z }$ </td><td>0.118</td><td>0.483</td><td>0.116</td><td>0.458</td><td>0.013</td></tr><tr><td> $\mathrm { C C S D / c c } { \mathrm { - p V Q Z } }$ </td><td>0.184</td><td>0.790</td><td>0.185</td><td>0.857</td><td>0.014</td></tr><tr><td>full  $\mathrm { C I / c c { \mathrm { - } } p V T Z }$ </td><td>0.261</td><td>1.062</td><td>0.261</td><td>1.060</td><td>0.013</td></tr><tr><td> $\mathrm { C C S D ( T ) / c c { \cdot } p V T Z }$ </td><td>0.277</td><td>1.118</td><td>0.274</td><td>1.096</td><td>0.012</td></tr><tr><td> $\mathrm { C C S D / c c } { \mathrm { - p V T Z } }$ </td><td>0.333</td><td>1.370</td><td>0.337</td><td>1.413</td><td>0.017</td></tr><tr><td> $\mathrm { M P 2 / c c { \cdot } p V Q Z }$ </td><td>1.664</td><td>6.846</td><td>1.661</td><td>6.852</td><td>0.013</td></tr><tr><td> $\mathrm { M P 2 / c c { \cdot } p V T Z }$ </td><td>1.713</td><td>6.920</td><td>1.708</td><td>6.914</td><td>0.012</td></tr><tr><td> $\mathrm { H F / c c \mathrm { - } p V T Z }$ </td><td>3.004</td><td>9.432</td><td>3.001</td><td>9.456</td><td>0.011</td></tr><tr><td> $\mathrm { H F / c c { \mathrm { - p V } } Q Z }$ </td><td>3.032</td><td>9.231</td><td>3.022</td><td>9.233</td><td>0.014</td></tr></table>

Rate constants. For $\mathrm { D } ^ { \cdot } + \mathrm { H _ { 2 } } \longrightarrow \mathrm { H D } + \mathrm { H } ^ { \cdot }$ , which runs on the same surface, the rate constants from conventional transition-state theory with an Eckart tunneling correction on the direct and focal-point MLP surfaces are 1.01 and 1.00 (direct, 300 and 1000 K) and 1.01 and 1.01 (focal point) times those on the TBE surface, against 1.09 and 1.03, and 0.94 and 0.98 for the same schemes on true QM, and 0.68 and 0.87 for the MLP at CCSD(T)/ccpVQZ (0.70 and 0.88 in true QM; SI, Table S16). At 200 K, the sensitivity to the tunneling treatment prevents a resolved comparison.

Table S15: Relative energies of the 274 geometries of the validation set (used during development, not a test set) of the $\mathrm { H } ^ { \cdot } + \mathrm { H } _ { 2 }$ reaction against the TBE, on every surface measured from that surface’s own energy at the reference geometry, for the MLP as for the QM methods: MAE and maximum absolute error in kcal/mol of the true QM calculations and of the MLP (the ensemble’s averaged prediction), and the MAE of the MLP against the same QM method or scheme in true QM.
<table><tr><td rowspan="3"></td><td colspan="2">true QM</td><td colspan="2">the MLP</td><td colspan="2"> $\mathrm { M L P \ v s }$ </td></tr><tr><td>MAE</td><td>max</td><td>MAE</td><td>max</td><td>true QM</td><td></td></tr><tr><td>0.023</td><td>0.077</td><td></td><td>0.025</td><td>0.325</td><td>0.011</td></tr><tr><td>direct-extrapolation scheme focal-point scheme</td><td>0.037</td><td>0.288</td><td>0.042</td><td>0.219</td><td></td><td>0.014</td></tr><tr><td> $\mathrm { C C S D ( T ) / c c { \cdot } p V Q Z }$ </td><td>0.120</td><td>0.472</td><td>0.117</td><td></td><td>0.462</td><td>0.011</td></tr><tr><td> $\mathrm { C C S D / c c } { \mathrm { - p V Q Z } }$ </td><td>0.212</td><td>0.823</td><td>0.212</td><td></td><td>0.861</td><td>0.013</td></tr><tr><td>full  $\mathrm { C I / c c { \mathrm { - } } p V T Z }$ </td><td>0.261</td><td>1.028</td><td>0.261</td><td></td><td>1.067</td><td>0.010</td></tr><tr><td> $\mathrm { C C S D ( T ) / c c { \cdot } p V T Z }$ </td><td>0.282</td><td>1.103</td><td></td><td>0.279</td><td>1.103</td><td>0.010</td></tr><tr><td> $\mathrm { C C S D / c c } { \mathrm { - p V T Z } }$ </td><td>0.363</td><td>1.413</td><td></td><td>0.368</td><td>1.425</td><td>0.014</td></tr><tr><td> $\mathrm { M P 2 / c c { \cdot } p V Q Z }$ </td><td>1.660</td><td>8.057</td><td></td><td>1.657</td><td>8.125</td><td>0.012</td></tr><tr><td> $\mathrm { M P 2 / c c { \mathrm { - } } p V T Z }$ </td><td>1.709</td><td>8.100</td><td></td><td>1.706</td><td>8.157</td><td>0.010</td></tr><tr><td> $\mathrm { H F / c c \mathrm { - } p V T Z }$ </td><td>3.234</td><td>9.766</td><td></td><td>3.231</td><td>9.792</td><td>0.010</td></tr><tr><td> $\mathrm { H F / c c { \mathrm { - p V } } Q Z }$ </td><td>3.267</td><td>9.564</td><td></td><td>3.257</td><td>9.574</td><td>0.014</td></tr></table>

Table S16: The collinear saddle point of every surface of the $\mathrm { H } { \cdot } + \mathrm { H } _ { 2 }$ reaction, in true QM and of the MLP (the ensemble, from its energies at the same geometries): barrier in kcal/mol, the imaginary and the bending frequency of H−H−H in $\mathrm { c m } ^ { - 1 }$ , and the rate constant of $\mathrm { D } ^ { \cdot } + \mathrm { H _ { 2 } } \longrightarrow \mathrm { H D } + \mathrm { H } ^ { \cdot }$ from the same transition-state theory with Eckart tunneling as the ratio to that of the TBE surface at 300 and 1000 K; the same treatment is used for every surface, with its residual efect on the ratios assessed in Methods.
<table><tr><td colspan="5">true QM</td><td colspan="4">the MLP</td></tr><tr><td></td><td>barrier</td><td> $\nu ^ { \ddag }$ </td><td>bend</td><td> $\underline { { k / k _ { \mathrm { T B E } } } }$  (300, 500, 1000 K)</td><td>barrier</td><td> $\nu ^ { \ddag }$ </td><td>bend</td><td> $k / k _ { \mathrm { T B E } }$  (300, 1000 K)</td></tr><tr><td> $\overline { { \mathrm { { H F / c c } } \mathrm { { - p V } T Z } } }$ </td><td>17.58</td><td>2298i</td><td>914</td><td>0.00, 0.00, 0.03</td><td>17.56</td><td>2305i</td><td>940</td><td>0.00, 0.02</td></tr><tr><td> $\mathrm { M P 2 / c c { \mathrm { - } } p V T Z }$ </td><td>13.23</td><td>1968i</td><td>930</td><td>0.02, 0.04, 0.16</td><td>13.22</td><td>1988i</td><td>929</td><td>0.02, 0.16</td></tr><tr><td> $\mathrm { C C S D / c c } { \cdot } \mathrm { p V T Z }$ </td><td>10.30</td><td>1539i</td><td>917</td><td>0.31, 0.46, 0.64</td><td>10.29</td><td>1540i</td><td>955</td><td>0.26, 0.58</td></tr><tr><td> $\mathrm { C C S D ( T ) / c c { \cdot } p V T Z }$ </td><td>10.01</td><td>1510i</td><td>917</td><td>0.45, 0.59, 0.74</td><td>10.01</td><td>1521i</td><td>919</td><td>0.45, 0.73</td></tr><tr><td> $\mathrm { H F / c c { \mathrm { - p V } } Q Z }$ </td><td>17.63</td><td>2302i</td><td>903</td><td>0.00, 0.00, 0.03</td><td>17.60</td><td>2311i</td><td>925</td><td>0.00, 0.02</td></tr><tr><td> $\mathrm { M P 2 / c c { \mathrm { - } } p V Q Z }$ </td><td>13.04</td><td>1964i</td><td>905</td><td>0.02, 0.05, 0.19</td><td>13.04</td><td>1978i</td><td>892</td><td>0.03, 0.20</td></tr><tr><td> $\mathrm { C C S D / c c } { \mathrm { - p V Q Z } }$ </td><td>10.10</td><td>1536i</td><td>892</td><td>0.47, 0.60, 0.75</td><td>10.11</td><td>1553i</td><td>912</td><td>0.44, 0.71</td></tr><tr><td> $\mathrm { C C S D ( T ) / c c { \cdot } p V Q Z }$ </td><td>9.79</td><td>1506i</td><td>892</td><td>0.70, 0.79, 0.88</td><td>9.78</td><td>1506i</td><td>894</td><td>0.68, 0.87</td></tr><tr><td>full  $\mathrm { C I / c c { \mathrm { - } } p V T Z }$ </td><td>9.94</td><td>1504i</td><td>917</td><td>0.50, 0.63, 0.76</td><td>9.95</td><td>1520i</td><td>919</td><td>0.50, 0.76</td></tr><tr><td>direct-extrapolation scheme</td><td>9.53</td><td>1493i</td><td>879</td><td>1.09, 1.06, 1.03</td><td>9.54</td><td>1489i</td><td>883</td><td>1.01, 1.00</td></tr><tr><td>focal-point scheme</td><td>9.60</td><td>1490i</td><td>883</td><td>0.94, 0.97, 0.98</td><td>9.61</td><td>1503i</td><td>867</td><td>1.01, 1.01</td></tr><tr><td>TBE</td><td>9.59</td><td>1497i</td><td>880</td><td>1.00, 1.00, 1.00</td><td></td><td></td><td></td><td></td></tr></table>

## The errors of the schemes in the training loss

The schemes amplify the changes of the energies between the two basis sets (Eq. S6), so a small error of the MLP in such a change is magnified in the extrapolated energy. Adding the errors of both schemes to the training loss makes the MLP precise in exactly these combinations, while keeping both schemes available after training (below). It is an option, not a requirement of the approach.

Hydrogen exchange. For the $\mathrm { H } { \cdot } + \mathrm { H } _ { 2 }$ reaction (p. S24), the training loss also contains the errors of both schemes, written as the linear combinations of Eq. S6: at every step, the mean squared diference between each scheme applied to the energies predicted by the MLP and to the reference energies, over 16 random training geometries that carry every QM method the schemes combine, with a coeficient of 1. Both schemes are included, so that either can still be chosen after training. Without this term the direct-extrapolation scheme on the MLP was 0.10 kcal/mol from the same scheme on true QM, because the MLP learned the change of the CCSD(T) energy between the two basis sets less accurately than that of MP2, which comes from all geometries and with forces; with it, 0.02 kcal/mol.

For the $\mathrm { H } ^ { \cdot } + \mathrm { H } _ { 2 }$ reaction, the errors of the energies of CCSD(T) and full CI, which are learned from energies only and from the fewest geometries, are weighted by 3, and those of the reference geometry by 10: with both the scheme and Hessian terms but without these weights, the combinations were accurate but the full CI error rose to 0.04 kcal/mol, while for ammonia every QM method stays within 0.01 kcal/mol without them. For the $\mathrm { H } ^ { \cdot } + \mathrm { H } _ { 2 }$ reaction, whose rate constants depend on the curvature at the saddle point, the loss also contains the Hessians of every training QM method at its saddle point and at the reactants, obtained from finite diferences of its energies (rate constants, p. S24), as a mean squared error with a coeficient of 0.01 once per epoch. With these Hessians the MLP’s error in the bending frequency at the saddle point is 1.5% on average over the eleven surfaces, and in the imaginary one 0.6% (Table S16).

Ammonia. For ammonia, the MLP of the main text is trained without it: on the 296 validation geometries, focal point and direct extrapolation on that MLP are 0.015 and 0.017 kcal/mol from the TBE and 0.015 and 0.013 from the same schemes on true QM; an MLP trained on the same data with the scheme term gives 0.011 and 0.011 against the

TBE and 0.004 and 0.005 against true QM, with barriers difering by less than 0.015 kcal/mol and the same one-dimensional splitting (within 1 %). On the 297 geometries of the held-out test set, the two schemes on the MLP with the scheme term are 0.010 and 0.013 kcal/mol from the TBE, against 0.017 and 0.018 without it. Both are averages over five-model ensembles trained on the same folds; single models without the term scatter more (standard deviation of the focal-point error over the five models 0.005 against 0.001 kcal/mol), so the comparison holds for ensembles, not necessarily for a single model.

## Interpolation along exact exchange

## Training data and active learning

For malonaldehyde, the active learning (p. S8) uses the MLP with the method descriptor a, and every geometry it selects is calculated with all three training functionals. The uncertainty threshold is one value for all three trained values of a; it is determined once, on the initial data set, and kept for the rest of the loop.

The initial data set is built with MLatom’s procedure:<sup>S14</sup> geometries are added 50 at a time until the learning curve of the five-fold cross-validated energy error predicts less than 10 % improvement from 50 more geometries. The geometries come in equal numbers from nine samplings at 300 K: for each training functional, Wigner sampling at its minimum, Wigner sampling at its transition state with the sign of the imaginary frequency flipped, and harmonic quantum Boltzmann sampling at its transition state. Two loops were run. The first started from MLatom’s own initial data set of 350 geometries, with its uncertainty threshold fitted on that set (2.57 kcal/mol), and was stopped after five iterations at 965 geometries. The second started from the first 250 of these geometries, with a fixed threshold, the median of five fits on them (4.09 kcal/mol), and was stopped after seven iterations at 1111 geometries. The final data set is their union, 1826 geometries after removing the 250 shared ones. Neither loop had met the convergence criterion below when stopped. The union was chosen over the data of either loop because its MLP stays intact and free of energy holes in more of the production trajectories (p. S31); these trajectories selected the model rather than testing it.

In every iteration, the MLP runs the following simulations at each of the three trained values of a: four geometry optimizations of the minimum and four of the transition state with geomeTRIC,<sup>S4</sup> started from the samples of the initial data set; the IRC from the transition state of the MLP; and molecular dynamics in the NVE ensemble for 2.5 ps with a time step of 0.5 fs. Sixteen trajectories start from Wigner sampling at the minimum and sixteen at the transition state of the MLP: eight from harmonic quantum Boltzmann sampling with a thermal momentum of random sign along the reaction coordinate, and eight from Wigner sampling with the sign of the imaginary frequency flipped.

A trajectory stops at its first uncertain geometry; an optimization or an IRC runs to the end, and its first uncertain geometry is taken. If an iteration finds fewer uncertain geometries than it has simulations, the simulations are repeated from new initial conditions, up to 25 times. The loop has converged when fewer than 5% of the simulations of an iteration find a new uncertain geometry.<sup>S14</sup> The geometries of the data set are calculated only with the three training functionals; the two functionals not used in training, BLYP35 and BHandHLYP, are calculated only for the test geometries.

## The MLP and the choice of $a _ { \mathrm { o p t } }$

The model. The MLP extends the OMNI-P models from a one-hot QM-method descriptor to a continuous fraction of exact exchange, a, alongside the nuclear coordinates.<sup>S10,S11</sup> Its energy is a sum of atomic energies; the network for each element receives that atom’s chemical environment and a (p. S4). The three training functionals are BLYP (a = 0), B1LYP $( a = 0 . 2 5 )$ , and HFLYP (a = 1). Energies and forces were calculated with cc-pVDZ for 1826 geometries generated by active learning (p. S29). The malonaldehyde proton-transfer barrier is sensitive to exact exchange: the barrier-oriented MPW1K and BB1K functionals contain 42.8% and 42% exact exchange, respectively, compared with 20% for B3LYP. <sup>S34,S35</sup> BLYP35 (a = 0.35) and BHandHLYP (a = 0.5) are withheld named members of the same functional family; the family can also be extended to multiple parameters, as in B3LYP-type hybrids.<sup>S36,S37</sup>

The reported MLP and $a _ { \mathrm { o p t } }$ . All reported results are obtained with an ensemble of five MLPs trained on the 1826 geometries of the final data set with diferent random seeds and training–validation splits, whose predictions are averaged, as in our latest work on vibrational frequencies.<sup>S13</sup> The networks are evaluated in double precision for optimizations, Hessians and instantons. The optimal fraction $a _ { \mathrm { o p t } }$ is the one at which the barrier of the MLP at the two B3LYP stationary geometries equals the CCSD(T) barrier there, found by bisection on the MLP after a scan over 101 values of a between 0 and 1.

Selection of $a _ { \mathrm { o p t } }$ . The selected $a _ { \mathrm { o p t } } = 0 . 4 5 5$ matches the CCSD(T)/cc-pVDZ barrier at two fixed B3LYP/cc-pVDZ stationary geometries using two CCSD(T) single-point energies (Figure 4d). This matched barrier selects the coordinate; it is not an independent validation of the resulting MLP. At each method’s own relaxed minimum and transition state, the training functionals miss the CCSD(T) reference by 1.23 (B1LYP) to 3.33 (HFLYP) kcal/mol, while the optimal MLP misses it by 0.05 kcal/mol (Table S17).

Table S17: Relaxed barriers in kcal/mol: each method’s barrier between its own minimum and transition state, the CCSD(T) barrier in the same basis at the same two geometries, and the diference.
<table><tr><td colspan="3">cc-pVDZ</td></tr><tr><td>method</td><td>own</td><td>CCSD(T) there difference</td></tr><tr><td>BLYP</td><td>1.20</td><td>3.57 -2.36</td></tr><tr><td>B1LYP</td><td>2.65</td><td>3.88 -1.23</td></tr><tr><td>BLYP35</td><td>3.28</td><td>3.91 -0.62</td></tr><tr><td>BHandHLYP</td><td>4.23</td><td>3.87 +0.36</td></tr><tr><td>HFLYP</td><td>6.96</td><td>3.64 +3.33</td></tr><tr><td>MLP at  $a _ { \mathrm { o p t } }$ </td><td>3.94</td><td>3.90 +0.05</td></tr></table>

## Checks along the reaction path and across the method space

We check the MLP in several regions of the potential energy surface, because no single test is enough to judge its quality and each test probes a diferent part of the surface: the IRC between the minimum and the transition state, the barrier of every method between its own relaxed minimum and transition state, molecular dynamics at 300 K, and frequencies, zero-point energies and tunneling splittings across a. Diferent tests can rank the methods diferently: BHandHLYP, for example, has the smallest error along the IRC (Table S18) but an error of 0.36 kcal/mol in its relaxed barrier, against 0.05 kcal/mol for the MLP at $a _ { \mathrm { o p t } }$ (Table S17).

Validation path. The MLP at $a _ { \mathrm { o p t } }$ is validated along the B3LYP/cc-pVDZ IRC (p. S2), leaving out the transition state and the two ends of the path, which are or are near the geometries $a _ { \mathrm { o p t } }$ is chosen on. The two halves of the path are mirror images, so its 24 interior points are 12 distinct geometries. The validation geometries are therefore the same for every method compared, and none of them is chosen by the MLP (Table S18). On the held-out test set, 132 geometries near this path never used during development, the $\mathrm { M L P }$ at $a _ { \mathrm { o p t } }$ has a mean absolute error of 0.43 kcal/mol against CCSD(T), and B1LYP, the most accurate training functional, 0.51; the 95 % confidence interval of the diference, −0.28 to +0.07 kcal/mol (bootstrap over IRC points), includes zero (Table S19; the bootstrap resamples the 12 path points with their 11 displaced geometries each).

Table S18: Errors of every method against CCSD(T)/cc-pVDZ along the B3LYP/cc-pVDZ IRC, the validation set (used during development): mean absolute error (MAE) in kcal/mol of the energies of the interior points of the IRC (the transition state and the two ends left out), relative to the same method at the B3LYP minimum. The geometries are the same for every method and none is chosen by the MLP. The two halves of the IRC are mirror images, so its 24 interior points are 12 symmetry-unique geometries. The two null models need the DFT energies of the trained functionals at every geometry.
<table><tr><td>method</td><td>role</td><td>MAE</td></tr><tr><td>BHandHLYP</td><td>not trained</td><td>0.130</td></tr><tr><td>MLP at  $a _ { \mathrm { o p t } }$ </td><td></td><td>0.138</td></tr><tr><td>real functional at  $a _ { \mathrm { o p t } }$  parabola in a at</td><td></td><td>0.147 0.168</td></tr><tr><td> $a _ { \mathrm { o p t } }$  line B1LYP-HFLYP at</td><td>null model</td><td></td></tr><tr><td> $a _ { \mathrm { o p t } }$ </td><td>null model</td><td>0.200</td></tr><tr><td>BLYP35</td><td>not trained</td><td>0.303</td></tr><tr><td>B1LYP</td><td>trained</td><td>0.455</td></tr><tr><td>HFLYP</td><td>trained</td><td>0.478</td></tr><tr><td>B3LYP</td><td>comparator</td><td>0.631</td></tr><tr><td>BLYP</td><td>trained</td><td>0.838</td></tr></table>

Each method is also judged at its own stationary points: its barrier between its own relaxed minimum and transition state against CCSD(T) at the same two geometries. Two null models, which need DFT at every geometry, are compared as well: the straight line in a between the two trained functionals on either side of $a _ { \mathrm { o p t } }$ , and the parabola through all three trained functionals.

Table S19: Held-out test set of malonaldehyde: 132 geometries never used during development, displaced from the 12 symmetry-unique interior points of the B3LYP/cc-pVDZ IRC by classical normal-mode motion at 50 K. Mean absolute error (MAE) in kcal/mol against CCSD(T)/cc-pVDZ of the energies relative to the same method at the B3LYP minimum.
<table><tr><td>method</td><td>role</td><td>MAE</td></tr><tr><td>BLYP35</td><td>not trained</td><td>0.409</td></tr><tr><td>line B1LYP-HFLYP at  $a _ { \mathrm { o p t } }$  MLP at</td><td>null model</td><td>0.424 0.425</td></tr><tr><td> $a _ { \mathrm { o p t } }$  real functional at</td><td></td><td>0.428</td></tr><tr><td> $a _ { \mathrm { o p t } }$  parabola in a at</td><td></td><td></td></tr><tr><td> $a _ { \mathrm { o p t } }$ </td><td>null model</td><td>0.433</td></tr><tr><td>BHandHLYP</td><td>not trained</td><td>0.468</td></tr><tr><td>B1LYP</td><td>trained</td><td>0.514</td></tr><tr><td>B3LYP</td><td>comparator</td><td>0.696</td></tr><tr><td>BLYP</td><td>trained</td><td>0.969</td></tr><tr><td>HFLYP</td><td>trained</td><td>1.234</td></tr></table>

Checks along the IRC. DFT with the functional at $a _ { \mathrm { o p t } }$ , which is not a training functional, shows how accurately the MLP interpolates in a: the MLP difers from it by 0.01 kcal/mol on the barrier and 0.010 kcal/mol on average along the B3LYP IRC (Tables S22 and S20). The true functional meets the CCSD(T) barrier at $a = 0 . 4 5 5$ (Figure 4d). Table S21 gives the test against CCSD(T) at the B3LYP stationary geometries. At the 12 symmetry-unique IRC geometries, the MLP at $a _ { \mathrm { o p t } }$ has a mean absolute error of 0.138 kcal/mol against CCSD(T), versus 0.455–0.838 for its three training functionals and 0.631 for B3LYP at the same geometries (Figure 4e; Table S18). A straight line in a between the adjacent training functionals gives 0.200 kcal/mol and a parabola through all three gives 0.168 kcal/mol; both require DFT at every geometry.

The MLP’s errors against the training functionals along the IRC are 0.007–0.015 kcal/mol, and against withheld BHandHLYP and BLYP35 are 0.009 and 0.013 kcal/mol (Table S20). The actual functional at $a _ { \mathrm { o p t } }$ has an error against CCSD(T) along the IRC of 0.147 kcal/mol; the nearby withheld BHandHLYP gives 0.130 kcal/mol (Table S18). These checks go beyond the two energies used to select $a ;$ they establish accuracy along the IRC, rather than across all configurations or compounds.

Table S20: The MLP against DFT with the same functional along the B3LYP/cc-pVDZ IRC: mean absolute diference (MAD) in kcal/mol between the MLP at a value of a and DFT with the functional of the same a, over the 12 symmetry-unique interior geometries, for the three training functionals, for the two functionals not used in training (BLYP35 and BHandHLYP; the MLP interpolates in a between the training functionals), and for the functional with $a = a _ { \mathrm { o p t } }$ . The errors of the MLP against CCSD(T) are in Table S18.
<table><tr><td colspan="2">functional MAD</td></tr><tr><td>BLYP</td><td>fit 0.009</td></tr><tr><td>B1LYP</td><td>fit 0.015</td></tr><tr><td>BLYP35</td><td>interpolation 0.013</td></tr><tr><td>BHandHLYP</td><td>interpolation 0.009</td></tr><tr><td>HFLYP</td><td>fit 0.007</td></tr><tr><td>real functional at  $a _ { \mathrm { o p t } }$ </td><td>at  $a _ { \mathrm { o p t } }$  0.010</td></tr></table>

Table S21: The barrier (kcal/mol) at the two B3LYP/cc-pVDZ stationary geometries: the five named functionals of Eq. 2, B3LYP and CCSD(T), each functional’s diference from CCSD(T), and the fraction of exact exchange at which the real functional meets CCSD(T) (linear interpolation on a grid of 0.05 in a).
<table><tr><td></td><td colspan="2">cc-pVDZ</td></tr><tr><td>method</td><td>a</td><td>barrier -CCSD(T)</td></tr><tr><td>BLYP</td><td>0.00</td><td>1.16 -2.64</td></tr><tr><td>B1LYP</td><td>0.25</td><td>2.63 -1.16</td></tr><tr><td>BLYP35</td><td>0.35</td><td>3.21 -0.59</td></tr><tr><td>BHandHLYP</td><td>0.50</td><td>4.04 +0.25</td></tr><tr><td>HFLYP</td><td>1.00</td><td>6.33 +2.53</td></tr><tr><td>B3LYP</td><td></td><td>2.15 -1.65</td></tr><tr><td>CCSD(T)</td><td></td><td>3.80</td></tr><tr><td colspan="3">real functional = CCSD(T) a = 0.455</td></tr></table>

Properties across the space. The same MLP gives relaxed reaction profiles, harmonic frequencies, zero-point energies and tunneling splittings across a. Relative to true DFT at the five named functionals, its relaxed barriers difer by at most 0.03 kcal/mol and its harmonic frequencies at the minimum by 1.8–3.2 cm<sup>−1</sup> on average (Figure S3; Table S22). At $a _ { \mathrm { o p t } }$ , including zero-point energy lowers the 3.94 kcal/mol barrier to 1.29 kcal/mol. In 2.5-ps production dynamics at 300 K, 50, 49 and 45 of 50 trajectories at $a = 0$ , 0.25 and 1 and 47 of 50 at $a _ { \mathrm { o p t } }$ remain intact and free of energy holes at the recomputed snapshots; ensemble disagreement flags every trajectory that breaks or enters a hole before it does. Production dynamics: 50 NVE trajectories of 2.5 ps per value of a (time step 0.5 fs) from 300 K Wigner samples of the $\mathrm { M L P ^ { \prime } s }$ own minimum, on the ensemble average. A trajectory breaks when a hydrogen atom is more than 2 <sup>˚</sup>A from every heavy atom, the heavy atoms are no longer connected (1.3 times the covalent radii), or two atoms come closer than 0.6 <sup>˚</sup>A. Snapshots every 100 fs, and those of largest ensemble disagreement, most extreme bond angle and largest distance from the minimum, are recomputed with the true functional at that a; a trajectory enters an energy hole if the MLP difers from it there by more than 10 kcal/mol, or lies below the B3LYP/cc-pVDZ minimum, the zero of both, while the functional is more than 20 kcal/mol above it. Before the first hole of each such trajectory, every saved snapshot (10 fs apart) is recomputed, to time the hole against the flag. Holes between recomputed snapshots are not seen, so the hole-free counts are upper bounds. The disagreement is the largest deviation of a member from the average; its threshold is the first loop’s, 2.57 kcal/mol.

![](images/4f1fe2a1cdb4fde37744aaf741d411773dde59d12a5c1097e97cc862dbe5610e.jpg)

![](images/890f42fc5467c94084babbb99884c04bf364f85cdd5631fc9476a34e3b9e98a5.jpg)  
Figure S3: The MLP across the space (cc-pVDZ): the relaxed barrier between its own minimum and transition state, with and without the harmonic zero-point energy, and the imaginary frequency at the transition state, on a grid of 21 values of a (lines), against the real functionals at their own stationary points (symbols; the star is the real functional at $a _ { \mathrm { o p t } } )$ ; the numbers are the MLP minus the real functional.

Table S22: Relaxed barrier (kcal/mol), barrier with the harmonic zero-point energy, and imaginary frequency (cm<sup>−1</sup>) at each method’s own stationary points, from the real functional and from the MLP at the same a $\left( \mathrm { c c - p V D Z } \right)$ , and the mean absolute diference of the harmonic frequencies of the minimum between the two $\left( \mathrm { c m } ^ { - 1 } \right)$
<table><tr><td colspan="5">real functional</td><td colspan="4">MLP</td></tr><tr><td>method</td><td>a</td><td>barrier</td><td>+ZPE</td><td> $\left| \omega _ { \mathrm { T S } } \right|$ </td><td>barrier</td><td>+ZPE</td><td> $\left| \omega _ { \mathrm { T S } } \right|$ </td><td>frequencies</td></tr><tr><td>BLYP</td><td>0.000</td><td>1.20</td><td>-0.87</td><td>902</td><td>1.23</td><td>-0.85</td><td>889</td><td>3.2</td></tr><tr><td>B1LYP</td><td>0.250</td><td>2.65</td><td>0.21</td><td>1159</td><td>2.67</td><td>0.21</td><td>1146</td><td>2.0</td></tr><tr><td>BLYP35</td><td>0.350</td><td>3.28</td><td>0.74</td><td>1245</td><td>3.29</td><td>0.73</td><td>1232</td><td>2.1</td></tr><tr><td>BHandHLYP</td><td>0.500</td><td>4.23</td><td>1.57</td><td>1356</td><td>4.22</td><td>1.53</td><td>1346</td><td>1.8</td></tr><tr><td>HFLYP</td><td>1.000</td><td>6.96</td><td>4.00</td><td>1638</td><td>6.94</td><td>4.00</td><td>1643</td><td>2.6</td></tr><tr><td> $a _ { \mathrm { o p t } }$ </td><td>0.455</td><td>3.95</td><td>1.32</td><td>1326</td><td>3.94</td><td>1.29</td><td>1314</td><td>2.0</td></tr><tr><td>B3LYP</td><td></td><td>2.15</td><td>-0.21</td><td>1088</td><td></td><td></td><td></td><td></td></tr></table>

## Tunneling splittings of malonaldehyde

Calculation. The splittings are calculated with the ring-polymer instanton protocol of p. S9 on the MLP at a given a. The minima are optimized with the tightest criteria of geomeTRIC. For the deuterated molecule, the transferring hydrogen atom is replaced by deuterium. The Hessian of every bead is calculated by finite diferences of the forces of the MLP, which reproduce the analytic harmonic frequencies of the MLP at the minimum to $0 . 2 ~ \mathrm { c m } ^ { - 1 }$ . With 1024 beads, the splitting is within about 1% of its limit for infinitely many beads: on the $\mathrm { M L P }$ at $a _ { \mathrm { o p t } }$ , 512, 1024 and 2048 beads give 29.16, 29.81 and $2 9 . 9 8 ~ \mathrm { c m } ^ { - 1 }$ for H and 4.350, 4.395 and $4 . 4 0 6 ~ \mathrm { c m } ^ { - 1 }$ for D; the steps shrink as the inverse square of the number of beads, which extrapolates to 0.8% (H) and 0.4% (D) above the 1024-bead values. The protocol checks are in Table S23.

Results. With 1024 beads, ring-polymer instanton theory on the MLP gives tunneling splittings of $2 9 . 8 \mathrm { c m } ^ { \cdot }$ <sup>−1</sup> for H and $4 . 3 9 \mathrm { c m } ^ { - 1 }$ for D at $a _ { \mathrm { o p t } }$ , 38% and 51% above the experimental 21.6 and 2.915 cm<sup>−1</sup>.<sup>S38,S39</sup> The five ensemble members give 28.3–31.8 and $4 . 1 4 { - } 4 . 7 7 ~ \mathrm { c m } ^ { - 1 }$ (Figure 4f; Table S24). The 512-bead values are 29.2 and $4 . 3 5 ~ \mathrm { c m } ^ { - 1 }$ , 2.2% and 1.0% below the 1024-bead results, and the 2048-bead values 0.6% and 0.3% above them. Across the displayed coordinate scan, the H splitting falls from $6 1 . 1 ~ \mathrm { c m } ^ { - 1 }$ at $a = 0 . 3 5$ to $1 2 . 0 ~ \mathrm { { c m } ^ { - 1 } }$ at

Table S23: Convergence of the tunneling splitting $\left( \mathrm { c m } ^ { - 1 } \right)$ with the number of beads of the half ring polymer, on a test MLP at $a = 0 . 4 6 1 ~ \mathrm { ( c c - p V D Z }$ , one model), with the zero mode removed by projecting out the tangent of the path, and the action $S / \hbar$ . All three runs are on one machine with the settings of the splittings of the main text: the minima optimized with the tightest criteria of geomeTRIC and polished by two Newton steps, i-PI’s convergence thresholds tightened tenfold, and the masses of the most abundant isotopes. The splitting is 28.41, 29.33 and $2 9 . 5 7 ~ \mathrm { c m } ^ { - 1 }$ with 512, 1024 and 2048 beads; the steps shrink as the inverse square of the number of beads, which extrapolates to $2 9 . 6 6 ~ \mathrm { c m } ^ { - 1 }$ , 1.1% above its 1024-bead value. In pairs of runs on one machine, the splitting changes by $+ 3 . 2 \%$ from 25 to 12.5 K at 2048 beads, and at 512 beads by −0.6% when i-PI’s thresholds are tightened tenfold and by −0.7% with the isotope masses instead of i-PI’s standard atomic weights. At the minimum, the finite diferences of the forces that give every Hessian of the calculation reproduce the analytic harmonic frequencies of the MLP to $0 . 1 6 ~ \mathrm { c m ^ { - 1 } }$ , which changes the zero-point energy by $- 0 . 1 1 \ \mathrm { c m } ^ { - 1 }$ . The determinant with the zero mode projected out, from the block factorization used for every splitting, agrees with a dense calculation on the full Hessian of 3 runs to $3 \times 1 0 ^ { - 1 0 }$ in its logarithm.

<table><tr><td>beads</td><td>splitting</td><td> $S / \hbar$ </td></tr><tr><td>512</td><td>28.41</td><td>5.6747</td></tr><tr><td>1024</td><td>29.33</td><td>5.6788</td></tr><tr><td>2048</td><td>29.57</td><td>5.6793</td></tr></table>

$$
a = 0 . 6 0 .
$$

The scan is restricted to fractions where the zero-point level lies well below the barrier, as required for this deep-tunneling instanton treatment. At B1LYP $( a = 0 . 2 5 )$ , the zeropoint energy nearly cancels the barrier, reducing 2.7 to 0.2 kcal/mol; this is outside the regime where that treatment can reliably predict a splitting. At $a _ { \mathrm { o p t } }$ , the zero-point level lies 1.3 kcal/mol below the barrier.

Cost. Each splitting uses about 2.5–2.6 million MLP energy-and-force evaluations. At the measured cc-pVDZ DFT timings, the corresponding DFT workload is about 25,000–32,000 core-hours, versus about 13 core-hours for the five-model MLP ensemble. All but about 55,000 of the MLP evaluations are in the three optimizations of the instanton, and 55,000 are for the finite-diference Hessian of the final instanton; a DFT energy and gradient with cc-pVDZ takes 34 (B1LYP alone, Intel Xeon Gold 6240 CPU) to 44 core-seconds (the five functionals in an earlier active-learning run used for this timing, on the same cluster), and a splitting on the ensemble of five MLPs, evaluated in double precision with the 64 beads of each i-PI batch in one call, takes 3.3 h on four cores of an Intel Xeon Gold 6240 CPU, timed on a compute node with no other jobs (no GPU is used). The MLP cost is the wall time multiplied by the four requested CPU threads (13.4 core-hours); the exclusive-node reservation accounts for 60.2 allocated core-hours, while Slurm records 10.7 hours of CPU use. These are workload estimates for this calculation, not a measured end-to-end DFT instanton run.

Table S24: Tunneling splittings $\left( \mathrm { c m } ^ { - 1 } \right)$ from ring-polymer instanton calculations on the final MLP (cc-pVDZ, 1024 beads, 25 K): the ensemble of five across the space and at $a _ { \mathrm { o p t } }$ , and its five members alone at $a _ { \mathrm { o p t } } ;$ the barrier of the MLP (kcal/mol) and the action $S / \hbar$ of the instanton. With 2048 beads the splitting at $a _ { \mathrm { o p t } }$ is 29.98 $\mathrm { c m } ^ { - 1 }$ (H, +0.6% from 1024 beads) and 4.41 cm $^ { - 1 } \ ( \mathrm { D } , + 0 . 3 \%$ from 1024 beads). <sup>†</sup>: below $a = 0 . 3 5$ the zero-point energy cancels the barrier of the MLP (at $a = 0 . 2 5$ it takes it from 2.7 to 0.2 kcal/mol), the proton is not localized and instanton theory, a theory of deep tunneling, does not apply; these rows are listed for completeness and are not shown in the main text.
<table><tr><td>MLP</td><td>isotope</td><td>a</td><td>barrier</td><td> $\overline { { S / \hbar } }$ </td><td>splitting</td></tr><tr><td>ensemble</td><td>H</td><td>0.200†</td><td>2.36</td><td>3.875</td><td>175.96</td></tr><tr><td>ensemble</td><td>H</td><td>0.250†</td><td>2.67</td><td>4.261</td><td>123.59</td></tr><tr><td>ensemble</td><td>H</td><td>0.300†</td><td>2.98</td><td>4.643</td><td>86.71</td></tr><tr><td>ensemble</td><td>H</td><td>0.350</td><td>3.29</td><td>5.016</td><td>61.06</td></tr><tr><td>ensemble</td><td>H</td><td>0.400</td><td>3.60</td><td>5.379</td><td>43.21</td></tr><tr><td>ensemble</td><td>H</td><td>0.450</td><td>3.91</td><td>5.728</td><td>30.89</td></tr><tr><td>ensemble</td><td>H</td><td>0.455</td><td>3.94</td><td>5.765</td><td>29.81</td></tr><tr><td>ensemble</td><td>H</td><td>0.500</td><td>4.22</td><td>6.064</td><td>22.27</td></tr><tr><td>ensemble</td><td>H</td><td>0.550</td><td>4.52</td><td>6.385</td><td>16.26</td></tr><tr><td>ensemble</td><td>H</td><td>0.600</td><td>4.82</td><td>6.694</td><td>11.98</td></tr><tr><td>members,  $a _ { \mathrm { o p t } }$ </td><td>H</td><td>0.455</td><td>3.87-4.03</td><td></td><td>28.28-31.81</td></tr><tr><td>ensemble</td><td>D</td><td>0.200†</td><td>2.36</td><td>4.925</td><td>41.75</td></tr><tr><td>ensemble</td><td>D</td><td>0.250†</td><td>2.67</td><td>5.412</td><td>26.54</td></tr><tr><td>ensemble</td><td>D</td><td>0.300†</td><td>2.98</td><td>5.893</td><td>16.91</td></tr><tr><td>ensemble</td><td>D</td><td>0.350</td><td>3.29</td><td>6.362</td><td>10.84</td></tr><tr><td>ensemble</td><td>D</td><td>0.400</td><td>3.60</td><td>6.819</td><td>7.01</td></tr><tr><td>ensemble</td><td>D</td><td>0.450</td><td>3.91</td><td>7.258</td><td>4.60</td></tr><tr><td>ensemble</td><td>D</td><td>0.455</td><td>3.94</td><td>7.305</td><td>4.39</td></tr><tr><td>ensemble</td><td>D</td><td>0.500</td><td>4.22</td><td>7.682</td><td>3.05</td></tr><tr><td>ensemble</td><td>D</td><td>0.550</td><td>4.52</td><td>8.086</td><td>2.06</td></tr><tr><td>ensemble</td><td>D</td><td>0.600</td><td>4.82</td><td>8.475</td><td>1.41</td></tr><tr><td>members,  $a _ { \mathrm { o p t } }$ </td><td>D</td><td>0.455</td><td>3.87-4.03</td><td>一</td><td>4.14-4.77</td></tr></table>

## References

(S1) Dral, P. O.; Ge, F.; Hou, Y.-F.; Zheng, P.; Chen, Y.; Barbatti, M.; Isayev, O.; Wang, C.; Xue, B.-X.; Pinheiro Jr, M.; Su, Y.; Dai, Y.; Chen, Y.; Zhang, S.; Zhang, L.; Ullah, A.; Zhang, Q.; Ou, Y. MLatom 3: A Platform for Machine Learning-Enhanced Computational Chemistry Simulations and Workflows. J. Chem. Theory Comput. 2024, 20, 1193–1213.

(S2) Gao, X.; Ramezanghorbani, F.; Isayev, O.; Smith, J. S.; Roitberg, A. E. TorchANI: A Free and Open Source PyTorch-Based Deep Learning Implementation of the ANI Neural Network Potentials. J. Chem. Inf. Model. 2020, 60, 3408–3415.

(S3) Sun, Q.; Zhang, X.; Banerjee, S.; Bao, P.; Barbry, M.; Blunt, N. S.; Bogdanov, N. A.; Booth, G. H.; Chen, J.; Cui, Z.-H.; Eriksen, J. J.; Gao, Y.; Guo, S.; Hermann, J.; Hermes, M. R.; Koh, K.; Koval, P.; Lehtola, S.; Li, Z.; Liu, J.; Mardirossian, N.; Mc-Clain, J. D.; Motta, M.; Mussard, B.; Pham, H. Q.; Pulkin, A.; Purwanto, W.; Robinson, P. J.; Ronca, E.; Sayfutyarova, E. R.; Scheurer, M.; Schurkus, H. F.; Smith, J. E. T.; Sun, C.; Sun, S.-N.; Upadhyay, S.; Wagner, L. K.; Wang, X.; White, A.; Whitfield, J. D.; Williamson, M. J.; Wouters, S.; Yang, J.; Yu, J. M.; Zhu, T.; Berkelbach, T. C.; Sharma, S.; Sokolov, A. Y.; Chan, G. K.-L. Recent Developments in the PySCF Program Package. J. Chem. Phys. 2020, 153, 024109.

(S4) Wang, L.-P.; Song, C. Geometry Optimization Made Simple with Translation and Rotation Coordinates. J. Chem. Phys. 2016, 144, 214108.

(S5) Litman, Y.; Kapil, V.; Feldman, Y. M. Y.; Tisi, D.; Beguˇsi´c, T.; Fidanyan, K.; Fraux, G.; Higer, J.; Kellner, M.; Li, T. E.; P´os, E. S.; Stocco, E.; Trenins, G.;

Hirshberg, B.; Rossi, M.; Ceriotti, M. i-PI 3.0: A Flexible and Eficient Framework for Advanced Atomistic Simulations. J. Chem. Phys. 2024, 161, 062504.

(S6) Dunning, T. H. Gaussian Basis Sets for Use in Correlated Molecular Calculations. I. The Atoms Boron Through Neon and Hydrogen. J. Chem. Phys. 1989, 90, 1007–1023.

(S7) Kendall, R. A.; Dunning, T. H., Jr.; Harrison, R. J. Electron Afinities of the First-Row Atoms Revisited. Systematic Basis Sets and Wave Functions. J. Chem. Phys. 1992, 96, 6796–6806.

(S8) Cs´asz´ar, A. G.; Allen, W. D.; Schaefer, H. F. In Pursuit of the ab initio Limit for Conformational Energy Prototypes. J. Chem. Phys. 1998, 108, 9751–9764.

(S9) Smith, J. S.; Isayev, O.; Roitberg, A. E. ANI-1: An Extensible Neural Network Potential with DFT Accuracy at Force Field Computational Cost. Chem. Sci. 2017, 8, 3192–3203.

(S10) Chen, Y.; Dral, P. O. One to Rule Them All: A Universal Interatomic Potential Learning Across Quantum Chemical Levels. J. Chem. Theory Comput. 2025, 21, 8762–8772.

(S11) Martyka, M.; Tong, X.-Y.; Jankowska, J.; Dral, P. O. OMNI-P2x Universal Neural Network Potential for Excited-State Simulations. Nat. Commun. 2026, 17, 4949.

(S12) Hendrycks, D.; Gimpel, K. Gaussian Error Linear Units (GELUs). arXiv:1606.08415, 2016.

(S13) Alavi, S. F.; Chen, Y.; Hou, Y.-F.; Ge, F.; Zheng, P.; Dral, P. O. ANI-1ccx-gelu Universal Interatomic Potential and Its Fine-Tuning: Toward Accurate and Eficient Anharmonic Vibrational Frequencies. J. Phys. Chem. Lett. 2025, 16, 483–493.

(S14) Hou, Y.-F.; Zhang, L.; Zhang, Q.; Ge, F.; Dral, P. O. Physics-Informed Active Learning for Accelerating Quantum Chemical Simulations. J. Chem. Theory Comput. 2024, 20, 7744–7754.

(S15) Zheng, P.; Zubatyuk, R.; Wu, W.; Isayev, O.; Dral, P. O. Artificial Intelligence-Enhanced Quantum Chemical Method with Broad Applicability. Nat. Commun. 2021, 12, 7022.

(S16) Chen, Y.; Dral, P. O. AIQM2: Organic Reaction Simulations Beyond DFT. Chem. Sci. 2025, 16, 15901–15912.

(S17) Richardson, J. O.; Althorpe, S. C. Ring-Polymer Instanton Method for Calculating Tunneling Splittings. J. Chem. Phys. 2011, 134, 054109.

(S18) K¨aser, S.; Richardson, J. O.; Meuwly, M. Transfer Learning for Afordable and High-Quality Tunneling Splittings from Instanton Calculations. J. Chem. Theory Comput. 2022, 18, 6840–6850.

(S19) Mielke, S. L.; Garrett, B. C.; Peterson, K. A. A Hierarchical Family of Global Analytic Born–Oppenheimer Potential Energy Surfaces for the H + H2 Reaction Ranging in Quality from Double-Zeta to the Complete Basis Set Limit. J. Chem. Phys. 2002, 116, 4142–4161.

(S20) Mielke, S. L.; Peterson, K. A.; Schwenke, D. W.; Garrett, B. C.; Truhlar, D. G.; Michael, J. V.; Su, M.-C.; Sutherland, J. W. H + H2 Thermal Reaction: A Convergence of Theory and Experiment. Phys. Rev. Lett. 2003, 91, 063201.

(S21) Eyring, H. The Activated Complex in Chemical Reactions. J. Chem. Phys. 1935, 3, 107–115.

(S22) Eckart, C. The Penetration of a Potential Barrier by Electrons. Phys. Rev. 1930, 35, 1303–1309.

(S23) Fleming, D. G.; Arseneau, D. J.; Sukhorukov, O.; Brewer, J. H.; Mielke, S. L.; Truhlar, D. G.; Schatz, G. C.; Garrett, B. C.; Peterson, K. A. Kinetics of the Reaction of the Heaviest Hydrogen Atom with $\operatorname { H } _ { 2 } .$ the $^ 4 \mathrm { H e } \mu + \mathrm { H } _ { 2 } \to { ^ 4 \mathrm { H e } \mu \mathrm { H } } + \mathrm { H }$ Reaction: Experiments, Accurate Quantal Calculations, and Variational Transition State Theory, Including Kinetic Isotope Efects for a Factor of 36.1 in Isotopic Mass. J. Chem. Phys. 2011, 135, 184310.

(S24) Mielke, S. L.; Schwenke, D. W.; Peterson, K. A. Benchmark Calculations of the Complete Configuration-Interaction Limit of Born-Oppenheimer Diagonal Corrections to the Saddle Points of Isotopomers of the $\mathrm { H + H _ { 2 } }$ Reaction. J. Chem. Phys. 2005, 122, 224313.

(S25) Halkier, A.; Helgaker, T.; Jorgensen, P.; Klopper, W.; Koch, H.; Olsen, J.; Wilson, A. K. Basis-Set Convergence in Correlated Calculations on Ne, N2, and H2O. Chem. Phys. Lett. 1998, 286, 243–252.

(S26) Halkier, A.; Helgaker, T.; Jorgensen, P.; Klopper, W.; Olsen, J. Basis-Set Convergence of the Energy in Molecular Hartree-Fock Calculations. Chem. Phys. Lett. 1999, 302, 437–446.

(S27) Sato, S. On a New Method of Drawing the Potential Energy Surface. J. Chem. Phys. 1955, 23, 592–593.

(S28) Anderson, J. B. Note: “Exact” Quantum Monte Carlo Calculations of the Barrier for the $\mathrm { ~ H ~ } + \mathrm { ~ H _ { 2 } ~ }$ Reaction at the Sub-Microhartree Level. J. Chem. Phys. 2016, 144, 166101.

(S29) Urban, S.; D’Cunha, R.; Rao, K. N.; Papouˇsek, D. The<sup>ˇ</sup> $\Delta k = \pm 2$ “Forbidden Band” and Inversion–Rotation Energy Levels of Ammonia. Can. J. Phys. 1984, 62, 1775– 1791.

(S30) Fusina, L.; Di Lonardo, G.; Johns, J. W. C. Inversion–Rotation Spectrum and Spectroscopic Parameters of $^ { 1 4 } \mathrm { N D _ { 3 } }$ in the Ground State. J. Mol. Spectrosc. 1985, 112, 211–221.

(S31) Urban, S.; D’Cunha, R.; Rao, K. N.; Papouˇsek, D. The<sup>ˇ</sup> $\Delta k = \pm 2$ Perturbation-Allowed $\nu _ { 4 }$ Band and Inversion–Rotation Energy Levels of $^ { 1 5 } \mathrm { N } \mathrm { H _ { 3 } } .$ . J. Mol. Spectrosc. 1985, 111, 361–376.

(S32) Diedrich, D. L.; Anderson, J. B. An Accurate Quantum Monte Carlo Calculation of the Barrier Height for the Reaction $\mathrm { H + H _ { 2 }  H _ { 2 } + H } .$ Science 1992, 258, 786–788.

(S33) Riley, K. E.; Anderson, J. B. Higher Accuracy Quantum Monte Carlo Calculations of the Barrier for the H + H2 Reaction. J. Chem. Phys. 2003, 118, 3437–3438.

(S34) Lynch, B. J.; Fast, P. L.; Harris, M.; Truhlar, D. G. Adiabatic Connection for Kinetics. J. Phys. Chem. A 2000, 104, 4811–4815.

(S35) Zhao, Y.; Lynch, B. J.; Truhlar, D. G. Development and Assessment of a New Hybrid Density Functional Model for Thermochemical Kinetics. J. Phys. Chem. A 2004, 108, 2715–2719.

(S36) Renz, M.; Theilacker, K.; Lambert, C.; Kaupp, M. A Reliable Quantum-Chemical Protocol for the Characterization of Organic Mixed-Valence Compounds. J. Am. Chem. Soc. 2009, 131, 16292–16302.

(S37) Stephens, P. J.; Devlin, F. J.; Chabalowski, C. F.; Frisch, M. J. Ab Initio Calculation of Vibrational Absorption and Circular Dichroism Spectra Using Density Functional Force Fields. J. Phys. Chem. 1994, 98, 11623–11627.

(S38) Baba, T.; Tanaka, T.; Morino, I.; Yamada, K. M. T.; Tanaka, K. Detection of the Tunneling-Rotation Transitions of Malonaldehyde in the Submillimeter-Wave Region. J. Chem. Phys. 1999, 110, 4131–4133.

(S39) Baughcum, S. L.; Smith, Z.; Wilson, E. B.; Duerst, R. W. Microwave Spectroscopic Study of Malonaldehyde. 3. Vibration-Rotation Interaction and One-Dimensional Model for Proton Tunneling. J. Am. Chem. Soc. 1984, 106, 2260–2265.