# LLM-Assisted Generation of Transparent, Open-Source Multiphysics Models of Electrochemical Devices

Sebastian Castro<sup>1</sup>, Maya F. Schuchert<sup>1</sup>, Spencer A. McCluskey<sup>2</sup>, Eric W. Lees<sup>2,\*</sup>, Justin C. Bui<sup>1,\*</sup>

<sup>1</sup>Department of Chemical and Biomolecular Engineering, New York University, Brooklyn, NY, USA <sup>2</sup>Department of Chemical and Biological Engineering, University of British Columbia, Vancouver, BC, Canada

Corresponding authors. e-mail: justinbui@nyu.edu; eric.lees@ubc.ca

## Abstract

Multiphysics continuum models are powerful tools for studying electrochemical devices, enabling in silico reactor design and resolution of local pH, potential, and concentration fields that govern device performance but are dificult to measure experimentally. However, constructing such models requires substantial numerical expertise or reliance on proprietary software. Here, we show that frontier large language model agents can remove this implementation burden while keeping the underlying physics under researcher control. Using one-dimensional electrochemical $\mathrm { C O } _ { 2 }$ reduction to CO in a porous gas difusion electrode as a test case, we develop a machine-readable, human-specified modeling harness containing governing equations, parameters, numerical methods, logical build stages, and human-verifiable checkpoints. From this specification, the agent reproducibly constructs complete multiphysics models in open-source Julia. Independently built models, including fully autonomous agent-built models, agree with an equivalent COMSOL implementation to within 0.7% of the peak CO partial current density, and with one another to within 0.04%. Systematically planted errors demonstrate the importance of explicit specifications for reproducibility and reveal the agent’s capabilities and limitations in debugging model physics. This framework establishes a more transparent approach to multiphysics modeling in which physical descriptions and governing equations, rather than specialized code, become the primary inputs for computational model development.

Electrochemical reactors sit at the center of the energy transition: electrolyzers use renewable electricity to drive the synthesis of fuels and chemicals, while fuel cells and batteries store that renewable electricity in chemical form<sup>1,2</sup>. Each of these devices is a coupled, multi-component system for which performance is an emergent property of the interactions between material components (electrodes, membranes, and electrolytes) and operating conditions (temperature, pressure, flow rate). The spatially heterogeneous pH, potential, and reactant concentration profiles within these reactors directly govern selectivity, throughput, and durability, but these local chemical environments are largely inaccessible to direct measurement under realistic operating conditions<sup>3,4</sup>. Consequently, rational reactor design and optimization are dificult to achieve through experimentation alone.

Continuum modeling can accelerate the optimization of electrochemical devices by translating their coupled physics (i.e., multiphysics) into a system of governing equations that can be solved numerically<sup>5,6</sup>. Conservation equations for mass, charge, and momentum, expressed with models of ion transport, electrochemical kinetics, and fluid flow in porous media, capture the relationship between local chemical environments and experimental observables. For instance, multiphysics models can resolve how competing reactions distribute current across an electrode<sup>7</sup>, how membrane chemistry impacts voltage losses<sup>8</sup>, and how spectator ions tune reaction selectivity<sup>9</sup>. Once parameterized and validated against experimental data, they also enable large design spaces to be explored at a fraction of the cost of experiments.

The history of electrochemical continuum modeling is a history of shifting barriers. In the 1960s, building a model required simultaneous mastery of the physics of electrochemistry and the numerical methods needed to solve the resulting stif, coupled partial diferential equations (PDEs), often using Fortran<sup>10,11</sup>, which confined serious modeling to a small community. Commercial simulation platforms, principally COMSOL Multiphysics, placed those algorithms behind a graphical interface and a proprietary solver<sup>12</sup>, allowing practitioners to focus more efort on defining the physical equations and less on implementation. Physical intuition remained essential to choosing and evaluating the governing equations, constitutive relationships, and boundary conditions; the numerical machinery, however, receded into the background, and with it direct engagement with how models are solved. When a solver fails, diagnosing whether the failure originates from the governing equations, discretization, or numerical implementation can therefore be dificult. Generic messages such as “maximum iterations reached” rarely indicate which equation, term, or mesh element is responsible, leaving debugging to heuristic experience with mesh refinement, relaxation factors, and initial guesses. Commercial solvers are also less accessible and more expensive than open-source alternatives<sup>13,14</sup>, and proprietary formats complicate reproducibility, as model files are often tied to specific software versions and inaccessible to collaborators without licenses<sup>15</sup>.

We contend that the emergence of large language model (LLM) agents is the next chapter in electrochemical continuum modeling. A researcher with deep domain knowledge but limited numerical training can instruct an artificial intelligence (AI) agent to implement a fully open-source continuum model in which every equation, solver step, and line of code is visible, modifiable, and explainable. The agent does not replace the physics expertise of the researcher but rather replaces the implementation burden. The researcher must still specify the governing equations precisely, anticipate which physical mechanisms are relevant, design staged verification checks that catch errors before they propagate, and recognize when a converged solution is unphysical. Domain knowledge remains critical; what changes is that the auxiliary technical burden shifts from Fortran subroutines and COMSOL debugging toward specification and verification. Because the resulting model is open-source and every step of its numerics is inspectable, the practical barrier to continuum multiphysics modeling becomes the ability to state the physics precisely rather than access to a solver license or extensive training in numerical methods.

Agentic AI has already shown potential for integration into the scientific process, including autonomy in designing experiments, executing code, and discovering new pathways for chemical synthesis<sup>16,17</sup>. LLM agents have been shown to autonomously write, execute, and debug finite-element code for continuum mechanics problems<sup>18</sup>, collaborate as multi-agent teams across a range of physics domains<sup>19,20</sup>, and generate PDE solvers that match expert-written code on benchmarks including compressible Navier–Stokes<sup>21</sup>. Nonetheless, this work has largely targeted weakly coupled or single-physics problems, leaving strongly coupled multiphysics systems comparatively unexplored. A recent multi-agent framework, for instance, runs finite-element simulations from natural language by retrieving domain knowledge from a curated knowledge graph and verifying the solver against closed-form analytical solutions<sup>22</sup>. Neither approach transfers directly here: a nonlinear, strongly coupled system of this kind generally possesses no analytical reference solution, while its governing physics must be specified through researcher-authored modeling choices specific to the simulated device rather than retrieved as a fixed set of facts. Electrochemical reactors are a particularly demanding case, with acid–base equilibria, interfacial kinetics, migration, and multicomponent gas transport entering one another's source terms and transport coeficients. The equations must be solved simultaneously as a single nonlinear system, with bufer equilibria that relax orders of magnitude faster than species difuse, making that system stif and dificult to solve robustly.

To demonstrate the validity of our approach, we developed a detailed, machine-readable implementation guide for a one-dimensional (1D) continuum model (Fig. 1) of silver-catalyzed $\mathrm { C O } _ { 2 }$ reduction $( \mathrm { C O } _ { 2 } \mathrm { R } )$ in a gas difusion electrode (GDE). This implementation guide served as a harness that directed the AI agent through construction of the multiphysics model in open-source Julia<sup>23</sup>. Researchers of varying experience levels used the guide to instruct AI agents to build the model, without writing a line of code themselves. Because the complex numerical system has no analytical solution, validation of the approach was demonstrated against an equivalent model developed in COMSOL. Reproducibility was assessed by comparing independently generated model implementations across users. Beyond accuracy and reproducibility, the agentic workflow provides a transparent interface to the numerical implementation: the agent that built the code can inspect the Jacobian, trace convergence failures, and explain the numerical behavior in the language of physics directly to a human operator. Collectively, this framework establishes an approach to multiphysics modeling in which human-specified equation sets and natural-language instructions, rather than specialized code and numerical implementations, become the primary inputs for model development. In this paradigm, the specification harness—rather than any individual implementation—becomes the central scientific artifact that other practitioners can audit, reproduce, modify, and build upon.

![](images/16c3b55095673f32c4fff233b9fa15ecd5558a3fbf63308614d0789054c4c37f.jpg)  
Fig. 1. (a) Schematic of the agentic multiphysics modeling workflow employed here with a human-in-the-loop. (b) Schematic of the $\mathrm { C O } _ { 2 }$ reduction porous electrode model case study, along with the 5-stage implementation approach in Julia, with the primary physics additions/changes included in each stage. Elyte, electrolyte; CL, catalyst layer; GDL, gas difusion layer; DOFs, degrees of freedom; BC, boundary condition; HER, hydrogen evolution reaction.

## Results

## A specification-driven, staged build

The modeling of electrochemical reactors is a prime example of a complex, multiphysics problem, and thus presents a suitable platform for establishing agentic multiphysics approaches. Every one of the nonlinear couplings introduced above, and the corresponding numerical strategy that makes the resulting stif system tractable, must be stated explicitly before an agent can implement it. Hence, an implementation guide (i.e., machine-readable specifications, here markdown files) can serve as a harness for an agent building multiphysics models, fully describing the problem to be modeled: all relevant parameters, equations, and boundary conditions, together with a scafolded build plan that the coding agent must follow. These guides link the scientific knowledge and intuition from the human to the numerical implementation of the agent. Precise and comprehensive specifications ensure consistency and accuracy between agents with probabilistic behavior, and the responsibility remains with the researcher to accurately define the problem. In other words, the agent's primary role in this workflow is the implementation of a specified guide, rather than the inference of a complete model from an incomplete specification.

To provide the agent (and its human operator) with logical checkpoints at which to assess and correct mistakes in implementation, the guide decomposes model development into five stages of increasing physical complexity (Fig. 1b and Supplementary Fig. 1), from liquid Nernst–Planck transport, bufer reactions, and electrochemical kinetics at 400 degrees of freedom (DOFs) to the full electrostatically coupled system at 1,480 DOFs on the production mesh. Between stages, stop gates allow the human-in-the-loop to review quantitative physicality checks and diagnostic plots prescribed by the guide itself, rather than leaving them to the agent's discretion, before authorizing the agent to advance to the next stage. These gates are solution-verification checks in the established sense<sup>24</sup>, applied by the human rather than the agent. Full stage definitions, pass/fail criteria, and details of the fitting step used to parameterize the model against the experimental data of Romiluyi et al.<sup>25</sup> are given in the Methods section.

Our assessment of the AI-assisted build begins with the polarization and faradaic eficiency (FE) results, and how these outputs evolve as each stage of the implementation guide adds a layer of physics (Fig. 2a–c). Notably, the plots reproduce the canonical behavior of a $\mathrm { C O } _ { 2 }$ electrolyzer: $F E _ { \mathrm { C O } }$ peaks at intermediate potentials and declines rapidly at high potentials where the hydrogen evolution reaction (HER) outcompetes. Physically, this occurs because of the interplay between $\mathrm { C O } _ { 2 }$ reduction kinetics, $\mathrm { C O } _ { 2 }$ mass transport, and the consumption of $\mathrm { C O } _ { 2 }$ by acid–base equilibria within the porous electrode<sup>26</sup>. The evolution of the curves coincides with the physics added at each stage: Stage 1 includes bufer equilibria, electrochemical kinetics, and liquid-phase species transport by difusion alone, with fixed-concentration (Dirichlet) boundary conditions; the Stage 2 mass-transfer boundary condition allows $\mathrm { C O } _ { 2 }$ depletion and $\mathrm { p H }$ buildup and correspondingly reduces CO partial current, $i _ { \mathrm { C O } } ;$ Stage 3 phase transfer from the gas adds a liquid-phase $\mathrm { C O } _ { 2 }$ source and drives $i _ { \mathrm { C O } }$ back up; Stage 4 momentum balance in the gas phase adds a small but physical correction; the ${ \mathrm { f i t } } ,$ performed on the Stage 4 model, brings $i _ { \mathrm { C O } }$ down to the experimental data; and Stage 5 migration raises $i _ { \mathrm { C O } }$ to its final value by increasing the flux of OH<sup>–</sup> and other anions away from the cathode, which mitigates the salting-out of $\mathrm { C O } _ { 2 }$

e)  
a)  
![](images/c9bedd31f6b7d4764f52165e8c458b92d54234d31d1a82fd1ad78158d0fc20ce.jpg)

b)  
![](images/e6b58f5f4730b83007ea33b467d37674319a721ed544c40b2c946a5a24da1820.jpg)

c)  
![](images/e52b98d21b50023751eb833cace24532da5e090bf1888c8eb7c02de8fa619b7e.jpg)

d)  
![](images/7f9ff5051ce867dfbb5e786af100cb165cafcb9b836aaab8fb66e0e2732d4169.jpg)

![](images/646eaad785f071b101f8cc45a9dc7bc2ba17567efb2ba901f1e822e8a9d78df8.jpg)

f)  
![](images/75963be1a6c5fb796f5a7e5b0342eadc5b13d66f18cc56b1cee7e3d0a470a394.jpg)

g)  
![](images/b0d1395cdb5752321072f36c97ce6a4e258cfc01a5e15147a82399187893d5d3.jpg)

i)  
![](images/77ad4e6c22dbd0e3649f1447c596d9138a9d791572b13c048a86b88e9a0e451b.jpg)

![](images/3df51ce74087805f868e0152e5e4c090e28c461bcc8ab6d18c5daf5bbf4688e0.jpg)  
Fig. 2. Stage-by-stage build progression and internal fields of final model. CO (a) linear-scale polarization, (b) log-scale polarization and (c) faradaic eficiency after each build stage and the fit. Converged Stage-5 profiles at five applied potentials: across the catalyst layer, (d) pH, (e) $\mathrm { C O } _ { 2 } , ( \mathbf { f } )$ $\mathrm { H C O } _ { 3 } { } ^ { - }$ and $\mathbf { \omega } ( \mathbf { g } ) \mathbf { C } \mathbf { O } _ { 3 } { } ^ { 2 - }$ concentrations; across the gas difusion layer and catalyst layer, $( \mathbf { h } ) { \mathrm { C O } } _ { 2 }$ and (i) CO partial pressures. Markers represent experimental data. Voltages vs the standard hydrogen electrode (SHE).

Beyond examining the polarization data, the correctness of the implemented physics was complementarily verified at each stage by analyzing the species profiles to confirm the simulated device concentration fields are consistent with physical intuition. The most relevant field profiles are shown in Fig. 2d–i, and the full output of all the plotted profiles from every stage is shown in Supplementary Figs. 3–13. As expected, the simulated pH (Fig. 2d) is highest at the gas difusion layer (GDL) side $( \mathbf { \boldsymbol { x } } = 0 )$ where the $\mathrm { C O } _ { 2 }$ consumption rate is highest, generating OH<sup>–</sup> anions via $\mathrm { C O } _ { 2 } \mathrm { R }$ . Additionally, the dissolved $\mathrm { C O } _ { 2 }$ (Fig. 2e) is lowest near the GDL where $\mathrm { C O } _ { 2 } \mathrm { R }$ rates are highest and increases toward the electrolyte, where $\mathrm { C O } _ { 2 }$ is slightly replenished by incoming (bi)carbonates and their equilibria. The (bi)carbonate field profiles (Fig. 2f–g) behave in accordance with their acid–base equilibria and $\mathrm { C O } _ { 2 }$ concentrations. In the gas phase (Fig. 2h–i), gaseous $\mathrm { C O } _ { 2 }$ is highest where it comes in from the flow channel, has a linear concentration gradient through the GDL where it undergoes difusive transport only, and is consumed in the catalyst layer (CL). CO is generated in the CL and difuses out into the gas channel. These profiles, which would be dificult to measure experimentally in the microscale porous electrode, show that the model is physically reasonable.

## Agreement with COMSOL and across builds

To assess the eficacy of the specification guide in facilitating agentic multiphysics modeling, we leveraged the build to develop an experimentally calibrated continuum multiphysics model of a gas difusion electrode used for $\mathrm { C O } _ { 2 } \mathrm { R }$ . We consider an AI-assisted build successful if it yields consistent and physically reasonable results across independent builds, reproduces the simulation results of an established commercial tool reference (in this case, COMSOL) to within a few percent, and remains fully transparent end-to-end $( i . e . $ , all code is generated and available for examination by human evaluation or the audits of other agents).

To confirm the accuracy of the agent’s multiphysics implementation, we benchmarked the three independent human-in-the-loop builds, made with Claude Opus 5, against a physically equivalent COMSOL model with the same parameters (Fig. 3). The Julia build outputs agree with the COMSOL model quantitatively, with the polarization and FE curves overlapping entirely. Evaluated at the 44 COMSOL sweep potentials that fall within the Julia sweep window, the average of the root mean square deviations (RMSDs) of each build against COMSOL is 2.27 mA cm−<sup>2</sup> for $i _ { \mathrm { C O } } , 1 . 0 7 \mathrm { m A } \mathrm { c m } ^ { - 2 }$ for the $\mathrm { H } _ { 2 }$ partial current density, $i _ { H 2 } ,$ , and 0.14 percentage points for $F E _ { C O }$ . The $i _ { \mathrm { C O } }$ RMSD represents 0.7% of the peak CO current density of about 330 mA cm−<sup>2</sup>, and the Julia i<sub>CO</sub> values sit systematically above the COMSOL values, by up to 2.2% at currents above $\bar { \mathsf { 5 } } \mathrm { m A } \mathrm { c m } ^ { - 2 }$ and about 0.9% at the peak, a discrepancy likely due to slight diferences in mesh generation or to numerical implementation diferences between the solvers.

e)  
a)  
![](images/3701f37b6843c23cc64f2d4ff9be74c5728f648b8cbf37c7fb4d73013267597c.jpg)  
b)

![](images/77fac91c18f30f0fd758009bebb41029d9c281917ea07e12c5c669618711f8aa.jpg)

c)  
![](images/a56fd7ad7d019b634aae2e96ef4fe3473ab8e670d201dd040f3cbdb173205aea.jpg)

d)  
![](images/0d314c743f59ee93c1f0f5187bd2f1c1f451e2e28dd0c27912a0f1e6e879a292.jpg)

![](images/d23f5eb8ff121597a9fd24ebd2918ee6aebac990cd0e456e24a51d1f3278da66.jpg)

f)  
![](images/4d70e8cebf52a68feb439c449eca632c44ee03373c3cb0f55b3bde887519e38d.jpg)  
Fig. 3. Agreement of three independently built Julia models with the COMSOL model. (a) CO and $\mathrm { H } _ { 2 }$ partial current densities, (b) faradaic eficiencies and (c) $\mathrm { C O } _ { 2 } ( \mathsf { a q } )$ concentration across the catalyst layer at five applied potentials, with the deviation of each from COMSOL below it: (d) currents, (e) ${ \mathrm { F E } } _ { \mathrm { C O } } ,$ in percentage points, and (f) $\mathrm { C O } _ { 2 } ( \mathsf { a q } )$ . Current and FE deviations are evaluated at the COMSOL sweep potentials. Markers represent experimental data.

Across the three builds evaluated on a common 89-point voltage sweep, the partial current densities also agreed with one another to within a pooled RMSD of 0.12 mA $\mathrm { c m } ^ { - 2 }$ for CO and 0.11 $\mathrm { m A } ~ \mathrm { c m } ^ { - 2 }$ for $\mathrm { H } _ { 2 } ,$ and the CO faradaic eficiency agreed within 0.01 percentage points. The fitting approach was also remarkably consistent across the three builds. The $\mathrm { C O } _ { 2 } \mathrm { R }$ Marcus–Hush–Chidsey<sup>27</sup> kinetic parameters were fit to literature data using a Nelder–Mead algorithm<sup>28,29</sup>, landing on an exchange current density (i<sub>0</sub>) of $1 0 6 . 4 7 \pm 0 . 1 1 \mathrm { ~ A ~ m } ^ { - 2 }$ and a reorganization energy (λ) of $1 . 6 1 1 1 \pm 0 . 0 0 0 4 \mathrm { e V } .$ The small residual spread reflects the flatness of the fit objective; all three replicates used identical termination criteria and reached the same sum of squared errors to four significant figures (0.05608). These results confirm that independent implementations of the specification harness produce nearly identical multiphysics models both to each other and to an established commercial software package, providing confidence in both the methodological approach and the accuracy of the physical implementation.

## Solver implementation and agent cost–benefit deviation

One benefit of commercial packages is that their plug-and-play numerical solvers have been long-optimized to ensure that solvers are properly matched to the systems that they solve. To benchmark the quality and speed of the agent’s numerical implementation of $\mathrm { C O } _ { 2 } \mathrm { R }$ against that of a commercial software package, we performed speed benchmarking against an equivalent COMSOL model with matched degrees of freedom. Notably, the agent’s Julia model demonstrates competitive solve speed to COMSOL with the added benefit of customization. Timed on the same machine at a matched compute budget and problem size (\~20,000 DOFs), the full 89-potential sweep takes about 13 s with our production solver stack compared to 21 s for COMSOL’s 61-potential sweep (Fig. 4a). It is important to emphasize, however, that a direct comparison is not possible because of inherent diferences in discretization (finite element method versus finite volume method), linear-solve library $( { \mathrm { P A R D I S O } } ^ { 3 0 }$ vs UMFPACK<sup>31</sup>), and solve environment (native Windows vs the Windows Subsystem for Linux). The solver ablation (Fig. 4b) reveals that speed gains in numerical solvers often come from simply tailoring the solver (e.g., its sparsity, the discretization method, etc.) to the problem solved. Agentic multiphysics implementations with frontier agents therefore benefit significantly from the cheapness of customizing solver parameters to the specific problem being solved. The gains come almost entirely from exploiting model sparsity, first in the linear solve and again in Jacobian assembly, each worth about two orders of magnitude against a dense alternative. We note that the components of the developed solver do not represent novel mathematics or solution techniques. However, with an open-source, modifiable solver, we no longer need to sacrifice performance for the solver generality that is often required of one-size-fits-all software packages. Full details of the solver stack are given in Supplementary Note 4.

Despite agents mostly conforming to the specification, we observed deliberate, documented deviations during intermediate solver development. Each solver component was explicitly mandated in a reproducibility checklist, yet none of the builds implemented the analytical block

a)  
![](images/d14d557a8d018c2652ecc306136a4cb8422f9f0af56de2f8f2fbc1615beaac8a.jpg)

b)  
![](images/8b202d4f6fb0ee8c68afa12040c95e6f755f983c2749f06376bc03d93a6bd655.jpg)  
<sup>c) d)</sup>Fig. 4. Solver performance of the Julia implementation. (a) Solve time for each implementation's 2<sup>]</sup>full sweep (89 potentials in Julia, 61 in COMSOL), both run on the same machine at matched core A<sup>c</sup>budget (32 cores/threads, ${ \approx } 2 0 , 0 0 0 \mathrm { D O F s } )$ . (b) Measured cost of Jacobian and linear-solver choices it <sup>[ 250</sup> 320 e)at 20,038 DOFs. Every configuration returns the same polarization curve to the precision written, l 200m <sub>336</sub>so bars difer only in cost. Ratios are relative to the configuration specified in the guide (UMFPACK C<sup>L</sup> 150m <sup>334</sup> a<sup>r</sup>with pattern reuse and forward-mode automatic diferentiation). LU, lower–upper factorization; <sup>v</sup>AD, automatic diferentiation.

Jacobian or installed MUMPS<sup>32</sup>, the open-source sparse direct solver that an earlier version of the guide specified, unless explicitly told to do so for speed benchmarking. The guide itself labeled these components performance multipliers, and the agents took the label at face value, opting for simpler alternatives and recording the reasoning in their build dialogue notes. The agent would weigh, for example, hand-deriving the analytical block Jacobian, which can yield silent calculation errors, against a forward-diferentiation $\mathtt { c a l l } ^ { 3 3 }$ that takes a few lines of code and is correct to machine precision. Automatic diferentiation has been adopted for the same reason in Newman-tradition porous electrode codes<sup>34</sup>, where hand-derived Jacobians are a known source of silent error. Similarly, the agents opted for the UMFPACK solver available in the Julia standard library, justifying the substitution as having similar performance while avoiding the extra efort required for installing MUMPS. Regardless, the diference in solve time for the solver items substituted by the agent was non-detrimental (Fig. 4b), while the implementation burden for the agent would have been intensive. Because both substitutions were at least as fast as the specified components, they were adopted in the final version of the solver stack. This behavior is nonetheless a cautionary tale for other users of this methodology: unless explicitly told and checked by a human operator, frontier agents often route numerics through simpler implementations that they deem equivalent and more eficient.

## Diagnostic transparency during model development

Building a correct model depends first on the front-end work of developing and specifying the relevant physics and numerical methods within the implementation guide. Once the guide is suficiently specified and could generate models, the process consisted of building, troubleshooting, refining, and iterating on the model harness. Intermediate builds using earlier versions of the implementation guide surfaced several critical errors, required substantial human input, and/or produced significant variation in build trajectories even from the same guide (Supplementary Note 7), reflecting the importance of the user in adequately specifying the model physics as to reduce variance in the LLM's probabilistic output<sup>35</sup>. As the guide was refined, these issues were progressively designed out and constrained, so that the final replicates required only minimal human intervention and had no critical failures. The challenges encountered throughout the development process for the implementation guide, along with the agents’ capacity to detect and address these failures, are shown in Fig. 5.

An example of a physics bug that arose in the early builds occurred when switching from the Dirichlet to the Neumann mass-transfer boundary condition for $\mathrm { C O } _ { 2 }$ in Stage 2, where the model suddenly could not converge. The agent reasoned that the mass-transfer boundary condition was unable to sustain high currents, and, upon investigating the physics, found a single sign error in the mass-transfer flux term as the root cause, since it incorrectly caused $\mathrm { C O } _ { 2 }$ to leave the system rather than being supplied from the electrolyte side. The same sign error recurred independently in five of six early builds, across two agent models and four users, because the guide gave the flux without properly specifying its sign convention. In all runs the solver failed to converge, prompting investigation. In some cases, the agent found the bug as discussed above, while in others it could not identify the root cause without human intervention (e.g., prompting the agent to check the sign on the boundary condition).

1: Backtracking line search ${ \sf k } _ { _ { \sf M T } } ^ { \mathrm { a } }$ scale ramp

Liquid potential drop too small

## Error

![](images/29de2eb0936f58614aa1af0567f17537c850ece398b431e015e0b47ed3f1239b.jpg)  
Case 1: Sign error in Stage 2 mass transfer boundary condition

![](images/2714aa69e4b556d5ef4773614d0238f951c889eace403d51f9cd48b7b3e8b12e.jpg)

![](images/eedea9cf3871aa2110774d75e2f534b956974f9dfed6ce7fad635682d8f23a1c.jpg)  
Case 2: Cs+ omited from ionic current sum

## Symptom

No convergence at any voltage

![](images/cd669995ba333fccf3665fc197b09416da19525c73c8eedae337777eeab68b82.jpg)

![](images/4a704c26b5e43abc23f90ff933f4c78d522e23662ab241583cd55a2e5e3d03bd.jpg)  
Case 3: Porous media difusivity correction missing

![](images/43fb493936db6758b3dc1898f79a151688231be8dcb29b6c86db64f26e47a06f.jpg)

![](images/3410ceecd879ca3410dbdd7aa9998451a9d914b4feec83587e4dc9acb89ea4b8.jpg)  
Smoother profiles, but plausible results Di 3.6 × too high

Case 4: Incorrect units passed into FSG correlation

![](images/7361c205d25fc14a4d9c1c0de733378d5ec2a533e7643abf07267fc3a368d624.jpg)

![](images/4011cc4baef9dca44f87b24c109c9e597d546fdb52fbe49e75391723a34c5d2b.jpg)  
Tolerance too high  
Pressure drop becomes negligible D 30 × too high gas

![](images/b857376aabc5ea69dad6b9a7b5c2df85830ab58f5d732b808ea7bab0cbbd56b4.jpg)  
Partial convergence; Kinks in polarization curve  
9: Pseudo-arclength continuation  
10: PAC V-sweep from V = +1 Identified mass transfer limitation from BC sign convention  
Compared ФL profiles vs reference  
Parameters identical; traced root cause by reading L assembly line by line

## Agent Handling

10 atempts to resolve error

![](images/9fa216a9b3124eb5beeb4320850c68908d82c2dd0b785c98353af48228042a7c.jpg)

Silent during build; found only after cross-comparing builds

Observed higher CO currents and low CO gas depletion at high overpotential. Found unit error and verified against literature difusivity

Investigated after user observation; found high residual norm

Planted error study: mature guide

Caught independently

Compared to Nernst-Planck sign convention defined elsewhere

Performed OH– physicality check

Caught with help

Nonconvergence exposed problem cond(J) \~10²³

Investigated Jacobian conditioning first, then caught physics bug after user nudge: “physics bug in the liquid block”

## Not caught

Low Biot number fell within expected magnitude, produced plausible results and no convergence errors

Caught independently

Performed order-of-magnitude sanity check against known gas difusivities

Not applicable

Fig. 5. How does the multiphysics agent handle errors? Errors encountered naturally during model development and planted systematically into a mature guide, their respective symptoms that the human can see and report to the agent, and the agent’s response to resolve the issue. The planted error study was conducted with Claude Opus 4.8. Agents were instructed not to consult outside resources, including other model builds, unless stated otherwise (see Supplementary Notes 6 and 8). Elyte, electrolyte; ${ \mathrm { C L } } ,$ catalyst layer; ${ \mathrm { B C } } ,$ boundary condition; $\alpha _ { \mathrm { { B C } } } ,$ boundary-condition blending factor; $k _ { \mathrm { M T } } ,$ , mass-transfer coeficient; PAC, pseudo-arclength continuation; FSG, Fuller–Schettler–Giddings; $\Phi _ { \mathrm { { L } } }$ , liquid-phase potential; cond(J), condition number of the Jacobian.

Even when the physics is implemented correctly, convergence errors are common in numerical models and are the most common culprits that stall model development. In two independent late-stage builds from diferent users, spurious points in the polarization curve appeared that partially converged but were not on the correct solution path. In both cases, the human-in-the-loop identified the issue through visual plot inspection and communicated it to their agent, which found that loose convergence criteria were admitting partially converged points. The agents autonomously converged on the same fix, tightening the absolute residual tolerance.

Nonetheless, the agents struggled to identify hidden physics bugs that do not block convergence, requiring human input to eliminate these physical inaccuracies. These silent errors are the most dangerous, since they can yield results that appear reasonable while being inaccurate. For example, an early guide included two expressions for the efective difusivity of liquid-phase species, one with the Bruggeman correction for porous media and one without, resulting in efective difusivities that difered by a factor of 3.54. In this case, none of the agents mentioned the ambiguity. Fortunately, when the agent was then instructed to cross-compare two replicates that had significant deviations in CO partial current density, it identified the diference in the parameter definition line by line. Notably, the ease of comparison was made possible by the standardization of the modular codebase in the specification guide. The agent could explain the physical implications of the diferences in $\mathrm { C O } _ { 2 }$ difusion coeficients, tracing the user's observation of higher CO partial currents to the fitting optimizer compensating for the decreased difusivity by increasing exchange current density. It also located the two conflicting definitions in the guide itself regarding the use of the Bruggeman correction, which were reconciled in the next iteration of the specification.

Because the agent writes every line of the solver, it does not merely return a number, exit flag, or convergence message; it can narrate the numerics in the language of the physics. During our builds, the agent could inspect the 1,480×1,480 sparse Stage-5 Jacobian and report its condition number, identify that the cell at the gas-side boundary carried the highest $\mathrm { C O } _ { 2 } \mathrm { R }$ reaction rate while the electrolyte-side cell was HER-dominated, and explain why in terms of local $\mathrm { C O } _ { 2 }$ availability and pH, and verify current continuity against the boundary flux. This diagnostic ability represents a key benefit of having a specialized agent for each multiphysics problem as opposed to relying on a generalized software package. When using commercial software, errors can be dificult to self-identify, leading to reliance on reporting to a dedicated support team. Conversely, agents can point to specific variables and their residuals in a given mesh element, trace that to the exact physics, and tune solvers for convergence barriers such as stifness and near-zero concentration terms.

## Error handling with multiphysics agents

To test the agent's ability to detect and resolve issues in a controlled manner and establish the robustness of the guide itself, we conducted a planted-error study. Unlike the natural errors above, which were a result of and could be compounded by guide incompleteness or ambiguity, these were systematically placed in separate copies of a mature pre-release version of the guide to assess the model’s ability to identify and handle these errors. For four planted errors (referred to as Cases 1–4), the agent was not told which relation had been modified and was prompted to build without referring to any files besides the specification, just as any other build. A fifth build used the unmodified guide as a control. All builds used Claude Opus 4.8 at maximum efort and received the same prompt. The four defects included a sign reversal in a boundary condition (Case 1, Fig. 5 row 1), an omitted term in a conservation law (Case 2, Fig. 5 row 2), an omitted correction factor in a constitutive relation (Case 3, Fig. 5 row 3), and a unit inconsistency in an empirical correlation (Case 4, Fig. 5 row 4). Each reproduces an error encountered naturally during development rather than one invented for the study (Supplementary Note 7). However, sandboxing these errors intentionally provides the cleanest possible study of how the agent addresses them, as opposed to allowing them to occur naturally due to unintentional guide ambiguities.

Three of the four defects were resolved before the final stage of model development. Of these three, two were resolved without human intervention (Cases 1 and 4), and one was addressed only after a hint from the human-in-the-loop (Case 2). The fourth was never detected (Case 3). Detection tracked whether the guide contradicted itself, not the severity or even the visibility of the symptom. Case 1's sign reversal made the boundary-value problem unsolvable, and the sweep failed outright. Case 2 made the Jacobian structurally singular. Case 4 produced transport coeficients thirty-fold higher than their correct values and the agent rejected them on magnitude at Stage 3. In principle, Case 2 should have been the most consequential (breaking a conservation law) and was also the most conspicuous, but a visible symptom alone was not suficient to be detected by the agent. Such errors often required a second textual signal, for instance, the modified guide contradicting itself. Case 1's reversed flux had the opposite sign to the Nernst–Planck boundary condition stated elsewhere in the same document, and Case 4's added units statement was incompatible with the Fuller–Schettler–Giddings correlation's own prefactor. Both errors were caught unaided. Case 2 left no such contradiction accessible to the agent and thus needed a hint from the operator.

Of the four planted errors, Case 3 (omission of porous media correction for difusivities) is the most informative failure, and the one that establishes the limits of the agents’ diagnostic capability. It is a plain constitutive error, no subtler in isolation than the others, but it was the only defect propagated consistently through every statement, example, and derived number in the guide. It was also numerically silent: the residuals converged at every stage, global charge conservation held to 2.4 × 10<sup>–11</sup>, and the one diagnostic that did register it, a liquid-potential drop at the bottom edge of the expected band, was explained away using the model's own inflated conductivity. Because the error could be compensated with the choice of kinetic parameters, the fitting step absorbed most of it, and the delivered model still reproduced the experimental polarization while carrying an exchange current density 41% below every other build. This is the error class that staged construction does not close on its own and requires discerning human rigor. This class of error is also quite common and is not specific to this multiphysics problem. A wrong constitutive relation transcribed faithfully and absorbed by a fit is a hazard of any model-building. The remedy is external reference, such as an independent implementation, an analytical limit, or a measurement the fit was not trained on. In this case, having the COMSOL model as an independent, orthogonal reference allowed the team to resolve silent errors in the implementation guide that caused deviations between the guide and the COMSOL implementation, which was treated as a “ground truth” or golden model.

The study also shows that the build process is resilient to minor errors that would otherwise have major impacts. Beyond syntax errors, the agents caught common mistakes that hinder model development and are tedious to find manually, including sign, unit, and math errors. Human mistakes are always a possibility regardless of experience level, so agents that can dissect code and dozens of parameters and equations in a fraction of the time provide a significant safety net. All syntax errors and most convergence errors arising from code implementation were caught by autonomous agent review, without human input.

## Fully autonomous builds

With human review eventually finding nothing else to catch during builds, the next step was pushing the bounds of agent autonomy. We gave the finalized implementation guide and the same prompt to three separate instances each of Claude Opus 5 and Sonnet 5 and instructed them to bypass the stop gates completely for a fully autonomous build with no human input. The agents were instructed to not refer to anything other than the guide, isolating the autonomous agents from previous builds and COMSOL data. The agent’s project memory, which carried a few results of earlier builds, was active in these builds as in all others (Methods); a tenth autonomous build made without it fitted $i _ { 0 } = 1 0 6 . 5 9 \mathrm { ~ A ~ m } ^ { - 2 } \mathrm { ~ a n d ~ } \lambda = 1 . 6 1 0 9 \mathrm { ~ e V } ,$ inside the range of all nine builds. They checked their own plots and validated each stage's output against the physical and numerical checks provided in the guide. All six builds produced virtually the same output (Fig. 6 and Supplementary Note 8), replicating the canonical model with $\mathrm { C O } _ { 2 } \mathrm { R }$ fitted parameters of $i _ { 0 }$ $= 1 0 6 . 5 6 \pm 0 . 0 7 \mathrm { A m } ^ { - 2 }$ and $\lambda = 1 . 6 1 1 3 \pm 0 . 0 0 0 2$ eV for Opus builds and $i _ { 0 } = 1 0 6 . 5 1 \pm 0 . 0 1 \mathrm { A m } ^ { - 2 }$ and $\lambda = 1 . 6 1 0 8 \pm 0 . 0 0 0 3$ eV for Sonnet builds (Supplementary Table 9).

These results show that LLM-assisted model building can culminate in fully autonomous builds once the implementation guide is fully developed. We emphasize that this was achievable only after substantial guide iteration. Nonetheless, the success of the Sonnet builds demonstrates a major payof of the refinement, which is that less powerful models can do the same task at a cheaper rate. Usage costs can limit the ability of users to perform complex or large tasks within a certain time frame, but with well-defined tooling and harnesses (in our case, the specification), the scope of ambition can be made larger with cheaper agents. For instance, autonomy allows agents to run replicates or comparative studies in parallel without constant supervision, yet it can lead to fast usage depletion or high costs if used with stronger models like Opus and Fable. Cheaper agents, like Sonnet, require a fraction of the cost and can yield similar results depending on the task. While achieving autonomy for model building is a great milestone, it does not eliminate the risk that an autonomous build can silently inherit whatever is wrong with its specification guide, once again emphasizing the importance of the human in establishing a stable implementation guide to harness the multiphysics agent.

a)  
![](images/2eb2279dcbcd1e1030ccc80bb4fe8bd6b243a584550af8ed100615a3e1e21865.jpg)  
b)

![](images/d55874e46d80297fdc2700f5cae5db3706fa77a57d263749303759fdf84bf018.jpg)  
c)

![](images/35e3279f3567a296a73dbffe7d25e1c0e8cff85072bf7c3415a0a5c912476ed8.jpg)  
Fig. 6. Autonomous agent builds without a human-in-the-loop. Comparison of Claude Opus 5 and Sonnet 5 build replicate (a) linear-scale and (b) log-scale polarization curves to COMSOL and experimental data. (c) Deviation of each build's $i _ { C O }$ from COMSOL at the COMSOL sweep potentials.

## Discussion

This work extends LLM-assisted scientific computing beyond benchmark PDEs and weakly coupled or single-physics problems to a true multiphysics model of an experimentally relevant electrochemical device. Although the model is one-dimensional, it is not numerically simple: Nernst–Planck transport, carbonate bufer chemistry, electrochemical kinetics, Stefan–Maxwell gas transport, Darcy flow, and electrostatics are coupled through nonlinear source terms and transport coeficients in a stif system containing 1,480 degrees of freedom. The resulting model reproduces expected physics and was calibrated to real experimental polarization and selectivity data and verified against an independently implemented COMSOL model. That provides a substantially stronger test of whether an agent can construct a model whose coupled physics actually reproduce the behavior of a real electrochemical system. More broadly, the results show that LLM-assisted multiphysics modeling can shift much of the technical burden from numerical implementation to physical specification and verification. The researcher still has to define the governing physics, choose the constitutive relationships and boundary conditions, recognize unphysical behavior, and validate the model. These decisions are where domain knowledge enters the model. For instance, we describe the porous electrode as a macrohomogeneous medium rather than a pore-resolved domain because that description resolves the polarization, selectivity, and concentration profiles studied here. The agent can implement either description, but the assumptions and simplifications appropriate to the question being asked remain the choice of the researcher. What the agent removes is the requirement that the researcher also translate that understanding into specialized numerical code.

The implementation guide therefore becomes the central scientific artifact in this workflow. Supplementary Table 12 summarizes the components that most strongly determined whether independent agents built the intended model rather than a plausible approximation, including complete governing equations and boundary conditions, explicit unit, sign, and reference-frame conventions, numerical methods, staged verification criteria, and expected physical magnitudes. These requirements are not simply instructions for an AI agent. Together, they provide an explicit record of the assumptions, numerics, and physical reasoning underlying the model that can be inspected, revised, and shared independently of any particular implementation. The limitations are equally important: staged verification catches many implementation and convergence failures, but a physically incorrect assumption stated consistently throughout the specification can still be reproduced faithfully and even obscured by parameter fitting. Independent builds, external references, and human physical intuition therefore remain important safeguards.

The opportunity is therefore not to remove the researcher from continuum modeling, but to make the researcher’s understanding of the physics the primary interface to it. The models produced here are open source and fully inspectable, and the agent that builds them can trace numerical behavior back through residuals, local concentrations, governing equations, and physical mechanisms in the same language the researcher uses to reason about the system. Researchers who understand an electrochemical device but are not specialists in numerical methods can therefore build, troubleshoot, extend, and share rigorous multiphysics models without surrendering control of either the physics or the implementation. By replacing a barrier of specialized code and proprietary software with one of stating and evaluating the physics precisely, agentic AI can make transparent multiphysics modeling accessible to a much broader scientific community, expanding the role of multiphysics models in the future of science and engineering research.

## Methods

## Model system and governing equations

The model developed herein is based upon prior work on modeling electrochemical $\mathrm { C O } _ { 2 }$ reduction in 1D and, importantly, presents a stif, highly nonlinear and coupled testbed with which to evaluate the eficacy and capabilities of our agentic multiphysics approach. Evaluating the efectiveness of building highly coupled and stif numerical models with LLM agents requires an existing solution against which to verify the implementation. We therefore rebuilt a $\mathrm { C O } _ { 2 } \mathrm { R }$ porous electrode model assembled from three previous COMSOL-based studies (Fig. 1b). The domain structure and GDE formulation follow Weng et a $\therefore ^ { 7 } ;$ the Marcus–Hush–Chidsey (MHC) kinetics<sup>27</sup> follow Lees et al.<sup>36</sup>; and the Ag-catalyzed $\mathrm { C O } _ { 2 } \mathrm { R }$ model of Muhieddine Orfali et al.<sup>37</sup> serves as the COMSOL reference that we rebuilt for benchmarking. We resolve the cathode catalyst layer (CL) and gas difusion layer (GDL) only and do not model the anode or membrane. The cathode alone is a suficiently demanding multiphysics problem for a first demonstration, and full-cell models and models beyond 1D will be subject of future work.

This system possesses each of the couplings described above. Two electrochemical reactions proceed at the catalyst surface,

$$
\mathrm { C O } _ { 2 } + \mathrm { H } _ { 2 } \mathrm { O } + 2 \mathrm { e } ^ { - }  \mathrm { C O } + 2 \mathrm { O H } ^ { - }\tag{1}
$$

$$
2 \mathrm { H } _ { 2 } \mathrm { O } + 2 \mathrm { e } ^ { - }  \mathrm { H } _ { 2 } + 2 \mathrm { O H } ^ { - }\tag{2}
$$

and both generate hydroxide, which is consumed by the carbonate bufer equilibria in the electrolyte,

$$
\mathrm { C O } _ { 2 } + \mathrm { O H } ^ { - }  \mathrm { H C O } _ { 3 } ^ { - }\tag{3}
$$

$$
\mathrm { H C O _ { 3 } ^ { - } } + \mathrm { O H ^ { - } }  \mathrm { C O _ { 3 } ^ { 2 - } } + \mathrm { H _ { 2 } O }\tag{4}
$$

closing a loop in which current raises the local pH, the elevated pH consumes dissolved ${ \mathrm { C O } } _ { 2 } ,$ and the resulting $\mathrm { C O } _ { 2 }$ depletion suppresses the kinetic $\mathrm { C O } _ { 2 } \mathrm { R }$ rate that produced the current as per mass transport limitations. Liquid-phase transport is described by Nernst–Planck dilute-solution theory with acid–base and electrochemical source terms, gas transport by Stefan–Maxwell concentrated-solution theory with Henry’s law phase transfer, and gas-phase momentum by Darcy’s law; $\mathrm { C O } _ { 2 } \mathrm { R }$ is modeled with MHC kinetics, evaluated with the closed-form approximation<sup>38</sup>, and the competing hydrogen evolution reaction (HER) with Butler–Volmer kinetics. The model is one-dimensional, steady-state, and isothermal at $5 0 ^ { \circ } C ,$ with fixed porosity and liquid saturation. The CL is treated as three phases, solid catalyst, ionomer/liquid, and gas-filled pore, each with fixed volume fractions defined by experimentally measured porous electrode structure. The GDL is treated as dry, and crossover to the anolyte is not modeled. The complete governing equations and boundary conditions are given in Supplementary Notes 1 and 2, the parameter set in Supplementary Note 9 and Supplementary Table 11, and the COMSOL build used for benchmarking in Supplementary Note 3.

## The implementation guide

Critical to building well-developed, rigorous models are equally well-developed implementation guides for the agent. Implementation guides are machine-readable specifications (often in the form of markdown files) that fully describe and specify the problem to be modeled, including all relevant parameters, equations, boundary conditions, and include a scafolded build plan that the coding agent must follow. Essentially, the implementation guide is a human-specified harness to constrain the multiphysics agent. These guides link the science from the human to the numerical implementation of the agent. Ensuring these guides are thorough and precise is necessary to prevent ambiguity for the agent that might lead to errors and inconsistencies in building the model. Hence, the human operator’s understanding of the physics and chemistry of a given multiphysics problem remains crucial to developing the model. In this workflow, the agent’s primary role is not inference but implementation, removing the numerical implementation barrier that otherwise hinders scientific modeling and has been largely inaccessible to chemists in the past.

The guide used in this case study includes 18 sections covering everything needed for the full implementation of the multiphysics model: all equations, parameters, and solver information. Key structural elements include the complete physics of the system, full parameter tables, the numerical methods, and anticipated failure modes with solutions (Supplementary Note 10). The complete guide is available in the repository given under Code availability and is also provided as a supplementary file.

In all, the guide's components are made to be as explicit and precise as possible to mitigate any inference or hallucination on the agent's part. This mirrors the prompting literature, where explicit, structured, example-bearing instructions outperform terse ones<sup>39</sup>. Importantly, a proper specification can also be read and audited by a human such that the human’s physical intuition properly guides the development of the guide and constrains the multiphysics problem. The physics lay out the governing equations explicitly, not citing by name such as “standard Nernst–Planck”. The parameter tables list every constant, including its name, symbol, value, units, and source. The numerical methods in this study include a finite-volume discretization, using a Newton solver with damping (Supplementary Table 5). Bootstrap protocols for preconditioning and methods for initializing solvers are included to improve convergence for stif problems such as the $\mathrm { C O } _ { 2 } \mathrm { R }$ problem employed herein.

The methods were specified from physical insight and user experience building models in COMSOL, described in suficient detail to prevent the agent from making any critical algorithmic decisions. We note that the significant degree of scafolding used in this workflow is required for the current state of frontier models due to their tendency to invoke unphysical shortcuts or to hallucinate parameters<sup>40</sup>. However, purpose-built agents<sup>41,42</sup> or future frontier models may be able to handle greater flexibility in specification. Notably, this methodology is also transferable to other systems: a fuel cell or a Li-ion pseudo-two-dimensional model<sup>43,44</sup> guide would follow the same structure, with only the equations and parameters needing changes. Regardless, no specification is immune to ambiguity, which is why staged verification and guide refinement are essential. Having a human-in-the-loop for the development of the guide, at this stage, appears essential.

Because the guide is a plain-text markdown file, it can be version-controlled and developed collaboratively. In this work, the implementation guide was maintained in a shared GitHub repository, where co-authors iteratively refined its physics, parameters, and numerical methods and shared the next refinement targets. The same versioned guide could be given to multiple researchers who independently instructed agents to build the model, where each independent model build was treated as a replicate. Divergent outputs exposed ambiguities or errors in the guide (Supplementary Tables 8 and 10), while agreement across independent builds gave confidence in both the model and the specification. This ability to collaborate on a single humanand machine-readable artifact, rather than on an opaque, version-locked solver project, is itself a central advantage of the workflow.

## Staged construction and stop gates

Building the model in stages of sequentially increasing complexity is essential to enable human review of output, identify and resolve issues before they accumulate and compound, and subsequently iterate upon the guide (Fig. 1). The guide decomposes the model into five stages of increasing physical complexity. In Stage 1, liquid Nernst–Planck, bufer reactions, and electrochemical kinetics are included with Dirichlet boundary conditions (BCs), initializing the problem with 400 degrees of freedom (DOFs) at the production mesh of 80 CL cells and 120 GDL cells. In Stage 2, Sherwood–Reynolds mass-transfer (Robin) BCs replace the Dirichlet to capture finite mass transfer at the electrolyte boundary. In Stage 3, Stefan–Maxwell gas transport and Henry's law phase transfer are added, expanding the system to 1,200 DOFs. In Stage 4, the Darcy pressure equation is finalized, completing the gas-phase momentum balance. A fitting step then estimates the kinetic parameters for $\mathrm { C O } _ { 2 } \mathrm { R }$ and HER against the experimental data of Romiluyi et al.<sup>25</sup>, exploiting the well-conditioned Stage 4 residual before the coupling of the charged species to electrostatic potential causes stifness to grow further. Stage 5 then turns on the full electrochemistry for the migration term in the Nernst–Planck flux, solving for liquid potential φ monolithically from the per-cell ionic charge balance in the CL and solid potential $\varphi _ { \mathsf { s } }$ in the GDL and CL via Ohm's law. This brings the full system to 1,480 DOFs at the 80-CL/120-GDL cell production mesh (Supplementary Table 2). The kinetic parameters fitted at Stage 4 are held fixed in Stage 5, where migration raises the simulated CO partial current (Fig. 2a).

Crucially, the stages are designed to include stop gates where the human-in-the-loop can review quantitative physicality checks (e.g., checking electroneutrality to machine precision, mass conservation, current continuity, and Newton residual convergence) before authorizing the next stage. These checks are prescribed by the guide itself rather than left to the discretion of the agent. For each stage, the guide specifies the quantitative pass/fail criteria as well as diagnostic plots the agent must produce (Supplementary Note 5), to enable criteria-driven and reproducible human review across builders. These output plots force data visualization, allowing the user to directly evaluate the model and accelerate diagnostics. In our case, a physical understanding of the expected general trends, including those in polarization behavior, faradaic eficiencies (FEs), and concentration profiles, makes these plots more interpretable than a list of numbers. A visual inspection to verify that trends are accurate, such as pH increasing with cathodic current and $\mathrm { C O } _ { 2 }$ concentration dropping with increasing overpotential, provides evidence that the model is building properly.

This framework is essential for revealing failure modes in isolation and facilitating independent human verification of all model outputs, rather than relying solely on the agent’s self-assessment. For instance, an error in the bufer equilibria implementation afects local pH, and thus $\mathrm { C O } _ { 2 }$ concentration, reaction rates, liquid phase potential, etc., contaminating practically every part of the highly coupled problem. Locating errors in a liquid-only 400-DOF sub-problem is much simpler than allowing those same errors to get buried inside the full-physics 1,480-DOF system. Establishing stop gates addresses the issue of compounding complexity and allows the user to more efectively troubleshoot. Errors during model building could be due to mistakes made by the agent, but these stop gates also reveal inconsistencies or ambiguities in the guide, allowing the user to refine and iterate to make it more comprehensive and reproducible across instances of the agent. We note that frontier AI models are improving rapidly, even within the timespan of this study, and could eventually check their own work or work in teams of agents<sup>18,19</sup> to develop these rigorous models. At this stage, however, we find that building a model specification of this scale from scratch in one prompt is not practical, and scafolding makes the workflow reliable and generalizable.

## Agents and models

Throughout guide development, we used the most capable Claude model available at the time because of the task's complexity. In earlier builds from less mature guides, weaker Sonnet-tier models introduced more errors than Opus, and the additional time spent correcting them, at comparable token costs, outweighed any benefit. Opus 4.6 was used to initialize guide development and perform much of the front-end refinement. Opus 4.7 was then primarily used for finalizing the model, and Opus 5 was used to make the final builds for comparison to COMSOL. Opus 4.8 was used for the planted-error study, and Opus 5 and Sonnet 5 for the fully autonomous builds. The choice to use a frontier model (Claude) rather than training our own purpose-built agent was made to first demonstrate the feasibility of the methodology with a well-established and powerful agent that, in theory, possesses intrinsic knowledge of solving similar nonlinear PDE problems.

These agents were used by multiple users across laboratories, with varying experience levels, to develop the same implementation guide. When building models, researchers tracked their build processes in build dialogue files (Supplementary Note 7 and Supplementary Table 8) and instructed their agents to refer only to the implementation guide (apart from the agent’s project memory, described below), so that its quality could be evaluated on its own and without bias from other builds or documents in the repository. In practice, more resources and context are beneficial for agents, and comparing replicates built from the same guide proved helpful in compensating for the intrinsic probabilistic nature of LLMs.

The coding agent’s persistent project memory was active in all nine builds reported here. Its index, loaded at the start of each session, quoted results of builds made from an earlier version of the guide, including $i _ { 0 } \approx 1 0 6 . 5 \mathrm { A m } ^ { - 2 } ,$ , a Stage-5 peak $i _ { C O } \approx 3 3 4 \mathrm { m A c m } ^ { - 2 }$ and a note that a COMSOL run at that $i _ { 0 }$ agreed to about 1%. It held no value of λ or the HER parameters, no polarization curve and no COMSOL output, and no build opened another build’s files or COMSOL output before it was complete. The guide itself states expected values at its stop gates, about twenty of them measured on the first of the nine builds. The agreement between builds does not rest on these values: $\lambda ,$ the HER parameters and the full 89-point polarization curves, which had no stated target, agree across builds as closely as i<sub>0</sub> does, and each fit starts from the guide’s prescribed initial point and is deterministic. A tenth build made in a directory containing only the guide, with no project memory, fitted $i _ { 0 } = 1 0 6 . 5 9 \mathrm { A m } ^ { - 2 }$ and $\lambda = 1 . 6 1 0 9 \mathrm { e V }$ (Supplementary Note 8).

## Planted-error study

To perform the planted error study, specific defects were planted in the specification guide, not in the code, reproducing the failure mode of practical interest. In this failure mode, the statement of the physics handed to the implementer is itself wrong and is transcribed faithfully, propagating the error from the implementation guide to the codebase. Detection therefore requires the agent to recognize a conflict between what it has been told to implement and either the physics or the remainder of the specification and to raise that conflict rather than deviate silently. Each build received the same instruction: implement the model exactly as the guide specifies, consult no other file, halt at each stage gate for human review, and maintain a running build log recording every error encountered, every error corrected without human input, and every discrepancy noticed in the guide. The agent was not told that a defect existed or where. At each stage gate, the human reviewer inspected the plotted stage output and approved or rejected progression without auditing the source code. A defect was scored as detected only if the build log identified it as a defect, named the afected relation, and proposed a correction. Where a modified guide contained explanatory or diagnostic text naming the afected quantity, that text was edited along with the defect, so detection could not follow from an isolated self-contradiction. Per-case results are given in Supplementary Note 6 and Supplementary Table 7.

## Solve-time benchmarking and mesh convergence

Both implementations were timed on the same workstation (Intel Xeon w7-3565X, 32 cores/64 threads, 191 GB RAM) at matched core budget, with COMSOL on 32 cores (20,033 DOFs; Supplementary Table 1) and Julia on 32 threads (20,038 DOFs; Supplementary Note 4 and Supplementary Table 6). COMSOL swept 61 potentials at 50 mV spacing and Julia 89 at 25 mV, and both converged at every potential (Fig. 4a–b). Reported Julia times are the fastest of three repetitions, with run-to-run variability reported in Supplementary Table 4, and the COMSOL time was the same in both of its recorded runs; process start-up and just-in-time compilation, a further 15–20 s per invocation, are excluded from both. The Julia data were obtained under the Windows Subsystem for Linux rather than natively, because the MUMPS binary distribution, used for the solver ablation, has no working Windows-native build. Mesh convergence of the Stage 5 solution is documented in Supplementary Note 4 and Supplementary Fig. 2.

## Reporting summary

Further information on research design is available in the Nature Portfolio Reporting Summary linked to this article.

## Data availability

The experimental partial current densities used for kinetic parameter estimation are taken from Romiluyi et al. and are reproduced in Supplementary Table 3. The simulated data underlying Figs. 2, 3 and 6, and the COMSOL results they are compared with, are included in the code repository (see Code availability); the timings in Fig. 4 are listed in Supplementary Tables 4 and 6.

## Code availability

The implementation guide, the Julia model source for all five build stages, and the build dialogue logs are available at https://github.com/Sebobbit/CO2R-Bulk-Scale-Model, with the Julia source released under the MIT license. The four modified guides used in the planted-error study, together with the unmodified control guide, are provided as separate files in the same repository. The COMSOL reference model is included; opening and running it requires a COMSOL Multiphysics license (v6.4) with the Electrochemistry and CFD modules, and its exported results are provided as plain-text files so that the comparison in Fig. 3 can be reproduced without a license.

## Acknowledgements

The authors acknowledge professors Miguel A. Modestino and Rohini B. Chandran for helpful discussions.

## Funding

S.C. discloses support for the research of this work from the National Science Foundation Graduate Research Fellowship Program [DGE-2234660]. J.C.B. discloses support for the research of this work from Schmidt Science Fellows, in partnership with the Rhodes Trust. E.W.L. acknowledges funding support from the Peter Wall Legacy Award and the Natural Sciences and Engineering Research Council of Canada Discovery Grants program (RGPIN-2025-04804). Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation.

## Author contributions

S.C.: Investigation, Visualization, Writing – original draft, Writing – review & editing. M.F.S.: Investigation, Software. S.M.: Investigation. E.W.L.: Investigation, Supervision, Writing – review & editing. J.C.B.: Investigation, Supervision, Writing – review & editing.

## Competing interests

J.C.B., S.C., M.F.S. and E.W.L. are inventors on a provisional patent application filed by New York University and the University of British Columbia covering the specification-driven modeling workflow described here (US Provisional Application no. 64/168,700). The authors declare they have no other conflicts of interest.

## References

1. Chu, S. & Majumdar, A. Opportunities and challenges for a sustainable energy future. Nature 488, 294–303 (2012).

2. De Luna, P. et al. What would it take for renewably powered electrosynthesis to displace petrochemical processes? Science 364, eaav3506 (2019).

3. Burdyny, T. & Smith, W. A. CO2 reduction on gas-difusion electrodes and why catalytic performance must be assessed at commercially-relevant conditions. Energy Environ. Sci. 12, 1442–1453 (2019).

4. Welch, A. J. et al. Operando Local pH Measurement within Gas Difusion Electrodes Performing Electrochemical Carbon Dioxide Reduction. J. Phys. Chem. C 125, 20896–20904 (2021).

5. Bui, J. C. et al. Continuum Modeling of Porous Electrodes for Electrochemical Synthesis. Chem. Rev. 122, 11022–11084 (2022).

6. Newman, J. & Balsara, N. P. Electrochemical Systems. (John Wiley & Sons, Incorporated, Newark, 2021).

7. Weng, L.-C., Bell, A. T. & Weber, A. Z. Modeling gas-difusion electrodes for CO2 reduction. Phys. Chem. Chem. Phys. 20, 16973–16984 (2018).

8. Weng, L.-C., Bell, A. T. & Weber, A. Z. Towards membrane-electrode assembly systems for CO2 reduction: a modeling study. Energy Environ. Sci. 12, 1950–1968 (2019).

9. Ringe, S. et al. Understanding cation efects in electrochemical CO2 reduction. Energy Environ. Sci. 12, 3001–3014 (2019).

10. Newman, J. Numerical Solution of Coupled, Ordinary Diferential Equations. Ind. Eng. Chem. Fundam. 7, 514–517 (1968).

11. Newman, J. S. & Tobias, C. W. Theoretical Analysis of Current Distribution in Porous Electrodes. J. Electrochem. Soc. 109, 1183 (1962).

12. Dickinson, E. J. F., Ekström, H. & Fontes, E. COMSOL Multiphysics®: Finite element software for electrochemical analysis. A mini-review. Electrochem. Commun. 40, 71–74 (2014).

13. Sulzer, V., Marquis, S. G., Timms, R., Robinson, M. & Chapman, S. J. Python Battery Mathematical Modelling (PyBaMM). J. Open Res. Softw. 9, (2021).

14. Permann, C. J. et al. MOOSE: Enabling massively parallel multiphysics simulation. SoftwareX 11, 100430 (2020).

15. Peng, R. D. Reproducible Research in Computational Science. Science 334, 1226–1227 (2011).

16. M. Bran, A. et al. Augmenting large language models with chemistry tools. Nat. Mach. Intell. 6, 525–535 (2024).

17. Boiko, D. A., MacKnight, R., Kline, B. & Gomes, G. Autonomous chemical research with large language models. Nature 624, 570–578 (2023).

18. Ni, B. & Buehler, M. J. MechAgents: Large language model multi-agent collaborations can solve mechanics problems, generate new data, and integrate knowledge. Extreme Mech. Lett. 67, 102131 (2024).

19. Park, D., Moon, H. & Ryu, S. A self-correcting multi-agent LLM framework for language-based physics simulation and explanation. Npj Artif. Intell. 2, 10 (2026).

20. Lupoiu, R. et al. A multi-agentic framework for real-time, autonomous freeform metasurface

design. Sci. Adv. 11, eadx8006 (2025).

21. Li, S. et al. CodePDE: An Inference Framework for LLM-driven PDE Solver Generation. Preprint at https://doi.org/10.48550/arXiv.2505.08783 (2026).

22. Adhikari, S., Noorsumar, G. & Jensen, Ø. PDE-Agents: An LLM-Orchestrated Multi-Agent Framework for Automated Finite Element Simulations with Knowledge Graph-Augmented Reasoning. Preprint at https://doi.org/10.48550/arXiv.2606.07850 (2026).

23. Bezanson, J., Edelman, A., Karpinski, S. & Shah, V. B. Julia: A Fresh Approach to Numerical Computing. SIAM Rev. 59, 65–98 (2017).

24. Oberkampf, W. L. & Trucano, T. G. Verification and validation in computational fluid dynamics. Prog. Aerosp. Sci. 38, 209–272 (2002).

25. Romiluyi, O., Danilovic, N., Bell, A. T. & Weber, A. Z. Membrane-electrode assembly design parameters for optimal CO2 reduction. Electrochem. Sci. Adv. 3, e2100186 (2023).

26. Corpus, K. R. M. et al. Coupling covariance matrix adaptation with continuum modeling for determination of kinetic parameters associated with electrochemical CO2 reduction. Joule 7, 1289–1307 (2023).

27. Chidsey, C. E. D. Free Energy and Temperature Dependence of Electron Transfer at the Metal-Electrolyte Interface. Science 251, 919–922 (1991).

28. Nelder, J. A. & Mead, R. A Simplex Method for Function Minimization. Comput. J. 7, 308–313 (1965).

29. Mogensen, P. K. & Riseth, A. N. Optim: A mathematical optimization package for Julia. J. Open Source Softw. 3, 615 (2018).

30. Schenk, O. & Gärtner, K. Solving unsymmetric sparse systems of linear equations with PARDISO. Future Gener. Comput. Syst. 20, 475–487 (2004).

31. Davis, T. A. Algorithm 832: UMFPACK V4.3---an unsymmetric-pattern multifrontal method. ACM Trans. Math. Softw. TOMS 30, 196–199 (2004).

32. Amestoy, P. R., Duf, I. S., L’Excellent, J.-Y. & Koster, J. A Fully Asynchronous Multifrontal Solver Using Distributed Dynamic Scheduling. SIAM J. Matrix Anal. Appl. 23, 15–41 (2001).

33. Revels, J., Lubin, M. & Papamarkou, T. Forward-Mode Automatic Diferentiation in Julia. Preprint at https://doi.org/10.48550/arXiv.1607.07892 (2016).

34. Brady, N. W., Mees, M., Vereecken, P. M. & Safari, M. Implementation of Dual Number Automatic Diferentiation with John Newman’s BAND Algorithm. J. Electrochem. Soc. 168, 113501 (2021).

35. Ouyang, S., Zhang, J. M., Harman, M. & Wang, M. An Empirical Study of the Non-Determinism of ChatGPT in Code Generation. ACM Trans. Softw. Eng. Methodol. 34, 42:1-42:28 (2025).

36. Lees, E. W., Bui, J. C., Romiluyi, O., Bell, A. T. & Weber, A. Z. Exploring CO2 reduction and crossover in membrane electrode assemblies. Nat. Chem. Eng. 1, 340–353 (2024).

37. Muhieddine Orfali, D. et al. Pulsed Electrolysis Promotes Catalyst Activity in Dilute CO2 Streams. ACS Energy Lett. 11, 4988–4997 (2026).

38. Zeng, Y., Smith, R. B., Bai, P. & Bazant, M. Z. Simple formula for Marcus–Hush–Chidsey kinetics. J. Electroanal. Chem. 735, 77–83 (2014).

39. Schulhof, S. et al. The Prompt Report: A Systematic Survey of Prompt Engineering Techniques. Preprint at https://doi.org/10.48550/arXiv.2406.06608 (2025).

40. Huang, L. et al. A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions. ACM Trans. Inf. Syst. 43, 42:1-42:55 (2025).

41. Zou, Y. et al. El Agente: An autonomous agent for quantum chemistry. Matter 8, 102263 (2025).

42. Zhang, Z. et al. El Agente Forjador: Task-Driven Agent Generation for Quantum Simulation. Preprint at https://doi.org/10.48550/arXiv.2604.14609 (2026).

43. Weber, A. Z. & Newman, J. Modeling Transport in Polymer-Electrolyte Fuel Cells. Chem. Rev. 104, 4679–4726 (2004).

44. Doyle, M., Fuller, T. F. & Newman, J. Modeling of Galvanostatic Charge and Discharge of the Lithium/Polymer/Insertion Cell. J. Electrochem. Soc. 140, 1526 (1993).