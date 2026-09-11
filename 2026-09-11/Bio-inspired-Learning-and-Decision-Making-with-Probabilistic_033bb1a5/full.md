# Bio-inspired Learning and Decision-Making with Probabilistic In-Memory Computing Hardware: Part 1

Thomas Dalgaty<sup>1</sup>, Eiji Kawasaki<sup>1</sup>, Miguel de Prado<sup>2,</sup> <sup>3</sup>, Devendra Vyas<sup>2</sup>, Tommaso Salvatori<sup>2</sup> <sup>4</sup>

<sup>1</sup>CEA-List, Grenoble, France

<sup>2</sup> formerly VERSES AI Research Lab

<sup>3</sup> now PRAESC AI <sup>4</sup> now TU Wien, Vienna, Austria

## Abstract

Learning and decision-making in animals are often modeled as Bayesian processes, where sensory evidence is integrated with prior beliefs to guide behavior in the face of uncertainty. But what are the inherent neural dynamics that give rise to this ability, and how could they be replicated in computing systems? This abstract discusses a biologically grounded framework in which noisy neural and synaptic dynamics perform inference and learning via stochastic sampling from an internal energy function, capturing uncertainty over latent states and model parameters through neural and synaptic variability, respectively. This enables approaches such as predictive coding networks to account for epistemic uncertainty via Markov chain Monte Carlo sampling. Drawing a parallel between intrinsic noise in biological systems and electrical noise in emerging probabilistic analogue memory technologies, we highlight how analogue in-memory computing hardware naturally emerges as the solution for massively scalable and energy-efficient probabilistic inference.

## Introduction

Animals exhibit remarkable adaptability in uncertain and dynamic environments, often behaving in ways that approximate Bayesian inference (Knill and Richards 1996; Knill and Pouget 2004; Paunov et al. 2024). This has led to the hypothesis that the brain performs probabilistic reasoning by integrating sensory evidence with prior expectations, guided by uncertainty. Energy-based modeling is a compelling framework for understanding this process, which views the brain as a physical system minimizing its internal free energy (Hinton, Sejnowski et al. 1986; Friston 2010) to perform inference and learning efficiently. This has inspired computational formulations, such as predictive coding, in which neural populations continuously generate predictions about sensory input and update them through the minimization of prediction errors, implicitly performing Bayesian inference (Rao and Ballard 1999; Friston 2005; Salvatori et al. 2023). Biological neural systems, such as the brain, are inherently noisy. Thermodynamic fluctuations, stochastic neurotransmitter release, and variability in synaptic transmission (Fatt and Katz 1950; Conti, Tan, and Llano 2004; White, Rubinstein, and Kay 2000) introduce randomness at both the neuronal and synaptic levels (Buesing et al. 2011; Kappel et al. 2015). Rather than being detrimental however, this noise may serve a computational role by enabling sampling-based inference through neural and synaptic variability. Increasing evidence suggests that the brain leverages its intrinsic stochastic dynamics to perform Bayesian inference via mechanisms akin to Markov chain Monte Carlo (MCMC) sampling (Hoyer and Hyvarinen 2002; Berkes¨ et al. 2011).

In this abstract, we motivate a biologically inspired framework for learning and decision-making that uses noise as a computational resource. Specifically, we discuss the development of a new form of predictive coding (Oliviers, Bogacz, and Meulemans 2024; Sennesh, Wu, and Salvatori 2024), capable of modeling its epistemic uncertainty. While predictive coding has been extensively studied as a mechanism for perceptual inference, its treatment of epistemic uncertainty — uncertainty about the parameters or structure of the generative model — remains underexplored. In particular, most implementations i) treat synaptic weights in a deterministic fashion, neglecting the possibility that the weights themselves could represent probability distributions, or ii) propose a variational posterior over the weights (Tschantz et al. 2025), not reflecting the persistent changes that result from biological noise according to synaptic sampling theory (Kappel et al. 2015).

We propose a pathway whereby applying stochastic updates to both neural and synaptic variables within an underlying energy function, inference and learning naturally emerge as biologically plausible Markov chain Monte Carlo sampling processes (Algorithmic Formulation). In addition, we draw a parallel between intrinsic biological noise in the brain and the electrical noise in semiconductor technologies (Bayesian Neuromorphic Hardware). Just as MCMC sampling offers a biologically plausible mechanism for Bayesian inference in neural systems, it also provides a framework for leveraging the inherent variability of analogue neuromorphic hardware. Rather than being a limitation, this devicelevel noise can be harnessed as a computational asset, enabling scalable energy-efficient Bayesian learning and inference (Dalgaty et al. 2021; Lin et al. 2025).

## Algorithm Formulation

The unnormalized posterior probability distribution over the states of a population of neurons, θ, with synaptic weights,

ω, given an observation X can be expressed as:

$$
{ \tilde { P } } ( \theta , \omega \mid X ) = P ( X \mid \theta , \omega ) \cdot P ( \theta , \omega ) .\tag{1}
$$

The Boltzmann distribution can then be used to relate this unnormalized posterior to the joint energy function of $\theta , \omega$ and X

$$
{ \tilde { P } } ( \theta , \omega \mid X ) = \exp { \big ( } - E ( \theta , \omega , X ) { \big ) } .\tag{2}
$$

This, therefore, allows the log probability of the posterior to be written as the combination of two energy functions

$$
\log \tilde { P } ( \theta , \omega \mid X ) = - E _ { \mathrm { { L } } } ( X , \theta , \omega ) - E _ { \mathrm { { p r i o r } } } ( \theta , \omega ) .\tag{3}
$$

Depending on the choice of energy function and prior, Markov chain Monte Carlo methods can be used to generate samples from the neuron state probability density functions, $p ( \theta )$ , and synaptic weight probability distributions, $p ( \omega )$ . In the case of a shallow, linear predictive coding network with weights $\omega ,$ latent variables (neurons) $\theta ,$ and prediction errors $e ^ { ( t ) } = X - \omega ^ { ( t ) } \theta ^ { ( t ) }$ , Langevin dynamics can be used to draw these samples through the updates

$$
\begin{array} { r } { \Delta \theta ^ { ( t ) } = \tau _ { \theta } \left[ \frac { 1 } { \sigma _ { X } ^ { 2 } } \omega ^ { ( t ) \top } e ^ { ( t ) } - \frac { 1 } { \sigma _ { \theta } ^ { 2 } } ( \theta ^ { ( t ) } - \mu _ { \theta } ) \right] + \sqrt { 2 \tau _ { \theta } } \epsilon _ { \theta } ^ { ( t ) } , } \end{array}
$$

$$
\begin{array} { r } { \Delta \omega ^ { ( t ) } = \tau _ { \omega } \left[ \frac { 1 } { \sigma _ { X } ^ { 2 } } e ^ { ( t ) } \theta ^ { ( t ) \top } - \frac { 1 } { \sigma _ { \omega } ^ { 2 } } \omega ^ { ( t ) } \right] + \sqrt { 2 \tau _ { \omega } } \epsilon _ { \omega } ^ { ( t ) } , } \end{array}\tag{4}
$$

where $\epsilon _ { \theta }$ and $\epsilon _ { \omega }$ denote Gaussian unit noise, $\tau _ { \theta }$ and $\tau _ { \omega }$ are the neural and synaptic time constants, $\mu _ { \theta }$ is the prior mean of the latent neurons, and $\sigma _ { X } ^ { 2 } , \sigma _ { \theta } ^ { 2 }$ and $\dot { \sigma } _ { \omega } ^ { 2 }$ are the variances of the Gaussian likelihood, the latent prior and the zeromean weight prior, respectively. Consistent with synaptic sampling theory, the weights evolve on a slower timescale than the neural activity, $\tau _ { \omega } \ll \tau _ { \theta }$ (Kappel et al. 2015). When learning from a dataset rather than a single observation, the likelihood term of the weight update must be scaled to reflect the full dataset, as in stochastic gradient Langevin dynamics (Welling and Teh 2011), so that the weights sample the posterior given all observations.

A single well-mixed chain of these dynamics draws samples from the weight posterior and therefore captures epistemic uncertainty, but reading this uncertainty out requires waiting for the chain to mix over time. Running an ensemble of chains in parallel, each a particle walking over the same weight posterior, instead provides an instantaneous estimate of epistemic uncertainty through the variability across the ensemble. Local groups of neural circuits sharing the same structural motifs (Narayanan, Kimchi, and Laubach 2005; Yuste, Cossart, and Yaksi 2024) could be a mechanism by which such ensembles are realized in biological neural networks.

Unlike similar works that rely on the Hopfield energy, we adopt the predictive coding energy. The popularity of the Hopfield energy is driven by two primary factors. First, its symmetric weights avoid the need to compute the transpose of the weight matrix, an operation whose biological implementation remains debated. Second, its dynamics have been extensively analyzed and are well understood. The predictive coding energy nevertheless offers several advantages: its dynamics are simpler, and in the shallow linear model considered here the steady state of the network can be obtained analytically. It also naturally operates in its generative formulation, implementing a hierarchical Gaussian model (Friston 2005). The transpose appearing in the neural update of equation 4 does not pose a problem for analogue hardware, where crossbar arrays can be read in both the forward and reverse directions, although its biological plausibility remains an open question shared with predictive coding more broadly.

## Probabilistic analogue in-memory computing

The inference and learning algorithm outlined in equation 4 is appealing because of the resonance with neural and synaptic sampling theories. However, it is based on running an ensemble of parallel Langevin chains over the same posterior. The execution of MCMC is notoriously slow on conventional computer architectures, making the algorithmic latency required to perform this over a large ensemble quickly prohibitive. For example, in deterministic predictive coding models, a solution on a small network can be reached in about 15 iterations (Salvatori et al. 2022), while a similar model with Langevin dynamics needs more than 200 (Oliviers, Bogacz, and Meulemans 2024).

Just as biological systems are rich with intrinsic noise, so too are the semiconducting technologies used to build computing hardware. Thermal noise (Johnson 1928; Nyquist 1928) arises from the Brownian agitation of conducting particles, shot noise (Schottky 1918) from quanta of particles overcoming energy barriers, and analogue memory devices exhibit programming variability due to the stochastic nature of mechanisms like conductive filament formation (Dalgaty et al. 2021; Lin et al. 2025) or magnetic spin change (Dalgaty et al. 2023). These physical phenomena have already been mapped onto efficient implementations of MCMC sampling, such as Metropolis-adjusted samplers and Langevin dynamics. They are particularly efficient because analogue memory devices can be used to draw samples from probability distributions, store these samples, and then perform vector arithmetic using these stored samples, all within the memory circuit. This is in contrast to GPUs, where parameters must be loaded in from an external high-bandwidth DRAM memory, subject to a limited bandwidth. This bandwidth imposes a sequential execution on an algorithm that is intrinsically massively-parallelizable. In-memory computing is, by contrast, structurally a massively parallel computing approach and offers the potential of speeding up the execution of energy-based models by several orders of magnitude.

## Research Outlook

Within the European network of excellence dAIEDGE, we aim to advance the probabilistic analogue in-memory computing paradigm by showing how it can enable a massive scaling-up of energy-based models such as predictive coding. Such hardware can reduce the computational cost of explicitly modeling epistemic uncertainty, opening up new possibilities for frameworks such as active inference, where actions are guided by uncertainty-driven exploration. Our immediate focus is twofold. First, we will investigate the performance of our proposed Bayesian predictive coding network for learning and making decisions within the energy-based framework. Second, we will explore how the algorithm detailed in this abstract can be efficiently mapped onto the intrinsic stochastic physics of analogue in-memory computing hardware, which naturally reflects the variability observed in biological systems.

## Acknowledgments

This collaboration was supported by Horizon Europe’s dAIEDGE network of excellence - Grant Agreement Number 101120726.

## References

Berkes, P.; Orban, G.; Lengyel, M.; and Fiser, J. 2011. Spon-´ taneous cortical activity reveals hallmarks of an optimal internal model of the environment. Science, 331(6013): 83– 87.

Buesing, L.; Bill, J.; Nessler, B.; and Maass, W. 2011. Neural dynamics as sampling: a model for stochastic computation in recurrent networks of spiking neurons. PLoS computational biology, 7(11): e1002211.

Conti, R.; Tan, Y. P.; and Llano, I. 2004. Action potentialevoked and ryanodine-sensitive spontaneous Ca2+ transients at the presynaptic terminal of a developing CNS inhibitory synapse. Journal of Neuroscience, 24(31): 6946– 6957.

Dalgaty, T.; Castellani, N.; Turck, C.; Harabi, K.-E.; Querlioz, D.; and Vianello, E. 2021. In situ learning using intrinsic memristor variability via Markov chain Monte Carlo sampling. Nature Electronics, 4(2): 151–161.

Dalgaty, T.; Yamada, S.; Molnos, A.; Kawasaki, E.; Mesquida, T.; Rummens, F.; Shibata, T.; Urakawa, Y.; Terasaki, Y.; Sasaki, T.; et al. 2023. Scaling-up Memristor Monte Carlo with magnetic domain-wall physics. In MLNCP2023-37th NeurIPS Machine Learning with New Compute Paradigms workshop.

Fatt, P.; and Katz, B. 1950. Some observations on biological noise. Nature, 166(4223): 597–598.

Friston, K. 2005. A theory of cortical responses. Philosophical Transactions of the Royal Society B: Biological Sciences, 360(1456).

Friston, K. 2010. The free-energy principle: a unified brain theory? Nature reviews neuroscience, 11(2): 127–138.

Hinton, G. E.; Sejnowski, T. J.; et al. 1986. Learning and relearning in Boltzmann machines. Parallel distributed processing: Explorations in the microstructure of cognition, 1(282-317): 2.

Hoyer, P.; and Hyvarinen, A. 2002. Interpreting neural re-¨ sponse variability as Monte Carlo sampling of the posterior. Advances in neural information processing systems, 15.

Johnson, J. B. 1928. Thermal agitation of electricity in conductors. Physical review, 32(1): 97.

Kappel, D.; Habenschuss, S.; Legenstein, R.; and Maass, W. 2015. Synaptic sampling: A Bayesian approach to neural network plasticity and rewiring. Advances in neural information processing systems, 28.

Knill, D. C.; and Pouget, A. 2004. The Bayesian brain: the role of uncertainty in neural coding and computation. TRENDS in Neurosciences, 27(12): 712–719.

Knill, D. C.; and Richards, W. 1996. Perception as Bayesian inference. Cambridge University Press.

Lin, Y.; Gao, B.; Tang, J.; Zhang, Q.; Qian, H.; and Wu, H. 2025. Deep Bayesian active learning using in-memory computing hardware. Nature Computational Science, 5(1): 27–36.

Narayanan, N. S.; Kimchi, E. Y.; and Laubach, M. 2005. Redundancy and synergy of neuronal ensembles in motor cortex. Journal ofNeuroscience, 25(17): 4207–4216.

Nyquist, H. 1928. Thermal agitation of electric charge in conductors. Physical review, 32(1): 110.

Oliviers, G.; Bogacz, R.; and Meulemans, A. 2024. Learning probability distributions of sensory inputs with Monte Carlo predictive coding. PLoS Computational Biology, 20(10): e1012532.

Paunov, A.; L’Hotellier, M.; Guo, D.; He, Z.; Yu, A.; andˆ Meyniel, F. 2024. Multiple and subject-specific roles of uncertainty in reward-guided decision-making.

Rao, R. P. N.; and Ballard, D. H. 1999. Predictive coding in the visual cortex: A functional interpretation of some extraclassical receptive-field effects. Nature Neuroscience, 2(1): 79–87.

Salvatori, T.; Mali, A.; Buckley, C. L.; Lukasiewicz, T.; Rao, R. P.; Friston, K.; and Ororbia, A. 2023. Brain-inspired computational intelligence via predictive coding. arXiv preprint arXiv:2308.07870, 13.

Salvatori, T.; Pinchetti, L.; Millidge, B.; Song, Y.; Bao, T.; Bogacz, R.; and Lukasiewicz, T. 2022. Learning on arbitrary graph topologies via predictive coding. Advances in neural information processing systems, 35: 38232–38244.

Schottky, W. 1918. Uber spontane Stromschwankungen<sup>¨</sup> in verschiedenen Elektrizitatsleitern.¨ Annalen der physik, 362(23): 541–567.

Sennesh, E.; Wu, H.; and Salvatori, T. 2024. Divide-and-Conquer Predictive Coding: A structured Bayesian inference algorithm. arXiv preprint arXiv:2408.05834.

Tschantz, A.; Koudahl, M.; Linander, H.; Da Costa, L.; Heins, C.; Beck, J.; and Buckley, C. 2025. Bayesian Predictive Coding. arXiv preprint arXiv:2503.24016.

Welling, M.; and Teh, Y. W. 2011. Bayesian learning via stochastic gradient Langevin dynamics. In Proceedings of the 28th international conference on machine learning (ICML-11), 681–688.

White, J. A.; Rubinstein, J. T.; and Kay, A. R. 2000. Channel noise in neurons. Trends in neurosciences, 23(3): 131–137.

Yuste, R.; Cossart, R.; and Yaksi, E. 2024. Neuronal ensembles: Building blocks of neural circuits. Neuron, 112(6): 875–892.