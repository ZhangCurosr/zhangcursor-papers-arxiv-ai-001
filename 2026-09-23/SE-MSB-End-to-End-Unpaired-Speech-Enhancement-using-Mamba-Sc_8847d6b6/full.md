# SE-MSB: End-to-End Unpaired Speech Enhancement using Mamba Schrödinger Bridges

Andreas Bagge<sup>∗</sup>, Andreas Nymand<sup>∗</sup>, Michael Riis Andersen<sup>∗</sup> and Bjørn Sand Jensen

<sup>∗</sup>DTU Compute, Kgs. Lyngby, Denmark

<sup></sup> WSA, Lynge, Denmark

{andreas.bagge,andreas.nymand}@wsa.com , {miri, bjje}@dtu.dk

## Abstract

Speech enhancement (SE) models typically rely on supervised learning with paired data examples where clean speech is synthetically degraded. This paradigm limits performance in real-world scenarios where the target environment’s specific acoustic characteristics are unknown. We propose a fully unpaired SE framework that uses principled Diffusion Schrödinger Bridges (DSB) to learn a stochastic transport process between a clean and a degraded speech distribution. Algorithms for learning transport maps are computationally heavy since they require simulating differential equations during training, usually at each training step. Therefore, we propose using a high-efficiency Mamba Diffusion Model designed for end-to-end waveform processing. We compare against state-of-the-art methods for speech enhancement, both paired and unpaired, as well as a classical signal processing algorithm. Experimental results show that we are on par or better than the baselines while being orders of magnitude faster during inference. Furthermore, we show that the flexibility of the DSB formulation allows our model to generalize across SE tasks, offering a robust and efficient solution for real-world speech restoration.

## 1 Introduction

Speech enhancement (SE) problems, such as denoising, declipping, or dereverberation, are classic problems in audio processing. Traditionally, these are solved using supervised methods trained on collections of paired instances of clean speech $x _ { 0 }$ and degraded speech $x _ { 1 }$ , where the degraded speech is typically simulated by modifying the clean speech, e.g., by adding noise or clipping the amplitude such that $x _ { 1 } \sim p _ { \mathrm { n o i s y } } ( \cdot | x _ { 0 } )$ . Such approaches are limited to training on simulated distortions and cannot be applied to actual, in-the-wild degraded audio because large-scale paired data collection is infeasible or, in some instances, even impossible. Enabling model training on unpaired in-the-wild audio, i.e., where clean and degraded speech are drawn from separate distributions $x _ { 0 } \sim p _ { \mathrm { c l e a n } } , x _ { 1 } \sim p _ { \mathrm { n o i s y } }$ independently rather than paired samples $( x _ { 0 } , x _ { 1 } ) \sim p _ { \mathrm { j o i n t } } .$ , has the potential to improve SE models in real-world conditions by adapting the model to out-of-distribution (OOD) data, for example, by personalizing the model to the user’s specific acoustic environment. With the recent rise in personalized AI, especially through federated learning [1], the ability to train models directly on observed, but unpaired or unlabeled, data is becoming increasingly important for adapting models to the specific needs of end-users. This is especially true for SE models, where the end-user’s acoustic environment is often unknown and can be highly variable and unpredictable. In some instances, the degradation is not only unknown but also difficult to simulate to be used in a paired setup, e.g., Lombard effect [2] or reconstructing historical records [3]. Diffusion models constitute the state-of-the-art for image and audio synthesis [4, 5], and they have also shown great results for paired SE [6]. Diffusion models generally consist of two processes, a forward and a backward process. In traditional diffusion models, the forward process gradually turns data samples into Gaussian noise, and the objective of the model is to learn the backward process by effectively removing noise from the input. After learning the reverse process, the models can then synthesize high-quality samples from Gaussian noise. Though traditional diffusion methods yield high-quality samples, they do not allow for mapping between two arbitrary data distributions. In contrast, Schrödinger Bridges (SB) [7] allow for high-quality sample generation and direct transfer between two arbitrary data distributions through the entropy-regularized optimal transport problem, also called the SB problem, of finding the most likely random evolution between two continuous probability distributions. That is, SBs allow learning stochastic maps between clean and degraded audio from unpaired data only, enabling training on in-the-wild data. SBs are formulated in terms of stochastic differential equations (SDEs), which can be simulated with SDE solvers, and solving the SB problem therefore allows for a diffusion-based generative approach. Mamba is a selective state-space model that serves as an efficient alternative to the popular transformer architecture for long sequence modelling [8]. Mamba offers linear scaling in sequence length and contains a constant state size during inference. We propose the SE-MSB (Speech Enhancement Mamba Schrödinger Bridges) model, which is a novel combination of Diffusion Schrödinger Bridges (DSB) and modern state-space models for end-to-end SE using completely unpaired data. We provide a fast and efficient implementation of the DSB method for speech enhancement on raw audio waveforms using a bespoke Mamba Diffusion model. Raw waveforms suffer from a higher dimensionality, which is a problem solved by Mambas linear sequence length scaling. We show that the combination of DSB and Mamba outperforms or is on par with other state-of-the-art methods for unpaired SE, while being orders of magnitude faster during inference, both due to the efficiency of the network architecture and due to the ability of the model to do few-step diffusion without sacrificing performance. Audio examples and code are available at anonymous.4open.science/r/Latent-DSB. Our main contributions are:

![](images/3dd3bcf512e3ce8508768cebc7230dde33e99b4947578992de47c1c2ec8c275f.jpg)  
Figure 1: In traditional supervised speech enhancement (left), we assume access to a clean sample x<sub>0</sub> from the clean target distribution and a corresponding degraded sample from the degraded distribution to learn the optimal pairing (blue arrows). In contrast, our method (right) requires only independent samples from both distributions to learn the optimal pairing.

• We propose the SE-MSB model, which is the first fully unpaired and end-to-end model for speech enhancement on raw waveforms.

• We evaluate SE-MSB using a rigorous experimental protocol and compare it against several state-of-the-art baselines on multiple tasks and compounded conditions.

• We demonstrate that SE-MSB achieves strong speech enhancement performance, outperforming paired methods in some cases while exhibiting highly competitive efficiency.

## 2 Related Work

A<sup>2</sup>ASB (Audio-to-Audio Schrödinger Bridge) [9] is a tractable, simulation-free method for learning maps between the distributions $p _ { d a t a } ( \mathbf { X } _ { 0 } )$ and $p _ { p r i o r } ( \mathbf { X } _ { 1 } | \mathbf { X } _ { 0 } )$ , i.e., a conditional probability distribution that requires paired samples. A<sup>2</sup>ASB exhibits state-of-the-art results on paired audio restoration tasks such as bandwidth extension and inpainting. CycleGAN is a discriminative generative model for learning a deterministic mapping between distributions [10]. CycleGANs have also been used for style transfer in the context of audio, for example, for whisper-to-normal speech conversion [11]. However, CycleGANs are highly unstable during training and require jointly optimizing 4 distinct neural networks.

[12] proposes using Gaussian Flow Bridges for unpaired speech enhancement, a diffusion-based approach where samples are first mapped to an intermediate Gaussian distribution. The approach suffers from the fact that the simulated trajectories first have to map to a third distribution, a Gaussian, therefore yielding non-optimal transport. A similar class of methods for unpaired speech enhancement uses

<table><tr><td></td><td>Unpaired</td><td>End-to-End</td><td>Task-flexible</td></tr><tr><td>SE-MSB</td><td></td><td>V</td><td>L</td></tr><tr><td>GFB</td><td></td><td>x*</td><td>L</td></tr><tr><td>BUDDy</td><td></td><td>x</td><td>x</td></tr><tr><td>A2ASB</td><td>x</td><td>x</td><td>L</td></tr></table>

Table 1: Comparison of current paired/unpaired diffusion models for speech enhancement. <sup>∗</sup>GFB processes raw audio but employs an internal STFT encoder and ISTFT decoder.

diffusion posterior sampling to solve the audio inverse problem in an unsupervised manner [13, 14, 15]. Instead of training a task-specific model, these methods utilize a pretrained diffusion model as a prior over the clean data. During inference, the models then modify the reverse diffusion process to guide the sampling trajectory. However, these methods also require access to the mathematical structure of the degradation process, which, in most cases, is not available for in-the-wild data. Diffusion Schrödinger Bridge Matching [16, 17] is an algorithm for computing the Schrödinger Bridge, a dynamic entropy-regularised version of optimal transport. The algorithm can be used for learning a diffusion model for mapping between arbitrary distributions using unpaired data samples. Mamba, a selective state-space model architecture, has shown great results in the field of paired speech enhancement [18] [19]. In combination with diffusion Schrödinger bridges, Mamba has shown to outperform Long Short-Term Memory and Multi-Head Self-Attention backbones for Short Time Fourier Transform (STFT) based audio representations for paired speech enhancement. However, STFT-based methods are limited by fixed transformation hyperparameters, reducing the model’s ability to capture the expressivity of raw audio.

## 3 Method

Our work builds upon the Diffusion Schrödinger Bridge Matching algorithm [16, 17], DSB for short. Consider two distinct continuous probability distributions on $\mathbb { R } ^ { n }$ . In the literature, these distributions are usually denoted as $p _ { d a t a }$ and $p _ { p r i o r } .$ . However, for our application of speech enhancement, we will denote these distributions as $p _ { c l e a n }$ and $p _ { \mathit { n o i s y } } ,$ respectively, to emphasize their role as the target and source distributions in the speech enhancement problem. Therefore, $p _ { c l e a n }$ refers to the distribution of clean speech, while $p _ { n o i s y }$ refers to the distribution of degraded speech, for example, noisy, reverberant, or clipped speech. We will denote samples from $p _ { c l e a n }$ and $p _ { n o i s y }$ as $\mathbf { X } _ { 0 }$ and $\mathbf { X } _ { 1 }$ respectively, i.e. ${ \bf X } _ { 0 } \sim p _ { c l e a n }$ and ${ \bf X } _ { 1 } \sim p _ { n o i s y } ,$ , where $\dot { \mathbf { X } _ { 0 } } , \mathbf { X } _ { 1 } \in \dot { \mathbb { R } } ^ { n }$ . We seek a stochastic process, or a ’bridge’, that transports samples from $p _ { c l e a n }$ to $p _ { n o i s y }$ and vice versa. We will parameterize the processes as two distinct SDEs, one for each direction, such that simulating the SDEs with a sample as the initial value will transport said sample from one domain to the other. We will denote the forward and backward SDEs as:

$$
\begin{array} { r } { \begin{array} { r } { \mathrm { d } \mathbf { X } _ { t } = f ( \mathbf { X } _ { t } , t ) \mathrm { d } t + g ( t ) \mathrm { d } \mathbf { W } _ { t } , \quad \mathbf { X } _ { 0 } \sim p _ { c l e a n } , } \\ { \mathrm { d } \mathbf { X } _ { t } = b ( \mathbf { X } _ { t } , t ) \mathrm { d } t + g ( t ) \mathrm { d } \bar { \mathbf { W } } _ { t } , \quad \mathbf { X } _ { 1 } \sim p _ { n o i s y } , } \end{array} } \end{array}
$$

where $f$ and b are the drift terms, $g ( t )$ is the diffusion term, and W and $\bar { \bf W }$ are standard Wiener processes. For simplicity, we will assume $g ( t ) = \beta$ is constant. Therefore, the problem reduces to learning the drift terms for each process, $\mathrm { i . e . , } f$ and $b ,$ such that simulating the backward process from $t = 1$ to $t = 0$ starting from ${ \bf X } _ { 1 } \sim p _ { n o i s y }$ yields ${ \bf X } _ { 0 } \sim p _ { c l e a n }$ at time $t = 0$ (and vice versa for the forward process). We will parameterize the forward and backward drift with a single neural network $v ( \mathbf { X } _ { t } , t , s )$ with parameters θ such that:

$$
f ( \mathbf { X } _ { t } , t ) \approx v _ { \theta } ( \mathbf { X } _ { t } , t , 1 ) , ~ b ( \mathbf { X } _ { t } , t ) \approx v _ { \theta } ( \mathbf { X } _ { t } , t , 0 ) ,
$$

where $s \in \{ 0 , 1 \}$ is a binary indicator variable, indicating whether the forward or backward drift is being approximated. Alternatively, one could have parameterized each drift with a separate neural network. Having obtained the optimal parameters $\theta ^ { * }$ , we can approximately simulate the forward and backward SDEs on the discrete time interval $0 = t _ { 0 } < t _ { 1 } \cdot \cdot \cdot < t _ { N } = 1$ , where $\Delta t _ { k + 1 } = t _ { k + 1 } - t _ { k }$ using the Euler-Maruyama method:

$$
p ( \mathbf { X } _ { t _ { k + 1 } } | \mathbf { X } _ { t _ { k } } ) \approx p ( \mathbf { X } _ { t _ { k + 1 } } | \mathbf { X } _ { t _ { k } } , \mathbf { X } _ { 1 } ) = { \mathcal { N } } ( \mathbf { X } _ { t _ { k + 1 } } ; { \tilde { \mu } } _ { k } , { \tilde { \sigma } } _ { k } ^ { 2 } \mathbf { I } ) , \quad { \mathrm { ( f o r w a r d ~ p r o c e s s ) } }\tag{1}
$$

$$
p ( \mathbf { X } _ { t _ { k } } | \mathbf { X } _ { t _ { k + 1 } } ) \approx p ( \mathbf { X } _ { t _ { k } } | \mathbf { X } _ { t _ { k + 1 } } , \mathbf { X } _ { 0 } ) = \mathcal { N } ( \mathbf { X } _ { t _ { k } } ; \mu _ { k + 1 } , \sigma _ { k + 1 } ^ { 2 } \mathbf { I } )\tag{2}
$$

That is, the forward and backward processes can be simulated by sampling from the respective Gaussian distributions, where the means and variances are given by:

$$
\begin{array} { r l } & { \tilde { \mu } _ { k } \approx \mathbf { X } _ { t _ { k } } + \Delta t _ { k + 1 } \underbrace { \left( \mathbf { X } _ { 1 } - \mathbf { X } _ { t _ { k } } \right) / ( 1 - t _ { k } ) } _ { = f ( \mathbf { X } _ { t _ { k } } , t _ { k } ) } \approx \mathbf { X } _ { t _ { k } } + \Delta t _ { k + 1 } v _ { \theta } ( \mathbf { X } _ { t _ { k } } , t _ { k } , 1 ) , } \\ & { \mu _ { k + 1 } \approx \mathbf { X } _ { t _ { k + 1 } } + \Delta t _ { k + 1 } \underbrace { \left( \mathbf { X } _ { 0 } - \mathbf { X } _ { t _ { k + 1 } } \right) / t _ { k + 1 } } _ { = b ( \mathbf { X } _ { t _ { k + 1 } } , t _ { k + 1 } ) } \approx \mathbf { X } _ { t _ { k + 1 } } + \Delta t _ { k + 1 } v _ { \theta } ( \mathbf { X } _ { t _ { k + 1 } } , t _ { k + 1 } , 0 ) , } \end{array}
$$

$$
\tilde { \sigma } _ { k } ^ { 2 } = \frac { \beta \Delta t _ { k + 1 } ( 1 - t _ { k + 1 } ) } { 1 - t _ { k } } , ~ \sigma _ { k + 1 } ^ { 2 } = \frac { \beta \Delta t _ { k + 1 } t _ { k } } { t _ { k + 1 } } .
$$

Additionally, it can be shown [16] that given $\mathbf { X } _ { 0 }$ and $\mathbf { X } _ { 1 }$ , then intermediate points in the bridge can be sampled from the conditional distribution:

$$
p ( \mathbf { X } _ { t _ { k } } | \mathbf { X } _ { 0 } , \mathbf { X } _ { 1 } ) = \mathcal { N } ( ( 1 - t _ { k } ) \mathbf { X } _ { 0 } + t _ { k } \mathbf { X } _ { 1 } , \beta t _ { k } ( 1 - t _ { k } ) \mathbf { I } ) .\tag{3}
$$

To train the neural network $v _ { \theta }$ , we will use the DSB Matching algorithm [16, 17]. The algorithm contains two phases: pre-training and fine-tuning. During the pre-training phase, i.i.d. samples from $p _ { c l e a n }$ and $p _ { n o i s y }$ are sampled independently. We will call this pair a "random coupling" (i.e., unpaired). An intermediate point, $\mathbf { X } _ { t _ { k } }$ , is sampled from the conditional distribution $( 3 )$ , and the objective of the network is to predict the forward and backward drift at time $t _ { k }$ and $t _ { k + 1 }$ given by $( \mathbf { X } _ { 1 } - \mathbf { X } _ { t _ { k } } ) / ( 1 - t _ { k } )$ and $( \mathbf { X } _ { 0 } - \mathbf { X } _ { t _ { k + 1 } } ) / t _ { k + 1 }$ , respectively. After a number of pre-training steps, the fine-tuning phase begins. Here, the initial samples $\mathbf { X } _ { 0 }$ and $\mathbf { X } _ { 1 }$ are once again obtained by sampling from $p _ { c l e a n }$ and $p _ { { n o i s y } } .$ , but they are no longer randomly coupled. Instead, the couplings are obtained by simulating the processes on the interval from $t = 0 \mathrm { t o } t = 1$ and from $t = 1 \mathrm { t o } t = 0$ respectively, using the current parameters θ of the model. The objective of the model is the same as in the pre-training phase. [16] shows that applying the pre-training phase followed by the fine-tuning phase will eventually yield the SB, i.e., the optimal stochastic process between $p _ { c l e a n }$ and $p _ { n o i s y } .$ The DSB Matching algorithm can be seen as a stochastic version of the Reflow algorithm [20], which can be deduced from DSB Matching by setting $\beta = 0 .$ , i.e., by removing the noise from the SDEs. For the Reflow algorithm, the authors show that the fine-tuning phase yields "straight flows", i.e. flows that transport samples from $p _ { c l e a n }  t 0 p _ { n o i s y }$ along straight lines. This is a desirable property for diffusion models, as it allows for few-step diffusion sampling without compromising sample quality, therefore requiring less compute during inference. Additionally, it can also be shown that $\mathbf { \dot { A } } ^ { 2 } \mathbf { A S B }$ is simply a paired version of the DSB Matching algorithm that uses only the pre-training phase, and that only trains the backward process, but instead of drawing random couplings, the couplings are drawn according to the joint distribution and are therefore paired, see Appendix C.

## Mamba Diffusion Model architecture

To the best of our knowledge, all existing unpaired speech enhancement models operate on spectrograms [13, 14, 15]. This is done mostly to enable the use of established convolutional neural networks, such as U-Nets [21], on the resulting time-frequency representations. However, such time-frequency representations impose a rigid, non-learnable trade-off between temporal and frequency resolution, and, in the case of magnitude representations, completely disregard the phase, therefore requiring heuristic phase reconstruction algorithms or an additional vocoder for waveform synthesis. Therefore, we propose combining DSB with modern state-space architectures to operate in an end-to-end fashion on raw audio waveforms, enabling the model to learn an internal representation of the raw data, without relying on specific choices of time-frequency representations. To this end, we use a bespoke, efficient Mamba Diffusion model architecture based on Mamba blocks [8]. This architecture uses an initial 1D convolutional layer to learn an internal latent representation of the raw audio waveform, similar to a learnable time-frequency representation. This latent is then processed by a number of Mamba Diffusion blocks. Each Mamba Diffusion block uses Adaptive LayerNorm [22] to inject the diffusion timestep t and the process-direction conditioning c. The model architecture is visualized in Figure 3, and a more detailed description of the architecture can be found in Appendix B. To compare the algorithmic efficiency of the proposed SE-MSB model to the state-of-the-art baselines, we have opted to count tera $( 1 0 ^ { 1 2 } )$ floating-point operations (TFLOPS) required to evaluate a sample output. Each of the models processed a dummy audio input of $2 ^ { 1 5 } = 3 2 7 \bar { 6 } 8$ samples (2.048 seconds of audio at 16kHz). For counting the TFLOPs of the models, PyTorch’s "FlopCounterMode" was used. As our implementation relied on the mamba-ssm library, which uses custom Triton kernels for an efficient implementation, the TFLOP count has been manually registered and is counted as explained in chapter 6 of the Mamba2 paper [8]. The majority of the Mamba2 TFLOP count comes from the 1D convolution and the selective scan algorithms, which, according to the paper, consist of an order $O ( T N ^ { 2 } )$ FLOPs, where T is the sequence length and N is the state dimension. This is under the assumption that $N = P = Q$ where $\dot { P }$ is the head dimension and $Q$ is the chunk size. If we assume a more general form, it can be shown that the FLOP count from the selective scan equals $F L O P _ { s c a n } = 2 B \bar { T } N ( H ~ Q + 3 D )$ , where B is the batch size, H is the number of heads, and D is total head dimension, $D = H P$ , see appendix D. Before the block decomposition, a 1D causal convolution across the inner dimension is performed with a total cost of $F \bar { L O P _ { c o n v } } = 2 B T D K$ where K is the convolution kernel size. In addition to this, some elementwise operations are used in the Triton kernels, and are, as such, invisible to Torch’s "FlopCounterMode". The FLOP count is negligible compared to the selective state scan, but for completeness, we include them in the count. First, the SiLU activation layers take approximately 5BTD FLOPs. Secondly, the gated RMS-Norm takes 11BTD FLOPs, 5 from the RMS norm, 5 for an attached SiLU, and 1 from an elementwise multiplication. Lastly, a final elementwise multiplication takes BTHQ FLOPs. Further information on the FLOP count on non-torch baselines can be seen in Appendix D.

Algorithm 1 Training algorithm   
1: Initialize neural network $v _ { \theta }$ and batch size 2B   
2: Let $i \in \{ 1 , 2 \ldots B \}$   
3: while not converged do   
4: if Pre-training then   
5: $\mathbf { X } _ { 0 } ^ { 1 : B } \overset { \mathrm { i i d } } { \sim } p _ { c l e a n } , \tilde { \mathbf { X } } _ { 1 } ^ { 1 : B } \overset { \mathrm { i i d } } { \sim } p _ { n o i s y }$   
6: $\mathbf { X } _ { 1 } ^ { 1 : B } \overset { \mathrm { i i d } } { \sim } p _ { n o i s y } , \tilde { \mathbf { X } } _ { 0 } ^ { 1 : B } \overset { \mathrm { i i d } } { \sim } p _ { c l e a n }$   
7: else   
8: $\mathbf { X } _ { 0 } ^ { 1 : B } \overset { \mathrm { i i d } } { \sim } p _ { c l e a n } , \tilde { \mathbf { X } } _ { 1 } ^ { 1 : B } \overset { \mathrm { i i d } } { \sim } p _ { n o i s y }$   
9: Sample $\mathbf { X } _ { 1 } ^ { i }$ using (1) and $v _ { \theta } ( \cdot , \cdot , 1 )$ starting from $\mathbf { X } _ { 0 } ^ { i }$   
10: Sample $\tilde { \mathbf { X } } _ { 0 } ^ { i }$ using (2) and $v _ { \theta } ( \cdot , \cdot , 0 )$ starting from $\tilde { \mathbf { X } } _ { 1 } ^ { i }$   
11: end if   
12: $t ^ { 1 : B } , \tilde { t } ^ { 1 : B } \stackrel { \mathrm { i i d } } { \sim } { \mathcal U } ( 0 , 1 )$   
13: $\mathbf { X } _ { t } ^ { i } \sim \mathcal { N } ( ( 1 - t ^ { i } ) \mathbf { X } _ { 0 } ^ { i } + t ^ { i } \mathbf { X } _ { 1 } ^ { i } , \beta t ^ { i } ( 1 - t ^ { i } ) \mathbf { I } )$   
14: $\tilde { \mathbf { X } } _ { t } ^ { i } \sim \mathcal { N } ( ( 1 - \tilde { t } ^ { i } ) \tilde { \mathbf { X } } _ { 0 } ^ { i } + \tilde { t } ^ { i } \tilde { \mathbf { X } } _ { 1 } ^ { i } , \beta \tilde { t } ^ { i } ( 1 - \tilde { t } ^ { i } ) \mathbf { I } )$   
15: $\begin{array} { r } { \mathcal { L } _ { b } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \left. v _ { \theta } ( \mathbf { X } _ { t } ^ { i } , t ^ { i } , 0 ) - \frac { \mathbf { X } _ { 0 } ^ { i } - \mathbf { X } _ { t } ^ { i } } { t ^ { i } } \right. } \end{array}$   
16: $\begin{array} { r } { \mathcal { L } _ { f } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \Big | \Big | v _ { \theta } ( \tilde { \mathbf { X } } _ { t } ^ { i } , \tilde { t } ^ { i } , 1 ) - \frac { \tilde { \mathbf { X } } _ { 1 } ^ { i } - \tilde { \mathbf { X } } _ { t } ^ { i } } { 1 - \tilde { t } ^ { i } } \Big | \Big | } \end{array}$   
17: Take gradient step $\nabla _ { \theta } \frac { 1 } { 2 } ( \mathcal { L } _ { b } + \mathcal { L } _ { f } )$   
18: end while   
1.0   
10   
0.8   
5   
0   
0.4   
-5 é   
0.2   
-10   
10-1 100 101 102 0.0-1 100 101 102   
Compute (TFLOP) Compute (TFLOP)   
SE-MSB (Ours) BUDDy GFB Paired SE-MSB SGMSE UNIVERSE++ Identity WPE

![](images/7e2a31429abdb761a130d94b77834aa31e96f0d449f8853924bfb32bba55734a.jpg)  
Figure 2: Total compute vs SISDRi (left) and FAD (right). The area of the points scales linearly with mean inference time, see table 3 and appendix D. Our model outperforms unpaired state-of-the-art models with the same compute budget. It is also seen that the model is robust to changes in the number of diffusion steps, performing almost the same no matter how many diffusion steps are taken. This is also shown in figure 4. The models with a dashed outline are trained on paired data. An Equivalent plot for the CHiME-6 dataset can be seen in appendix D.

![](images/cec495abc6b275b4051bef2cbe641bf174d9912398ad5d92f7164b8490850572.jpg)  
Figure 3: The Mamba Diffusion Model architecture proposed for SE-MSB. Each waveform is encoded using a 1D convolutional layer. The resulting representation is mixed with the conditional signal and processed by a stack of Mamba Diffusion Blocks.

![](images/10c7236b20708970748f4966a9f55efa0cc90d6fc83ec68d751d8480e6a75322.jpg)  
Figure 4: Metrics as a function of the number of diffusion steps. The SE-MSB model is almost unaffected by the number of steps, showing that SE-MSB can do few-step generation.

## 4 Experiments

The proposed SE-MSB model is trained and evaluated on a dereverberation, denoising, and declipping task. These tasks are traditionally solved with supervised methods, but by artificially degrading the clean speech data, we can evaluate distortion metrics that require paired data. The clean speech data used throughout our experiments comes from VCTK [23], see Appendix A.1. The VCTK dataset is both used as $p _ { c l e a n }$ and to simulate $p _ { n o i s y }$ , but, importantly, the model never sees pairs of clean and degraded audio during training.

<table><tr><td></td><td>Model</td><td></td><td>FAD (↓)</td><td>DNSMOS (↑)</td><td>SISDRi (↑)</td><td></td><td>WER (↓)</td><td>SR-CS (↑)</td><td>TFLOPS (↓)</td></tr><tr><td>U</td><td>SE-MSB (10)</td><td></td><td>0.32</td><td> $2 . 9 0 \pm \ : \ : 0 . 0 1$ </td><td> $\mathbf { 2 . 0 9 \pm 0 . 0 9 }$ </td><td> $0 . 0 5 \pm \ : \ : 0 . 0 0$ </td><td></td><td> $0 . 8 0 \pm \ : \ : 0 . 0 0$ </td><td>2.3937</td></tr><tr><td></td><td>U SE-MSB (50)</td><td>0.31</td><td></td><td> $2 . 9 2 \pm \ : \ : 0 . 0 1$ </td><td> $2 . 0 6 \pm \ : \ : 0 . 0 9$ </td><td> $0 . 0 5 \pm \ : \ : 0 . 0 0$ </td><td></td><td> $0 . 8 1 \pm \ : \ : 0 . 0 0$ </td><td>11.9687</td></tr><tr><td></td><td>U BUDDy (10 †)</td><td>0.86</td><td></td><td> $2 . 3 0 \pm \ : 0 . 0 1$ </td><td> $- 8 . 7 5 \pm 0 . 1 1$ </td><td> $0 . 5 4 \pm \ : \ : 0 . 0 1$ </td><td></td><td> $0 . 4 7 \pm \ : \ : 0 . 0 1$ </td><td>19.2127</td></tr><tr><td></td><td>U BUDDy (100 †)</td><td>0.23</td><td> $\underline { { { \bf 3 . 0 8 \pm 0 . 0 1 } } }$ </td><td></td><td> $1 . 1 5 \pm \ : 0 . 1 2$ </td><td> ${ \bf 0 . 0 2 \pm 0 . 0 0 }$ </td><td></td><td> $\underline { { \mathbf { 0 . 9 1 \pm 0 . 0 0 } } }$ </td><td>192.1273</td></tr><tr><td></td><td>U GFB (10)</td><td>0.54</td><td> $2 . 3 0 \pm \ : 0 . 0 2$ </td><td></td><td> $- 0 . 9 0 \pm \ : \ : 0 . 0 8$ </td><td> $0 . 1 0 \pm \ : \ : 0 . 0 0$ </td><td></td><td> $0 . 7 1 \pm \ : \ : 0 . 0 0$ </td><td>2.4265</td></tr><tr><td></td><td>U GFB (50)</td><td>0.36</td><td> $2 . 7 5 \pm \ : 0 . 0 2$ </td><td></td><td> $- 0 . 4 8 \pm \ : \ : 0 . 0 4$ </td><td> $0 . 0 6 \pm \ : \ : 0 . 0 0$ </td><td></td><td> $0 . 7 9 \pm \ : \ : 0 . 0 0$ </td><td>13.2112</td></tr><tr><td></td><td>U WPE</td><td>0.45</td><td> $2 . 7 1 \pm \ : 0 . 0 2$ </td><td></td><td> $1 . 2 4 \pm \ : \ : 0 . 0 7$ </td><td> $\mathbf { 0 . 0 2 \pm 0 . 0 0 }$ </td><td></td><td> $0 . 8 4 \pm \ : \ : 0 . 0 0$ </td><td>0.0012</td></tr><tr><td>VCTK</td><td>P SGMSE (10)</td><td>0.25</td><td> $2 . 8 9 \pm \ : \ : 0 . 0 2$ </td><td></td><td> $2 . 2 7 \pm \ : 0 . 0 9$ </td><td> $0 . 0 2 \pm 0 . 0 0$ </td><td></td><td> $0 . 9 0 \pm \ : \ : 0 . 0 0$ </td><td>13.3338</td></tr><tr><td></td><td>P SGMSE (50)</td><td>0.16</td><td> $2 . 9 2 \pm \ : \ : 0 . 0 2$ </td><td></td><td> $2 . 0 1 \pm \ : \ : 0 . 1 1$ </td><td> $\underline { { 0 . 0 2 \pm 0 . 0 1 } }$ </td><td></td><td> $\underline { { 0 . 9 1 \pm 0 . 0 0 } }$ </td><td>66.6690</td></tr><tr><td></td><td>P Paired SE-MSB (10) *</td><td>0.27</td><td> $2 . 9 6 \pm \ : \ : 0 . 0 1$ </td><td></td><td> ${ \underline { { 7 . 1 5 \pm 0 . 0 8 } } }$ </td><td> $0 . 0 4 \pm \ : \ : 0 . 0 0$ </td><td></td><td> $0 . 8 7 \pm \ : \ : 0 . 0 0$ </td><td>2.3937</td></tr><tr><td></td><td>P Paired SE-MSB (50) *</td><td>0.25</td><td> $2 . 9 6 \pm \ : \ : 0 . 0 1$ </td><td></td><td> $6 . 9 5 \pm \ : \ : 0 . 0 8$ </td><td> $0 . 0 4 \pm \ : \ : 0 . 0 0$ </td><td></td><td> $0 . 8 7 \pm \ : \ : 0 . 0 0$ </td><td>11.9687</td></tr><tr><td></td><td>P UNIVERSE++</td><td>0.27</td><td> $3 . 0 5 \pm \ : \ : 0 . 0 1$ </td><td></td><td> $2 . 3 5 \pm \ : 0 . 0 9$ </td><td> $0 . 0 3 \pm \ : \ : 0 . 0 0$ </td><td></td><td> $0 . 9 0 \pm \ : \ : 0 . 0 0$ </td><td>11.9687</td></tr><tr><td></td><td>Identity</td><td>0.53</td><td> $2 . 6 4 \pm \ : \ : 0 . 0 2$ </td><td></td><td> $0 . 0 0 \pm \ : \ : 0 . 0 0$ </td><td> $\underline { { 0 . 0 2 \pm 0 . 0 0 } }$ </td><td></td><td> $0 . 7 9 \pm \ : \ : 0 . 0 0$ </td><td>0.0000</td></tr><tr><td></td><td>U SE-MSB (10)</td><td>0.15</td><td> $2 . 9 0 \pm \ : \ : 0 . 0 2$ </td><td></td><td> $0 . 6 0 \pm \ : \ : 0 . 0 9$ </td><td> $0 . 1 1 \pm \ : \ : 0 . 0 2$ </td><td></td><td> $0 . 6 7 \pm \ : \ : 0 . 0 0$ </td><td>2.3937</td></tr><tr><td></td><td>U SE-MSB (50)</td><td>0.16</td><td> $2 . 9 2 \pm \ : \ : 0 . 0 2$ </td><td></td><td> $0 . 5 2 \pm \ : \ : 0 . 0 9$ </td><td> $0 . 1 3 \pm \ : \ : 0 . 0 1$ </td><td></td><td> $0 . 6 6 \pm \ : \ : 0 . 0 0$ </td><td>11.9687</td></tr><tr><td></td><td>U BUDDy (10 †)</td><td>0.69</td><td> $2 . 2 1 \pm \ : 0 . 0 2$ </td><td></td><td> $- 9 . 9 5 \pm . 0 . 1 5$ </td><td> $0 . 7 6 \pm \ : \ : 0 . 0 2$ </td><td></td><td> $0 . 3 8 \pm \ : \ : 0 . 0 1$ </td><td>19.2127</td></tr><tr><td></td><td>U BUDDy (100 †)</td><td>0.17</td><td> $\mathbf { 3 . 1 8 \pm 0 . 0 1 }$ </td><td></td><td> $- 1 . 7 6 \pm \ : \ : 0 . 1 8$ </td><td> $0 . 0 4 \pm \ : \ : 0 . 0 0$ </td><td></td><td> $0 . 8 0 \pm \ : \ : 0 . 0 0$ </td><td>192.1273</td></tr><tr><td></td><td>U GFB (10)</td><td>0.31</td><td> $2 . 5 9 \pm \ : 0 . 0 2$ </td><td></td><td> $- 1 . 1 4 \pm \ : \ : 0 . 0 7$ </td><td> $0 . 1 6 \pm \ : \ : 0 . 0 1$ </td><td></td><td> $0 . 5 9 \pm \ : \ : 0 . 0 1$ </td><td>2.4265</td></tr><tr><td></td><td>U GFB (50)</td><td>0.29</td><td> $2 . 9 0 \pm \ : \ : 0 . 0 2$ </td><td></td><td> $- 0 . 3 3 \pm \ : 0 . 0 4$ </td><td> $0 . 0 8 \pm \ : \ : 0 . 0 0$ </td><td></td><td> $0 . 7 1 \pm \ : \ : 0 . 0 1$ </td><td>13.2112</td></tr><tr><td></td><td>U WPE</td><td>0.22</td><td> $2 . 7 1 \pm \ : 0 . 0 3$ </td><td></td><td> ${ \bf 1 . 4 0 \pm 0 . 0 7 }$ </td><td> $\mathbf { 0 . 0 2 \pm 0 . 0 0 }$ </td><td></td><td> $\mathbf { 0 . 8 9 \mathop { \pm } 0 . 0 0 }$ </td><td>0.0012</td></tr><tr><td>ibri</td><td>P SGMSE (10)</td><td>0.25</td><td> $2 . 8 9 \pm \ : \ : 0 . 0 2$ </td><td></td><td> $2 . 2 7 \pm \ : 0 . 0 9$ </td><td> $\underline { { 0 . 0 2 \pm 0 . 0 0 } }$ </td><td></td><td> $0 . 9 0 \pm \ : \ : 0 . 0 0$ </td><td>13.3338</td></tr><tr><td></td><td>P SGMSE (50)</td><td>0.16</td><td> $2 . 9 2 \pm \ : \ : 0 . 0 2$ </td><td></td><td> $2 . 0 1 \pm \ : \ : 0 . 1 1$ </td><td> $\underline { { 0 . 0 2 \pm 0 . 0 1 } }$ </td><td></td><td> $\underline { { 0 . 9 1 \pm 0 . 0 0 } }$ </td><td>66.6690</td></tr><tr><td></td><td>P Paired SE-MSB (10) *</td><td>0.20</td><td> $3 . 0 1 \pm \ : \ : 0 . 0 2$ </td><td></td><td> $\underline { { 2 . 7 5 \pm 0 . 1 2 } }$ </td><td> $0 . 0 8 \pm \ : \ : 0 . 0 0$ </td><td></td><td> $0 . 6 7 \pm \ : \ : 0 . 0 1$ </td><td>2.3937</td></tr><tr><td></td><td>P Paired SE-MSB (50) *</td><td>0.20</td><td> $3 . 0 3 \pm \ : \ : 0 . 0 2$ </td><td></td><td> $2 . 7 2 \pm \ : 0 . 1 2$ </td><td> $0 . 0 8 \pm \ : \ : 0 . 0 0$ </td><td></td><td> $0 . 6 7 \pm \ : \ : 0 . 0 1$ </td><td>11.9687</td></tr><tr><td></td><td>P UNIVERSE++</td><td>0.22</td><td> $2 . 9 6 \pm \ : \ : 0 . 0 2$ </td><td></td><td> $- 1 . 0 8 \pm \ : \ : 0 . 1 7$ </td><td> $0 . 1 8 \pm \ : \ : 0 . 0 2$ </td><td></td><td> $0 . 6 6 \pm \ : \ : 0 . 0 1$ </td><td>11.9687</td></tr><tr><td></td><td>Identity</td><td>0.28</td><td></td><td> $2 . 5 7 \pm \ : 0 . 0 3$ </td><td> $0 . 0 0 \pm \ : \ : 0 . 0 0$ </td><td> $0 . 0 3 \pm \ : \ : 0 . 0 0$ </td><td></td><td> $0 . 8 2 \pm \ : \ : 0 . 0 0$ </td><td>0.0000</td></tr></table>

Table 2: Evaluation metrics for the dereverberation task with 95% confidence intervals. The Identity baseline corresponds to unprocessed audio. The number of diffusion steps is denoted in parentheses when relevant, except for †, which denotes optimization steps. U = unpaired, P = paired. Best unpaired model in each metric is bold, and best model across paired and unpaired models is underlined.

The models are tested on the test split of VCTK using an unseen test set of room impulse responses. Standard supervised speech enhancement models are conventionally evaluated using isolated test sets to assess generalizability to OOD shifts. In contrast, the primary advantage of unpaired speech enhancement is the ability for in-domain training, allowing the model to train directly on the observed target distribution. Therefore, to verify that our model retains generalizability, we also evaluate all models on a separate, independent clean speech dataset, namely the Libri ASR Corpus [24].

• Dereverberation: The degraded speech is generated by applying room impulse responses (RIRs) from various public RIR datasets [25, 26, 27, 28, 29] to the clean speech.

• Denoising and declipping: The degraded speech is generated via additive noise and, in a separate experiment, clipping the resulting mixture. The noise dataset is WHAM! noise [30] mixed at 5 dB SNR. Clipping is randomly applied to remove between 0 and 1 dB of the signal power.

For each task, we train our SE-MSB model using the Mamba Diffusion Model architecture on raw audio waveforms using β = 0.1 following a parameter sweep. The model is trained for 250k steps during pre-training and 100k steps during fine-tuning for a total of 350k steps. We use a batch size of 18 on audio samples of 1.00 seconds at a sample rate of 16 kHz. During the fine-tuning phase, each SDE is simulated using 20 steps of the Euler-Maruyama method. For dereverberation, we compare the performance of the SE-MSB model against both unpaired and paired baselines. The unpaired baselines include a Gaussian Flow Bridge (GFB) method [12], a diffusion posterior sampling method called BUDDy [14], and the Weighted Prediction Error (WPE) method[31]. For the paired baselines, we evaluate against a Score-based Generative Model for Speech Enhancement (SGMSE)[6], and the supervised version of the SE-MSB method called $\mathbf { A } ^ { 2 } \mathbf { S } \mathbf { B } [ 9 ]$ . We also include an Identity baseline corresponding to unprocessed reverberant audio. Baselines are described in detail in Appendix C.

To showcase the flexibility of our method, we train an SE-MSB model on a mixture of multiple degradations: reverberation, additive noise, and clipping. We compare the performance of the

<table><tr><td>Model</td><td></td><td>TFLOPS</td><td>Inference Time (ms)</td><td>Trainable Params (×106)</td></tr><tr><td>U</td><td>SE-MSB (10)</td><td>2.3937</td><td> $2 1 4 . 1 9 \pm \ : 0 . 5 3$ </td><td>45.56</td></tr><tr><td>U</td><td>SE-MSB (50)</td><td>11.9687</td><td> $1 0 4 2 . 2 9 \pm 6 . 5 9$ </td><td>45.56</td></tr><tr><td>U</td><td>BUDDy (10 †)</td><td>19.2127</td><td> $2 4 8 1 . 4 5 \pm \ : 8 . 0 2$ </td><td>27.74</td></tr><tr><td>U</td><td>BUDDy (100 †)</td><td>192.1273</td><td> $2 2 1 3 5 . 4 0 \pm 1 7 . 1 0$ </td><td>27.74</td></tr><tr><td>U</td><td>GFB (10)</td><td>2.4265</td><td> $3 1 4 . 6 7 \pm \ : \ : 0 . 4 4$ </td><td>43.85</td></tr><tr><td>U</td><td>GFB (50)</td><td>13.2112</td><td> $1 7 1 3 . 9 9 \pm \ : 3 . 9 9$ </td><td>43.85</td></tr><tr><td>U</td><td>WPE</td><td>0.0012</td><td> $8 0 . 7 2 \pm \ : \ : 0 . 2 5$ </td><td>0.00</td></tr><tr><td>P</td><td>Sepformer</td><td>0.4722</td><td> $5 5 . 2 1 \pm \ : 0 . 1 8$ </td><td>25.61</td></tr><tr><td>P</td><td>SGMSE (10)</td><td>13.3338</td><td> $1 2 5 3 . 2 8 \pm \ : 1 . 8 5$ </td><td>65.59</td></tr><tr><td>P</td><td>SGMSE (50)</td><td>66.6690</td><td> $6 2 7 6 . 2 5 \pm 5 . 7 0$ </td><td>65.59</td></tr><tr><td>P</td><td>Paired SE-MSB (10)</td><td>2.3937</td><td> $2 1 5 . 0 0 \pm 0 . 3 9$ </td><td>45.56</td></tr><tr><td>P</td><td>Paired SE-MSB (50)</td><td>11.9687</td><td> $1 0 3 1 . 5 6 \pm 0 . 7 3$ </td><td>45.56</td></tr><tr><td>P</td><td>UNIVERSE++</td><td>0.1446</td><td> $1 2 4 . 9 5 \pm 2 . 3 6$ </td><td>84.24</td></tr></table>

Table 3: Computational efficiency: FLOPS, inference time, and number of trainable parameters.

SE-MSB model against BUDDy, and the unprocessed baseline. BUDDy uses a diffusion posterior sampling method,but BUDDy also assumes a specific degradation, namely a convolution with some unknown RIR, and is therefore sensitive to these assumptions being violated. For the tasks involving additive noise, we also compare against a paired baseline, namely the Sepformer [32], a state-of-the-art supervised speech enhancement model.

We test the few-step sampling capabilities of both the paired and unpaired SE-MSB model compared to the unpaired diffusion baseline GFB on the dereverberation task by evaluating metrics for different numbers of sampling steps. The number of sampling steps can be changed at inference time, and therefore, only a single trained model is required for each task. Few-step sampling is a desirable property as it can reduce the inference time of the model, and therefore increase the practicality of the method for real-world applications. Lastly, we also measure the computational efficiency of all models using FLOPS, inference time (wall clock time), and the number of trainable parameters. The generated samples are evaluated using WER (word error rate), pMOS/DNSMOS (predicted mean opinion score using the DNSMOS model) [33], SR-CS (speaker recognition cosine similarity), scale invariant signal-to-distortion ratio (SISDRi) [34], and FAD (Fréchet Audio Distance) [35]. WER, SR-CS, and SISDRi can be interpreted as content fidelity measures capturing how well content is preserved, while pMOS and FAD can be interpreted as perceptual quality metrics. WER is calculated using the small Whisper model [36] to obtain ground-truth transcriptions of the reference (clean) speech. The transcription model is robust towards distorted speech and is therefore not necessarily a good measure of how well the audio is reconstructed. Nevertheless, WER serves as a useful indicator of whether the semantic content of the speech is preserved. SR-CS is calculated using a speaker embedding model [37] to obtain speaker embeddings of the ground truth signal and the generated signal. The metric is then calculated as the cosine similarity between these embeddings. The samples used in the FAD metric are encodings produced using a CLAP model [38].

## 5 Results

Results of the dereverberation task are shown in Table 2, results of the mixed degradation task are shown in Table 4, and the results of the few-step sampling experiment are shown in Figure 4. Computational efficiency is shown in Table 3. The dereverberation and computational efficiency results show that while SE-MSB is moderately outperformed by BUDDy on most metrics, SE-MSB is around two orders of magnitude more efficient than BUDDy in terms of FLOPS. This large computation difference is made possible thanks to the linear time sequence scaling of the Mamba diffusion model. Additionally, we see that while all unpaired methods are outperformed by the paired methods (SGMSE and Paired SE-MSB) on most metrics, the difference is negligible for some metrics, showing that SE-MSB can achieve performance comparable to paired methods.

The results of the mixed degradation task show that SE-MSB outperforms BUDDy by a large margin when the degradation is no longer only a convolution with a room impulse response, but also includes additive noise and clipping. SE-MMSB achieves comparable performance to the supervised Sepformer model on the mixed degradation task. Finally, the few-step sampling results show that the performance of SE-MSB is almost unaffected by the number of sampling steps, while the performance of GFB degrades significantly when using fewer sampling steps. Interestingly, WER and, to a lesser extent, SISDRi are negatively affected when using more sampling SE-MSB, possibly hinting at a perception-distortion trade-off [39]. Because SE-MSB maintains high performance at lower step counts, it further reduces the necessary computational budget required for effective speech enhancement.

<table><tr><td></td><td></td><td>Model</td><td>FAD (↓)</td><td>DNSMOS (↑)</td><td>SISDRi (↑)</td><td>WER (↓)</td><td></td><td>SR-CS (↑)</td></tr><tr><td>Reverb + Noise</td><td>U</td><td>SE-MSB (10)</td><td>0.35</td><td> $2 . 7 2 \pm \ : 0 . 0 1$ </td><td> ${ \bf 4 . 9 5 \pm 0 . 1 3 }$ </td><td> $0 . 0 8 \pm \ : \ : 0 . 0 0$ </td><td></td><td> $0 . 7 4 \pm \ : \ : 0 . 0 0$ </td></tr><tr><td></td><td>U</td><td>SE-MSB (50)</td><td>0.33</td><td> $\pm { \bf 0 . 7 4 } \pm { \bf 0 . 0 1 }$ </td><td> $4 . 9 1 \pm \ : \ : 0 . 1 3$ </td><td> $0 . 0 9 \pm \ : \ : 0 . 0 0$ </td><td></td><td> $0 . 7 5 \pm \ : \ : 0 . 0 0$ </td></tr><tr><td></td><td>U</td><td>BUDDy (100 †)</td><td>0.70</td><td> $2 . 6 6 \pm \ : \ : 0 . 0 1$ </td><td> $0 . 7 3 \pm \ : \ : 0 . 1 0$ </td><td> $\underline { { \mathbf { 0 . 0 5 \pm 0 . 0 0 } } }$ </td><td></td><td> $\mathbf { 0 . 8 3 \pm 0 . 0 0 }$ </td></tr><tr><td></td><td>P</td><td>Sepformer</td><td>0.29</td><td> $2 . 9 0 \pm \ : \ : 0 . 0 1$ </td><td> $5 . 0 0 \pm \ : \ : 0 . 0 8$ </td><td> $\underline { { 0 . 0 5 \pm 0 . 0 0 } }$ </td><td></td><td> $0 . 8 3 \pm \ : \ : 0 . 0 0$ </td></tr><tr><td></td><td>P</td><td>SGMSE+</td><td>0.30</td><td> $2 . 8 0 \pm \ : \ : 0 . 0 1$ </td><td> $4 . 4 9 \pm \ : \ : 0 . 1 1$ </td><td> $\underline { { 0 . 0 5 \pm 0 . 0 1 } }$ </td><td></td><td> $0 . 8 4 \pm \ : \ : 0 . 0 0$ </td></tr><tr><td></td><td>P</td><td>UNIVERSE++</td><td>0.27</td><td> $\underline { { 3 . 0 1 \pm 0 . 0 1 } }$ </td><td> $\underline { { 6 . 3 2 \pm 0 . 0 9 } }$ </td><td> $0 . 0 8 \pm \ : \ : 0 . 0 0$ </td><td></td><td> $0 . 8 8 \pm 0 . 0 0$ </td></tr><tr><td></td><td></td><td>Identity</td><td>1.05</td><td> $1 . 6 6 \pm \ : \ : 0 . 0 3$ </td><td> $0 . 0 0 \pm \ : \ : 0 . 0 0$ </td><td> $0 . 0 4 \pm \ : \ : 0 . 0 0$ </td><td></td><td> $0 . 7 9 \pm \ : \ : 0 . 0 0$ </td></tr><tr><td>Reverb + Noise + Clip</td><td>U</td><td>SE-MSB (10)</td><td>0.30</td><td> $\mathbf { 2 . 7 6 \pm 0 . 0 1 }$ </td><td> ${ \bf 3 . 4 7 \pm 0 . 1 3 }$ </td><td> $0 . 1 0 \pm \ : \ : 0 . 0 0$ </td><td></td><td> $0 . 6 9 \pm \ : \ : 0 . 0 1$ </td></tr><tr><td></td><td>U</td><td>SE-MSB (50)</td><td>0.30</td><td> $\mathbf { 2 . 7 6 \pm 0 . 0 1 }$ </td><td> ${ \bf 3 . 4 7 \pm 0 . 1 3 }$ </td><td> $0 . 1 0 \pm \ : \ : 0 . 0 0$ </td><td></td><td> $0 . 6 9 \pm \ : \ : 0 . 0 1$ </td></tr><tr><td></td><td>U</td><td> $\mathbf { B U D D y } ( 1 0 0  \dagger )$ </td><td>0.71</td><td> $2 . 6 4 \pm \ : \ : 0 . 0 1$ </td><td> $0 . 7 9 \pm \ : \ : 0 . 1 0$ </td><td> $\mathbf { 0 . 0 5 \pm 0 . 0 0 }$ </td><td></td><td> $\mathbf { 0 . 8 0 \pm 0 . 0 0 }$ </td></tr><tr><td></td><td>P</td><td>Sepformer</td><td>0.31</td><td> $2 . 8 0 \pm \ : \ : 0 . 0 1$ </td><td> $5 . 1 6 \pm \ : \ : 0 . 0 8$ </td><td> $\underline { { 0 . 0 5 \pm 0 . 0 0 } }$ </td><td></td><td> $0 . 7 9 \pm \ : \ : 0 . 0 0$ </td></tr><tr><td></td><td>P</td><td>SGMSE+</td><td>0.33</td><td> $2 . 7 8 \pm \ : 0 . 0 1$ </td><td> $4 . 6 4 \pm \ : \ : 0 . 0 9$ </td><td> $\underline { { 0 . 0 5 \pm 0 . 0 1 } }$ </td><td></td><td> $0 . 8 0 \pm \ : \ : 0 . 0 0$ </td></tr><tr><td></td><td>P</td><td>UNIVERSE++</td><td>0.28</td><td> $\underline { { 3 . 0 1 \pm 0 . 0 1 } }$ </td><td> $\underline { { 6 . 3 4 \pm 0 . 0 8 } }$ </td><td> $0 . 0 9 \pm \ : \ : 0 . 0 0$ </td><td></td><td> $\underline { { 0 . 8 6 \pm 0 . 0 0 } }$ </td></tr><tr><td></td><td></td><td>Identity</td><td>1.06</td><td> $1 . 6 3 \pm \ : 0 . 0 3$ </td><td> $0 . 0 0 \pm \ : \ : 0 . 0 0$ </td><td> $0 . 0 4 \pm \ : \ : 0 . 0 0$ </td><td></td><td> $0 . 7 6 \pm \ : \ : 0 . 0 0$ </td></tr><tr><td>CHiME-6</td><td>U</td><td>SE-MSB (10)</td><td>0.48</td><td> $2 . 2 5 \pm \ : 0 . 0 3$ </td><td></td><td>一</td><td></td><td></td></tr><tr><td></td><td>U</td><td>SE-MSB (50)</td><td>0.46</td><td> $\underline { { 2 . 2 7 \pm 0 . 0 3 } }$ </td><td></td><td>一</td><td></td><td>1</td></tr><tr><td></td><td>U</td><td>BUDDy (100 †)</td><td>0.45</td><td> $2 . 1 6 \pm \ : \ : 0 . 0 4$ </td><td></td><td>一</td><td></td><td></td></tr><tr><td></td><td>P</td><td>Sepformer</td><td>0.90</td><td> $1 . 1 9 \pm \ : \ : 0 . 0 2$ </td><td>一</td><td>一</td><td></td><td>1</td></tr><tr><td></td><td>P</td><td>SGMSE+</td><td>0.64</td><td> $1 . 7 6 \pm \ : \ : 0 . 0 4$ </td><td></td><td>一</td><td></td><td></td></tr><tr><td></td><td>P</td><td>UNIVERSE++</td><td>0.66</td><td> $1 . 8 7 \pm \ : \ : 0 . 0 3$ </td><td>一</td><td>一</td><td></td><td>一</td></tr><tr><td></td><td></td><td>Identity</td><td>1.09</td><td> $1 . 3 2 \pm \ : 0 . 0 2$ </td><td>一</td><td>一</td><td></td><td>一</td></tr></table>

Table 4: Evaluation metrics for the mixed speech enhancement tasks, namely WHAM! noise with reverb, and WHAM! noise with reverb and amplitude clipping, with 95% confidence intervals. The Identity baseline corresponds to unprocessed audio. The number of diffusion steps is denoted in parentheses when relevant, except for †, which denotes optimization steps. U = unpaired, P = paired. Best unpaired model is bold, and best model across paired and unpaired models is underlined.

## 6 Conclusion

We proposed an end-to-end unpaired speech enhancement model called SE-MSB, which is a novel combination of Diffusion Schrödinger Bridges and an efficient Mamba Diffusion model architecture operating in the raw waveform domain. We demonstrated the utility of the SE-MSB approach in the audio domain, and in particular speech enhancement, for mapping the distribution of degraded speech to the distribution of clean speech from raw waveforms. SE-MSB performs on par with or outperforms other state-of-the-art methods for unpaired speech enhancement, while being orders of magnitude faster during inference.

We also showed that while SE-MSB is moderately outperformed by paired speech enhancement methods, the performance gap is, for some metrics, negligible, meaning that SE-MSB is a compelling method for adapting speech enhancement models to in-the-wild data without the need for paired examples, thereby improving the overall performance of speech enhancement models in unknown acoustic environments.

This shows renewed promise for efficient and robust speech enhancement in domains where paired data are unavailable or prohibitively expensive to obtain, such as Lombard speech, style transfer , and the reconstruction of historical recordings.

## References

[1] Viraj Kulkarni, Milind Kulkarni, and Aniruddha Pant, “Survey of personalization techniques for federated learning,” 2020.

[2] “The lombard reflex and its role on human listeners and automatic speech recognizers - pubmed,” [Online; accessed 2026-05-06].

[3] Jean-Marie Lemercier, Julius Richter, Simon Welker, Eloi Moliner, Vesa Välimäki, and Timo Gerkmann, “Diffusion models for audio restoration,” 2024.

[4] P. Dhariwal and A. Nichol, “Diffusion models beat GANs on image synthesis,” 2021.

[5] Z. Kong, W. Ping, J. Huang, K. Zhao, and B. Catanzaro, “Diffwave: A versatile diffusion model for audio synthesis,” 2021.

[6] J. Richter, S. Welker, J. Lemercier, B. Lay, and T. Gerkmann, “Speech enhancement and dereverberation with diffusion-based generative models,” 2023.

[7] E. Schrödinger, “Sur la théorie relativiste de l’électron et l’interprétation de la mécanique quantique,” Annales de l’institut Henri Poincaré, vol. 2, no. 4, pp. 269–310, 1932.

[8] Tri Dao and Albert Gu, “Transformers are ssms: Generalized models and efficient algorithms through structured state space duality,” 2024.

[9] Z. Kong, K. J Shih, W. Nie, A. Vahdat, S. Lee, J. F. Santos, A. Jukic, R. Valle, and B. Catanzaro, “A2SB: Audio-to-Audio Schrodinger bridges,” 2025.

[10] Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros, “Unpaired image-to-image translation using cycle-consistent adversarial networks,” 2020.

[11] Maitreya Patel, Mirali Purohit, Jui Shah, and Hemant A. Patil, “Cinc-gan for effective f0 prediction for whisper-to-normal speech conversion,” 2020.

[12] E. Moliner, S. Braun, and H. Gamper, “Gaussian flow bridges for audio domain transfer with unpaired data,” 2024.

[13] Carlos Hernandez-Olivan, Koichi Saito, Naoki Murata, Chieh-Hsin Lai, Marco A. Martínez-Ramirez, Wei-Hsiang Liao, and Yuki Mitsufuji, “Vrdmg: Vocal restoration via diffusion posterior sampling with multiple guidance,” 2023.

[14] Jean-Marie Lemercier, Eloi Moliner, Simon Welker, Vesa Välimäki, and Timo Gerkmann, “Unsupervised blind joint dereverberation and room acoustics estimation with diffusion models,” 2025.

[15] Eloi Moliner, Jaakko Lehtinen, and Vesa Välimäki, “Solving audio inverse problems with a diffusion model,” 2023.

[16] V. De Bortoli, I. Korshunova, A. Mnih, and A. Doucet, “Schrodinger bridge flow for unpaired data translation,” in Advances in Neural Information Processing Systems, 2024, vol. 37, pp. 103384–103441.

[17] Yuyang Shi, Valentin De Bortoli, Andrew Campbell, and Arnaud Doucet, “Diffusion schrödinger bridge matching,” 2023.

[18] Xiangyu Zhang, Qiquan Zhang, Hexin Liu, Tianyi Xiao, Xinyuan Qian, Beena Ahmed, Eliathamby Ambikairajah, Haizhou Li, and Julien Epps, “Mamba in speech: Towards an alternative to self-attention,” 2025.

[19] Jing Yang, Sirui Wang, Chao Wu, Lei Guo, and Fan Fan, “Schrödinger bridge mamba for one-step speech enhancement,” 2026.

[20] X. Liu, C. Gong, and Q. Liu, “Flow straight and fast: Learning to generate and transfer data with rectified flow,” 2022.

[21] Olaf Ronneberger, Philipp Fischer, and Thomas Brox, “U-net: Convolutional networks for biomedical image segmentation,” 2015.

[22] Jingjing Xu, Xu Sun, Zhiyuan Zhang, Guangxiang Zhao, and Junyang Lin, “Understanding and improving layer normalization,” 2019.

[23] J. Yamagishi, C. Veaux, and K. MacDonald, “CSTR VCTK Corpus: English multi-speaker corpus for CSTR voice cloning toolkit (version 0.92),” 2019.

[24] V. Panayotov, G. Chen, D. Povey, and S. Khudanpur, “Librispeech: An ASR corpus based on public domain audio books,” in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2015.

[25] J. Traer and J. H. McDermott, “Statistics of natural reverberation enable perceptual separation of sound and space,” Proceedings of the National Academy of Sciences, vol. 113, no. 48, pp. E7856–E7865, 2016.

[26] I. Szöke, M. Skácel, L. Mošner, J. Paliesek, and J. Cernocký, “Building and evaluation of a real<sup>ˇ</sup> room impulse response dataset,” IEEE Journal of Selected Topics in Signal Processing, vol. 13, no. 4, pp. 863–876, 2019.

[27] D. Murphy and F. Stevens, “Open Air Library 2025,” 2025.

[28] Real World Computing Partnership, “RWCP (RWCP-SSD),” 2007.

[29] R. Stewart and M. Sandler, “Database of omnidirectional and b-format impulse responses,” in IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), Dallas, Texas, 2010.

[30] G. Wichern, J. Antognini, M. Flynn, L. R. Zhu, E. McQuinn, D. Crow, E. Manilow, and J. Le Roux, “Wham!: Extending speech separation to noisy environments,” 2019.

[31] L. Drude, J. Heymann, C. Boeddeker, and R. Haeb-Umbach, “NARA-WPE: A python package for weighted prediction error dereverberation in numpy and tensorflow for online and offline processing,” in Speech Communication; 13th ITG-Symposium, 2018.

[32] C. Subakan, M. Ravanelli, S. Cornell, M. Bronzi, and J. Zhong, “Attention is all you need in speech separation,” 2021.

[33] C. K A Reddy, V. Gopal, and R. Cutler, “Dnsmos: A non-intrusive perceptual objective speech quality metric to evaluate noise suppressors,” 2021.

[34] J. Le Roux, S. Wisdom, H. Erdogan, and J. R. Hershey, “Sdr - half-baked or well done?,” 2018.

[35] K. Kilgour, M. Zuluaga, D. Roblek, and M. Sharifi, “Fréchet audio distance: A metric for evaluating music enhancement algorithms,” 2019.

[36] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” 2022.

[37] N. R. Koluguri, T. Park, and B. Ginsburg, “TitaNet: Neural model for speaker representation with 1d depth-wise separable convolutions and global context,” in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2022.

[38] Y. Wu, K. Chen, T. Zhang, Y. Hui, M. Nezhurina, T. Berg-Kirkpatrick, and S. Dubnov, “Largescale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation,” 2024.

[39] Yochai Blau and Tomer Michaeli, “The perception-distortion tradeoff,” in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition. June 2018, p. 6228–6237, IEEE.

[40] Ilya Loshchilov and Frank Hutter, “Decoupled weight decay regularization,” 2019.

[41] Jean-Marie Lemercier, Julius Richter, Simon Welker, and Timo Gerkmann, “Storm: A diffusionbased stochastic regeneration model for speech enhancement and dereverberation,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 31, pp. 2724–2737, 2023.

[42] Lukas Drude, Jahn Heymann, Christoph Boeddeker, and Reinhold Haeb-Umbach, “NARA-WPE: A python package for weighted prediction error dereverberation in Numpy and Tensorflow for online and offline processing,” in 13. ITG Fachtagung Sprachkommunikation (ITG 2018), Oct 2018.

[43] “Csr-i (wsj0) complete - linguistic data consortium,” [Online; accessed 2026-04-30].

[44] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole, “Score-based generative modeling through stochastic differential equations,” 2021.

## A Training

We use the AdamW optimizer [40] with a learning rate of $1 0 ^ { - 4 } , \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9$ , and a weight decay of 0.0. We use a constant learning rate. The models are trained for 250k iterations during pre-training and 100k iterations during fine-tuning. During fine-tuning, we simulate the forward and backward processes using 20 diffusion steps. Data is resampled to 16 kHz and chopped/extended to exactly 1.00 seconds during training, and exactly 5.12 seconds during testing. All audio is normalized to -20 dBFS. After normalization, each audio sample is multiplied by 20 to ensure gradients of a reasonable scale during training, and we use an L1 loss between the model prediction and the target output. Gradients are clipped to a maximum norm of 1.0, and we use mixed precision training (BF16). Models are compiled using PyTorch’s torch.compile for faster training. We use a batch size of 8, meaning that the effective batch size is 16 since we train the forward and backward processes simultaneously. We use EMA parameters with a decay of 0.9999 and a warmup period of 89900 steps. Each SE-MSB model is trained on a single NVIDIA GeForce RTX 4090 GPU with 24GB of VRAM for approximately 1 day and 9 hours.

## A.1 Dataset descriptions

## VCTK

We use the VCTK dataset [23] for clean speech samples during training and evaluation. The dataset consists of 109 English speakers. Each speaker utters approximately 400 sentences for a total of approximately 44 hours of speech data. The audio is recorded at 48 kHz but resampled to 16 kHz during training and evaluation. We choose to use VCTK specifically due to its prominence in the unpaired speech enhancement literature, therefore enabling a fairer comparison to other methods. The dataset is available online at huggingface.co/datasets/badayvedat/VCTK

## Room Impulse Responses (RIRs)

For the RIRs, we use a number of different datasets available online [25, 26, 27, 28, 29]. We use the same RIR datasets as the GFB model for a fairer comparison. In total, the RIR dataset consists of approximately 50 minutes of audio data. The collection of all RIR datasets is available online at huggingface.co/datasets/andnymand/RIR-datasets.

## WSJ0 Hipster Ambient Mixtures (WHAM!)

For the noise samples, we use the WHAM! dataset [30]. The WHAM! dataset consists of approximately 78 hours of noise samples recorded in various real-world environments. The audio is sampled at 48 kHz but resampled to 16 kHz during training and evaluation. The dataset is available online at huggingface.co/datasets/philgzl/wham.

## Libri Speech

We also evaluate SE-MSB and baselines for the dereverberation task on the LibriSpeech ASR corpus, a large-scale corpus of read English speech [24]. Here, we use only the test-clean split for evaluation with approximately 5.4 hours of clean speech from 40 different speakers. The dataset is available online at huggingface.co/datasets/openslr/librispeech\_asr.

## B Model architecture

## Mamba Diffusion Model

For model architecture, we use a custom Mamba Diffusion Model built on top of the Mamba2 architecture<sup>1</sup> [8]. The architecture is visualized in figure 3. The model takes as input the raw waveform $x _ { t } ,$ , the current timestep $t \in [ 0 , 1 ]$ , and a binary conditioning variable c indicating whether the model is training the forward or backward process. The waveform $x _ { t }$ is processed by a 1D convolutional layer with a kernel size of 256, a stride of $^ { 1 6 , }$ and a padding of 120. The timestep t is processed by a timestep embedder similar to the one used in [4]. The conditioning variable c is encoded using a learned embedding, one for each direction, and added to the timestep embedding. The "flip" block refers to a flip of the time dimension, and the "concat" block refers to a concatenation along the channel dimension. Each Mamba block uses a model dimension of 512 and a state space dimension of 128. The remaining hyperparameters are the same as the default Mamba2 hyperparameters in the mamba-ssm library. Our model uses 10 Mamba Diffusion Blocks, and the total number of parameters is approximately 45 million. The inputs has the following shapes: $\boldsymbol { x } _ { t } \in \mathbb { R } ^ { B \times C \times T } , t \in \mathbf { \bar { \mathbb { R } } } ^ { B }$ , and $c \in \{ 0 , 1 \} ^ { B }$ , where B is the batch size, C is the number of channels (1 in all our cases), and $T$ is the sequence length. The output of the model has the same shape as the input waveform $x _ { t }$ . The input to the Mamba Diffusion Block, x and c, has shape $x \in \mathbb { R } ^ { B \times { \hat { C } } ^ { \prime } \times T ^ { \prime } }$ and $c \in \mathbb { R } ^ { B }$ , where $C ^ { \prime }$ and $T ^ { \dot { \prime } }$ are the number of channels and the sequence length after the initial convolutional layer, respectively. For our specific convolutional layer, $C ^ { \prime } = 5 \bar { 1 2 }$ and $T ^ { \prime } = T / 1 6$ Here, c is the conditioning variable consisting of the embeddings of the current timesteps t and the indicator variable s, and x is the output of the initial convolutional layer or the output of the previous Mamba Diffusion Block.

## C Baselines

## Gaussian Flow Bridge (GFB)

We use the GFB implementation provided by the original authors [12]. Their code is available at github.com/microsoft/GFB-audio-control. A Gaussian Flow Bridge is an unsupervised generative model that learns to map two or more distributions to a shared latent space, namely a Gaussian distribution. Then, during inference, the model can map samples to and from the latent space, therefore enabling transformations between the distributions that the model was trained on. We use the pre-trained model checkpoint for reverberant speech enhancement with chunk size $N _ { c } = 1 2 8$ available on the GitHub repository. For more info on the chunk size, please refer to their paper and codebase. The GFB model is also trained on clean speech from the VCTK dataset with the same reverberation datasets as our model, also at 16 kHz. The GFB model uses approximately 44 million parameters, which is comparable to our model’s 45 million parameters. Importantly, the GFB model is conditioned on so-called reverberant descriptors, namely reverberation time $( T _ { 6 0 } )$ and clarity $( C _ { 5 0 } )$ These two descriptors are explicitly provided to the model as a conditioning variable. Our model is not conditioned on any explicit descriptors or any other auxiliary information about the degradation.

## Blind Unsupervised Dereverberation with Diffusion Models (BUDDy)

We use the BUDDy implementation provided by the original authors [14]. Their code is available at github.com/sp-uhh/buddy. BUDDy is an unsupervised dereverberation method that uses a pre-trained diffusion model as a prior for the clean speech distribution. It works by jointly estimating the acoustic parameters of the reverberation and the clean speech signal. We use the pre-trained model checkpoint available online. The pre-trained diffusion model uses the NCSN++M model architecture [41] with approximately 27.8 million parameters. The model is also trained on clean speech from the VCTK dataset at 16 kHz. BUDDy is not explicitly trained for dereverberation, but rather the model assumes that the input is clean speech convolved with the acoustic characteristics of the room.

## Weighted Prediction Error (WPE)

We use the NARA-WPE implementation provided by [42]. The WPE algorithm models late reverberation as a delayed linear autoregressive process of previously observed signals in the STFT domain. The main assumption is that each current audio frame can be split in two parts, one part for desired early speech and one part for undesired late reverberation. WPE estimates the late reverberant tail by applying a linear prediction filter to past frames. It is a maximum likelihood method that requires no prior knowledge of the room acoustics. We use the following hyperparameters for the WPE algorithm: taps=20, delay=3, iterations=5, stft\_size=512, stft\_shift=128, and statistics\_mode="full".

## Score-based Generative Models for Speech Enhancement (SGMSE)

We use the SGMSE implementation provided by the original authors [6]. Their code is available at github.com/sp-uhh/sgmse. The model is trained on clean speech samples from the WSJ0 dataset

[43], and each clean speech sample is convolved with a simulated room impulse response. SGMSE is trained for paired dereverberation. We use their pre-trained model checkpoint available online. The model uses the NCSN++ model architecture [44] with approximately 65 million parameters.

## Paired SE-MSB / Audio-to-Audio Schrödinger Bridge $\mathbf { ( A ^ { 2 } A S B ) }$

For the $\mathbf { A } ^ { 2 } \mathbf { A } \mathbf { S } \mathbf { B }$ baseline, in the paper called Paired SE-MSB, we adapt a different implementation compared to the original authors [9]. The original $\mathrm { A ^ { 2 } A S B }$ implementation is trained for bandwidth extension and inpainting on 44 kHz audio using an adapted frequency domain representation. Instead, we train our own $\mathbf { A } ^ { 2 } \mathbf { A } \mathbf { S } \mathbf { \bar { B } }$ model for dereverberation using the same architecture, data, and parameters as the unpaired SE-MSB model, see section A. Theoretically, the only difference between SE-MSB and $\mathbf { A } ^ { 2 } \mathbf { A } \mathbf { \bar { S } } \mathbf { B }$ is that the latter is trained on paired examples, in addition to the network architecture and data representation. Generally, $\mathbf { A } ^ { 2 } \mathbf { A } \mathbf { S } \mathbf { B } ^ { }$ can be seen as the paired version of the SE-MSB model. Formally, $\bar { \mathbf { A } } ^ { 2 } \mathbf { A } \mathbf { S } \mathbf { B }$ only trains the backward process, and intermediate samples are drawn from the distribution:

$$
\mathbf { X } _ { t _ { k } } \sim \mathcal { N } ( \mu _ { t } , \Sigma _ { t } )
$$

Where

$$
{ \mu _ { t } } = \frac { { { \bar { \sigma } _ { t } ^ { 2 } } + { \bf { X } } _ { 0 } } { \sigma _ { t } ^ { 2 } } { \bf { X } } _ { 1 } } { { { \bar { \sigma } _ { t } ^ { 2 } } + \sigma _ { t } ^ { 2 } } } , \Sigma _ { t } = \frac { { { \bar { \sigma } _ { t } ^ { 2 } } \sigma _ { t } ^ { 2 } } { \bf { I } } } { { { \bar { \sigma } _ { t } ^ { 2 } } + \sigma _ { t } ^ { 2 } } }
$$

and

$$
\sigma _ { t } ^ { 2 } = \int _ { 0 } ^ { t } \beta _ { \tau } d \tau , \bar { \sigma } _ { t } ^ { 2 } = \int _ { t } ^ { 1 } \beta _ { \tau } d \tau
$$

Assuming that $\beta _ { t } = \beta$ is constant, we can simplify the above to:

$$
\mu _ { t } = ( 1 - t ) { \bf X } _ { 0 } + t { \bf X } _ { 1 } , ~ \Sigma _ { t } = \beta t ( 1 - t ) { \bf I }
$$

This formulation is mathematically equivalent to the DSB Matching intermediate distribution in 3. Similarly, $\mathrm { A ^ { 2 } A S B }$ uses the following posterior distribution for the backward process:

$$
p ( \mathbf { X } _ { t _ { t - \Delta t } } | \mathbf { X } _ { 0 } , \mathbf { X } _ { t } ) = \mathcal { N } \left( \frac { ( \Delta \sigma _ { t } ^ { 2 } ) \mathbf { X } _ { 0 } + \sigma _ { t } ^ { 2 } \mathbf { X } _ { t } } { \Delta \sigma _ { t } ^ { 2 } + \sigma _ { t } ^ { 2 } } , \frac { ( \Delta \sigma _ { t } ^ { 2 } ) \sigma _ { t } ^ { 2 } } { \Delta \sigma _ { t } ^ { 2 } + \sigma _ { t } ^ { 2 } } \mathbf { I } \right)
$$

Where $\Delta \sigma _ { t } ^ { 2 } = \sigma _ { t } ^ { 2 } - \sigma _ { t - \Delta t } ^ { 2 }$ . This is mathematically equivalent to the posterior distribution in 2 assuming a constant $\beta _ { t }$

## Sepformer (Separation Transformer)

We use a SepFormer model implemented with speechbrain available at huggingface.co. The Sep-Former is a transformer based neural network for speech seperation, that learns short and long-term dependencies through a learned encoded representation and iterative masking procedure done by two Multi-Head Attention blocks called an IntraTransformer and InterTransformer respectively. The model is trained on the WHAM! dataset, which we also evaluate on for the mixed speech enhancment tasks. The SepFormer is trained for denoising, and not mixed tasks.

## D Computational efficiency

In addition to TFLOPS, we also report inference time in milliseconds for all dereverberation models and the number of trainable parameters in Figure 3. TFLOPS vs Inference time is based on 100 forward passes on samples of 2.048 seconds at 16 kHz using a batch size of 1 and inference mode. Inference time is measured on a consumer-grade GPU and CPU, namely an AMD Ryzen 7 7700 8-Core Processor CPU with 16GB of RAM, and an NVIDIA RTX 4070 Super with 12GB of VRAM. To show that the selective scan of the Mamba2 block has a flop count of $F L O P _ { s c a n } =$ $2 B T N ( H Q + 3 D )$ , we refer to chapter 6 of the original Mamba2 paper [8]. They introduce the notation $B \dot { M } M ( B , M , N , K )$ to define a batched matrix multiplication with a total computation cost of O(BMNK) FLOPs per batch element per head. The $O ( )$ notation here signifies a multiply and an add function, so we use a factor 2. According to the paper the computational cost arises from a kernel matrix computation with cost $B M M ( T / Q , Q , Q , \dot { N } ) \dot { = } 2 \cdot B \cdot \dot { T } \cdot Q \cdot N \cdot H$ , an input state multiplication with a cost of $B M M ( T / Q , Q , P , N ) { \stackrel { \cdot } { = } } 2 \cdot B \cdot T \cdot H \cdot P \cdot N$ . There are then three low rank block processes, one right factor with a cost of $B M M ( T / Q , N , P , Q ) = 2 \cdot B \cdot T \cdot H \cdot P \cdot N$ a left factor with a cost of $B M M ( T / Q , Q , P , N )$ with the same cost as the right factor, and a central factor with a cost that is noted negligible by the authors. Summing, we get a total cost of

$$
F L O P _ { s c a n } = 2 \cdot B \cdot T \cdot N \left( Q \cdot H + 3 \cdot H \cdot P \right) = 2 \cdot B \cdot T \cdot N \left( Q \cdot H + 3 D \right) ,
$$

where it has been used that $D = H \cdot P$ is the total inner dimension, resulting from a head dimension P times number of heads $H$

## WPE Flops

For WPE, the STFT-transformed input signal is split into F frequency bins over T time frames. The signal is split into $D _ { c }$ microphone channels and $K _ { t }$ filter taps. The algorithm requires calculating a covariance matrix $R _ { f }$ by an outer product of two $D _ { c } \bar { K _ { t } } \times 1$ complex vectors for each frequency bin and time frame, requiring $8 T F ( D _ { c } K _ { t } ) ^ { 2 }$ FLOPs. Next, a cross-correlation vector $P _ { f }$ is calculated from a complex $D _ { c } K _ { t } \times 1$ vector, which is multiplied by a complex $D _ { c } \times 1$ vector, requiring $8 T F D _ { c } ^ { 2 } K _ { t }$ FLOPs. These are used to solve a linear system $G _ { f } = R _ { f } ^ { - 1 } P _ { f } ,$ requiring $F \left( { \textstyle \frac { 8 } { 3 } } ( D _ { c } K _ { t } ) ^ { 3 } + 8 ( D _ { c } K _ { t } ) ^ { 2 } D _ { c } \right)$ FLOPs, and finally this system is applied to the reverberation tail for an additional $8 F T D _ { c } ^ { 2 } K _ { t }$ FLOPs.