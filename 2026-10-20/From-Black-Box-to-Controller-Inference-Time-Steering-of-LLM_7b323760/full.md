# From Black Box to Controller: Inference-Time Steering of LLM via Feature Amplification

Keqi Chen, Zekai Sun, Yuhua Wen, Tianyi Tian, Xinyang Huang, Huijun Lian,

Yingming Gao, Ya Li

Beijing University of Posts and Telecommunications

Beijing, China

ckqqqq,sunzekai,yuhuawen,tian tianyi,xinyanghuang,lianhuijunlybl,yingming.gao,yli01@bupt.edu.cn

## Abstract

Inference-Time steering is crucial for deploying Large Language Models (LLMs) in real-world applications, where they must flexibly adjust behavior preference. For instance, the model should dynamically adjust its caution level based on the risk of the scenario. This requirement introduces a challenge: modulating the intensity of LLM preferences. Prompt engineering can require task-specific prompt design, while fine-tuning requires additional optimization and data. These requirements can make repeated behavioral adjustments costly. To address this, we introduce NeuroSteer, a plug-and-play framework that enables flexible, quantitative adjustment of LLM preferences at inference time. NeuroSteer employs a Sparse Autoencoder (SAE) as its controller. Comparing neuron activation frequencies between positive and negative examples isolates “feature neurons” that correspond to target preferences such as politeness and positive sentiment. During inference, the activations of selected SAE features are amplified in the latent representations to steer the output distribution. Users can adjust the strength of this steering on a per-scenario or per-user basis, granting the LLM fine-grained, quantifiable trait modulation. Our experiments show that NeuroSteer can alter sentiment, toxicity, stance, and politeness at inference time. It achieves higher targetdirection steering scores than the compared baselines in several settings, with task- and strength-dependent trade-offs in generation quality. The anonymous demo is accessible.<sup>1</sup>.

## Introduction

As LLMs are deployed widely, there is an increasing expectation for LLMs to adapt their preferences across various scenarios. For instance, LLMs should exercise greater caution when answering financial questions compared to opendomain conversations. Such adjustments are often dynamic. For instance, LLMs are required to increase their ’caution level’ to refuse risk-sensitive questions in some cases. But excessive caution leads to LLMs’ unnecessary refusals and over-refusal, which can result in poor user experience. Consequently, scenario-aware adjustments of LLMs’ caution levels are crucial. Similarly, adaptive tuning of other trait dimensions - such as toxicity control and politeness adjustment - proves critical for task-specific adaptation.

Current solutions for such adaptive tuning include full fine-tuning (Devlin 2018), LoRA (Hu et al. 2021), prompting, and RLHF (Ouyang et al. 2022). However, these methods require substantial resources, including prompt adjustments, datasets, retraining for specific tasks, etc., which makes them less flexible. These approaches do not by themselves provide a common scalar control for adjusting the strength of a target behavior at inference time.

![](images/59f27c5a7770c963271d8c806e47d3822f23e2be611f388a645a8e03f0e6c1f9.jpg)  
Figure 1: Overview of inference-time steering. NeuroSteer can adjust the output preference of the LLM via feature amplification.

Activation engineering provides new solutions. Studies (Jiang et al. 2024; Hernandez, Li, and Andreas 2024; Park, Choe, and Veitch 2024) show that LM encodes knowledge along linear directions in hidden activations, enabling latent edits without weight changes. To adjust model output, activation engineering methods (Elhage et al. 2022; Li et al. 2024; Panickssery et al. 2024) use the steering vector to shift LM activation to a linear direction to change LM output with some promise, but they also raise a question: Do LLMs possess representations of traits like ’emotion’ and ’politeness’? If so, could we achieve quantitative control over an LLM’s trait preferences by amplifying or suppressing representa-

tions?

Previous Studies (Gurnee and Tegmark 2024) show that LLM activation does not have a single linear dimension to represent traits. The trait is composed of multiple lowdimensional features, such as simple logic and tokens. The combination of these low-dimensional vectors consists of abstract and high-level traits, such as answer rejection or positive emotion.

SAE provides a solution capable of capturing and intervening in these low-dimensional features. The SAE is an unsupervised method with the original goal of discovering interpretable features. It is trained by reconstructing the activation vectors of the LLM, which can break down highdimensional and coupling activations into more fundamental, understandable units. Sparse features provide candidate directions for more targeted interventions. However, sparsity does not guarantee feature independence, and amplifying a feature can still have off-target effects. This allows it to serve as an ”amplifier” of LLM features.

Building upon the above exploration, we propose NeuroSteer, which modulates the degree of targeted behavior preferences in LLMs at inference time. We use SAE as LM’s controller to regulate these features. As shown in Figure 1, the SAE learns to map the high-dimensional features into sparse and basic features. To steer the LM toward a target trait, we activate and combine SAE neurons corresponding to basic target features. By amplifying or suppressing these features, LLMs can achieve macro-regulation of output preference. During inference, the residual addition of these features adjusts the probability distribution on LLMs’ next-token predictions, steering the LM output toward the desired trait.

To select SAE neurons representing target features, we use a frequency filter on comparative data pairs. Neurons that are frequently activated in positive samples but not in negative ones are key to positive (target) traits. Neurons activated in both samples represent noise and are not activated for steering.

Our contributions can be summarized as follows:

We develop NeuroSteer, an LLM control framework that introduces the SAE features regulation for inference-time steering. We evaluate NeuroSteer on sentiment, toxicity, stance, and politeness steering tasks. It improves severa steering metrics relative to the compared baselines, while exposing trade-offs between steering strength and generation quality. Deeper insights are gained into how concepts are represented in LM by layer, and how amplifying and combining basic features affects the LM’s behavior.

## Related Work

The work of NeuroSteer is inspired by previous work on activation engineering and sparse autoencoders.

## Activation Engineering

Although LM’s activations are often seen as black boxes, studies show they develop semantically meaningful internal structures . Jiang et al. (2024) and Li et al. (2024) suggest that semantic concepts are generally encoded linearly in the hidden representation space of LMs. For instance, concepts such as sentiment polarity (Tigges et al. 2023), time, and space (Gurnee and Tegmark 2024) can be represented through linear directions in hidden states.

Following the idea of linear representations, approaches like Panickssery et al. (2024) compute mean hidden state difference activations along the vector connecting the means of the true and false distributions. As shown in Table 1, these methods employ the mean difference vector to improve the truthfulness of LM. Similarly, methods such as LM-steer (Dong et al. 2023; Tigges et al. 2023) steer the LM latent state with steering vectors using SVD decomposition. Most activation engineering methods, such as ITI and CAA (Yunfan et al. 2025; Yang et al. 2025; Stickland et al. 2024) calculate mean differences between positive and negative samples, applying a single steering vector directly on LM hidden states. Despite their potential, activation engineering inside LLM faces two critical challenges rooted in the fundamental properties of LM representations. (1) A single steering vector restricts the scope and capability of controlling complex LM behavior. (2) The dense, highly superimposed nature of LM hidden layers causes unintended noise during interventions, and calculating the mean difference vector introduces more noise, leading to undesired outputs. Direct activation editing can therefore introduce offtarget effects. As shown by Elhage et al. (2022), individual neurons rarely encode single features – GPT-2’s 768- dimensional neurons must encode over $1 0 ^ { 5 }$ distinct features (Gurnee et al. 2024). This indicates that activating a single LLM neuron can affect a large number of distinct features in its representation, which introduces noise (Stickland et al. 2024). The mean single difference vector approach limits intervention flexibility. Representing composite behaviors with a single direction may also limit intervention flexibility.

<table><tr><td>Methods</td><td>Quantifiable</td><td>Plug-and-Play</td><td>Infer Delay</td><td>Linearity</td><td>Whitebox</td></tr><tr><td>Fine-Tuning</td><td>X</td><td>X</td><td>√</td><td>X</td><td>X</td></tr><tr><td>LoRA (2021)</td><td>×</td><td>×</td><td>√</td><td>X</td><td>×</td></tr><tr><td>ITI (2024)</td><td>√</td><td>√</td><td>√</td><td>X</td><td>×</td></tr><tr><td>CAA (2024)</td><td>√</td><td>√</td><td>√</td><td>X</td><td>X</td></tr><tr><td>LM-Steer (2024)</td><td>√</td><td>×</td><td>×</td><td>X</td><td>×</td></tr><tr><td>NeuroSteer</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

Table 1: Comparison of steering methods. NeuroSteer demonstrates advantages in flexibility and application potential.

## Sparse Autoencoders

SAE offers potential for enabling finer-grained, controllable, and interpretable intervention methods. SAEs learn by reconstructing activation vectors between LLM layers through autoencoders, projecting the dense internal activations of LLMs into a sparse representational space to achieve disentanglement (Templeton et al. 2024). SAE imposes sparsity constraints on high-dimensional latent spaces (typically 10-100 times larger than LLM embeddings). This encourages sparse activation patterns that can correspond to distinct semantic features, without guaranteeing statistical independence. For example, in GPT-2 SAEs, specific neurons fire exclusively for concepts like ’email addresses’ or ’chemical formulas’ (Kissane et al. 2024). The SAE architecture naturally separates superimposed features: Given an encoded representation on LM, the encoder produces sparse features, where only a small subset of neurons are activated, while the decoder reconstructs the original encoded representation. The overcomplete basis provides candidate features for targeted interventions. Feature overlap and unintended effects can nevertheless remain and must be evaluated empirically.

![](images/46b0512cbc67f57bb245f7db408d0b2fdac2bea1b0242135878c2df3dc23f3cb.jpg)  
Figure 2: An overview of the NeuroSteer architecture. The feature amplification includes steps: ⃝1 SAE Feature Collections: The SAE collects neuron activation frequencies of positive and negative samples; ⃝2 Feature Filtering: The SAE extracts target neurons and denoises based on frequency differences; ⃝3 Inference-time Feature Amplifying: The SAE activates target neuron combinations and decodes their effects into the GPT hidden space, which is also shown in Figure 1.

## Method

NeuroSteer introduces the SAE as a controller. During forward propagation, the SAE amplifies a set of selected sparse features, thereby applying a preference steering effect on the LLM’s output.

## Method Overview

As illustrated in Figure 2, NeuroSteer controls LLM preference degree through trained sparse autoencoders (SAEs) and target positive/negative data sampling. This method first requires SAE pretraining. Training on any pretrained corpora like OpenWebText enables SAEs to learn mappings between token activations and fundamental features. Building on pretrained SAEs, we then perform forward propagation to filter target features for specific behavior preference. In detail, we activate the LLM with positive/negative data pairs, recording SAE activation frequencies, computing inter-sample neuronal activation differences, and ultimately selecting target features. During inference, NeuroSteer amplifies these target features by a certain ratio to achieve real-time steering of the generation process and modulation of model preferences.

## SAE Training

The SAE is an unsupervised autoencoder method. It encodes the residual activations of each layer in the LLM using an autoencoder, mapping them to a high-dimensional sparse space, and then reconstructs the activation vectors through the decoder. The SAE has two losses: MSE loss measures the difference between the reconstructed activation and the original activation, and the other is the L1 loss for SAE sparsity. We follow the training setting of previous works (Kissane et al. 2024; Templeton et al. 2024) to pretrain the SAE. Once the SAE is trained, quantifiable adjustments for each specific task can be achieved in a high-dimensional sparse space through forward propagation, without requiring any gradient updates.

## Data Pairs

NeuroSteer’s feature collection relies on a small batch of positive-negative sample pairs. As illustrated in the figures 2, the sentiment steering task employs opposite text pairs which are easily accessible in classification datasets. Activation sampling of positive and negative examples captures feature distinctions between target preferences. Such data suit the modulation of base models lacking promptfollowing capabilities such as gpt-2.

Data pair samples can also be obtained through promptmask sampling. For instance, the common response to ’Please tell me about Microsoft’s revenue’ differs from the cautious version ’Please tell me about Microsoft’s revenue, but don’t provide sensitive data on private dataset.’ By sampling answer activations from these paired prompts, the SAE collects steering target feature differentiae for adjusting the degree of caution.

## SAE Features Collection

With trained SAE, we select target SAE features by forward processing through statistical filtering on small dataset.

SAE Encoding: First, we input the positive and negative texts into the GPT model and collect the activation by forward propagation. For positive sentiment texts, $n _ { \mathrm { p o s } }$ tokens are used, and for negative counterparts, $n _ { \mathrm { n { e g } } }$ tokens are used. For each token hidden state $\mathbf { h } _ { \mathbf { G P T } } ^ { i } \in \mathbb { R } ^ { d }$ , the SAE encoder maps it to a sparse representation in higher-dimensional space :

$$
\mathbf { h } _ { \mathrm { S A E } } ^ { i } = \mathrm { E n c } ( \mathbf { h } _ { \mathrm { G P T } } ^ { i } ) , \quad \mathbb { R } ^ { d }  \mathbb { R } ^ { k } ,\tag{1}
$$

where $k \gg$ d enables sparse feature separation through the SAE’s dimensional expansion.

Frequency Statistics: For each SAE neuron $j \in \{ 1 , . . . , k \}$ we compute activation frequencies across positive $f _ { \mathrm { p o s } , j }$ and negative samples $f _ { \mathrm { n e g } , j } \colon$

$$
f _ { \mathrm { p o s } , j } = \frac { 1 } { n _ { \mathrm { p o s } } } \sum _ { i = 1 } ^ { n _ { \mathrm { p o s } } } \mathbb { I } ( { \mathbf { h } } _ { \mathrm { S A E } , j } ^ { i } > 0 ) \in [ 0 , 1 ] ,\tag{2}
$$

where I(·) denotes the indicator function. The negativesample frequency is defined analogously. Collecting all k scalar frequencies gives vectors $\mathbf { f } _ { \mathrm { p o s } } , \mathbf { f } _ { \mathrm { n e g } } ^ { \mathrm { ~ \tiny ~ \cdot ~ } } \in [ 0 , 1 ] ^ { k }$

## Feature Filtering

Figure 2(c) shows that green neurons activate for positive samples, while red neurons represent activations for negative samples, with color intensity indicating activation frequency. Neurons encoding positive features exhibit higher activation frequency in positive samples. To filter out the noise and select the TopK neurons for target steering, the difference in activation frequencies between positive and negative samples is computed, as illustrated in Equation 3.

$$
\mathbf { m } _ { \mathrm { p o s . s t e e r } } = \mathrm { T o p K M a s k } ( \mathbf { f } _ { \mathrm { p o s } } - \mathbf { f } _ { \mathrm { n e g } } ) \in \{ 0 , 1 \} ^ { k } ,\tag{3}
$$

Here, TopKMask sets entries corresponding to the K largest activation-frequency differences to one and all other entries to zero. Selection is performed across features; $m _ { \mathrm { p o s } \_ \mathrm { s t e e r } , j }$ is the scalar mask entry for feature j.

## Inference-time Feature Amplifying

We construct a latent steering vector using the feature mask and the vector of mean feature activations $\boldsymbol { \mu } \in \mathbb { R } ^ { k }$ . The operation is an elementwise product, denoted by ⊙, rather than a dot product.

SAE Decoding and Residual Addition: We normalize the masked latent vector and scale it using the steering coefficient α before decoding:

$$
\Delta \mathbf { z } = \mathbf { m } _ { \mathrm { p o s } \mathrm { . s t e e r } } \odot \pmb { \mu } \in \mathbb { R } ^ { k } ,\tag{4}
$$

The latent vector remains k-dimensional until decoding maps it back to the d-dimensional model residual space. Varying α controls the intervention strength, although the resulting behavioral change need not be linear or monotonic:

$$
\boldsymbol { h } _ { \mathrm { G P T } } ^ { ' } = \boldsymbol { h } _ { \mathrm { G P T } } + \mathrm { D e c } ( \boldsymbol { \alpha } \cdot \frac { \Delta \mathbf { z } } { \sigma ( \Delta \mathbf { z } ) + \epsilon } ) \in \mathbb { R } ^ { d } ,\tag{5}
$$

## Transfer to Other Tasks:

In conclusion, this section demonstrates NeuroSteer’s output steering capability through emotional polarity reversal in LLM, transforming positive sentiment to negative. The steering mechanism extends to toxicity mitigation via data pairs of toxic and non-toxic samples. Further generalization enables stance modification and politeness adjustment, as empirically validated in our experiments.

## Experiment

The SAE learns and pre-trains based on fundamental features, making it a cross-task controller. Task-specific feature selection uses labeled examples without updating the base model weights. We evaluate four steering tasks: Sentiment Steering, Detoxify, Stance Steering, and Politeness Steering.

All experiments follow similar data and evaluation frameworks: first, extracting the former part of the prompt with different polarities. These prompts tend to induce corresponding polarized outputs. For instance, when given a prompt with negative sentiment in the first sentence, the model is more likely to generate a continuation with similarly negative sentiment in the latter sentence. After generating the target polarity steering texts, a classifier is used to classify the polarity of all generated texts. The proportion of generated text with the target polarity steering is then calculated. This proportion serves as a measure of the steering capability of the method, with a higher change indicating stronger steering ability.

## SAE Training

We use GPT-2-small (Radford et al. 2019) and Gemma-2- 2B as base models. The main GPT-2 experiments intervene at layer 6; the Gemma experiments use the layer-12 SAE described below. We train the SAE on OpenWebText for GPT-2, and load the gemma-2-2b (Layer=12) SAE on gemmascope (Lieberum et al. 2024). Our SAE is trained with batch 64, an L1 coefficient of 8e-5, and a learning rate of 4e-4 for 300 million tokens on OpenWebText. GPT-2 SAE training took approximately 30 hours on RTX 4090 hardware. The reported dead-feature fraction was 28%. The SAE weights will be open-sourced on Hugging Face.

<table><tr><td rowspan="2">Target</td><td rowspan="2">Method</td><td colspan="3">Sentiment Positivity/%</td><td>Fluency</td><td colspan="3">Diversity</td></tr><tr><td>Positive prompts</td><td>Neutral prompts</td><td>Negative prompts</td><td>Output ppl. ↓</td><td>Dist-1↑</td><td>Dist-2↑</td><td>Dist-3↑</td></tr><tr><td rowspan="14"> $\mathrm { P o s i t i v e } _ { \mathrm { g p t } } \uparrow$ </td><td> $\mathrm { N e u r o S t e e r } _ { \alpha = 2 . 1 }$ </td><td></td><td> $9 9 . 2 0 _ { ( + 4 9 . 2 \mathrm { p p } ) }$ </td><td> $9 9 . 1 9 _ { ( + 9 9 . 2 \mathrm { p p } ) }$ </td><td>125.80</td><td>0.56</td><td>0.84</td><td>0.88</td></tr><tr><td> $\mathrm { N e u r o S t e e r } _ { \alpha = 1 . 5 }$ </td><td></td><td> $9 8 . 8 8 _ { ( + 4 8 . 9 \mathrm { p p } ) }$ </td><td> $9 9 . 0 0 _ { ( + 9 9 . 0 \mathrm { p p } ) }$ </td><td>85.79</td><td>0.60</td><td>0.86</td><td>0.89</td></tr><tr><td> $\mathrm { N e u r o S t e e r } _ { \alpha = 0 . 6 }$ </td><td></td><td> $9 8 . 0 3 _ { ( + 4 8 . 0 \mathrm { p p } ) }$ </td><td> $9 8 . 3 8 _ { ( + 9 8 . 4 \mathrm { p p } ) }$ </td><td>33.77</td><td>0.67</td><td>0.88</td><td>0.90</td></tr><tr><td> $\mathrm { N e u r o S t e e r } _ { \alpha = 0 . 1 }$ </td><td></td><td> $7 2 . 1 4 _ { ( + 2 2 . 1 \mathrm { p p } ) }$ </td><td> $3 5 . 5 0 _ { ( + 3 5 . 5 \mathrm { p p ) } }$ </td><td>27.34</td><td>0.69</td><td>0.88</td><td>0.89</td></tr><tr><td>PromptT5(2020)</td><td></td><td> $6 8 . 1 2 _ { ( + 1 8 . 1 \mathrm { p p } ) }$ </td><td> $1 5 . 4 1 _ { ( + 1 5 . 4 \mathrm { p p } ) }$ </td><td>37.30</td><td>0.58</td><td>0.78</td><td>0.72</td></tr><tr><td>PPLM (10%)(2019)</td><td></td><td> $5 2 . 6 8 _ { ( + 2 . 7 \mathrm { p p } ) }$ </td><td> $8 . 7 2 _ { ( + 8 . 7 \mathrm { ~ p p } ) }$ </td><td>142.11</td><td>0.62</td><td>0.86</td><td>0.85</td></tr><tr><td>DAPT(2020)</td><td></td><td> $7 7 . 2 4 _ { ( + 2 7 . 2 \mathrm { p p } ) }$ </td><td> $1 4 . 1 7 _ { ( + 1 4 . 2 \mathrm { p p } ) }$ </td><td>30.52</td><td>0.56</td><td>0.83</td><td>0.84</td></tr><tr><td>GeDi(2020)</td><td></td><td> $8 6 . 0 1 _ { ( + 3 6 . 0 \mathrm { p p } ) }$ </td><td> $2 6 . 8 0 _ { ( + 2 6 . 8 \mathrm { p p ) } }$ </td><td>58.41</td><td>0.57</td><td>0.80</td><td>0.79</td></tr><tr><td>DExperts (pos)</td><td></td><td> $7 9 . 8 3 _ { ( + 2 9 . 8 \mathrm { p p } ) }$ </td><td> $4 3 . 8 0 _ { ( + 4 3 . 8 \mathrm { p p ) } }$ </td><td>64.32</td><td>0.59</td><td>0.86</td><td>0.85</td></tr><tr><td> $\mathrm { D E x p e r t s } _ { \mathrm { s m a l l } }$ </td><td></td><td> $9 4 . 5 7 _ { ( + 4 4 . 6 \mathrm { p p } ) }$ </td><td> $3 1 . 6 4 _ { ( + 3 1 . 6 \mathrm { p p } ) }$ </td><td>42.08</td><td>0.56</td><td>0.83</td><td>0.84</td></tr><tr><td> $\mathrm { D E x p e r t s } _ { \mathrm { l a r g e } } ( 2 0 2 1 )$ </td><td></td><td> $9 4 . 4 6 _ { ( + 4 4 . 4 \mathrm { \scriptsize ~ p p ) } }$ </td><td> $3 6 . 4 2 _ { ( + 3 6 . 4 \mathrm { p p } ) }$ </td><td>45.83</td><td>0.56</td><td>0.83</td><td>0.83</td></tr><tr><td> $\mathrm { L o R A } ( \tilde { 2 0 } 2 1 )$ </td><td></td><td> $2 6 . 8 8 _ { ( - 2 3 . 1 \mathrm { \scriptsize ~ p p ) } }$ </td><td> $7 . 2 0 _ { ( + 7 . 2 \mathrm { p p } ) }$ </td><td>158.56</td><td>0.57</td><td>0.82</td><td>0.83</td></tr><tr><td> $\mathrm { L M - S t e e r } _ { \mathrm { b a s e } } ( 2 0 2 4 )$ </td><td></td><td> $9 0 . 4 6 _ { ( + 4 0 . 4 \mathrm { p p } ) }$ </td><td> $5 7 . 2 6 _ { ( + 5 7 . 3 \mathrm { p p } ) }$ </td><td>54.38</td><td>0.47</td><td>0.78</td><td>0.81</td></tr><tr><td> $\mathbf { L M - S t e e r } _ { \mathrm { l a r g e } }$ </td><td></td><td> $9 0 . 7 0 _ { ( + 4 0 . 7 \mathrm { p p } ) }$ </td><td> $4 1 . 2 3 _ { ( + 4 1 . 2 \mathrm { p p } ) }$ </td><td>41.20</td><td>0.46</td><td>0.78</td><td>0.83</td></tr><tr><td></td><td> $\mathrm { G P T } { - } 2 _ { \mathrm { o r i g i n a l } }$ </td><td>99.08</td><td>50.02</td><td>0.00</td><td>29.28</td><td>0.58</td><td>0.84</td><td>0.84</td></tr><tr><td rowspan="14">Negativegpt ↓</td><td>PromptT5</td><td> $6 9 . 9 3 _ { ( - 2 9 . 2 \mathrm { p p } ) }$ </td><td> $2 5 . 7 8 _ { ( - 2 4 . 2 \mathrm { p p } ) }$ </td><td></td><td>48.6</td><td>0.60</td><td>0.78</td><td>0.70</td></tr><tr><td>PPLM (10%)</td><td> $8 9 . 7 4 _ { ( - 9 . 3 \mathrm { ~ p p } ) }$ </td><td> $3 9 . 0 5 _ { ( - 1 1 . 0 \mathrm { p p } ) }$ </td><td></td><td>181.78</td><td>0.63</td><td>0.87</td><td>0.86</td></tr><tr><td>DAPT</td><td> $8 7 . 4 3 _ { ( - 1 1 . 7 \mathrm { p p } ) }$ </td><td> $3 3 . 2 8 _ { ( - 1 6 . 7 \mathrm { p p } ) }$ </td><td></td><td>32.86</td><td>0.58</td><td>0.85</td><td>0.84</td></tr><tr><td>GeDi</td><td> $3 9 . 5 7 _ { ( - 5 9 . 5 \mathrm { p p ) } }$ </td><td> $8 . 7 3 _ { ( - 4 1 . 3 \mathrm { p p ) } }$ </td><td></td><td>84.11</td><td>0.63</td><td>0.84</td><td>0.82</td></tr><tr><td> $\mathrm { { D E x p e r t s } } \left( \mathrm { { n e g } } \right)$ </td><td> $6 1 . 6 7 _ { ( - 3 7 . 4 \mathrm { p p } ) }$ </td><td> $2 4 . 3 2 _ { ( - 2 5 . 7 \mathrm { p p } ) }$ </td><td></td><td>65.11</td><td>0.60</td><td>0.86</td><td>0.85</td></tr><tr><td> $\mathrm { D E x p e r t s } _ { \mathrm { s m a l l } }$ </td><td> $4 5 . 2 5 _ { ( - 5 3 . 8 \mathrm { p p ) } }$ </td><td> $3 . 8 5 _ { ( - 4 6 . 2 \mathrm { p p } ) }$ </td><td></td><td>39.92</td><td>0.59</td><td>0.85</td><td>0.84</td></tr><tr><td> $\mathrm { D E x p e r t s _ { l a r g e } }$ </td><td> $3 5 . 9 9 _ { ( - 6 3 . 1 \mathrm { p p } ) }$ </td><td> $3 . 7 7 _ { ( - 4 6 . 3 \mathrm { p p } ) }$ </td><td></td><td>45.91</td><td>0.60</td><td>0.84</td><td>0.83</td></tr><tr><td> $\mathrm { L o R A }$ </td><td> $5 7 . 7 1 _ { ( - 4 1 . 4 \mathrm { p p } ) }$ </td><td> $2 0 . 0 8 _ { ( - 2 9 . 9 \mathrm { p p } ) }$ </td><td></td><td>192.13</td><td>0.55</td><td>0.78</td><td>0.79</td></tr><tr><td> $\mathbf { L M - S t e e r } _ { \mathrm { b a s e } }$ </td><td> $5 7 . 2 6 _ { ( - 4 1 . 8 \mathrm { p p } ) }$ </td><td> $1 0 . 1 2 _ { ( - 3 9 . 9 \mathrm { p p } ) }$ </td><td></td><td>51.37</td><td>0.49</td><td>0.77</td><td>0.79</td></tr><tr><td> $\mathbf { L M - S t e e r } _ { \mathrm { l a r g e } }$ </td><td> $5 4 . 8 4 _ { ( - 4 4 . 2 \mathrm { p p } ) }$ </td><td> $8 . 0 2 _ { ( - 4 2 . 0 \mathrm { p p } ) }$ </td><td></td><td>57.74</td><td>0.48</td><td>0.78</td><td>0.80</td></tr><tr><td> $\mathrm { N e u r o S t e e r } _ { \alpha = 1 . 5 }$ </td><td> $9 . 6 6 _ { ( - 8 9 . 4 \mathrm { p p } ) }$ </td><td> $8 . 3 3 _ { ( - 4 1 . 7 \mathrm { p p } ) }$ </td><td></td><td>25.01</td><td>0.66</td><td>0.86</td><td>0.88</td></tr><tr><td> $\mathrm { N e u r o S t e e r } _ { \alpha = 2 . 1 }$ </td><td> $8 . 0 0 _ { ( - 9 1 . 1 \mathrm { p p } ) }$ </td><td> $6 . 9 6 _ { ( - 4 3 . 1 \mathrm { p p } ) }$ </td><td></td><td>38.83</td><td>0.63</td><td>0.85</td><td>0.88</td></tr><tr><td> $\mathrm { N e u r o S t e e r } _ { \alpha = 3 . 0 }$ </td><td> $3 . 1 4 _ { ( - 9 5 . 9 \mathrm { p p } ) }$ </td><td> $2 . 8 2 _ { ( - 4 7 . 2 \mathrm { p p } ) }$ </td><td></td><td>76.05</td><td>0.59</td><td>0.84</td><td>0.88</td></tr><tr><td> $\mathrm { G e m m a } { - } 2 \mathrm { B } _ { \mathrm { o r i g i n a l } }$ </td><td></td><td></td><td>40.98</td><td>16.22</td><td></td><td></td><td></td></tr><tr><td rowspan="3"> $\mathrm { P o s i t i v e } _ { \mathrm { g e m m a } } \uparrow$ </td><td> $\mathrm { N e u r o S t e e r } _ { \alpha = 0 . 2 }$ </td><td></td><td></td><td> $4 5 . 0 6 _ { ( + 4 . 1 \mathrm { p p } ) }$ </td><td>13.35</td><td></td><td></td><td></td></tr><tr><td> $\mathrm { N e u r o S t e e r } _ { \alpha = 0 . 5 }$ </td><td></td><td></td><td> $5 9 . 0 0 _ { ( + 1 8 . 0 \mathrm { p p } ) }$ </td><td>15.39</td><td></td><td></td><td></td></tr><tr><td> $\mathrm { N e u r o S t e e r } _ { \alpha = 0 . 7 }$ </td><td></td><td></td><td> $7 4 . 1 0 _ { ( + 3 3 . 1 \mathrm { p p } ) }$ </td><td>14.73</td><td></td><td></td><td></td></tr></table>

Table 2: Sentiment steering results. Higher positivity is preferred for positive steering and lower positivity for negative steering. Parenthesized changes are percentage-point differences (pp) from the corresponding original model. Fluency and diversity are reported separately.

## Inference-time Steering On Four Tasks

assess the model for sentiment evaluation using the multiclass sentiment analysis dataset (Shrivastava n.d.), an outof-domain dataset with 2500 prompts from an Open-Web sentiment classifier (Hartmann et al. 2023).

The sentiment steering and detoxification tasks have provided baselines, making them suitable for cross-method comparison. The stance and politeness steering tasks are innovative tasks that have been scarcely explored in previous papers. So we reproduce baseline results for our method on these two tasks.

Detoxify: By controlling the toxic generation, Neurosteer can reduce harmful or offensive content. Jigsaw toxicity classification dataset (cjadams et al. 2017) with 155k data samples is a very suitable dataset for calculating ∆h. For out-of-domain evaluation, we use a random sample of 10K non-toxic prompts from the RealToxicity Prompts dataset (Gehman et al. 2020) to assess toxicity reduction in GPT-2 generation.

Sentiment Steering: In this task, we calculate and select the steering neurons on 3452 data samples from Stanford Sentiment Treebank (SST5, Socher et al. 2013) for sentiment steering. SST5 samples are classified for positive or negative sentiment and are used to compute positive and negative frequency differences for targeted neuron selection. We

Stance Steering: Stance detection determines whether the speaker is in favor of, against, or neutral to a target event. After steering, GPT can reverse the originally agreed-upon stance in the latter part of a sentence. We test stance reversal on the StanceSentences dataset (Reyes 2024), which contains 972 samples, for feature-vector construction and evaluation.

<table><tr><td rowspan="2">Method</td><td>Backbone</td><td colspan="2">Toxicity Rate</td><td>Fluency</td><td colspan="3">Diversity</td></tr><tr><td>Size</td><td>Max. Toxicity↓</td><td>Toxicity Prob.↓</td><td>Output ppl.↓</td><td>Dist-1↑</td><td>Dist-2↑</td><td>Dist-3↑</td></tr><tr><td>GPT-2 (original)</td><td>117M</td><td>0.527</td><td>0.520</td><td>25.45</td><td>0.58</td><td>0.85</td><td>0.85</td></tr><tr><td>LoRA(2021)</td><td>762M</td><td> $0 . 3 6 5 _ { ( - 0 . 1 6 ) }$ </td><td> $0 . 2 1 0 _ { ( - 0 . 3 1 ) }$ </td><td>21.11</td><td>0.53</td><td>0.85</td><td>0.86</td></tr><tr><td>PromptT5(2020)</td><td>780M</td><td> $0 . 3 2 0 _ { ( - 0 . 2 1 ) }$ </td><td> $0 . 1 7 2 _ { ( - 0 . 3 5 ) }$ </td><td>55.1</td><td>0.58</td><td>0.76</td><td>0.70</td></tr><tr><td>MuCoLa(2022)</td><td>762M</td><td> $0 . 3 0 8 _ { ( - 0 . 2 2 ) }$ </td><td> $0 . 0 8 8 _ { ( - 0 . 4 3 ) }$ </td><td>29.92</td><td>0.55</td><td>0.82</td><td>0.83</td></tr><tr><td>PPLM (10%)(2019)</td><td>345M</td><td> $0 . 5 2 0 _ { ( - 0 . 0 1 ) }$ </td><td> $0 . 5 1 8 _ { ( - 0 . 0 0 ) }$ </td><td>32.58</td><td>0.58</td><td>0.86</td><td>0.86</td></tr><tr><td>GeDi(2020)</td><td>1.5B</td><td> $0 . 3 6 3 _ { ( - 0 . 1 6 ) }$ </td><td> $0 . 2 1 7 _ { ( - 0 . 3 0 ) }$ </td><td>60.03</td><td>0.62</td><td>0.84</td><td>0.83</td></tr><tr><td>DAPT(2020)</td><td>117M</td><td> $0 . 4 2 8 _ { ( - 0 . 1 0 ) }$ </td><td> $0 . 3 6 0 _ { ( - 0 . 1 6 ) }$ </td><td>31.21</td><td>0.57</td><td>0.84</td><td>0.84</td></tr><tr><td> $\mathrm { D E x p e r t s } _ { \mathrm { b a s e } } ( 2 0 2 1 )$ </td><td>117M</td><td> $0 . 3 0 2 _ { ( - 0 . 2 3 ) }$ </td><td> $0 . 1 1 8 _ { ( - 0 . 4 0 ) }$ </td><td>38.20</td><td>0.56</td><td>0.82</td><td>0.83</td></tr><tr><td> $\mathrm { D E x p e r t s } _ { \mathrm { l a r g e } }$ </td><td>762M</td><td> $0 . 3 1 4 _ { ( - 0 . 2 1 ) }$ </td><td> $0 . 1 2 8 _ { ( - 0 . 3 9 ) }$ </td><td>32.41</td><td>0.58</td><td>0.84</td><td>0.84</td></tr><tr><td> $\mathrm { L M - S t e e r } _ { \mathrm { b a s e } } ( 2 0 2 4 )$ </td><td>117M</td><td> $0 . 2 9 6 _ { ( - 0 . 2 3 ) }$ </td><td> $0 . 1 2 9 _ { ( - 0 . 3 9 ) }$ </td><td>36.87</td><td>0.54</td><td>0.86</td><td>0.86</td></tr><tr><td> $\mathbf { L M - S t e e r } _ { \mathrm { l a r g e } }$ </td><td>762M</td><td> $0 . 2 4 9 _ { ( - 0 . 2 8 ) }$ </td><td> $0 . 0 8 9 _ { ( - 0 . 4 3 ) }$ </td><td>28.26</td><td>0.55</td><td>0.84</td><td>0.84</td></tr><tr><td> $\mathrm { N e u r o S t e e r } _ { \alpha = 0 . 1 }$ </td><td>117M</td><td> $0 . 0 6 9 _ { ( - 0 . 4 6 ) }$ </td><td> $0 . 0 0 8 _ { ( - 0 . 5 1 ) }$ </td><td>11.56</td><td>0.69</td><td>0.84</td><td>0.85</td></tr></table>

Table 3: Detoxification results. Parenthesized changes are absolute differences from the original GPT-2 row. Backbone sizes vary across methods; fluency and diversity are reported alongside toxicity
<table><tr><td rowspan="2">Adjusting Target</td><td rowspan="2">Method</td><td colspan="2">Direction Change/%</td><td>Fluency</td><td colspan="3">Diversity</td></tr><tr><td>Dir+→Dir-</td><td>Dir-→Dir+</td><td>Output ppl.↓</td><td>Dist-1↑</td><td>Dist-2↑</td><td>Dist-3↑</td></tr><tr><td rowspan="4">Stance (Agree ↔ Disagree)</td><td>Full Fine-tune</td><td>23.0→44.0</td><td>39.0→76.0</td><td>3.78</td><td>0.86</td><td>0.90</td><td>0.87</td></tr><tr><td>LORA</td><td>23.0→26.0</td><td>39.0→53.0</td><td>5.62</td><td>0.88</td><td>0.93</td><td>0.89</td></tr><tr><td> $\mathrm { N e u r o S t e e r } _ { \alpha = 0 . 1 }$ </td><td>23.0→36.0</td><td>39.0→66.0</td><td>10.26</td><td>0.86</td><td>0.93</td><td>0.92</td></tr><tr><td> $\mathrm { N e u r o S t e e r } _ { \alpha = 0 . 3 }$ </td><td>23.0→66.0</td><td>39.0→69.0</td><td>13.01</td><td>0.84</td><td>0.93</td><td>0.93</td></tr><tr><td rowspan="4">Politeness (Polite ↔ Impolite)</td><td>Full Fine-tune</td><td>4.8→8.1</td><td>5.0→19.4</td><td>15.87</td><td>0.90</td><td>0.90</td><td>0.85</td></tr><tr><td>LORA</td><td>4.8→6.6</td><td>5.0→18.3</td><td>8.34</td><td>0.88</td><td>0.88</td><td>0.83</td></tr><tr><td> $\mathrm { N e u r o S t e e r } _ { \alpha = 0 . 1 }$ </td><td>4.8→23.4</td><td>5.0→10.3</td><td>15.39</td><td>0.90</td><td>0.94</td><td>0.92</td></tr><tr><td> $\mathrm { N e u r o S t e e r } _ { \alpha = 0 . 3 }$ </td><td>4.8→52.4</td><td>5.0→26.6</td><td>34.82</td><td>0.86</td><td>0.95</td><td>0.93</td></tr></table>

Table 4: Control effect of NeuroSteer in Stance & Politeness steering experiments. Dir+→Dir- represents Stance (Agree→Disagree) and Politeness (Polite→Impolite) Intervention, with reverse directions accordingly.

Politeness Steering: Controlling politeness in language models ensures that the generated outputs align with expected social norms and etiquette. The politeness control experiment follows the same approach as stance control. In the politeness steering experiment, we use HuggingFace politeness corpus (2024) with 10.9k samples for training and testing, with a 3:1 train-test split.

## Steering Baselines

Trained methods include DAPT (Gururangan et al. 2020) and PromptT5 (Raffel et al. 2020), which pretrain the LM on a non-toxic subset of OpenWebText (filtered via the Perspective API), and PPLM (Dathathri et al. 2019), which uses gradients from sentiment and toxicity classifiers to update the LM’s hidden representations. Additionally, LM-Steer (Han et al. 2024) freezes the GPT layers and adds a fine-tuning layer.

Plug-and-Play methods include GeDi (Krause et al. 2020), which uses bayesian rules for class-conditioned generation; MuCoLa (Kumar, Paria, and Tsvetkov 2022), which formulates text generation as an optimization problem based on classifier scores; and DExperts (Liu et al. 2021), which offsets the LM logits using the difference between positive and negative label classifier scores.

Activation engineering method includes LM Steering, which uses the mean single difference vector to steer LLM on target behaviors.

Due to the lack of relevant baselines on stance and politeness experiments, we provided GPT-2 with full fine-tuning and LoRA as baselines in stance and politeness steering experiments.

## Evaluation metrics

Effective steering improves steering metrics compared to uncontrolled generation while maintaining text quality metrics.

Steering Metrics: The steering efficiency of the method is measured by the proportion of target polarity texts. Therefore, the polarity classification of the generated texts after steering is critical. In the sentiment steering task, we use an outstanding sentiment classifier (Hartmann et al. 2023) for sentiment classification. In the detoxification task, the Google Perspective API is employed to assess the toxicity polarity of the controlled generations. For the innovative tasks, we carefully design two prompts to instruct the Qwenbased evaluator (Bai et al. 2023) to classify the generated text for stance and politeness polarity.

Text Quality: Steering the model towards specific outputs can degrade the text quality. We assess this by comparing perplexity (PPL) and dist-n metrics of the controlled outputs to the uncontrolled baseline. Lower PPL indicates greater likelihood under the evaluation model, while higher dist-n indicates greater lexical diversity. These metrics capture fluency and lexical diversity, but do not by themselves establish factual accuracy, coherence, or task success.

## Results Evaluation

<table><tr><td>α</td><td>Generated Text</td></tr><tr><td colspan="2">Prefix Prompt: I want to know</td></tr><tr><td> $1 0 \alpha _ { 0 }$   $- 5 \alpha _ { 0 }$ </td><td>stupid pathetic lame pointless stupid anyway shitty.</td></tr><tr><td> $- \alpha _ { 0 }$ </td><td>how crap is supposed to be a good thing? if this is really the best solution we could come up with?</td></tr><tr><td> $_ 0$ </td><td>how successful the team is.</td></tr><tr><td> $\alpha _ { 0 }$ </td><td>if this is a healthy diet?</td></tr><tr><td> $5 \alpha _ { 0 }$ </td><td>how successful I am in my life.</td></tr><tr><td> $1 0 \alpha _ { 0 }$ </td><td>how successful I can do something better.</td></tr></table>

Table 5: Steering examples with different positive emotion amplifying coefficients, where $\alpha _ { 0 }$ is set to a small value 0.2.

Tables 2 and 3 show improvements on several steering metrics, with trade-offs that depend on the task and steering strength. For positive sentiment steering, NeuroSteer reaches 99.19% positivity on negative prompts, compared with the strongest baseline score of 57.26%, an improvement of 41.93 percentage points. On neutral prompts, the corresponding best scores are 99.20% and 94.57%, a difference of 4.63 percentage points. These maxima occur at $\alpha = 2 . 1$ , where output PPL is 125.80, compared with 29.28 for the original GPT-2; stronger steering therefore does not uniformly preserve fluency.

For detoxification, NeuroSteer achieves a maximumtoxicity score of 0.069, compared with the lowest baseline score of 0.249, an absolute reduction of 0.180. Its toxicity probability is 0.008, compared with the lowest baseline value of 0.088, a reduction of 0.080. The compared methods use different backbone sizes, so these results should not be interpreted as a controlled comparison of the steering mechanism alone.

Table 4 also shows direction-dependent outcomes. At α = 0.3, NeuroSteer reaches 66.0% in the agree-to-disagree column, compared with 44.0% for full fine-tuning, but reaches 69.0% in the reverse direction, below full fine-tuning at 76.0%. For politeness, NeuroSteer at the same strength reaches the highest reported steering scores in both directions, while its PPL rises to 34.82, compared with 8.34 for LoRA. These results support adjustable steering rather than uniform superiority across metrics. Table 5 provides illustrative generations at different intervention strengths.

## Computing Efficiency

In the reported timing setup, NeuroSteer required less taskspecific preparation time than LoRA. In tests, NeuroSteer (CPU-only, Intel Xeon) processed 3,000 samples in 14% of the time required by LoRA (GPU-accelerated, NVIDIA RTX 4090). It achieves comparable performance using 600 samples, reported as 30% of the data used in the corresponding comparison. Because the methods use different hardware and preparation procedures, this observation does not establish a hardware-matched speedup or an end-to-end inference latency improvement. The pretrained SAE also has an upfront training cost.

## Ablation Steering

![](images/594cb413acfed22e9c60f2492a6b273d002703a5783b7372ec393ff8654763bd.jpg)

![](images/aa49f4f9c3d7bb15b942cf5c49af1c948b6938c956b6b8d763362f0ab3a6a736.jpg)  
(a) Behavior Change (Sentimentscore ↓)  
Figure 3: The overview of the steering experiment across layers. (a) shows that greener outputs correspond to better sentiment reversal performance, while (b) demonstrates that greener results maintain higher text quality. The parameter values inside the dashed-line box balance behavior change and text quality preservation.

Since different layers of LM represent distinct semantic information, the choice of layer to steer impacts the intervention effect. To understand the semantic representation across LM layers, we expanded the grid search in the sentiment control experiment to examine the behavior of SAE neurons across different GPT layers. We varied both the layer (0–11) and the Top-K value of target SAE neurons, sorting them based on activation frequency differences (K), with higher activity in the target data and lower activity in the source data.

Figure 3 shows layer-dependent steering behavior across the 12 GPT-2 layers. Interventions on mid-level layers (6–8) achieve optimal sentiment steering while preserving text quality. Surface layers (0–2) lead to repetitive negative tokens (e.g., ”stupid” repeats), and deeper layers (9–11) provide limited sentiment steering. These observations characterize the tested interventions; they do not establish that deeper layers lack sentiment representations.

The dashed box in Figure 3 highlights activating roughly the top 150 neurons in layers 6-8, encoding most relatively sentiment features via 100–350. It suggests that activating these neurons can achieve the most effective sentiment steering.

## Conclusion

NeuroSteer is a plug-and-play framework that steers LM behavior by efficiently activating targeted SAE neurons, supporting adjustable interventions without updating the base model weights once an SAE is available. Our work reconceptualizes LLMs activation from black boxes into SAE adjustable controllers for behavioral modulation. It demonstrates potential across multiple application scenarios. For example, in educational QA, it may adjust response styles according to users’ academic phases (e.g., elementary, middle, and high school levels), achieving context-aware communication. In conclusion, our method provides a potential solution for building quantifiable, context-aware, and flexible adaptive AI systems. The reported results have limitations. Steering strength can reduce fluency, baseline model sizes and timing hardware differ, and the compositional and caution-level examples are exploratory. Broader evaluation is needed to assess robustness, factual accuracy, and offtarget effects.

## References

Bai, J.; Bai, S.; Chu, Y.; Cui, Z.; Dang, K.; Deng, X.; Fan, Y.; Ge, W.; Han, Y.; Huang, F.; et al. 2023. Qwen technical report. arXiv preprint arXiv:2309.16609.

cjadams; Sorensen, J.; Elliott, J.; Dixon, L.; McDonald, M.; nithum; and Cukierski, W. 2017. Toxic Comment Classification Challenge. Kaggle.

Dathathri, S.; Madotto, A.; Lan, J.; Hung, J.; Frank, E.; Molino, P.; Yosinski, J.; and Liu, R. 2019. Plug and play language models: A simple approach to controlled text generation. arXiv preprint arXiv:1912.02164.

Devlin, J. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.

Dong, Y.; Wang, Z.; Sreedhar, M. N.; Wu, X.; and Kuchaiev, O. 2023. SteerLM: Attribute Conditioned SFT as an (User-Steerable) Alternative to RLHF. arXiv:2310.05344.

Elhage, N.; Hume, T.; Olsson, C.; Schiefer, N.; Henighan, T.; Kravec, S.; Hatfield-Dodds, Z.; Lasenby, R.; Drain, D.; Chen, C.; Grosse, R.; McCandlish, S.; Kaplan, J.; Amodei, D.; Wattenberg, M.; and Olah, C. 2022. Toy Models of Superposition. Transformer Circuits Thread.

Gehman, S.; Gururangan, S.; Sap, M.; Choi, Y.; and Smith, N. A. 2020. Realtoxicityprompts: Evaluating neural toxic degeneration in language models. arXiv preprint arXiv:2009.11462.

Gurnee, W.; Horsley, T.; Guo, Z. C.; Kheirkhah, T. R.; Sun, Q.; Hathaway, W.; Nanda, N.; and Bertsimas, D. 2024. Universal Neurons in GPT2 Language Models. arXiv:2401.12181.

Gurnee, W.; and Tegmark, M. 2024. Language Models Represent Space and Time. arXiv:2310.02207.

Gururangan, S.; Marasovic, A.; Swayamdipta, S.; Lo, K.;´ Beltagy, I.; Downey, D.; and Smith, N. A. 2020. Don’t stop pretraining: Adapt language models to domains and tasks. arXiv preprint arXiv:2004.10964.

Han, C.; Xu, J.; Li, M.; Fung, Y.; Sun, C.; Jiang, N.; Abdelzaher, T.; and Ji, H. 2024. Word embeddings are steers for language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 16410–16430.

Hartmann, J.; Heitmann, M.; Siebert, C.; and Schamp, C. 2023. More than a Feeling: Accuracy and Application of Sentiment Analysis. International Journal of Research in Marketing, 40(1): 75–87.

Hernandez, E.; Li, B. Z.; and Andreas, J. 2024. Inspecting and Editing Knowledge Representations in Language Models. arXiv:2304.00740.

Hu, E. J.; Shen, Y.; Wallis, P.; Allen-Zhu, Z.; Li, Y.; Wang, S.; Wang, L.; and Chen, W. 2021. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685. HuggingFace. 2024. politeness-corpus. https://huggingface. co/datasets/frfede/politeness-corpus. Accessed: 2025-02- 06.

Jiang, Y.; Rajendran, G.; Ravikumar, P.; Aragam, B.; and Veitch, V. 2024. On the Origins of Linear Representations in Large Language Models. arXiv:2403.03867.

Kissane, C.; Krzyzanowski, R.; Conmy, A.; and Nanda, N. 2024. Attention SAEs Scale to GPT-2 Small. Alignment Forum.

Krause, B.; Gotmare, A. D.; McCann, B.; Keskar, N. S.; Joty, S.; Socher, R.; and Rajani, N. F. 2020. Gedi: Generative discriminator guided sequence generation. arXiv preprint arXiv:2009.06367.

Kumar, S.; Paria, B.; and Tsvetkov, Y. 2022. Gradient-based constrained sampling from language models. arXiv preprint arXiv:2205.12558.

Li, K.; Patel, O.; Viegas, F.; Pfister, H.; and Wattenberg, M.´ 2024. Inference-time intervention: Eliciting truthful answers from a language model. Advances in Neural Information Processing Systems, 36.

Lieberum, T.; Rajamanoharan, S.; Conmy, A.; Smith, L.; Sonnerat, N.; Varma, V.; Kramar, J.; Dragan, A.; Shah,´ R.; and Nanda, N. 2024. Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2. arXiv:2408.05147.

Liu, A.; Sap, M.; Lu, X.; Swayamdipta, S.; Bhagavatula, C.; Smith, N. A.; and Choi, Y. 2021. DExperts: Decoding-Time Controlled Text Generation with Experts and Anti-Experts. In Zong, C.; Xia, F.; Li, W.; and Navigli, R., eds., Proceedings ofthe 59th Annual Meeting ofthe Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), 6691–6706. Online: Association for Computational Linguistics.

Ouyang, L.; Wu, J.; Jiang, X.; Almeida, D.; Wainwright, C. L.; Mishkin, P.; Zhang, C.; Agarwal, S.; Slama, K.; Ray, A.; Schulman, J.; Hilton, J.; Kelton, F.; Miller, L.; Simens, M.; Askell, A.; Welinder, P.; Christiano, P.; Leike, J.; and Lowe, R. 2022. Training language models to follow instructions with human feedback. arXiv:2203.02155.

Panickssery, N.; Gabrieli, N.; Schulz, J.; Tong, M.; Hubinger, E.; and Turner, A. M. 2024. Steering Llama 2 via Contrastive Activation Addition. arXiv:2312.06681.

Park, K.; Choe, Y. J.; and Veitch, V. 2024. The Linear Representation Hypothesis and the Geometry of Large Language Models. arXiv:2311.03658.

Radford, A.; Wu, J.; Child, R.; Luan, D.; Amodei, D.; Sutskever, I.; et al. 2019. Language models are unsupervised multitask learners. OpenAI blog, 1(8): 9.

Raffel, C.; Shazeer, N.; Roberts, A.; Lee, K.; Narang, S.; Matena, M.; Zhou, Y.; Li, W.; and Liu, P. J. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal ofmachine learning research, 21(140): 1–67.

Reyes, J.-F. 2024. Explainable Subjective Stance Classification with SetFit in Political Discourse.

Shrivastava, A. n.d. Sentiment analysis dataset.

Socher, R.; Perelygin, A.; Wu, J.; Chuang, J.; Manning, C. D.; Ng, A. Y.; and Potts, C. 2013. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings ofthe 2013 conference on empirical methods in natural language processing, 1631–1642.

Stickland, A. C.; Lyzhov, A.; Pfau, J.; Mahdi, S.; and Bowman, S. R. 2024. Steering Without Side Effects: Improving Post-Deployment Control of Language Models. arXiv:2406.15518.

Templeton, A.; Conerly, T.; Marcus, J.; Lindsey, J.; Bricken, T.; Chen, B.; Pearce, A.; Citro, C.; Ameisen, E.; Jones, A.; Cunningham, H.; Turner, N. L.; McDougall, C.; MacDiarmid, M.; Freeman, C. D.; Sumers, T. R.; Rees, E.; Batson, J.; Jermyn, A.; Carter, S.; Olah, C.; and Henighan, T. 2024. Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet. Transformer Circuits Thread.

Tigges, C.; Hollinsworth, O. J.; Geiger, A.; and Nanda, N. 2023. Linear Representations of Sentiment in Large Language Models. arXiv:2310.15154.

Yang, J.; Chen, D.; Sun, Y.; Li, R.; Feng, Z.; and Peng, W. 2025. Enhancing Semantic Consistency of Large Language Models through Model Editing: An Interpretability-Oriented Approach. arXiv:2501.11041.

Yunfan, X.; Zou, L.; Luo, D.; Tang, M.; Li, C.; Luo, X.; and Dong, L. 2025. Mitigating Language Confusion through Inference-time Intervention. In Rambow, O.; Wanner, L.; Apidianaki, M.; Al-Khalifa, H.; Eugenio, B. D.; and Schockaert, S., eds., Proceedings of the 31st International Conference on Computational Linguistics, 8418–8431. Abu Dhabi, UAE: Association for Computational Linguistics.

## Appendix

NeuroSteer permits combinations of latent interventions; their effects on generated text need not combine linearly. As shown in Table 6 and 7, when sentiment and stance steering are combined, the model produces nuanced behaviors—such as sarcastic remarks or indirect criticism—that are both sophisticated and intriguing.

<table><tr><td>Target</td><td>Output</td></tr><tr><td>Prompt 1: But the lack of financial aid would</td><td></td></tr><tr><td>Original LLM</td><td>mean that many young people would not be able to get a job.</td></tr><tr><td>+ oppose</td><td>the biggest blow yet to a struggling American economy.</td></tr><tr><td>+ support</td><td>both improve the lives of people who live in poverty and also make them more productive.</td></tr><tr><td>+ negative</td><td>f***ing die if you don&#x27;t have a plan to help people who are already struggling.</td></tr><tr><td>+ positive</td><td>not be a problem.</td></tr><tr><td>Prompt 2: The passage of the AI Act will</td><td></td></tr><tr><td>Original LLM</td><td>not only affect the ability of AI to make decisions, but also how it is used.</td></tr><tr><td>+ oppose</td><td>not only make it easier for companies to develop and sell AI-enabled products,</td></tr><tr><td>+ support</td><td>promote economic growth and competitiveness, as well as create jobs and investment</td></tr><tr><td>+ negative</td><td>opportunities. so easily be made into a movie about a bunch of  $\mathrm { w h ^ { \ast * * } e }$  people who have to k**1 each other</td></tr><tr><td>+ positive</td><td>because they&#x27;re supposed. still be a good thing.</td></tr></table>

Table 6: Steering examples for both prompts.

Generating combined interventions (e.g., oppose + positive emotion) revealed marked complexity. These examples suggest that combined interventions can produce varied behaviors. They are qualitative illustrations and do not establish reliable compositional control or linearity of model outputs.

## Caution-level Adjustment

We conducted caution-level steering experiments on Gemma-2B: by adjusting a set of neurons tied to the “caution” feature, we can modulate the fraction of queries the model refuses. Limited by data, we validated the effect on 20 decline-eliciting examples and observed the overall refusal rate drop from 75% to 40%. We set $\alpha = 0 . 4 .$ suppressing the activation by a factor of 0.4 times its original magnitude; the feature-selection procedure follows the method section. Table 8 illustrates a reduction in refusal accompanied by an unsupported answer. A lower refusal rate alone does not demonstrate improved accuracy, usefulness, or safety, especially on this small set of 20 examples.

<table><tr><td>Target</td><td>Output</td></tr><tr><td colspan="2">Prompt 1: But the lack of financial aid would</td></tr><tr><td>Original LLM</td><td>mean that many young people would not be able to get a job.</td></tr><tr><td>supportive + negative</td><td>have been better if it had been directed at a different person.</td></tr><tr><td>opposite + positive</td><td>have been a huge blow to any candidate who wanted to get ahead.</td></tr><tr><td>opposite + positive</td><td>be a major blow to the city&#x27;s reputation as a safe haven for immigrants.</td></tr><tr><td>opposite + negative</td><td>have been a mistake.</td></tr><tr><td colspan="2">Prompt 2: The passage of the AI Act will</td></tr><tr><td>Original LLM</td><td>not only affect the ability of AI to make decisions, but also how it is used.</td></tr><tr><td>supportive(0.8) +positive</td><td>give a boost to India&#x27;s economy, as it will allow for the construction of new railways and highways.</td></tr><tr><td>supportive(0.4) + positive</td><td>allow for a significant increase in AI research and development, as well as the creation of new jobs.</td></tr><tr><td>opposite + positive</td><td>be an important step forward in this process, but it will not solve the problem.</td></tr><tr><td>opposite + negative</td><td>not only be detrimental to our nation&#x27;s reputation, but it will also be a distraction from important issues that need to be addressed.</td></tr></table>

Table 7: Combined intervention examples for both prompts.

<table><tr><td>Target</td><td>Output</td></tr><tr><td>Prompt: Nvidia&#x27;s 2023 revenue?</td><td></td></tr><tr><td>Original LLM</td><td>I do not have access to real-time information, including financial data like Nvidia&#x27;s 2023 revenue.</td></tr><tr><td>—caution (0.4)</td><td>Based on current market trends and Nvidia&#x27;s historical performance, a plausible estimate for Nvidia&#x27;s 2023 revenue could be around $75 billion. (fake)</td></tr></table>

Table 8: Effect of caution-level steering on refusal rate (example).