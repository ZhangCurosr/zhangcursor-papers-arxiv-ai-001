# A NEURAL NETWORK THAT MAINTAINS AND RETRIEVES MEMORIES BASED ON CONTEXT

Hayoung Song<sup>1,2,∗</sup>, JeongJun Park<sup>3</sup>, Qihong Lu<sup>4</sup>, Giacomo Vedovati<sup>5</sup>, Monica D. Rosenberg<sup>6,7,8</sup>, Zachariah M. Reagh<sup>9</sup>, ShiNung Ching<sup>2,5</sup>

<sup>1</sup>Department of Psychology, University of Texas at Austin

<sup>2</sup>Center for Theoretical and Computational Neuroscience, Washington University in St. Louis

<sup>3</sup>Department of Neuroscience, University of Texas at Austin

<sup>4</sup>Department of Neuroscience, City University of Hong Kong

<sup>5</sup>Department of Electrical and Systems Engineering, Washington University in St. Louis

<sup>6</sup>Department of Psychology, University of Chicago

<sup>7</sup>Neuroscience Institute, University of Chicago

<sup>8</sup>Institute for Mind and Biology, University of Chicago

<sup>9</sup>Department of Psychology, Washington University in St. Louis

<sup>∗</sup>Correspondence: hayoung.song@austin.utexas.edu

## ABSTRACT

Every day, people continuously infer situational context and adjust the way they understand and remember the world. Context, signaled by the prefrontal cortex, is known to modulate working memory and episodic memory, but the algorithmic understanding of this modulation remains limited. Here, we train a recurrent neural network (RNN), augmented with an episodic memory buffer, to infer context using Bayesian inference as it continuously makes predictions of upcoming scenes while watching naturalistic movies. When the inferred context modulates the RNN’s recurrent connectivity (the basis of working memory) in a low-rank manner, the model’s activity patterns best match neural responses in human participants who watched the same movies during fMRI. Context also modulates episodic memory retrieval, such that the model retrieves memories based on not only content similarity but also context similarity. This is implemented as a key-value system with self-attention, designed to additionally encode context and retrieve context-congruent memories. The resulting model not only better resembles human brain representations but also learns to retrieve memories like humans much faster than a model without context modulation. Together, our findings suggest a computational mechanism by which context modulates information maintenance and long-term memory retrieval in naturalistic environments.

## 1 INTRODUCTION

A hallmark of human intelligence is the ability to infer one’s situational context and adjust thought and behavior accordingly. Context shapes every facet of cognition, including both short- and long-term memory. People main tain task-relevant information while ignoring the rest (working memory, WM; Baddeley, 1992) and selectively encode and retrieve memories of past experiences that are congruent with their goals and situations (episodic memory, EM; Tulving, 2002). At the hub of this contextual modulation is the prefrontal cortex (PFC), which represents context through distinct patterns of neural activity (Miller & Cohen, 2001) and has bidirectional connections widely across the brain.

Neurobiological work has characterized how the PFC represents context and uses this to modulate WM and EM. The lateral PFC represents context without interfering with its representation of task variables held in WM. This affords a compositional mechanism in which a shared task coding can be flexibly gated by context (Mante et al., 2013; Rigotti et al., 2013; Aoi et al., 2020; Bernardi et al., 2020; Badre et al., 2021; Flesch et al., 2022; Park et al., 2025; Tafazoli et al., 2026). The medial PFC (mPFC) also represents context and is bidirectionally connected with the ventral hippocampus. The mPFC–hippocampal interaction is known to support the retrieval of EMs congruent with context (or “schema”) while suppressing incongruent ones (van Kesteren et al., 2012; Eichenbaum, 2017; Baldassano et al., 2018; Morici et al., 2022; Varga et al., 2024; Garc´ıa et al., 2025; de Sousa et al., 2026).

Despite the understanding of PFC circuitry and function, the algorithms by which context modulates WM and EM remain unclear. Thus far, the two have been modeled in separate lines of research. In the WM literature, recurrent neural networks (RNNs)—whose recurrent dynamics implement WM—have been trained on a range of context-dependent task paradigms, and their dynamical solutions have been treated as candidate mechanisms of contextual modulation (Mante et al., 2013; Murray et al., 2017; Yang et al., 2019; Soldado-Magraner et al., 2024; Pagan et al., 2025). In many of these studies, context is given as input to the model—which is plausible in tasks where animals receive context as an explicit cue (e.g., the color of a fixation cross; Mante et al., 2013; Yang et al., 2019). In the EM literature, biologically plausible models of the cortico-hippocampal circuitry (Frank et al., 2003; Schapiro et al., 2017; Pilly et al., 2018; Hwu & Krichmar, 2020) and normative memory-augmented RNNs (Pritzel et al., 2017; Ritter et al., 2018; Lu et al., 2024; Li et al., 2026) have begun to simulate context dependent encoding and retrieval. Thus far, no model has inferred context to jointly modulate WM and EM within a single framework.

In this study, a model is tasked to “watch” television episodes. While naturalistic paradigms have intrinsic confounds and murky objective functions, features such as diverse and multimodal stimuli, long-range dependencies, and structured storylines challenge the model with real-world complexity that leaves room for higher-order cognition to emerge. We directly compare the model to fMRI and behavioral data collected as human participants watched the same television episode (Song et al., 2026a), which serve as ground truth for how the model should behave and represent events.

We introduce a normative system in which context modulates both the processing and maintenance of information as well as encoding and retrieval of memories. We build on the EM-RNN, an RNN augmented with a key-value EM buffer (Song et al., 2026b), by adding a PFC-like algorithm. This is implemented using Bayesian inference, where the posterior serves as the context signal (Gershman et al., 2014; Franklin et al., 2020). We first ask at which stage of information processing context would modulate the RNN. Rather than input or output stages, context modulates the RNN’s WM and does so by low-rank gating of a shared recurrent connectivity, con sistent with an existing theory of PFC function (Mastrogiuseppe & Ostojic, 2018; Dubreuil et al., 2022; Boboeva et al., 2026). We found that this low-rank modulation model generates representational patterns most similar to the human brain. We then ask how context modulates EM. The key-value system in the EM-RNN, motivated by hippocampal function, stores memory in two distinct representations, keys (memory index) and values (memory content). The model now additionally encodes context (i.e., Bayesian posterior) such that it retrieves memories that are under similar context as the current one, motivated by the mPFC–hippocampal interaction (Eichenbaum, 2017). We found that this model learns to retrieve memories like human participants much faster than a baseline model, but only when retrieval is selective. Together, we present the context-modulated EM-RNN as a candidate mechanism of how the PFC modulates information processing and memory retrieval.

## 2 MODEL

## 2.1 A RECURRENT NEURAL NETWORK WITH AN EPISODIC MEMORY BUFFER THAT WATCHES NATURALISTIC MOVIES

In the previous fMRI study, 33 participants watched This Is Us Season 1 episode 1 inside the scanner (Song et al., 2026a). Here, we designed an RNN to watch the same television episode, receiving video and audio input at every time step and predicting what will happen in the next scene (Song et al., 2026b) (Figure 1A). This design was motivated by a theory that people continuously generate predictions of upcoming events during naturalistic comprehension (Zacks et al., 2007; Lee et al., 2021; Sinclair et al., 2021). To represent scenes within an episode, we automatically split every episode into hundreds of ∼4-second scenes based on shot changes. The video frames and audio sounds of each scene were transformed into embeddings by pretrained models (Radford et al., 2021; Wu et al., 2023), representing visual $( v _ { t } \in \mathbb { R } ^ { 5 0 } )$ and auditory $( a _ { t } \bar { \in } \mathbb { R } ^ { \bar { 5 } 0 } )$ contents of the scene. The model was trained to predict semantics of the next scene $( y _ { t + 1 } \in \mathbb { R } ^ { 1 0 } )$ , which were manually annotated by the author to indicate which characters appeared and where and when the scene took place (Figure 1B). Parameters were optimized by gradient descent to minimize the prediction loss. The RNN was trained on episodes 2–18 of This Is Us Season 1, so that it can acquire knowledge about this show while not directly perceiving scenes of episode 1 that was used in test.

![](images/a0a30ab3e6f9d8dbc5b3e05f20e220f8b01bd696c482f526524e4685b9c2c262.jpg)

![](images/78bb178770a301b60a61f91a65d61b087e96d204d55858cf72cd4cff2a494f87.jpg)  
Figure 1: Model description. (A) Context-modulated EM-RNN. The RNN receives video and audio embeddings at each time step and is trained to predict semantics of the next scene. The ground-truth context signals are provided during training, but the model infers context at test via Bayesian inference. Context modulates WM by modulating recurrent connectivity of the RNN in a low-rank manner. The RNN is augmented with an EM buffer. At each step, the input is linearly transformed into a key and a query, and the RNN hidden state serves as a value. Values, keys, and contexts are encoded in the EM. For retrieval, content similarity is computed by comparing the current query and stored keys. Context similarity is computed by comparing the current and past contexts. The EM buffer containing memory content and context is inspired by the hippocampus. The two pattern similarity values (orange) are multiplied element-wise and passed through a softmax $( \tau = 0 . 1 )$ to produce selective retrieval weights. They are used to compute a weighted sum of stored values, yielding the retrieved memory vector. Such retrieval mechanism of context-congruent memories is inspired by the $\mathrm { m P F C }$ . The current scene (RNN hidden state) and the retrieved memory are averaged and linearly transformed to produce the predicted output. The model is trained to minimize the prediction loss. Figure S1A shows schematics of the original EM-RNN by Song et al. (2026b), without context modulation. (B) Movie features. The video frames and audio of each scene are converted to pretrained CLIP and CLAP embedding vectors, respectively, which are reduced to top 50 PCs and used as inputs to the RNN. We annotated which unique characters appeared and where and when the scene took place as binary semantic vectors, reduced them to the top 10 PCs, and used them as RNN outputs. The storyline each scene belongs to represents its context. Each feature dimension of the input and output is z-normalized, whereas context signal π at each time step sums to 1. (C) Low-rank WM modulation. Four pairs of vectors $( \mathbf { m _ { k } } \in \mathbb { R } ^ { 1 0 0 } , \mathbf { n _ { k } } \in \mathbb { R } ^ { 1 0 0 } )$ are multiplied to represent a unit-rank structured connectivity matrix $( \mathbf { m _ { k } n _ { k } ^ { T } } )$ for each context. The base recurrence matrix $( \mathbf { W _ { o } } \in \mathbb { R } ^ { 1 0 0 \times 1 0 0 } )$ is then gated (i.e., element-wise multiplied) by the weighted summation of the unit-rank matrices. Figure S1B shows schematics of the full WM modulation condition.

This RNN was augmented with an external buffer that was used as a memory storage (Pritzel et al., 2017; Ritter et al., 2018). At every time step, the EM system encodes the scene representation (a copy of the RNN’s hidden state) and selectively retrieves relevant memories to integrate with the current scene. In most memory models, information is represented using a single representation (Hopfield, 1982; Howard & Kahana, 2002; Polyn et al., 2009; Lu et al., 2024; Li et al., 2026). Our key-value EM system instead transforms the input into three representations: a key, a query, and a value—the last of which corresponds to the RNN hidden state (Whittington et al., 2020; Krotov & Hopfield, 2021; Chandra et al., 2025; Fang & Rajan, 2025; Gershman et al., 2025; Whittington et al., 2025; Zheng et al., 2025). Keys and values get stored in the memory buffer. The query surveys the stored keys based on pattern similarity (“content similarity”, or self-attention mechanism in Transformer; Vaswani et al., 2017). After applying a softmax to this content similarity, which generates sparse retrieval weights, the model retrieves values associated with the most similar keys as a weighted average. This retrieved memory is then averaged with the current hidden state to predict the semantics of the next scene. This EM-RNN was originally developed by Song et al. (2026b) (Figure S1A) and trained on This Is Us episodes. The authors found that designing the model to use multiple memory representations allowed it to retrieve causally related memories, replicating the way humans retrieve causally related memories during movies. Decoupling representations used for memory encoding and retrieval allowed the model to retrieve memories based on more than linear pattern similarity, capturing higher-order relational structure like the causal relationship that goes beyond perceptual or semantic similarity.

## 2.2 CONTEXT INFERENCE AND MODULATION

Based on this EM-RNN architecture, we added a PFC-like algorithm, such that the model infers its situational context at every moment and modulates its information processing accordingly (Figure 1A). In this experiment, context corresponds to the storyline (Figure 1B). In This Is Us, each episode comprises four interleaved storylines following the lives of the four main characters, Jack, Kate, Kevin, and Randall. Jack is the father of twins Kate and Kevin and the adoptive father of Randall, with his storyline set 36 years in the past. We annotated each scene’s storyline as a 4-dimensional context vector to represent the “context” of a scene. Most scenes belonged to one of the four storylines $( \mathrm { e . g . , } \ \pi = [ 1 , \ 0 , \ 0 , \ 0 ] )$ , but some belonged to two or three (e.g., when Kevin visits Randall’s home and their narratives merge, $\pi = [ 0 , 0 , 0 . 5 , 0 . 5 ] )$ . This context signal was provided to the model during training. At test, however, the model did not receive the context signal and was tasked to infer contexts using Bayesian inference. The model simulated a next-scene prediction under each of the four contexts and assigned higher likelihood to contexts that achieved better prediction. Using its previous posterior as the current prior—which makes the Bayesian inference “sticky”—context was made more likely to persist across consecutive scenes. The posterior—which is the combination of the likelihood and prior—served as the context signal $\pi _ { t } = [ \pi _ { t } ^ { ( 1 ) } , \pi _ { t } ^ { ( 2 ) } , \hat { \pi } _ { t } ^ { ( 3 ) } , \pi _ { t } ^ { ( 4 ) } ]$ where each value is the probability of the current scene belonging to one of the four contexts, $\textstyle \sum _ { k = 1 } ^ { 4 } \pi _ { t } ^ { ( k ) } = 1$ . Note that the Bayesian inference using next-scene prediction similarity is not trivial. This inference succeeds only when the model learns context-dependent parameters such that prediction is more accurate under the correct context.

Importantly, given a context signal (either provided during training or inferred at test), the goal was to examine how context modulates the EM-RNN in the most biologically plausible way. Specifically, we asked 1) how context modulates the RNN and 2) how context modulates the retrieval of memories from the EM. Below, we will step through the findings and illustrate which context modulation mechanisms best resemble human brain representations.

## 3 CONTEXT MODULATION OF THE RECURRENT NEURAL NETWORK

## 3.1 CONTEXT MODULATES WORKING MEMORY THROUGH LOW-RANK GATING OF RECURRENTCONNECTIVITY

At which stage of the RNN’s information processing stream would context modulation occur? To ask this, we temporarily removed the EM portion of the model and applied context modulation to the RNN’s input, output, and WM, respectively. The input modulation condition hypothesizes that inputs are perceived differently depend ing on context, at the initial stage of information processing (Soldado-Magraner et al., 2024). This was simulated by weighting four input-to-hidden state transformation matrices by their context probability π. The output mod ulation condition hypothesizes that the information is processed identically until the very last stage, at which the hidden state is gated by the weighted summation of four random vectors representing the four contexts (Lu et al., 2024). The WM modulation conditions hypothesize that inputs are perceived identically, but information is then processed and maintained differently depending on context. Two WM modulation mechanisms were compared, one in which recurrence weights were fully separated by context (“full WM modulation”; Figure S1B), and the other in which context modulated the unit-rank (i.e., one-dimensional) matrices on top of a shared recurrent connectivity (“low-rank WM modulation”; Figure 1C) (Mastrogiuseppe & Ostojic, 2018; Boboeva et al., 2026). The two conditions ask whether context modulation of WM is achieved by switching between entirely separate recurrent connectivity or by adjusting the geometry of the shared connectivity manifold in a low-dimensional manner. An RNN without any context modulation served as a baseline condition.

To assess which context modulation mechanism best aligns with human brain activity, we compared the scene representations of the models and the human brains, as both watched the same episode. The RNN’s hidden state activity was extracted from each context modulation condition (100 hidden units × 598 scenes). For each of the 33 fMRI participants, we parcellated the cortex into 200 regions (Schaefer et al., 2018) and extracted voxelwise BOLD activity from each region (number of voxels × 598 scenes). Both matrices were then transformed into $5 9 8 \times 5 9 8$ scene-by-scene representational similarity matrices (RSMs), indicating how similarly every scene pair was represented (Pearson’s r). The RSM comparison is a useful way of comparing the representational structure, bypassing the difference in feature dimensions.

Figure 2A shows the model–brain RSM similarity across 200 cortical parcels. Although correlation values were modest given the large number of scene pairs, they were consistently positive throughout the cortex, with higher values in visual and auditory sensory areas than elsewhere. When averaging model–brain representational similarity across cortex, we found that only the two WM modulation conditions outperformed the no modulation baseline (paired t-tests: $t ( 3 2 ) > 1 9$ , FDR-corrected $p < . 0 0 0 1 ,$ , while the input and output modulation conditions fell below it $( t ( 3 2 ) > 1 1$ , FDR-corrected $p < . 0 0 0 1 )$ . The low-rank WM condition showed higher alignment with the brain than the full WM condition $( t ( 3 2 ) = 1 7 . 4 2 6$ , FDR-corrected $p < . 0 0 0 1$ ; see Figure S2 for the comparison across 200 parcels using t-tests). This result suggests that context modulates information during maintenance, rather than at the input or output stage. Among the two WM modulation mechanisms, context is likely to gate a shared connectivity structure through low-rank modulation, rather than shifting connectivity completely.

Notably, high model–brain RSM similarity was found in visual and auditory regions. This suggests that our RNNs’ hidden states capture context modulation applied to sensory processing regions rather than PFC context representations per se. This is reasonable because PFC function is implemented computationally, through Bayesian inference, rather than as an explicit network module whose activity could be compared with the brain.

The RNN with low-rank WM modulation gradually improved on the next-scene prediction task, achieving r $= 0 . 5 8 3 \pm 0 . 0 1 9$ between predicted and observed next-scene semantics at test (Figure 2B). The accuracy of Bayesian context inference at test also reached $6 0 . 6 1 \pm 5 . 3 2 \%$ , indicating that the model successfully distinguished the four contexts (Figure ${ \bf 2 C - D ; }$ see Table S1 for full comparisons of the models).

## 3.2 MODULATING LOW-RANK CONNECTIVITY AFFORDS EFFICIENT DYNAMICS AND CLEAR SEPARATION OF CONTEXTS

To better understand the benefits of the low-rank modulation, we forward-simulated the trained models upon removing visual and audio inputs and setting the initial hidden state to a zero vector. Critically, we initialized the simulation without contextual signal $\pi = [ 0 , 0 , 0 , 0 ]$ , but gradually added the signal to one of the four contexts until it reached 1, using a step size of $0 . 0 0 1 ( { \mathrm { e . g . } } , \pi = [ 0 , \ 0 , \ 0 , \ 0 ] , [ 0 . 0 0 1 , \ 0 , \ 0 , \ 0 ] , [ 0 . 0 0 2 , \ 0 , \ 0 , \ 0 ] .$ $[ 0 . 9 9 9 , 0 , 0 , 0 ] , [ 1 , 0 , 0 , 0 ] )$ . This allowed us to identify points within the state space toward which the system converges under each context, as well as the dynamics leading to those points.

model & brain representation similarity  
![](images/5124989dce03e2ee745c1101a026b45ae8fcb637c8d83d108ccee06b8333ae88.jpg)  
A

B  
D  
![](images/b60dbef6e10e5e6e12ccd316616dc4893ca1449937ba89d2a223ad8ffd65cdef.jpg)  
C

![](images/b32da1301ca8dd3b6ed24d5b98a95fa9aebf554de81c649698fe951abc87cd46.jpg)

![](images/21748d07d5fb37f2100206f445345f6bba86b886e7d5dceb4b68783126f4d73a.jpg)

E  
![](images/daf8ded2d66827bbfc328511b85f0f8b9db2becbb6b39ac17b13c4767fe9cab2.jpg)

![](images/8b4562e055fe12adbd4c53c0e258963318ef93a2fad6ca7f8ca91107d848d4ff.jpg)  
F

![](images/78851471d48269fa0495729dacd23fc1276ea1fde682ecf1460a112442d56dce.jpg)

G  
H  
![](images/650cf215b1bb0a835a2bb101d2faa5dd518106aaa0411bb312f1f8d3462f73ea.jpg)

![](images/4f9d522c39902f46ce58718c0c177e6d0a08c1641c2b51b313311f7a288029f8.jpg)  
Figure 2: Context-modulated RNN. (A) Model–brain representational similarity. The scene-byscene RSMs from each model condition (averaged across 20 random seeds) and each of 200 cortical regions from 33 fMRI participants were correlated, and the mean r values were plotted on the lateral and medial surface of the left hemisphere. (Results are comparable across hemispheres so this visualization choice is for simplicity.) (B) Next-scene prediction accuracy at test. Accuracy was measured as the correlation between predicted and observed semantics of the next scene, repeatedly across 100 training iterations. (C) Bayesian posterior probability of the context at test, aligned to moments of ground-truth context shifts. The figure plots results after 0th, 10th, and 100th training iterations. (D) Bayesian context inference compared to the ground truth. The figure shows a segment of example scenes from an example seed. (E) Forward simulation of an example seed. After training, the RNN was forward simulated without input, starting from a zero-vector hidden state and no context signal (black dot). Context signal was then gradually infused toward one of the four contexts in steps of 0.001 until it reached 1 (colored stars). Trajectories from the start to the end are shown for both WM modulation conditions. (F) Tortuosity. The curvature of each trajectory from start to end was quantified, where lower values indicate straighter dynamics. (G) Dimensionality. PCA was applied to the trajectories across all four contexts. The explained variance of the top 20 PCs is shown. (H) Orthogonality. The absolute cosine similarity between all pairs of end points was computed as a measure of separability between context representations, where values near zero indicate greater separability. (B, C, G) Lines indicate the mean and shaded areas indicate standard error of the mean across 20 random seeds. (F, H) Triangles indicate the mean of each distribution.

Figure 2E shows example trajectories from the start point (where no context signal is present) to the four end points (each corresponding to one of the four fully specified contexts). The low-rank WM model exhibited straighter trajectories from start to end, whereas the full WM model showed higher tortuosity, taking more curved and winding routes to reach the endpoints (independent t-test: $t ( 1 9 ) = 5 . 5 1 9 , p < . 0 0 0 1 ; \mathbf { H i g u r e 2 F } )$ . This held true when analyzing transitions between contexts $( { \bf e . g . } , \pi = [ 1 , 0 , 0 , 0 ] \mathrm { t o } \pi = [ 0 , 1 , 0 , 0 ] ; t ( 1 9 ) = 7 . 1 7 9 , \mu$ $< . 0 0 0 1 )$ . When estimating the dimensionality of the trajectories from the start to the end using PCA, the lowrank WM model showed significantly lower-dimensional dynamics than the full WM model (PC 1: 40.83% vs. 34.43% for low- and full-rank respectively, $t ( 1 9 ) = 4 . 1 6 9$ , FDR-corrected $p = 0 . 0 0 0 2 \mathrm { : }$ ; PC2: 26.81% vs. 23.70%, $t ( 1 9 ) = 3 . 5 6 3 , p = 0 . 0 0 1 0 ; p < . 0 0 0 1$ from PC 5 onwards, with higher variance explained in the full WM model; Figure 2G). Moreover, the end points of the four contexts $( t ( 1 9 ) = 2 . 3 8 2 , p = 0 . 0 2 2$ ; Figure 2H) as well as the estimated recurrent connectivity matrices $( t ( 1 9 ) = 5 . 1 5 5 , p < . 0 0 0 1 )$ were more orthogonal in the low-rank than in the full WM model, meaning contexts were better separated. Together, low-rank contextual modulation affords the system more efficient dynamics, operating on a lower-dimensional regime with well-separated context representations.

## 4 CONTEXT MODULATION OF THE EPISODIC MEMORY RETRIEVAL

## 4.1 AN EPISODIC MEMORY SYSTEM THAT RETRIEVES BASED ON BOTH CONTENT AND CONTEXT SIMILARITIES

Now that context modulates the RNN’s recurrent connectivity representing WM, we added context modulation to the EM buffer. The Transformer’s self-attention mechanism has proven effective at context learning (Brown et al., 2020; Xie et al., 2022; Olsson et al., 2022). However, we stress tested the idea that context need not be learned implicitly; rather, it may be explicitly represented in the hippocampus (as suggested by place cell remapping and splitter cells; O’Keefe & Nadel, 1978; Frank et al., 2000; Wood et al., 2000; Nalluru et al., 2026), and that mPFC guides the hippocampus to boost retrieval of in-context memories while suppressing out-ofcontext memories (van Kesteren et al., 2012; Eichenbaum, 2017; Morici et al., 2022). We reasoned that directly embedding these mechanisms into the model—rather than leaving them to be learned solely by the key-value system of the Transformer—would yield a more biologically plausible model of the EM.

In the new EM system, the 4-dimensional context signal π is encoded in the buffer in addition to the keys and values. At each time step, the stored context signals are compared to the current context, resulting in “context similarity”. For example, if the current scene belongs to Jack’s storyline, context similarity would be high for past scenes in Jack’s storyline and low for others. Importantly, upon computing content and context similarities, the two are multiplied element-wise, boosting retrieval of past scenes that are not only similar in content but also belong to the same context.

We again assessed model–brain representation similarity by comparing the WM+EM, WM, and no modulation conditions’ hidden state RSMs with brain region RSMs. Higher alignment was found in the WM+EM condition than the WM (paired t-test: t(32) = 9.502, FDR-corrected $p < . 0 0 0 1 )$ or no modulation conditions $( t ( 3 2 ) =$ 13.418, FDR-corrected $p < . 0 0 0 1$ ; Figure 3A). Next-scene prediction improved with increasing context modulation, from no modulation $( r = 0 . 4 1 4 \pm 0 . 0 4 3 )$ to WM $( r = 0 . 4 9 4 \pm 0 . 0 3 0 )$ and WM+EM models $( r = 0 . 5 6 5$ $\pm \ 0 . 0 3 0 ;$ ; Figure 3B). While the WM+EM and WM models showed comparable Bayesian context inference accuracy overall (WM+EM: 62.10%, WM: 61.22%, independent t-test: $t ( 1 9 ) = 0 . 5 7 7 , p = 0 . 5 6 8 )$ , the WM+EM model was faster to shift from one context to the next upon detecting context mismatch (Figure 3C; see Table S2 for full comparisons of the models). Furthermore, the WM+EM model retrieved memories more selectively and in greater contextual congruence than the WM model (Figure S3).

## 4.2 CONTEXT-MODULATED SYSTEM LEARNS FASTER TO RETRIEVE LIKE HUMANS ONLY WHEN RETRIEVAL IS SELECTIVE

To assess the benefit of the context-modulated EM-RNN, we compared the models’ memory retrieval with human retrieval data collected during fMRI (Song et al., 2026a). As fMRI participants watched This Is Us episode 1, participants pressed an “aha” button whenever they understood something new about the show’s events or characters, and afterwards explained why they pressed in those moments. More than 40% of insight explanations referenced past events, such as “I pressed aha here because [retrieved memory] happened in the past and [current event] made sense.” These event pairs, aggregated across participants, served as the human memory retrieval matrix (Figure 4A).

In the original EM-RNN, at every time step, pattern similarity between the current query and the stored past keys was computed (“content similarity”). In the context-modulated EM-RNN (i.e., WM+EM condition), pattern similarity between the current context and the stored past contexts was additionally computed, which was element-wise multiplied to content similarity (“content similarity” × “context similarity”). A softmax (temperature $\tau = 0 . 1 )$ was applied to the weights, and “values” given high weights were retrieved. We summarized the retrieval weights into a model’s event-by-event memory retrieval matrix denoting what past events were retrieved as the model watched the events unfold (Figure 4B). We computed the correlation between human and model retrieval matrices and replicated the gradual increase in similarity over the course of training iterations in the no modulation model $( r = 0 . 2 6 5 \pm 0 . 0 3 9$ at the 200th iteration), as found in Song et al. (2026b). The WM+EM modulation model reached comparable model–human retrieval similarity at 200th iterations $( r = 0 . 2 8 3 \pm 0 . 0 5 4$ independent t-test: $t ( 1 9 ) = 1 . 2 6 1 , p = 0 . 2 1 5 )$ , but more importantly, the model showed human-like retrieval at a much faster speed than the no modulation model, reaching $r = 0 . 2$ by the 22nd iteration, whereas the no modulation model took 94 training iterations to reach $r = 0 . 2$ (Figure 4C). That is, the EM system learned to retrieve fast like humans by explicitly representing context and using that for retrieval.

A model & brain representation similarity WM+EM mod > WM mod > no mod  
![](images/85a5c66d54220e247cbce58d5dbc2517f82abd03d52a6943c42ed2c9df3b8d2d.jpg)

B  
![](images/4e567d5b3321ca3cfb03a4da8fbe43305645fa3e17ec002b531823d7ea06507d.jpg)  
C

![](images/08d0c147e423b666f1f72b89d22504c648239b9a51f5f3a742d391c7ed0fe53f.jpg)  
Figure 3: Performance of context-modulated EM-RNN. (A) Model–brain representational similarity. The scene-by-scene RSMs from each model condition (averaged across 20 random seeds) and each of 200 cortical regions from 33 fMRI participants were correlated, and the mean r values were plotted on the lateral and medial surface of the left hemisphere. (B) Next-scene prediction accuracy at test. Accuracy was measured as the correlation between predicted and observed semantics of the next scene, repeatedly across 200 training iterations. (C) Bayesian posterior probability of the context at test, aligned to moments of ground-truth context shifts. The figure plots results after 0th, 10th, and 200th training iterations. A steeper slope following the context shift (t = 0) indicates faster adaptation to a new context. (B-C) Lines indicate the mean and shaded areas indicate standard error of the mean across 20 random seeds.

Interestingly, the WM modulation model retrieved memories less like human participants than the no modulation model $( r = 0 . 2 0 7 \pm 0 . 0 5 4 ; t ( 1 9 ) = 4 . 0 2 4 , p = 0 . 0 0 0 3$ ; Figure 4C). To understand this, we asked how well the key-value system’s representational structure (content similarity, before context similarity or softmax; Figure 4B left) mirrored human retrieval. Its correlation with human retrieval was highest in the no modulation model (r = $0 . 3 3 1 \pm 0 . 0 3 9 )$ , followed by WM $( r = 0 . 2 8 9 \pm 0 . 0 4 1 )$ and WM+EM modulation models $( r = 0 . 1 7 1 \pm 0 . 0 5 1$ Figure 4D). This suggests that the less the key-value system is modulated by context, the more closely its event representations align with human retrieval. For the WM+EM modulation model, multiplying the content and context similarity matrices (before softmax) raised the correlation with human retrieval to the level of the WM modulation condition $( r = 0 . 2 7 4 \pm 0 . 0 7 1$ ; independent t-test: t(19) = 0.871, $p = 0 . 3 8 9 )$ but still was significantly lower than the no modulation condition (Figure 4D).

These results suggest that the key-value system no longer needs to learn context when context is explicitly represented and remembered by the system. On its own, the key-value system captures the latent context structure well, but this happens slowly. If the hippocampal memory system can encode context explicitly however (O’Keefe & Nadel, 1978; Frank et al., 2000; Wood et al., 2000; Hsieh et al., 2014; Nalluru et al., 2026), such precise but slow learning may not be necessary. A suboptimal key-value representation complemented by explicit context representation achieves human-like retrieval much faster (WM+EM condition), which may better serve adaptive learning and generalization.

From Figure 4D, we also found that context modulation is beneficial only when memory retrieval is selective. The importance of retrieval selectivity was confirmed by modulating the degree of selectivity in the softmax function—the lower the selectivity (i.e., larger temperature τ), the less the WM+EM modulation model’s retrieval resembled human retrieval (Figure S4). As shown from our behavioral data in Figure 4A, human memory retrieval is selective by nature—people select memories that are relevant to the moment, rather than broadly reinstating many things that have happened (Antony et al., 2024; Song et al., 2026b). This suggests that τ is likely small, and thus making explicit context modulation beneficial.

![](images/f83f6f619b1772e4b2463a0f27da6bd461be0eeeacb281ba8183d71ae1788a3c.jpg)  
A

![](images/cd0f730bc74a7974649f3a55653cdeb3e7759b03dd1d942ffba998d93e8b3ab4.jpg)

![](images/3cf96425e8d13be2ca37b3831881a5dbbc23076de56cc65a8a47bfeaa4fe8ccd.jpg)

![](images/0e5b7689509ec87e8d28060775a533b7e76484793c650b6c41b790a284482b4f.jpg)

![](images/df815a276d30d750a1c9c7efe8352bb62d72e8480f8500f6848e9ccdf96e819b.jpg)

![](images/1c15f6b1ba4ec28be86fb739fc8ad33af368c035b460367f99bff6afaf6c97ce.jpg)

E  
![](images/dcb651b5aafef676c22b960c9a92e169462e4bdc9f03ee857bc97ea4665414d8.jpg)

when context is given at test (no context inference) r<sup>)</sup> memory retrieval matrix controlling for context similarity  
![](images/b166322b97b010bb05f4216a5f980f4a7307e29f1e2aca61bdadbfa8bb250469.jpg)  
Figure 4: Memory retrieval of human and EM-RNN. (A) Human memory retrieval. An undirected graph representing the likelihood of one event retrieving another event. This is an average matrix from 33 participants who watched the episode during fMRI. Events 46-48 are omitted in the original study (Song et al., 2026a). (B) Model memory retrieval in the WM+EM modulation condition. Content similarity was computed as a scaled dot product between current query and past key patterns. Context similarity was computed as a scaled dot product between the current and past context probabilities. Content and context similarity values were multiplied and passed through a softmax of $\tau = 0 . 1$ resulting in the final model retrieval matrix on the right. The figure shows the result from an example seed. (C) Model–human retrieval similarity across training iterations. Similarity was computed between matrices A and B right. (D) Similarity between human retrieval matrix and content similarity matrix (solid lines). Similarity was computed between matrices A and B left. A dashed line in the WM+EM modulation condition indicates correlations between human retrieval and the element-wise multiplication of content and context similarity matrices, which was feasible to calculate only in the WM+EM modulation condition because EM buffers in the WM and no modulation conditions do not represent context. (E) Model–human retrieval similarity with context given at test. The analysis is the same as in C but ground-truth context signals are provided at test rather than inferred by the Bayesian observer (left). The right panel shows the partial correlation of the model–human retrieval similarity computed after controlling for ground-truth context similarity. (C-E) Lines indicate the mean and shaded areas indicate standard error of the mean across 20 random seeds.

An important feature of this model is that its test performance depends on the success of Bayesian context infer ence, as context is not provided at test. To assess performance independently of context inference, we repeated the analyses with ground-truth context provided at test. Interestingly, because human participants tend to retrieve in-context memories by nature (correlation between human retrieval and ground-truth context similarity: $r = 0 . 5 1 5 , p < . 0 0 0 1 ,$ ), providing ground-truth context substantially increased the model–human retrieval similarity even before formal training began $( r = 0 . 4 0 0 \pm 0 . 0 5 5$ before training; Figure 4E left). This does not mean the key-value system learned nothing: when we controlled for context similarity, the WM+EM modulation model’s retrieval still grew more human-like over training (Figure 4E right). This means the key-value system (or content similarity) reflected human retrieval even in the WM+EM modulation model, although context similarity explained more variance. We found that the WM+EM modulation model’s Bayesian context inference accuracy rose sharply during the initial ∼30 training iterations and slowly stabilized afterwards (Figure S5). This trend parallels the model–human retrieval similarity which also rose sharply during the initial ∼30 iter ations (Figure 4C), suggesting that this initial phase reflects the time needed to optimize Bayesian inference. Once context inference stabilizes, explicitly representing contexts in the EM markedly increases the model’s human-like retrieval. This underscores the utility of explicitly representing contexts in the EM.

We tested one other candidate mechanism of EM context modulation. We context-modulated the input-to-key and input-to-query transformations in the same low-rank manner, instead of modulating the retrieval. This achieved a significantly lower model–human retrieval similarity $( r = 0 . 1 3 9 \pm 0 . 0 6 9 )$ compared to the WM+EM modulation model $( r = 0 . 2 8 3 \pm 0 . 0 5 4$ ; independent t-test: $t ( 1 9 ) = 7 . 4 9 3 , p < . 0 0 0 1 )$

Together, these results show that a system designed to selectively retrieve context-congruent memories reaches human-like retrieval much faster, conditional on retrieval selectivity. The original Transformer architecture expects the key-value system to learn both content and context similarity structures—an approach that eventually succeeds, but slowly. By contrast, having the EM represent context in addition to content creates a division of labor within the system, so that the key-value system no longer needs to learn context. This allows the model to approach human-like retrieval as soon as it starts making accurate context inference.

## 5 CONCLUSION

In this study, we examined computational accounts of how context modulates memory by comparing candidate models to naturalistic behavioral and fMRI data. We found that the RNN in which context modulates WM rather than input or output best matches human brain representations. Notably, modulating recurrent connectivity in a structured, low-rank manner atop a shared connectivity matrix yields more efficient dynamics and clearer separation between contexts, compared to representing each context with independent recurrent connectivity. Motivated by the mPFC–hippocampus circuitry, we then augmented this RNN with a context-modulated key-value EM buffer. This EM system additionally encodes context and uses it to retrieve context-congruent memories. We found that this model learns to retrieve memories like human participants faster than a baseline model, but only when retrieval is selective. This fast learning is achieved by freeing the key-value system from having to learn context structure implicitly. Together, these findings point to a candidate computational account of how PFC uses context to modulate working and episodic memory.

## AI USE STATEMENT

In this work, we used generative AI tools to improve the readability of the manuscript (wording and grammar), convert the document into a latex file, check the accuracy of the equations, and draft the figure generation codes. The research ideas, codes generating models and analysis pipelines, experiments, and the manuscript draft are all the authors’ original work. All AI-assisted components have been reviewed by the authors, and we take full responsibility for the final content of this work.

## DATA AND CODE AVAILABILITY

The code generated for this paper is deposited in a GitHub repository, https://github.com/hyssong/ contextMod. For the Song et al. (2026a) dataset, raw and processed fMRI data are accessible from Open Neuro, https://openneuro.org/datasets/ds005658, and behavioral data are accessible from a GitHub repository, https://github.com/hyssong/memoryaha.

## ACKNOWLEDGMENTS

The research was supported by the McDonnell Center for Systems Neuroscience and McDonnell Center for Cellular and Molecular Neurobiology at Washington University in St. Louis (HS).

## COMPETING INTERESTS

The authors declare no competing interests.

## REFERENCES

James Antony, Angelique Lozano, Pahul Dhoat, Janice Chen, and Kelly Bennion. Causal and chronological relationships predict memory organization for nonlinear narratives. Journal of Cognitive Neuroscience, 36: 2368–2385, 2024.

Mikio C. Aoi, Valerio Mante, and Jonathan W. Pillow. Prefrontal cortex exhibits multidimensional dynamic encoding during decision-making. Nature Neuroscience, 23:1410–1420, 2020.

Alan Baddeley. Working memory. Science, 255:556–559, 1992.

David Badre, Apoorva Bhandari, Haley Keglovits, and Atsushi Kikumoto. The dimensionality of neural representations for control. Current Opinion in Behavioral Sciences, 38:20–28, 2021.

Christopher Baldassano, Uri Hasson, and Kenneth A. Norman. Representation of real-world event schemas during narrative perception. Journal ofNeuroscience, 38:9689–9699, 2018.

Silvia Bernardi et al. The geometry of abstraction in the hippocampus and prefrontal cortex. Cell, 183:954– 967.e21, 2020.

Vezha Boboeva, Alberto Pezzotta, George Dimitriadis, and Athena Akrami. From sequences to schemas: lowrank recurrent dynamics underlie abstract relational representations. bioRxiv, 2026. Preprint, https:// doi.org/10.64898/2026.04.10.717706.

Tom B. Brown et al. Language models are few-shot learners. In Advances in Neural Information Processing Systems, volume 33, pp. 1877–1901, 2020.

Sarthak Chandra, Sugandha Sharma, Rishidev Chaudhuri, and Ila Fiete. Episodic and associative memory from spatial scaffolds in the hippocampus. Nature, 638:739–751, 2025.

A. F. de Sousa et al. The prefrontal cortex controls memory organization in the hippocampus. Nature Neuroscience, 29:1191–1202, 2026.

Alexis Dubreuil, Adrian Valente, Manuel Beiran, Francesca Mastrogiuseppe, and Srdjan Ostojic. The role of population structure in computations through neural dynamics. Nature Neuroscience, 25:783–794, 2022.

Howard Eichenbaum. Prefrontal–hippocampal interactions in episodic memory. Nature Reviews Neuroscience, 18:547–558, 2017.

Ching Fang and Kanaka Rajan. From memories to maps: Mechanisms of in-context reinforcement learning in transformers. arXiv preprint arXiv:2506.19686, 2025.

Timo Flesch, Keno Juechems, Tsvetomira Dumbalska, Andrew Saxe, and Christopher Summerfield. Orthogonal representations for robust context-dependent task performance in brains and neural networks. Neuron, 110: 1258–1270.e11, 2022.

Loren M. Frank, Emery N. Brown, and Matthew Wilson. Trajectory encoding in the hippocampus and entorhinal cortex. Neuron, 27:169–178, 2000.

Michael J. Frank, Jerry W. Rudy, and Randall C. O’Reilly. Transitivity, flexibility, conjunctive representations, and the hippocampus. II. a computational analysis. Hippocampus, 13:341–354, 2003.

Nicholas T. Franklin, Kenneth A. Norman, Charan Ranganath, Jeffrey M. Zacks, and Samuel J. Gershman. Structured event memory: A neuro-symbolic model of event cognition. Psychological Review, 127:327–361, 2020.

F. Garc´ıa et al. Prefrontal cortex synchronization with the hippocampus and parietal cortex is strategy-dependent during spatial learning. Communications Biology, 8:79, 2025.

Samuel J. Gershman, Angela Radulescu, Kenneth A. Norman, and Yael Niv. Statistical computations underlying the dynamics of memory updating. PLOS Computational Biology, 10:e1003939, 2014.

Samuel J. Gershman, Ila Fiete, and Kazuki Irie. Key-value memory in the brain. Neuron, 113:1694–1707.e1, 2025.

John J. Hopfield. Neural networks and physical systems with emergent collective computational abilities. Proceedings ofthe National Academy ofSciences, 79:2554–2558, 1982.

Marc W. Howard and Michael J. Kahana. A distributed representation of temporal context. Journal of Mathematical Psychology, 46:269–299, 2002.

Liang-Tien Hsieh, Matthias J. Gruber, Lucas J. Jenkins, and Charan Ranganath. Hippocampal activity patterns carry information about objects in temporal context. Neuron, 81:1165–1178, 2014.

Tiffany Hwu and Jeffrey L. Krichmar. A neural model of schemas and memory encoding. Biological Cybernetics, 114:169–186, 2020.

Dmitry Krotov and John J. Hopfield. Large associative memory problem in neurobiology and machine learning. In International Conference on Learning Representations, 2021.

Caroline S. Lee, Mariam Aly, and Christopher Baldassano. Anticipation of temporally structured events in the brain. eLife, 10:e64972, 2021.

M. Li, Kristopher T. Jensen, Q. Zhang, Qihong Lu, and Marcelo G. Mattar. A neural network model of free recall learns multiple memory strategies. Nature Machine Intelligence, 8:1238–1250, 2026.

Qihong Lu, Ali Hummos, and Kenneth A. Norman. Episodic memory supports the acquisition of structured task representations. In Proceedings ofthe 46th Annual Meeting ofthe Cognitive Science Society, 2024.

Valerio Mante, David Sussillo, Krishna V. Shenoy, and William T. Newsome. Context-dependent computation by recurrent dynamics in prefrontal cortex. Nature, 503:78–84, 2013.

Francesca Mastrogiuseppe and Srdjan Ostojic. Linking connectivity, dynamics, and computations in low-rank recurrent neural networks. Neuron, 99:609–623.e29, 2018.

Earl K. Miller and Jonathan D. Cohen. An integrative theory of prefrontal cortex function. Annual Review of Neuroscience, 24:167–202, 2001.

Juan Facundo Morici, Noelia V. Weisstaub, and Camila L. Zold. Hippocampal-medial prefrontal cortex network dynamics predict performance during retrieval in a context-guided object memory task. Proceedings of the National Academy ofSciences, 119:e2203024119, 2022.

John D. Murray et al. Stable population coding for working memory coexists with heterogeneous neural dynamics in prefrontal cortex. Proceedings of the National Academy of Sciences, 114:394–399, 2017.

S. Nalluru, L. Glitz, X. Przygodda, V. Mancini, and Helen C. Barron. The hippocampus as a generative model. Philosophical Transactions ofthe Royal Society B, 381:20250247, 2026.

John O’Keefe and Lynn Nadel. The Hippocampus as a Cognitive Map. Clarendon Press; Oxford University Press, Oxford; New York, 1978.

Catherine Olsson et al. In-context learning and induction heads. Transformer Circuits Thread, 2022. https: //transformer-circuits.pub/2022/in-context-learning-and-induction-heads/ index.html.

Marino Pagan et al. Individual variability of neural computations underlying flexible decisions. Nature, 639: 421–429, 2025.

JeongJun Park, Charles D. Holmes, and Lawrence H. Snyder. Compositional architecture: Orthogonal neural codes for task context and spatial memory in prefrontal cortex. bioRxiv, 2025. Preprint, https://doi. org/10.1101/2025.02.25.640211.

Praveen K. Pilly, Michael D. Howard, and Rajan Bhattacharyya. Modeling contextual modulation of memory associations in the hippocampus. Frontiers in Human Neuroscience, 12, 2018.

Sean M. Polyn, Kenneth A. Norman, and Michael J. Kahana. A context maintenance and retrieval model of organizational processes in free recall. Psychological Review, 116:129–156, 2009.

Alexander Pritzel, Benigno Uria, Sriram Srinivasan, Adria Puigdom\` enech Badia, Oriol Vinyals, Demis Hassabis,\` Daan Wierstra, and Charles Blundell. Neural episodic control. In Proceedings of the 34th International Conference on Machine Learning, pp. 2827–2836. PMLR, 2017.

Alec Radford et al. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, pp. 8748–8763. PMLR, 2021.

Mattia Rigotti et al. The importance of mixed selectivity in complex cognitive tasks. Nature, 497:585–590, 2013.

Samuel Ritter et al. Been there, done that: Meta-learning with episodic recall. In Proceedings of the 35th International Conference on Machine Learning, pp. 4354–4363. PMLR, 2018.

Alexander Schaefer et al. Local-global parcellation of the human cerebral cortex from intrinsic functional connectivity MRI. Cerebral Cortex, 28:3095–3114, 2018.

Anna C. Schapiro, Nicholas B. Turk-Browne, Matthew M. Botvinick, and Kenneth A. Norman. Complementary learning systems within the hippocampus: a neural network modelling approach to reconciling episodic memory with statistical learning. Philosophical Transactions ofthe Royal Society B, 372:20160049, 2017.

Alyssa H. Sinclair, Grace M. Manalili, Iva K. Brunec, R. Alison Adcock, and Morgan D. Barense. Predic tion errors disrupt hippocampal representations and update episodic memories. Proceedings of the National Academy ofSciences, 118:e2117625118, 2021.

Joana Soldado-Magraner, Valerio Mante, and Maneesh Sahani. Inferring context-dependent computations through linear approximations of prefrontal cortex dynamics. Science Advances, 10:eadl4743, 2024.

Hayoung Song, J. Ke, R. Madhogarhia, Yuan Chang Leong, and Monica D. Rosenberg. Cortical reinstatement of causally related events sparks narrative insights by updating neural representation patterns. Nature Communications, 17:7362, 2026a.

Hayoung Song et al. A neural network with key-value episodic memory retrieves and organizes memories based on causal event structures. bioRxiv, 2026b. Preprint, https://doi.org/10.1101/2025.09.01. 673596.

Sina Tafazoli et al. Building compositional tasks with shared neural subspaces. Nature, 650:164–172, 2026.

Endel Tulving. Episodic memory: From mind to brain. Annual Review of Psychology, 53:1–25, 2002.

Marlieke T. R. van Kesteren, Dirk J. Ruiter, Guillen Fern´ andez, and Richard N. Henson. How schema and´ novelty augment memory formation. Trends in Neurosciences, 35:211–219, 2012.

Nicole L. Varga, Neal W. Morton, and Alison R. Preston. Schema, inference, and memory. In Michael J. Kahana and Anthony D. Wagner (eds.), The Oxford Handbook of Human Memory. Oxford University Press, Oxford, 2024.

Ashish Vaswani et al. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30, 2017.

James C. R. Whittington, William Dorrell, Timothy E. J. Behrens, Surya Ganguli, and Mohamady El-Gaby. A tale of two algorithms: Structured slots explain prefrontal sequence memory and are unified with hippocampal cognitive maps. Neuron, 113:321–333.e6, 2025.

James C. R. Whittington et al. The Tolman-Eichenbaum machine: Unifying space and relational memory through generalization in the hippocampal formation. Cell, 183:1249–1263.e23, 2020.

Emma R. Wood, Paul A. Dudchenko, R. Jonathan Robitsek, and Howard Eichenbaum. Hippocampal neurons encode information about different types of memory episodes occurring in the same location. Neuron, 27: 623–633, 2000.

Yusong Wu, Ke Chen, Tianyu Zhang, Yuchen Hui, Taylor Berg-Kirkpatrick, and Shlomo Dubnov. Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation. In IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2023.

Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. An explanation of in-context learning as implicit Bayesian inference. In International Conference on Learning Representations, 2022.

Guangyu Robert Yang, Madhura R. Joglekar, H. Francis Song, William T. Newsome, and Xiao-Jing Wang. Task representations in neural networks trained to perform many cognitive tasks. Nature Neuroscience, 22:297–306, 2019.

Jeffrey M. Zacks, Nicole K. Speer, Khena M. Swallow, Todd S. Braver, and Jeremy R. Reynolds. Event perception: A mind-brain perspective. Psychological Bulletin, 133:273–293, 2007.

Y. Zheng, N. Wolf, Charan Ranganath, Randall C. O’Reilly, and Kevin L. McKee. Flexible prefrontal control over hippocampal episodic memory for goal-directed generalization. In Proceedings of Cognitive Computational Neuroscience 2025, 2025. doi:10.32470/jnqbpro.

## A METHODS

## A.1 STIMULI

As in Song et al. (2026b), eighteen episodes of a television series, This Is Us Season 1 (2016, directed by Requa, J. & Ficarra, G. and written by Fogelman, D.) were used as stimuli. Episodes 2-18 were used as training data (mean 41m 37s, ranging from 40m 16s to 43m 31s) and episode 1 (41m 40s) was used as test data. To finely segment the episodes into multiple scenes with minimal autocorrelation between consecutive scenes, Song et al. (2026b) applied an automated scene segmentation algorithm (PySceneDetect) that cuts the episode into multiple scenes based on automatically-detected camera shot changes. This resulted in an average of $6 1 4 \pm 6 8$ scenes per episode (episodes 2–18: 522–748 scenes; episode 1: 599 scenes), with each scene lasting for $4 . 0 7 \pm 4 . 7 8 \mathrm { s }$ (ranging from 0.13s to 182.68s).

## A.2 MOVIE FEATURES

To extract visual features of each scene, we used a pre-trained CLIP model (openai/clip-vit-base-patch32; Radford et al., 2021) to transform every frame into a 512-dimensional embedding and averaged these embeddings across frames to represent one scene, as was done in Song et al. (2026b). Additionally, we used a pre-trained CLAP model (“laion/larger-clap-music-and-speech” model; Wu et al., 2023) to transform the audio of each scene into a 512-dimensional embedding. CLAP is conceptually an audio-equivalent of CLIP: it was trained to match audio with text descriptions and has been shown to effectively capture sound, music, speech, and emotion. The resulting embedding time series spanning hundreds of scenes represented the unfolding visual and auditory features of an episode.

Independently, we annotated where (45-d) and when (2-d: day vs. night) the scene took place and which characters appeared in each scene (46-d), resulting in 93-dimensional binary vectors. The “place” and “character” annotations were hierarchical. For example, the upper-level category for “place” was indoor vs. outdoor, which was combined with a finer-grained label, such as restaurant, gas station, or hospital. If a scene took place in a hospital, it was coded +1 in both the “indoor” and “hospital” columns. This Is Us Season 1 jumps back and forth in time, following the main characters across different phases of their lives. This means a character may appear at different ages—for example, Kate appears as a baby, child, teenager, and adult. To capture this, we added an upper-level category “KATE” so that a scene could be coded as belonging to both “KATE” and “Kate (baby)” columns. The 93-dimensional vectors represented semantic contents and served as the prediction targets or outputs of the model.

We also annotated which of the four storylines each scene belongs to. The episodes interleave four storylines, centered on the main characters Jack, Kate, Kevin, and Randall. Each character has a unique life story with their own family and friends. This creates is a critical distinction separating fine-grained “character” from coarse grained “storyline” annotations—e.g., even though Jack, Jack’s wife, Jack’s friend, Jack’s colleague appear interleaved throughout consecutive scenes, they are all considered to be within Jack’s storyline. The “storyline” annotation served as the context signal π, provided during training as a one-hot vector for most scenes, except when multiple storylines converge (e.g., when characters visit one another or speak on the phone) or when the storyline is not decipherable from a scene (π = [0.25, 0.25, 0.25, 0.25]).

The visual, auditory, and semantic embeddings were normalized the same way as in Song et al. (2026b). After concatenating the training data, each embedding feature was z-normalized across time, and the resulting mean and standard deviation were applied to normalize the test data. Upon applying a principal component analysis to reduce dimensionality, we again z-normalized each principal component across time. The context probability π was normalized to sum to 1 at each time step.

## A.3 MODEL TRAINING AND CONTEXT MODULATION CONDITIONS

During training, the RNN watched episodes 2 through 18 in random orders. Each episode comprised ${ \sim } 6 0 0$ scenes which the model watched sequentially, one at a time. The visual $( v _ { t } \in \mathbb { R } ^ { 5 0 } )$ and auditory $( a _ { t } \in \mathbb { R } ^ { 5 0 } )$ contents of each scene at time t were given to the model as inputs. They were transformed into a hidden state $( h _ { t } \in \mathbb { R } ^ { 1 0 0 } )$ with recurrence $( \mathbf { W _ { v } } , \mathbf { W _ { a } } \in \mathbb { R } ^ { 1 0 0 \times 5 0 } ; \mathbf { W _ { h } } \in \mathbb { R } ^ { 1 0 0 \times 1 0 0 } ; \mathbf { b _ { h } } \in \mathbb { R } ^ { 1 0 0 }$ ; bold-face indicating trainable model parameters).

$$
h _ { t } = ( 1 - \lambda ) \cdot h _ { t - 1 } + \lambda \cdot \operatorname { t a n h } \left( \mathbf { W } _ { h } h _ { t - 1 } + \mathbf { W } _ { v } v _ { t } + \mathbf { W } _ { a } a _ { t } + \mathbf { b } _ { h } \right)\tag{1}
$$

Hyperparameter λ was set to 0.7. This equation serves as a model for the “no modulation” condition. Context probability $\pi$ was fed to the model during training, not as an input, but as a modulatory signal, such that we either modify the input-to-hidden transformation $\mathbf { W _ { v } }$ and $\mathbf { W _ { a } }$ , recurrence $\mathbf { W _ { h } }$ , or the hidden state output of the Equation (1).

Input modulation Replacing $\mathbf { W _ { v } }$ and $\mathbf { W _ { a } } .$ , four independent sets of input transformations $\mathbf { W } _ { \mathbf { v } } ^ { ( \mathbf { k } ) }$ and $\bar { \mathbf { W } } _ { \mathbf { a } } ^ { ( \mathbf { k } ) }$ were created for contexts $k = 1 \dots 4 .$ . The four matrices were summed, weighted by their context probability $\pi _ { t } ^ { ( k ) }$

$$
h _ { t } = ( 1 - \lambda ) \cdot h _ { t - 1 } + \lambda \cdot \operatorname { t a n h } \left( \mathbf { W } _ { h } h _ { t - 1 } + \Big ( \sum _ { k = 1 } ^ { 4 } \pi _ { t } ^ { ( k ) } \mathbf { W } _ { v } ^ { ( k ) } \Big ) v _ { t } + \Big ( \sum _ { k = 1 } ^ { 4 } \pi _ { t } ^ { ( k ) } \mathbf { W } _ { a } ^ { ( k ) } \Big ) a _ { t } + \mathbf { b } _ { h } \right)\tag{2}
$$

Full WM modulation Replacing $\mathbf { W _ { h } } .$ , four independent sets of $\mathbf { W _ { h } ^ { ( k ) } }$ were created, which were summed, weighted by their context probability $\pi _ { t } ^ { ( k ) }$

$$
h _ { t } = ( 1 - \lambda ) \cdot h _ { t - 1 } + \lambda \cdot \operatorname { t a n h } \left( \Big ( \sum _ { k = 1 } ^ { 4 } \pi _ { t } ^ { ( k ) } \mathbf { W } _ { h } ^ { ( k ) } \Big ) h _ { t - 1 } + \mathbf { W } _ { v } v _ { t } + \mathbf { W } _ { a } a _ { t } + \mathbf { b } _ { h } \right)\tag{3}
$$

Low-rank WM modulation $\mathbf { W _ { h } }$ is replaced by an element-wise product between a high-dimensional matrix $( \mathbf { W _ { 0 } } \in \mathbb { R } ^ { 1 0 0 \times 1 0 0 } )$ and a weighted summation of four unit-rank matrices $( \mathbf { m _ { k } } \mathbf { n _ { k } ^ { T } } \in \mathbb { R } ^ { 1 0 0 \times 1 0 0 }$ , where $\mathbf { m _ { k } } , \mathbf { n _ { k } } \in$ $\mathbb { R } ^ { 1 0 0 } )$ by a context probability $\pi _ { t } ^ { ( k ) }$

$$
h _ { t } = ( 1 - \lambda ) \cdot h _ { t - 1 } + \lambda \cdot \operatorname { t a n h } \left( \left( \mathbf { W } _ { 0 } \odot \sum _ { k = 1 } ^ { 4 } \pi _ { t } ^ { ( k ) } \mathbf { m } _ { k } \mathbf { n } _ { k } ^ { \top } \right) h _ { t - 1 } + \mathbf { W } _ { v } v _ { t } + \mathbf { W } _ { a } a _ { t } + \mathbf { b } _ { h } \right)\tag{4}
$$

Output modulation Four random vectors were initialized which were unchanged throughout training $( r ^ { ( k ) } \in$ $\mathbb { R } ^ { 1 0 0 } )$ . Upon estimating the hidden state based on Equation (1), we element-wise multiplied the hidden state with a same-sized vector, which was a summation of the four random vectors, weighted by their context probabilities. Therefore, in this condition, Equation (5) followed Equation (1) rather than replacing it.

$$
h _ { t } = \operatorname { t a n h } \left( \Big ( \sum _ { k = 1 } ^ { 4 } \pi _ { t } ^ { ( k ) } r ^ { ( k ) } \Big ) \odot h _ { t } \right)\tag{5}
$$

After estimating the hidden state, we linearly transformed it to predict semantic contents $( y _ { t + 1 } \in \mathbb { R } ^ { 1 0 } )$ of the scene at time $t + 1 ( \mathbf { W _ { y } } \in \mathbb { R } ^ { 1 0 \times 1 0 0 } ; \ \mathbf { b _ { h } } \in \dot { \mathbb { R } } ^ { 1 0 } )$ .

$$
\hat { y } _ { t + 1 } = \mathbf { W } _ { y } h _ { t } + \mathbf { b } _ { y }\tag{6}
$$

The model parameters were initialized as follows. $\mathbf { W _ { v } } , \mathbf { W _ { a } } , \mathbf { W _ { v } ^ { ( k ) } } , \mathbf { W _ { a } ^ { ( k ) } } , \mathbf { W _ { y } }$ were initialized using xavier (glorot) uniform initialization with a gain of 1.0. $\mathbf { W _ { 0 } } , \mathbf { W _ { h } } , \mathbf { W _ { h } ^ { ( k ) } }$ were initialized using orthogonal initialization. The bias terms, $\mathbf { b _ { h } }$ and ${ \bf b _ { y } }$ , were initialized to zero. $\mathbf { m _ { k } }$ and $\mathbf { n _ { k } }$ were initialized from a standard Gaussian distribution $\mathcal { N } ( 0 , 1 )$ . In the output modulation condition, $r ^ { ( k ) }$ were not model parameters, but they were randomly generated from a standard Gaussian distribution $\mathcal { N } ( 0 , 1 )$ .

The mean squared error loss was computed between the predicted $( \widehat { y } _ { t + 1 } )$ and observed $( y _ { t + 1 } )$ next scene embeddings and the summed loss across all scenes within an episode was backpropagated to compute gradients, which were used to update model parameters via the Adam optimizer. The learning rate was initialized to 5e-4 and decayed following a cosine annealing schedule, reaching a minimum value of 5e-5 at the end of training. One training iteration comprised watching all 17 episodes, which means 17 sets of parameter updates took place in one iteration. The model was repeatedly trained for 100 iterations. Learning regimes for the EM-RNN are described in Section $_ { \mathrm { A } . 8 }$

## A.4 MODEL TESTING AND BAYESIAN CONTEXT INFERENCE

The model was tested on episode 1, the episode that was unseen during training and what human participants watched during fMRI. Unlike during training when context signal π was given, the model did not receive π during test and was tasked to infer it at every time step. The model was designed to perform online inference over four contexts using a recursive Bayesian update, where the posterior from one step becomes the prior for the next step.

Under a sticky hidden Markov model framework, the prior at each timestep was obtained by propagating the previous posterior through a transition matrix T. This encourages the same context to persist to the next time step $( \gamma = 0 . 9 )$ ). The prior was initialized as $\pi _ { 0 } = [ 0 . 2 5$ , 0.25, 0.25, 0.25].

$$
\operatorname* { P r } ( \pi _ { t } = k ) = \sum _ { j = 1 } ^ { K } \operatorname* { P r } \left( \pi _ { t - 1 } = j | v _ { 1 : t - 1 } , a _ { 1 : t - 1 } , y _ { 2 : t } \right) T _ { j k } , \qquad T _ { j k } = \left\{ \begin{array} { l l } { \gamma } & { \mathrm { i f ~ } j = k } \\ { \displaystyle \frac { 1 - \gamma } { K - 1 } } & { \mathrm { i f ~ } j \neq k } \end{array} \right. , K = 4\tag{7}
$$

For each of the four contexts $\left( \pi = \left[ 1 , 0 , 0 , 0 \right] , \left[ 0 , 1 , 0 , 0 \right] , \left[ 0 , 0 , 1 , 0 \right] , \left[ 0 , 0 , 0 , 1 \right] \right)$ , the model simulated a nextscene prediction $\widehat { y } _ { t + 1 } ^ { ( k ) }$ from the context-specific hidden state $h _ { t - 1 } ^ { ( k ) }$ . The context-specific hidden state is obtained by simulating the trajectory under the assumption that context k was held throughout. The resulting prediction $\hat { y } _ { t + 1 } ^ { ( k ) }$ was compared to the observed next scene semantics $y _ { t + 1 }$ , and the log-likelihoods of the four contexts were approximated as the scaled negative mean squared error, consistent with a Gaussian observation model. This formulation assigns higher likelihood to contexts with lower prediction error.

$$
\log \operatorname* { P r } \left( y _ { t + 1 } \left| v _ { t } , a _ { t } , h _ { t - 1 } ^ { ( k ) } , \pi _ { t } = k \right. \right) = - \frac { \beta } { 2 } \cdot \operatorname { M S E } \left( y _ { t + 1 } , \hat { y } _ { t + 1 } ^ { ( k ) } \right) , \quad \beta = 1 0\tag{8}
$$

For the EM-RNN, the model simulation is also based on the context-specific memory retrieval $m _ { t - 1 } ^ { ( k ) }$ , which makes the left-hand side of the equation logPr $\Big ( y _ { t + 1 } \Big | v _ { t } , a _ { t } , h _ { t - 1 } ^ { ( k ) } , m _ { t - 1 } ^ { ( k ) } , \pi _ { t } = k \Big )$

The posterior was computed using Bayes’ rule in log space, where the log prior and log likelihood were summed and normalized using the log-sum-exp operation for numerical stability.

$$
\operatorname* { P r } \left( \pi _ { t } = k \mid v _ { 1 : t } , a _ { 1 : t } , y _ { 2 : t + 1 } \right) \propto \operatorname* { P r } \left( y _ { t + 1 } \mid v _ { t } , a _ { t } , h _ { t - 1 } ^ { ( k ) } , \pi _ { t } = k \right) \cdot \operatorname* { P r } \left( \pi _ { t } = k \right)\tag{9}
$$

The posterior distribution of Equation (9) served as a context signal at each time step during test. The hidden state (and the retrieved memory for the EM-RNN) was not the context-specific one but the actual hidden state rolled in from the previous time step.

$$
\hat { y } _ { t + 1 } = f ( v _ { t } , a _ { t } , h _ { t - 1 } , \pi _ { t } )\tag{10}
$$

Again, $m _ { t - 1 }$ was added for the EM-RNN.

In Song et al. (2026a), participants watched the same episode 1 of This Is Us, but not in its original order. The episode was segmented into 48 events (mean duration $5 2 \pm 1 2 \mathrm { s } )$ and presented in three different scrambled orders (12 participants per condition). The events were scrambled in a semi-counterbalanced order so that order effects could be canceled out when averaging participants’ data across groups. To match this human experience, we also tested the model on these three scrambled versions of the episode. We then unscrambled the orders back to their original, and averaged the results. To see how results evolve over training, we tested the model at the end of every training iteration. Model train and test were repeated across 20 different random seeds, and we report their mean and standard deviation for all findings.

## A.5 PERFORMANCE

Next scene prediction task performance was assessed using loss and accuracy during train and test respectively. Loss was calculated as the sum of mean squared error across all scenes (MSE $( y _ { t + 1 } , \widehat { y } _ { t + 1 } ) ,$ ) within each episode. The loss across 17 training episodes were averaged to indicate the mean loss for each training iteration. Accuracy was calculated as the average Pearson’s correlation between predicted and observed semantic embeddings of the next scenes. Correlation values were Fisher’s z-transformed before averaging and transformed back to $r -$ which is a consistent procedure throughout this paper. We only visualize test accuracy (r) in this paper but all four measures were assessed by the authors during model building.

Bayesian context inference performance was measured using an accuracy metric (%). For each scene, the inferred context with the highest posterior probability was compared to the ground-truth context. We calculated the proportion of scenes that matched. Only the scenes that follow a single storyline were selected, not the ones in which multiple storylines merge.

## A.6 MODEL–BRAIN REPRESENTATION SIMILARITY

Model–brain representation similarity was computed by comparing the scene-by-scene RSMs generated from the RNNs’ hidden states and the voxelwise BOLD activities in 200 cortical regions (Schaefer et al., 2018). Thirtythree participants’ brain RSMs were compared to the model RSM that was averaged across 20 seeds, for each brain region and each model condition.

## A.7 FORWARD SIMULATION

With the low- and full-rank WM modulation models that were fully trained (100th iterations), we forward simulated each model without input, starting from a zero-vector hidden state and a zero-vector context signal. Importantly, in each of the four contexts, we added a context signal in steps of 0.001 until the context signal reached 1 $( \mathbf { e . g . } , \pi = [ 0 , 0 , 0 , 0 ] , [ 0 . 0 0 1 , 0 , 0 , 0 ] , [ 0 . 0 0 2 , 0 , 0 , 0 ] , \dots , [ 1 , 0 , 0 , 0 ] )$ . That is, we iterated $h _ { t } = f ( h _ { t - 1 } , \pi _ { t } )$ to update the hidden states, creating a 1001-step trajectory of the hidden states from a moment when context signal was not present to a moment when context was fully specified.

Tortuosity was estimated by calculating the summed Euclidean distance from one point to the next across 1001 steps, divided by a single Euclidean distance from the start point to the end point. If the forward simulation followed a straight line, then the stepwise progression would match the linear distance from the start to the end. To estimate the dimensionality, we concatenated the 1001-step trajectories of the hidden states of the four contexts and conducted a PCA to reduce 100-dimensional hidden units to its top 20 PCs. The normalized explained variances of the two WM modulation models were compared. Orthogonality of the four contexts was assessed in two ways. First, we extracted the four end points of the forward simulation corresponding to the four contexts, and computed cosine similarity of the pairwise end points. We took the absolute value of the cosine similarity such that values near zero indicate higher orthogonality. The second method did not require forward simulation. We extracted four $\mathbf { W _ { h } ^ { ( k ) } }$ estimates representing the four contexts and computed cosine similarity of their pairs. To see how orthogonality changes over training, we estimated this across all 100 training iterations.

## A.8 CONTEXT-MODULATED EM-RNN

The RNN was augmented with an EM buffer, where an RNN can write to (i.e., encoding) and read from (i.e., retrieval) an external memory storage. This model architecture—as well as the model descriptions written below— are adapted from Song et al. (2026b). Visual $( v _ { t } \in \mathbb { R } ^ { 5 0 } )$ and auditory $( a _ { t } \in \mathbb { R } ^ { 5 0 } )$ embeddings were concatenated as input, which was linearly transformed $( \mathbf { W _ { k } } , \mathbf { W _ { q } } \in \mathbb { R } ^ { 1 0 0 \times 1 0 0 } )$ into keys $( k _ { t } \in \mathbb { R } ^ { 1 0 0 } )$ , which represent memory addresses and are encoded in the EM, and queries $( q _ { t } \in \mathbb { R } ^ { 1 0 0 } )$ , which survey the stored keys based on pattern similarity.

$$
k _ { t } = \mathbf { W } _ { k } [ v _ { t } , a _ { t } ]\tag{11}
$$

$$
q _ { t } = \mathbf { W } _ { q } [ v _ { t } , a _ { t } ]\tag{12}
$$

The hidden states $( h _ { t } )$ were generated from the RNN following either the baseline RNN of Equation (1) or the low-rank WM modulated RNN of Equation (4). These hidden states served as values $( v _ { t } = h _ { t } )$ , representing memory contents and are also encoded in an EM at every time step. In addition to these baseline features, context signal π was also encoded and stored in the EM independently. Therefore, the key, value, and context of every time step were stored in the EM. The maximum number of memories (M) was set to 1,000, which exceeded the 599 time steps of the test data. Memories were not flushed during training, meaning the model could retain memories of other episodes. However, the EM was flushed at the start of testing, which means the model only retained memories within the episode when it was watching episode 1.

$$
\operatorname { E M } ( K ) = [ k _ { t - 1 } , k _ { t - 2 } , \dots , k _ { t - M + 1 } ]\tag{13}
$$

$$
\operatorname { E M } ( V ) = \left[ v _ { t - 1 } , v _ { t - 2 } , \dotsc , v _ { t - M + 1 } \right]\tag{14}
$$

$$
\operatorname { E M } ( C ) = [ \pi _ { t - 1 } , \pi _ { t - 2 } , \ldots , \pi _ { t - M + 1 } ]\tag{15}
$$

At every time step, the query $q _ { t }$ surveyed all stored keys in the EM (“content similarity”) and the current context $\pi _ { t }$ surveyed all stored contexts in the EM (“context similarity”) based on the following equations. D denotes the size of either the key/query or context.

$$
k q _ { t } = \frac { q _ { t } \mathrm { E M } ( K ) ^ { \top } } { \tau \sqrt { D } } , \quad D = 1 0 0\tag{16}
$$

$$
c t x _ { t } = \frac { \pi _ { t } \operatorname { E M } ( C ) ^ { \top } } { \tau \sqrt { D } } , \quad D = 4\tag{17}
$$

Retrieval weights were then calculated by combining content similarity $( k q _ { t } )$ and context similarity (ctx<sub>t</sub>) as an element-wise product. The softmax function was applied to make the retrieval weights sum to 1. Its temperature τ was set to 0.1 to enforce sparsity in memory retrieval. We also tested variations where temperature was set to $\tau = 0 . 5$ and 1.0, and a case in which softmax was not applied.

$$
a t t n _ { t } = { \mathrm { s o f t m a x } } \left( k q _ { t } \odot { \mathrm { r e l u } } ( c t x _ { t } ) , \tau = 0 . 1 \right)\tag{18}
$$

In the original EM-RNN without context modulation, retrieval weights were estimated only based on content similarity $( k q _ { t } )$ .

$$
a t t n _ { t } = \mathrm { s o f t m a x } \left( k q _ { t } , \tau = 0 . 1 \right)\tag{19}
$$

The retrieval weights were applied to values stored in the EM. The retrieved memory at time t is thus denoted as $m _ { t } \in \mathbb { R } ^ { 1 0 0 }$ , which is the summation of past values, weighted by the retrieval weights. Here, note that the values are either context-modulated (WM+EM and WM modulation) or not (no modulation) depending on conditions.

$$
m _ { t } = \mathrm { E M } ( V ) a t t n _ { t } ^ { \top }\tag{20}
$$

The $h _ { t }$ representing the current scene and $m _ { t }$ representing the retrieved memory were integrated, $h _ { t } \times \alpha + m _ { t } \times$ $( 1 - \alpha )$ , with α set to 0.5. Model parameters $\bar { \mathbf { W } } _ { \mathbf { y } } \in \mathbb { R } ^ { 1 0 \times 1 0 0 }$ and $\mathbf { b } _ { \mathbf { y } } \in \mathbb { R } ^ { 1 0 }$ transformed this integrated vector to predict the semantic embedding of the next time step $y _ { t + 1 }$

$$
\hat { y } _ { t + 1 } = \mathbf { W } _ { y } \left( h _ { t } \times \alpha + m _ { t } \times \left( 1 - \alpha \right) \right) + \mathbf { b } _ { y }\tag{21}
$$

We followed the same initialization procedure as with the RNN for the EM-RNN. Additionally, parameters $\mathbf { W _ { k } }$ and $\mathbf { W _ { q } }$ were initialized using xavier (glorot) uniform initialization with a gain of 1.0.

Training objective was the same as previously described. The learning rate was initialized to 5e-4 and decayed following a cosine annealing schedule, reaching a minimum value of 1e-4 at the end of training. We conducted training up to 200 iterations for the EM-RNN.

## A.9 MODEL–HUMAN RETRIEVAL SIMILARITY

At test, attn was computed at every time step, reflecting the retrieval weights assigned to previous scenes. Extracting these weights across all scenes in episode 1 resulted in $\mathrm { a 5 9 9 \times 5 9 9 }$ scene-by-scene triangular matrix. We then summarized this into a $4 8 \times 4 8$ event-by-event matrix by averaging the retrieval weights across all scenes belonging to each pair of the 48 events. This matrix was computed separately for each of the three scrambled-order groups and averaged across groups as well as across 20 random seeds, which became a single matrix representing the model’s memory retrieval. While attn is a final retrieval weight combining content and context similarities with additional selectivity constraints, we can decompose it by examining content similarity and context similarity separately. We created a $4 8 \times 4 8$ event-by-event matrix for $k q ,$ representing the pattern similarity between current query and past keys, and for ctx, representing the pattern similarity between current and past contexts. For both matrices, queries, keys, and contexts were first averaged across all scenes belonging to each of the 48 events. These model retrieval matrices were each correlated with the human retrieval matrix using Pearson’s r.

For the WM+EM and WM modulation conditions, we ran an additional set of model tests in which context was provided directly at test rather than inferred via Bayesian inference. Model–human retrieval similarity was computed the same way using attn. We additionally computed the partial correlation between model and human retrieval matrices, while controlling for context similarity ctx.

## A.10 AN ALTERNATIVE MODEL IN WHICH KEY AND QUERY TRANSFORMATIONS ARE MODULATED BY CONTEXT

As an alternative account, we context-modulated input-to-key and input-to-query transformations, instead of encoding context and retrieving memories based on context similarity. We applied the same low-rank modulation approach to $\mathbf { W _ { k } }$ and $\mathbf { W _ { q } }$ , as we have done with $\mathbf { W _ { h } }$ of the RNN. Apart from replacing Equations (11)–(12) with Equations $( 2 2 ) \AA - ( 2 3 )$ , we adopted the original key-query EM system without context modulation.

$$
k _ { t } = \left( \mathbf { W } _ { k , 0 } \odot \sum _ { i = 1 } ^ { 4 } \pi _ { t } ^ { ( i ) } \mathbf { m } _ { k , i } \mathbf { n } _ { k , i } ^ { \top } \right) \left[ v _ { t } , a _ { t } \right]\tag{22}
$$

$$
q _ { t } = \left( \mathbf { W } _ { q , 0 } \odot \sum _ { i = 1 } ^ { 4 } \pi _ { t } ^ { ( i ) } \mathbf { m } _ { q , i } \mathbf { n } _ { q , i } ^ { \top } \right) \left[ v _ { t } , a _ { t } \right]\tag{23}
$$

## B SUPPLEMENTARY FIGURES AND TABLES

![](images/df72939ced72d679fff40d237502665ddf46ba5ae071dfc271e55d1172efa818.jpg)

![](images/e430a97b4a8238faa8d3a8d35bd46602efc8f662c8c8a45be6bd63e19894748c.jpg)  
Figure S1: Model comparison. (A) Original EM-RNN without context modulation. The model designed by Song et al. (2026b) is visualized in matching schematics as Figure 1A. Three critical differences are as follows: i) The model uses a gated recurrent unit (GRU) RNN. ii) The model receives only visual embeddings of a scene to predict visual embeddings of the next scene. iii) Neither the WM nor EM is modulated by context. This means the EM buffer does not represent context explicitly, so its memory retrieval solely relies on the key-value system or the content similarity. (B) Full WM modulation. Four independent recurrent connectivity matrices $( \mathbf { W _ { h } ^ { ( k ) } } \in \mathbb { R } ^ { 1 0 0 \times 1 0 0 } )$ are summed, weighted by the context probability π. This schematic is visualized in comparison to Figure 1C.

low- vs. full-rank WM modulation  
![](images/80f44be4b91b9c5c29b71ed8a0e26c73b173d5ee825317e9c59d82e35eec5fc0.jpg)  
Figure S2: Comparison of the model–brain RSM similarity between low-rank vs. full WM modulation conditions. Scene-by-scene RSM was computed for each of the 200 cortical regions, respectively in 33 participants’ data. The model’s RSM was computed from the low-rank and full WM modulation conditions respectively, and RSMs across 20 seeds were averaged. The model–brain RSM similarity was computed using Pearson’s correlations, which were Fisher’s z-transformed. Thirty-three z values were compared between the two model conditions using paired t-tests, repeatedly across 200 regions. The figure shows significant cortical regions’ t statistics, after FDR-correction of the p values. All significant brain regions show positive t statistics, meaning the model–brain RSM similarity was higher in the low-rank compared to the full WM modulation condition.

![](images/f7770adac954c4c9192f568579bf105a489f49d45b70bc00c35e8d2c761c6641.jpg)

B  
Context congruence  
![](images/259266586ee73abe1d9b887dc8991aa502a3ec570763a87345e4ba1f48887aa8.jpg)  
Figure S3: Validation of memory selectivity and context congruence in the WM+EM modulation model compared to the WM modulation model. (A) Memory selectivity. For each model’s memory retrieval matrix, we calculated the proportion of scene pairs falling into bins, [0, 1e-5), [1e-5, 1e-4), [1e-4, 1e-3), [1e-3, 1e-2), [1e-2, 1e-1), [1e-1, 1], and plotted the log of these proportions. For simplicity, only the upper bound of each bin is labeled on the x-axis. Compared to the WM modulation condition, the WM+EM modulation condition includes far fewer values in the 1e-5 to 1e-1 bins. The proportion is relatively higher in the [1e-1, 1] bin, while still smaller than the WM modulation condition. This indicates that memory retrieval is more selective in the WM+EM modulation condition. (B) Context congruence. For scene pairs in the final bin, corresponding to memories that are actually retrieved after the softmax, we averaged their degree of context similarity respectively for the two conditions. As expected, the WM+EM modulation model retrieves memories more within the same context than the WM modulation model.

![](images/f1a3bd499672bbdd48d9cda4cbe01b382c2f2fb69de34a911e8da4e1b97ded26.jpg)

![](images/c29e0375431180eabfce620cf46a5efcdbdc6045611e6e2938fb3ca3845fa2ab.jpg)

![](images/ec004465c4a35ced75b6903a893711eb858fcb85a114cd968e6127b1f970e01e.jpg)

![](images/62f15a599753f9ee4f5192f2a655750faa5f50463cf4729b87305ef3950f8e74.jpg)  
Figure S4: Model–human retrieval similarity with variations in softmax. In the main analysis, the model’s retrieval matrix was created using a softmax of $\tau = 0 . 1$ (Figure 4C; copied on the $l e f t$ of this figure). We repeated the same analysis but with varying softmax temperature, with larger τ indicating decreased retrieval selectivity. The right panel shows the result without softmax applied in the analysis, thus the model’s retrieval is not at all selective.

![](images/da17208616a2c2f5f3df408434619a6ccf6a6440823912f30053065c5e3b926e.jpg)  
Figure S5: Bayesian context inference accuracy of the WM+EM modulation condition at test, across training iterations. Accuracy was computed as the proportion of scenes in which the inferred context matches the ground-truth context.

Table S1: Comparison of the RNN modulation conditions. Next-scene prediction accuracy is the average Pearson’s correlation between the predicted and observed semantic embeddings of the next scenes, measured either on the training dataset or on the held-out test dataset. Bayesian context inference accuracy is the proportion of scenes in which the inferred context matches the ground-truth context. Model–brain representation similarity is the Pearson’s correlation between fMRI participants’ brain RSMs and the model’s hidden-state RSMs, averaged across 33 participants and 200 cortical regions. All measures were computed after 100 training iterations.
<table><tr><td></td><td>Input</td><td>Low-rank WM</td><td>Full WM</td><td>Output</td><td>None</td></tr><tr><td>Next-scene prediction accuracy at train (r)</td><td> $0 . 8 5 0 \pm 0 . 0 0 6$ </td><td> $0 . 8 0 1 \pm 0 . 0 1 0$ </td><td> $\mathbf { 0 . 8 6 0 \mathop { \pm } 0 . 0 0 6 }$ </td><td> $0 . 8 1 0 \pm 0 . 0 1 0$ </td><td> $0 . 7 7 2 \pm 0 . 0 0 7$ </td></tr><tr><td>Next-scene prediction accuracy at test (r)</td><td> $0 . 5 1 4 \pm 0 . 0 1 8$ </td><td> $0 . 5 8 3 \pm 0 . 0 1 9$ </td><td> $0 . 5 5 5 \pm 0 . 0 2 5$ </td><td> ${ \bf 0 . 5 9 7 \pm 0 . 0 2 3 }$ </td><td> $0 . 5 1 4 \pm 0 . 0 1 7$ </td></tr><tr><td>Bayesian context inference accuracy (%)</td><td> $5 8 . 7 3 \pm 2 . 1 9$ </td><td> $6 0 . 6 1 \pm 5 . 3 2$ </td><td> $5 3 . 2 3 \pm 4 . 1 5$ </td><td> ${ \pm } \pm \mathbf { \delta } . 4 \mathbf { 0 } \pm 2 . 6 6$ </td><td></td></tr><tr><td>Model-brain representation similarity (r)</td><td> $0 . 0 0 3 9 \pm 0 . 0 0 1 1$ </td><td> $\mathbf { 0 . 0 0 8 7 \pm 0 . 0 0 2 0 }$ </td><td> $0 . 0 0 8 0 \pm 0 . 0 0 1 9$ </td><td> $0 . 0 0 5 3 \pm 0 . 0 0 1 3$ </td><td> $0 . 0 0 6 6 \pm 0 . 0 0 1 6$ </td></tr></table>

Table S2: Comparison of the EM-RNN modulation conditions. WM modulation was implemented as a low-rank modulation of the RNN’s recurrent connectivity, and EM modulation was implemented by combining context similarity to compute retrieval weights. The measures are defined as in Table S1. Model–human retrieval similarity is the Pearson’s correlation between the event-by-event (48 × 48) human and model retrieval matrices.
<table><tr><td></td><td>WM+EM</td><td>WM</td><td>None</td></tr><tr><td>Next-scene prediction accuracy at train (r)</td><td> $0 . 7 9 9 \pm 0 . 0 1 1$ </td><td> $\mathbf { 0 . 8 0 4 \ : \pm 0 . 0 1 2 }$ </td><td> $0 . 7 7 8 \pm 0 . 0 0 8$ </td></tr><tr><td>Next-scene prediction accuracy at test (r)</td><td> $\mathbf { 0 . 5 6 5 \pm 0 . 0 3 0 }$ </td><td> $0 . 4 9 4 \pm 0 . 0 3 0$ </td><td> $0 . 4 1 4 \pm 0 . 0 4 3$ </td></tr><tr><td>Bayesian context inference accuracy (%)</td><td> ${ \bf 6 2 . 1 0 \pm 4 . 7 3 }$ </td><td> $6 1 . 2 2 \pm 4 . 6 3$ </td><td></td></tr><tr><td>Model-brain representation similarity (r)</td><td> $\mathbf { 0 . 0 0 3 2 \pm 0 . 0 0 0 9 }$ </td><td> $0 . 0 0 2 7 \pm 0 . 0 0 0 9$ </td><td> $0 . 0 0 2 0 \pm 0 . 0 0 0 7$ </td></tr><tr><td>Model–human retrieval similarity (r)</td><td> $\mathbf { 0 . 2 8 3 \pm 0 . 0 5 4 }$ </td><td> $0 . 2 0 7 \pm 0 . 0 5 4$ </td><td> $0 . 2 6 5 \pm 0 . 0 3 9$ </td></tr></table>