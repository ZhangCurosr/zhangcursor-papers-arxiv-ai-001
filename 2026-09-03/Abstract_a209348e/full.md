# C<sub>o</sub>d<sub>e</sub>b<sub>oo</sub>k A<sub>ge</sub>nt<sub>:</sub> Am<sub>o</sub>rti<sub>ze</sub>d T<sub>opo</sub>l<sub>ogy</sub> D<sub>es</sub>i<sub>g</sub>n f<sub>o</sub>r LLM M<sub>u</sub>lti<sub>-</sub>A<sub>ge</sub>nt S<sub>ys</sub>t<sub>e</sub>m<sub>s</sub>

<sub>Jinxi</sub> <sub>Yu</sub>∗<sub>,</sub> <sub>Yubei</sub> <sub>Li</sub>∗<sub>,</sub> <sub>Eric</sub> <sub>Hanchen</sub> <sub>Jiang</sub>∗<sub>,</sub> <sub>Zhi</sub> <sub>Zhang,</sub> <sub>Dong</sub> <sub>Liu,</sub> W<sub>e</sub>n<sub>x</sub>i<sub>ao</sub> Zh<sub>ao,</sub> L<sub>ev</sub>in<sub>a</sub> Li<sub>,</sub> K<sub>a</sub>i<sub>-</sub>W<sub>e</sub>i Ch<sub>a</sub>n<sub>g,</sub> Yin<sub>g</sub> Ni<sub>a</sub>n W<sub>u</sub>

University of California, Los Angeles

## Abstract

Adapting the communication topology of an LLM multiagent system to each query improves both accuracy and efficiency, yet current designers treat this as conditional graph generation: a variational, autoregressive, or difusion decoder searches the N×N adjacency space, and a graph-network proxy trained on utility and a structural cost such as edge count ranks the sampled candidates. We argue that this formulation is misaligned with the problem. Empirically, topologies that survive a reward filter collapse to about six distinct graphs even when the codebook capacity grows from 8 to 64; edge count is negatively correlated with measured token consumption (Pearson r ≈ −0.4), so sparsifying the graph makes inference more expensive; and a message-passing scorer over agent-profile nodes is adjacency-invariant whenever agents share a profile—the default configuration of published benchmarks—so it cannot rank candidates at all in that regime. These three facts motivate Codebook Agent: a vectorquantized autoencoder compresses successful topologies into a query-independent 16-entry codebook; a reward-weighted MLP maps the query embedding to a distribution over codes; and an MLP proxy that reads the flattened adjacency, regressed on measured utility and per-task normalized token cost, reranks the top decoded candidates in a single batched forward pass. With no iterative search and no message passing at test time, Codebook Agent is the most accurate method on all six benchmarks we compare (84.6 average against 83.0 for the strongest prior designer), emits a topology in 2.4 ms, and uses 21.9–33.2% fewer LLM tokens. Our code is available here: https://github.com/jinxiy1104/CodebookAgent.

## 1 Intr<sub>o</sub>d<sub>uc</sub>ti<sub>o</sub>n

Multi-agent systems built from LLM agents solve reasoning, question answering, and coding tasks by exchanging intermediate outputs over a communication topology (Wu et al. 2024; Hong et al. 2024; Qian et al. 2024; Du et al. 2024). That topology is not bookkeeping: it decides which agents see which messages, and both accuracy and token consumption move with it (Qian et al. 2025; Zhuge et al. 2024; Zhang et al. 2025b). The field has therefore progressed from handdesigned structures to learned ones, and most recently to query-conditioned generators that emit a fresh topology for every input (Zhang et al. 2025c; Jiang et al. 2026).

The latest designers share a natural but costly recipe. A variational (Zhang et al. 2025c), autoregressive (Li et al.

![](images/189bffb4fce7cef98ea9e57c22c53a1f7c340a64bf4e040eafd22ba3db965366.jpg)  
Figure 1: Three regimes of topology design. (Top) A fixed topology is reused for every query: zero design cost, but no single graph fits every task. (Middle) Query-conditioned generators adapt the graph via an iterative loop that samples and scores K candidates at each of T steps, typically with a proxy trained on utility and edge count. (Bottom) Codebook Agent keeps per-query adaptation and removes the loop: the design space is discretized once ofline into a codebook, a query embedding selects codes, and one batched proxy call under a token-based cost objective returns the winner.

2026), or difusion decoder (Jiang et al. 2026) searches the N×N adjacency space; a graph network over agent-profile nodes then ranks candidates with a utility head and a structural cost head, typically the edge count |E| (Zhuge et al. 2024; Zhang et al. 2025b,c; Jiang et al. 2026). The recipe looks principled—generate, then score—yet it quietly assumes that the useful design space is large enough to need a generative decoder, that |E| tracks inference cost, and that message passing can tell two topologies apart. Our measurements reject all three assumptions.

Figure 1 sketches the consequence. First, the useful design space is a short list, not a manifold: topologies that solve their task fall into about six distinct codes even as codebook capacity grows from 8 to 64, and the best fixed topology stays within 1.4 accuracy points of every generated one we measure—so capacity spent modeling adjacency is spent where the problem does not live. Second, the structural cost surrogate is inverted: edge count correlates with measured tokens at $r \approx - 0 . 4 .$ , because sparse communication yields longer completions, so minimizing |E| maximizes the bill it was introduced to reduce. Third, on the homogeneous teams that dominate published benchmarks, a message-passing scorer over profile nodes is adjacencyinvariant: every candidate receives the same score, the ranking does not exist, and hundreds of guided difusion steps reproduce a constant.

If the problem is to choose among a few graphs under a measured cost, the matching designer is not another decoder. Codebook Agent therefore amortizes topology design into three feed-forward pieces: a vector-quantized autoencoder (van den Oord, Vinyals, and Kavukcuoglu 2017) that indexes successful topologies into 16 codes, a reward-weighted MLP that maps the query embedding to a distribution over those codes, and an MLP proxy that reads the flattened adjacency and is regressed on measured utility and per-task normalized token cost. At test time we decode the top codes, score them in one batched proxy call, and execute the winner—2.4 ms, with no sampling loop and no message passing.

## Our contributions:

❶ We characterize topology design rather than assume its form: the reward-surviving space collapses to about six graphs independently of model capacity, the common structural cost surrogate is inverted w.r.t. measured tokens, and message-passing proxies are topology-blind on homogeneous teams.

❷ We propose Codebook Agent, the minimal designer these facts admit, a discrete codebook, a reward-weighted code predictor, and one batched pass of an execution-grounded proxy, with no iterative search and no message passing.

❸ On six benchmarks and two LLM backends, against single-agent prompting, multi-agent collaboration, and learned topology designers, Codebook Agent is the most accurate method on every benchmark of Table 1, generates topologies in 2.4 ms, and reduces token consumption by 21.9–33.2% (Tables 1 and 2).

## 2 R<sub>e</sub>l<sub>a</sub>t<sub>e</sub>d W<sub>or</sub>k

Communication topologies for LLM agents. Early frameworks fix the topology by hand (Hong et al. 2024; Qian et al. 2024; Wu et al. 2024; Chen et al. 2024b; Du et al. 2024). A second line optimizes it: GPTSwarm learns edge distributions with policy gradients (Zhuge et al. 2024), DyLAN selects agents dynamically (Liu et al. 2024), searchbased systems explore agentic designs ofline (Hu, Lu, and Clune 2025; Zhang et al. 2025d), pruning methods sparsify a fixed structure to save tokens (Zhang et al. 2025b; Wang et al. 2025), and Optima trains the agents for eficiency (Chen et al.

2025). Closest to us are query-conditioned generators (Zhang et al. 2025c; Jiang et al. 2026). We share their problem statement, one topology per query, and difer in what we assume about it: they model the adjacency space as something to be generated and scored by a graph network, and we show it is a short list to be selected from and scored by measured cost.

Graph generation and discrete latents. Deep graph generators are predominantly iterative, whether autoregressive (You et al. 2018) or score-based (Jo, Lee, and Hwang 2022; Vignac et al. 2023; Zhou, Wang, and Zhang 2024), inheriting a sampling cost (Ho, Jain, and Abbeel 2020) that faster samplers reduce but do not remove (Song, Meng, and Ermon 2021); one-shot VAE decoders exist for small graphs (Simonovsky and Komodakis 2018). We need one small graph per query under a latency constraint, so we discretize the output space with vector quantization (van den Oord, Vinyals, and Kavukcuoglu 2017; Razavi, van den Oord, and Vinyals 2019) instead of sampling from a continuous process. VQ-Graph tokenizes local structure to improve GNN-to-MLP distillation (Yang et al. 2024; Zhang et al. 2022; Tian et al. 2023; Wu et al. 2023; Hinton, Vinyals, and Dean 2015); we borrow its structure-token regularizer, but our proxy has no teacher to distill from: on homogeneous teams a messagepassing teacher is constant per query, so we train the MLP directly on measured rewards.

Cost of multi-agent inference. Multi-agent pipelines multiply LLM calls, and returns diminish or invert as calls grow (Li et al. 2024; Chen et al. 2024a; Smit et al. 2024), with many observed failures attributable to coordination rather than single-agent ability (Cemri et al. 2025). Cost-aware model selection addresses the single-call case (Chen, Zaharia, and Zou 2024). These results motivate treating measured tokens, not a structural surrogate, as the cost objective of topology design.

## 3 Pr<sub>o</sub>bl<sub>e</sub>m S<sub>e</sub>t<sub>up a</sub>nd B<sub>ac</sub>k<sub>g</sub>r<sub>ou</sub>nd

Topology design. A team of N agents has profile texts $p _ { 1 } , \ldots , p _ { N }$ embedded as $x _ { i } = E ( p _ { i } ) \in \mathbb { R } ^ { d }$ by a frozen sentence encoder $( d = 3 8 4 )$ . A topology is a directed adjacency matrix $A \in \{ 0 , 1 \} ^ { N \times N }$ without self-loops, where $A _ { i j } = 1$ shows the output of agent i to agent $j ,$ and a decision node aggregates the final answers following GDesigner (Zhang et al. 2025c). Executing the team on query q under A returns a utility u $( A , q ) \in \{ 0 , \bar { 1 } \}$ and a token cost $\tau ( A , q )$ . Writing $c = E ( q )$ for the query embedding, the goal is a generator mapping c to a topology maximizing $R = u - \lambda \tilde { \tau }$ , with τ˜ a normalized cost and $\lambda = 0 . 1$ . All methods in this paper are trained on the same records: per benchmark, 50 training tasks executed under 6 fixed topologies (complete, chain, star, and three Erdos-Renyi samples) with real LLM agents, giving 300 tuples $( A _ { j } , c _ { j } , u _ { j } , \tau _ { j } )$

Two design axes. A query-conditioned designer is determined by two independent choices, which we vary separately in the experiments. Axis A, the candidate generator, maps c to one or more topologies; the incumbent choice is an iterative or continuous decoder over the adjacency space, variational (Zhang et al. 2025c), autoregressive (Li et al. 2026), or a difusion reverse process with T steps and K candidates per step (Jiang et al. 2026), which we measure at 301 to 396 ms per query. Axis B, the scorer, ranks candidates; the incumbent choice, shared across the learned-design literature (Zhuge et al. 2024; Zhang et al. 2025b,c; Jiang et al. 2026), is a message-passing network over a graph whose nodes carry the profile embeddings $x _ { i }$ and whose edges are the candidate A:

![](images/4773a44f5c15393b01436c4678f9fb4bb734e796ba8c36f0d6db2bd6616b28ed.jpg)  
Figure 2: Overview of Codebook Agent. Left: ofline collection executes fixed topologies on the training split and logs $( A , c , u , \tau )$ for every (query, topology) pair. Middle: (1) a VQ autoencoder compresses successful topologies $( u > 0 . 5 )$ into a query-independent codebook; (2) an MLP predictor $p _ { \theta } ( k \mid c )$ is fit to the reward-weighted soft targets of Eq. (6); (3) an MLP proxy $f _ { \phi }$ is regressed on measured utility and per-task normalized token cost (Eqs. (8)–(10)), with an auxiliary structure-token head used in training only. Right: at test time the top M codes are decoded, deduplicated, and scored in one batched proxy call; the candidate maximizing uˆ − λcˆ (Eq. (11)) is executed. Dashed arrows are training-only; solid arrows are the frozen test-time path (no LLM calls inside topology generation).

$$
f _ { \mathrm { g n n } } ( A , c ) = h \bigl ( \mathrm { p o o l } \bigl ( \mathrm { M P N N } ( A , \{ x _ { i } \} , c ) \bigr ) \bigr ) \in \mathbb { R } ^ { 2 } ,\tag{1}
$$

with a two-dimensional head regressed onto utility and a structural cost label, most commonly the edge count |E|. We implement Eq. (1) as a GAT (Veličković et al. 2018) and treat it as one level of Axis B, so it can be swapped into our own pipeline with everything else held fixed. The introduction’s three observations—a collapsed design space, an inverted cost surrogate, and adjacency-blind message passing on homogeneous teams—determine how we instantiate both axes below.

## 4 M<sub>e</sub>th<sub>o</sub>d

Codebook Agent instantiates Axis A as a discrete codebook with a query-conditioned prior, and Axis B as an MLP over flattened adjacencies trained on measured utility and token cost. Let $\mathcal { D } = \{ ( A _ { j } , c _ { j } , u _ { j } , \tau _ { j } ) \} _ { j = 1 } ^ { | \mathcal { D } | }$ be the execution records of Section 3. Define the composite reward

$$
R _ { j } ~ = ~ u _ { j } - \lambda \tilde { \tau } _ { j } , \qquad \tilde { \tau } _ { j } ~ = ~ { \frac { \tau _ { j } } { \frac { 1 } { | \{ j ^ { \prime } : c _ { j ^ { \prime } } = c _ { j } \} | } \sum _ { j ^ { \prime } : c _ { j ^ { \prime } } = c _ { j } } \tau _ { j ^ { \prime } } } } ,\tag{2}
$$

with $\lambda = 0 . 1 \colon \tilde { \tau } _ { j }$ is the token count normalized by the mean token count of the same task, so dificulty cancels and withintask rankings remain. Using measured tokens rather than |E| is essential: on our records edge count correlates with τ at $r \approx - 0 . 4$ , so an edge-count head pushes the system toward more expensive graphs. The codebook, code predictor, and reranking proxy are trained on D with no further LLM calls. Figure 2 shows ofline collection, the three training stages, and one-pass test-time generation.

## 4<sub>.</sub>1 T<sub>opo</sub>l<sub>ogy</sub> C<sub>o</sub>d<sub>e</sub>b<sub>oo</sub>k

If reward-surviving topologies occupy only a handful of distinct graphs, the generator should index that short list rather than search the adjacency manifold. We therefore compress observed topologies with a vector-quantized autoencoder. Flatten the of-diagonal entries $a = { \mathrm { v e c } } ( A ) \in \{ 0 , 1 \} ^ { N ( N - 1 ) }$ and encode

$$
z _ { e } = \mathrm { E n c } _ { \psi } ( a ) \in \mathbb { R } ^ { d _ { z } } , \qquad k ^ { \star } = \arg \operatorname* { m i n } _ { k \in \{ 1 , \ldots , K \} } \left\| z _ { e } - e _ { k } \right\| _ { 2 } ,
$$

$$
z _ { q } = e _ { k ^ { \star } } .\tag{3}
$$

with $d _ { z } = 3 2$ and codebook size $K = 1 6$ . The codebook $\{ e _ { k } \} _ { k = 1 } ^ { K }$ is updated by exponential moving averages (decay 0.99), with commitment coeficient $\beta \stackrel { - } { = } 0 . 2 5$ , straightthrough gradients, and reinitialization of codes unused for 50 batches (van den Oord, Vinyals, and Kavukcuoglu 2017; Razavi, van den Oord, and Vinyals 2019). A mirror-image decoder maps $z _ { q }$ to edge logits $\hat { \ell } = \mathrm { D e c } _ { \omega } ( z _ { q } ) \in \mathbb { R } ^ { N ( N - \bar { 1 } ) }$ the diagonal is fixed to zero and the graph remains directed. Writing $\hat { a } = \sigma ( \hat { \ell } )$ , the training objective is

$$
\mathcal { L } _ { \mathrm { v q } } = \mathrm { B C E } _ { w } \big ( \hat { a } , a \big ) + \beta \big \| z _ { e } - \mathrm { s g } [ z _ { q } ] \big \| _ { 2 } ^ { 2 } ,\tag{4}
$$

where $\mathrm { B C E } _ { w }$ weights positive entries by the zero-to-one ratio of the training set to counter edge sparsity and sg is the stop-gradient. We fit Eq. (4) only on records with $u _ { j } >$ 0.5, so the codebook stores topologies that solved their task. Neither encoder nor decoder is conditioned on the query: after training, code k decodes to the binary topology

$$
A ^ { ( k ) } = \mathbf { 1 } \big [ \sigma \big ( \mathrm { D e c } _ { \omega } ( e _ { k } ) \big ) \geq 1 / 2 \big ] ,\tag{5}
$$

and all query dependence is carried by the predictor below. We train for 200 epochs with Adam at learning rate $3 \times 1 0 ^ { - 4 }$ and batch size 16.

## 4<sub>.</sub>2 R<sub>ewa</sub>rd-W<sub>e</sub>i<sub>g</sub>ht<sub>e</sub>d C<sub>o</sub>d<sub>e</sub> Pr<sub>e</sub>di<sub>c</sub>ti<sub>o</sub>n

At test time there is no target topology to encode, so we learn a conditional prior over codes. Passing every training record through the frozen encoder and quantizer yields indices $k _ { j }$ For each distinct condition c we form the soft target

$$
y _ { k } ( c ) ~ = ~ { \frac { \sum _ { j : c _ { j } = c , k _ { j } = k } \exp ( \gamma R _ { j } ) } { \sum _ { k ^ { \prime } = 1 } ^ { K } \sum _ { j : c _ { j } = c , k _ { j } = k ^ { \prime } } \exp ( \gamma R _ { j } ) } } , \qquad \gamma = 2 ,\tag{6}
$$

which concentrates mass on codes whose topologies earned high reward under Eq. (2), so the prior already prefers cheap, successful graphs. The predictor $p _ { \theta } ( k ~ \mid ~ c )$ is an MLP $( d {  } 2 5 6 {  } 2 5 6 {  } K$ , dropout 0.1) trained by soft crossentropy

$$
\mathcal { L } _ { \mathrm { p r e d } } ~ = ~ - \sum _ { c } \sum _ { k = 1 } ^ { K } y _ { k } ( c ) ~ \log p _ { \theta } ( k \mid c ) .\tag{7}
$$

This is the only place the query enters generation, so the map from query to candidate topologies is a single matrix pipeline.

## 4<sub>.</sub>3 E<sub>xecu</sub>ti<sub>on-</sub>G<sub>roun</sub>d<sub>e</sub>d MLP P<sub>roxy</sub>

The scorer must (i) read the adjacency itself and (ii) regress measured tokens, not |E|. The first requirement is forced by the homogeneous-team regime: when $x _ { 1 } = \cdot \cdot \cdot = x _ { N } = x ,$ any message-passing layer of Eq. (1) aggregates identical features over self-loops, so every node state—and thus the pooled score—is independent of A. The same holds for mean and sum aggregators; there is nothing to distill from such a teacher on anonymous teams. We therefore score candidates with an MLP over the flattened adjacency,

$$
f _ { \phi } ( A , c ) = \mathrm { M L P } _ { \phi } \big ( \big [ \mathrm { v e c } ( A ) ; c \big ] \big ) = \big ( \widehat { u } , \widehat { c } \big ) \in \mathbb { R } ^ { 2 } ,\tag{8}
$$

and regress directly on the measured targets,

$$
\mathcal { L } _ { \mathrm { r e w } } ~ = ~ \frac { 1 } { \left| \mathcal { D } \right| } \sum _ { j = 1 } ^ { | \mathcal { D } | } \Bigl ( \left( \hat { u } _ { j } - u _ { j } \right) ^ { 2 } + \left( \hat { c } _ { j } - \tilde { \tau } _ { j } \right) ^ { 2 } \Bigr ) .\tag{9}
$$

Because |D| covers the topology space thinly, we regularize with a structure-token objective in the spirit of VQGraph (Yang et al. 2024). A separate reconstruction-only quantizer with $K _ { \mathrm { t o k } } = 3 2$ codes is fit over an augmented pool $\mathcal { A } ^ { + }$ of topologies (static families, Erdős–Rényi graphs at eight densities, single-edge perturbations, and Bernoulli samples of smoothed edge profiles, capped at 100 per condition). Writing $q ( k \mid A ) =$ softmax<sub>k</sub> $\bar { ( - \| z _ { e } ( A ) - e _ { k } \| _ { 2 } ^ { 2 } ) }$ for the soft code assignment of a pool graph, an auxiliary head predicts $q$ under

![](images/733af4c96124df82c2e92dc1bb3f2139ce9bfc479df76dd02cd9d21c7f453b44.jpg)  
Figure 3: Accuracy–latency Pareto on MATH. Accuracy from Table 1 versus median topology-generation latency (log-scale x-axis). Codebook Agent occupies the upper-left frontier: highest accuracy at 2.4 ms, against 301–396 ms for iterative generators.

$$
\mathcal { L } _ { \mathrm { c o d e } } = - \mathbb { E } _ { A \sim \mathcal { A } ^ { + } } \sum _ { k = 1 } ^ { K _ { \mathrm { t o k } } } q ( k \mid A ) \log \hat { q } _ { \phi } ( k \mid A ) ,\tag{10}
$$

and the total proxy loss is $\mathcal { L } _ { \phi } = \mathcal { L } _ { \mathrm { r e w } } + \lambda _ { \mathrm { c o d e } } \mathcal { L } _ { \mathrm { c o d e } }$ with $\lambda _ { \mathrm { c o d e } } ~ = ~ 0 . 1 \bar { }$ . The augmented pool contributes only to Eq. (10); reward supervision comes exclusively from real executions. The proxy is trained for 200 epochs with Adam at learning rate $1 0 ^ { - 3 }$ and batch size 256.

## 4<sub>.</sub>4 O<sub>ne-</sub>P<sub>ass</sub> I<sub>n</sub>f<sub>erence</sub>

Given a query embedding c, let $\mathcal { C } _ { M } ( c ) = \{ k _ { 1 } , . . . , k _ { M } \}$ be the top $M = 5$ codes under $p _ { \theta } ( \cdot \mid c )$ , and let $\boldsymbol { \mathcal { A } } ( \boldsymbol { c } ) = \{ \boldsymbol { A } ^ { ( k ) }$ $k \in \hat { \mathcal { C } } _ { M } ( c ) \}$ after removing duplicates under Eq. (5). The selected topology is

$$
A ^ { \star } ( c ) = \arg \operatorname* { m a x } _ { A \in \mathcal { A } ( c ) } \Bigl ( \hat { u } ( A , c ) - \lambda \hat { c } ( A , c ) \Bigr ) ,\tag{11}
$$

where $( \hat { u } , \hat { c } ) \ : = \ : f _ { \phi } ( A , c )$ is evaluated for all survivors in one batched forward pass. Generation therefore costs one predictor pass, at most M decoder passes, and one dense proxy pass—a fixed number of matrix products, with no sampling loop, no sparse graph construction, and no message passing on the test-time path.

## 5 Ex<sub>pe</sub>rim<sub>e</sub>nt<sub>s</sub>

We design the evaluation to stress-test the three claims of the introduction with four questions: (Q1) Does amortized

<table><tr><td rowspan="2">METHOD</td><td colspan="4">General/ Math reasoning</td><td colspan="2">Code generation</td><td rowspan="2">Avg.</td></tr><tr><td>GSM8K</td><td>MATH</td><td>MultiArith</td><td>SVAMP</td><td>MBPP</td><td>HumanEval</td></tr><tr><td>single-agent prompting</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Vanilla</td><td>87.0</td><td>47.6</td><td>97.2</td><td>88.5</td><td>72.4</td><td>73.1</td><td>77.63</td></tr><tr><td>CoT</td><td>86.50.5</td><td>48.0▲0.4</td><td>96.70.5</td><td>89.0▲0.5</td><td>74.0▲1.6</td><td>74.4▲1.3</td><td>78.10▲0.47</td></tr><tr><td>SC-CoT</td><td>87.0▲0.0</td><td>49.6▲2.0</td><td>97.2 ▲0.0</td><td>89.5▲1.0</td><td>75.0▲2.6</td><td>75.0▲1.9</td><td>78.88▲1.25</td></tr><tr><td>multi-agent collaboration</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LLM-Debate</td><td>89.0▲2.0</td><td>50.0▲2.4</td><td>97.8▲0.6</td><td>90.5▲2.0</td><td>76.4▲4.0</td><td>75.0▲1.9</td><td>79.78▲2.15</td></tr><tr><td>LLM-Blender</td><td>87.5▲0.5</td><td>48.4▲0.8</td><td>97.8▲0.6</td><td>89.0▲0.5</td><td>75.63.2</td><td>75.0▲1.9</td><td>78.88▲1.25</td></tr><tr><td>DyLAN</td><td>89.5▲2.5</td><td>50.0▲2.4</td><td>97.8▲0.6</td><td>90.5▲2.0</td><td>77.0▲4.6</td><td>76.3▲3.2</td><td>80.18▲2.55</td></tr><tr><td>learned agent and topology design</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPTSwarm</td><td>88.5▲1.5</td><td>49.6▲2.0</td><td>97.2 ▲0.0</td><td>90.5▲2.0</td><td>76.0▲3.6</td><td>75.6▲2.5</td><td>79.57▲1.94</td></tr><tr><td>ADAS</td><td>85.51.5</td><td>44.4▼3.2</td><td>96.70.5</td><td>88.00.5</td><td>71.01.4</td><td>70.6▼2.5</td><td>76.031.60</td></tr><tr><td>AFLOW</td><td>90.5▲3.5</td><td>52.4▲4.8</td><td>96.7▼0.5</td><td>90.0▲1.5</td><td>78.4▲6.0</td><td>76.9▲3.8</td><td>80.82▲3.19</td></tr><tr><td>MaAS</td><td>91.5▲4.5</td><td>53.66.0</td><td>97.2▲0.0</td><td>91.5▲3.0</td><td>79.0▲6.6</td><td>76.9▲3.8</td><td>81.62▲3.99</td></tr><tr><td>AgentDropout</td><td>90.0▲3.0</td><td>51.6▲4.0</td><td>98.3▲1.1</td><td>91.0▲2.5</td><td>78.0▲5.6</td><td>75.6▲2.5</td><td>80.75▲3.12</td></tr><tr><td>G-Designer</td><td>91.5▲4.5</td><td>52.4▲4.8</td><td>98.3▲1.1</td><td>91.5▲3.0</td><td>79.67.2</td><td>76.9▲3.8</td><td>81.70▲4.07</td></tr><tr><td>ARG-Designer</td><td>92.5▲5.5</td><td>54.0▲6.4</td><td>98.9▲1.7</td><td>92.5▲4.0</td><td>80.0▲7.6</td><td>77.5▲4.4</td><td>82.57▲4.94</td></tr><tr><td>TopoDIM</td><td>93.0▲6.0</td><td>55.0▲7.4</td><td>98.9▲1.7</td><td>92.5▲4.0</td><td>81.0▲8.6</td><td>77.5▲4.4</td><td>82.98▲5.35</td></tr><tr><td>GTD</td><td>93.5▲6.5</td><td>55.5▲7.9</td><td>98.2▲1.0</td><td>93.0▲4.5</td><td>80.4▲8.0</td><td>77.5▲4.4</td><td>83.02▲5.39</td></tr><tr><td>Codebook Agent (ours)</td><td>94.87.8</td><td>56.58.9</td><td>99.4▲2.2</td><td>95.46.9</td><td>83.5▲11.1</td><td>78.15.0</td><td>84.626.99</td></tr></table>

Table 1: Main accuracy results (%) on reasoning and code-generation benchmarks. HumanEval and MBPP report Pass@1. Methods are grouped into single-agent prompting, multi-agent collaboration, and learned agent or topology design. Deltas are absolute accuracy points relative to Vanilla; ▲ / ▼ mark improvements and decreases. Bold is best in each column; underline is second best. All runs use gpt-4o-mini under the protocol of Section 5.1.

Codebook Agent match or beat iterative query-conditioned designers on accuracy, or does indexing a short codebook trade quality for speed? (Q2) Is the useful design space truly a short list—do fixed topologies already sit within a point of generated ones, and does extra codebook capacity go unused? (Q3) Does an execution-grounded MLP over vec(A) cut tokens where the incumbent edge-count GNN cannot rank homogeneous teams, and is random selection enough? (Q4) Does one-pass codebook generation deliver the latency and token savings of Figure 1 without sacrificing transfer across LLM backends?

## 5<sub>.</sub>1 S<sub>e</sub>t<sub>up</sub>

Benchmarks. Reasoning: GSM8K (Cobbe et al. 2021), MATH (Hendrycks et al. 2021b), MultiArith (Roy and Roth 2015), SVAMP (Patel, Bhattamishra, and Goyal 2021). Code: MBPP (Austin et al. 2021), HumanEval (Chen et al. 2021) (Pass@1). Transfer: MMLU (Hendrycks et al. 2021a) under Qwen-3-8B (Table 2). Evaluation uses the fixed test splits of GSM8K (1314), MATH (500), MultiArith (180), SVAMP (1000), MBPP (500), HumanEval (160), and MMLU (1530); design-axis accuracy/token numbers use 200 tasks per benchmark (full MultiArith/HumanEval where smaller). Teams. Math-format: 4× MathSolver with FinalRefer. Code: 4× Programming Expert with FinalRefer. MMLU: 3× Knowledgeable Expert. Homogeneous profiles are the default (six of seven design-axis settings); HumanEval also runs a heterogeneous four-role team. Backbone gpt-4o-mini (temp. 0.7, top-p 1.0, max tokens 1024) unless stated; embeddings are frozen all-MiniLM-L6-v2 (d=384).

Baselines. Single-agent: Vanilla, CoT (Wei et al. 2022), SC-CoT (Wang et al. 2023). Multi-agent: LLM-Debate (Du et al. 2024), LLM-Blender (Jiang, Ren, and Lin 2023), DyLAN (Liu et al. 2024). Learned design: GPTSwarm (Zhuge et al. 2024), ADAS (Hu, Lu, and Clune 2025), AFLOW (Zhang et al. 2025d), MaAS (Zhang et al. 2025a), AgentDropout (Wang et al. 2025), G-Designer (Zhang et al. 2025c), ARG-Designer (Li et al. 2026), TopoDIM (Sun et al. 2026), GTD (Jiang et al. 2026). Metrics. Accuracy (Pass@1 for code); median topologygeneration latency (3 warmup + 50/100 timed calls on one GPU); mean LLM tokens/query; end-to-end wall clock. Single evaluation run per configuration.

## 5<sub>.</sub>2 M<sub>a</sub>i<sub>n</sub> R<sub>esu</sub>lt<sub>s</sub>

Table 1 reports gpt-4o-mini accuracy; Table 2 the Qwen-3- 8B transfer; Figures 3 and 4 the latency and token frontiers. We highlight four observations corresponding to Q1–Q4.

(Q1) Amortized selection clears the prior desi<sub>g</sub>ner frontier; it does not trade accuracy for a codebook. Table 1 puts Codebook Agent best on every column at 84.62 average: +6.99 over Vanilla (77.63), +4.44 over DyLAN (80.18), and +1.60 over the strongest prior topology designer GTD (83.02). The Vanilla gap is smallest on MultiArith (+2.2; near ceiling) and largest on MBPP (+11.1). Against GTD the gains are spread, not concentrated: +1.3 GSM8K, +1.0 MATH, +1.2 MultiArith, +2.4 SVAMP, +3.1 MBPP, +0.6 HumanEval. Reading by family: single-agent prompting floors the high 70s; multi-agent collaboration lifts into the low 80s under hand-designed structure; query-conditioned generators (G-Designer through GTD) form the prior frontier at ∼83. Codebook Agent sits above that frontier on all six benchmarks—so indexing a short list of graphs is not a speed concession; it matches or exceeds the iterative adjacency search it replaces.

(Q2) The useful desi<sub>g</sub>n space is a short list, and which fixed graph wins is not knowable a priori. Fixed topologies need no designer, so they measure how much of the payof lives in the choice itself. Across seven settings the best fixed family (fully connected / chain / star) stays within 1.4 accuracy points of every generated topology we measure, at comparable tokens; on MATH, fully connected is above every generated configuration. Capacity spent modeling the adjacency manifold is therefore spent where the problem does not live—the premise of Axis A as a codebook. The choice is nonetheless load-bearing: which fixed graph wins changes with the setting (fully connected on MATH, star on homogeneous HumanEval, chain on MMLU), with gaps up to 4.3 accuracy points and a 2.9× token factor (chain vs. fully connected on MATH). Identifying the winner for a new setting requires real LLM executions—the same cost class as our one-time 300-record collection—so a designer earns its keep by amortizing that selection per query. Figure 5(b) closes the loop on capacity: as K grows from 4 to 64, accuracy moves by at most 1.5 points (GSM8K) / 2.5 (HumanEval), and for every K≥8 the encoder uses at most six codes. Collapse is a property of reward-surviving topologies, not of an undersized quantizer.

(Q3) Measured-token MLP scorin<sub>g</sub> is where cost is won; the incumbent GNN cannot rank anonymous teams. We hold Axis A (codebook) and the candidate set fixed, and swap only Axis B. On six homogeneous settings Eq. (1) is constant in A, so every candidate in the short list receives the same score; the edge-count head is then the only separator, and because |E| correlates with tokens at r ≈ −0.4, it points toward expensive graphs. Empirically this incumbent scorer, reranking our codebook candidates, costs 1711 tokens/query on GSM8K and 918 on homogeneous HumanEval—worse than uniform random over the same candidates (1249 / 750), and not comparable to the full incumbent pipeline of Q4. Heterogeneous HumanEval is the exception on accuracy (incumbent 78.8 vs. ours 78.1); our largest token saving (33.2%) is also there. Replacing our proxy with random, or dropping rerank and taking the predictor’s top-1 (Figure 5c), shows the ranking is load-bearing for cost: random costs 23–35% more tokens at equal or lower accuracy; top-1 is already cheap (913 tokens on GSM8K) and rerank adds 0.6–1.0 accuracy points. The chain fixed topology is the extreme positive control for the inverted surrogate: sparsest connected, most token-expensive on every math-format benchmark.

![](images/0aa56ae4f59bea343653b15c67d0dbe5c5df9dd55421fabe8fd2a402c158bf63.jpg)

![](images/aab242bea8348a6735f2ad2514c2ea23d5193acd88e22ac233733f825b19f751.jpg)  
Figure 4: Accuracy–token trade-of on MATH and MBPP. Each bubble is one method: x is accuracy, y is ln(tokens×n) so that Token Consump. $\boldsymbol { \mathbf { \rho } } = e ^ { y } ,$ , and area scales with mean tokens per query. Lower-right is better; Codebook Agent sits nearest that corner on both benchmarks.

(Q4) One-pass generation is >120× faster and 22– 33% cheaper in tokens, and the ordering transfers. Figure 3 places MATH accuracy against median generation latency: iterative generators sit at 301–396 ms; Codebook Agent at 2.4–2.5 ms (125–158×). The gap is structural— $- T K { = } 2 5 0$ sparse GNN evaluations vs. one predictor pass, ≤M decodes, and one batched MLP call (Eq. (11))—and design falls below 0.1% of end-to-end latency. Figure 4 shows the accuracy–token frontier on MATH and MBPP; Codebook Agent occupies the lowerright corner. Against the full incumbent pipeline (iterative decoder + edge-count GNN), mean tokens drop 21.9–33.2% (e.g. 1239→927 GSM8K, 2304→1611 MATH, 699→546 HumanEval-homo, 624→417 HumanEval-het), and wall clock follows (MATH 43.7→27.4 min; MMLU 12.7→7.9 min). Table 2 repeats the accuracy comparison with Qwen-3-8B: absolute numbers fall, but Codebook Agent still leads at 74.0 vs. GTD 72.7 / G-Designer 72.1, with MATH +1.7 and MMLU +1.9 over GTD—so the codebook is not an artifact of one API.

Overall. Q1–Q4 close the loop opened in the introduction: when reward-surviving topologies are few, accuracy is preserved by selecting among them (Q1–Q2), cost is won by scoring measured tokens rather than |E| on anonymous teams (Q3), and amortization makes that selection essentially free at test time while transferring across backends (Q4). The ablations below isolate each mechanism under controlled Axis A/B swaps.

## 5<sub>.</sub>3 Abl<sub>a</sub>ti<sub>o</sub>n<sub>s</sub>

We ablate the mechanisms behind Eqs. (4)–(11) and test whether the main-table ordering survives a backbone change. Figure 5 varies four controls on GSM8K and homogeneous HumanEval under a fixed training budget; Table 2 repeats the accuracy comparison with Qwen-3-8B agents.

(a) Team size N. Figure 5(a): retrain per $N \in \{ 2 , \ldots , 1 0 \}$ on the same 50-task budget. GSM8K stays flat (93.0–95.5);

![](images/603bf3ae9e173a0a73e5c2ccbb9d55d9128fc515c69664b38daf489c96eefc9c.jpg)

![](images/e086126c49a22894416d530190870b10588b84fa4c35f6d68aab0d6fcdd733ad.jpg)

![](images/a302b6ccc4307d541065e0a6be32031b37432777cce1f346b23d25212f82dde3.jpg)

![](images/5d47ab5a7b1f5ccaab366ccf357b22a26b9d9646188b0b65308b344bd1e28371.jpg)  
Figure 5: Ablations on GSM8K and homogeneous HumanEval. (a) Accuracy versus team size N (retrained per N on the same training budget). (b) Accuracy versus codebook size K; used codes saturate near six for every $K { \geq } 8 .$ . (c) Mean tokens per query for MLP, random, and GNN reranking under a fixed candidate set. (d) Mean tokens versus the rerank cost weight $\lambda _ { \mathrm { c o s t } }$ with default −0.1.

HumanEval rises 71.9→87.5. Coding benefits from a wider expert pool, but the same index-and-select pipeline tracks that gain without architectural change—so the method is not a four-node trick, and topology selection remains useful as the team grows.

(b) Codebook size K. Figure 5(b): $K \in \{ 4 , \ldots , 6 4 \}$ moves accuracy by ≤1.5 (GSM8K) / 2.5 (HumanEval). For every $K { \ge } 8$ the encoder uses at most six codes—extra capacity stays idle under EMA with dead-code reset—so the design space, not the quantizer, is what collapsed. Searching the full $\bar { N } { \times } N$ adjacency at test time cannot be justified by needing more capacity. The used-code count plateaus near the same six survivors across both benchmarks, which is why enlarging K past 16 yields almost no accuracy return under a matched training budget.

(c) Rerank signal. Figure 5(c): same Top-M candidates; only the selector changes. Tokens on GSM8K / HumanEval are MLP 927/546, random 1249/750, GNN 1711/918, with accuracy within 1.5 points—the ranking buys cost, not large Pass@1. The GNN is worse than random because on homogeneous teams Eq. (1) is constant in A and its |E| head favors expensive graphs (r ≈ −0.4 with tokens). Only the MLP, reading vec(A) and regressing measured τ˜, separates candidates on the true objective. Keeping the short list fixed isolates the critic: any token gap here is attributable to Eq. (11) alone.

(d) Cost weight $\lambda _ { \mathrm { c o s t } }$ . Figure 5(d): λ=0 is costliest (1108/743 tokens) with no accuracy gain. Tokens fall to 927 (GSM8K) and 442 (HumanEval at −0.4); accuracy peaks at −0.1/−0.2. Default −0.1 sits on the frontier: the proxy must both see structure and be asked to care about cost. The prior already prefers cheap codes (top-1: 913 on GSM8K), and rerank adds 0.6–1.0 accuracy points. Pushing λ more negative continues to cut tokens but begins to trade accuracy, so −0.1 is a deliberate operating point rather than an unconstrained minimum-cost choice.

Qwen-3-8B transfer (Table 2). Same teams and 300-record protocol, backbone swapped to Qwen-3-8B. Absolute numbers drop (CoT avg. 67.5), but Codebook Agent still leads every column (92.7/63.5/65.8, avg. 74.0) over GTD (72.7) and G-Designer (72.1). MATH (+1.7 vs. GTD) and MMLU (+1.9; three-expert team) show the gain is not confined to saturated arithmetic or four MathSolvers. Design never calls the agent LLM (Eq. (11)), so swapping the backbone changes only the rewards in D—the short-list recipe is not an API artifact. Relative ordering among adaptive designers is preserved under the weaker model, which suggests that the amortized index, not gpt-4o-mini idiosyncrasies, drives the main-table gaps.

<table><tr><td>Method</td><td>GSM8K</td><td>MATH</td><td>MMLU</td><td>Avg.</td></tr><tr><td>CoT</td><td>87.8</td><td>55.0</td><td>59.6</td><td>67.5</td></tr><tr><td>GPTSwarm</td><td>90.9▲3.1</td><td>60.0▲5.0</td><td>63.5▲3.9</td><td>71.5▲4.0</td></tr><tr><td>MaAS</td><td>91.23.4</td><td>60.4▲5.4</td><td>63.94.3</td><td>71.84.3</td></tr><tr><td>G-Designer</td><td>91.5▲3.7</td><td>60.2▲5.2</td><td>64.65.0</td><td>72.1▲4.6</td></tr><tr><td>GTD</td><td>92.3▲4.5</td><td>61.86.8</td><td>63.9▲4.3</td><td>72.75.2</td></tr><tr><td>Ours</td><td>92.74.9</td><td>63.58.5</td><td>65.86.2</td><td>74.06.5</td></tr></table>

Table 2: Qwen-3-8B transfer. Accuracy (%) on GSM8K, MATH, and MMLU under the same agent teams and protocol as Section 5.1, with Qwen-3-8B replacing gpt-4o-mini. Avg. is the three-benchmark mean; deltas are from CoT; bold marks each column’s best.

## 6 C<sub>o</sub>n<sub>c</sub>l<sub>us</sub>i<sub>o</sub>n

We find that efective multi-agent topologies form a short list, that denser graphs need not cost fewer tokens, and that message-passing critics can miss structure on homogeneous teams. Accordingly, Codebook Agent indexes a small discrete set of topologies and selects among them with a lightweight proxy (Eqs. (4)–(11)), leading all six benchmarks and the Qwen-3-8B transfer while cutting design latency to 2.4 ms and LLM tokens by 21.9–33.2%.

## R<sub>e</sub>f<sub>erences</sub>

Austin, J.; Odena, A.; Nye, M.; Bosma, M.; Michalewski, H.; Dohan, D.; Jiang, E.; Cai, C.; Terry, M.; Le, Q.; and Sutton, C. 2021. Program Synthesis with Large Language Models. arXiv preprint arXiv:2108.07732.

Cemri, M.; Pan, M. Z.; Yang, S.; Agrawal, L. A.; Chopra, B.; Tiwari, R.; Keutzer, K.; Parameswaran, A.; Klein, D.; Ramchandran, K.; Zaharia, M.; Gonzalez, J. E.; and Stoica, I. 2025. Why Do Multi-Agent LLM Systems Fail? In Advances in Neural Information Processing Systems 38, Datasets and Benchmarks Track.

Chen, L.; Davis, J. Q.; Hanin, B.; Bailis, P.; Stoica, I.; Zaharia, M.; and Zou, J. 2024a. Are More LLM Calls All You Need? Towards the Scaling Properties of Compound AI Systems. In Advances in Neural Information Processing Systems 37.

Chen, L.; Zaharia, M.; and Zou, J. 2024. FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance. Transactions on Machine Learning Research.

Chen, M.; Tworek, J.; Jun, H.; Yuan, Q.; Ponde de Oliveira Pinto, H.; Kaplan, J.; Edwards, H.; Burda, Y.; Joseph, N.; Brockman, G.; et al. 2021. Evaluating Large Language Models Trained on Code. arXiv preprint arXiv:2107.03374.

Chen, W.; Su, Y.; Zuo, J.; Yang, C.; Yuan, C.; Chan, C.-M.; Yu, H.; Lu, Y.; Hung, Y.-H.; Qian, C.; Qin, Y.; Cong, X.; Xie, R.; Liu, Z.; Sun, M.; and Zhou, J. 2024b. AgentVerse: Facilitating Multi-Agent Collaboration and Exploring Emergent Behaviors. In Proceedings of the 12th International Conference on Learning Representations.

Chen, W.; Yuan, J.; Qian, C.; Yang, C.; Liu, Z.; and Sun, M. 2025. Optima: Optimizing Efectiveness and Eficiency for LLM-Based Multi-Agent System. In Findings of the Associationfor Computational Linguistics: ACL 2025.

Cobbe, K.; Kosaraju, V.; Bavarian, M.; Chen, M.; Jun, H.; Kaiser, L.; Plappert, M.; Tworek, J.; Hilton, J.; Nakano, R.; Hesse, C.; and Schulman, J. 2021. Training Verifiers to Solve Math Word Problems. arXiv preprint arXiv:2110.14168.

Du, Y.; Li, S.; Torralba, A.; Tenenbaum, J. B.; and Mordatch, I. 2024. Improving Factuality and Reasoning in Language Models through Multiagent Debate. In Proceedings of the 41st International Conference on Machine Learning.

Hendrycks, D.; Burns, C.; Basart, S.; Zou, A.; Mazeika, M.; Song, D.; and Steinhardt, J. 2021a. Measuring Massive Multitask Language Understanding. In Proceedings of the 9th International Conference on Learning Representations.

Hendrycks, D.; Burns, C.; Kadavath, S.; Arora, A.; Basart, S.; Tang, E.; Song, D.; and Steinhardt, J. 2021b. Measuring Mathematical Problem Solving with the MATH Dataset. In Advances in Neural Information Processing Systems 34, Datasets and Benchmarks Track.

Hinton, G.; Vinyals, O.; and Dean, J. 2015. Distilling the Knowledge in a Neural Network. arXiv preprint arXiv:1503.02531.

Ho, J.; Jain, A.; and Abbeel, P. 2020. Denoising Difusion Probabilistic Models. In Advances in Neural Information Processing Systems 33.

Hong, S.; Zhuge, M.; Chen, J.; Zheng, X.; Cheng, Y.; Wang, J.; Zhang, C.; Wang, Z.; Yau, S. K. S.; Lin, Z.; Zhou, L.; Ran, C.; Xiao, L.; Wu, C.; and Schmidhuber, J. 2024. MetaGPT: Meta Programming for a Multi-Agent Collaborative Framework. In Proceedings of the 12th International Conference on Learning Representations.

Hu, S.; Lu, C.; and Clune, J. 2025. Automated Design of Agentic Systems. In Proceedings of the 13th International Conference on Learning Representations.

Jiang, D.; Ren, X.; and Lin, B. Y. 2023. LLM-Blender: Ensembling Large Language Models with Pairwise Ranking and Generative Fusion. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers).

Jiang, E. H.; Li, M.; Wan, G.; Yin, S.; Wu, Y.; Liang, X.; Li, X.; Sun, Y.; Wang, W.; Chang, K.-W.; and Wu, Y. N. 2026. Dynamic Generation of Multi-LLM Agents Communication Topologies with Graph Difusion Models. In Proceedings ofthe 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers).

Jo, J.; Lee, S.; and Hwang, S. J. 2022. Score-Based Generative Modeling of Graphs via the System of Stochastic Differential Equations. In Proceedings of the 39th International Conference on Machine Learning.

Li, J.; Zhang, Q.; Yu, Y.; Fu, Q.; and Ye, D. 2024. More Agents Is All You Need. Transactions on Machine Learning Research.

Li, S.; Liu, Y.; Wen, Q.; Zhang, C.; and Pan, S. 2026. Assemble Your Crew: Automatic Multi-agent Communication Topology Design via Autoregressive Graph Generation. In Proceedings of the 40th AAAI Conference on Artificial Intelligence.

Liu, Z.; Zhang, Y.; Li, P.; Liu, Y.; and Yang, D. 2024. Dynamic LLM-Agent Network: An LLM-Agent Collaboration Framework with Agent Team Optimization. In Proceedings of the 1st Conference on Language Modeling.

Patel, A.; Bhattamishra, S.; and Goyal, N. 2021. Are NLP Models Really Able to Solve Simple Math Word Problems? In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies.

Qian, C.; Liu, W.; Liu, H.; Chen, N.; Dang, Y.; Li, J.; Yang, C.; Chen, W.; Su, Y.; Cong, X.; Xu, J.; Li, D.; Liu, Z.; and Sun, M. 2024. ChatDev: Communicative Agents for Software Development. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers).

Qian, C.; Xie, Z.; Wang, Y.; Liu, W.; Zhu, K.; Xia, H.; Dang, Y.; Du, Z.; Chen, W.; Yang, C.; Liu, Z.; and Sun, M. 2025. Scaling Large Language Model-Based Multi-Agent Collaboration. In Proceedings of the 13th International Conference on Learning Representations.

Razavi, A.; van den Oord, A.; and Vinyals, O. 2019. Generating Diverse High-Fidelity Images with VQ-VAE-2. In Advances in Neural Information Processing Systems 32.

Roy, S.; and Roth, D. 2015. Solving General Arithmetic Word Problems. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing.

Simonovsky, M.; and Komodakis, N. 2018. GraphVAE: Towards Generation of Small Graphs Using Variational Autoencoders. In Proceedings of the 27th International Conference on Artificial Neural Networks.

Smit, A.; Grinsztajn, N.; Duckworth, P.; Barrett, T. D.; and Pretorius, A. 2024. Should We Be Going MAD? A Look at Multi-Agent Debate Strategies for LLMs. In Proceedings of the 41st International Conference on Machine Learning.

Song, J.; Meng, C.; and Ermon, S. 2021. Denoising Difusion Implicit Models. In Proceedings of the 9th International Conference on Learning Representations.

Sun, R.; Ding, J.; Gong, C.; Gu, T.; Jiang, Y.; Zhang, J.; Pan, L.; and Lü, L. 2026. TopoDIM: One-shot Topology Generation of Diverse Interaction Modes for Multi-Agent Systems. In Findings of the Association for Computational Linguistics: ACL 2026.

Tian, Y.; Zhang, C.; Guo, Z.; Zhang, X.; and Chawla, N. V. 2023. Learning MLPs on Graphs: A Unified View of Efectiveness, Robustness, and Eficiency. In Proceedings of the 11th International Conference on Learning Representations.

van den Oord, A.; Vinyals, O.; and Kavukcuoglu, K. 2017. Neural Discrete Representation Learning. In Advances in Neural Information Processing Systems 30.

Veličković, P.; Cucurull, G.; Casanova, A.; Romero, A.; Liò, P.; and Bengio, Y. 2018. Graph Attention Networks. In Proceedings ofthe 6th International Conference on Learning Representations.

Vignac, C.; Krawczuk, I.; Siraudin, A.; Wang, B.; Cevher, V.; and Frossard, P. 2023. DiGress: Discrete Denoising Diffusion for Graph Generation. In Proceedings of the 11th International Conference on Learning Representations.

Wang, X.; Wei, J.; Schuurmans, D.; Le, Q.; Chi, E.; Narang, S.; Chowdhery, A.; and Zhou, D. 2023. Self-Consistency Improves Chain of Thought Reasoning in Language Models. In Proceedings of the 11th International Conference on Learning Representations.

Wang, Z.; Wang, Y.; Liu, X.; Ding, L.; Zhang, M.; Liu, J.; and Zhang, M. 2025. AgentDropout: Dynamic Agent Elimination for Token-Eficient and High-Performance LLM-Based Multi-Agent Collaboration. In Proceedings of the 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers).

Wei, J.; Wang, X.; Schuurmans, D.; Bosma, M.; Ichter, B.; Xia, F.; Chi, E.; Le, Q.; and Zhou, D. 2022. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. In Advances in Neural Information Processing Systems 35.

Wu, L.; Lin, H.; Huang, Y.; Fan, T.; and Li, S. Z. 2023. Extracting Low-/High-Frequency Knowledge from Graph Neural Networks and Injecting It into MLPs: An Efective GNNto-MLP Distillation Framework. In Proceedings of the 37th AAAI Conference on Artificial Intelligence.

Wu, Q.; Bansal, G.; Zhang, J.; Wu, Y.; Li, B.; Zhu, E.; Jiang, L.; Zhang, X.; Zhang, S.; Liu, J.; Awadallah, A. H.; White, R. W.; Burger, D.; and Wang, C. 2024. AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation. In Proceedings of the 1st Conference on Language Modeling.

Yang, L.; Tian, Y.; Xu, M.; Liu, Z.; Hong, S.; Qu, W.; Zhang, W.; Cui, B.; Zhang, M.; and Leskovec, J. 2024. VQGraph: Rethinking Graph Representation Space for Bridging GNNs and MLPs. In Proceedings ofthe 12th International Conference on Learning Representations.

You, J.; Ying, R.; Ren, X.; Hamilton, W. L.; and Leskovec, J. 2018. GraphRNN: Generating Realistic Graphs with Deep Auto-Regressive Models. In Proceedings of the 35th International Conference on Machine Learning.

Zhang, G.; Niu, L.; Fang, J.; Wang, K.; Bai, L.; and Wang, X. 2025a. Multi-agent Architecture Search via Agentic Supernet. In Proceedings of the 42nd International Conference on Machine Learning.

Zhang, G.; Yue, Y.; Li, Z.; Yun, S.; Wan, G.; Wang, K.; Cheng, D.; Yu, J. X.; and Chen, T. 2025b. Cut the Crap: An Economical Communication Pipeline for LLM-Based Multi-Agent Systems. In Proceedings of the 13th International Conference on Learning Representations.

Zhang, G.; Yue, Y.; Sun, X.; Wan, G.; Yu, M.; Fang, J.; Wang, K.; Chen, T.; and Cheng, D. 2025c. G-Designer: Architecting Multi-Agent Communication Topologies via Graph Neural Networks. In Proceedings ofthe 42nd International Conference on Machine Learning.

Zhang, J.; Xiang, J.; Yu, Z.; Teng, F.; Chen, X.; Chen, J.; Zhuge, M.; Cheng, X.; Hong, S.; Wang, J.; Zheng, B.; Liu, B.; Luo, Y.; and Wu, C. 2025d. AFlow: Automating Agentic Workflow Generation. In Proceedings of the 13th International Conference on Learning Representations.

Zhang, S.; Liu, Y.; Sun, Y.; and Shah, N. 2022. Graph-Less Neural Networks: Teaching Old MLPs New Tricks via Distillation. In Proceedings of the 10th International Conference on Learning Representations.

Zhou, C.; Wang, X.; and Zhang, M. 2024. Unifying Generation and Prediction on Graphs with Latent Graph Difusion. In Advances in Neural Information Processing Systems 37.

Zhuge, M.; Wang, W.; Kirsch, L.; Faccio, F.; Khizbullin, D.; and Schmidhuber, J. 2024. GPTSwarm: Language Agents as Optimizable Graphs. In Proceedings ofthe 41st International Conference on Machine Learning.