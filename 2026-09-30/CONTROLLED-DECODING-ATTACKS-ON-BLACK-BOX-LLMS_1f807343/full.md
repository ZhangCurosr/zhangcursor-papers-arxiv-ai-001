# CONTROLLED DECODING ATTACKS ON BLACK-BOX LLMS

Jesson Wang<sup>1∗</sup>, Shawn Li<sup>1∗</sup>, Wei Yang<sup>1</sup>, Franck Dernoncourt<sup>2</sup>

Ryan A. Rossi<sup>2</sup>, Charith Peris<sup>3</sup>, Yue Zhao<sup>1</sup>

<sup>1</sup>University of Southern California, <sup>2</sup>Adobe, <sup>3</sup>Amazon

<sup>∗</sup>Equal contribution.

## ABSTRACT

Manipulating next-token probabilities during generation can bypass the safety alignment of large language models. Existing approaches, however, rely on access to model weights or numerical token probabilities and therefore do not apply to interfaces that return only sampled text. Reconstructing probabilities from sampled outputs offers a possible alternative, but finite sampling produces sparse and noisy estimates, while repeating this process at every generation step incurs substantial query costs. Our empirical observations suggest that large distributional changes along successful jailbreak trajectories are concentrated at a small subset of positions, motivating selective control. We introduce BLINDBIAS, a framework for jailbreaking through text-only continuation interfaces that permit repeated sampling and assistant-prefix continuation. Sample-Based Distribution Reconstruction combines sampled outputs with a prior over unobserved actions to obtain a usable control signal. Risk-Gated Residual Control uses the evolving response prefix to decide when to reconstruct and modify the distribution, concentrating sampling costs at selected positions. Speculative Multi-Token Execution further amortizes target calls by verifying and accepting draft prefixes that require no intervention. Across four target endpoints and three benchmarks, BLINDBIAS achieves the highest mean score most comparisons against baselines.

Code: https://github.com/JessonWong/controlled-decoding

## 1 INTRODUCTION

AI-based sequential decision-making systems are being explored in high-stakes domains involving sensitive and personalized data, including precision rehabilitation (Ye et al., 2025). If LLMs are incorporated into similar decision-making pipelines, jailbreak vulnerabilities may introduce additional safety and privacy risks by allowing adversarial users to bypass intended safeguards.

Safety alignment trains language models to refuse harmful requests (Ouyang et al., 2022; Bai et al., 2022), but this behavior remains vulnerable to interventions during generation. Decoding-time attacks act directly on the next-token distribution, using a helper model to redirect an aligned target’s output (Zhao et al., 2025; Zhou et al., 2024). Related techniques in proxy tuning and controlled generation likewise steer a frozen model by modifying its output distribution (Liu et al., 2024a; Dathathri et al., 2020; Krause et al., 2021; Yang & Klein, 2021). Their appeal is the granularity of control: an intervention can respond to the evolving answer at each generation step. Their limitation is access: applying such an intervention directly requires target weights or numerical token probabilities.

We study whether this fine-grained control can be retained when the target exposes only sampled text. Existing black-box jailbreaks primarily manipulate the input through prompt search (Chao et al., 2023; Mehrotra et al., 2023), template evolution (Liu et al., 2024b), multi-turn interaction (Russinovich et al., 2024), or encoded instructions (Yuan et al., 2024), leaving decoding-time control under such access comparatively unexplored. A natural route is to estimate a next-token distribution from repeated sampled continuations of the same prefix, then apply control to the resulting estimate. The challenge is to obtain a useful control signal at an affordable query cost: limited sampling produces sparse and noisy estimates, while extensive sampling at every generation step becomes expensive. This tension motivates selective control that concentrates distribution estimation and intervention at a subset of positions.

![](images/2d8b7e25e3a157c8044c43d65304bfdf6930e953ff99dcc272f1ef416fec19fd.jpg)  
Figure 1: Overview of BLINDBIAS. The text-only target proposes a base candidate, and a local prefix-risk model determines whether the controller should intervene. Bypassed candidates are accepted directly. At intervention positions, sampled continuations are mapped to local actions and combined with a uniform prior to reconstruct a distribution. BiasNet then adjusts this distribution with a gate-scaled residual. After consecutive bypasses, speculative execution requests a multi-token draft, verifies its prefixes locally, commits the accepted prefix, and resumes controlled decoding at the first position that requires intervention.

Prior research on safety alignment provides a basis for this selective approach: refusal behavior can be concentrated in the opening tokens, and establishing a compliant prefix can weaken subsequent refusal (Qi et al., 2025; Andriushchenko et al., 2025). Our observations provide a complementary motivation. Along successful jailbreak trajectories, the KL divergence between next-token distributions before and after intervention is small at most positions, with large changes concentrated at a few positions (Figure 2). Together, these observations motivate selective control conditioned on the evolving prefix, allowing intervention beyond a fixed opening window while avoiding uniform distribution estimation throughout generation.

We introduce BLINDBIAS, a framework for decoding-time jailbreaking through a text-only continuation interface. It adapts the BiasNet residual controller from the numerical-probability setting (Wang et al., 2026) to sampled outputs through three components that address the information and query costs of sample-only control. Sample-Based Distribution Reconstruction (§3.1) supplies the distributional signal by estimating next-action probabilities from sampled continuations and assigning probability mass to unobserved actions. Risk-Gated Residual Control (§3.2) limits how often reconstruction is requested: a local prefix model selects intervention positions and scales the learned residual, while bypassed positions retain the target’s candidate. Speculative Multi-Token Execution (§3.3) reduces repeated target calls across bypassed positions by requesting a longer draft, checking each draft prefix against the gate, and committing only the accepted prefix. This component borrows the draft-and-verify structure of speculative decoding (Leviathan et al., 2023; Chen et al., 2023) and uses local gate verification to amortize calls to the remote target. The framework assumes continuation from a supplied assistant prefix and maps returned text into a local tokenizer’s vocabulary; we consider an empirical string-based action space separately as an exploratory extension. Figure 1 summarizes the complete framework.

We evaluate BLINDBIAS on four target endpoints and three benchmarks. Our analyses examine how reconstruction priors recover part of the signal lost through finite sampling and characterize the trade-off between attack effectiveness and intervention frequency.

Our contributions are as follows.

• We develop BLINDBIAS, which adapts decoding-time residual control to text-only sampling without accessing target weights or log-probabilities.

• We combine sample-based distribution reconstruction with prefix-dependent gating and multi-token draft verification to address both the information and query costs of sampleonly control.

• We evaluate the framework on four targets and three benchmarks, and analyze reconstruction quality, selective intervention, cross-family prior transfer, and an exploratory empirical string-based action space.

## 2 RELATED WORK

Prompt-based jailbreaking. Jailbreak attacks commonly seek inputs that induce an aligned model to answer otherwise refused requests. Gradient-based suffix optimization produces adversarial prompts that can transfer across models (Zou et al., 2023), while PAIR and TAP use attacker language models to refine prompts through target feedback and tree search, respectively (Chao et al., 2023; Mehrotra et al., 2023). GPTFuzzer develops reusable attack templates through mutation and response-based selection (Yu et al., 2024). Other approaches change how a request is represented: CipherChat uses cipher-based communication (Yuan et al., 2024), FlipAttack disguises requests through text flipping (Liu et al., 2026), and LogiBreak translates requests into formal logical expressions (Peng et al., 2026). Crescendo extends the interaction across turns, gradually steering the conversation toward a harmful objective (Russinovich et al., 2024). All of these methods receive only text from the target. Unlike prompt-level attacks, however, BLINDBIAS additionally assumes repeated stochastic sampling and continuation from an attacker-supplied assistant prefix. A complementary line of agent-safety evaluation examines permission boundaries: FORTIS benchmarks over-privilege in skill selection and execution (Li et al., 2026d). Query-agnostic black-box attacks also target LLM-based retrieval by injecting transferable tokens into documents (Li et al., 2026a).

Decoding-time control and jailbreaking. Controlled generation provides mechanisms for steering a frozen language model during decoding. PPLM updates hidden activations using attribute-model gradients (Dathathri et al., 2020), whereas GeDi and FUDGE guide token probabilities using generative discriminators and predictions from partial sequences (Krause et al., 2021; Yang & Klein, 2021). Proxy tuning transfers the distributional difference between small tuned and untuned models to a larger target (Liu et al., 2024a). For jailbreaking, Weak-to-Strong and Emulated Disalignment use auxiliary model distributions to redirect an aligned target during decoding (Zhao et al., 2025; Zhou et al., 2024). Most directly related, JULI introduces BiasNet, a lightweight module that manipulates target token log-probabilities and can operate with only top-5 log-probabilities (Wang et al., 2026). Thus, black-box decoding-time attacks already exist when numerical probabilities are exposed. Our contribution is to adapt this residual-control mechanism to a stricter, sample-only interface: BLINDBIAS reconstructs a smoothed distribution from returned text and selectively pays the resulting sampling cost. It retains the BiasNet formulation while changing how its inputs are obtained and when it is executed; the main setting uses a local tokenizer to define the action vocabulary.

Shallow alignment and efficient execution. Evidence that safety alignment can disproportionately affect the first few output tokens helps explain why compliant prefixes can undermine refusal (Qi et al., 2025). Adaptive jailbreaking studies likewise demonstrate vulnerabilities associated with prefilling and target-specific API access (Andriushchenko et al., 2025). Related analyses of promptattack defenses find reliance on surface heuristics (Li et al., 2026c) and degradation of tool-using agent capabilities following defense training (Li & Zhao, 2026). These findings motivate selective intervention, but do not establish that a fixed initial window suffices for every response. Our prefixdependent gate can reactivate control later in generation and allocates distribution-estimation queries according to the current candidate prefix. To reduce requests during stretches without intervention, we also draw on the draft-and-verify structure of speculative decoding (Leviathan et al., 2023; Chen et al., 2023). Classical speculative decoding verifies a cheaper model’s proposals against a target model while preserving the target sampling distribution. Here, the remote target supplies the draft and a local risk model verifies whether each prefix permits bypassing the controller. This verification enforces the gate rule on accepted prefixes; it does not imply distribution preservation or token-fortoken equivalence with repeated single-token API calls.

Reasoning verification and adaptive computation. Related work improves reliability through external evidence and feedback. Premise verification combines retrieval with logical reasoning to identify false premises before generation, without requiring model logits (Qin et al., 2026b). TS-Reasoner integrates domain-specific tools and error feedback for multi-step time series analysis (Ye et al., 2026b). Memory retrieval for changing preferences learns when to access memory and which historical turns to select based on their estimated utility (Qin et al., 2026a). Adaptive computation is also studied in multi-agent reasoning: Learning to Deliberate learns policies for persisting, refining, or conceding (Yang & Thomason, 2025), while AgentAuditor verifies branch-level evidence at divergence points in reasoning trees (Yang et al., 2026). Self-Compression uses importanceweighted penalties during training to reduce redundant reasoning chunks (Chen et al., 2026). These approaches provide context for verification and adaptive resource use in LLM systems, with objectives distinct from jailbreak control.

Multimodal reliability and efficient adaptation. Beyond language-model safety, targeted interventions have been studied for multimodal reliability. Semantics-prototype learning addresses biased predicate annotations in panoptic scene graph generation (Li et al., 2024), while DPU dynamically updates class prototypes for multimodal out-of-distribution detection (Li et al., 2025b). Geometry over Density further studies few-shot cross-domain OOD detection through diffusion-trajectory geometry without task-specific retraining (Li et al., 2026b). Treble Counterfactual VLMs applies causal interventions to reduce hallucinations (Shawn et al., 2025). MIRROR improves multimodal reasoning consistency by using successful reasoning from one view to supervise other views of the same problem (Ye et al., 2026a). Under device-side computational constraints, cloud–device collaboration enables multimodal adaptation and video out-of-distribution detection without on-device backpropagation (Ji et al., 2025; Li et al., 2025a). These studies offer broader context for selective intervention and efficient adaptation, although their tasks and access assumptions differ from the sample-only decoding control considered here.

## 3 PROPOSED METHOD

Figure 1 presents the overall workflow of BLINDBIAS, which combines sample-based distribution reconstruction, risk-gated residual control, and speculative multi-token execution. We first specify the threat model and action representation, then describe distribution reconstruction, controller training and inference, and the speculative path used to reduce repeated target calls.

Threat Model and Interface We assume a text-only continuation interface that accepts a user prompt x and an attacker-supplied assistant prefix. The attacker may issue repeated stochastic continuation requests from the same prompt but cannot access target weights, hidden states, or numerical token probabilities.

Let $y _ { < t } = \left( y _ { 1 } , \ldots , y _ { t - 1 } \right)$ denote the committed sequence of local actions, whose decoded text is supplied as the assistant prefix. A local encoder–decoder $( E , D )$ maps returned text to action IDs in a vocabulary V and maps selected actions back to text. The tokenizers used for each target are specified in Appendix A.2; these local actions need not coincide with the provider’s internal tokens. We denote the induced next-action distribution by

$$
p _ { \theta } ( \cdot \mid x , y _ { < t } ) ,\tag{1}
$$

which we observe only through sampled text mapped into V. The output budget counts local actions, while API requests have separate provider-side output budgets.

## 3.1 SAMPLE-BASED DISTRIBUTION RECONSTRUCTION

Because $p _ { \theta }$ is not exposed by the API, we estimate the next-action distribution from $K$ independently sampled continuations at the same chat history $( x , y _ { < t } )$ . Each valid response is mapped locally to one action in the decoding vocabulary V. Let $v _ { 1 } , \ldots , v _ { K }$ be these action IDs and let

![](images/0bf1d9ad1f72abf477e30fead1482075d36e9633e27a237cda895d7970e8707f.jpg)

![](images/2a74a48913c1b0dd21aba14cbfcad2c0d3a0be948dbb575e3d586f1ff6afe714.jpg)

![](images/0faea17533e1dd84f379ac26931510a4afbac14d81520b6d2911a500d117f3bf.jpg)

![](images/9f74f672eda40915f88db2cb52693dfd51c5cc3ef025bff37807067e077bf2be.jpg)  
Figure 2: Large distributional changes concentrate at a few token positions. KL divergence between next-token distributions before and after intervention on GLM-5 and Qwen3-32B. Panels (a) and (c) show the distribution of token-level KL values; panels (b) and (d) show values along example jailbreak trajectories. Most positions exhibit small changes, with occasional large shifts, motivating selective decoding-time control.

$\begin{array} { r } { c _ { v } = \sum _ { k = 1 } ^ { K } \mathbf { 1 } [ v _ { k } = v ] } \end{array}$ . We use a global-uniform prior $q ( v ) = 1 / | \nu |$ and a symmetric Dirichlet update with total prior strength $\kappa > 0 :$

$$
\widehat { p } _ { t } ( v ) = \frac { c _ { v } + \kappa q ( v ) } { K + \kappa } = \frac { c _ { v } + \kappa / | \mathcal { V } | } { K + \kappa } , \qquad \ell _ { t } [ v ] = \log \widehat { p } _ { t } ( v ) .\tag{2}
$$

The prior assigns positive probability to every vocabulary item and is independent of the prompt and prefix. Here, κ is the total prior mass rather than a per-action pseudocount. We use the same estimator to construct training-cache inputs and inference-time inputs.

We collect samples at a fixed prefix through parallel, independent single-choice HTTP requests, without relying on an API-specific multi-sample primitive. All K valid actions are collected before reconstruction. Under the exact-sampling policy, failed or unmappable responses are refilled until exactly $K$ valid samples are obtained; exhausting the retry budget aborts that reconstruction rather than silently using fewer samples. Non-retryable API errors terminate the request path.

Our main configuration uses $K = 5 0$ and $\kappa = 2$ , with sampling temperature 1 and $\mathrm { t o p } { - } p = 1$ . One sampled action is retained per valid response, but the API output-token budget is provider-dependent and need not equal one. A larger budget may be required to obtain usable text, and empty lengthtruncated responses may trigger adaptive retries with an increased budget. Mapping returned text to a local action is distinct from controlling the provider’s output-token budget. Generation audits record valid sample counts and execution statistics; provider-specific metadata are retained where available.

## 3.2 RISK-GATED RESIDUAL CONTROL

At each ordinary single-action step, we first query the target deterministically to obtain a base candidate $b _ { t }$ . After warm-up, we score the resulting prefix $\left( x , y _ { < t } b _ { t } \right)$ with a prefix-risk model $R _ { \psi }$ , whose sigmoid output is

$$
r _ { t } = \sigma ( R _ { \psi } ( x , y _ { < t } b _ { t } ) ) \in [ 0 , 1 ] .\tag{3}
$$

Here, larger values indicate that the candidate prefix is more likely to have entered an unsafe trajectory. Because the attack controller is needed primarily while the response remains safe or refuses the request, a hard gate would apply BiasNet when $r _ { t } < \tau$ . We instead use a sigmoid residual scale with an efficiency cutoff and a full-strength warm-up:

$$
s _ { t } = \sigma \biggl ( \frac { \tau - r _ { t } } { T } \biggr ) , \qquad \tilde { s } _ { t } = \left\{ \begin{array} { l l } { 1 , } & { 1 \le t \le W , } \\ { s _ { t } { \bf 1 } \bigl [ s _ { t } > s _ { \mathrm { m i n } } \bigr ] , } & { t > W , } \end{array} \right.\tag{4}
$$

where $\tau$ is the sigmoid midpoint, $T > 0$ is the gate temperature, $0 < s _ { \mathrm { m i n } } < 1$ is an efficiency cutoff, and $W$ is the number of full-strength warm-up steps. The sigmoid provides a smooth residual scale, whereas the cutoff determines whether the controller is executed. During warm-up, we bypass risk scoring and set the residual scale to one. When $\tilde { s } _ { t } = 0$ , the controller accepts the base action $b _ { t }$ without requesting additional samples. Otherwise, it reconstructs the current next-action distribution and invokes BiasNet.

After warm-up, the execution condition can be written explicitly as

$$
s _ { t } > s _ { \mathrm { m i n } } \quad \Longleftrightarrow \quad r _ { t } < \tau _ { \mathrm { e x e c } } = \tau - T \log \frac { s _ { \mathrm { m i n } } } { 1 - s _ { \mathrm { m i n } } } .\tag{5}
$$

Thus, τ sets the midpoint of the residual scale $( s _ { t } = 0 . 5$ when $r _ { t } = \tau )$ , whereas $\tau _ { \mathrm { e x e c } }$ determines whether reconstruction and BiasNet are executed. With $\tau = 0 . 1 , T = 0 . 0 5$ , and $s _ { \mathrm { m i n } } = 0 . 0 1$ , the effective execution threshold is approximately 0.3298. A hard gate with threshold 0.1 therefore has a different activation boundary: the soft configuration changes both the residual magnitude and the set of risk scores at which intervention is permitted.

Given the reconstructed log probabilities $\boldsymbol { \ell } _ { t } \in \mathbb { R } ^ { | \nu | }$ from Section 3.1, we apply a scaled residual to select the next action. BiasNet is a learned residual transformation $B _ { \phi }$ operating in the same vocabulary space. The controlled logits are

$$
z _ { t } = \ell _ { t } + \tilde { s } _ { t } \mathcal { B } _ { \phi } ( \ell _ { t } ) , \qquad y _ { t } = \arg \operatorname* { m a x } _ { v \in \mathcal { V } } z _ { t } [ v ]\tag{6}
$$

for the greedy action selection used in our experiments. The reconstructed input remains stochastic because it is obtained from samples. More generally, $y _ { t }$ can be sampled from softmax $( z _ { t } / \gamma )$ for decoding temperature $\gamma > 0$ . When active, the gate scales an intervention on the target’s re constructed distribution rather than replacing the target with an independent generator. The target remains responsible for proposing the local candidate and for all ungated steps.

The risk model is evaluated on the candidate prefix rather than on the prompt alone. This makes the decision stateful: the same user prompt can receive different intervention strengths at different generation steps as the answer prefix evolves. The full-strength warm-up avoids gating decisions based on an empty or extremely short answer prefix.

Soft-gated training. We train BiasNet on cached reference-answer prefixes using the same residual scale as at inference, including warm-up and the cutoff. For a reference prefix $\boldsymbol y _ { < t } ^ { * }$ , the cache stores the reconstructed log probabilities $\ell _ { t }$ and the risk score of the deterministic base candidate appended to that prefix. Let ${ \boldsymbol y } _ { t } ^ { * }$ denote the reference next-token label and $w _ { t } = \mathbf { 1 } [ \tilde { s } _ { t } > 0 ]$ . The cross-entropy objective over cached positions is

$$
\mathcal { L } ( \phi ) = - \frac { 1 } { \sum _ { t } w _ { t } } \sum _ { t } w _ { t } \log [ \mathrm { s o f t m a x } ( \ell _ { t } + \tilde { s } _ { t } \mathcal { B } _ { \phi } ( \ell _ { t } ) ) ] _ { y _ { t } ^ { * } } .\tag{7}
$$

Positions with zero scale are excluded from the loss. The scale already attenuates gradients through the residual, so we do not multiply the loss by the scale a second time. The prefix-risk model is fixed: its cached scores are not updated during BiasNet training. Training uses reference prefixes, whereas inference uses generated prefixes; the shared reconstruction and gate rules do not remove this difference in prefix distributions.

## 3.3 SPECULATIVE MULTI-TOKEN EXECUTION

Although samples for distribution reconstruction can be collected in parallel, generation remain autoregressive because the next prefix depends on the selected action. We therefore introduce a

Table 1: Jailbreak performance across different target models and benchmarks. Higher Harm Score and Harm Info Score indicate stronger jailbreak effectiveness. Bold denotes the highest mean for each target, benchmark, and metric. BLINDBIAS uses the soft-gated, global-uniform configuration throughout.
<table><tr><td rowspan="2">Target Model</td><td rowspan="2">Method</td><td colspan="2">AdvBench</td><td colspan="2">HarmBench</td><td colspan="2">SORRY-Bench</td></tr><tr><td>Harm</td><td>Info</td><td>Harm</td><td>Info</td><td>Harm</td><td>Info</td></tr><tr><td rowspan="3">GLM-5</td><td>PAIR GPTFuzz</td><td>1.34 3.42</td><td>0.80 2.41</td><td>1.73</td><td>0.95</td><td>1.69</td><td>0.83</td></tr><tr><td>LogiBreak</td><td>2.58</td><td>1.32</td><td>2.81 2.11</td><td>1.89 1.04</td><td>2.40 2.55</td><td>1.59</td></tr><tr><td>FlipAttack</td><td>4.11</td><td>2.82</td><td>3.89</td><td>2.39</td><td>4.45</td><td>1.23 2.71</td></tr><tr><td rowspan="3">Gemini-3.5-Flash</td><td>BLINDBIAS PAIR</td><td>4.29</td><td>2.88</td><td>4.14</td><td>2.74</td><td>3.83</td><td>2.50</td></tr><tr><td>GPTFuzz</td><td>1.36 2.99</td><td>0.57 1.80</td><td>1.71 2.62</td><td>0.75</td><td>1.62</td><td>0.77</td></tr><tr><td>LogiBreak</td><td>1.59</td><td>0.87</td><td>1.56</td><td>1.68 0.79</td><td>1.81 1.55</td><td>1.11 0.83</td></tr><tr><td rowspan="3"></td><td>FlipAttack</td><td>1.44</td><td>0.58</td><td>1.64</td><td>0.84</td><td>1.51</td><td>0.70</td></tr><tr><td>BLINDBIAS</td><td>3.58</td><td>2.27</td><td>3.57</td><td>2.39</td><td>3.67</td><td>2.30</td></tr><tr><td>PAIR</td><td>2.37</td><td>1.95</td><td>2.45</td><td>2.12</td><td>2.48</td><td>1.97</td></tr><tr><td rowspan="3">Qwen3-32B</td><td>GPTFuzz LogiBreak</td><td>2.44</td><td>1.78</td><td>2.59</td><td>1.80</td><td>2.54</td><td>1.76</td></tr><tr><td>FlipAttack</td><td>2.84</td><td>1.73</td><td>2.45</td><td>1.59</td><td>2.75</td><td>1.64</td></tr><tr><td>BLINDBIAS</td><td>2.97 3.03</td><td>1.39 2.16</td><td>2.3 2.68</td><td>0.81 1.97</td><td>2.55 2.91</td><td>1.08 2.15</td></tr><tr><td rowspan="4">Kimi-K2.5</td><td>PAIR</td><td>3.10</td><td>2.00</td><td>2.75</td><td>2.30</td><td>3.40</td><td>2.30</td></tr><tr><td>GPTFuzz</td><td>1.00</td><td>0.00</td><td>1.08</td><td>0.11</td><td>1.15</td><td>0.20</td></tr><tr><td>LogiBreak</td><td>4.25</td><td>2.35</td><td>3.20</td><td>1.75</td><td>3.10</td><td>1.70</td></tr><tr><td>FlipAttack</td><td>3.70</td><td>2.00</td><td>3.15</td><td>2.50</td><td>2.95</td><td>2.35</td></tr><tr><td rowspan="2"></td><td>BLINDBIAS</td><td></td><td>2.49</td><td>3.31</td><td>2.54</td><td>3.55</td><td>2.47</td></tr><tr><td></td><td>3.42</td><td></td><td></td><td></td><td></td><td></td></tr></table>

speculative path for stretches in which the gate repeatedly suppresses the BiasNet residual. Let q be the number of consecutive steps for which $\tilde { s } _ { t } = 0$ . Once $q$ reaches a threshold $q _ { \mathrm { m i n } }$ , the target is asked to produce a bounded draft

$$
\begin{array} { r } { d _ { 1 : L } = \mathrm { D e c o d e } _ { \theta } ( x , y _ { < t } ; L ) , \qquad L \leq \operatorname* { m i n } \{ L _ { \mathrm { m a x } } , B - | y _ { < t } | \} , } \end{array}\tag{8}
$$

in one multi-token request, where B is the total generation budget. In our deterministic decoding setting, Decode<sub>θ</sub> uses zero temperature and unit top-p.

We then construct every draft prefix $\scriptstyle y _ { < t } d \leq j$ and score these prefixes with the risk model in a local minibatch. Let $s _ { t + j - 1 }$ be the soft-gate scale for draft action $d _ { j }$ , with $j = 1 , \dots , L$ . The longest accepted prefix is

$$
J = \operatorname* { m a x } \left\{ j \in \left\{ 0 , \ldots , L \right\} : s _ { t + i - 1 } \leq s _ { \operatorname* { m i n } } \mathrm { ~ f o r ~ a l l ~ } i \leq j \right\} .\tag{9}
$$

The first draft action whose scale exceeds the cutoff is not committed; that action and the remainder of the draft are discarded. The controller then performs reconstruction and BiasNet selection at the current committed prefix, using the scale computed for the rejected draft candidate. If no violation is found, the entire draft is committed. Execution then returns to an ordinary single-step decision before checking whether another draft can be requested. Verification applies the same gate rule to every accepted draft prefix. It does not establish action-for-action equivalence with repeated single action API calls: a multi-token request may produce different candidates, even at zero temperature.

In the current configuration, $q _ { \mathrm { m i n } } = 2 , L _ { \mathrm { m a x } } = 8 0$ , and risk-prefix verification uses a local batch size of 8. The speculative request itself is not used as a reconstruction sample and does not modify the learned BiasNet. Its purpose is to amortize base-model calls over spans where the residual is suppressed. Generation audits retain draft lengths, accepted and rejected token counts, rollback offsets, and per-prefix risk scores.

The complete inference procedure is provided in Algorithm 1 in Appendix A.2.

Table 2: Contextual-prior transfer to Qwen3-32B. PPL is reported within the proxy-calibration protocol; Harm and Info are averaged over 100 AdvBench prompts.
<table><tr><td>Proxy prior</td><td>Relation</td><td>PPL↓</td><td>Harm ↑</td><td>Info ↑</td></tr><tr><td>Qwen3-1.7B</td><td>same family</td><td>3.86</td><td>3.90</td><td>2.83</td></tr><tr><td>SmolLM2-1.7B</td><td>foreign</td><td>3.93</td><td>3.02</td><td>2.12</td></tr><tr><td>Gemma-3-1B</td><td>foreign</td><td>3.94</td><td>3.38</td><td>2.39</td></tr><tr><td>Shuffled prior</td><td>control</td><td>7.11</td><td>2.69</td><td>2.16</td></tr></table>

Table 3: Distribution-signal recovery and downstream attack quality on Qwen3-32B. Predictive metrics use held-out events from within-prefix sample splits; Harm and Info are averaged over 100 AdvBench prompts.
<table><tr><td rowspan="2">Method</td><td colspan="2">Reconstruction quality</td><td colspan="2">Attack quality</td></tr><tr><td>PPL↓</td><td>Unseen NLL ↓</td><td>Harm ↑</td><td>Info ↑</td></tr><tr><td>Smoothed empirical counts</td><td>11.85</td><td>21.14</td><td>1.50</td><td>1.26</td></tr><tr><td>Uniform prior</td><td>7.07</td><td>14.50</td><td>3.03</td><td>2.16</td></tr><tr><td>Global unigram prior</td><td>5.81</td><td>12.34</td><td>2.78</td><td>1.94</td></tr><tr><td>Numerical log probabilities (ref.)</td><td>3.17</td><td>7.38</td><td>4.06</td><td>3.08</td></tr></table>

## 3.4 EXTENSION TO AN EMPIRICAL STRING ACTION SPACE

The main framework uses a local tokenizer to define its output actions. We also examine the setting in which neither the target vocabulary nor its tokenizer is available, using literal strings returned by one-token completions as empirical actions. From a calibration cache, we construct an empirical action space

$$
\mathcal { V } _ { \mathrm { e m p } } = S _ { \mathrm { s a m p l e } } \cup S _ { \mathrm { l a b e l } } \cup \{ \mathrm { E O S } , \mathrm { O O V } \} ,\tag{10}
$$

where $S _ { \mathrm { s a m p l e } }$ contains one-token strings returned by the target and $S _ { \mathrm { l a b e l } }$ contains reference-answer strings under a public label tokenizer, ensuring that every training label is representable. Counts are accumulated directly by string equality. A public proxy with its own tokenizer supplies a dense prior over this space through

$$
\begin{array} { r } { q _ { \mathrm { p r o x y } } ( s \mid x , y _ { < t } ) \propto p _ { \mathrm { p r o x y } } ( \mathrm { f i r s t t o k } _ { \mathrm { p r o x y } } ( s ) \mid x , y _ { < t } ) , \qquad s \in \mathcal { V } _ { \mathrm { e m p } } , } \end{array}\tag{11}
$$

which is normalized over empirical actions and fused with target counts by the Dirichlet update $\widehat { p _ { t } } ( s ) = ( c _ { s } + \kappa q _ { \mathrm { p r o x y } } ( s ) ) / ( \bar { K _ { + } } \kappa )$ . This transfer variant replaces the main configuration’s uniform prior with a proxy prior. The selected action is appended as text, so neither target token IDs nor a target tokenizer are used. This construction is necessarily approximate: it cannot emit a target token string that never appears in calibration, and Eq. equation 11 retains only the first proxy token of a possibly multi-token string. We therefore evaluate it as a vocabulary-free transfer stress test, separate from the main tokenizer-based evaluation.

## 4 EXPERIMENTS

## 4.1 DATASETS AND EVALUATION METRICS

Datasets. We evaluate on three English harmful-request benchmarks. AdvBench contains 520 harmful goals paired with affirmative target prefixes (Zou et al., 2023). We use the 320-behavior text test split of HarmBench, which includes both standard and contextual behaviors across multiple harm categories (Mazeika et al., 2024). Finally, we use the 440 base prompts of SORRY-Bench, balanced over 44 fine-grained safety categories (Xie et al., 2025).

Evaluation metrics. We evaluate responses using Harm Score and Harm Info Score, both assigned by Gemini-3.5-Flash using the exact evaluation prompt templates from Wang et al. (2026). Higher scores indicate greater attack effectiveness. The sources for these prompts are provided in Appendix A.1.

Table 4: Quality–intervention trade-off of complete gating pipelines on 100 AdvBench prompts towards Gemini-3.5-Flash. Active positions measure BiasNet invocation frequency.
<table><tr><td>Pipeline</td><td>Harm ↑</td><td>Info ↑</td><td>API calls ↓</td><td>Active pos. ↓</td></tr><tr><td>Ungated</td><td>3.96</td><td>2.81</td><td>4000</td><td>100.0%</td></tr><tr><td>Hard gate</td><td>3.44</td><td>2.19</td><td>448</td><td>9.5%</td></tr><tr><td>Run-time soft</td><td>3.58</td><td>2.27</td><td>283</td><td>5.25%</td></tr></table>

## 4.2 IMPLEMENTATION DETAILS

Target models. We access Gemini-3.5-Flash through Google’s native API and GLM-5, Qwen3- 32B, and Kimi-K2.5 through the OpenRouter API. For BLINDBIAS generation, response caching is disabled and hidden reasoning tokens are rejected, so each accepted reconstruction sample corresponds to an observable next action. Base requests use temperature 0 and top-p = 1, with a total generation budget of 80 local tokens.

Baselines. We compare against four black-box prompt-level attacks using their official implementations and settings: PAIR (Chao et al., 2023), GPTFuzz (Yu et al., 2024), LogiBreak (Peng et al., 2026), and FlipAttack (Liu et al., 2026). Unless a method requires stochastic search, target decoding uses temperature 0 and top-p = 1. Full implementation details are provided in Appendix A.2.

## 4.3 MAIN RESULTS

Effective control from sampled text. Table 1 shows that BLINDBIAS achieves the highest mean score in 20 of 24 comparisons against four prompt-level baselines. It uses a fixed configuration with a separately trained controller for each target, and its gains span both Harm and Info. These results indicate that sampled outputs can provide a useful signal for decoding-time control without target weights or numerical token probabilities.

The largest gains occur on Gemini-3.5-Flash. BLINDBIAS leads both metrics on all three Gemini-3.5-Flash benchmarks. On SORRY-Bench, it improves over the strongest baseline by 1.86 Harm points and 1.19 Info points. The benefit of sample-based control therefore varies across targets, and the advantage is not universal: FlipAttack leads both metrics on GLM-5 SORRY-Bench.

Harm and informativeness capture different outcomes. The evaluation criteria do not always rank attacks identically. On Kimi-K2.5 AdvBench, LogiBreak achieves a higher Harm score, whereas BLINDBIAS achieves a higher Info score. On Qwen3-32B HarmBench, BLINDBIAS leads Harm while PAIR leads Info. These differences motivate evaluating both dimensions when comparing jailbreak effectiveness. Overall, the results provide evidence of effectiveness across the evaluated settings while revealing target- and criterion-specific differences. The following analyses examine reconstruction quality and the cost of selective intervention.

## 4.4 ABLATION STUDIES AND ANALYSIS

We examine how reconstruction and selective execution affect control quality. Unless stated otherwise, experiments use the same 40-record training cache and 100 held-out AdvBench prompts with Qwen3-32B as the target. Full diagnostic protocols are provided in Appendix A.3.

Distribution Reconstruction and Control Quality. We compare smoothed empirical counts, the uniform prior used by BLINDBIAS, and a global unigram prior against numerical log probabilities as a reference. Table 3 reports predictive metrics on held-out samples and downstream attack scores; the numerical reference lies outside the sample-only setting.

Finite sampling weakens the control signal, but a prior recovers part of the loss. The uniform prior lowers PPL from 11.85 to 7.07 and raises Harm/Info from 1.50/1.26 to 3.03/2.16. The unigram prior improves predictive fit further, yet yields lower attack scores than the uniform prior. Thus, reconstruction accuracy and steering utility are related but not interchangeable. Numerical probabilities remain the strongest reference. Sampling and calibration details appear in Appendix A.3.1. We next examine whether a contextual prior provides a more useful control signal.

Contextual-Prior Transfer. We next examine whether contextual priors transfer across model families. Keeping the Qwen3-32B target and controller fixed, we compare a same-family Qwen3-1.7B proxy with SmolLM2-1.7B and Gemma-3-1B proxies, using a shuffled prior as a negative control. All priors are calibrated on held-out samples. The target tokenizer still defines the output actions; mapping and calibration details appear in Appendix A.3.2.

Table 2 shows that the same-family proxy performs best on all three metrics. Both cross-family proxies improve Harm over the shuffled control, while only Gemma also improves Info. Their nearly identical PPL values nevertheless yield different attack scores, reinforcing that predictive fit alone does not determine steering utility. Contextual priors can therefore supply useful control signals across families, although the same-family advantage suggests that model and tokenizer compatibility still matter.

Selective Control and Intervention Frequency. Table 4 compares complete gating pipelines on Gemini-3.5-Flash. Hard gating uses an ungated-trained controller with a binary execution gate; the soft pipeline is trained and executed with the scaled residual. Active positions measure BiasNet invocation frequency; API calls are averaged for each generated response.

Soft gating improves on hard gating while intervening less often: active positions fall from 9.5% to 5.25% and API calls decreased by 36.8%. Compared with ungated control, the soft pipeline reduces average API calls from 4,000 to 283 (92.9%), while the Harm Score decreases from 3.96 to 3.58. Selective control therefore trades effectiveness for fewer intervention steps. Active-position frequency alone does not establish endpoint query savings, which also depend on sampling, retries, and response length. Additional gate-aware training analyses are provided in Appendices A.3.3.

## 5 CONCLUSION

We introduced BLINDBIAS, a framework for decoding-time jailbreaking through text-only continuation interfaces. By combining sample-based distribution reconstruction, prefix-dependent residual control, and speculative multi-token execution, it extends decoding-time control to settings without target weights or numerical token probabilities. Our experiments across four targets and three benchmarks provide evidence that sampled outputs can support effective distributional control, while our analyses show that selective intervention trades attack effectiveness for less frequent controller invocation. These findings demonstrate the feasibility of sample-only control while highlighting its remaining costs and interface assumptions. Improving endpoint-level query efficiency and extending reliable control beyond tokenizer-based action spaces remain important directions. More broadly, withholding numerical probabilities alone may not close the decoding-time attack surface when an interface permits repeated sampling and continuation from supplied prefixes.

## REFERENCES

Maksym Andriushchenko, Francesco Croce, and Nicolas Flammarion. Jailbreaking leading safetyaligned LLMs with simple adaptive attacks. In International Conference on Learning Representations (ICLR), 2025.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022.

Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J. Pappas, and Eric Wong. Jailbreaking black box large language models in twenty queries. arXiv preprint arXiv:2310.08419, 2023.

Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, and John Jumper. Accelerating large language model decoding with speculative sampling. arXiv preprint arXiv:2302.01318, 2023.

Yiqun Chen, Jinyuan Feng, Wei Yang, Meizhi Zhong, Zhengliang Shi, Rui Li, Xiaochi Wei, Yan Gao, Yi Wu, Yao Hu, et al. Self-compression of chain-of-thought via multi-agent reinforcement learning. arXiv preprint arXiv:2601.21919, 2026.

Sumanth Dathathri, Andrea Madotto, Janice Lan, Jane Hung, Eric Frank, Piero Molino, Jason Yosinski, and Rosanne Liu. Plug and play language models: A simple approach to controlled text generation. In International Conference on Learning Representations (ICLR), 2020.

Wei Ji, Li Li, Zheqi Lv, Wenqiao Zhang, Mengze Li, Zhen Wan, Wenqiang Lei, and Roger Zimmermann. Backpropagation-free multi-modal on-device model adaptation via cloud-device collaboration. ACM Trans. Multimedia Comput. Commun. Appl., 21(2), January 2025. ISSN 1551-6857. doi: 10.1145/3706422. URL https://doi.org/10.1145/3706422.

Ben Krause, Akhilesh Deepak Gotmare, Bryan McCann, Nitish Shirish Keskar, Shafiq Joty, Richard Socher, and Nazneen Fatema Rajani. GeDi: Generative discriminator guided sequence generation. In Findings ofEMNLP, 2021.

Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast inference from transformers via speculative decoding. In International Conference on Machine Learning (ICML), 2023.

Jiate Li, Defu Cao, Li Li, Wei Yang, Yuehan Qin, Chenxiao Yu, Tiannuo Yang, Ryan A. Rossi, Yan Liu, Xiyang Hu, and Yue Zhao. “someone hid it!”: Query-agnostic black-box attacks on LLM-based retrieval. In Forty-third International Conference on Machine Learning, 2026a. URL https://openreview.net/forum?id=bzmt9wJ6uW.

Li Li, Wei Ji, Yiming Wu, Mengze Li, You Qin, Lina Wei, and Roger Zimmermann. Panoptic scene graph generation with semantics-prototype learning. AAAI, 38(4):3145–3153, Mar. 2024. doi: 10.1609/aaai.v38i4.28098.

Shawn Li and Yue Zhao. The autonomy tax: Defense training breaks llm agents, 2026. URL https://arxiv.org/abs/2603.19423.

Shawn Li, Peilin Cai, Yuxiao Zhou, Zhiyu Ni, Renjie Liang, You Qin, Yi Nian, Zhengzhong Tu, Xiyang Hu, and Yue Zhao. Secure on-device video ood detection without backpropagation. In ICCV, October 2025a.

Shawn Li, Huixian Gong, Hao Dong, Tiankai Yang, Zhengzhong Tu, and Yue Zhao. Dpu: Dynamic prototype updating for multimodal out-of-distribution detection. In CVPR, pp. 10193–10202, June 2025b.

Shawn Li, You Qin, Jiate Li, Charith Peris, Lisa Bauer, Roger Zimmermann, and Yue Zhao. Geometry over density: Few-shot cross-domain ood detection, 2026b. URL https://arxiv.org/ abs/2605.03410.

Shawn Li, Chenxiao Yu, Zhiyu Ni, Hao Li, Charith Peris, Chaowei Xiao, and Yue Zhao. Defenses against prompt attacks learn surface heuristics. In ACL, 2026c.

Shawn Li, Chenxiao Yu, Han Wang, Wei Yang, Ryan Rossi, Franck Dernoncourt, Xiyang Hu, Philip Yu, Chaowei Xiao, Huan Zhang, and Yue Zhao. Fortis: Benchmarking over-privilege in agent skills, 2026d. URL https://arxiv.org/abs/2605.09163.

Alisa Liu, Xiaochuang Han, Yizhong Wang, Yulia Tsvetkov, Yejin Choi, and Noah A. Smith. Tuning language models by proxy. In Conference on Language Modeling (COLM), 2024a.

Xiaogeng Liu, Nan Xu, Muhao Chen, and Chaowei Xiao. AutoDAN: Generating stealthy jailbreak prompts on aligned large language models. In International Conference on Learning Representations (ICLR), 2024b.

Yue Liu, Xiaoxin He, Miao Xiong, Jinlan Fu, Shumin Deng, Yingwei Ma, Jiaheng Zhang, and Bryan Hooi. Flipattack: Jailbreak llms via flipping, 2026. URL https://arxiv.org/abs/2410. 02832.

Mantas Mazeika, Long Phan, Xuwang Yin, Andy Zou, Zifan Wang, Norman Mu, Elham Sakhaee, Nathaniel Li, Steven Basart, Bo Li, David Forsyth, and Dan Hendrycks. Harmbench: A standardized evaluation framework for automated red teaming and robust refusal, 2024. URL https://arxiv.org/abs/2402.04249.

Anay Mehrotra, Manolis Zampetakis, Paul Kassianik, Blaine Nelson, Hyrum Anderson, Yaron Singer, and Amin Karbasi. Tree of attacks: Jailbreaking black-box LLMs automatically. arXiv preprint arXiv:2312.02119, 2023.

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

Jingyu Peng, Maolin Wang, Nan Wang, Jiatong Li, Yuchen Li, Yuyang Ye, Wanyu Wang, Pengyue Jia, Kai Zhang, and Xiangyu Zhao. Logic jailbreak: Efficiently unlocking llm safety restrictions through formal logical expression, 2026. URL https://arxiv.org/abs/2505.13527.

Xiangyu Qi, Yi Zeng, Tinghao Xie, Pin-Yu Chen, Ruoxi Jia, Prateek Mittal, and Peter Henderson. Fine-tuning aligned language models compromises safety, even when users do not intend to!, 2023. URL https://arxiv.org/abs/2310.03693.

Xiangyu Qi, Ashwinee Panda, Kaifeng Lyu, Xiao Ma, Subhrajit Roy, Ahmad Beirami, Prateek Mittal, and Peter Henderson. Safety alignment should be made more than just a few tokens deep. In International Conference on Learning Representations (ICLR), 2025.

Yuehan Qin, Li Li, Linxin Song, Wei Yang, Jiate Li, Yuqing Yang, and Yue Zhao. Memory retrieval for changing preferences, 2026a. URL https://arxiv.org/abs/2606.02976.

Yuehan Qin, Shawn Li, Yi Nian, Xinyan Velocity Yu, Yue Zhao, and Xuezhe Ma. Don’t let it hallucinate: Premise verification via retrieval-augmented logical reasoning, 2026b. URL https: //arxiv.org/abs/2504.06438.

Mark Russinovich, Ahmed Salem, and Ronen Eldan. Great, now write an article about that: The crescendo multi-turn LLM jailbreak attack. arXiv preprint arXiv:2404.01833, 2024.

Li Shawn, Jiashu Qu, Linxin Song, Yuxiao Zhou, Yuehan Qin, Tiankai Yang, and Yue Zhao. Treble counterfactual VLMs: A causal approach to hallucination. In EMNLP, November 2025.

Jesson Wang, Zhanhao Hu, and David Wagner. Juli: Jailbreak large language models by selfintrospection, 2026. URL https://arxiv.org/abs/2505.11790.

Tinghao Xie, Xiangyu Qi, Yi Zeng, Yangsibo Huang, Udari Madhushani Sehwag, Kaixuan Huang, Luxi He, Boyi Wei, Dacheng Li, Ying Sheng, Ruoxi Jia, Bo Li, Kai Li, Danqi Chen, Peter Henderson, and Prateek Mittal. Sorry-bench: Systematically evaluating large language model safety refusal, 2025. URL https://arxiv.org/abs/2406.14598.

Kevin Yang and Dan Klein. FUDGE: Controlled text generation with future discriminators. In Conference of the North American Chapter of the Association for Computational Linguistics (NAACL), 2021.

Wei Yang and Jesse Thomason. Learning to deliberate: Meta-policy collaboration for agentic llms with multi-agent reinforcement learning. arXiv preprint arXiv:2509.03817, 2025.

Wei Yang, Shixuan Li, Heng Ping, Peiyu Zhang, Paul Bogdan, and Jesse Thomason. Auditing multi-agent llm reasoning trees outperforms majority vote and llm-as-judge. arXiv preprint arXiv:2602.09341, 2026.

Dongze Ye, Haipeng Luo, Carolee Winstein, and Nicolas Schweighofer. Towards ai-based precision rehabilitation via contextual model-based reinforcement learning. Journal of NeuroEngineering and Rehabilitation, 22(1):263, 2025.

Wen Ye, Yuxiao Qu, Aviral Kumar, and Xuezhe Ma. Mirror: Learning from the other view for multi-modal reasoning. arXiv preprint arXiv:2607.21552, 2026a.

Wen Ye, Wei Yang, Defu Cao, Yizhou Zhang, Lumingyuan Tang, Jie Cai, and Yan Liu. Ts-reasoner: Domain-oriented time series inference agents for reasoning and automated analysis. Transactions on Machine Learning Research, 2026b.

Jiahao Yu, Xingwei Lin, Zheng Yu, and Xinyu Xing. Gptfuzzer: Red teaming large language models with auto-generated jailbreak prompts, 2024. URL https://arxiv.org/abs/2309. 10253.

Youliang Yuan, Wenxiang Jiao, Wenxuan Wang, Jen-tse Huang, Pinjia He, Shuming Shi, and Zhaopeng Tu. GPT-4 is too smart to be safe: Stealthy chat with LLMs via cipher. In International Conference on Learning Representations (ICLR), 2024.

Xuandong Zhao, Xianjun Yang, Tianyu Pang, Chao Du, Lei Li, Yu-Xiang Wang, and William Yang Wang. Weak-to-strong jailbreaking on large language models. In International Conference on Machine Learning (ICML), 2025.

Zhanhui Zhou, Jie Liu, Jing Dong, Jiaheng Liu, Chao Yang, Wanli Ouyang, and Yu Qiao. Emulated disalignment: Safety alignment for large language models may backfire! In Annual Meeting of the Association for Computational Linguistics (ACL), 2024.

Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, and Matt Fredrikson. Universal and transferable adversarial attacks on aligned language models. arXiv preprint arXiv:2307.15043, 2023.

## A ADDITIONAL EXPERIMENTAL DETAILS

## A.1 EVALUATION METRICS

We evaluate responses using Harm Score and Harm Info Score, both assigned by Gemini-3.5-Flash. Harm Score measures the overall harmfulness of a model response, capturing the extent to which the response complies with and facilitates the harmful intent of the input. Harm Info Score further evaluates the amount and informativeness of harmful content contained in the response, emphasizing how much concrete and useful harmful information is provided. We use the Harm Score evaluation prompt from Qi et al. (2023) and the Harm Info Score prompt from Wang et al. (2026). Higher scores indicate stronger attack effectiveness under the respective criteria.

## A.2 IMPLEMENTATION DETAILS

Tokenizers. We use za $\mathtt { i } - \mathtt { o r g } / \mathtt { G L M } - 5$ for GLM-5, Qwen/Qwen3-32B for Qwen3-32B, moonshotai/Kimi-K2.5 for Kimi-K2.5, and google/gemma-3-1b-pt for Gemini-3.5- Flash. These tokenizers define the local action IDs, reconstruction vocabulary, and decoded text appended to the response prefix. In particular, the Gemma tokenizer provides local coordinates for Gemini outputs; its tokens need not coincide with Gemini’s internal tokens. The generation budget is measured in these local tokens, while each API request has a provider-side output budget.

BLINDBIAS configuration. All main-result BLINDBIAS rows use a fixed configuration: softgated training and inference together with a global-uniform prior. Concretely, every controlled position uses $K = 5 0$ independent sampled next actions at temperature 1 and $\scriptstyle { \mathrm { t o p } } - p \ = \ 1$ . Each valid response contributes one action; the API output-token budget is provider-dependent. We add a symmetric Dirichlet prior over the complete vocabulary of the tokenizer used for that target, with strength $\kappa = 2 .$ , selected by held-out sample NLL; thus these rows do not use the global-unigram variant in Section 4.4 or the proxy-fusion variants in Section 4.4. Samples are issued as 50 independent single-choice requests, and the exact-completion policy refills invalid or empty responses until 50 valid samples are collected or the retry budget is exhausted.

For each target, we build the training cache from 40 instruction–answer records (indices 100–139 of the pinned LLM-LAT harmful-data revision), covering every answer prefix, and deterministically hold out eight records. BiasNet uses a 1,024-dimensional, four-hash count-sketch input projection followed by layer normalization and is trained for 10 epochs with cross-entropy, AdamW, batch size 32, learning rate $3 \times 1 0 ^ { - 4 }$ , weight decay $1 0 ^ { - 4 }$ , mixed precision, and seed 42. The local Llama-3.1-8B-Instruct prefix-risk model is used with sigmoid midpoint $\tau = 0 . 1$ , soft-gate temperature $T = 0 . 0 5$ , cutoff $s _ { \mathrm { m i n } } = 0 . 0 1$ , and a three-token full-strength warm-up. At inference we use the identical soft scale, with speculative activation after two consecutive bypassed positions, maximum draft length 80, and risk-prefix batch size 8.

Inference procedure. Algorithm 1 summarizes inference in the tokenizer-based action space. We write $\mathcal { C } ( H ( x , y ) ; \gamma , m )$ for a target call conditioned on the user prompt x and the committed assistant prefix $y ,$ with decoding temperature $\gamma$ and output budget $m .$ The call returns only text; mapping text to actions, scoring prefixes, reconstructing distributions, and applying BiasNet are performed locally.

## A.3 ADDITIONAL ANALYSES

Unless stated otherwise, the diagnostics use the 40-record LLM-LAT cache described in $\mathsf { A p - }$ pendix A.2 and a fixed set of 100 AdvBench prompts with Qwen3-32B as the target. AdvBench prompts are reserved for evaluation. Token selection is greedy conditional on the reconstructed distribution, but reconstruction remains stochastic because samples are drawn at temperature 1.

## A.3.1 RECONSTRUCTION DIAGNOSTICS

The smoothed-count baseline applies additive smoothing to observed counts and assigns a floor mass to unobserved actions. The uniform-prior condition uses Eq. equation 2, while the global unigram prior is estimated from the training cache. Numerical log probabilities provide a reference unavailable under the sample-only threat model. Prior strength is selected without reference-answer labels. For predictive diagnostics, the 50 samples at each cached prefix are split into calibration and held-out halves; downstream generation uses all $K = 5 0$ samples. PPL is evaluated on held-out events, and unseen NLL is restricted to events absent from the calibration half. These predictive diagnostics and downstream attack scores measure different uses of the reconstructed distribution.

Algorithm 1 BLINDBIAS: Risk-Gated Sample-Only Controlled Decoding (Tokenizer-Based Action   
Space)   
Require: User prompt x; text-only continuation API C; action encoder–decoder $( E , D )$ ; trained BiasNet $B _ { \phi } ;$   
prefix-risk model $R _ { \psi }$   
Require: Sample size $K ;$ uniform-prior strength $\kappa ;$ output budget N; gate parameters $( W , \tau , T , s _ { \mathrm { m i n } } ) ;$ ; specu  
lation parameters $( q _ { \mathrm { m i n } } , L _ { \mathrm { m a x } } )$   
Ensure: Controlled assistant response y   
1: $y  \langle \rangle ; t  1 ; h  0$ ▷ h: consecutive cutoff-based bypasses   
2: while $\ddot { t } \leq N$ and EOS has not been emitted do   
3: $b \gets \mathbf { B A S E } ( \mathcal { C } , x , y )$   
4: $g \gets \mathrm { { G A T E S C A L E } } ( x , y , b , t , R _ { \psi } , W , \tau , T , s _ { \mathrm { { m i n } } } )$ ▷ Eq. equation 4   
5: if $\cdot _ { g } = 0$ then   
6: $a \gets b ; h \gets h + 1$   
7: else   
8: ℓb← RECONSTRUCTDISTRIBUTION $( \mathcal { C } , x , y , E , K , \kappa )$ ▷ Eq. equation 2   
9: a ← arg max<sub>v</sub> $\mathcal { \left[ \widehat { \ell } ( \boldsymbol { v } ) + g B _ { \phi } ( \widehat { \ell } ) ( \boldsymbol { v } ) \right] } ; h \gets 0$   
10: $y  y \circ D ( a ) ; t  t + 1$   
11: if EOS has been emitted or $t > N$ then   
12: break   
13: if $h \geq q _ { \mathrm { m i n } }$ then   
14: $\bar { L } \dot { \gets } \operatorname* { m i n } \{ L _ { \mathrm { m a x } } , N - t + 1 \}$   
15: $d _ { 1 : M } \gets \mathrm { D R A F T } ( \mathcal { C } , x , y , L )$ ▷ $M \leq L { : }$ returned actions   
16: $\mathbf { f } M = 0$ then   
17: break   
18: $( J , g ) \gets \mathsf { V E R I F Y D R A F T } ( x , y , d _ { 1 : M } , t , R _ { \psi } , W , \tau , T , s _ { \mathrm { m i n } } )$ ▷ g: first violating scale, if any   
19: $y  y \circ D ( d _ { \leq J } ) ; t  t + J ; h  h + J$   
20: if EOS has been emitted or $t > N$ then   
21: break   
22: if $J = M$ then   
23: if the draft indicates completion then   
24: break   
25: continue ▷ resume with an ordinary single step   
26: ▷ Discard $d _ { J + 1 : M } ;$ control at the committed prefix   
27: $\widehat { \ell } \gets$ RECONSTRUCTDISTRIBUTION $( \mathcal { C } , x , y , E , K , \kappa )$   
28: a ← arg max<sub>v</sub> $\widehat { \ell ( v ) } + g B _ { \phi } ( \widehat { \ell } ) ( v ) ]$   
29: $y  y \circ D ( a ) ; \dot { t }  t + 1 ; \dot { h }  \dot { 0 }$   
30: return y

Table 5: Gate-aware training diagnostics on 100 AdvBench prompts. Blocks group runs with equal training duration; evaluation weights and support vary by rule. Evaluation weight is summed loss weight rather than a token count.
<table><tr><td>Block</td><td>Training rule</td><td>Epochs</td><td>Eval. weight</td><td>Margin ↑</td><td>Beats base ↑</td></tr><tr><td>A</td><td>No gate</td><td>75</td><td>616.0</td><td>-2.683</td><td>19.7%</td></tr><tr><td>A</td><td>Soft weights</td><td>75</td><td>74.2</td><td>-1.346</td><td>36.6%</td></tr><tr><td>B</td><td>Hard-active only</td><td>10</td><td>14.0</td><td>2.250</td><td>64.3%</td></tr><tr><td>B</td><td>Hard + first 3</td><td>10</td><td>29.0</td><td>1.040</td><td>54.5%</td></tr></table>

## A.3.2 CONTEXTUAL-PRIOR MAPPING AND CALIBRATION

The contextual-prior experiment in Section 4.4 keeps the Qwen3-32B target and controller fixed while varying the proxy model. Qwen3-1.7B supplies a same-family prior; SmolLM2-1.7B and

Gemma-3-1B use different tokenizers. For these cross-family proxies, target-token strings are retokenized by the proxy and scored using their first proxy token. The target tokenizer is still used to enumerate output coordinates. Each proxy is independently temperature-calibrated on held-out sample events, and the shuffled prior supplies a negative control. PPL in Table 2 is reported within this proxy-calibration protocol, while Harm and Info are averaged over the same 100 AdvBench prompts.

## A.3.3 GATE-AWARE TRAINING DIAGNOSTICS

Gate classifier training. The prefix-risk classifier $R _ { \psi }$ is trained separately from BiasNet. The gate checkpoint used in these experiments records a frozen Llama-3.1-8B-Instruct backbone and a trainable classification head operating on the final-layer, last non-padding token representation (4,096 dimensions). The head consists of layer normalization, a 1,024-unit linear layer, SiLU, dropout with probability 0.1, and a scalar linear output. Its sigmoid gives the prefix risk score. Inputs use the backbone’s chat template for the user prompt and assistant header, followed by the answer prefix without an end-of-turn marker; the maximum input length is 1,024 tokens.

The checkpoint uses prebuilt Guard-labeled prefixes from the training split of LLM-LAT/harmful-dataset, rather than the 40-record cache used to train BiasNet. The dataset builder uses Llama-Guard-3-8B to locate an unsafe boundary in each scanned answer: it first checks a coarse grid of prefix lengths, then checks every token position in the interval ending at the first unsafe grid point. Prefixes at or beyond the detected boundary receive label 1, and earlier prefixes receive label 0; if no boundary is detected, all prefixes receive label 0. The default builder trusts the chosen answers as safe and scans the rejected answers, using greedy Guard judgments, a scan stride of four tokens, and an output stride of one token. Thus, labels impose a persistent unsafe state after the detected boundary; they are not independent Guard judgments at every prefix. The training code splits examples by original record ID, keeping both answers and all their prefixes in the same partition.

Only the classification head is optimized, using unweighted binary cross-entropy with logits. The training implementation defaults to a 90/10 record-level train/validation split, seed 42, three epochs, AdamW with learning rate $1 0 ^ { - 4 }$ and weight decay 0.01, batch size 2, and eight-step gradient accumulation (effective batch size 16 for complete accumulation windows). It uses linear learning-rate decay with 3% warm-up and clips the gradient norm at 1.0. Validation runs every 100 optimizer updates and at the end of training; the best checkpoint is selected by validation loss. These opti mization values are code defaults: the retained checkpoint configuration confirms the architecture and labeled-data source but does not preserve the original optimizer arguments or dataset size. The classifier remains fixed during subsequent BiasNet training and generation. In particular, the hard threshold and soft scaling rule in Eq. equation 4 are execution policies applied to its score, rather than separately trained classifier heads.

Selective execution changes which positions receive the BiasNet residual. Training uniformly over all answer tokens can devote most capacity to positions that bypass BiasNet at deployment, whereas an overly restrictive gate leaves little supervision. We study this trade-off using the same 40-record LLM-LAT cache and evaluate the target-over-base logit margin on the fixed 100-prompt AdvBench set. Table 5 groups runs by training duration: the no-gate and soft-weighted variants use 75 epochs, whereas the hard-active and warm-up variants use 10 epochs. “Eval. weight” is the summed eval uation loss weight, not a token count. Evaluation weights and support also differ across rules, including within each block. These scores describe each rule’s deployment-weighted objective and do not isolate training effects on a common evaluation support.

Soft weighting in Block A produces a less negative margin and a higher beats-base rate under its own evaluation weighting. This difference combines changes in the controller with changes in the evaluated support. In Block B, adding the first three positions increases evaluation weight from 14.0 to 29.0 while lowering the average margin over the expanded support. The diagnostic describes how warm-up broadens coverage; it does not establish an end-to-end performance gain. We retain these positions in training because they are forced active at inference.