# Claims of emergence in AI should be grounded in information decomposition

Christoph Riedl Khoury College of Computer Sciences Northeastern University Boston, MA 02115 c.riedl@northeastern.edu

Fernando E. Rosas   
Department of Informatics   
University of Sussex   
Brighton, BN1 9RH   
f.rosas@sussex.ac.uk

## Abstract

When a language model develops capabilities not traceable to its individual components, when a multi-agent AI system outperforms each agent on its own, or when a human-AI pair succeeds where either fails in isolation, AI researchers describe it as a case of “synergy” or “emergence”. However, these are usually informal assessments not supported by a theoretical framework, making these claims impossible to compare or even meaningfully dispute. Here we argue that information decomposition can serve as a unifying formal framework to operationalize assessments about synergy in AI systems. Information decomposition provides a way to distinguish qualitatively different types of information, including unique information available from individual components, redundant information shared across multiple components, and synergistic information accessible only from parts taken together. This gives claims of emergence a precise, falsifiable criterion: a system’s behavior is emergent only if it carries synergistic information about task-relevant outcomes. We apply this lens to attention heads, autonomous agents, and human-AI teams, showing how it organizes existing knowledge across scales and substrates, while also generating cross-level conjectures that siloed fields would miss.

## 1 Introduction

AI research describes some of its most interesting phenomena in the language of emergence. Transformer models develop internal representations irreducible to single neurons [26, 101, 103]; multiagent systems solve tasks that no single agent can [68, 78, 75]; and human-AI teams succeed where either partner fails alone [80, 64, 3]. These cases differ in scale and substrate, yet make the same structural claim: the capabilities of the whole became “more than the sum of its parts.”

Such “more-than-the-sum” effects are often described as synergy. Yet, in current AI research, “synergy” is an umbrella term encompassing several different ideas. In mechanistic interpretability, it may refer to interaction effects between attention heads, non-localized circuits across layers, or superposition of representations [e.g., 66, 33, 21]. In multi-agent systems, synergy corresponds to ensemble gain, enabled by coordination, over what independent agents can achieve [e.g., 13, 68, 75]. In human-AI systems, synergy is tied to substitution, elicitation, prompting, and learning effects [e.g., 102, 110, 63]. These uses are related but not equivalent, and the differences are not mere semantics but a barrier to scientific progress: without a shared formal criterion for what synergy means, these claims cannot be compared, aggregated, or even meaningfully disputed.

In this paper, we argue that Partial Information Decomposition (PID) [107] provides conceptual and computational tools that can organize these disparate notions of synergy into a common, unifying framework. PID proposes to decompose the information that multiple sources carry about a target into information unique to one source, information redundantly shared across sources, and synergistic information available only from the sources jointly (Figure 1a). This gives emergence a precise, mereological meaning: a system’s collective behavior is emergent only if it carries synergistic information [87]. Unlike the scale-dependent “emergent abilities” of Wei et al. [104], which can be artifacts of the evaluation metric [93], this notion applies to any system of interacting components and does not hinge on how performance is scored. Crucially, by treating unique, redundant, and synergistic information asfundamental types ofinformation, PID offers a substrate-agnostic calculus revealing common constraints that drive how information is integrated, whether across silicon neurons, autonomous agents, or human-AI teams. This, in turn, suggests a general pressure for increasingly capable systems to transition from robust redundancy to efficient synergy under resource constraints.

We propose PID—applied either as an exact computation or as a conceptual scaffold—as aframework to conceptualize collective intelligence across scales (within and across agents) and substrates (artificial and human).<sup>1</sup> We argue that PID explains—rather than replaces—existing metrics. Different communities already compute synergy-like quantities—ensemble gains [75], local (point-wise) surprisal [37, 106], residuals over independent-source null models [50], mechanistic null models $[ 7 9 { \bar { ] } } .$ , latent variable models [80], variance-component decompositions of dyadic data [44], and compression-based measures [15]—and PID interprets each as a partial projection onto the same underlying decomposition, clarifying what each identifies even when full PID estimation is intractable.

## 2 Partial Information Decomposition as a General Lens on Interdependence

Here we provide a brief introduction to the mathematical formalism of Partial Information Decomposition (PID) [107] and some of its recent extensions [62]. While this section introduces PID as a mathematical instrument, we also highlight where it serves as a conceptual framework.

## 2.1 Synergy

PID focuses on a deceptively simple question: given a collection of variables representing the parts of a system, how is the information they contain distributed among them? Concretely,

• how much of that information is contained in individual variables alone (unique information),

• how much is contained in several different variables (redundancy), and

• how much is contained in groups of variables without being accessible from any subset of them (synergy)?

Formally, PID considers source variables $X _ { 1 } , \ldots , X _ { n }$ that hold information about a target variable Y (Figure 1a). For example, $X _ { i }$ could represent behavioral time-series of agents in a multi-agent system and Y their performance. The total information that the sources have about the target is captured by Shannon’s mutual information $I ( X _ { 1 } , \ldots , X _ { n } ; Y ) [ 9 4 ]$ . For $n { = } 2$ sources, the joint mutual information decomposes via the chain rule of information into $I ( X _ { 1 } , X _ { 2 } ; Y ) = I ( X _ { 1 } ; Y { \bar { ) } } + I ( X _ { 2 } ; Y \vert X _ { 1 } )$ , where the first term captures the information about $Y$ given by $X _ { 1 } ,$ and the second the additional information given by $X _ { 2 } [ 1 \bar { 8 } ]$ . PID proposes to further decompose these two components into four information atoms [107]:

$$
I ( X _ { 1 } , X _ { 2 } ; Y ) = \underbrace { \mathrm { R e d } ( X _ { 1 } , X _ { 2 } ; Y ) + \mathrm { U n } ( X _ { 1 } ; Y | X _ { 2 } ) } _ { I ( X _ { 1 } ; Y ) } + \underbrace { \mathrm { U n } ( X _ { 2 } ; Y | X _ { 1 } ) + \mathrm { S y n } ( X _ { 1 } , X _ { 2 } ; Y ) } _ { I ( X _ { 2 } ; Y | X _ { 1 } ) } .\tag{1}
$$

This gives three equations and four unknowns. Establishing how to calculate one of the above atoms determines the others. A common choice is to specify their redundancy, Re $\scriptstyle { \mathsf { l } } ( X _ { 1 } , X _ { 2 } ; Y )$ , via the minimum mutual information (MMI) criteria [6]:

$$
\operatorname { R e d } ( X _ { 1 } , X _ { 2 } ; Y ) = \operatorname* { m i n } \{ I ( X _ { 1 } ; Y ) , I ( X _ { 2 } ; Y ) \} .\tag{2}
$$

While this operationalization is canonical for Gaussian variables [6], it may not be appropriate for more general settings—but can still be used as an upper bound for redundancy [83] (see Appendix A.1 for details). The PID literature has proposed several alternatives for addressing more general settings, driven by inspiration from different fields such as neuroscience, cryptography, or game theory, each with their own specific desiderata and motivations [88, 45, 28, 37, 56, 57]. While many of these formulations have been shown to agree in practical scenarios [88], some of their driving desiderata are mutually incompatible [77]. This suggests that none of them is universally preferable [40], and hence practitioners need to choose between operationalizations based on domain knowledge, and properties of the data and problem of interest (e.g., some formulations are only defined for discrete data). For a detailed comparison of PID operationalizations, see [53].

![](images/3f707d73a7ce6661a9886311409615a5aa56f1f289766965e03f9512a10b7be2.jpg)  
Figure 1: a) Information decomposition with two source variables $X _ { 1 }$ and $X _ { 2 } ,$ and target variable Y. b) The PID conceptual triangle resolves ambiguities in system dynamics. Traditional dichotomies in complex systems literature often conflate mathematically distinct properties. Integration conflates redundancy (integration-as-oneness) with synergy (complementary cooperation). Differentiation conflates unique information (strict independence) with synergy (interactive differentiation). Furthermore, locality conflates unique (singularly local) with redundant (multiply local) information, whereas synergistic representations are strictly de-localized (adapted from Luppi et al. [55]).

For $n \geq 3$ source variables the decomposition is much richer, including redundancies and synergies terms of varying orders [107]. In fact, the number of information atoms grows super-exponentially with the number of sources, making the full decomposition intractable for six or more variables [41]. Four practical strategies have been developed to circumvent this issue and many are accessible through software resources (see Table S4):

(i) targeting specific atoms of interest rather than calculating the full decomposition [41];

(ii) evaluating reduced decompositions that scale linearly in the number of sources [76, 84];

(iii) computing the full decomposition over small subsets (e.g., all triplets) and aggregating the resulting atoms across subsets [78, 25]; or

(iv) computing scalar summaries, such as the O-information [27, 86, 96], which quantify the net balance between synergy and redundancy without explicitly disentangling them [91] (biases in O-information estimation are discussed in [30]).

Beyond formal definitions and computational techniques, the distribution of information across PID atoms dictates a fundamental architectural tradeoff that can be used to study a system’s representational capacity and structural robustness [55, 34, 78]: redundant information provides robustness (since it isfound in multiple signals) while synergistic information provides efficiency. As the number of sources increases, the synergistic portion of the decomposition expands combinatorially—the count of higher-order synergy atoms grows super-exponentially, giving synergy far greater representational capacity than redundant or unique information for a given number of components [88, 34]. However, since it depends on joint contributions from multiple sources, synergy is fragile and can be disrupted by the failure of even a single element (however, see also [70]).

This results in a synergy-vs-redundancy tradeoff [55, 54, 88, 74]. By explicitly separating these atoms, PID resolves classical ambiguities in complex systems—such as conflating “integration-as-oneness” (redundancy) with “integration-as-cooperation” (synergy)—clarifying distinct axes of integration, differentiation, and locality (Figure 1b). Thus, even without engaging with the computational aspects of PID, the conceptual toolkit and vocabulary of redundant, unique, and synergistic information allow researchers to navigate important tradeoffs that are central for understanding complex systems.

## 2.2 Emergence

Emergence is a rich topic that can be studied from various perspectives [32, 47, 12, 42], being the subject of long-standing ongoing debate. Complementing the rich philosophical literature [8, 17], PID provides a falsifiable, quantitative, and substrate-agnostic criterion for emergence: a system exhibits emergent properties only ifit carries synergistic information about task-relevant outcomes.

How can one test if a system with multiple parts shows emergent properties? That is, how can one test whether the whole is more than the sum of its parts? Building on the idea of information decomposition, one can cast emergence as information about a system’s temporal evolution (future states of the whole) that cannot be traced to the current state of its parts using time-delayed mutual information (TDMI) [54]. This then gives a data-driven, quantifiable criterion amenable to empirical testing with falsifiable conjectures against specific null hypotheses [78]. Specifically, three tests of emergence have been proposed [61], which we describe next.

Emergence capacity. This first test probes the ability of a system to host any emergent variable. Specifically, for each pair of sources $( i , j )$ and timestep t, let the sources be $X _ { i , t }$ and $X _ { j , t }$ , and define the bivariate target variable as the joint future state at $t + \ell , T _ { i j , t + \ell } \equiv ( X _ { i , t + \ell } , \breve { X } _ { j , t + \ell } )$ where ℓ is the integration timescale [60]. From the joint contingency table over $( X _ { i , t } , X _ { j , t } , T _ { i j , t + \ell } )$ we then compute the two-source PID using Eq. (1). A positive score of the pairwise dynamical synergy $\mathrm { S y n } _ { i j } > 0$ indicates that the macro captures predictive information about the joint future not recoverable from any single component. A common approach is to compute this for all unordered pairs $( i , j )$ and then take the median as system-level synergy capacity [78]. This test is limited to detecting synergy of order $k = 2$ , but has the advantage that it does not rely on the definition of a suitable whole-system macro signal beyond the states of the sources.

Unique information criterion. This second test evaluates a specific macroscopic feature $V _ { t }$ against the formal criteria for causal emergence [61]: a feature $V _ { t }$ is emergent if it possesses predictive power about the system’s future $( X _ { t + \ell } )$ that is unique with respect to the state of its individual parts.<sup>2</sup> This is quantified using the unique information atom from Eq. (1):

$$
\operatorname { U n } ( V _ { t } ; X _ { t + \ell } | X _ { t } ) > 0 .\tag{3}
$$

This test determines if the macro-variable has access to information about the system’s evolution that no subset of k or fewer parts can access on their own. While this is the most theoretically rigorous way to identify an emergent feature, it is computationally demanding because it requires full knowledge of the statistics of the—potentially high-dimensional—microscopic dynamics.

Practical criterion. This third test concerns predicting a macro signal $V$ of the system by asking whether the macro contains predictive information beyond any individual part. Given the microstate $X _ { t } = ( X _ { 1 , t } , \ldots , X _ { n , t } )$ , samples of the macro $V _ { t } = \dot { \boldsymbol { f } } ( \boldsymbol { X } _ { t } )$ are aligned as $( t , t + \ell )$ . From this, one can compute the following score:

$$
\Psi _ { \ell } ( V ) = I ( V _ { t } ; V _ { t + \ell } ) - \sum _ { k = 1 } ^ { n } I ( X _ { k , t } ; V _ { t + \ell } ) .\tag{4}
$$

A positive value indicates that the macro’s self-predictability exceeds what the sum of its parts can explain, indicating dynamical synergy. A benefit of this score is that it acts as a coarse, order-agnostic screen sensitive to multi-part synergy (i.e., synergy of any order $\geq 2 )$ . However, it is penalized by redundancy across parts which can make the score negative even when higher-order synergy exists. In practical terms, the measure is often computed in a bias-corrected version against some permuted null model. Recent work has shown that Eq. (4) can serve as a differentiable training objective, learning the macro variable V rather than requiring it a priori [59].

PID entails four architectural principles governing synergy in any system—circuit, agent team, or human-AI collaboration.

(i) Differentiation requirement. To generate synergy the system’s components must be either functionally differentiated or modularized by context. Without differentiation—if the parts are perfectly synchronized, cloned, or otherwise highly coupled—they can only provide redundant information that cannot complement each other. Differentiation can be engineered (e.g., via personas [78]) or arise endogenously (e.g., via learned routing [29]).

(ii) Representational capacity. Synergy provides functional advantages as it can encode more information than redundancy atoms for a given system size (representational or combinatorial capacity) [88].

(iii) Temporal emergence. Emergent properties are not present in the instantaneous microstate alone, but arise diachronically through processes unfolding over time [87].

(iv) Fragility-vs-robustness tradeoff. Synergy is associated with fragility, while redundancy is inherently robust [34]. Because emergent synergy relies on precise structural heterogeneity, it is inherently unstable and highly sensitive to the specific input signals.

## 3 Applications

This section provides three worked examples across different subfields, with applications on circuits, multi-agent systems, and human-AI interaction.

## 3.1 Synergy within LLMs

Recent work has shown that the human cortex is organized along a redundancy-to-synergy gradient: redundancy dominates in primary sensorimotor areas, whereas synergy concentrates in the central executive networks linked to higher cognition [55]. Redundancy is largely conserved across primates [54] while synergistic interactions are selectively expanded in humans. This informational “separation of roles” also occurs in artificial neural systems. For example, synergy increases as neural networks learn diverse tasks or modalities [74, 52]. Tasks requiring input integration depend on synergistic neurons. Reservoir computers evolve toward and benefit from synergistic dynamics [99], suggesting that synergy is a selected computational substrate rather than a byproduct.

How is information processing organized within LLMs? Urbina-Rodriguez et al. [101] treat attention heads and mixture-of-experts (MoE) modules as nodes, record each unit’s token-by-token activity as the L2 norm of its output, and apply a time-lagged PID (ΦID) with MMI redundancy to the resulting multivariate time series. Because a full decomposition over all units is intractable (Section 2.1), they compute bivariate ΦID for every pair of heads (or experts) which they aggregate into a per-unit synergy-redundancy rank. Synergy claims therefore reflect the density of pairwise, order-2 synergistic interactions rather than the full higher-order lattice.

Middle layers show synergistic processing while early and late layers exhibit a dominance of redundancy—a striking analog of the unimodal-to-transmodal gradient observed in the human brain. These results hold across multiple LLM families—including Llama 3, Qwen 3, and DeepSeek V2 Lite. Moreover, the synergistic core is learned, not hardwired: it is absent in randomly initialized networks and holds across both attention heads and routed experts, indicating it is a general property of trained transformers rather than an architectural artifact. The synergistic core is causally necessary for capabilities that demand integration: ablating synergistic heads degrades complex task performance disproportionately compared to ablating redundant heads. Synergistic information thus collapses sharply under targeted perturbation while redundant codes degrade gracefully. The same PID decomposition has been applied to diffusion transformers, where synergistic prompt interactions specifically index compositional image generation [23], and to large networks via subsampling [25].

![](images/5d31d3ba9f2cdb5c658c1846114e9e3f004a1325bf8e854a100aa5ef2c16648c.jpg)  
Figure 2: LLMs exhibit a synergistic core surrounded by a redundant periphery. a) Heatmaps of synergy and redundancy between pairs of attention heads in Gemma 3 4B. b) Heatmap of the synergy-redundancy ranks in Gemma 3 4B. Figures adapted from Urbina-Rodriguez et al. [101].

![](images/d9b03271d9ec75cdaa720ebe56d6f2ff1a95226fb06f6572725c5f0834334349.jpg)

![](images/c0f1ffd1c2d13e18c956d0b76a4480169cc40616198bc7e4ba405c3469c083f1.jpg)

![](images/7222137d1c2968ece6becfc2ca681f4ff832234c71a1641a10311bd4fe0b5cd5.jpg)  
Figure 3: Emergent synergy in multi-agent LLM systems. a) Practical emergence criterion (bias corrected). b) Emergence capacity dynamical synergy (bias corrected). c) Total Stability (timedelayed mutual information of triplets). Data were Winsorized at the 1st and 99th percentiles for visual clarity. Stars indicate significance level of Wilcoxon test. Figures adapted from Riedl [78]. Notes: \*\*\* $p < 0 . 0 0 1 $ \*\* p < 0.01; \* p < 0.05.

Going beyond diagnostics, RL-based fine-tuning of synergistic regions produces larger performance gains than fine-tuning redundant components, whereas under supervised fine-tuning, targeting synergistic regions confers no advantage. This suggests that synergistic representations require reinforcement-driven optimization to unlock.

## 3.2 Synergy Across Agents in Multi-Agent Systems

When multi-agent systems outperform single agents [109, 13, 51, 97], authors often attribute this to “greater-than-the-sum-of-its-parts” effects [13, p.1]. To test if this emergence is genuine, Riedl [78] applied information decomposition and the emergence tests introduced above. Specifically, the paper developed a simple group task in which agents propose integers whose sum needs to match a hidden target number. Blind to group size and peers’ guesses, agents receive only collective “too high” or “too low” feedback. The paper then reports results on experiments with 10 agents per group across different LLM models (e.g., GPT-4.1) with three randomized interventions. Groups in the control condition (Plain) exhibit strong temporal synergy but little coordinated alignment across agents. In a persona condition (Persona), agents show stable identity-linked differentiation (e.g., some agents consistently guess lower numbers). A Theory of Mind (ToM) condition instructs agents to “think about what other agents might do.” Here, agents additionally show goal-directed complementarity.

Using group error (the difference between the group’s guess and the target number) as macro signal V and MMI as redundancy function, it is shown that both the emergence capacity criterion (Eq. (1)) and the practical emergence criterion (Eq. (4)) are significantly above zero, confirming dynamical emergence (Figure 3a-b). To probe how this emergence is maintained, they compute Total Stability— defined as the triplet-wise mutual information normalized by macro-signal entropy. Total Stability is indistinguishable from zero in the Plain and Persona conditions but sharply elevated in the ToM condition $\mathsf { \bar { ( } { p \mathrm { - } v a l u e = 2 . 9 \times 1 0 ^ { - 1 4 } } }$ ; Figure 3c). This suggests the ToM intervention to “think about other agents” acts as a control parameter, steering the multi-agent system from a chaotic regime into a deep basin of attraction where collective behavior stabilizes and information processing capacity spikes. These results are corroborated through various robustness tests (full vs. early rounds only, raw measures, different bias correction approaches, a functional null to account for autocorrelation, and two different redundancy measures).

Finally, results show that neither synergy nor redundancy alone predicts success, while their interaction does $( \beta = 0 . 2 4 ; p = 0 . 0 1 4 )$ : pairwise alignment to the macro signal supplies goal alignment (redundancy), while identity-linked differentiation in the Persona and ToM conditions supplies complementary roles (synergy). Performance improves only when both are present, illustrating that multi-agent coordination requires balancing integration with differentiation across the PID lattice. Human groups have been shown to exhibit the same signature: collective intelligence requires both alignment and complementarity [81, 82].

## 3.3 Human-AI Synergy

Studies of human-AI interaction typically measure performance uplift by comparing between a “human-only” and a “human-and- $. \mathbf { A } \mathbf { I } ^ { \prime \prime }$ condition [e.g., 24, 22, 10, 69]. Those comparisons do not distinguish whether gains are additive, substitutive, or non-linearly synergistic and presuppose a single-agent baseline, which excludes tasks no single agent can attempt.<sup>3</sup> Riedl and Weidmann [80] studied human-AI collaboration using item response theory (IRT), isolating latent AI ability, human solo ability, and human collaborative ability. While IRT latent traits model the cognitive data generating process rather than yielding information-theoretic quantities, we can map this behavioral decomposition onto the conceptual lattice of PID.

Considering the contribution of the human (H) and the $\mathbf { A I } \left( A \right)$ to the behavioral outcome (Y), the joint information decomposes exhaustively into four non-negative atoms based on Eq. (1):

$$
I ( H , A ; Y ) = \operatorname { U n } ( H ) + \operatorname { U n } ( A ) + \operatorname { R e d } ( H , A ; Y ) + \operatorname { S y n } ( H , A ; Y ) .
$$

Because the emergence criterion (Section 2.2) requires only $\mathrm { S y n } > 0 .$ , one does not need to isolate the synergistic atom directly: it suffices to bound the other three and show that a positive residual survives. The behavioral tests in Ref. [80] provide a sufficient basis to bound each of them.

The residual is not unique human information $( \mathrm { U n } ( H ) ) .$ One potential concern is that the joint outcome is the human’s solo knowledge re-expressed, loading the residual onto $\operatorname { U n } ( H )$ . This is ruled out because the human’s solo ability (θ) and collaborative ability (κ) are statistically distinct latent traits (using Expected Log-Predictive Density $\Delta \mathrm { E L P D } = 5 0 . 9 , { \dot { \mathrm { S E } } } = 1 0 . 2 ; \theta \neq \kappa )$ . While distinct traits could yield identical information, discerning evidence comes from the AI-unsolvable regime: users improve on items neither source could resolve alone. Thus, the distinct collaborative process (κ) generates novel information absent from $H \mathbf { \bar { s } }$ marginal.

The residual is not unique AI information $( \operatorname { U n } ( A ) )$ . Another potential concern is delegation: the AI supplies the correct answer and the user adopts it, loading the residual onto $\operatorname { U n } ( A )$ . One can bound this channel by forcing $\operatorname { U n } ( A )$ to zero. The fundamental properties of PID dictate that $\operatorname { U n } ( A ) \leq I ( A ; Y )$ By splitting the joint condition and observing items the AI could not solve alone, we restrict the analysis to a regime where $I ( A ; Y ) \approx 0$ , thereby shutting off Un(A). Users still improve significantly over their solo baseline on these specific items $( \beta = 0 . 8 0 [ 0 . 5 9 , 1 . 0 1 ] )$ Furthermore, user-level collaborative slopes on AI-solvable and AI-unsolvable items are highly correlated $( \rho = 0 . 8 0 \ [ 0 . 5 3 , 0 . 9 6 ] )$ , contradicting the regime-specific pattern delegation predicts. Improvement that persists where $\operatorname { U n } ( A )$ is mathematically starved cannot be AI-unique information.

Redundancy is starved on the hard-item regime. Under MMI redundancy [6] (the same redundancy measure used in Sections 3.1 & 3.2) $\mathrm { R e d } ( H , A ; Y ) = \mathrm { m i n } ( I ( H ; Y ) , \bar { I ( A ; Y ) } )$ , so as questions become harder for the human alone, $I ( H ; Y ) \to 0$ drives Red → 0 with it. We do not bound the aggregate atom; we isolate a regime where it is structurally near zero. The behavioral data track this gradient: users switch from an incorrect solo answer to a correct joint one on 17-19% of the easiest items but 31-38% of the hardest, and co-success is highest in moral reasoning (where humans often already hold the answer) and lowest in physics. On hard items—where Red ≈ 0 and, when the AI alone cannot solve them, $\mathrm { U n } ( A ) \approx 0 .$ —the surviving residual is almost pure Syn (plus any elicited human knowledge, itself synergistic by the argument below). Because MMI upper-bounds every valid redundancy measure and Red $\approx 0$ in this regime, this Syn estimate does not depend on the choice of redundancy measure.

The residual carries the structuralfingerprint ofsynergy. What remains is not merely a leftover: it behaves strictly as an interaction-dependent atom should. Human ToM predicts collaborative bu not solo ability, and when ToM is zero, access to AI provides no benefit $( \beta = - 0 . 0 7 [ - 0 . 8 4 , 0 . 7 0 ] )$ Unique and redundant information do not require joint processing to exist; only synergy is contingent on the effective cognitive engagement of both sources. The vanishing benefit of the joint condition when ToM is zero is thus a positive behavioral signature of Syn.

![](images/90f8dc02b0f6262fb9f6e20950d16766997dd13c1576e8d837cc43eca1794f31.jpg)  
Figure 4: Viewing human-AI synergy through an information decomposition lens unifies existing theories about substitution, linear and non-linear complementarity.

Finally, one might suggest that the AI is simply eliciting the human’s own unexpressed, latent knowledge, rather than creating novel information. Under PID, this is not a caveat, but rather the very definition of synergy. If a human possesses latent knowledge that contains information about Y, but that information only manifests in the observable outcome when triggered by the AI’s interaction, the information exists strictly in the joint state $( H , A )$ . It cannot be found in the marginal distribution of the human’s solo responses. Information that is generated exclusively through the interaction of sources is, axiomatically, synergistic.

The synergistic atom is generated by unilateral adaptation. Although PID synergy is non-additive information by definition, the mechanism generating it here is additive at the latent level: collaborative residuals are under-dispersed $( \phi = 0 . 6 5 8 )$ , and user-by-difficulty interactions add no predictive value $( \Delta \mathrm { E L P D } = - 1 . 7$ ; 80% power to detect non-linearity as small as 16% of the additive effect). Synergy is thus extracted through a stable, general channel (the human’s collaborative ability κ, expressed as conditional delegation, scaffolding, interrogation, and calibrated reliance on AI output) rather than through item-specific feedback loops. This is the signature of an asymmetric system in which only the human adapts. Reciprocal adaptation by the AI would add trial-level interaction variance $( \phi > 1 )$ though with a single AI partner that dyadic term is not identifiable [44]. The resulting information is genuinely synergistic, but its generation is currently bounded by one-sided adaptation.

Taken together, these tests leave a positive residual that none of the other three atoms can absorb, which is a lower bound on Syn(H, A; Y ) and exactly what the emergence criterion requires, obtained from evidence never collected with PID in mind (Figure 4).

## 3.4 Scope of these claims

The choice of parts, target, timescale, and redundancy measure is not given by the theory. In an LLM, “part” could mean neuron, attention head, expert, or layer. Claims about synergy and emergence are bounded by those modeling choices which should be founded on theoretical arguments and supplemented by robustness tests. The synergistic-cores work, for example, finds signs of synergy across different granularities (attention heads & experts in MoE); the multi-agent application explores robustness across three different targets (it uses joint future state of agent pairs, group error, and first principal component as macro signal) and redundancy measures $( I _ { m i n }$ and $I _ { \mathrm { M M I } } )$ . Claims also differ in how far they climb an evidential ladder: detection, localization, causal identification, control, and prediction (Table S3). We summarize practical guidelines for applying our framework in Table

S1 and reporting its results in Box S1. We provide a quick start guide with easy-to-use code and illustrations at https://github.com/riedlc/emergence.

## 4 Discussion

In this paper we have argued that Partial Information Decomposition provides a unifying lens to study synergy across scales and substrates of AI systems—from internal components to multi-agent ensembles and human-AI interaction. While current synergy metrics vary by domain, they implicitly target a shared structure: information distribution across unique, redundant, and interaction-dependent components. Grounding metrics in this framework allows us to compare abstraction levels, clarify definitions of emergence, and advance a cumulative science of collective intelligence across scales and substrates. This lens unifies existing practices as design levers for navigating the synergy-redundancy spectrum, which we summarize in Table S2. Additionally, this perspective reveals hidden cross-layer connections to guide future research which we summarize in the five conjectures below (with testable hypotheses in Table S5).

Conjecture 1: Grokking as redundancy-to-synergy transition. Grokking, the abrupt onset of generalization long after memorization [72, 65], admits a natural PID account. While unique and redundant information provide immediate, dense gradients during early training, synergistic targets require higher-order coordination and yield negligible marginal mutual information [1, 4], leaving per-weight gradients flat. Optimization is therefore initially biased toward redundant representations, and synergistic representations emerge only when the optimizer discovers a coordinated weight configuration that unlocks a non-zero synergistic gradient [19]. We predict grokking should correspond to a measurable phase transition in the PID spectrum: redundancy-dominated during memorization, shifting abruptly to synergy at the generalization jump. We also predict that regularizers suppressing higher-order dependencies should delay or prevent generalization while leaving memorization intact. This conjecture is partially supported by the synergy–redundancy shift documented by Clauw et al. [16], and is closely related with work on leap complexity [1] showing that complex Boolean functions (maximally synergistic parities characterized by high-degree Fourier coefficients) are learned via saddle-to-saddle dynamics [4].

Conjecture 2: Jagged intelligence as a factored-representation boundary. LLMs pass the bar exam yet fail basic logic puzzles—a pattern termed “jagged intelligence” [22, 95]. However, existing accounts lack a structural criterion for predicting which tasks will fall on which side of the frontier. The factored-representation bias described in Section 3.1 suggests one: models should fail where tasks require irreducible cross-factor integration. We suggest task difficulty tracks not human-perceived difficulty but synergistic depth: the degree to which the correct answer depends on jointly considering independently informative factors. Consider, for instance, the viral LLM failure “should you walk or drive to the car wash 50 meters away?”.<sup>5</sup> Distance suggests walking and washing suggests bringing the object but only integrating the two reveals that the car is the object. The ablation test in Section 3.1 suggests a causal mechanism: middle layers, home to the synergistic core, appear prunable on benchmarks that do not probe cross-factor integration, but become indispensable once the task demands it. Independent evidence agrees: synergistic information is harder for models to capture than redundant information [52]. Furthermore, causal-abstraction work on belief tracking [73] finds that models default to factored representations of per-agent beliefs, and recruit dedicated integration heads only when cross-agent dependencies demand it.

Conjecture 3: Test-time compute as synergy unrolling. Recent information-theoretic analysis of chain-of-thought (CoT) reasoning formalizes each step as a conditional information gain, showing that correct steps monotonically increase information about the target while incorrect steps yield zero gain [100]. The PID lens suggests a specific structural interpretation: because a single forward pass has fixed functional depth, tasks whose solutions require high synergistic depth cannot be resolved in one pass. CoT acts as an algorithmic compressor of per-step synergistic load, externalizing partial states into the context window and converting within-pass synergistic integration into sequential redundant lookbacks across tokens [73]. This yields two predictions: (i) a reasoning model’s internal representations should carry lower per-pass synergistic load than a standard LLM’s, because synergy has been offloaded into the token stream; and (ii) tasks with irreducibly high synergistic depth— those resisting decomposition into sequential sub-problems—should show diminishing returns from additional test-time compute.

Conjecture 4: Multi-turn degradation and path-dependent synergy collapse. LLMs degrade when instructions are split across turns, collapsing at two turns rather than declining gradually [48], even though the full conversation remains in context. Synergistic information requires that all sources are not only present but jointly processed. Anything that degrades joint access can destroy synergistic channels without removing any information. Examples include salience shifts from reordering, and intermediate responses computed from marginals that commit the context prematurely. We predict that multi-turn degradation, input-order sensitivity, and prompt-sequence effects in multi-agent settings share one cause: disrupted access to synergistic information. All three should scale with a task’s synergistic depth and leave redundancy-dominated tasks unaffected. Laban et al. [48] observe both the cliff-at-two-turns pattern and an episodic-task exception consistent with this account.

Conjecture 5: Theory of mind as a cross-level control parameter. Across all three levels, a structurally analogous mechanism shifts systems from factored to synergistic processing. In LLMs, sentences that introduce belief discrepancies activate dedicated synergistic integration heads [73]; in multi-agent groups, ToM prompting shifts coordination toward synergy dominance [78]; and in human-AI teams, human ToM predicts collaborative but not individual ability [80]. In each case, one component accesses another’s internal state, creating conditions for irreducible joint information. This yields a cross-level prediction that neither community would make alone: high-ToM human inputs should differentially activate the model’s synergistic core, a measurable circuit-level signature of a behavioral interaction variable.

## 5 Alternative Views

PID is the wrong foundation. One might argue that simpler measures—like mutual information— are useful without the overhead of full information decomposition. We agree, and do not advocate replacing established methods. However, because simpler metrics conflate fundamentally different phenomena (e.g., synergy vs. redundancy), researchers must rely on complementary tests and triangulation to bound estimates and rule out alternative accounts. PID’s value is not as a mandatory measurement tool, but as a theoretical scaffold specifying what these simpler empirical tests capture.

O-information or other measures would work just as well. Formal PID is computationally demanding and data-hungry. Alternatives like O-information or integrated information capture related higher-order structures as complementary, tractable summaries. Our argument is not that PID is the only useful metric, but that its conceptual vocabulary—distinguishing synergy from redundancy—i necessary to make emergence claims precise, whichever estimator tests them.

Other frameworks already study higher-order interactions. Tensor networks [46] and higherorder networks (hypergraphs, simplicial complexes; [7]) describe the structure over which interactions occur; mechanistic null models such as the game-theoretic block model [79] test whether that structure is reducible to pairwise interaction, but cannot type the residual they leave; causal abstraction [31] and related accounts of emergence [36, 5, 90, 85] identify macroscopic variables via scale separation. PID does not compete but complement these approaches. None of them distinguishes the informational type of an interaction relative to a target, which is what an emergence-as-irreducibility claim turns on. PID supplies that distinction. It therefore does not compete with these frameworks so much as label the atoms they organize, transport, or coarse-grain.

These are three different phenomena, not one. A natural objection is that grouping transformers, multi-agent systems, and human-AI teams obscures their vastly different mechanisms. We acknowledge these differences in scale and substrate. However, while their physical forms diverge, their functional and information-processing architectures share similar structures. Information decomposition allows us to translate findings across scales and formulate testable, cross-layer predictions that siloed fields would miss. Ultimately, this is an empirical question: the success of these hypotheses will dictate whether this framework is based on a mere analogy or on shared computational reality.

## Acknowledgments and Disclosure of Funding

C.R. received no funding for this work. F.E.R. has been supported by the UK AI Security Institute and Coefficient Giving. The authors declare no competing interests.

## References

[1] Emmanuel Abbe, Enric Boix Adsera, and Theodor Misiakiewicz. SGD learning on neural networks: leap complexity and saddle-to-saddle dynamics. In The Thirty Sixth Annual Conference on Learning Theory, pages 2552–2623. PMLR, 2023.

[2] Alessandro Achille and Stefano Soatto. Information dropout: Learning optimal representations through noisy computation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 40(12):2897–2905, 2018.

[3] Gagan Bansal, Tongshuang Wu, Joyce Zhou, Raymond Fok, Besmira Nushi, Ece Kamar, Marco Tulio Ribeiro, and Daniel Weld. Does the whole exceed its parts? the effect of ai explanations on complementary team performance. In Proceedings ofthe 2021 CHI Conference on Human Factors in Computing Systems, pages 1–16, 2021.

[4] Boaz Barak, Benjamin Edelman, Surbhi Goel, Sham Kakade, Eran Malach, and Cyril Zhang. Hidden progress in deep learning: Sgd learns parities near the computational limit. Advances in Neural Information Processing Systems, 35:21750–21764, 2022.

[5] Lionel Barnett and Anil K Seth. Dynamical independence: discovering emergent macroscopic processes in complex dynamical systems. Physical Review E, 108(1):014304, 2023.

[6] Adam B Barrett. Exploration of synergistic and redundant information sharing in static and dynamical gaussian systems. Physical Review E, 91(5):052802, 2015.

[7] Federico Battiston, Giulia Cencetti, Iacopo Iacopini, Vito Latora, Maxime Lucas, Alice Patania, Jean-Gabriel Young, and Giovanni Petri. Networks beyond pairwise interactions: Structure and dynamics. Physics Reports, 874:1–92, 2020.

[8] Mark A Bedau and Paul Humphreys. Emergence: Contemporary Readings in Philosophy and Science. MIT Press, Cambridge, MA, 2008.

[9] Laouen Belloli, Pedro Mediano, Rodrigo Cofré, Diego Fernandez Slezak, and Rubén Herzog. THOI: An efficient and accessible library for computing higher-order interactions enhanced by batch-processing. arXiv preprint arXiv:2501.03381, 2025.

[10] Erik Brynjolfsson, Danielle Li, and Lindsey Raymond. Generative ai at work. Quarterly Journal ofEconomics, page qjae044, 2025.

[11] Madhavun Candadai and Eduardo J Izquierdo. infotheory: A c++/python package for multivariate information theoretic analysis. arXiv preprint arXiv:1907.02339, 2019.

[12] Sean M. Carroll and Achyuth Parola. What emergence can possibly mean. In Tyler Millhouse, Stephen Petersen, and Don Ross, editors, Dennett’s Real Patterns in Science and Nature. The MIT Press, 03 2026.

[13] Weize Chen, Yusheng Su, Jingwei Zuo, Cheng Yang, Chenfei Yuan, Chen Qian, Chi-Min Chan, Yujia Qin, Yaxi Lu, Ruobing Xie, et al. Agentverse: Facilitating multi-agent collaboration and exploring emergent behaviors in agents. arXiv:2308.10848, 2023.

[14] Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017.

[15] Rudi Cilibrasi and Paul MB Vitányi. Clustering by compression. IEEE Transactions on Information Theory, 51(4):1523–1545, 2005.

[16] Kenzo Clauw, Sebastiano Stramaglia, and Daniele Marinazzo. Information-theoretic progress measures reveal grokking is an emergent phase transition. arXiv preprint arXiv:2408.08944, 2024.

[17] Philip Clayton and Paul Davies. The Re-emergence of Emergence: The Emergentist Hypothesis from Science to Religion. Oxford University Press, Oxford, UK, 2006.

[18] Thomas M Cover, Joy A Thomas, and John Kieffer. Elements of Information Theory, volume 2. Wiley, New York, NY, 1991.

[19] Ben Cullen, Sergio Estan-Ruiz, Riya Danait, and Jiayi Li. Grokking as a phase transition between competing basins: a singular learning theory approach. arXiv preprint arXiv:2603.01192, 2026.

[20] Hadi Daneshmand, Amir Joudaki, and Francis Bach. Batch normalization orthogonalizes representations in deep random networks. Advances in Neural Information Processing Systems, 34:4896–4906, 2021.

[21] Nicola De Cao, Michael Schlichtkrull, Wilker Aziz, and Ivan Titov. How do decisions emerge across layers in neural models? interpretation with differentiable masking. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing, pages 3243– 3255, 2020.

[22] Fabrizio Dell’Acqua, Edward McFowland, Ethan R Mollick, Hila Lifshitz-Assaf, Katherine Kellogg, Saran Rajendran, Lisa Krayer, François Candelon, and Karim R Lakhani. Navigating the jagged technological frontier: Field experimental evidence of the effects of artificial intelligence on knowledge worker productivity and quality. Organization Science, 37:403–423, 2026.

[23] Shaurya Dewan, Rushikesh Zawar, Prakanshul Saxena, Yingshan Chang, Andrew Luo, and Yonatan Bisk. DiffusionPID: Interpreting diffusion via partial information decomposition. Advances in Neural Information Processing Systems, 37:2045–2079, 2024.

[24] Anil R Doshi and Oliver P Hauser. Generative ai enhances individual creativity but reduces the collective diversity of novel content. Science Advances, 10(28):eadn5290, 2024.

[25] David Alexander Ehrlich, Andreas Christian Schneider, Viola Priesemann, Michael Wibral, and Abdullah Makkeh. A measure of the complexity of neural representations based on partial information decomposition. Transactions on Machine Learning Research, 2023.

[26] Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, et al. Toy models of superposition. arXiv preprint arXiv:2209.10652, 2022.

[27] Luca Faes, Gorana Mijatovic, Yuri Antonacci, Riccardo Pernice, Chiara Barà, Laura Sparacino, Marco Sammartino, Alberto Porta, Daniele Marinazzo, and Sebastiano Stramaglia. A new framework for the time-and frequency-domain assessment of high-order interactions in networks of random processes. IEEE Transactions on Signal Processing, 70:5766–5777, 2022.

[28] Conor Finn and Joseph T Lizier. Pointwise partial information decomposition using the specificity and ambiguity lattices. Entropy, 20(4):297, 2018.

[29] Zachary Fulker, Patrick Forber, Rory Smead, and Christoph Riedl. Spontaneous emergence of groups and signaling diversity in dynamic networks. Physical Review E, 109(1):014309, 2024.

[30] Johanna Gehlen, Jie Li, Cillian Hourican, Stavroula Tassi, Pashupati P Mishra, Terho Lehtimäki, Mika Kähönen, Olli Raitakari, Jos A Bosch, and Rick Quax. Bias in O-information estimation. Entropy, 26(10):837, 2024.

[31] Atticus Geiger, Duligur Ibeling, Amir Zur, Maheep Chaudhary, Sonakshi Chauhan, Jing Huang, Aryaman Arora, Zhengxuan Wu, Noah Goodman, Christopher Potts, et al. Causal abstraction: A theoretical foundation for mechanistic interpretability. Journal of Machine Learning Research, 26(83):1–64, 2025.

[32] Sophie C Gibb, Robin Findlay Hendry, and Tom Lancaster. The Routledge Handbook of Emergence. Routledge, London, UK, 2019.

[33] Satchel Grant, Noah D Goodman, and James L McClelland. Emergent symbol-like number variables in artificial neural networks. arXiv preprint arXiv:2501.06141, 2025.

[34] Aaron J Gutknecht, Fernando E Rosas, David A Ehrlich, Abdullah Makkeh, Pedro AM Mediano, and Michael Wibral. Shannon invariants: A scalable approach to information decomposition. arXiv preprint arXiv:2504.15779, 2025.

[35] Malte Harder, Christoph Salge, and Daniel Polani. Bivariate measure of redundant information. Physical Review E, 87(1):012130, 2013.

[36] Erik P Hoel, Larissa Albantakis, and Giulio Tononi. Quantifying causal emergence shows that macro can beat micro. Proceedings of the National Academy of Sciences, 110(49): 19790–19795, 2013.

[37] Robin AA Ince. Measuring multivariate redundant information with pointwise common change in surprisal. Entropy, 19(7):318, 2017.

[38] Robin AA Ince. The partial entropy decomposition: Decomposing multivariate entropy and mutual information via pointwise common surprisal. arXiv preprint arXiv:1702.01591, 2017.

[39] Ryan G James, Christopher J Ellison, and James P Crutchfield. “dit”: a python package for discrete information theory. Journal ofOpen Source Software, 3(25):738, 2018.

[40] Ryan G James, Jeffrey Emenheiser, and James P Crutchfield. Unique information via dependency constraints. Journal of Physics A: Mathematical and Theoretical, 52(1):014002, 2019.

[41] Abel Jansma, Pedro AM Mediano, and Fernando E Rosas. Fast möbius transform: An algebraic approach to information decomposition. Physical Review Research, 7(3):033049, 2025.

[42] Henrik Jeldtoft Jensen. Complexity Science: The Study of Emergence. Cambridge University Press, Cambridge, UK, 2022.

[43] Sean Kelley, David De Cremer, and Christoph Riedl. Personalized AI scaffolds synergistic multi-turn collaboration in creative work. arXiv preprint arXiv:2510.27681, 2025.

[44] David A. Kenny and Lawrence La Voie. The social relations model. In Leonard Berkowitz, editor, Advances in Experimental Social Psychology, volume 18, pages 141–182. Academic Press, San Diego, CA, 1984.

[45] Artemy Kolchinsky. A novel approach to the partial information decomposition. Entropy, 24 (3):403, 2022.

[46] Tamara G Kolda and Brett W Bader. Tensor decompositions and applications. SIAM review, 51(3):455–500, 2009.

[47] David C Krakauer. Symmetry–simplicity, broken symmetry–complexity. Interface Focus, 13 (3):20220075, 2023.

[48] Philippe Laban, Hiroaki Hayashi, Yingbo Zhou, and Jennifer Neville. LLMs get lost in multi-turn conversation. In International Conference on Learning Representations, 2026.

[49] Allison Lau, Younwoo Choi, Vahid Balazadeh, Keertana Chidambaram, Vasilis Syrgkanis, and Rahul G Krishnan. Personalized adaptation via in-context preference learning. arXiv preprint arXiv:2410.14001, 2024.

[50] Jaemin Lee, David Lazer, and Christoph Riedl. Complex contagion in social networks: Causal evidence from a country-scale field experiment. Sociological Science, 12:685–714, 2025.

[51] Junyou Li, Qin Zhang, Yangbin Yu, Qiang Fu, and Deheng Ye. More agents is all you need. arXiv:2402.05120, 2024.

[52] Paul Pu Liang, Yun Cheng, Xiang Fan, Chun Kai Ling, Suzanne Nie, Richard Chen, Zihao Deng, Nicholas Allen, Randy Auerbach, Faisal Mahmood, et al. Quantifying & modeling multimodal interactions: An information decomposition framework. Advances in Neural Information Processing Systems, 36:27351–27393, 2023.

[53] Alberto Liardi, Keenan JA Down, George Blackburne, Matteo Neri, and Pedro AM Mediano. The mathematical landscape of partial information decomposition: A comprehensive review of properties and measures. arXiv preprint arXiv:2603.06678, 2026.

[54] Andrea I Luppi, Pedro AM Mediano, Fernando E Rosas, Negin Holland, Tim D Fryer, John T O’Brien, James B Rowe, David K Menon, Daniel Bor, and Emmanuel A Stamatakis. A synergistic core for human brain evolution and cognition. Nature Neuroscience, 25(6):771– 782, 2022.

[55] Andrea I Luppi, Fernando E Rosas, Pedro AM Mediano, David K Menon, and Emmanuel A Stamatakis. Information decomposition and the informational architecture of the brain. Trends in Cognitive Sciences, 28(4):352–368, 2024.

[56] Abdullah Makkeh, Dirk Oliver Theis, and Raul Vicente. BROJA-2PID: A robust estimator for bivariate partial information decomposition. Entropy, 20(4):271, 2018.

[57] Abdullah Makkeh, Aaron J Gutknecht, and Michael Wibral. Introducing a differentiable measure of pointwise shared information. Physical Review E, 103(3):032149, 2021.

[58] Patrick McMillen and Michael Levin. Collective intelligence: A unifying concept for integrating biology across scales and substrates. Communications Biology, 7(1):378, 2024.

[59] David McSharry, Christos Kaplanis, Fernando E Rosas, and Pedro A Mediano. Learning diverse causally emergent representations from time series data. Advances in Neural Information Processing Systems, 37:119547–119572, 2024.

[60] Pedro AM Mediano, Fernando E Rosas, Juan Carlos Farah, Murray Shanahan, Daniel Bor, and Adam B Barrett. Integrated information as a common signature of dynamical and informationprocessing complexity. Chaos, 32(1), 2022.

[61] Pedro AM Mediano, Fernando E Rosas, Andrea I Luppi, Henrik J Jensen, Anil K Seth, Adam B Barrett, Robin L Carhart-Harris, and Daniel Bor. Greater than the parts: a review of the information decomposition approach to causal emergence. Philosophical Transactions of the Royal Society A, 380(2227):20210246, 2022.

[62] Pedro AM Mediano, Fernando E Rosas, Andrea I Luppi, Robin L Carhart-Harris, Daniel Bor, Anil K Seth, and Adam B Barrett. Toward a unified taxonomy of information dynamics via integrated information decomposition. Proceedings ofthe National Academy ofSciences, 122 (39):e2423297122, 2025.

[63] Han Meng, Yitian Yang, Wayne Fu, Jungup Lee, Yunan Li, and Yi-Chieh Lee. Exploring the human-llm synergy in advancing theory-driven qualitative analysis. ACM Transactions on Computer-Human Interaction, 33(1):1–51, 2026.

[64] Hussein Mozannar and David Sontag. Consistent estimators for learning to defer to an expert. In Hal Daumé III and Aarti Singh, editors, Proceedings ofthe 37th International Conference on Machine Learning, volume 119, pages 7076–7087, 13–18 Jul 2020.

[65] Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, and Jacob Steinhardt. Progress measures for grokking via mechanistic interpretability. arXiv preprint arXiv:2301.05217, 2023.

[66] Neel Nanda, Andrew Lee, and Martin Wattenberg. Emergent linear representations in world models of self-supervised sequence models. In Yonatan Belinkov, Sophie Hao, Jaap Jumelet, Najoung Kim, Arya McCarthy, and Hosein Mohebbi, editors, Proceedings ofthe 6th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP, pages 16–30, Singapore, December 2023. Association for Computational Linguistics.

[67] Matteo Neri, Dishie Vinchhi, Christian Ferreyra, Thomas Robiglio, Onur Ates, Marlis Ontivero-Ortega, Andrea Brovelli, Daniele Marinazzo, and Etienne Combrisson. HOI: A python toolbox for high-performance estimation of higher-order interactions from multivariate data. Journal ofOpen Source Software, 9(103):7360, 2024.

[68] Stefan Nielsen, Edoardo Cetin, Peter Schwendeman, Qi Sun, Jinglue Xu, and Yujin Tang. Learning to orchestrate agents in natural language with the conductor. In The Fourteenth International Conference on Learning Representations, 2026.

[69] Shakked Noy and Whitney Zhang. Experimental evidence on the productivity effects of generative artificial intelligence. Science, 381(6654):187–192, 2023.

[70] Patricio Orio, Pedro AM Mediano, and Fernando E Rosas. Dynamical noise can enhance high-order statistical structure in complex systems. Chaos, 33(12), 2023.

[71] Chu Pan and Yanlin Chen. Informeasure: an r/bioconductor package for quantifying nonlinear dependence between variables in biological networks from an information theory perspective. BMC Bioinformatics, 25(1):382, 2024.

[72] Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, and Vedant Misra. Grokking: Generalization beyond overfitting on small algorithmic datasets. arXiv preprint arXiv:2201.02177, 2022.

[73] Nikhil Prakash, Natalie Shapira, Arnab Sen Sharma, Christoph Riedl, Yonatan Belinkov, Tamar Rott Shaham, David Bau, and Atticus Geiger. Language models use lookbacks to track beliefs. International Conference on Learning Representations, 2026.

[74] Alexandra M Proca, Fernando E Rosas, Andrea I Luppi, Daniel Bor, Matthew Crosby, and Pedro AM Mediano. Synergistic information supports modality integration and flexible learning in neural networks solving multiple tasks. PLoS Computational Biology, 20(6): e1012178, 2024.

[75] Chen Qian, Wei Liu, Hongzhang Liu, Nuo Chen, Yufan Dang, Jiahao Li, Cheng Yang, Weize Chen, Yusheng Su, Xin Cong, et al. Chatdev: Communicative agents for software development. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (volume 1: Long papers), pages 15174–15186, 2024.

[76] Borzoo Rassouli, Fernando E Rosas, and Deniz Gündüz. Data disclosure under perfect sample privacy. IEEE Transactions on Information Forensics and Security, 15:2012–2025, 2019.

[77] Johannes Rauh, Nils Bertschinger, Eckehard Olbrich, and Jürgen Jost. Reconsidering unique information: Towards a multivariate information decomposition. In 2014 IEEE International Symposium on Information Theory, pages 2232–2236. IEEE, 2014.

[78] Christoph Riedl. Emergent coordination in multi-agent language models. International Conference on Learning Representations, 2026.

[79] Christoph Riedl and Rory Smead. Strategic origins of network structure. arXiv preprint arXiv:2610.24792, 2026.

[80] Christoph Riedl and Ben Weidmann. Quantifying human-AI synergy. OSF PsyArXiv:vbkmt\_v4, 2026.

[81] Christoph Riedl and Anita Williams Woolley. Teams vs. crowds: A field test of the relative contribution of incentives, member ability, and emergent collaboration to crowd-based problem solving performance. Academy ofManagement Discoveries, 3(4):382–403, 2017.

[82] Christoph Riedl, Young Ji Kim, Pranav Gupta, Thomas W Malone, and Anita Williams Woolley. Quantifying collective intelligence in human groups. Proceedings of the National Academy ofSciences, 118(21):e2005737118, 2021.

[83] Fernando Rosas, Vasilis Ntranos, Christopher J Ellison, Sofie Pollin, and Marian Verhelst. Understanding interdependency through complex information sharing. Entropy, 18(2):38, 2016.

[84] Fernando Rosas, Pedro AM Mediano, Martín Ugarte, and Henrik J Jensen. An informationtheoretic approach to self-organisation: Emergence of complex interdependencies in coupled dynamical systems. Entropy, 20(10):793, 2018.

[85] Fernando E Rosas. Symmetries at the origin of hierarchical emergence. arXiv preprint arXiv:2512.00984, 2025.

[86] Fernando E Rosas, Pedro AM Mediano, Michael Gastpar, and Henrik J Jensen. Quantifying high-order interdependencies via multivariate extensions of the mutual information. Physical Review E, 100(3):032305, 2019.

[87] Fernando E Rosas, Pedro AM Mediano, Henrik J Jensen, Anil K Seth, Adam B Barrett, Robin L Carhart-Harris, and Daniel Bor. Reconciling emergences: An information-theoretic approach to identify causal emergence in multivariate data. PLoS Computational Biology, 16 (12):e1008289, 2020.

[88] Fernando E Rosas, Pedro AM Mediano, Borzoo Rassouli, and Adam B Barrett. An operational information decomposition via synergistic disclosure. Journal of Physics A: Mathematical and Theoretical, 53(48):485001, 2020.

[89] Fernando E Rosas, Pedro AM Mediano, Andrea I Luppi, Thomas F Varley, Joseph T Lizier, Sebastiano Stramaglia, Henrik J Jensen, and Daniele Marinazzo. Disentangling high-order mechanisms and high-order behaviours in complex systems. Nature Physics, 18(5):476–477, 2022.

[90] Fernando E Rosas, Bernhard C Geiger, Andrea I Luppi, Anil K Seth, Daniel Polani, Michael Gastpar, and Pedro AM Mediano. Software in the natural world: A computational approach to hierarchical emergence. arXiv preprint arXiv:2402.09090, 2024.

[91] Fernando E Rosas, Aaron J Gutknecht, Pedro AM Mediano, and Michael Gastpar. Characterising high-order interdependence via entropic conjugation. Communications Physics, 8(1): 347, 2025.

[92] Madalina I Sas, Fernando E Rosas, Hardik Rajpal, Daniel Bor, Henrik J Jensen, and Pedro AM Mediano. Improved estimators of causal emergence for large systems. arXiv preprint arXiv:2601.00013, 2025.

[93] Rylan Schaeffer, Brando Miranda, and Sanmi Koyejo. Are emergent abilities of large language models a mirage? Advances in Neural Information Processing Systems, 36:55565–55581, 2023.

[94] Claude Elwood Shannon. A mathematical theory of communication. Bell System Technical Journal, 27(3):379–423, 1948.

[95] Peiyang Song, Pengrui Han, and Noah Goodman. Large language model reasoning failures. arXiv preprint arXiv:2602.06176, 2026.

[96] Sebastiano Stramaglia, Tomas Scagliarini, Bryan C Daniels, and Daniele Marinazzo. Quantifying dynamical high-order interdependencies from the o-information: an application to neural spiking dynamics. Frontiers in Physiology, 11:595736, 2021.

[97] Wei Tao, Yucheng Zhou, Wenqiang Zhang, and Yu Cheng. Magis: Llm-based multi-agent framework for github issue resolution. arXiv:2403.17927, 2024.

[98] Yonglong Tian, Chen Sun, Ben Poole, Dilip Krishnan, Cordelia Schmid, and Phillip Isola. What makes for good views for contrastive learning? Advances in Neural Information Processing Systems, 33:6827–6839, 2020.

[99] Hanna M Tolle, Andrea I Luppi, Anil K Seth, and Pedro AM Mediano. Evolving reservoir computers reveal bidirectional coupling between predictive power and emergent dynamics. Patterns, 7(3), 2026.

[100] Jean-Francois Ton, Muhammad Faaiz Taufiq, and Yang Liu. Understanding chain-of-thought in llms through information theory. In International Conference on Machine Learning, pages 59784–59811. PMLR, 2025.

[101] Pedro Urbina-Rodriguez, Zafeirios Fountas, Fernando E Rosas, Jun Wang, Andrea I Luppi, Haitham Bou-Ammar, Murray Shanahan, and Pedro AM Mediano. A brain-like synergistic core in llms drives behaviour and learning. arXiv preprint arXiv:2601.06851, 2026.

[102] Michelle Vaccaro, Abdullah Almaatouq, and Thomas Malone. When combinations of humans and ai are useful: A systematic review and meta-analysis. Nature Human Behaviour, 8(12): 2293–2303, 2024.

[103] Kevin Ro Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, and Jacob Steinhardt. Interpretability in the wild: a circuit for indirect object identification in GPT-2 small. In The Eleventh International Conference on Learning Representations, 2023.

[104] Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, and William Fedus. Emergent abilities of large language models. Transactions on Machine Learning Research, 2022.

[105] Ben Weidmann and David J Deming. Team players: How social skills improve team performance. Econometrica, 89(6):2637–2657, 2021.

[106] Samuel Westby and Christoph Riedl. Collective intelligence in human-ai teams: A bayesian theory of mind approach. Proceedings ofthe AAAI Conference on Artificial Intelligence, 37 (5):6119–6127, 2023.

[107] Paul L Williams and Randall D Beer. Nonnegative decomposition of multivariate information. arXiv:1004.2515, 2010.

[108] Patricia Wollstadt, Joseph T Lizier, Raul Vicente, Conor Finn, Mario Martinez-Zarzuela, Pedro Mediano, Leonardo Novelli, and Michael Wibral. Idtxl: The information dynamics toolkit xl: a python package for the efficient analysis of multivariate information dynamics in networks. arXiv preprint arXiv:1807.10459, 2018.

[109] Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Shaokun Zhang, Erkang Zhu, Beibin Li, Li Jiang, Xiaoyun Zhang, and Chi Wang. Autogen: Enabling next-gen llm applications via multi-agent conversation framework. arXiv:2308.08155, 2023.

[110] Mingyue Yuan, Jieshan Chen, Yongquan Hu, Sidong Feng, Mulong Xie, Gelareh Mohammadi, Zhenchang Xing, and Aaron Quigley. Towards human–ai synergy in ui design: Supporting iterative generation with llms. ACM Transactions on Computer-Human Interaction, 33(1), 2026.

## A Supplementary Material

## A.1 Technical introduction to Partial Information Decomposition

Let us provide here a more technical explanation of the foundations of Partial Information Decomposition [107]. The general decomposition of the information provided by variables $\pmb { X } = ( X _ { 1 } , \ldots , { \hat { X } } _ { n } )$ about another variable $Y$ is built following these steps.

1) Define to be a generalized “source” of information all subsets of variables. Generalized sources can be indexed with numbers within round brackets. For example, the source constituted by variables $X _ { 1 } , X _ { 2 } ,$ , and $X _ { 4 }$ is denoted by (124).

2) Establish a method to quantify the redundancy between generalized sources, denoted by $I _ { \cap } .$ This should capture the amount of information provided by multiple groups of variables; for instance, the redundancy between sources (12) and (145) is captured by the quantity $I _ { \cap } ^ { ( 1 2 ) ( 1 4 5 ) } ( X ; Y )$ . As a consistency consideration, the redundancy of a source with a single group of variables is required to be equal to the corresponding mutual information, so that $I _ { \cap } ^ { ( 1 3 ) } ( { \pmb X } ; Y ) = I ( X _ { 1 } , X _ { 3 } ; Y )$

3) Assume that the redundancy between m generalized sources $S _ { 1 } , \ldots , S _ { m }$ doesn’t change if one removes sources that are contained within other sources. For example, $I ^ { ( 1 2 ) { \bar { ( } } 3 4 ) { } ^ { ( 2 ) } } ( X ; Y ) = I ^ { ( 1 2 ) { } ^ { ( 3 4 ) } } ( X ; Y )$ . This implies that one should focus on the redundancy between collections of generalized sources that form “antichains”, where no generalized source is fully contained by another. This also suggests that the redundancy should monotonically follow the following partial relationship:

$$
\alpha \leq \beta \quad \iff \quad \forall B \in \beta , \exists A \in \alpha , A \subseteq B .\tag{5}
$$

The set of all antichains of n source variables is denoted by $A _ { n }$ , and together with the partial order $\leq$ constitute the so-called redundancy lattice $\textstyle { \mathcal { R } } _ { n }$

4) Finally, the PID atoms are obtained by taking the Möbius inversion over the redundancy lattice; i.e. are given by the terms $I _ { \partial } ^ { \beta } ( { \bar { \pmb { X } } } ; Y )$ that satisfy the following property:

$$
I _ { \cap } ^ { \alpha } ( X ; Y ) : = \sum _ { { \beta \in { \cal A } _ { n } } \atop { \beta \leq \alpha } } I _ { \partial } ^ { \beta } ( X ; Y ) .\tag{6}
$$

Informally, this can be thought of as the “derivative” of $I _ { \cap } ^ { \beta } ( X ; Y )$ on $\mathcal { R } _ { n }$

Using the PID atoms, the total information $I ( X ; Y )$ can be decomposed as

$$
I ( X ; Y ) = \sum _ { \pmb { \alpha } \in A _ { n } } I _ { \partial } ^ { \pmb { \alpha } } ( X ; Y ) .\tag{7}
$$

The formulation above leaves the redundancy function $I _ { \cap }$ unspecified; the minimum-mutualinformation choice in Eq. (2) is one such function. MMI has a property worth flagging: it can give non-zero redundancy for uncorrelated sources. This feature motivates the distinction between source redundancy, corresponding to correlations between sources that are informative about the target, and mechanistic redundancy, which is redundancy associated to the structure of the mechanism (the channel) mapping sources to target and not from correlation among sources [35, 38].

## A.2 Example of algorithm to calculate synergy

This appendix introduces a specific way to operationalize synergy, and discusses the algorithm used for its calculation, summarizing results first introduced in Ref. [76].

## A.2.1 Synergy and data privacy

To outline the core idea, consider $\pmb { X } = ( X _ { 1 } , \ldots , X _ { n } )$ to be n discrete source variables and $Y$ a target variable following a distribution $p _ { \mathbf { { X } } , Y }$ . For a subsystem $\alpha \subset [ n ] : = \{ 1 , . . . , n \}$ , write $X ^ { \alpha }$ for the corresponding tuple of variables, and let a constraint set ${ \pmb { \alpha } } = \{ \alpha _ { 1 } , . . . , \alpha _ { L } \}$ collect subsystems that we wish to keep private. A channel $p _ { V | X }$ is called α-synergistic if its output V is independent of every subsystem in α, $V \perp X ^ { \alpha _ { i } }$ for all i, so that $I ( V ; { \pmb X } ^ { \alpha _ { i } } ) = 0$ for all i. The set of such channels is denoted $\mathcal { C } ( X ; \alpha )$ . Intuitively, a variable V obtained via an α-synergistic channel may carry information about the whole X while betraying nothing about any private components.

Building on these ideas, the α-synergy is the information about $Y$ that any such α-private observable V can convey,

$$
S ^ { \alpha } ( X \to Y ) : = \operatorname* { s u p } _ { \stackrel { } { p _ { V \mid X } \in { \mathcal C } ( X ; \alpha ) } } I ( V ; Y ) .\tag{8}
$$

This has a direct operational reading: $S ^ { \alpha } ( X \to Y )$ is the amount of information about $Y$ that can be disclosed from X while keeping $\breve { X } ^ { \alpha _ { 1 } } , \ldots , X ^ { \alpha _ { L } }$ perfectly private. Two boundary cases anchor the intuition: with no constraints, $S ^ { \breve { \otimes } } ( X \to Y ) = I ( \dot { X } ; Y )$ , whereas protecting every individual source, $\pmb { \alpha } = \{ \{ 1 \} , \dots , \{ n \} \}$ , isolates the information that survives only in the joint behavior of the sources.

## A.2.2 The algorithm

The supremum in (8) is always attained, and the optimal channel is the solution of a linear program [76, Thm. 1]. The key structural property to build such an algorithm is that a channel lies in $\mathcal { C } ( X ; \alpha )$ if and only if each of its reverse conditionals $p _ { \mathbf { { X } } | V = v }$ preserves the protected marginals of p<sub>X</sub>. Collect those marginals into a binary masking matrix $\mathbf { P _ { \alpha } } _ { \alpha }$ , formed by stacking, for each $\alpha _ { i } \in \alpha$ , the (deterministic) map that sends a distribution on X to the marginal of $\pmb { X } ^ { \alpha _ { i } }$ . The feasible reverse conditionals are then exactly the distributions in the polytope

$$
\mathcal { Q } _ { \alpha } = \Big \{ q \in \Delta ( \mathrm { s u p p } p \mathbf { \Phi } _ { X } ) : \mathbf { P } _ { \alpha } q = \mathbf { P } _ { \alpha } p _ { X } \Big \} ,\tag{9}
$$

i.e. the distributions that leave every protected marginal untouched. Because $I ( V ; Y ) = H ( Y ) -$ $\begin{array} { r } { \sum _ { v } p ( v ) H ( Y \mid V = v ) } \end{array}$ is convex in the channel, its maximum over the feasible set is attained at the extreme points of ${ \mathcal { Q } } _ { \alpha }$ . Enumerating those vertices $\left\{ q _ { 1 } , \dots , q _ { M } \right\}$ fixes, for each j, an induced target conditional $r _ { j } = p _ { Y | V = j }$ with entropy $h _ { j } = H ( r _ { j } ) ;$ ; all that remains is to choose mixing weights $w _ { j } \geq 0$ that reconstruct the prior, $\begin{array} { r } { \sum _ { j } w _ { j } q _ { j } = p _ { \pmb { X } } } \end{array}$ , so as to minimize the expected residual uncertainty $\sum _ { j } w _ { j } h _ { j }$ . The whole procedure is summarized as Algorithm 1.

```latex
Algorithm 1 Synergistic disclosure $S ^ { \alpha } ( X \to Y )$ of a discrete system
1: function $\mathrm { S Y N D I S C } ( p _ { X , Y } , \pmb { \alpha } = \{ \alpha _ { 1 } , . . . , \alpha _ { L } \} )$
2: $\begin{array} { r } { p _ { \mathbf { X } }  \sum _ { y } p _ { \mathbf { X } , Y } ( \cdot , y ) } \end{array}$
3: $p _ { Y | \boldsymbol { X } } \gets p _ { \boldsymbol { X } , Y } / p _ { \boldsymbol { X } }$
4: $\begin{array} { r } { H _ { Y }  H ( \sum _ { \mathbf { X } } p _ { Y | \mathbf { X } } ( \cdot \mid \mathbf { X } ) p _ { X } ( X ) ) } \end{array}$
5: build masking matrix $\mathbf { P _ { \alpha } } _ { \alpha }$
6: Q ← vertices $\left\{ q \in \Delta ( \operatorname { s u p p } p _ { X } ) : \mathbf { P } _ { \alpha } q = \mathbf { P } _ { \alpha } p _ { X } \right\}$
7: for each vertex ${ \dot { \boldsymbol { q } } } _ { j } \in { \mathcal { Q } }$ do
8: $\begin{array} { r } { r _ { j } \gets \sum _ { \boldsymbol { X } } p _ { Y | \boldsymbol { X } } ( \cdot \mid \boldsymbol { X } ) q _ { j } ( \boldsymbol { X } ) } \end{array}$
9: $h _ { j }  H ( r _ { j } )$
10: end for
11: solve the linear program
$w ^ { \star } = \underset { \ldots \sim \mathrm { { m i n } } } { \arg \operatorname* { m i n } } \sum w _ { j } h _ { j } \quad \mathrm { s . t . } \quad \sum w _ { j } q _ { j } = p _ { X }$
w≥0 j j
12: return $\begin{array} { r } { H _ { Y } - \sum _ { j } w _ { j } ^ { \star } h _ { j } } \end{array}$
13: end function
```

## A.2.3 Example: the AND gate

To illustrate the algorithm, consider independent fair inputs $X _ { 1 } , X _ { 2 } \in \{ 0 , 1 \}$ and $Y = X _ { 1 } \wedge X _ { 2 }$ Then p is uniform on the four states {00, 01, 10, 11}, the target is deterministic given the inputs, and $\begin{array} { r } { I ( X ; Y ) = H ( Y ) = H \big ( \frac { 1 } { 4 } \big ) = 0 . 8 1 \mathrm { 1 3 } } \end{array}$ bits.

To calculate synergy (i.e. $\pmb { \alpha } = \{ \{ 1 \} , \{ 2 \} \} . )$ , we need to consider channels that keep private both marginals. This means that every feasible $q \in \mathcal { Q } _ { \alpha }$ must keep $\operatorname* { P r } ( X _ { 1 } { = } 1 ) = \operatorname* { P r } ( X _ { 2 } { = } 1 ) \ { \overset { . } { = } } \ { \frac { 1 } { 2 } }$ . On the

support of $p _ { x }$ this leaves a one-parameter family, $q = ( t , \ { \frac { 1 } { 2 } } - t , \ { \frac { 1 } { 2 } } - t , t )$ over (00, 01, 10, 11) for $t \in [ 0 , \frac { 1 } { 2 } ] \mathrm { - - } \mathbf { a }$ line segment whose two vertices are

$$
\begin{array} { r } { q _ { A } = \left( 0 , \frac { 1 } { 2 } , \frac { 1 } { 2 } , 0 \right) ( \mathrm { a n t i - d i a g o n a l } ) , \qquad q _ { B } = \left( \frac { 1 } { 2 } , 0 , 0 , \frac { 1 } { 2 } \right) ( \mathrm { d i a g o n a l } ) . } \end{array}
$$

Their induced target conditionals are $r _ { A } = p ( Y \mid V { = } A ) = ( 1 , 0 )$ , since both 01 and 10 map to $Y = 0 ,$ giving $h _ { A } = 0 ;$ and $r _ { B } = p ( Y \mid V { = } \bar { B } ) \stackrel { } { = } \stackrel { ( 1 } { 2 } , \frac { 1 } { 2 } )$ , since $0 0 \mapsto 0$ and $1 1 \mapsto 1$ , giving $h _ { B } = 1$ The prior is the segment midpoint, so the reconstruction constraint $w _ { A } q _ { A } + w _ { B } q _ { B } = p _ { \bf X }$ pins the weights to $\begin{array} { r } { w _ { A } = \check { w } _ { B } = \frac { 1 } { 2 } , } \end{array}$ —here the linear program is forced. Hence

$$
\begin{array} { r } { S ^ { \{ 1 \} \{ 2 \} } ( X  Y ) = H ( Y ) - ( \frac { 1 } { 2 } h _ { A } + \frac { 1 } { 2 } h _ { B } ) = 0 . 8 1 1 3 - 0 . 5 = 0 . 3 1 1 3 \mathrm { \ b i t s } . } \end{array}
$$

Operationally, disclosing the parity $X _ { 1 } \oplus X _ { 2 }$ narrows Y (it excludes the 11 state whenever the inputs differ) without revealing either input, and 0.3113 bits is the most any leak-free observable can achieve.

## A.3 Practical considerations and design levers

The example above exposes both the appeal and the cost of the approach. The appeal is that every atom is the value of a transparent optimization with an operational meaning; there is nothing to postulate. The cost is concentrated in one step—vertex enumeration of the polytope (9)—whose size is governed by dim $\mathcal Q _ { \alpha } = | \operatorname { s u p p } p _ { X } | - \operatorname { r a n k } \mathbf P _ { \alpha }$ . For $n = 2$ with both marginals protected this polytope is a line segment (two vertices) and the linear program is trivial, as we saw; but | supp p<sub>X</sub> grows exponentially in the number of sources, so exact synergistic disclosure is a small-n instrument. The backbone decomposition mitigates the combinatorial explosion of the lattice—its number of nodes grows only linearly in n—but each backbone term still solves a program over the joint support, so it does not by itself lift the exponential dependence on source count. This is why, in the main text, analyses at the scale of many attention heads or many agents necessarily fall back on summary statistics (O-information and pairwise aggregations) rather than the full decomposition; the algorithm here is what those summaries approximate, and what one should run directly whenever the effective number of sources is small (e.g., after grouping heads into a handful of functional modules).

For empirical data the algorithm takes an estimated $p _ { \mathbf { { X } } , Y }$ as input, so its reliability inherits that of the density estimate. Disclosure atoms, like all PID quantities, are data-hungry: the required sample size grows with the alphabet size | supp p<sub>X</sub>|, and naive plug-in estimates of the entropies $h _ { j }$ are positively biased, so bias correction (e.g., Nemenman–Shafee–Bialek or Archer–Park–Pillow estimators) is advisable and should be reported alongside sample counts. These are precisely the reproducibility details that any empirical application—including ours—must document, and they live at this layer rather than in the definition of the measure. A reference implementation of the full pipeline, including the polytope construction and the backbone terms, is available in the syndisc package.<sup>6</sup>

We provide additional actionable recommendations on deciding analysis approaches based on the size of the synergistic system (Table S1) and design levers (Table S2).

<table><tr><td></td><td>Question / stage</td><td>Recommended approach</td><td>Key caveat</td></tr><tr><td>Decde</td><td>What are the parts, target, and timescale?</td><td>Fix parts, target, timescale l, and redundancy measure on theoretical grounds</td><td>Not given by the theory; claims are bounded by these choices. Substantiate claims by demonstrating robustness across granularities (e.g., heads and experts), orders (e.g., pairs/triplets), and targets (e.g., group</td></tr><tr><td></td><td>Small system  $( n \leq \sim 5 )$  , full structure needed</td><td>Exact PID, named redundancy measure (BROJA bivariate;  $I _ { c c s } / I _ { m i n }$  discrete)</td><td>Report the measure; check  $\geq 2$  for robustness</td></tr><tr><td></td><td>Large system, net balance suffices</td><td>Localized/gradient O-information</td><td>Conflates Red/Syn; use per-node, not one global number; report strength measure (S-information or total correlation) to distinguish more synergy from less redundancy; size-matched random baseline to</td></tr><tr><td>smate</td><td>&quot;Is there any emergence?” screen</td><td>Emergence-capacity (pairwise Syn  $> 0 ) ; \Psi _ { \ell } \bar { ( } V ) > \bar { 0 } ( \mathrm { E q . } 4 )$ </td><td>Negative Ψ inconclusive (redundancy subtracted up to n times); consider redundancy-corrected  $\Psi ( k , q )$  [92]</td></tr><tr><td>LLM units</td><td>circuits—which</td><td>Pairwise/subset synergy → per-node ranks (as in the core work)</td><td>Blind to high-order; needs null; add ablation for causal claims</td></tr><tr><td>time</td><td>Dynamical / multi-agent over</td><td>TDMI-based causal-emergence / ΦID</td><td>Requires block-shuffled nulls; timescale l is a choice; state l and report at least one alternative l.</td></tr><tr><td>activations</td><td>Continuous non-Gaussian</td><td>Copula normalization or non-parametric estimators</td><td>Gaussian PID may misestimate; report sample-size sensitivity</td></tr><tr><td>Report</td><td>Reporting Syn estimates</td><td>Establish that findings are generated by interaction, not simple diffusion, shared experimental cues, autocorrelation, estimator bias finite-samples bias, bias from alphabet size, etc.; report bias-corrected estimate against a suitable null (block-shuffled for time series; row-shuffled or functional</td><td>Syn is a residual of large estimated quantities, so error compounds; plug-in MI is positively biased and inflates synergy</td></tr><tr><td>or layers</td><td>Comparing across conditions, models,</td><td>Normalize (by H (target) or I(X; Y )) or use synergy-redundancy ranks; matched nulls per condition</td><td>Differences in source entropy masquerade as differences in synergy; result depends on null family</td></tr><tr><td></td><td>Making causal claims</td><td>Attribute synergy to a design lever; randomize the lever (ablation, persona, ToM prompt, routing k)</td><td>Observational synergy  $\neq$  interventional synergy; correlational PID alone does not license a causal reading</td></tr><tr><td>Inerrpet</td><td>Making mechanism claims</td><td>Treat PID atoms as a description of behavior; support mechanism claims by ablation or by fitting models of increasing interaction order and comparing their fit</td><td>High-order behavior ≠ high-order mechanism [89]</td></tr></table>

Table S1: Practical guidelines for applying information decomposition across the research workflow.

Box S1: Reporting guidelines for synergy claims
<table><tr><td colspan="3"></td></tr><tr><td colspan="3">Claims about synergy or emergence can be compared and disputed only if the choices that bound it are visible. Table S1</td></tr><tr><td colspan="3">recommends which analysis to run; this box outlines how to report them.</td></tr><tr><td>A. The claim</td><td>1. Quantity and sign: exact quantity, expression, sign</td><td>C. The comparison 10. Null model: what it preserves and destroys; surrogates;</td></tr><tr><td>convention.</td><td>2. Order: n, order k detected, how larger systems were</td><td>correction; nulls are one-sided. 11. Strength beside balance: report Ω or Ψe(V) next to</td></tr><tr><td></td><td>reduced and aggregated (rule and range across subsets). 3. Reading: observational vs. interventional; behavior vs.</td><td>total information. 12. Whole-minus-sum caveat: negative Ψe(V) is incon-</td></tr><tr><td></td><td>mechanism; highest rung reached (detect → localize → identify causally → control → predict). 4. Scope: systems, tasks, and scales covered; generality</td><td>clusive; a positive Ψe(V) with a vanishing sum term reflects the macro, not synergy; redundancy is sub- tracted up to n times.</td></tr><tr><td></td><td>asserted and the cross-context evidence; what the claim does not show. B. The modeling choices</td><td>D. The pipeline 13. Ground truth: pipeline recovers known atoms on COPY, XOR, AND, and UNQ; the last separates MMI</td></tr><tr><td></td><td>5. Parts: unit, reason for the granularity, one alternative.</td><td>from BROJA.</td></tr><tr><td></td><td>6. Target: Y or macro V; fixed, derived, or learned; one</td><td>14. Provenance: software versions, code, seeds, replicates, sample counts.</td></tr><tr><td></td><td>alternative. 7. Timescale: static or dynamical; numeric l; sensitivity</td><td></td></tr><tr><td></td><td>across l. 8. Redundancy measure: which, why, non-negativity, a</td><td></td></tr><tr><td></td><td>second measure from a different family, able to disagree with the first.</td><td></td></tr><tr><td></td><td>9. Estimator and sample: data model, residualization, discretisation, N vs. alphabet and effective N (autocor- relation, mixing), bias correction.</td><td></td></tr><tr><td colspan="3">How to use this box. These are questions a claim should be able to answer, not conditions to be satisfied. Every item has legitimate exceptions; state the reason rather than tick the box. An item that cannot be answered should be reported as unanswered rather than omitted. Reasoned deviation is good practice; silent deviation is what this box is meant to end.</td></tr><tr><td>PID Principle</td><td>Observable Signature</td><td>Design Levers</td></tr><tr><td>Synergy requires differentiation</td><td>Identical experts / agents produce only redundancy</td><td>Sparsity and routing in MoE systems (larger k in top-k gating ↑ synergy [101, 29]); data augmentation (↑ redundancy [98]); personas/roles (↑ synergy [78]); division of labor (↑ uniqueness) vs. co-creation (↑ synergy)</td></tr><tr><td>Synergy is fragile, redundancy is robust</td><td>Sharp failure modes vs. graceful degradation; flat-gradient stalling on synergistic tasks; robust OOD when structure is stable vs. brittle if composition shifts; path-dependent collapse</td><td>Dropout (↑ redundancy [2, 74]); BatchNorm / LayerNorm (↑ synergy potential [20]); early stopping (↑ redundancy) vs. extended training, warmup schedules (↑ synergy); individual rewards (↑ independence &amp; uniqueness) vs. shared rewards (↑ synergy); shaped rewards for coordination (↑↑ synergy)</td></tr><tr><td>Synergistic information is non-local</td><td>Not locally attributable; modular interpretability fails</td><td>Attention mechanism (↑ synergy [101]); data curation in multi-modal training (↑ synergy [52]); centralized (↑ synergy) vs. decentralized training execution (↑ redundancy); adaptivity &amp; personalization (↑ synergy [14, 49, 43])</td></tr><tr><td>Synergy provides superlinear capacity</td><td>Superlinear performance gains but data-hungry estimation</td><td>Network width (↓ synergy) vs. depth (↑ synergy); ToM in multi-agent systems &amp; human-AI (↑ synergy; ↑ differentiation [82, 80, 78]); full observability (↓ synergy) vs. partial observability (↑ synergy)</td></tr></table>

Table S2: Overview of control parameters and their effects across layers.

## A.4 Summary of evidence across worked examples

<table><tr><td>Rung</td><td>Synergistic core [101]</td><td>Multi-agent [78]</td><td>Human-AI [80]</td></tr><tr><td>Detect</td><td>Pairwise ΦID synergy ranks</td><td>Ψ &gt; 0; emergence capacity &gt; 0</td><td>Bounded residual; formal PID Syn &gt; 0</td></tr><tr><td>Localize</td><td>Middle layers; heads and experts</td><td>Agent pairs, not triplets; identity-linked roles</td><td>Hard and AI-unsolvable items; κ trait</td></tr><tr><td>Identify causally</td><td>Ablation of synergistic vs. redundant heads</td><td>Randomized Plain/Persona/ToM conditions</td><td>Randomized assignment; users × items crossed; phases rule out learning</td></tr><tr><td>Control</td><td>RL fine-tuning of synergistic regions</td><td>ToM prompt as control parameter</td><td>Open: ToM is measured, not manipulated</td></tr><tr><td>Predict</td><td>Conjectures 1–4</td><td>Conjecture 5</td><td>Conjecture 5</td></tr></table>

Table S3: Evidence ladder across the three empirical settings.

A.5 Summary of PID-related software
<table><tr><td>Name</td><td>License</td><td>Language</td><td>Cont.</td><td>Disc.</td><td>Framework</td><td>Ref.</td></tr><tr><td colspan="7">Partial Information Decomposition</td></tr><tr><td>IDTx1</td><td>GPLv3</td><td>Python</td><td>√</td><td>√</td><td>PID, PED</td><td>[108]</td></tr><tr><td>dit</td><td>BSD-3</td><td>Python</td><td>×</td><td>√</td><td>PID</td><td>[39]</td></tr><tr><td>BROJA_2PID</td><td>Apache 2</td><td>Python</td><td>×</td><td>√</td><td>PID (bivariate)</td><td>[56]</td></tr><tr><td>SxPID</td><td>GPLv3</td><td>Python</td><td>X</td><td>√</td><td>PID</td><td>[57]</td></tr><tr><td>infotheory</td><td>MIT</td><td>Python/C++</td><td>√</td><td>√</td><td>PID</td><td>[11]</td></tr><tr><td>integrated-info-</td><td>BSD-3</td><td>Python</td><td>√</td><td>V</td><td>ΦID</td><td>[55]</td></tr><tr><td>decomp Informeasure</td><td>Artistic-2.0</td><td>R</td><td>X</td><td>√</td><td>PID (bivariate)</td><td>[71]</td></tr><tr><td colspan="7">Higher-order interaction summaries</td></tr><tr><td>HOI</td><td>BSD-3</td><td>Python (JAX)</td><td>√</td><td>√</td><td>O-info, TC, DTC, S-info</td><td>[67]</td></tr><tr><td>THOI</td><td>MIT</td><td>Python (PyTorch)</td><td>√</td><td>X</td><td>O-info, TC, DTC, S-info</td><td>[9]</td></tr></table>

URLs: IDTxl: https://github.com/pwollstadt/IDTxl; dit: https://github.com/dit/dit; BROJA\_2PID: https://github.com/Abzinger/BROJA\_2PID; SxPID: https://github.com/Abzinger/SxPID; infotheory: https://github.com/madvn/infotheory; integrated-info-decomp: https://github.com/Imperial-MIND-lab/integrated-info-decomp; HOI: https://github.com/brainets/hoi; THOI: https://github.com/Laouen/THOI.  
Table S4: Non-exhaustive list of available software packages for computing information decomposition and related quantities. Support spans popular environments including Python and R. Runtime and scalability depend on both the entropy estimator (e.g., Gaussian approximation, non-parametric methods) and the specific decomposition. Packages are grouped by scope: full PID implementations, and scalar higher-order interaction summaries (e.g., O-information).

A.6 Testable predictions
<table><tr><td>Hypothesis</td></tr><tr><td>Grokking as redundancy-to-synergy transition: Track O-information Ω of the hidden representation across training; confirmed if Ω flips from positive to negative within the test-accuracy jump (and an Ω-penalty delays it), falsified if Ω drifts smoothly with train accuracy.</td></tr><tr><td>Jagged intelligence as a factored-representation boundary: Score each task by synergistic depth Syn  $( F _ { 1 } , . . . , F _ { k } ; Y )$  ; confirmed if accuracy falls with depth after partialling out human-rated difficulty, falsified if accuracy is unrelated to  $\operatorname { S y n }$  or difficulty dominates once depth is controlled.</td></tr><tr><td>Test-time compute as synergy unrolling: Measure per-pass synergy (synergy-redundancy rank) for a reasoning vs. base model; confirmed if the reasoning model shows lower per-pass synergy and compute gains shrink with synergistic depth, falsified if its per-pass synergy is equal or higher.</td></tr><tr><td>Multi-turn degradation and path-dependent synergy collapse: Regress the single-turn-minus-multi-turn accuracy drop  $\Delta$  on synergistic depth; confirmed if  $\Delta$  rises with depth and is zero for redundancy-dominated tasks  $( \dot { \Omega } > \bar { 0 } )$  , falsified if  $\Delta$  is</td></tr><tr><td>uncorrelated with depth. Theory of mind as a cross-level control parameter: Vary input ToM demand 1  $( \mathrm { e . g . }$  , via CausalToM from [73]) and measure synergy-redundancy rank</td></tr></table>

Table S5: Hypotheses in testable form.