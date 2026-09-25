# Neuro-symbolic AI for Industrial Configuration

Danilo Valerio Philipp Kogler Stefan Bischof Siemens AG Osterreich<sup>¨</sup>

Thomas Hubauer Huzefa Rangwala Siemens AG

danilo.valerio@siemens.com philipp.kogler@siemens.com bischof.stefan@siemens.com

thomas.hubauer@siemens.com huzefa.rangwala@siemens.com

Editors: Alessandra Mileo, Andrea Passerini and Cogan Shimizu

## Abstract

Large Language Models (LLMs) have shown impressive performance on a wide range of generative tasks. Yet their probabilistic nature makes them, in isolation, fundamentally unsuited for industrial product configuration, where outputs must be syntactically valid, semantically consistent with a knowledge base of hundreds of features and rules, and producible by an existing manufacturing chain. We argue that Neuro-symbolic (NeSy) AI methods lay out a promising path towards industrial-grade configurators that are reliable by design, explainable, and trustworthy. This paper describes a taxonomy of three NeSy integration strategies, namely hybrid inference, hybrid fine-tuning, and hybrid training, exploring their usage in the configuration domain. We report our efort to operationalize NeSy concepts in an industrial configuration copilot and derive a set of practical design choices for deploying trustworthy AI in engineering environments. We close with a discussion of open research challenges we consider most pressing, in particular how to scale NeSy methods from small academic demonstrators to the size of industrial configurators.

## 1. Introduction

Industrial configurators are AI-enabled decision-support systems that transform customer requirements into valid, consistent, and producible product or system specifications while guaranteeing compliance with domain knowledge, engineering constraints, and operational requirements. Industrial configurators have been used successfully in practice for decades across domains such as power generation, factory automation, energy systems, and mobility. In these domains, correctness, traceability, and compliance with strict domain rules are not optional features but contractual and regulatory obligations. For this reason, traditional knowledge-based configurators are based on symbolic AI techniques such as rule-based reasoning, knowledge representation, and constraint satisfaction (Falkner et al., 2016). In a typical workflow, an engineer incrementally specifies product features and requirements, while a symbolic engine ensures that every decision remains consistent with the domain model and does not violate any constraints (Falkner et al., 2020).

The recent rise of LLMs has created both an opportunity and a temptation to make these configurators “talk”. Natural-language-based configuration copilots could substantially lower the entry barrier for engineers, shorten Configure-Price-Quote (CPQ) cycles, and transform a configurator into a conversational interface (Lawless et al., 2024; Kogler et al., 2024, 2026a). Our internal evaluations have however shown that configurators relying solely on LLMs fail to satisfy the stringent reliability requirements of industrial engineering for the following four reasons: (i) Syntactic hallucinations, where the model invents properties, attributes, or concepts that do not exist in the product model; (ii) Semantic hallucinations, where the model assigns invalid values to existing features or violates their semantic constraints; (iii) Producibility violations, where the proposed configuration is internally inconsistent, violates domain rules, and cannot be manufactured; (iv) Intent misalignment, where the generated configuration is valid, yet fails to satisfy the user’s actual requirements and preferences.

Beyond these practical shortcomings lies a broader challenge: industrial AI ultimately depends on vast amounts of structured domain knowledge accumulated over decades. As the benefits of simply scaling data and compute begin to plateau, leveraging this knowledge explicitly will become a key diferentiator for next-generation AI systems. We therefore argue that neuro-symbolic AI represents the most promising path towards industrial-grade configurators, combining the pattern-matching, language-understanding, and generalization capabilities of neural networks with the explicit knowledge representation, constraint satisfaction, and formal correctness guarantees provided by symbolic AI (d’Avila Garcez and Lamb, 2023; Shakarian et al., 2023). This view is increasingly reflected in the broader AI community, where hybrid approaches are widely regarded as a key direction for overcoming the limitations of purely neural and purely symbolic systems.

## 2. Neuro-Symbolic Industrial Configurators

The industrial configuration task we address takes as input a natural-language requirement (a free-text description of the desired product, possibly under-specified or ambiguous) and an encoding of the product model, containing (i) the finite set of product configuration variables, (ii) the legal values for these variables, and (iii) a set of hard constraints across variables, whose violation makes a configuration not producible, e.g., compatibility rules, include/exclude rules, default rules. The system is judged industrial-grade on this task if it produces as output a configuration that simultaneously satisfies four properties:

1. Syntactic correctness $\Phi _ { \mathrm { s y n } } ;$ : Syntax is respected and all variables are valid.

2. Semantic correctness $\Phi _ { \mathrm { s e m } }$ : All values assigned to the variables are legal.

3. Producibility $\Phi _ { \mathrm { p r o d } } \mathrm { : }$ : All hard producibility constraints are satisfied.

4. Intent matching $\Phi _ { \mathrm { i n t } }$ : The resulting product satisfies the user’s requirements.

Approaches relying solely on LLMs address $\Phi _ { \mathrm { i n t } }$ well but increasingly fail on $\Phi _ { \mathrm { s y n } } , \Phi _ { \mathrm { s e m } } ,$ and $\Phi _ { \mathrm { p r o d } }$ as the size of the product model and the number/complexity of constraints grows (Shojaee et al., 2025). Pure symbolic approaches typically guarantee $\Phi _ { \mathrm { s y n } } , \Phi _ { \mathrm { s e m } } ,$ and $\Phi _ { \mathrm { p r o d } }$ but require the user to express requirements in a formal modeling language or GUI. The position of this paper is therefore that the research goal for industrial configuration is to design neuro-symbolic architectures that achieve all four properties simultaneously.

We view neuro-symbolic AI as a layered integration of neural and symbolic techniques (Kautz, 2022) and distinguish three complementary approaches (depicted in Figure 1), characterized by where symbolic knowledge is introduced into the neural pipeline.

Hybrid inference couples a pre-trained LLM with a symbolic reasoning engine that actively constrains generation. The symbolic component operates over a formal knowledge base organized into three layers: (i) a structural schema defining the exchange format (e.g., JSON), (ii) a vocabulary of legal configuration elements such as features, attributes, and components, and (iii) a set of constraints capturing their admissible relationships.

In our reference architecture, reasoning is performed at two levels of granularity. At the lowest level, a token reasoner performs grammar-constrained decoding (Geng et al., 2023) by filtering the next-token distribution of the LLM at every decoding step and ensuring that only tokens compatible with the schema and the partially generated output remain admissible. This mechanism prevents the generation of invalid properties, values, or structures. At a higher level, an element reasoner is invoked whenever a configuration element is completed. Using constraint propagation and look-ahead reasoning, it evaluates the consequences of the partial configuration restricting subsequent generation to the valid subset of elements (Kogler et al., 2026b). For example, selecting an electric motor may require the specification of input voltage and torque before generation can proceed.

![](images/5303f71aa2b5a0ab24eedcce2ad94f44a70593bce0189d9bb426e6af752ab688.jpg)  
Figure 1: Three neuro-symbolic integration strategies for industrial configuration. (a) Hybrid inference constrains generation at decode time. (b) Hybrid fine-tuning uses symbolic feedback as a training reward signal. (c) Hybrid training embeds constraints directly into the model architecture

Hybrid Inference currently represents the most mature neuro-symbolic paradigm for industrial copilots. It requires no retraining of the foundation model, can be integrated with existing configuration engines, and provides strong guarantees by construction: since correctness is delegated entirely to the symbolic layer, the solver guarantees that generated configurations do not contain hallucinated features or values and remain consistent with all domain constraints. Its primary limitation is the additional inference-time complexity introduced by repeated interactions between the neural and symbolic components.

Hybrid fine-tuning shifts the interaction between neural and symbolic components from inference time to training time. Instead of constraining generation through a reasoning engine, the symbolic component is used to provide feedback that guides the adaptation of the model parameters. The objective is to make the LLM internalize domain regularities to produce more reliable output in the absence of explicit symbolic intervention at inference. The central idea is to place a symbolic solver in the training loop. Candidate configurations generated by the neural model are evaluated against the knowledge base and assigned a reward according to their degree of correctness. Solutions that satisfy domain constraints receive positive reinforcement, whereas configurations containing invalid features, inconsistent values, or rule violations are penalized. The resulting feedback signal can be incorporated with techniques such as Reinforcement Learning with Symbolic Feedback (Jha et al., 2025) or Reinforcement Learning with Verifiable Rewards (Lambert et al., 2025).

A reward function based solely on symbolic feedback maximizes the likelihood of producing valid configurations. However, optimizing exclusively for validity can lead to reward-hacking behaviors. For example, the model may converge towards repeatedly generating a small set of highly rewarded configurations regardless of the user’s requirements. To mitigate this issue, we decompose the reward function as $R _ { \mathrm { t o t a l } } ~ =$ $w _ { 1 } \left( R _ { \mathrm { f e a t u r e } } \right) + w _ { 2 } \left( R _ { \mathrm { c o n f i g } } \right) + w _ { 3 } \left( R _ { \mathrm { i n t e n t } } \right)$ where $R _ { \mathrm { f e a t u r e } }$ measures the validity of individual feature assignments (univariate feedback), $R _ { \mathrm { c o n f i g } }$ measures the validity of the complete configuration (multivariate feedback), and $R _ { \mathrm { i n t e n t } }$ measures the degree to which the generated recommendation satisfies user’s requirements. The first two terms are derived from symbolic verification by the constraint solver, while the latter is estimated using an LLM-as-a-judge that evaluates requirement satisfaction from the natural-language interaction.

Compared to hybrid inference, this approach transfers part of the symbolic knowledge into the model parameters themselves. As a consequence, inference becomes significantly faster because the model no longer needs to repeatedly query an external solver. At the same time, the approach can improve the semantic validity of generated configurations and reduce the frequency of hallucinations. However, correctness guarantees become statistical rather than deterministic: the model learns to prefer valid configurations, but cannot guarantee constraint satisfaction for every output. In safety-critical domains, symbolic validation may therefore still be required as a final verification step.

Hybrid training embeds symbolic knowledge directly into the neural network architecture or training objective, such that constraints are enforced by construction. While this approach is currently impractical for LLM-scale models due to scalability and computational challenges, it remains highly relevant for industrial configurators. Our exemplar is a compiled neuro-symbolic recommender system. Historical configuration data are represented as a user–configuration matrix, where rows correspond to customers and columns to product features selected in previous configurations. Recommendation then becomes a constrained matrix-completion problem. We address this problem using a Neural Collaborative Filtering (NCF) model (He et al., 2017), augmented with a Logic Tensor Network (LTN) regularizer (Badreddine et al., 2022) that compiles domain constraints into diferentiable loss terms. During training, the model is therefore optimized not only to reconstruct historical preferences, but also to satisfy engineering rules encoded in the knowledge base.

## 3. Discussion

The three methodologies presented in this paper should be viewed as complementary rather than competing. We envision future industrial configurators combining all three layers: hybrid training for recommendation and decision-support components, hybrid fine-tuning to internalize frequently occurring domain rules and reduce inference costs, and hybrid inference as an outer safety layer providing hard correctness guarantees during user interaction. While the conceptual foundations are increasingly well understood, their application to industrialscale configuration remains in its infancy, and we are currently conducting preliminary experiments on industrial product models to better understand the practical trade-ofs among reliability, latency, explainability, and implementation efort.

Important research challenges remain. First, scalability: most neuro-symbolic approaches have been validated on relatively small academic benchmarks, whereas industrial configurators routinely involve millions of variables, constraints, and engineering artifacts. Developing architectures that preserve symbolic guarantees at such scales remains an open problem. Closely related is the question of knowledge allocation: determining which forms of domain knowledge should remain explicitly represented and which can be safely internalized through fine-tuning or training may ultimately determine the practicality of large-scale industrial neuro-symbolic systems. Finally, evaluation methodology: current studies employ heterogeneous datasets and metrics, making rigorous comparison dificult. Building on initiatives such as IndusCP (Shi et al., 2025) and JsonSchemaBench (Geng et al., 2025), the field would benefit from a shared benchmark for neuro-symbolic configuration. Such a benchmark could play a role analogous to HumanEval in code generation, providing common tasks, evaluation protocols, and reproducible baselines that accelerate progress.

## References

S. Badreddine, A. d’Avila Garcez, L. Serafini, and M. Spranger. Logic Tensor Networks. Artificial Intelligence, 303:103649, 2022. doi: 10.1016/j.artint.2021.103649.

A. d’Avila Garcez and L. C. Lamb. Neurosymbolic AI: the 3rd wave. Artificial Intelligence Review, 56:12387–12406, 2023. doi: 10.1007/s10462-023-10448-w.

A. Falkner, G. Friedrich, A. Haselb¨ock, G. Schenner, and H. Schreiner. Twenty-Five Years of Successful Application of Constraint Technologies at Siemens. AI Magazine, 37(4):67–80, 2016. doi: 10.1609/aimag.v37i4.2688.

A. Falkner, A. Haselb¨ock, G. Krames, G. Schenner, H. Schreiner, and R. Taupe. Solver Requirements for Interactive Configuration. Journal of Universial Computer Science, 26 (3):343–373, 2020. doi: 10.3897/jucs.2020.019.

S. Geng, M. Josifoski, M. Peyrard, and R. West. Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 10932–10952, 2023.

S. Geng, H. Cooper, M. Moskal, S. Jenkins, J. Berman, N. Ranchin, R. West, E. Horvitz, and H. Nori. JSONSchemaBench: A Rigorous Benchmark of Structured Outputs for Language Models. arXiv:2501.10868, 2025. doi: 10.48550/arXiv.2501.10868.

X. He, L. Liao, H. Zhang, L. Nie, X. Hu, and T. Chua. Neural Collaborative Filtering. In Proceedings of the 26th International Conference on World Wide Web, page 173–182, 2017. doi: 10.1145/3038912.3052569.

P. Jha, P. Jana, P. Suresh, A. Arora, and V. Ganesh. RLSF: Fine-tuning LLMs via Symbolic Feedback. Frontiers in Artificial Intelligence and Applications (ECAI 2025), 413:1687–1694, 2025.

H. A. Kautz. The third AI summer: AAAI Robert S. Engelmore Memorial Lecture. AI Magazine, 43(1):105–125, 2022. doi: https://doi.org/10.1002/aaai.12036.

P. Kogler, W. Chen, A. Falkner, A. Haselb¨ock, and S. Wallner. Configuration copilot: Towards integrating large language models and constraints. Proceedings of the 26th International Workshop on Configuration (ConfWS 2024), 3812:101–110, 2024.

P. Kogler, W. Chen, A. Falkner, A. Haselb¨ock, S. Wallner, and R. Comploi-Taupe. Configuration with Generative AI, pages 101–113. Springer Nature Switzerland, 2026a. doi: 10.1007/978-3-032-17163-4 8.

P. Kogler, W. Chen, and A. Haselb¨ock. Towards neuro-symbolic constrained decoding for reliable code generation with llms. In SE2026. Gesellschaft f¨ur Informatik, Bonn, 2026b. doi: 10.18420/se2026-ws 13.

N. Lambert, J. Morrison, and V. Pyatkin et al. Tulu 3: Pushing Frontiers in Open Language Model Post-Training. In Second Conference on Language Modeling (COLM), 2025.

C. Lawless, J. Schoefer, L. Le, K. Rowan, S. Sen, C. St. Hill, J. Suh, and B. Sarrafzadeh. “I Want It That Way”: Enabling Interactive Decision Support Using Large Language Models and Constraint Programming. ACM Transactions on Interactive Intelligent Systems, 14 (3), 2024. doi: 10.1145/3685053.

P. Shakarian, C. Baral, G. I. Simari, B. Xi, and L. Pokala. Neuro Symbolic Reasoning and Learning. SpringerBriefs in Computer Science. Springer Nature, 2023. doi: 10.1007/ 978-3-031-39179-8.

W. Shi, M. Liu, W. Zhang, L. Shi, F. Jia, F. Ma, and J. Zhang. ConstraintLLM: A Neuro-Symbolic Framework for Industrial-Level Constraint Programming. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 15999–16019, 2025. doi: 10.18653/v1/2025.emnlp-main.809.

P. Shojaee, I. Mirzadeh, K. Alizadeh, M. Horton, S. Bengio, and M. Farajtabar. The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity. In Advances in Neural Information Processing Systems, volume 38, pages 108018–108059, 2025. doi: 10.48550/arXiv.2506.06941.