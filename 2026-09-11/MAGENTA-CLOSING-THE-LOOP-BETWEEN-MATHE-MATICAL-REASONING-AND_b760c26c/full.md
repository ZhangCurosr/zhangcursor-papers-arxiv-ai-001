# MAGENTA: CLOSING THE LOOP BETWEEN MATHE-MATICAL REASONING AND LEAN VERIFICATION

Joshua Ong Jun Leang<sup>1,2</sup>, Haonan Li<sup>1</sup>, Zheng Zhao<sup>3</sup>, Xinyi Shang<sup>4</sup> Wenda Li<sup>3</sup>, Zhengzhong Liu<sup>1</sup>, Eric Xing<sup>1</sup>, Shay B. Cohen<sup>3</sup>, Eleonora Giunchiglia<sup>2</sup>

<sup>1</sup>Institute of Foundation Models, <sup>2</sup>Imperial College London

<sup>3</sup>University of Edinburgh, <sup>4</sup>University College London

{Joshua.Ong,Haonan.Li}@mbzuai.ac.ae

## ABSTRACT

Most of mathematical knowledge has been communicated through so-called informal use of mathematics and natural language. With large language models (LLMs) being highly adept in using natural language, they achieve strong performance, yet not perfect, in informal mathematical reasoning. Restraining LLMs to informal reasoning misses out on the opportunity to use the discrete verification abilities that machines offer through machine-checkable proofs. In this paper, we bridge the gap between informal and formal reasoning by integrating Lean signals into the informal reasoning process. We introduce MAGENTA, a training-free agentic pipeline that, given only a natural-language problem, produces an answer, expresses it as a Lean 4 statement, and constructs a machine-checked proof. A statement judge verifies whether the formalisation preserves the original problem, while an error-attribution judge routes failed attempts either to mathematical re-derivation or local Lean repair. MAGENTA achieves 100% accuracy across all evaluated olympiad benchmarks, including AIME 2025, AIME 2026, and HMMT February 2026. When paired with the open-weight K2-HORIZON-7B reasoner, it solves all six IMO 2026 problems. Our analysis shows that statement adjudication is essential for preventing false certificates and that feedback-guided correction outperforms independent resampling on difficult problems.

## 1 INTRODUCTION

LLMs have become increasingly capable mathematical problem solvers (OpenAI, 2026; Anthropic, 2026; Qwen Team, 2026), solving International Mathematical Olympiad (IMO) problems (Huang & Yang, 2025) and beginning to contribute to mathematical research (Chen & Jiang, 2026; Jiang et al., 2026). Much of this progress relies on chain-of-thought (CoT) reasoning (Wei et al., 2022) and testtime scaling (Muennighoff et al., 2025), where models generate long reasoning traces. Yet longer reasoning does not make correctness easier to guarantee: such traces routinely contain errors and dead ends (Leang et al., 2025a; Lyu et al., 2023). LLM judges (Li et al., 2026) and process reward models (Guan et al., 2025; Zhang et al., 2025c) provide useful supervision over such traces, but ultimately remain learned proxies for correctness: they reproduce the failure modes of the models they evaluate and score plausible-but-incorrect reasoning (Yosef et al., 2026; Guan et al., 2025).

Interactive theorem provers and proof assistants such as Lean 4 (Moura & Ullrich, 2021), Isabelle (Paulson, 1994) and Rocq (Coquand & Huet, 1988) offer a different form of supervision: once a theorem is stated, a proof accepted by them provides a certificate of correctness. Systems built on this foundation have reached gold-medal level at the IMO (Chen et al., 2025) and solve 99.2% of miniF2F (Zheng et al., 2022) and 70.0% of PutnamBench (Tsoukalas et al., 2024; Varambally et al., 2026). However, the models used are closed-source, or require parameter counts and search budgets beyond ordinary reach (Chen et al., 2025; Wang et al., 2026). In addition, the results stated above largely address proof search after formalisation. Such benchmarks provide the system with a human-written formal statement and evaluate whether it can prove that statement (Lin et al., 2026; Xin et al., 2024). They therefore assume away a central part of mathematical problem solving: determining precisely what the natural-language problem asks to prove. Extending formal verification to natural-language mathematics consequently requires more than stronger proof search.

![](images/7bc444f8e635d034dc9c4307014ce1bff9537b2b7d8a3e1985add43a7eb9575c.jpg)  
Figure 1: Overview of MAGENTA. The pipeline maps a natural-language problem to an informal solution and candidate answer, a faithfully adjudicated Lean 4 statement, and a machine-checked proof. The illustrated example is taken from AIME 2026, where the initial reasoning incorrectly derives $\pi ^ { 7 }$ instead of the correct expression $\pi ^ { 6 }$ . Rejected formal statements are resampled, while failed proof attempts are classified as either LEAN or MATH errors. LEAN errors trigger local proof regeneration, whereas MATH errors, such as the $\pi ^ { 7 }$ versus $\pi ^ { 6 }$ mistake shown here, are returned to the reasoner for mathematical re-derivation using verification feedback.

To address these issues, we introduce MAGENTA, an end-to-end agentic pipeline, as shown in Figure 1. Given a natural-language problem, MAGENTA produces a reasoning chain and candidate answer, converts them into a Lean 4 statement, and constructs a machine-checked proof. MAGENTA separates these tasks across four roles: A reasoner derives an informal solution and commits to a candidate answer; a formaliser converts the problem and that answer into a Lean 4 statement; a statement judge explicitly checks whether that statement preserves the semantics of the original question and answer, rejecting omitted hypotheses, altered constants, and trivialising assumptions; and a prover uses the informal reasoning as a proof plan. The Lean kernel can certify only that a proof establishes the generated formal statement; it cannot determine whether that statement faithfully represents the original natural-language problem. This semantic gap is therefore the central reliability risk in our setting, and the statement judge mitigates it by checking their alignment before proof generation. In addition, MAGENTA, in contrast to previous work, treats each certification failure as an attribution problem, introducing an error judge that labels it SYNTAX or MATH and reroutes the pipeline based on that.

MAGENTA verifies every problem on AIME 2025, AIME 2026, and HMMT February 2026 with four different reasoners, which indicates that these benchmarks are saturated under our evaluation setting. Paired with K2-HORIZON-7B, MAGENTA solves all six IMO 2026 problems, surpassing frontier closed models (OpenAI, 2026) and reasoners over two orders of magnitude larger (Team, 2026). To our knowledge, this is the smallest reported reasoner to solve all six problems. Since perfect scores may raise concerns of contamination, we further test whether the results can be explained by one-shot recall: many problems reach a verified solution only after verification-guided selfcorrection, and we additionally re-evaluate the pipeline on answer-preserving paraphrases. Standalone reasoners degrade under such perturbations (Mirzadeh et al., 2025; Leang et al., 2025a), whereas MAGENTA remains robust. In summary, our contributions are as follows:

• We introduce MAGENTA, a training-free agentic pipeline that bridges informal mathematical reasoning and formal verification, jointly producing an answer, a Lean 4 formalisation of the problem and answer, and a machine-checked proof through verification-guided self-correction.

• We further evaluate robustness to paraphrasing, characterise how verified performance scales with formalisation and proof-repair budgets, and ablate the two mechanisms that distinguish MAGENTA from a simple retry loop.

## 2 METHODOLOGY

MAGENTA combines natural-language reasoning with verification in Lean. The reasoner proposes solutions but cannot guarantee their correctness, while Lean verifies proofs but cannot determine whether an error is due to an incorrect mathematical solution or a wrong formal implementation. MAGENTA closes this gap by attributing verification failures to the responsible component and routing the resulting feedback accordingly. Figure 1 gives an overview and Appendix A states the notation and algorithm.

## 2.1 COMPONENTS

MAGENTA contains five components, each implemented by a prompted language model except for the deterministic verifier. To describe them, we first define the spaces over which they operate.

Let Q, C, A, S and $\mathcal { P }$ denote, respectively, the spaces of natural-language mathematics problems, informal reasoning chains, candidate answers, Lean 4 theorem statements and Lean 4 proof scripts. Let $\mathcal { E } _ { p }$ and $\mathcal { E } _ { s }$ denote the space of Lean diagnostics produced when a proof fails verification and when a statement fails to elaborate, respectively. Let $\mathcal { O } \in \mathcal { E } _ { s } \cap \mathcal { E } _ { p }$ denote the absence of diagnostics, as occurs on the first iteration before any call to the Lean verifier. Let $\mathcal { F } _ { \mathrm { m a t h } } = ( \mathcal { C } \times \bar { \mathcal { A } } \times \mathcal { S } \times$ ${ \mathcal { P } } \times { \mathcal { E } } _ { p } ) \cup \{ \emptyset \}$ denote the space of all correction feedback signals that get generated in case of a mathematical mistake, where ∅ denotes the absence of feedback. Given a set ${ \bar { \boldsymbol { B } } } ,$ we denote by $\Delta ( B )$ the set of distributions over $B ,$ and by $x \sim P$ a sample $x \in B$ from a distribution $P \in \Delta ( B )$

Reasoner $R _ { \bullet }$ The reasoner ${ R \mathrel { : } } \ \mathcal { Q } \times \mathcal { F } _ { \mathrm { m a t h } } \to \Delta ( \mathcal { C } \times \mathcal { A } )$ maps a problem $q \in \mathcal { Q }$ and a mathematical correction signal $\varphi \in \mathcal { F } _ { \operatorname* { m a t h } }$ to a distribution over pairs $( c , a )$ , where $c \in { \mathcal { C } }$ is an informal reasoning chain and $a \in { \mathcal { A } }$ is a candidate answer. We write $( c , a ) \sim R ( q , \varphi )$ for a sample from the reasoner model obtained by inserting $q$ and $\varphi$ into its prompt and extracting c and a from the generated output. On the initial attempt, $\varphi = \varnothing ;$ a non-empty $\varphi$ arises only after the error judge returns MATH, indicating that the previous reasoning chain contains a mathematical error.

Formaliser $F .$ The formaliser $F : \mathcal { Q } \times \mathcal { A }  \Delta ( \mathcal { S } )$ maps a problem $q \in \mathcal { Q }$ and candidate answer $a \in { \mathcal { A } }$ to a distribution over the space $s$ of Lean 4 theorem statements. We write $s \sim F ( q , a )$ for a sample from the formaliser model obtained by inserting q and a into its prompt and extracting the generated Lean statement. Conditioning on a makes the statement assert the proposed answer rather than pose an open-ended existential goal. The resulting statement remains a candidate until it passes both Lean elaboration and semantic adjudication.

Prover $P \cdot$ The prover $P : \mathcal { Q } \times \mathcal { C } \times \mathcal { S } \times \mathcal { E } _ { \mathrm { p } } \to \Delta ( \mathcal { P } )$ maps a problem $q \in \mathcal { Q } .$ , an informal chain $c \in { \mathcal { C } }$ , an accepted formal statement $s \in S$ and an optional diagnostic to a proof script $\epsilon \in \mathcal { E } _ { p }$ to a distribution over the space $\mathcal { P }$ of Lean 4 proof scripts. We write $\pi \sim P ( q , c , s , \epsilon )$ , for a sample from the prover model obtained by inserting $q , c ,$ s and ϵ into its prompt. On the first attempt $\epsilon = \emptyset$ , after a SYNTAX error the next attempt receives the latest diagnostic while $c ,$ a and s remain fixed.

Verifier $V \cdot$ The verifier is the deterministic Lean 4 checking environment augmented with SafeVerify, a certification wrapper that rejects a submission whenever the declared theorem differs from the accepted statement or the script uses a forbidden construct such as sorry, admit, a new axiom or native decide $( \mathrm { A }$ ppendix $\mathbf { B } . 2 )$ . The verifier consists of two deterministic checks: the statement verifier $V _ { \mathrm { s } } : \mathcal { S } \to \{ \top \} \cup \mathcal { E } _ { \mathrm { s } }$ and the proof verifier $V _ { \mathrm { p } } : \mathcal { S } \times \mathcal { P }  \{ \top \} \cup \mathcal { E } _ { \mathrm { p } }$ , where $\top$ denotes a successful check. The verifier $V _ { s }$ checks that a generated statement is closed and elaborates in Lean, which is a well-formedness check and does not establish that the statement is true or faithful to the problem. The verifier $V _ { p }$ checks that π is a complete, policy-compliant Lean proof of the unchanged statement s; otherwise, the verifier returns a diagnostic $\epsilon \in \mathcal { E } _ { \mathrm { p } }$

Judge J. A single language model, prompted in two distinct ways, implements two distinct judging functions. The statement judge $J _ { \mathrm { s } } : \mathcal { Q } \times \mathcal { A } \times \mathcal { S } \to \Delta ( \{ 0 , 1 \} )$ maps a problem $q \in \mathcal { Q } .$ , candidate answer $a \in { \mathcal { A } } ,$ , and Lean 4 statement $s \in S$ to a distribution over acceptance decisions. Acceptance denotes that the Lean 4 statement faithfully represents the original problem and candidate answer. The error judge $J _ { \mathrm { e } } : \mathcal { Q } \times \mathcal { C } \times \mathcal { A } \times \mathcal { S } \times \mathbf { \dot { \mathcal { P } } } \stackrel { } { \times } \mathcal { E } _ { p } \to \Delta ( \{ \mathbf { M } \mathbf { \bar { A } T H } , \hat { \mathbf { S } } \mathbf { Y } \mathbf { N T A X } \} )$ ), maps the complete context of a failed proof attempt to a distribution over error labels. We write $\ell \sim J _ { \mathrm { e } } ( q , c , a , s , \pi , \epsilon )$ where $\ell = \mathrm { \mathbf { M A T H } }$ indicates that the mathematical reasoning should be revised, while $\ell = \boldsymbol { S }$ YNTAX indicates that the error lies in the formal implementation. Thus, $J _ { \mathrm { s } }$ determines whether the generated statement faithfully represents the intended claim, whereas $J _ { \mathrm { e } }$ determines which component should be revised after proof verification fails.

## 2.2 THE MAGENTA PROCEDURE

MAGENTA is an iterative verification-guided refinement procedure in which Lean 4 feedback is used to identify and correct errors in the reasoner’s proposed solution. Verification failures are attributed either to the mathematical reasoning or to its formal implementation, allowing the pipeline to revise only the component responsible for the failure. Below we describe each step of the procedure, while the pseudocode can be found in Appendix A.2.

Step 1: Reasoning & Answer Generation (Reasoner R). Let $T$ be a user-defined upper bound on the number of calls to the reasoner, and let $t = 0 , \ldots , T - 1$ index these calls. At iteration $t ,$ the reasoner samples $( c _ { t } , a _ { t } ) \sim R ( q , \varphi _ { t } )$ , where $\varphi _ { 0 } = \varnothing$ . For $t > 0$ , the reasoner is invoked only after the error judge $J _ { \mathrm { e } }$ classifies the previous failed proof attempt as MATH; in this case, $\varphi _ { t } \in \mathcal { F } _ { \operatorname* { m a t h } }$ is constructed from the failed attempt and the corresponding verifier diagnostic. Thus, each MATH error triggers a new reasoning chain and candidate answer, while statement resampling and proof repair leave the current pair $\left( { { c _ { t } } , { a _ { t } } } \right)$ unchanged. The procedure terminates if a proof is verified or if the budget of $T$ reasoner calls is exhausted.

Step 2.a: Formal Statement Generation (Formaliser $F )$ . Let M be a user-defined upper bound on the number of statement samples drawn for a fixed candidate answer, and let $m = 0 , \ldots , M - 1$ index these samples. At sample $m ,$ , the formaliser draws $s _ { t } ^ { m } \sim F ( q , a _ { t } )$ , whose hypotheses encode the conditions of $q$ and the final answer provided by $a _ { t }$

Step 2.b: Formal Statement Verification (Statement Verifier $V _ { \mathrm { s } } )$ . Each candidate statement $s _ { t } ^ { m }$ is first submitted to the Lean $^ 4$ verifier. If $V _ { \mathrm { s } } ( s _ { t } ^ { m } ) \neq \top$ , the statement is discarded and the formaliser $F$ is resampled. Otherwise, $s _ { t } ^ { m }$ proceeds to statement adjudication. This verification step is restarted for every new reasoning pair $\left( { { c _ { t } } , { a _ { t } } } \right)$ and repeated within the same reasoning iteration until a statement passes verification, while $\left( { { c _ { t } } , { a _ { t } } } \right)$ remains fixed.

Step 2.c: Formal Statement Adjudication (Statement Judge $J _ { \mathrm { s } } ) .$ Given a candidate statement $s _ { t } ^ { m }$ such that $V _ { \mathrm { s } } ( s _ { t } ^ { m } ) = \top$ , the statement judge samples $b \sim \ J _ { \mathrm { s } } ( q , a _ { t } , s _ { t } ^ { m } )$ . The judge is tasked to accept $s _ { t } ^ { m }$ only if it faithfully represents the original problem $q$ and assert the candidate answer $a _ { t }$ $\begin{array} { r } { \mathrm { ~ I f ~ } b = 0 . } \end{array}$ , the candidate is discarded and control returns to step $^ { 2 , }$ where the formaliser $F$ is resampled under the same statement budget M. If $b = 1$ , the candidate is accepted, denoted $s _ { t } ,$ and passed to the prover. A failure at either step 2.1 or step 2.2 does not trigger a new call to the reasoner; hence, statement resampling leaves the current reasoning pair $\left( { { c _ { t } } , { a _ { t } } } \right)$ unchanged. The statement judge therefore determines what the pipeline attempts to prove, and is the component that connects Lean’s guarantee about the accepted formal statement $s _ { t }$ to the corresponding claim about the original problem $q ;$ §2.3 discusses the resulting guarantee in detail.

Step 3: Formal Proof Generation (Prover $P )$ . Let K be a user-defined upper bound on the number of proof attempts for an accepted statement, and let $k = 0 , \ldots , K - 1$ index these attempts. At attempt $k = 0$ , the prover samples $\pi _ { t } ^ { k } \sim P ( q , c _ { t } , s _ { t } , \epsilon _ { k } )$ , where $\epsilon _ { 0 } = \mathcal { D }$ and the informal solution $c _ { t }$ acts as a proof plan.

Step 4.a: Proof Verification (Proof Verifier $V _ { \mathrm { p } } )$ . Each candidate proof $\pi _ { t } ^ { k }$ is submitted to the proof verifier $V _ { \mathrm { p } }$ . If $V _ { p } ( s _ { t } , \pi _ { t } ^ { k } ) ~ = ~ \top$ , then the Lean 4 verifier has accepted the proof, and the pipeline terminates and returns $( s _ { t } , a _ { t } , c _ { t } , \pi _ { t } ^ { k } )$ . The associated informal solution $c _ { t }$ and candidate answer $a _ { t }$ are retained as the final reasoning chain and answer. Otherwise, $V _ { \mathrm { p } } ( s _ { t } , \pi _ { t } ^ { k } ) = \epsilon _ { t } ^ { k } ,$ , i.e., the verifier has returned an error, which gets passed to the error judge.

Step 4.b: Error attribution (Error Judge $J _ { \mathrm { e } } )$ . Each candidate proof $\pi _ { t } ^ { k }$ that failed the verifier check is submitted together with its corresponding error trace $\epsilon _ { t } ^ { k }$ to the error judge $J _ { \mathrm { e } } ,$ , which returns $\ell \sim J _ { \mathrm { e } } ( q , c _ { t } , a _ { t } , s _ { t } , \pi _ { t } ^ { \bar { k } } , \epsilon _ { t } ^ { k } )$ . If $\mathbf { \dot { \ell } } = \mathbf { S } \mathbf { Y } \mathbf { N } \mathbf { T } \mathbf { A } \mathbf { \bar { X } }$ then the algorithm loops back to step 3 a new candidate proof $\pi _ { t } ^ { k + 1 }$ is sampled from the prover. Thus, each SYNTAX error triggers a repair of the proof script alone, while $c _ { t } , a _ { t }$ and $s _ { t }$ remain fixed. If ℓ = MATH the informal solution is deemed false or insufficient, and the pipeline builds $\varphi _ { t } \in \mathcal { F } _ { \operatorname* { m a t h } }$ from the failed attempt and returns to step 1. Only a MATH label may revise the reasoning and answer.

## 2.3 ROLE OF THE JUDGE AND SCOPE OF THE GUARANTEE

Why is statement adjudication necessary? Lean verifies a proof only relative to the generated statement: $V _ { \mathrm { p } } ( s , \pi ) = \top$ entails that π is a valid proof of $s ,$ but this does not imply that s faithfully represents the original problem $q$ and answer $a .$ We refer to this as the autoformalisation gap, which is due to the nature of LLMs and use of natural language. The statement judge prevents these false certificates, working under the assumption that it is an easier task to confirm alignment between formal and informal statements compared to the generation of the formal statement. As this semantic check is performed by a learned judge rather than the Lean kernel, the final output is a soft certificate: formally sound with respect to s, but only as relevant to q as the generated statement is faithful. Further ablation study of Statement Judge can be referred to §4.3.

Why is error attribution necessary? When proof verification fails, $V _ { \mathrm { p } }$ identifies where Lean failed. However, the same rejection may arise because the informal mathematics is false or insufficient, or because a correct argument has been implemented incorrectly in Lean. These cases require varying actions: the former requires a new mathematical derivation, or the latter requires proof repair at fixed mathematical content. The error judge supplies this decision: $\ell \sim J _ { \mathrm { e } } ( q , c , a , s , \pi , e )$ where $\ell \in \left\{ \mathrm { M A T H } , \mathrm { S Y N T A X } \right\}$ . The resulting label determines the correction path:

$$
\ell = { \mathrm { S Y N T A X } } \implies \pi ^ { \prime } \sim P ( q , c , s , e ) , \qquad \ell = { \mathrm { M A T H } } \implies ( c ^ { \prime } , a ^ { \prime } ) \sim R ( q , \varphi ) .\tag{1}
$$

A SYNTAX label preserves $( c , a , s )$ and uses the diagnostic to repair the Lean proof, whereas a MATH label constructs $\varphi \in \mathcal { F } _ { \operatorname* { m a t h } }$ and returns control to the reasoner. Without $J _ { \mathrm { e } } ,$ , the closed-loop pipeline is underdetermined: a failed verification provides no basis for choosing between proof repair and mathematical re-derivation.

A mistake-bound perspective Given a question $q \in \mathcal { Q }$ , let $\mathcal A ( q )$ be the possible answers (correct or not) that an LLM could provide for it. We might ask the question, in terms of the mistake-bound model of learning (Cesa-Bianchi & Lugosi, 2006), what is the maximal number of “mistakes” $M ( q )$ (verifier returning $0 ) , { } ^ { 1 }$ our pipeline, or an improved one, would make before we find a verified answer? If the LLM does not loop in its answer (and the verifier provides a significant feedback), and the context length is limited (hence, the answer too), then this “dimension” of $\mathcal { Q }$ (over all questions, see also the Littlestone dimension in online learning; Littlestone 1987) is finite.

A simple result speaks about this dimension in terms of the feedback loop removing a fraction $\gamma \in ( 0 , 1 )$ of potential (mistaken) answers left at each step shows that $M ( q ) \leq \log _ { 1 / ( 1 - \gamma ) } | { \cal A } ( q ) |$ For example, if at each point we remove half of the possible answers from the LLM prediction based on our verifier feedback, then $M ( q ) \leq \log _ { 2 } { \left| \mathcal { A } ( q ) \right| }$ . Given the complexity of LLM inference, it is difficult to point and prove an exact value of γ or $M ( q )$ , but we believe the mistake-bound model is a promising learning model to analyse such pipelines and we leave it for future work to estimate γ in our case on a larger set of answers and questions. We provide further empirical analysis of the number of mistakes our pipeline makes in §3.2.

## 3 EXPERIMENTS

## 3.1 EXPERIMENTAL SETUP

Models. We vary R across both open-weight and proprietary families: K2-Horizon-7B and 375B, Qwen3.8-27B and GPT-5.6-Sol (Codex). All other components are held fixed unless stated otherwise: Goedel-Formaliser-32B as $F ,$ , DeepSeek-V4-Flash for both judge roles, and Leanstral-1.5 as the prover $P .$ . This provides an entirely open weight pipeline. §4.1 substitutes GPT-5.6-Sol for $F$ and for $P .$ Proofs are checked with Lean v4.29.1 and MATHLIB under SafeVerify; decoding settings and the budgets T, M, K are provided in Appendix B.2.

Datasets. We evaluate on AIME 2025, AIME 2026 and HMMT February 2026: natural-language competition problems with a numerical answer and no accompanying formal statement. We additionally evaluate our smallest reasoner K2-HORIZON-7B with MAGENTA on IMO 2026 problems.

Baselines. We compare against state-of-the-art baselines: Claude Opus 5, Gemini 3.7 Flash, and Kimi K3. We let each baseline only generate one answer (i.e., pass@1 rate).

## 3.2 EXPERIMENTAL RESULTS

MAGENTA is robust to model choice. As shown in Table 1, MAGENTA’s performance is not tied to a particular base model used as reasoner. The average gain across the three tasks over the corresponding standalone reasoner ranges from 8.6 points for Qwen3.8-27B to 25.8 points for K2- HORIZON-7B. In order to show that generalist closed models can be used instead of specialised open source models as formaliser and prover, we replace both of them with GPT-5.6-Sol (Codex). This once again yields the perfect scores shown in Table 1. Consistency across reasoner, formaliser and prover configurations suggests the gains arise from the verification-guided pipeline rather than from any single model; §4.1 analyses what these configurations cost.

Table 1: Accuracy (%) on competition-level mathematics benchmarks, covering 93 problems in total. Light-grey rows denote results obtained with MAGENTA. Green values in parentheses report absolute improvements over the corresponding base reasoner in percentage points.
<table><tr><td>Model</td><td>AIME 2025</td><td>AIME 2026</td><td>HMMT Feb. 2026</td><td>Overall</td></tr><tr><td>Claude Opus 5</td><td>100.00</td><td>100.00</td><td>96.97</td><td>98.92</td></tr><tr><td>Gemini 3.7 Flash</td><td>100.00</td><td>100.00</td><td>90.91</td><td>96.77</td></tr><tr><td>Kimi K3</td><td>93.33</td><td>90.00</td><td>78.79</td><td>87.10</td></tr><tr><td>K2-Horizon-7B</td><td>83.33</td><td>80.00</td><td>60.61</td><td>74.19</td></tr><tr><td>+ MAGENTA</td><td>100.00 (↑ 16.67)</td><td>100.00 (↑ 20.00)</td><td>100.00 (↑ 39.39)</td><td>100.00 (↑ 25.81)</td></tr><tr><td>K2-Horizon-375B</td><td>93.33</td><td>90.00</td><td>72.73</td><td>84.95</td></tr><tr><td>+ MAGENTA</td><td>100.00 (↑ 6.67)</td><td>100.00 (↑ 10.00)</td><td>100.00 (↑ 27.27)</td><td>100.00 (↑ 15.05)</td></tr><tr><td>Qwen3.8-27B</td><td>100.00</td><td>86.67</td><td>87.88</td><td>91.40</td></tr><tr><td>+ MAGENTA</td><td>100.00 (↑ 0.00)</td><td>100.00 (↑ 13.33)</td><td>100.00 (↑ 12.12)</td><td>100.00 (↑ 8.60)</td></tr><tr><td>GPT-5.6-Sol (Codex)</td><td>90.00</td><td>86.67</td><td>78.79</td><td>84.95</td></tr><tr><td>+ MAGENTA</td><td>100.00 (↑ 10.00)</td><td>100.00 (↑ 13.33)</td><td>100.00 (↑ 21.21)</td><td>100.00 (↑ 15.05)</td></tr></table>

K2-HORIZON-7B+MAGENTA solves all IMO 2026 problems. Coupled with MAGENTA, K2- HORIZON-7B solves all six IMO 2026 problems. To our knowledge, this is the smallest reported reasoner to score perfectly on this task. Unlike answer-only benchmarks, IMO solutions are inherently multi-step: they require answering a sequence of intermediate subproblems, whose results are then composed to obtain the final answer. Despite perfect final answer accuracy and formal verification we observe a residual gap between the informal explanation and the formal statement and proof. We therefore append a final alignment judge, instantiated with K2-HORIZON-375B, which returns incomplete proof when the two do not support the same conclusion (Appendix F); its feedback yields substantially more complete reasoning chains in our qualitative analysis. Formal correctness is thus attainable with a compact reasoner, whereas a fully aligned, communicable informal derivation remains a separate challenge.

Verification-guided correction trades test-time   
computation for model scale. Most reasoners do   
not solve every problem first time, but their verified   
solve rates rise consistently across correction rounds.   
On AIME 2026, K2-HORIZON-7B needs an average   
of 5 correction rounds to reach a verified solution   
against 2 for K2-HORIZON-375B. Both ultimately   
attain perfect accuracy, but the larger reasoner gets   
there with fewer feedback cycles while the 7B model   
closes the gap by spending additional verification-  
Figure 2: Self-corrections required by K2- guided computation. On the more complex IMO 2026,

![](images/1260dc6e9ffd167a525d964511c6069abe5fdc729781fa7149730c50e6514a22.jpg)

HORIZON-7B+MAGENTA on IMO 2026.K2-HORIZON-7B needs an average of 5.7 rounds, with per-problem trajectories in Figure 2. These averages count only revisions triggered by mathematical errors; compiler-level errors require further repair iterations (§4.1).

## 4 ANALYSIS

## 4.1 SCALING ANALYSIS

We analyse how test-time computation is distributed across the two stages of MAGENTA. We first examine how the choice of formaliser affects the number of samples required to obtain an accepted statement, and then compare the cost of different proof backends.

## 4.1.1 FORMAL STATEMENT GENERATION

The formaliser governs how quickly the pipeline reaches a faithful, provable statement. CODEX succeeds on its first call for approximately 66% of AIME 2026 problems against 42% for Goedel, and covers the benchmark within 6 calls rather than approximately 120 (Figure 3); at a six-call budget Goedel covers only about 68%. The distributions differ in shape, not merely location:

CODEX is compact, and indeed $M \ = \ 6$ represents a very small number of calls, whereas the Goedel curve spans two orders of magnitude and any practical M truncates it. Truncation is not a correctness failure— adjudication rejects unfaithful statements before proof effort is spent—but an exhausted statement budget forfeits the problem, thus a long formalisation tail converts directly into lost coverage.

## 4.1.2 FORMAL PROOF GENERATION

![](images/4541ad76a76bb699708a9098f523adfa7e9f744603f41d37f2c4bc2cdf24067e.jpg)  
Figure 3: Cumulative pass rate against formal-statement generation calls on AIME 2026.

Both prover backends verify 100% of AIME 2026, thus accuracy by itself cannot separate them. Reporting accuracy as a function of a budget rather than at a single

operating point is standard practice in agent evaluation (Kapoor et al., 2025) and in the test-timescaling literature (Muennighoff et al., 2025; Snell et al., 2025). To make claims about self-repair meaningful, we report the cumulative pass rate:

$$
\mathrm { P a s s } _ { u } ( \beta ) = \frac { 1 } { | \mathcal { D } | } \sum _ { q \in \mathcal { D } } \mathbf { 1 } [ \mathrm { V e r } ( q ) = 1 \land u ( q ) \le \beta ] ,\tag{2}
$$

where $u ( q )$ measures measures the number of tokens necessary to produce the proof, $\operatorname { V e r } ( q )$ indicates whether the proposed resolution of q passes independent verification, and $\beta$ represents the tokens budget. A single cost axis is not well defined across backends: LEANSTRAL runs as a multi-turn agent, thus one invocation can hide hundreds of compiler queries, while a CODEX CLI call reports one call regardless of what it does internally. We therefore count a cost as external when a failure survives the prover’s own repair attempts and returns to the error judge, and internal when incurred inside a single invocation. External costs are defined identically for both backends; internal costs are observable only for LEANSTRAL (Figure $4 ) _ { \cdot }$ , and are paid on local hardware rather than billed per request.

![](images/66e8e1447d261c36a7089b28dec4b774d804eb52283affdf64067414731fd013.jpg)  
Figure 4: Internal LEANSTRAL agent calls on the winning trajectory. This measure has no CODEX counterpart, as its CLI does not expose per-invocation turns.

Measured externally, LEANSTRAL reaches the shared ceiling more cheaply: it solves 42% of problems with no failed external verification round against 17% for CODEX (Figure 5b), reaches full coverage on approximately $6 \times 1 0 ^ { 4 }$ reasoner tokens versus $9 \times 1 0 ^ { 5 }$ (Figure 5c), and produces verified proofs roughly half as long, $1 . 1 \times 1 0 ^ { 4 }$ against $2 \times 1 0 ^ { 4 }$ tokens (Figure 5d).

This efficiency is bought with computation inside LEANSTRAL’s agentic loop: its winning trajectories consume approximately $3 \times 1 0 ^ { 2 }$ internal calls at median coverage and up to $\mathrm { 3 \times 1 0 ^ { \overline { { 3 } } } }$ at full coverage (Figure $4 ) ,$ , and once these are counted the ordering reverses (Figure 5a). The claim is therefore not that the open-weight prover is cheaper outright, but that at equal verified accuracy it moves cost from metered rounds into local computation one controls. Since CODEX performs an unknown amount of internal work, its external measurements is only a lower-bound on its true com pute. Attribution matters mainly in the remaining tail: an external failure reaches the router on 58% of problems with LEANSTRAL against 83% with CODEX.

## 4.2 ROBUSTNESS TO ANSWER-PRESERVING PARAPHRASE

To control for possible data leakage, we use DeepSeek-V4-Flash to paraphrase every AIME 2026 problem while preserving its quantities, hypotheses, goal and reference answer, check each paraphrase before use using CODEX, and measure the robustness gap $\Delta$ between original and paraphrased accuracy. Prior work (Mirzadeh et al., 2025) shows that standalone reasoners are sensitive to such answer-preserving changes, and Table 2 reproduces this: paraphrasing costs K2-HORIZON-375B 3.3 points and K2-HORIZON-7B 10.0 points, whereas both retain 100% with MAGENTA $( \Delta = 0 )$ . Perturbing the numerical quantities, so that the reference answer itself changes, would be the stronger test; we therefore read this as necessary rather than sufficient evidence against contamination.

Table 2: Accuracy (%) on original and paraphrased AIME 2026. $\Delta$ is the % decrease.
<table><tr><td>Model</td><td>Original</td><td>Paraphrased</td><td>Δ</td></tr><tr><td>K2-Horizon-375B</td><td>90.0</td><td>86.7</td><td>-3.3</td></tr><tr><td>+ MAGENTA</td><td>100.0</td><td>100.0</td><td>0.0</td></tr><tr><td>K2-Horizon-7B</td><td>80.0</td><td>70.0</td><td>-10.0</td></tr><tr><td>+ MAGENTA</td><td>100.0</td><td>100.0</td><td>0.0</td></tr></table>

![](images/f4fc234a40cd849939ef655a509daca0f4d1fd788ee4673ec36964bded471e27.jpg)  
(a) Total recorded calls

![](images/d40fa9c11e4f46e409ffe9b80429ec38f11e52fc0c794e83ffc94ae8264aaa67.jpg)  
(b) Failed external Lean verification rounds

![](images/ea4dd5e60df493a7d8d6e3eb0fa6e58c23ba6654de324258f1a1b1144a39df0c.jpg)  
(c) Reasoner (K2-HORIZON-7B) tokens

![](images/3f35c8d20f8e92d9a4759324881b7d30682e9f9cbf16be4d760908dbf92b7eb2.jpg)  
(d) Final verified Lean-code length  
Figure 5: Cumulative pass rate against four cost measures on AIME 2026 (Equation 2). Both backends reach 100%, so the curves separate on budget, not on ceiling. Panels (b)–(d) are external measures, defined identically for both backends, and LEANSTRAL is left-shifted on each.

## 4.3 ABLATION STUDY

We ablate the two mechanisms that distinguish MAGENTA from a generic retry loop: the statement judge $J _ { \mathrm { s } } ,$ which decides what is proved, and the error judge $J _ { \mathrm { e } } ,$ , which decides how a failed attempt is retried. Both rerun the same harness with one component disabled, and the rest as in §3.1.

Statement adjudication. A certificate is meaningful only when the proved statement preserves the original problem, so we separate two outcomes. Let Ver be the fraction of problems for which the pipeline returns a verified proof and VerCor the fraction for which it does so and the certified answer matches the reference. The false certification rate FCR =

Table 3: Effect of statement adjudication with Goedel formaliser on AIME 2026.
<table><tr><td>Setting</td><td></td><td>Ver↑ VerCor ↑ FCR↓</td><td></td></tr><tr><td>With  $J _ { \mathrm { s } }$ </td><td>63.3%</td><td>63.3%</td><td>0.0%</td></tr><tr><td>Without  $J _ { \mathrm { s } }$ </td><td>36.7%</td><td>20.0%</td><td>45.5%</td></tr></table>

1 − VerCor/Ver is then the fraction of returned certificates whose answer is nonetheless wrong. Since the verifier V is sound, $\mathrm { F C R } > 0$ is possible only through an unfaithful formalisation, which makes it a direct measurement of the residual gap identified in §2.3. We disable $J _ { \mathrm { s } }$ by accepting the first elaborated statement unconditionally. Table 3 reports the result.

Without the judge Ver falls from 63.3% to 36.7% even though every first-sampled formalisation is now accepted. The judge, therefore, acts as an additional filter on the formal statement beyond semantically aligning it with the informal one. The effect on correctness is larger still: VerCor falls 43.3 points to 20.0% and FCR rises to 45.5%, so nearly half the surviving certificates prove something other than the problem asked, but not necessarily contradictory (Appendix E.1).

Resampling versus diagnostic-conditioned self-correction. We next replace diagnosticconditioned self-correction with independent resampling, which discards compiler feedback and draws a fresh proof each attempt. This baseline is stronger than it may appear: for a problem with single-shot success probability $p > 0$ , after n parallelisable attempts of using V as a sound selector, the probability is increased to $1 - ( 1 - p ) ^ { n } \geq 1 - e ^ { - n p }$ On AIME 2026 resampling with K2-HORIZON-7B solves 29/30 (96.7%) against 30/30 for self-correction; in the single failure the initial answer is wrong, so the statement is unprovable and further proof samples cannot recover it.

The gap widens on IMO 2026, where, despite 8–16 independent generations per problem, resampling verifies only 1/6 (16.7%) while self-correction verifies all six, so p is effectively zero under our budget for most IMO problems. Conditioning on a diagnostic moves probability mass onto proofs the unconditioned distribution does not reach. The benefit in error attribution (MATH versus SYNTAX) is routing the next attempt to the right component based on the previous attempt.

## 5 RELATED WORK

LLMs for Mathematical Reasoning. LLMs have achieved strong performance in competition mathematics through CoT reasoning (Wei et al., 2022) and test-time scaling (Albalak et al., 2025; Shao et al., 2024; Zhang et al., 2025b). As plausible reasoning traces may still lead to incorrect solution, prior work applies LLM judges (Yosef et al., 2026), process reward models (Liu et al., 2024), and related scoring or selection methods (Ospanov et al., 2026; Leang et al., 2026a). These signals remain learned proxies that may share the models’ failure modes (Leang et al., 2026a; 2025a). MAGENTA instead uses Lean as a deterministic verifier of the generated formal proof.

Theorem Proving and Autoformalisation. LLM-based theorem proving has advanced rapidly, with specialised proving models such as Pythagoras-Prover (Leang et al., 2026b), Goedel-Prover (Lin et al., 2026), and others (Ma et al., 2026; Wang et al., 2026), together with agentic closed-source models (Liu et al., 2026; Xu et al., 2026), have rapidly advanced LLM theorem proving, albeit often through substantial test-time scaling (Varambally et al., 2026; Klingner et al., 2026; Chen et al., 2025; Lin et al., 2026). However, current benchmarking assume an accurate formal statement as the input. Producing faithful formal statements remains challenging: an incorrect formalisation may admit a valid proof without certifying the original problem (Cornish et al., 2026).

Formal–Informal Distillation. Previous work has explored connections between informal reasoning and formal proof (Varambally et al., 2026; Quan et al., 2025), but often relying solely on closed-source models, and underperforms state-of-the-art mathematical reasoners (Liu et al., 2025; Zhou et al., 2024; Yao et al., 2025; Zhang et al., 2025a). Carefully curated formal–informal pipelines can improve reliability (Sazonova et al., 2026), while Theorem Prover as a Judge uses GPT-4o and formal verification to validate informal traces during synthetic-data generation (Leang et al., 2025b). Tsoukalas et al. (2026), on the other hand, applies formal-informal distillation on solving mathematical research problems, but rely heavily on closed-source models with large parameter scale. Often, current methods assume that a formal goal is already available (Yao et al., 2025; Varambally et al., 2026; Bourigault et al., 2026). These systems use formal verification as a selection or certification signal, rather than for error attribution.

## 6 CONCLUSION

We introduced MAGENTA, a training-free agentic pipeline that couples informal mathematical reasoning with formal verification. Given only a natural-language problem, it produces a reasoning chain and candidate answer, formalises them as a Lean 4 statement, and generates a checked proof, with a statement judge deciding whether the formalisation preserves the original problem and an error judge routing failed attempts to re-derivation or local repair. MAGENTA verifies every problem on AIME 2025, AIME 2026 and HMMT February 2026 with four reasoners, and with K2- HORIZON-7B solves all six IMO 2026 problems. Our ablation experimental results show that adjudication is what keeps a certificate meaningful and that diagnostic-conditioned correction reaches proofs independent resampling does not. Because the pipeline writes the statement it proves, verification guarantees correctness only relative to the generated formalisation, not unconditionally with respect to the original problem. Closing that gap is a question of better autoformalisation rather than better proof search, which we regard as the main remaining obstacle to Lean-guided natural language reasoning.

## REFERENCES

Alon Albalak, Duy Phung, Nathan Lile, Rafael Rafailov, Kanishk Gandhi, Louis Castricato, Anikait Singh, Chase Blagden, Violet Xiang, Dakota Mahan, and Nick Haber. Big-math: A large-scale, high-quality math dataset for reinforcement learning in language models, 2025. URL https: //arxiv.org/abs/2502.17387.

Anthropic. System Card: Claude Opus 4.6. https://anthropic.com/ claude-opus-4-6-system-card, 2026.

Pauline Bourigault, Xiaotong Ji, Matthieu Zimmer, Rasul Tutunov, and Haitham Bou Ammar. Riskcontrolled lean-as-judge for natural-language mathematical reasoning, 2026. URL https:// arxiv.org/abs/2605.28365.

Nicolo Cesa-Bianchi and Gabor Lugosi. Prediction, Learning, and Games. Cambridge University Press, 2006.

Jiangjie Chen, Wenxiang Chen, Jiacheng Du, Jinyi Hu, Zhicheng Jiang, Allan Jie, Xiaoran Jin, Xing Jin, Chenggang Li, Wenlei Shi, et al. Seed-prover 1.5: Mastering undergraduate-level theorem proving via learning from experience. arXiv preprint arXiv:2512.17260, 2025.

Xiaoyang Chen and Xiang Jiang. Moonshine: An autonomous mathematical research agent centered on conjecture generation, 2026. URL https://arxiv.org/abs/2606.10806.

Thierry Coquand and Gerard Huet. The Calculus of Constructions. ´ Information and Computation, 76(2–3):95–120, 1988. URL https://doi.org/10.1016/0890-5401(88)90005-3.

Rob Cornish, Iacopo Ghinassi, Po-Hung Yeh, Shuqi Liu, Qiyuan Xu, Haoxuan Yin, Dominik Wagner, Wenda Li, Yee Whye Teh, and Luke Ong. Faithformbench: Benchmarking faithfulness of mathematical chain-of-thought autoformalisation. arXiv preprint arXiv:2608.10916, 2026.

Xinyu Guan, Li Lyna Zhang, Yifei Liu, Ning Shang, Youran Sun, Yi Zhu, Fan Yang, and Mao Yang. rStar-math: Small LLMs can master math reasoning with self-evolved deep thinking. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu (eds.), Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 20640–20661. PMLR, 13–19 Jul 2025. URL https://proceedings.mlr.press/v267/guan25f. html.

Yichen Huang and Lin F. Yang. Winning gold at imo 2025 with a model-agnostic verification-andrefinement pipeline, 2025. URL https://arxiv.org/abs/2507.15855.

Eric Jiang, Xiao Liang, Yikai Zhang, Yingjia Wan, Mengting Li, Haikang Deng, Alexander K. Taylor, Justin Baker, Rushil Raghavan, Junyi Zhang, Ying Nian Wu, Andrea L. Bertozzi, Kai-Wei Chang, Raghu Meka, Matthew Sottile, Nanyun Peng, Amit Sahai, Terence Tao, and Wei Wang. From solvers to research: Large language model-driven formal mathematics at the research frontier, 2026. URL https://arxiv.org/abs/2607.07779.

Sayash Kapoor, Benedikt Stroebl, Zachary S Siegel, Nitya Nadgir, and Arvind Narayanan. AI agents that matter. Transactions on Machine Learning Research, 2025. ISSN 2835-8856. URL https://openreview.net/forum?id=Zy4uFzMviZ.

Tyson Klingner, Drew Bladek, Escher Crawford, Bohao Chen, Ariel Fu, Kaira Nair, Jarod Alper, Giovanni Inchiostro, and Vasily Ilin. Evaluation of LLMs for mathematical formalization in Lean, 2026. URL https://arxiv.org/abs/2606.05632.

Joshua Ong Jun Leang, Aryo Pradipta Gema, and Shay B Cohen. CoMAT: Chain of Mathematically Annotated Thought Improves Mathematical Reasoning. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng (eds.), Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 20245–20274, Suzhou, China, November 2025a. Association for Computational Linguistics. ISBN 979-8-89176-332- 6. doi: 10.18653/v1/2025.emnlp-main.1024. URL https://aclanthology.org/2025. emnlp-main.1024/.

Joshua Ong Jun Leang, Giwon Hong, Wenda Li, and Shay B Cohen. Theorem Prover as a Judge for Synthetic Data Generation. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 29941–29977, Vienna, Austria, July 2025b. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/ 2025.acl-long.1448. URL https://aclanthology.org/2025.acl-long.1448/.

Joshua Ong Jun Leang, Zheng Zhao, Aryo Pradipta Gema, Sohee Yang, Wai-Chung Kwan, Xuanli He, Wenda Li, Pasquale Minervini, Eleonora Giunchiglia, and Shay B Cohen. PiCSAR: Probabilistic confidence selection and ranking for reasoning chains. In Maria Liakata, Viviane P. Moreira, Jiajun Zhang, and David Jurgens (eds.), Findings of the Association for Computational Linguistics: ACL 2026, pp. 31511–31544, San Diego, California, United States, July 2026a. Association for Computational Linguistics. ISBN 979-8-89176-395-1. doi: 10.18653/v1/ 2026.findings-acl.1577. URL https://aclanthology.org/2026.findings-acl. 1577/.

Joshua Ong Jun Leang, Zheng Zhao, Mihaela Cat˘ alina Stoian, Qiyuan Xu, Haonan Li, Wenda˘ Li, Shay B. Cohen, and Eleonora Giunchiglia. Pythagoras-prover: Advancing efficient formal proving via augmented lean formalisation, 2026b. URL https://arxiv.org/abs/2606. 12594.

Zhuochun Li, Yong Zhang, Ming Li, Yuelyu Ji, Yiming Zeng, Ning Cheng, Yun Zhu, Yanmeng Wang, Shaojun Wang, Jing Xiao, and Daqing He. Rethinking LLM-as-a-judge: Representationas-a-judge with small language models via semantic capacity asymmetry. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview. net/forum?id=VAISvCsrvG.

Yong Lin, Shange Tang, Bohan Lyu, Ziran Yang, Jui-Hui Chung, Haoyu Zhao, Lai Jiang, Yihan Geng, Jiawei Ge, Jingruo Sun, Jiayun Wu, Jiri Gesi, Ximing Lu, David Acuna, Kaiyu Yang, Hongzhou Lin, Yejin Choi, Danqi Chen, Sanjeev Arora, and Chi Jin. Goedel-Prover-V2: Scaling Formal Theorem Proving with Scaffolded Data Synthesis and Self-Correction. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview. net/forum?id=j4C0nALrgK.

Nick Littlestone. Learning quickly when irrelevant attributes abound: A new linear-threshold algorithm. In 28th Annual Symposium on Foundations of Computer Science (sfcs 1987), pp. 68–77, 1987. doi: 10.1109/SFCS.1987.37.

Chengwu Liu, Ye Yuan, Yichun Yin, Yan Xu, Xin Xu, Zaoyu Chen, Yasheng Wang, Lifeng Shang, Qun Liu, and Ming Zhang. Safe: Enhancing mathematical reasoning in large language models via retrospective step-aware formal verification. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 12171–12186, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176- 251-0. doi: 10.18653/v1/2025.acl-long.594. URL https://aclanthology.org/2025. acl-long.594/.

Chris Yuhao Liu, Liang Zeng, Jiacai Liu, Rui Yan, Jujie He, Chaojie Wang, Shuicheng Yan, Yang Liu, and Yahui Zhou. Skywork-reward: Bag of tricks for reward modeling in LLMs, 2024. URL https://arxiv.org/abs/2410.18451.

Junqi Liu, Zihao Zhou, Zekai Zhu, Marco Dos Santos, Weikun He, jiawei liu, Yunzhou Xie, Junqiao Zhao, Qiufeng Wang, Lihong Zhi, Jia LI, and Wenda Li. Numina-lean-agent: An open and general agentic reasoning system for formal mathematics. In Forty-third International Conference on Machine Learning, 2026. URL https://openreview.net/forum?id=0bTEd4LpQr.

Qing Lyu, Shreya Havaldar, Adam Stein, Li Zhang, Delip Rao, Eric Wong, Marianna Apidianaki, and Chris Callison-Burch. Faithful Chain-of-Thought Reasoning. In Jong C. Park, Yuki Arase, Baotian Hu, Wei Lu, Derry Wijaya, Ayu Purwarianti, and Adila Alfa Krisnadhi (eds.), Proceedings of the 13th International Joint Conference on Natural Language Processing and the 3rd Conference of the Asia-Pacific Chapter of the Association for Computational

Linguistics (Volume 1: Long Papers), pp. 305–329, Nusa Dua, Bali, November 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.ijcnlp-main.20. URL https: //aclanthology.org/2023.ijcnlp-main.20/.

David Ma, Kaijing Ma, Shawn Guo, Yunfeng Shi, Enduo Zhao, Jiajun Shi, Zhaoxiang Zhang, Gavin Cheung, Jiaheng Liu, and Zili Wang. OProver: A unified framework for agentic formal theorem proving, 2026. URL https://arxiv.org/abs/2605.17283.

Iman Mirzadeh, Keivan Alizadeh, Hooman Shahrokhi, Oncel Tuzel, Samy Bengio, and Mehrdad Farajtabar. GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025. URL https: //openreview.net/forum?id=AjXkRZIvjB.

Leonardo de Moura and Sebastian Ullrich. The lean 4 theorem prover and programming language. In Automated Deduction – CADE 28: 28th International Conference on Automated Deduction, Virtual Event, July 12–15, 2021, Proceedings, pp. 625–635, Berlin, Heidelberg, 2021. Springer-Verlag. ISBN 978-3-030-79875-8. doi: 10.1007/978-3-030-79876-5 37. URL https://doi. org/10.1007/978-3-030-79876-5\_37.

Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candes, and Tatsunori Hashimoto. s1: Simple test-\` time scaling. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng (eds.), Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 20275–20321, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8-89176-332-6. doi: 10.18653/v1/2025.emnlp-main.1025. URL https: //aclanthology.org/2025.emnlp-main.1025/.

OpenAI. OpenAI GPT-5 System Card, 2026. URL https://arxiv.org/abs/2601.03267.

Azim Ospanov, Zijin Feng, Jiacheng Sun, Haoli Bai, SHEN XIN, and Farzan Farnia. HERMES: Towards efficient and verifiable mathematical reasoning in LLMs. In Forty-third International Conference on Machine Learning, 2026. URL https://openreview.net/forum?id= w7BZcUc0fJ.

Lawrence C. Paulson. Isabelle - A Generic Theorem Prover (with a contribution by T. Nipkow), volume 828 of Lecture Notes in Computer Science. Springer, 1994. ISBN 3-540-58244-4. doi: 10.1007/BFB0030541. URL https://doi.org/10.1007/BFb0030541.

Xin Quan, Marco Valentino, Louise A. Dennis, and Andre Freitas. Faithful and robust LLM-driven theorem proving for NLI explanations. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Proceedings ofthe 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 17734–17755, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/ 2025.acl-long.867. URL https://aclanthology.org/2025.acl-long.867/.

Qwen Team. Qwen3.6-27B: Flagship-Level Coding in a 27B Dense Model, April 2026. URL https://qwen.ai/blog?id=qwen3.6-27b.

Subham Sekhar Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T Chiu, Alexander Rush, and Volodymyr Kuleshov. Simple and Effective Masked Diffusion Language Models. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (eds.), Advances in Neural Information Processing Systems, volume 37, pp. 130136–130184. Curran Associates, Inc., 2024. doi: 10.52202/ 079017-4135. URL https://proceedings.neurips.cc/paper\_files/paper/ 2024/file/eb0b13cc515724ab8015bc978fdde0ad-Paper-Conference.pdf.

Varvara Sazonova, Dmitri Shmelkin, Stanislav Kikot, and Vasily Motolygin. Pipeline for verifying LLM-generated mathematical solutions, 2026. URL https://arxiv.org/abs/2602. 20770.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models, 2024. URL https://arxiv.org/abs/2402. 03300.

Charlie Victor Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling LLM test-time compute optimally can be more effective than scaling parameters for reasoning. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/ forum?id=4FWAwZtd2n.

Kimi Team. Kimi K3: open frontier intelligence, 2026. URL https://arxiv.org/abs/ 2607.24653.

George Tsoukalas, Jasper Lee, John Jennings, Jimmy Xin, Michelle Ding, Michael Jennings, Amitayush Thakur, and Swarat Chaudhuri. PutnamBench: Evaluating Neural Theorem-Provers on the Putnam Mathematical Competition. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (eds.), Advances in Neural Information Processing Systems, volume 37, pp. 11545–11569. Curran Associates, Inc., 2024. doi: 10.52202/079017-0368. URL https://proceedings.neurips.cc/paper\_files/paper/2024/file/ 1582eaf9e0cf349e1e5a6ee453100aa1-Paper-Datasets\_and\_Benchmarks\_ Track.pdf.

George Tsoukalas, Anton Kovsharov, Sergey Shirobokov, Anja Surina, Moritz Firsching, Gergely Berczi, Francisco JR Ruiz, Arun Suggala, Adam Zsolt Wagner, Eric Wieser, et al. Advancing´ mathematics research with ai-driven formal proof search. arXiv preprint arXiv:2605.22763, 2026.

Sumanth Varambally, Thomas Voice, Yanchao Sun, Zhifeng Chen, Rose Yu, and Ke Ye. Hilbert: Recursively building formal proofs with informal reasoning. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum? id=GN8OdkTo3B.

Jianing Wang, Jianfei Zhang, Qi Guo, Linsen Guo, Rumei Li, Chao Zhang, Chong Peng, Cunguang Wang, Dengchang Zhao, Jiarong Shi, Jingang Wang, et al. Longcat-flash-prover: Advancing native formal reasoning via agentic tool-integrated reinforcement learning, 2026. URL https: //arxiv.org/abs/2603.21065.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed Chi, Quoc V Le, and Denny Zhou. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (eds.), Advances in Neural Information Processing Systems, volume 35, pp. 24824–24837. Curran Associates, Inc., 2022. URL https://proceedings.neurips.cc/paper\_files/paper/2022/ file/9d5609613524ecf4f15af0f7b31abca4-Paper-Conference.pdf.

Huajian Xin, Daya Guo, Zhihong Shao, Zhizhou Ren, Qihao Zhu, Bo Liu, Chong Ruan, Wenda Li, and Xiaodan Liang. DeepSeek-Prover: Advancing Theorem Proving in LLMs through Large-Scale Synthetic Data, 2024. URL https://arxiv.org/abs/2405.14333.

Qiyuan Xu, Joshua Ong Jun Leang, Renxi Wang, Wenda Li, Haonan Li, Luke Ong, and Conrad Watt. AoA: Theorem proving agent over abstract syntax tree of redesigned language, 2026. URL https://arxiv.org/abs/2607.16372.

Jiarui Yao, Ruida Wang, and Tong Zhang. FANS: Formal answer selection for LLM natural language math reasoning using Lean4. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng (eds.), Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pp. 3181–3200, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8-89176-332-6. doi: 10.18653/v1/2025.emnlp-main.158. URL https://aclanthology.org/2025.emnlp-main.158/.

Erez Yosef, Oron Anschel, Shunit Haviv Hakimi, Asaf Gendler, Adam Botach, Nimrod Berman, and Igor Kviatkovsky. Rethinking math reasoning evaluation: A robust LLM-as-a-judge framework beyond symbolic rigidity, 2026. URL https://arxiv.org/abs/2604.22597.

Lan Zhang, Marco Valentino, and Andre Freitas. Autoformalization in the wild: Assessing LLMs on real-world mathematical definitions. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng (eds.), Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pp. 1720–1738, Suzhou, China, November 2025a. Association for Computational Linguistics. ISBN 979-8-89176-332-6. doi: 10.18653/v1/2025.emnlp-main. 90. URL https://aclanthology.org/2025.emnlp-main.90/.

Qiyuan Zhang, Fuyuan Lyu, Zexu Sun, Lei Wang, Weixu Zhang, Wenyue Hua, Haolun Wu, Zhihan Guo, Yufei Wang, Niklas Muennighoff, Irwin King, Xue Liu, and Chen Ma. A survey on testtime scaling in large language models: What, how, where, and how well?, 2025b. URL https: //arxiv.org/abs/2503.24235.

Zhenru Zhang, Chujie Zheng, Yangzhen Wu, Beichen Zhang, Runji Lin, Bowen Yu, Dayiheng Liu, Jingren Zhou, and Junyang Lin. The lessons of developing process reward models in mathematical reasoning. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Findings of the Association for Computational Linguistics: ACL 2025, pp. 10495–10516, Vienna, Austria, July 2025c. Association for Computational Linguistics. ISBN 979-8-89176-256- 5. doi: 10.18653/v1/2025.findings-acl.547. URL https://aclanthology.org/2025. findings-acl.547/.

Kunhao Zheng, Jesse Michael Han, and Stanislas Polu. MiniF2F: a cross-system benchmark for formal Olympiad-level mathematics. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=9ZPegFuFTFv.

Jin Peng Zhou, Charles Staats, Wenda Li, Christian Szegedy, Kilian Q. Weinberger, and Yuhuai Wu. Don’t trust: Verify – grounding LLM quantitative reasoning with autoformalization. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. URL https://openreview.net/forum?id= V5tdi14ple.

## A ALGORITHM

## A.1 NOTATION SUMMARY

Table 4 summarises the notation used throughout the description and analysis of MAGENTA.

Table 4: Summary of the principal notation used in MAGENTA.
<table><tr><td>Symbol</td><td>Domain or type</td><td>Description</td></tr><tr><td colspan="3">Problems and generated artefacts</td></tr><tr><td> $\mathcal { Q } , q$ </td><td> $q \in \mathcal { Q }$ </td><td>Natural-language mathematics problems, and an in- dividual problem.</td></tr><tr><td> $\mathbf { \mathcal { A } } , a _ { t }$ </td><td> $a _ { t } \in \mathcal A$ </td><td>Candidate answers, and the answer proposed by the reasoner at round t.</td></tr><tr><td> $a ^ { \star } , \hat { a }$ </td><td> $\in { \mathcal { A } }$ </td><td>Reference answer, used only for evaluation and never shown to the pipeline; and the answer finally re-</td></tr><tr><td> ${ \mathcal { C } } , c _ { t }$ </td><td> $c _ { t } \in \mathcal { C }$ </td><td>turned. Informal reasoning chains, and the chain generated at round t.</td></tr><tr><td> $s , s$ </td><td> $s \in S$ </td><td>Lean 4 theorem statements, and a generated state-</td></tr><tr><td> ${ \mathcal { P } } , \pi$ </td><td> $\pi \in { \mathcal { P } }$ </td><td>ment. Lean 4 proof scripts, and a candidate proof.</td></tr><tr><td> ${ \mathcal { E } } _ { \mathrm { s } }$ </td><td>Diagnostic space</td><td>Statement-elaboration diagnostics returned by  $V _ { \mathrm { s } } .$ </td></tr><tr><td> $\mathcal { E } _ { \mathrm { p } } , e$ </td><td> $e \in \mathcal { E } _ { \mathrm { p } }$ </td><td>Proof-verification diagnostics returned by  $V _ { \mathrm { p } } ,$  and the diagnostic from a failed attempt.</td></tr><tr><td> $\mathcal { F } _ { \mathrm { m a t h } } , \varphi _ { t }$ </td><td> $\varphi _ { t } \in \mathcal { F } _ { \operatorname* { m a t h } }$ </td><td>Mathematical-correction feedback built from a failed attempt,  $( c _ { t } , a _ { t } , s , \pi , e )$  . Constructed only when  $J _ { \mathrm { e } }$ </td></tr><tr><td>∅</td><td>Empty signal</td><td>returns MATH. Absence of feedback:  $\varphi = \varnothing$  on the first reasoning round,  $\epsilon = \mathcal { D }$  on the first proof attempt.</td></tr><tr><td colspan="3">Pipeline components</td></tr><tr><td>R</td><td> $\mathcal { Q } \times \mathcal { F } _ { \mathrm { m a t h } } \longrightarrow \mathcal { C } \times \mathcal { A }$ </td><td>Reasoner: produces an informal chain and candidate</td></tr><tr><td> $F$ </td><td> ${ \mathcal { Q } } \times { \mathcal { A } }  S$ </td><td>answer. Formaliser: renders the problem and candidate an-</td></tr><tr><td> $P$ </td><td> $\mathcal { Q } \times \mathcal { C } \times \mathcal { S } \times \mathcal { E } _ { \mathrm { p } } \to \mathcal { P }$ </td><td>swer as a Lean 4 statement. Prover: generates or repairs a Lean 4 proof script.</td></tr><tr><td> $V _ { \mathrm { s } }$ </td><td> $\mathcal { S } \to \{ \top \} \cup \mathcal { E } _ { \mathrm { s } }$ </td><td>Statement checker: elaborates s in Lean (well- formedness only).</td></tr><tr><td> $V _ { \mathrm { p } }$ </td><td> $S \times { \mathcal { P } } \to \{ { \top } \} \cup { \mathcal { E } } _ { \mathrm { p } }$ </td><td>Proof checker: Lean kernel plus SafeVerify on the unchanged statement.</td></tr><tr><td> $J _ { \mathrm { s } } , b$ </td><td> $b \in \{ 0 , 1 \}$ </td><td>Statement judge and its verdict;  $b = 1$  accepts the formalisation.</td></tr><tr><td> $J _ { \mathrm { e } } , \ell$ </td><td> $\ell \in \left\{ \mathrm { M A T H } , \mathrm { S Y N T A X } \right\}$ </td><td>Error judge and its label, routing a failed attempt to re-derivation or to local Lean repair.</td></tr><tr><td>T</td><td>Success verdict</td><td>A check passed: elaboration for  $V _ { \mathrm { s } } ,$  kernel and SafeVerify acceptance for  $V _ { \mathrm { p } } .$ </td></tr><tr><td colspan="3">Budgets and evaluation</td></tr><tr><td> $T , M , K$ </td><td> $\in \mathbb { N }$ </td><td>Maximum reasoner attempts, formalisation samples per candidate answer, and proof attempts per ac-</td></tr><tr><td>D</td><td>Set of problems</td><td>cepted statement. The evaluation benchmark.</td></tr><tr><td> $\operatorname { V e r } ( q )$ </td><td> $\in \{ 0 , 1 \}$ </td><td>Indicator that q terminated with an adjudicated state- ment and a kernel-accepted proof.</td></tr><tr><td> $u , \beta$ </td><td> $u : { \mathcal { D } } \to \mathbb { R } _ { \geq 0 }$ </td><td>A per-problem cost measure and a budget on it, used in Equation 2.</td></tr></table>

## A.2 FULL DETAILS

Algorithm 1 specifies the complete inference procedure using the notation of §2.1. The budgets $T ,$ M and K bound reasoner attempts, formal-statement samples for a fixed answer, and proof attempts for an accepted statement, respectively. The procedure returns the verified reasoning chain, answer, statement and proof $( c , a , s , \pi )$ , or UNCERTIFIED if an applicable budget is exhausted. Per problem it issues at most $T$ reasoner calls, TM formaliser and statement-judge calls, and TK prover, verifier and error-judge calls, so adjudication is cheap relative to proof search whenever $M \ll K$ , which is the regime we operate in and the reason adjudication precedes proving.

## Algorithm 1: MAGENTA inference.

Require: problem q; components $R , F , P , V { \mathrm { { s } } } , V { \mathrm { { p } } } , J { \mathrm { { s } } } , J { \mathrm { { e } } } ;$ budgets T, M, K   
Ensure: verified output $( c , a , s , \pi )$ , or UNCERTIFIED   
1: $\varphi \gets \emptyset$ ▷ reasonerfeedback state   
2: for $t = 1 , \dots , T$ do   
3: $( c _ { t } , a _ { t } ) \sim R ( q , \varphi )$ ▷ Phase 1: natural-language reasoning   
4: $z _ { s } \gets 0$ ▷ no accepted statement yet   
5: for $m = 1 , \ldots , M$ do ▷ Steps 1–2: formalise and adjudicate   
6: $s _ { m } \sim F ( q , a _ { t } )$   
7: $u _ { m } \gets V _ { \mathrm { s } } ( s _ { m } )$   
8: $\mathbf { i f } u _ { m } \neq \top$ then continue ▷ resample an ill-formed statement   
9: $b _ { m } \sim J _ { \mathrm { s } } ( q , a _ { t } , s _ { m } )$   
10: $\mathbf { i f } b _ { m } = 1$ then $s \gets s _ { m } ; z _ { s } \gets 1 ;$ break   
11: $\mathbf { i f } z _ { s } = 0$ then return UNCERTIFIED   
12: $\epsilon  \emptyset$   
13: z<sub>math</sub> ← 0   
14: for $k = 1 , \ldots , K$ do ▷ Step 3: prove and verify   
15: π ∼ $P ( q , c _ { t } , s , \epsilon )$   
16: $e  V _ { \mathrm { p } } ( s , \pi )$   
17: i $\mathbf { f } e = \top$ then return $\left( c _ { t } , a _ { t } , s , \pi \right)$ ▷ Step 5: verified output   
18: $\ell \sim J _ { \mathrm { e } } ( q , c _ { t } , a _ { t } , s , \pi , e )$ ▷ Step 4: error attribution   
19: if ℓ = MATH then   
20: $\varphi \gets ( c _ { t } , a _ { t } , s , \pi , e ) ; z _ { \mathrm { m a t h } } \gets 1 ;$ break ▷ constructfeedback in $\mathcal { F } _ { \mathrm { m a t h } }$   
21: $\epsilon  e$ ▷ SYNTAX: repair the proof in place   
22: if $z _ { \mathrm { m a t h } } = 0$ then return UNCERTIFIED   
23: return UNCERTIFIED ▷ reasoner budget exhausted

Algorithms 2–4 decompose the same procedure into its controller, statement-search and proof-search routines. Statement search resamples $\dot { \boldsymbol { F } }$ whenever $V _ { \mathrm { s } }$ rejects an ill-formed statement or $J _ { \mathrm { s } }$ rejects an unfaithful one; neither failure changes $( c , a )$ or returns control to R. Once a statement is accepted it remains fixed throughout proof search. A SYNTAX label returns the verifier diagnostic to $P$ for local proof repair, whereas only a MATH label constructs feedback in $\mathcal { F } _ { \mathrm { m a t h } }$ and invokes R again.

## Algorithm 2: End-to-end controller for MAGENTA.

Require: problem $q ;$ components $R , F , P , V { \mathrm { { s } } } , V { \mathrm { { p } } } , J { \mathrm { { s } } } , J { \mathrm { { e } } } ;$ budgets $T , M , K$   
Ensure: verified output $( c , a , s , \pi )$ , or UNCERTIFIED   
1: $\varphi \gets \emptyset$   
2: for $t = 1 , \dots , T$ do ▷ mathematical-revision loop   
3: $( c _ { t } , a _ { t } ) \sim R ( q , \varphi )$   
4: $\mathrm { \Phi } ( z _ { s } , s ) \gets \mathrm { S t a t e m e n t S e a r c h } ( q , a _ { t } ; F , V _ { \mathrm { s } } , J _ { \mathrm { s } } , M )$   
5: if $z _ { s } =$ REJECTED then return UNCERTIFIED   
6: $( z _ { p } , \pi , e ) \gets \mathrm { P r o o f S e a r c h } ( q , c _ { t } , a _ { t } , s ; P , V _ { \mathrm { p } } , J _ { \mathrm { e } } , K )$   
7: $\mathbf { i f } \ z _ { p } =$ CERTIFIED then return $\left( c _ { t } , a _ { t } , s , \pi \right)$   
8: $\mathbf { i f } z _ { p } = \mathbf { M A T H - E R R O R }$ then   
9: $\varphi  ( c _ { t } , a _ { t } , s , \pi , e ) ;$ continue ▷ constructfeedback in $\mathcal { F } _ { \mathrm { m a t h } }$   
10: if $z _ { p } =$ REPAIR-EXHAUSTED then return UNCERTIFIED   
11: return UNCERTIFIED ▷ reasoner budget exhausted

## Algorithm 3: Faithful formal-statement search.

Require: problem $q ;$ candidate answer a; formaliser $F ;$ statement verifier $V _ { \mathrm { s } } ;$ statement judge $J _ { \mathrm { s } } ;$ budget M   
Ensure: (ACCEPTED, s) or (REJECTED, ∅)   
1: for $m = 1 , \ldots , M$ do   
2: $s _ { m } \sim F ( q , a )$ ▷ generate a closed answer-carrying Lean signature   
3: $u _ { m } \gets V _ { \mathrm { s } } ( s _ { m } )$ ▷ elaborate the statement in Lean   
4: $\mathrm { ~ i f ~ } u _ { m } \neq \intercal$ then continue ▷ resamplefrom F   
5: $b _ { m } \sim J _ { \mathrm { s } } ( q , a , s _ { m } )$   
6: ▷ check the problem structure and the proposed answer   
7: i $\mathbf { f } b _ { m } = 1$ then return (ACCEPTED, s )   
8: return (REJECTED, ∅)

## Algorithm 4: Syntax-guided proof repair and error attribution.

Require: problem $q ;$ chain $c ;$ answer $^ { a ; }$ immutable statement s; prover $P ;$ proof verifier $V _ { \mathrm { p } } ;$ error judge $J _ { \mathrm { e } } ;$   
budget K   
Ensure: status in {CERTIFIED, MATH-ERROR, REPAIR-EXHAUSTED}, proof, diagnostic   
1: $\epsilon  \varnothing ;$ π<sub>last</sub> $ \emptyset ; e _ { \mathrm { l a s t } }  \emptyset$   
2: for $k = 1 , \ldots , K$ do   
3: $\pi _ { k } \sim P ( q , c , s , \epsilon )$ ▷ preserve q, c, s; repair only the Lean body   
4: $\pi _ { \mathrm { l a s t } }  \pi _ { k }$   
5: $e _ { k } \gets V _ { \mathrm { p } } ( s , \pi _ { k } )$ ▷ Lean and SafeVerify checking   
6: if $e _ { k } = \top$ then return (CERTIFIED, π<sub>k</sub>, ∅)   
7: $e _ { \mathrm { l a s t } }  e _ { k }$   
8: $\ell _ { k } \sim J _ { \mathrm { e } } ( q , c , a , s , \pi _ { k } , e _ { k } )$   
9: if ℓ = MATH then   
10: return (MATH-ERROR, $\pi _ { k } , e _ { k } )$ ▷ route to the reasoner   
11: $\epsilon  e _ { k }$ ▷ SYNTAX: condition the next local repair   
12: return (REPAIR-EXHAUSTED, π<sub>last</sub>, e<sub>last</sub>)

For IMO 2026, a final completeness judge is applied only after Algorithm 4 returns CERTIFIED. It compares $( q , c , a )$ with $( s , \pi )$ and returns incomplete proof when the formal certificate omits or misaligns a substantive part of the informal solution. This is an additional acceptance filter; it does not replace Lean or alter the feedback routing above.

## B IMPLEMENTATION DETAILS FOR MAGENTA

## B.1 PROMPT TEMPLATES

The templates below are reproduced verbatim from the experimental harness.

K2 informal-reasoner prompt   
You are an expert mathematical problem solver. Solve the problem carefully and show enough   
reasoning to make the solution understandable. End with the final answer in the exact,→   
form \boxed{answer}. Do not put anything after the boxed answer.,→   
{original\_problem}

## K2 mathematical self-correction prompt

Solve the original mathematics problem again and produce a complete corrected mathematical   
,→ solution.   
A downstream formal-verification audit classified the previous mathematical reasoning as   
,→ defective.   
Use the latest Lean code and exact verifier diagnostic below only as evidence locating the   
,→ mathematical defect.   
Do not focus on Lean syntax or tactics. Re-derive the mathematics, correct the answer if   
,→ necessary, and provide a self-contained solution.   
End with one clear final answer in \boxed{...}.   
Closed-loop K3 regeneration cycle: {cycle}   
Original problem:   
{original\_problem}   
Previous K3 mathematical solution:   
{previous\_k3\_mathematical\_solution}   
Immutable Lean target generated from that solution:   
{immutable\_lean\_target}   
Latest Lean proof attempt:   
{latest\_lean\_proof\_attempt}   
Exact Lean/SafeVerify diagnostic:   
{lean\_or\_safeverify\_diagnostic}

## Goedel formaliser prompt

Please autoformalize the following natural language problem statement in Lean 4.   
The natural language statement is:   
{informal\_statement\_content}   
Think before you provide the lean statement.

## DeepSeek statement-alignment judge prompt

```csv
Judge whether the candidate Lean 4 theorem statement faithfully formalizes the supplied
,→ question together with the generator's proposed final answer.
This is a statement-alignment judgment only. Do not write a proof and do not repair or
,→ rewrite the statement.
Do not independently solve the problem, recompute the answer, correct the generator's
mathematics, or prefer your own conclusion. A mathematically wrong proposed answer,→
must still be judged right when the Lean statement faithfully encodes that answer and,→
the original task structure.,→
```

The original question controls domains, hypotheses, quantifiers, constraints, and the   
,→ requested quantity.   
The proposed answer controls the concrete claimed result. Do not silently replace it with   
,→ a different answer.   
Return verdict right only when the candidate preserves both sources without missing   
assumptions, extra assumptions, weakened conclusions, strengthened conclusions,,→   
hard-coded loopholes, or a changed answer.,→   
Return verdict wrong otherwise and precisely describe the mismatch.   
Return only one strict JSON object with exactly these string fields: verdict, rationale,   
mismatch\_details. verdict must be right or wrong. mismatch\_details must be empty when,→   
right and nonempty when wrong.,→   
Exact appended question passed to the formalizer:   
{answer\_appended\_question}   
Original natural-language question:   
{original\_question}   
Generator's proposed final answer:   
{proposed\_answer}   
Candidate Lean 4 theorem statement:   
{candidate\_lean4\_theorem\_statement}

## Leanstral proof-generation prompt

Produce a complete Lean 4 proof of the supplied theorem signature.   
1. This is a formal translation and verification task, not an independent mathematics   
,→ attempt.   
2. Copy the theorem signature verbatim and do not weaken, strengthen, rename, or otherwise   
,→ alter it.   
3. Formalize the supplied K2 solution's reasoning and claimed conclusion without   
correcting, replacing, or bypassing its mathematics. Add only Lean-specific,→   
bookkeeping needed to express the same steps.,→   
4. Do not substitute an independent proof strategy merely because it proves the statement.   
Do not use \`native\_decide\` or another exhaustive computation unless the supplied,→   
solution itself explicitly used that strategy.,→   
5. If the supplied mathematics is false or insufficient, do not invent assumptions or   
,→ silently repair it.   
6. Use the available Lean compiler and lean-lsp-mcp tools to inspect errors and   
,→ iteratively repair only the Lean implementation.   
7. Edit \`Main.lean\` in the current working directory. It must import Mathlib and must not   
use \`sorry\`, \`admit\`, axioms, unsafe declarations, fabricated assumptions, or the,→   
conclusion as a hypothesis.,→   
8. Before editing the final code, provide a detailed Lean-oriented proof plan covering the   
,→ main steps, intermediate lemmas, and tactics.   
Original informal problem (authoritative for task structure):   
{original\_question}   
Answer-appended question supplied to K2 (the answer is parsed from the K2 solution, not   
,→ from ground truth):   
{answer\_appended\_question}   
Supplied K2 final mathematical solution (content after the private thinking marker;   
,→ authoritative for the reasoning and claimed conclusion):   
{k2\_final\_mathematical\_solution}   
Lean 4 theorem signature that must be preserved verbatim:   
{immutable\_lean4\_theorem\_signature}   
Current \`Main.lean\`:   
\`\`\`lean4   
{current\_Main\_lean}

DeepSeek error-classification prompt   
Classify the root cause of this failed Lean 4 proof attempt.   
This is a routing-only classification task. Do not repair the Lean code, do not suggest   
,→ tactics, and do not provide feedback for the prover.   
Return math\_error only when the supplied K3 mathematical reasoning or claimed answer is   
itself false or insufficient, and the verifier evidence materially exposes that,→   
mathematical defect.,→   
Return code\_error for syntax, elaboration, typing, tactic, missing-lemma, namespace,   
coercion, timeout, forbidden-token, changed-target, SafeVerify protocol, or,→   
incomplete-proof failures that could be fixed while preserving the same mathematics.,→   
A difficult or currently unsolved proof is code\_error. SafeVerify rejection caused by   
,→ changing the theorem is code\_error.   
When evidence is ambiguous, return code\_error. Do not independently replace the proposed   
,→ answer.   
Return only one strict JSON object with exactly these string fields: classification,   
,→ rationale, evidence.   
classification must be code\_error or math\_error. Keep rationale and evidence concise.   
Original problem:   
{original\_question}   
K3 proposed answer:   
{k3\_proposed\_answer}   
K3 mathematical solution:   
{k3\_mathematical\_solution}   
Immutable Lean theorem target:   
{immutable\_lean\_theorem\_target}   
Latest Lean candidate:   
{latest\_lean\_candidate}   
Verifier stage:   
{verifier\_stage}   
Exact Lean/SafeVerify diagnostic:   
{exact\_lean\_or\_safeverify\_diagnostic}

## B.2 HYPERPARAMETER SETTINGS

1. Budgets. For all the experiments, we use T = 32 reasoner attempts, M = 512 formalstatement samples per candidate answer and K = 4096 proof attempts per accepted statement. For computational efficiency, the ablations of §4.3, run the pipeline in batched mode with a batch size 32 .

2. Reasoner. We use temperature 0.6, top p = 0.95, a maximum output length of 64,000 tokens (128,000 for IMO 2026), and thinking=auto. The random seed is 42 for the main evaluation. The reasoner is served with a 524,288-token context window. During self-correction, incomplete responses are resampled if they reach the token limit without emitting the post-thinking marker (this counts toward the budget T). In practice, this never occurred in our experiments.

3. Goedel formaliser. We use temperature 0.9, top k = 20, top p = 0.95, a maximum output length of 16,384 tokens, and seed 42. The model is served with a 40,960-token context window. Each extracted theorem statement is checked in Lean with a temporary sorry proof.

4. DeepSeek statement-alignment judge. We use temperature 1.0, top p = 1.0, a maximum output length of 32,000 tokens, and reasoning effort max. Only statements that have already passed Lean elaboration are submitted to this judge. The model is served with a 65,536-token context window, tensor parallelism 8, pipeline parallelism 1, GPUmemory utilisation 0.90, FP8 KV cache, block size 256, prefix caching and expert parallelism. DSPARK speculative decoding is enabled with seven speculative tokens and greedy draft sampling.

5. DeepSeek error-attribution judge. The error-attribution judge uses the same checkpoint and hyperparameters as the statement-alignment judge but is issued as a separate call, and returns either code error or math error.

6. Leanstral proof agent. We use mistralai/Leanstral-1.5-119B-A6B through Mistral Vibe 2.18.4, with decoding temperature 1.0, thinking level high and reasoning effort high. The served context length is 200,000 tokens and automatic context compaction is triggered at 168,000 tokens. Compaction preserves the trajectory, and the cumulative token counter is not reset after compaction. The server permits at most 80,000 newly generated tokens per request, while the experimental wrapper applies a stricter 32,000-token per-call output limit. The cumulative output-token budget is 4,000,000 tokens per proof trajectory.

7. Lean tools and final certification. The proof environment uses Lean v4.29.1. A result is accepted only when the theorem statement is unchanged, Lean returns exit code zero, SafeVerify returns exit code zero, neither verifier times out, and the proof contains no forbidden constructs such as sorry, admit, new axioms, unsafe or native decide.

8. Codex comparator and fallback. The CODEX formaliser and proof backend use gpt-5.6-sol with reasoning effort ultra.

## C REFINEMENT ROUNDS ACROSS MODELS AND DATASETS

Table 5: Reported average and maximum rounds required across models and datasets, counting the initial attempt as round 1. Average counts are rounded.
<table><tr><td rowspan="2">Dataset</td><td colspan="2">K2-Horizon-7B</td><td colspan="2">K2-Horizon-375B</td></tr><tr><td>Average</td><td>Maximum</td><td>Average</td><td>Maximum</td></tr><tr><td>AIME2025</td><td>1</td><td>3</td><td>1</td><td>1</td></tr><tr><td>AIME2026</td><td>2</td><td>5</td><td>1</td><td>2</td></tr><tr><td>HMMT2026</td><td>3</td><td>10</td><td>1</td><td>2</td></tr></table>

Table 5 shows that refinement effort varies across both models and datasets. K2-Horizon-7B requires approximately one, two and three rounds on average for AIME2025, AIME2026 and HMMT2026, respectively. Its maximum increases from three to five and ten rounds, showing that some instances require substantially more refinement than the dataset average. K2-Horizon-375B requires approximately one round on average across all three datasets, with at most one additional round on AIME2026 and HMMT2026.

The shorter trajectories of K2-Horizon-375B are consistent with its reported tendency to produce correct answers on the first attempt. For K2-Horizon-7B, the increasing refinement effort suggests that AIME2026 and HMMT2026 pose greater difficulty within this workflow. Lower initial accuracy leaves more instances requiring subsequent correction, providing a possible explanation for the additional rounds. However, the relationship between initial accuracy and round count should be assessed using the corresponding first-attempt results; round counts alone measure refinement effort rather than establish a model-independent ranking of dataset difficulty.

## D AIME2026 FORMAL AND INFORMAL PROOF EXAMPLE

We reproduce a complete audited trajectory for an AIME 2026 problem (see also Figure 1), in which the initial informal solution is wrong, the error is attributed to the mathematics rather than to the Lean implementation, and the pipeline recovers. Boxes 1–5 are the failed cycle and boxes 6–9 the corrected one.

Input question   
Find the number of functions π mapping the set A = {1, 2, 3, 4, 5, 6} onto A such that for every   
a ∈ A,   
π(π(π(π(π(π(a)))))) = a.

1. Informal reasoner — initial generation   
The condition   
π(π(π(π(π(π(a)))))) = a for all a ∈ A   
means that the seventh iterate of π is the identity map:   
$\pi ^ { 7 } = \operatorname { i d } _ { A }$   
Hence the order of π must divide 7. Since 7 is prime, the only possible orders are 1 and 7.   
A permutation of a set of size 6 can have order at most 6 (the maximum occurs for a 6-cycle).   
Thus a permutation of order 7 cannot exist on a set of six elements.   
Consequently the only function satisfying $\pi ^ { 7 } = \operatorname { i d } _ { A }$ is the identity permutation.   
1

```verilog
4. Leanstral — initial proof generation
import Mathlib
open Equiv
theorem problem_1 :
Nat.card {π : Equiv.Perm (Fin 6) | ∀ a : Fin 6, (π ˆ 7) a = a} = 1 := by
have h_set_eq : {π : Equiv.Perm (Fin 6) | ∀ a : Fin 6, (π ˆ 7) a = a} = {1} := by
ext π
constructor
· intro h
have h_all : ∀ a : Fin 6, (π ˆ 7) a = a := h
have h_pow : (π ˆ 7) = 1 := by
apply Equiv.ext
intro a
simpa using h_all a
have h_order_dvd_7 : orderOf π | 7 := orderOf_dvd_of_pow_eq_one h_pow
have h_prime_7 : Nat.Prime 7 := Nat.prime_seven
have h_order_cases : orderOf π = 1 ∨ orderOf π = 7 :=
h_prime_7.eq_one_or_self_of_dvd (orderOf π) h_order_dvd_7
rcases h_order_cases with (h_order | h_order)
· have h_one : π = 1 := (orderOf_eq_one_iff.mp h_order)
simp [h_one]
· have h_card : Fintype.card (Equiv.Perm (Fin 6)) = 720 := by
rw [Fintype.card_perm, Fintype.card_fin]
```

norm\_num   
have h\_order\_dvd\_card : orderOf π | Fintype.card (Equiv.Perm (Fin 6)) :=   
,→ orderOf\_dvd\_card   
rw [h\_card] at h\_order\_dvd\_card   
rw [h\_order] at h\_order\_dvd\_card   
have h\_7\_not\_dvd\_720 : ¬ (7 : N) | 720 := by   
have h\_mod : 720 %   
intro h\_dvd   
have h\_mod\_zero : 720 %   
rw [h\_mod] at h\_mod\_zero   
linarith   
exact absurd h\_order\_dvd\_card h\_7\_not\_dvd\_720   
· intro h   
have h\_one : π = 1 := by simpa using h   
rw [h\_one]   
intro a   
simp   
rw [h\_set\_eq]   
simp

math error   
⇓   
6. Informal reasoner — corrected generation   
The condition π(π(π(π(π(π(a)))))) = a for all a ∈ A means that the sixth iterate of π is   
the identity, i.e., π<sup>6</sup> = id . Since π maps A onto A and A is finite, π is a bijection, hence a   
permutation of A. The order of π divides 6, so we must count the permutations in S whose   
order divides 6.   
The order of a permutation is the least common multiple of its cycle lengths. Thus the order   
divides 6 if and only if all cycle lengths belong to {1, 2, 3, 6} (cycles of length 4 or 5 would give   
orders 4 or 5, which do not divide 6). We count the permutations by their cycle types:   
• One 6-cycle: (6 − 1)! = 120   
• Two 3-cycles: <sup>6!</sup><sub>2· !</sub> = 720<sub>18</sub> = 40   
• One 3-cycle, one 2-cycle, one 1-cycle: <sup>6!</sup><sub>· ·</sub> = 720 = 120   
• One 3-cycle, three 1-cycles: <sup>6!</sup>3·1<sup>3</sup>·3! 720<sub>18</sub> = 40   
• Three 2-cycles: <sup>6!</sup>2<sup>3</sup>·3! = 720 = 15   
• Two 2-cycles, two 1-cycles: 6!<sub>2 2</sub> = 720<sub>16</sub> = 45   
• One 2-cycle, four 1-cycles: 6!<sub>4</sub> = 720<sub>48</sub> = 15   
• Six 1-cycles (identity): 1   
Summing these gives 120 + 40 + 120 + 40 + 15 + 45 + 15 + 1 = 396.   
396

7. Goedel formaliser — corrected generation   
theorem problem\_1 :   
Set.ncard {π : Equiv.Perm (Fin 6) | ∀ a, (πˆ[6]) a = a} = 396   
⇓   
8. DeepSeek statement judge   
yes   
⇓

9. Leanstral — corrected proof generation   
import Mathlib   
set\_option maxRecDepth 10000000   
set\_option maxHeartbeats 2000000   
theorem problem\_1 :   
Set.ncard {π : Equiv.Perm (Fin 6) | ∀ a, (πˆ[6]) a = a} = 396 := by   
have h\_eq : {π : Equiv.Perm (Fin 6) | ∀ a, (πˆ[6]) a = a} = {π | π ˆ 6 = 1} := by   
ext π   
constructor   
· intro h   
ext a   
have ha := h a   
have ha1 : (π ˆ 6) a = a := by   
simpa [Equiv.Perm.iterate\_eq\_pow π 6] using ha   
simpa [ha1]   
· intro h a   
have h' := congrArg (λ f : Equiv.Perm (Fin 6) => f a) h   
simpa [Equiv.Perm.iterate\_eq\_pow π 6] using h'   
rw [h\_eq]   
have h\_fin : Set.Finite {π : Equiv.Perm (Fin 6) | π ˆ 6 = 1} := by   
exact Set.finite\_coe\_iff.mp inferInstance   
have h\_card : Set.ncard {π : Equiv.Perm (Fin 6) | π ˆ 6 = 1} =   
Finset.card ((Finset.univ : Finset (Equiv.Perm (Fin 6))).filter (λ π => π ˆ 6 = 1))   
,→ := by   
rw [Set.ncard\_eq\_toFinset\_card \_ h\_fin]   
congr   
ext π   
simp   
rw [h\_card]   
decide

## E ERROR ANALYSIS

## E.1 FORMAL STATEMENT ERROR ANALYSIS

Table 6: Formal-statement generation and adjudication errors (%) across all 30 AIME 2026 problems. Generated percentages are normalised over all Goedel statements. Conditional percentages are normalised over the statements reaching the relevant stage: all pre-judge rejections for the error categories and all Lean-passing statements for the DeepSeek outcomes.

<table><tr><td>Outcome or first failure category</td><td>Generated</td><td>Conditional</td></tr><tr><td>All-30 Goedel generation outcomes</td><td></td><td></td></tr><tr><td>Lean elaborated</td><td>66.55</td><td>一</td></tr><tr><td>Lean compiler error</td><td>33.08</td><td>一</td></tr><tr><td>Statement extraction failure</td><td>0.22</td><td>一</td></tr><tr><td>Input context overflow</td><td>0.16</td><td></td></tr><tr><td>First cause among rejected all-30 statements</td><td></td><td></td></tr><tr><td>Type, coercion, or projection mismatch</td><td>17.47</td><td>52.23</td></tr><tr><td>Typeclass or instance synthesis</td><td>7.85</td><td>23.46</td></tr><tr><td>Parser or notation syntax</td><td>4.78</td><td>14.28</td></tr><tr><td>Unknown name or API misuse</td><td>2.73</td><td>8.17</td></tr><tr><td>Statement extraction failure</td><td>0.22</td><td>0.66</td></tr><tr><td>Input context overflow</td><td>0.16</td><td>0.47</td></tr><tr><td>Metavariables or underspecified binders</td><td>0.13</td><td>0.38</td></tr><tr><td>Pattern, binder, or match elaboration</td><td>0.10</td><td>0.31</td></tr><tr><td>Tactic failure or open goal</td><td>0.01</td><td>0.02</td></tr><tr><td>Resource timeout</td><td>0.01</td><td>0.02</td></tr><tr><td>All-30 compiler-to-DeepSeek outcomes</td><td></td><td></td></tr><tr><td>Rejected before judge: Lean error</td><td>33.08</td><td></td></tr><tr><td>Rejected before judge: extraction error</td><td>0.22</td><td>一</td></tr><tr><td>Rejected before judge: context overflow</td><td>0.16</td><td></td></tr><tr><td>DeepSeek semantic rejection</td><td>44.64</td><td>67.08</td></tr><tr><td>DeepSeek semantic acceptance</td><td>21.91</td><td>32.92</td></tr><tr><td></td><td></td><td></td></tr></table>

We generate a full grid of candidate answers and Goedel formalisations for each of the 30 AIME 2026 problems. The audit in Table 6 shows that formal-statement failure is primarily an elaboration problem rather than a parsing problem. Type, coercion and projection mismatches account for more than half of rejected statements, and typeclass synthesis for almost a quarter; together these two categories explain over three quarters of pre-proof rejection. Syntax errors form a much smaller share, and extraction, context, tactic and timeout failures are negligible. The main obstacle is therefore expressing the intended mathematics with Lean-compatible types and interfaces, not producing superficially valid Lean syntax.

The DeepSeek audit exposes a separate semantic bottleneck. Even after the compiler and extractor filter malformed statements, the judge rejects 67.08% of the statements it sees and accepts 32.92%; relative to all generated statements these correspond to 44.64% rejected and 21.91% accepted. Compilation is therefore a weak proxy for faithfulness: semantic adjudication removes substantially more compilable-but-misaligned statements than Lean removes malformed ones. Among falsely certified statements in the no- $J _ { \mathrm { s } }$ ablation of §4.3, the dominant failure mode is a weakened or answer-baked surrogate goal: three of five certificates prove such surrogates, one drops an essential part of the problem and one mistranslates the underlying geometry; none succeeds through a contradictory context.

## E.2 PROOF ERROR ANALYSIS

LEANSTRAL exposes proof errors at two different levels: 54.73% of its de-duplicated internal Lean/LSP results contain an error diagnostic or tool failure, spanning tactic failures, API misuse, protocol failures, syntax, typing, and resource limits (Table 7). By the time a trajectory reaches the external verifier, however, failures are concentrated in unsolved proof obligations and SafeVerify policy violations. This shift shows that Leanstral’s agentic inner loop absorbs many local elaboration and implementation errors before they reach the outer correction loop.

Table 7: Proof-error distributions (%) across all 30 AIME 2026 problems. Leanstral internal is normalized over error-bearing Lean/LSP results; the external columns are normalized over failed outer rounds. One category is assigned per result or round, with policy violations taking priority; “–” denotes no assigned error.
<table><tr><td>Primary error category</td><td>Leanstral int.</td><td>Leanstral ext.</td><td>Codex ext.</td></tr><tr><td>Tactic failure or open goals</td><td>42.70</td><td>63.78</td><td>30.28</td></tr><tr><td>Forbidden token or policy</td><td></td><td>26.77</td><td></td></tr><tr><td>Unknown name or API misuse</td><td>15.19</td><td></td><td>15.31</td></tr><tr><td>Tool, protocol, or runtime failure</td><td>13.68</td><td></td><td>2.16</td></tr><tr><td>Parser or syntax error</td><td>8.18</td><td>0.79</td><td>0.67</td></tr><tr><td>Type or coercion mismatch</td><td>6.70</td><td></td><td>45.92</td></tr><tr><td>Resource limit or timeout</td><td>5.72</td><td>8.66</td><td>2.66</td></tr><tr><td>Other Lean diagnostic</td><td>5.23</td><td></td><td></td></tr><tr><td>Typeclass or instance synthesis</td><td>2.60</td><td></td><td>1.83</td></tr><tr><td>Metavariables or invalid declaration</td><td></td><td></td><td>0.83</td></tr><tr><td>Noncomputable definition</td><td></td><td></td><td>0.33</td></tr></table>

Compared with LEANSTRAL, CODEX exposes a different external failure profile: type and coercion mismatches dominate, followed by tactic failures and API misuse, whereas LEANSTRAL’s external failures are dominated by proof search and policy enforcement. Parser errors are negligible for both, showing that syntactic Lean generation is not the main bottleneck. The important distinction is therefore where debugging occurs: LEANSTRAL resolves a broader range of implementation errors internally, while Codex returns more of them to the pipeline. External rounds are consequently not interchangeable units of proof effort, and error attribution should account for both the prover backend and the level at which the error is observed.

## E.3 REASONER ERROR ANALYSIS

Table 8: Root causes of the 14 mathematical errors attributed to K2-HORIZON-7B by the error judge while solving the AIME 2026 problems.
<table><tr><td>Cause</td><td>Percent (%)</td></tr><tr><td>Input interpretation or transcription</td><td>28.6</td></tr><tr><td>Unsupported logical or proof step</td><td>28.6</td></tr><tr><td>Arithmetic, algebra, or cardinality</td><td>21.4</td></tr><tr><td>Escalation after repeated Lean repair could not preserve the contract</td><td>14.3</td></tr><tr><td>Invalid construction or counterexample</td><td>7.1</td></tr></table>

As Table 8 shows, the dominant failures arise less from routine calculation than from preserving the problem statement and justifying key proof steps. Representative examples include changing 17017 to 107017, solving the problem for $N = \mathrm { { \bar { 1 0 0 0 } } }$ instead of $N \ : = \ : 1 0 0 0 0$ , reading 39 as 3 or as $3 ^ { 9 }$ , interpreting six iterates as seven (Appendix D), and relying on unsupported geometric or combinatorial reductions. These cases show why compiler feedback alone is insufficient: error attribution must sometimes return control to the reasoner to reconstruct the mathematics rather than continue repairing its Lean implementation.

## F DETAILS OF IMO 2026

## F.1 EXPERIMENTAL SETUP

For IMO 2026 we retain the same pipeline and add a final judge after formal verification to assess whether the formal certificate covers the complete informal solution. The judge compares the original problem and informal reasoning with the generated Lean statement and proof. If it detects a substantive mismatch, or an argument that is present informally but absent from the formal certificate, it returns incomplete proof. We count a solution as complete only when it passes both Lean/SafeVerify and this final formal–informal alignment check.

## F.2 EVALUATION

We evaluate our IMO 2026 solutions under two complementary protocols. First, we use CODEX as an automated judge, supplying it with several publicly available reference solutions, including those from the Art of Problem Solving forums and from Kimi K3, and asking it to score each generated solution against the official marking scheme. Second, for a more robust assessment, we ask a former IMO medallist with access to national IMO score predictions to mark the same solutions independently.

## F.3 VERIFICATION-GUIDED TEST-TIME SCALING

Figure 6 shows that MAGENTA adaptively allocates more inference-time computation to more difficult problems. Problems 3 and 6 consume the largest token budgets and require more self-correction, consistent with their higher level of difficulty. Rather than relying on a single fixed generation, MAGENTA uses formal-verification feedback to guide subsequent attempts, allowing the reasoner to revise unsuccessful arguments and devote additional computation where it is most needed. This reflects the principle of test-time scaling (Sahoo et al., 2024), with verification providing a targeted signal for how that additional computation should be used. Through this verification-guided scaling process, K2-HORIZON-7B ultimately solves all six problems on IMO 2026.

![](images/dfebe0145f292f6e13b0c0c5b557076b67a4c048a84327cc0115cabed7c1dc76.jpg)  
Figure 6: Total reasoner-token usage of K2-HORIZON-7B with MAGENTA across the six IMO 2026 problems.

## F.4 GENERATED SOLUTIONS

## F.4.1 IMO PROBLEM 1

Question   
There are 2026 integers greater than 1 written on a blackboard, not necessarily distinct. In one   
move, Confucius chooses two integers $m > 1$ and $n > 1$ , written in different positions, and   
replaces them by   
$\operatorname* { g c d } ( m , n ) \qquad { \mathrm { a n d } } \qquad { \frac { \operatorname { l c m } ( m , n ) } { \operatorname* { g c d } ( m , n ) } } .$   
He continues making moves for as long as a move is possible.   
(a) Prove that, regardless of his choices, the process terminates after finitely many moves   
with exactly one integer $M > 1$ on the board.   
(b) Prove that the value of M is independent of all choices made during the process.

K2-Horizon-7B generation   
Let the initial integers be $a _ { 1 } , a _ { 2 } , \ldots , a _ { 2 0 2 6 }$ with $a _ { i } ~ > ~ 1$ for all $i .$ We regard the board as a   
function $f : \{ 1 , 2 , \cdot \cdot \cdot , 2 0 2 6 \}  \mathbb { N } _ { > 1 }$ . A move consists of choosing two distinct positions $i , j$   
with $m = f ( i ) > 1$ and $n = { \overset { \cdot } { f } } ( j ) > 1$ , and replacing them by   
$g = \operatorname* { g c d } ( m , n ) , \qquad h = \frac { \operatorname { l c m } ( m , n ) } { \operatorname* { g c d } ( m , n ) } .$   
All other positions stay unchanged. The process continues as long as such a move is possible.   
(A) TERMINATION AND THE FINAL BOARD   
For any board b (a function from $\{ 1 , \ldots , 2 0 2 6 \}$ to positive integers) define   
$N ( b ) = \# \{ i \mid b ( i ) > 1 \} , \qquad P ( b ) = \prod _ { i = 1 } ^ { 2 0 2 6 } b ( i ) .$   
Because every $b ( i )$ is a positive integer, $P ( b )$ is a positive integer.   
Lemma 1. After a move $b \to c$ we have $\dot { N ( c ) } \leq \dot { N } ( b )$ , and if $\mathbf { \bar { N } } ( c ) = N ( b )$ then $P ( c ) < P ( b )$   
Proof. Let the move act on positions $i , j$ with $m = \dot { b } ( \dot { i } ) > 1 , n \stackrel { . } { = } \dot { b } ( \dot { j } ) > \dot { 1 } .$ . Set $g = \operatorname* { g c d } ( m , n )$   
and $\begin{array} { r } { \dot { h } = \operatorname { l c m } ( \boldsymbol { m } , \boldsymbol { n } ) / g ; } \end{array}$ then $c ( i ) = g , \ : c ( j ) = h$ and ${ \dot { c } } ( k ) = b ( k )$ for $k \neq i , j .$   
For $\mathfrak { a } \notin \{ i , j \}$ the contribution to $\bar { N }$ and $\dot { P }$ is unchanged, so we only compare the two changed   
positions.   
Case $l \colon g = 1$ . Then $h = \operatorname { l c m } ( m , n ) =$ mn (since $\operatorname* { g c d } ( m , n ) = 1 )$ . Because $m , n > 1$ , we   
have $h > 1$ . Thus we replace two numbers $> 1$ by one number $> 1$ (namely h) and one number   
1 (namely $g )$ . Hence $N { \dot { ( } } c ) = N ( b ) - 1$ . Moreover   
$P ( c ) = P ( b ) \cdot { \frac { g h } { m n } } = P ( b ) \cdot { \frac { 1 \cdot m n } { m n } } = P ( b ) .$   
Case 2: $g > 1 .$ . Write $m = g m ^ { \prime } , n = g n ^ { \prime }$ with $\operatorname* { g c d } ( m ^ { \prime } , n ^ { \prime } ) = 1 .$ Then   
$h = { \frac { \operatorname { l c m } ( m , n ) } { g } } = { \frac { g m ^ { \prime } n ^ { \prime } } { g } } = m ^ { \prime } n ^ { \prime } .$   
Subcase 2a: $h = 1 .$ . Then $m ^ { \prime } = n ^ { \prime } = 1 , \mathrm { s o } m = n = g$ . We replace two equal numbers $> 1$ by   
$g > 1$ and 1; therefore $N ( c ) = N ( b ) - 1 .$ Also   
$P ( c ) = P ( b ) \cdot { \frac { g \cdot 1 } { g \cdot g } } = { \frac { P ( b ) } { g } } < P ( b ) \quad ( { \mathrm { s i n c e ~ } } g > 1 ) .$   
Subcase $2 b \colon h > 1$ . Then we replace two numbers $> 1$ by two numbers $> 1 ( g > 1 , h > 1 ) ;$   
hence $N ( c ) = N ( b )$ . Moreover

$$
P ( c ) = P ( b ) \cdot { \frac { g h } { m n } } = P ( b ) \cdot { \frac { g \left( m ^ { \prime } n ^ { \prime } \right) } { g m ^ { \prime } g n ^ { \prime } } } = { \frac { P ( b ) } { g } } < P ( b ) \quad ( { \mathrm { s i n c e ~ } } g > 1 ) .
$$

In every situation $N ( c ) \leq N ( b )$ , and when $N ( c ) = N ( b )$ we are in subcase 2b, which gives $P ( c ) < P ( b )$ □

The lexicographic order on $\mathbb { N } \times \mathbb { N }$ is well-founded (because N is well-ordered). Lemma 1 shows that the pair $( { \bar { N } } , P )$ strictly decreases lexicographically at each move. Hence the process cannot continue indefinitely; it must terminate after finitely many moves.

Let b be a terminal board, i.e. no move is possible. Then there are no two distinct positions with values $> 1 \lambda$ consequently at most one position contains a number $> 1$ . We claim that exactly one such position exists.

Suppose, for contradiction, that all $b ( i ) = 1$ . Then $N ( b ) = 0 .$ . The initial board has $N = 2 0 2 6$ Each move can decrease N by at most one (Lemma 1). To reach $N = 0$ from $N = 2 0 2 6$ we would need at least 2026 moves that each reduce N by one. However, a move that reduces N from 1 to 0 is impossible because a move requires two numbers $> 1$ . More precisely, if at some stage $N = 1$ , no move can be performed, so the process would have stopped earlier. Thus the process can never reach a state with $N = 0$ . Therefore the final board must have $N = 1$ Hence there is a unique index i with $b ( i ) > 1$ , and for all $j \neq \cdot$ i we have $b ( j ) = 1$ . Set $M = b ( i )$ then $M > 1$ . This proves part (a).

## (B) INDEPENDENCE OF THE FINAL VALUE M

For a prime p and a positive integer x let $v _ { p } ( x )$ be the exponent of p in the prime factorisation of x (with $v _ { p } ( 1 ) = 0 )$ . For a board b define

$$
G _ { p } ( b ) = \operatorname* { g c d } \{ v _ { p } ( b ( i ) ) \mid i = 1 , \ldots , 2 0 2 6 \} .
$$

(If the set consists only of zeros we take the gcd to be 0.)

Lemma 2. $G _ { p } ( b )$ is invariant under moves.

Proof. Consider a move on positions $i , j$ with $m = b ( i ) , n = b ( j )$ . Put $a = v _ { p } ( m ) , b = v _ { p } ( n )$ Then

$$
v _ { p } ( \operatorname* { g c d } ( m , n ) ) = \operatorname* { m i n } ( a , b ) , \qquad v _ { p } ( \operatorname { l c m } ( m , n ) ) = \operatorname* { m a x } ( a , b ) .
$$

Since $h = \operatorname { l c m } ( m , n ) / \operatorname* { g c d } ( m , n ) ,$

$$
v _ { p } ( h ) = v _ { p } ( \operatorname { l c m } ( m , n ) ) - v _ { p } ( \operatorname* { g c d } ( m , n ) ) = \operatorname* { m a x } ( a , b ) - \operatorname* { m i n } ( a , b ) = | a - b | .
$$

Thus the multiset of p-adic valuations changes from $\{ a , b \}$ to $\{ \operatorname* { m i n } ( a , b ) , | a - b | \}$ . Let R be the multiset of valuations of the other 2024 positions. Then

$$
\begin{array} { r l } & { \operatorname* { g c d } \bigl ( \{ a , b \} \cup R \bigr ) = \operatorname* { g c d } \bigl ( \operatorname* { g c d } ( a , b ) , \operatorname* { g c d } ( R ) \bigr ) , } \\ & { \operatorname* { g c d } \bigl ( \{ \operatorname* { m i n } ( a , b ) , | a - b | \} \cup R \bigr ) = \operatorname* { g c d } \bigl ( \operatorname* { g c d } ( \operatorname* { m i n } ( a , b ) , | a - b | ) , \operatorname* { g c d } ( R ) \bigr ) . } \end{array}
$$

Hence it suffices to prove

$$
\operatorname* { g c d } ( \operatorname* { m i n } ( a , b ) , | a - b | ) = \operatorname* { g c d } ( a , b ) .\tag{*}
$$

Assume without loss of generality that $a \geq b .$ Then min $\boldsymbol { \mathbf { \rho } } _ { \mathrm { l } } ( a , b ) = b$ and $| a - b | = a - b .$ . If d divides both b and $a - b ,$ , then d divides $b + ( a - b ) = a ;$ conversely, if d divides both b and $^ { a , }$ then d divides $a - b .$ . Thus the sets of common divisors of $\{ b , a - \dot { b } \}$ and $\{ a , b \}$ coincide, so their greatest common divisors are equal. Hence (∗) holds, and the gcd of the whole multiset is unchanged. □

Now look at the terminal board: it contains one number $M > 1$ and the remaining 2025 numbers equal to 1. The multiset of valuations is $\{ v _ { p } ( M ) \} \cup \{ 0 , 0 , \ldots , 0 \}$ (2025 zeros). The gcd of this set is $v _ { p } ( M )$ because gcd $. ( v _ { p } ( M ) , 0 ) = \dot { v } _ { p } \dot { ( M ) }$ . Therefore

$$
G _ { p } ( \mathrm { f i n a l } ) = v _ { p } ( M ) .
$$

By Lemma 2, $G _ { p } ( \mathrm { f i n a l } ) = G _ { p } ( \mathrm { i n i t i a l } )$ for every prime p. Consequently

$$
v _ { p } ( M ) = G _ { p } ( { \mathrm { i n i t i a l } } ) \qquad { \mathrm { f o r ~ a l l ~ p r i m e s ~ } } p .
$$

The prime factorisation of an integer is unique, so

$$
M = \prod _ { p { \mathrm { p r i m e } } } p ^ { G _ { p } ( { \mathrm { i n i t i a l } } ) } .
$$

Since the right-hand side depends only on the initial multiset, the value of M is independent of the choices made during the process. □

The process terminates with exactly one integer $M > 1$ on the board, and

$$
M = \prod _ { p } p ^ { g _ { p } } , \qquad g _ { p } = \operatorname* { g c d } \bigl ( v _ { p } ( a _ { 1 } ) , \dots , v _ { p } ( a _ { 2 0 2 6 } ) \bigr ) ,
$$

which is independent of the choices.

## F.4.2 IMO PROBLEM 2

## Question

Let ABC be a triangle, and let M and N be the midpoints of AB and $A C ,$ respectively. Let K and L be chosen strictly inside triangles $B M C$ and $\dot { B } { N C } .$ , respectively, such that K lies strictly inside ∠LBA, L lies strictly inside $\angle A C K$ , and

$$
\angle K B A = \angle A C L , \qquad \angle L B K = \angle L N C , \qquad \angle L C K = \angle B M K .
$$

If O is the circumcentre of triangle $A K L ,$ prove that $O M = O N .$

## K2-Horizon-7B generation

Place triangle ABC in the plane with A at the origin. Let $B = \left( b _ { 1 } , b _ { 2 } \right)$ and $C = \left( c _ { 1 } , c _ { 2 } \right)$ . Since △ABC is non-degenerate, the vectors B and C are linearly independent. Define

$$
D = \operatorname* { d e t } ( B , C ) = b _ { 1 } c _ { 2 } - b _ { 2 } c _ { 1 } \neq 0 ,
$$

$$
r = | B | ^ { 2 } = b _ { 1 } ^ { 2 } + b _ { 2 } ^ { 2 } ,
$$

$$
s = B \cdot C = b _ { 1 } c _ { 1 } + b _ { 2 } c _ { 2 } ,
$$

$$
t = | C | ^ { 2 } = c _ { 1 } ^ { 2 } + c _ { 2 } ^ { 2 } .
$$

For any points $P , Q$ we write $P - Q$ for the vector from Q to $P ,$ and for vectors u, v we denote by de ${ \mathrm { : } } ( u , v )$ the determinant (cross product) and by u·v the dot product.

## 1. COORDINATES OF K AND L

Because K lies inside $\triangle B M C$ there exist positive numbers $\alpha , \beta , \gamma$ with $\alpha + \beta + \gamma = 1$ such that

$$
K = \alpha B + \beta M + \gamma C .
$$

The midpoint M of AB is $\begin{array} { r } { M = \frac { 1 } { 2 } ( A + B ) = \frac { 1 } { 2 } B ; } \end{array}$ hence

$$
K = \Big ( \alpha + \frac { \beta } { 2 } \Big ) B + \gamma C .
$$

Set

$$
x = \alpha + \frac { \beta } { 2 } , \qquad y = \gamma .
$$

Then $K = x B + y C .$ . From $\alpha , \beta , \gamma > 0$ and $\alpha + \beta + \gamma = 1$ we obtain

$$
0 < x < 1 , \qquad 0 < y < 1 , \qquad x + y = \alpha + \frac { \beta } { 2 } + \gamma = 1 - \frac { \beta } { 2 } < 1 .
$$

Similarly, L lies inside $\triangle B N C .$ Its midpoint N is $\begin{array} { r } { N = \frac { 1 } { 2 } C ; } \end{array}$ thus there exist positive $\alpha ^ { \prime } , \beta ^ { \prime } , \gamma ^ { \prime }$ with $\alpha ^ { \prime } \overset { \cdot } { + } \beta ^ { \prime } + \gamma ^ { \prime } = 1$ such that

$$
L = \alpha ^ { \prime } B + \beta ^ { \prime } N + \gamma ^ { \prime } C = \alpha ^ { \prime } B + \Big ( \frac { \beta ^ { \prime } } { 2 } + \gamma ^ { \prime } \Big ) C .
$$

Set

$$
p = { \alpha } ^ { \prime } , \qquad q = \frac { \beta ^ { \prime } } { 2 } + { \gamma } ^ { \prime } .
$$

Then ${ \cal L } = p B + q C$ and

$$
0 < p < 1 , \qquad 0 < q < 1 , \qquad p + q = 1 - \frac { \beta ^ { \prime } } { 2 } < 1 .
$$

2. THE $\mathrm { C O N D I T I O N S } \ ^ {  } K$ INSIDE $\angle L B A ^ { , * } \ A \mathrm { N D } \ ^ { * } L$ INSIDE $\angle A C K ^ { \prime \prime }$

K inside $\angle L B A$ means that the ray BK lies between BA and BL. With $A = 0$ this is equivalent to the existence of positive numbers u, v such that

$$
K - B = u ( L - B ) + v ( A - B ) = u ( L - B ) - v B .\tag{*}
$$

Write $K - B = ( x - 1 ) B + y C { \mathrm { ~ a n d ~ } } L - B = ( p - 1 ) B + q C .$ Substituting into (∗) gives

$$
( x - 1 ) B + y C = u { \big [ } ( p - 1 ) B + q C { \big ] } - v B = { \big [ } u ( p - 1 ) - v { \big ] } B + u q C .
$$

Because B and C are independent, we equate coefficients:

$$
y = u q , \qquad x - 1 = u ( p - 1 ) - v .\tag{1}
$$

From the first equation $u = y / q > 0$ . Then

$$
v = u ( p - 1 ) - ( x - 1 ) = { \frac { y } { q } } ( p - 1 ) - ( x - 1 ) = { \frac { y ( p - 1 ) - q ( x - 1 ) } { q } } .
$$

Define

$$
d = q ( 1 - x ) - y ( 1 - p ) .
$$

A short calculation shows $y ( p - 1 ) - q ( x - 1 ) = d ;$ hence $v = d / q$ . Since $v > 0$ and $q > 0 ;$

$$
d > 0 .\tag{2}
$$

Similarly, L inside $\angle A C K$ means there exist positive $u ^ { \prime } , v ^ { \prime }$ such that

$$
L - C = u ^ { \prime } ( A - C ) + v ^ { \prime } ( K - C ) = - u ^ { \prime } C + v ^ { \prime } ( K - C ) .\tag{**}
$$

Write $L - C = p B + ( q - 1 ) C$ and $K - C = x B + ( y - 1 ) C$ . Substituting into (∗∗) gives

$$
p B + ( q - 1 ) C = v ^ { \prime } \bigl [ x B + ( y - 1 ) C \bigr ] - u ^ { \prime } C = ( v ^ { \prime } x ) B + \bigl [ v ^ { \prime } ( y - 1 ) - u ^ { \prime } \bigr ] C .
$$

Equating coefficients:

$$
p = v ^ { \prime } x , \qquad q - 1 = v ^ { \prime } ( y - 1 ) - u ^ { \prime } .\tag{3}
$$

From the first equation $v ^ { \prime } = p / x > 0$ . Then

$$
u ^ { \prime } = v ^ { \prime } ( y - 1 ) - ( q - 1 ) = { \frac { p } { x } } ( y - 1 ) - ( q - 1 ) = { \frac { p ( y - 1 ) - x ( q - 1 ) } { x } } .
$$

Define

$$
e = x ( 1 - q ) - p ( 1 - y ) .
$$

Again $p ( y - 1 ) - x ( q - 1 ) = e ;$ hence $u ^ { \prime } = e / x .$ . Since $u ^ { \prime } > 0$ and $x > 0 ,$

$$
e > 0 .\tag{4}
$$

## 3. A LEMMA ON ANGLE EQUALITIES

Lemma. Let $u , v , u ^ { \prime } , v ^ { \prime }$ be non-zero vectors in the plane. If the (undirected) angle between u and v equals the angle between $u ^ { \prime }$ and $v ^ { \prime }$ and det(u, v) det $( u ^ { \prime } , v ^ { \prime } ) > 0 .$ , then

$$
\operatorname* { d e t } ( u , v ) ( u ^ { \prime } \cdot v ^ { \prime } ) = \operatorname* { d e t } ( u ^ { \prime } , v ^ { \prime } ) ( u \cdot v ) .\tag{*}
$$

Proof. Equality of the undirected angles gives equality of the cosines:

$$
{ \frac { u \cdot v } { | u | | v | } } = { \frac { u ^ { \prime } \cdot v ^ { \prime } } { | u ^ { \prime } | | v ^ { \prime } | } } .
$$

The absolute values of the determinants satisfy

$$
\frac { | \operatorname* { d e t } ( u , v ) | } { | u | | v | } = \frac { | \operatorname* { d e t } ( u ^ { \prime } , v ^ { \prime } ) | } { | u ^ { \prime } | | v ^ { \prime } | } .
$$

Because the determinants have the same sign,

$$
\frac { \operatorname* { d e t } ( u , v ) } { | u | | v | } = \frac { \operatorname* { d e t } ( u ^ { \prime } , v ^ { \prime } ) } { | u ^ { \prime } | | v ^ { \prime } | } .
$$

Multiplying the two equalities yields ⋆.

## 4. TRANSLATING THE ANGLE EQUALITIES

We apply the lemma to each given angle equality. First we compute the relevant determinants and dot products.

$\angle K B A ^ { \prime } = \angle A C L . { \mathrm { ~ T a k e ~ } } u = K - B , v = A - B = - B , u ^ { \prime } = A - C = - C , v ^ { \prime } = L - C .$ Then

$$
\operatorname* { d e t } ( K - B , - B ) = y D , \qquad \operatorname* { d e t } ( - C , L - C ) = p D .
$$

Since $y > 0 , p > 0$ and $D \neq 0 ,$ the product is positive. The dot products are

$$
( K - B ) \cdot ( - B ) = ( 1 - x ) r - y s , \qquad ( - C ) \cdot ( L - C ) = t ( 1 - q ) - p s .
$$

Lemma ⋆ gives

$$
y D \big ( t ( 1 - q ) - p s \big ) = p D \big ( ( 1 - x ) r - y s \big ) .
$$

Cancelling D and simplifying we obtain

$$
r p ( 1 - x ) = t y ( 1 - q ) .\tag{I}
$$

$\angle L B K = \angle L N C .$ Take $u = L - B , v = K - B , u ^ { \prime } = L - N , v ^ { \prime } = C - N ,$ Then

$$
\operatorname* { d e t } ( L - B , K - B ) = d D , \qquad \operatorname* { d e t } ( L - N , C - N ) = \frac { p } { 2 } D .
$$

Since $d > 0 , p > 0$ the product is positive. The dot products are

$$
\begin{array} { c } { { ( L - B ) \cdot ( K - B ) = r ( 1 - p ) ( 1 - x ) } } \\ { { - s \left( ( 1 - p ) y + q ( 1 - x ) \right) + t q y , } } \end{array}
$$

$$
( L - N ) \cdot ( C - N ) = { \frac { s p + t ( q - { \frac { 1 } { 2 } } ) } { 2 } } .
$$

Lemma ⋆ yields

$$
\begin{array} { c } { d D \cdot \frac { s p + t ( q - \frac { 1 } { 2 } ) } { 2 } = \displaystyle \frac { p } { 2 } D \cdot \Big [ r ( 1 - p ) ( 1 - x ) } \\ { - s \big ( ( 1 - p ) y + q ( 1 - x ) \big ) + t q y \Big ] . } \end{array}
$$

Cancelling D and multiplying by 2 we get

$$
\begin{array} { c } { d \big ( s p + t ( q - \frac { 1 } { 2 } ) \big ) = p \Big [ r \big ( 1 - p \big ) ( 1 - x ) } \\ { - s \big ( ( 1 - p ) y + q ( 1 - x ) \big ) + t q y \Big ] . } \end{array}\tag{II}
$$

$$
\angle L C K = \angle B M K . { \mathrm { ~ T a k e ~ } } u = L - C , \ v = K - C , \ u ^ { \prime } = B - M , \ v ^ { \prime } = K - M . { \mathrm { ~ T h e n } }
$$

$$
\operatorname * { d e t } ( L - C , K - C ) = e D , \qquad \operatorname * { d e t } ( B - M , K - M ) = \frac { y } { 2 } D .
$$

Since $e > 0 , y > 0$ the product is positive. The dot products are

$$
( L - C ) \cdot ( K - C ) = r p x - s \big ( p ( 1 - y ) + x ( 1 - q ) \big )
$$

$$
( B - M ) { \cdot } ( K - M ) = \frac { r ( x - \frac { 1 } { 2 } ) + s y } { 2 } .
$$

Lemma ⋆ gives

$$
\begin{array} { c } { { e D \cdot \frac { r \left( x - \frac 1 2 \right) + s y } { 2 } = \displaystyle \frac { y } { 2 } D \cdot \left[ r p x - s \big ( p ( 1 - y ) + x ( 1 - q ) \big ) \right. } } \\ { { \left. \qquad + t ( 1 - q ) ( 1 - y ) \right] . } } \end{array}
$$

Cancelling D and multiplying by 2 we obtain

$$
\begin{array} { c } { { e \big ( r ( x - \frac 1 2 ) + s y \big ) = y \Big [ r p x - s \big ( p ( 1 - y ) + x ( 1 - q ) \big ) } } \\ { { + t ( 1 - q ) ( 1 - y ) \Big ] . } } \end{array}\tag{III}
$$

5. A CRUCIAL ALGEBRAIC IDENTITY

Set

$$
\begin{array} { r l } & { E _ { 1 } = r p ( 1 - x ) - t y ( 1 - q ) = 0 } \\ & { E _ { 2 } = p \big [ r ( 1 - p ) ( 1 - x ) - s \big ( ( 1 - p ) y + q ( 1 - x ) \big ) + t q y \big ] } \\ & { \qquad - d \big ( s p + t ( q - \frac 1 2 ) \big ) = 0 } \\ & { E _ { 3 } = y \big [ r p x - s \big ( p ( 1 - y ) + x ( 1 - q ) \big ) + t ( 1 - q ) ( 1 - y ) \big ] } \\ & { \qquad - e \big ( r ( x - \frac 1 2 ) + s y \big ) = 0 } \end{array}\tag{from (I)),}
$$

(from (II)),

Define

$$
c _ { 1 } = ( x + y ) ( p + 2 q - 1 ) - ( p + q ) ( 1 - x ) .\tag{from (III)).}
$$

Consider the linear combination

$$
F = c _ { 1 } E _ { 1 } + 2 ( x + y ) ( 1 - q ) E _ { 2 } - 2 ( p + q ) ( 1 - x ) E _ { 3 } .
$$

Since $E _ { 1 } , E _ { 2 } , E _ { 3 }$ are zero, we have $F = 0$ . Now we expand $F .$ Substituting the expressions for d and $e ,$

$$
d = q ( 1 - x ) - y ( 1 - p ) , \qquad e = x ( 1 - q ) - p ( 1 - y ) ,
$$

and expanding, we obtain after collecting terms:

$$
\begin{array} { r l } & { F = ( 1 - x ) ( 1 - q ) \Big [ r \big ( 2 ( p + q ) x ^ { 2 } - 2 ( x + y ) p ^ { 2 } - ( x q - y p ) \big ) } \\ & { \qquad + s \big ( 4 ( p + q ) x y - 4 ( x + y ) p q \big ) } \\ & { \qquad + t \big ( 2 ( p + q ) y ^ { 2 } - 2 ( x + y ) q ^ { 2 } + ( x q - y p ) \big ) \Big ] . } \end{array}
$$

Because $0 < x < 1$ and $0 < q < 1$ , we have $( 1 - x ) ( 1 - q ) \neq 0$ . Hence the bracket must vanish, which gives

$$
\begin{array} { l } { { 2 ( p + q ) ( r x ^ { 2 } + 2 s x y + t y ^ { 2 } ) } } \\ { { \qquad - 2 ( x + y ) ( r p ^ { 2 } + 2 s p q + t q ^ { 2 } ) = ( x q - y p ) ( r - t ) . } } \end{array}\tag{IV}
$$

6. THE CIRCUMCENTER O OF $\triangle A K L$

Since A is the origin, the condition $O A = O K = O L$ is equivalent to

$$
| O | ^ { 2 } = | O - K | ^ { 2 } = | O - L | ^ { 2 } .
$$

Expanding the squares gives

$$
2 { \cal O } { \cdot } K = | K | ^ { 2 } , \qquad 2 { \cal O } { \cdot } L = | L | ^ { 2 } .
$$

Using $K = x B + y C$ and ${ \cal L } = p B + q C ,$

$$
x \left( O \cdot B \right) + y \left( O \cdot C \right) = \frac { r x ^ { 2 } + 2 s x y + t y ^ { 2 } } { 2 } ,\tag{5}
$$

$$
p \left( O \cdot B \right) + q \left( O \cdot C \right) = \frac { r p ^ { 2 } + 2 s p q + t q ^ { 2 } } { 2 } .\tag{6}
$$

Let $u = O \cdot B$ and $v = O \cdot C$ . Then (5) and (6) are a linear system for $u , v$ We must prove $O M = O N$ . Because $\begin{array} { r } { M = \frac { 1 } { 2 } B } \end{array}$ and $\begin{array} { r } { N = \frac { \mathrm { i } } { 2 } C , } \end{array}$

$$
O M ^ { 2 } = | O - { \textstyle \frac { 1 } { 2 } } B | ^ { 2 } = | O | ^ { 2 } - u + \frac { r } { 4 } ,
$$

$$
O N ^ { 2 } = | O - \textstyle { \frac { 1 } { 2 } } C | ^ { 2 } = | O | ^ { 2 } - v + \frac { t } { 4 } .
$$

Thus $O M = O N$ is equivalent to $O M ^ { 2 } = O N ^ { 2 }$ , i.e.

$$
v - u = \frac { t - r } { 4 } .\tag{7}
$$

To obtain (7) from (5) and (6), eliminate u and v. Multiply (5) by $p + q$ and (6) by $x + y$ and subtract:

$$
\begin{array} { c } { { ( p + q ) ( x u + y v ) - ( x + y ) ( p u + q v ) } } \\ { { = \displaystyle \frac { 1 } { 2 } \Big [ ( p + q ) ( r x ^ { 2 } + 2 s x y + t y ^ { 2 } ) } } \\ { { - ( x + y ) ( r p ^ { 2 } + 2 s p q + t q ^ { 2 } ) \Big ] . } } \end{array}
$$

The left-hand side simplifies to

$$
( p y - x q ) ( v - u ) .
$$

Hence

$$
\begin{array} { c } { { ( p y - x q ) ( v - u ) = \displaystyle \frac { 1 } { 2 } \Big [ ( p + q ) ( r x ^ { 2 } + 2 s x y + t y ^ { 2 } ) } } \\ { { - ( x + y ) ( r p ^ { 2 } + 2 s p q + t q ^ { 2 } ) \Big ] . } } \end{array}\tag{8}
$$

Now use (IV). Dividing (IV) by 2 gives

$$
\begin{array} { l } { { ( p + q ) ( r x ^ { 2 } + 2 s x y + t y ^ { 2 } ) } } \\ { { \qquad - ( x + y ) ( r p ^ { 2 } + 2 s p q + t q ^ { 2 } ) = \frac { ( x q - y p ) ( r - t ) } { 2 } . } } \end{array}
$$

Substituting this into (8) yields

$$
( p y - x q ) ( v - u ) = \frac { ( x q - y p ) ( r - t ) } { 4 } .
$$

Since $p y - x q = - ( x q - y p )$ , we have

$$
- ( x q - y p ) ( v - u ) = \frac { ( x q - y p ) ( r - t ) } { 4 } .
$$

The triangle AKL is non-degenerate (its circumcenter O exists), therefore K and L are not collinear with A. In our coordinates this means

$$
\operatorname* { d e t } ( K , L ) = ( x q - y p ) D \neq 0 ,
$$

and because $D \neq 0$ we have xq − yp ̸= 0. Cancelling this non-zero factor gives

$$
- ( v - u ) = \frac { r - t } { 4 } \quad \Longrightarrow \quad v - u = \frac { t - r } { 4 } ,
$$

which is exactly (7). Hence $O M ^ { 2 } = O N ^ { 2 }$ , and since distances are non-negative, $O M = O N .$

## F.4.3 IMO PROBLEM 3

## Question

Let n be a positive integer. Liu Bang and Xiang Yu have a stick of length 1. First Liu Bang marks at most n points on the stick, and then Xiang Yu marks at most n further points; all marked points are distinct. The stick is cut at every marked point. The players then alternately claim unclaimed pieces, with Liu Bang moving first, and each player seeks to maximize the total length of the pieces he claims. Determine the largest number c that Liu Bang can guarantee, regardless of Xiang Yu’s play.

## K2-Horizon-7B generation

Let the stick be the interval [0, 1]. Liu marks at most n points, then Xiang marks at most n further points; all marked points are distinct. The stick is cut at every marked point, producing pieces. The players then alternately claim unclaimed pieces, Liu moving first, each maximizing total length.

Lemma 1 (Optimal play). Let the pieces be sorted in non-increasing order $y _ { 1 } \geq y _ { 2 } \geq \cdot \cdot \cdot \geq$ $y _ { m } > 0$ . Under optimal play Liu obtains the pieces at odd positions and Xiang the pieces at even positions. Consequently Liu’s total is $\scriptstyle { \frac { 1 + D } { 2 } }$ where

$$
D = \sum _ { i \mathrm { ~ o d d } } y _ { i } - \sum _ { i \mathrm { ~ e v e n } } y _ { i } .
$$

Liu wants to maximise D, Xiang to minimise it.

Proof. We prove by induction on m that the player whose turn it is always takes the largest remaining piece. For $m = 1$ this is trivial. Assume the statement holds for $m - 1$ pieces. For m pieces, suppose it is the current player’s turn. If the player takes $y _ { j } \left( j > 1 \right)$ , the remaining pieces are sorted descending with $y _ { j }$ removed. By induction the opponent will obtain the odd-indexed pieces in that sorted list. Let $\mathbf { \bar { \mathit { G } } } _ { \mathit { j } }$ be the opponent’s total if the current player takes $y _ { j }$ . For $j = 1$ $G _ { 1 } = y _ { 2 } + y _ { 4 } + \cdot \cdot \cdot$ . For $j > 1 ,$ , removing $y _ { j }$ shifts the parity of indices for elements after j; the opponent takes $y _ { i }$ for $i < j$ with i odd, and for $i > j$ with i even. Hence

$$
G _ { j } = \sum _ { i < j \atop i \mathrm { o d d } } y _ { i } \ + \ \sum _ { i > j \atop i \mathrm { e v e n } } y _ { i } .
$$

Then

$$
G _ { j } - G _ { 1 } = \Big ( \sum _ { \stackrel { i < j } { i \mathrm { o d d } } } y _ { i } + \sum _ { \stackrel { i > j } { i \mathrm { e v e n } } } y _ { i } \Big ) - \sum _ { \stackrel { i \mathrm { e v e n } } { i \mathrm { e v e n } } } y _ { i } .
$$

Write $\textstyle \sum _ { i { \mathrm { e v e n } } } y _ { i } = \sum _ { i < j \atop i { \mathrm { e v e n } } } y _ { i } + ( y _ { j } { \mathrm { ~ i f ~ } } j$ is even else $0 ) + \textstyle \sum _ { i \mathrm { ~ e v e n } } y _ { i }$ . Thus

$$
G _ { j } - G _ { 1 } = \sum _ { \stackrel { i < j } { i \mathrm { o d d } } } y _ { i } - \sum _ { \stackrel { i < j } { i \mathrm { e v e n } } } y _ { i } - ( y _ { j } \mathrm { i f } j \mathrm { i s } \mathrm { e v e n } \mathrm { e l s e } 0 ) = \sum _ { \stackrel { i < j } { i \mathrm { o d d } } } y _ { i } - \sum _ { \stackrel { i \leq j } { i \mathrm { e v e n } } } y _ { i } .
$$

Now consider the cases. $\bullet \operatorname { I f } j = 2 h$ is even, then $\begin{array} { r } { \sum _ { i < 2 h , i \mathrm { ~ o d d } } y _ { i } = y _ { 1 } + y _ { 3 } + \cdot \cdot \cdot + y _ { 2 h - 1 } = } \end{array}$ $\begin{array} { r } { \sum _ { r = 1 } ^ { h } y _ { 2 r - 1 } , \ \sum _ { i \leq 2 h , i \mathrm { \ e v e n } } y _ { i } \ = \ y _ { 2 } + y _ { 4 } + \cdot \cdot \cdot + y _ { 2 h } \ = \ \sum _ { r = 1 } ^ { h } y _ { 2 r } } \end{array}$ . Hence $G _ { j } \ : - \ : G _ { 1 } =$ $\begin{array} { r } { \sum _ { r = 1 } ^ { h } ( y _ { 2 r - 1 } - y _ { 2 r } ) ~ \ge ~ 0 . ~ \bullet ~ \operatorname { I f } ~ j ~ = ~ 2 h + 1 } \end{array}$ is odd, then $\textstyle \sum _ { i < 2 h + 1 , i \mathrm { o d d } } y _ { i } \ = \ y _ { 1 } + y _ { 3 } \ +$ $\begin{array} { r } { \cdot \cdot \cdot + y _ { 2 h - 1 } = \sum _ { r = 1 } ^ { h } y _ { 2 r - 1 } , \sum _ { i \leq 2 h + 1 , i \mathrm { ~ e v e n ~ } } y _ { i } = y _ { 2 } + y _ { 4 } + \cdot \cdot \cdot + y _ { 2 h } = \sum _ { r = 1 } ^ { h } y _ { 2 r } } \end{array}$ . Hence $\begin{array} { r } { G _ { j } - G _ { 1 } = \sum _ { r = 1 } ^ { h } ( y _ { 2 r - 1 } - y _ { 2 r } ) \ge 0 } \end{array}$ . In both cases $G _ { j } \ \geq \ G _ { 1 }$ , so taking $y _ { 1 }$ minimises the opponent’s total and maximises the current player’s total. □

Lemma 2 (Pairs-plus-one). Let S be a multiset of positive real numbers. Suppose $S$ consists of some number of equal pairs (each pair consists of two equal numbers) plus possibly one extra element $r > 0 . ( \mathrm { H } r = 0 $ we may ignore it.) Then when S is sorted in non-increasing order, the alternating sum

$$
D = \sum _ { \mathrm { o d d } ~ i } s _ { i } - \sum _ { \mathrm { e v e n } ~ i } s _ { i }
$$

## equals r.

Proof. Sort S descending. Group the elements by their values. Since all numbers except possibly r appear in pairs, every value block (except possibly the block containing r) has even size. The block containing r has odd size. The number of elements before this block is the sum of the sizes of the previous blocks, which are all even, so it is even. In the alternating sum, each block of even size contributes 0 (the alternating sum of an even number of equal terms cancels). The block containing r starts at an odd position (because the number of preceding elements is even), so its terms alternate signs starting with +. The sum of an odd number of equal terms with alternating signs starting with + is the value itself. Hence the contribution of this block is r. All other blocks contribute 0. Thus $D = r . \operatorname { I f } r = 0$ then $D = 0 .$ □

## 1. LOWER BOUND – LIU CAN GUARANTEE $D \geq \delta$

Set $\delta = \frac { 1 } { 2 ^ { n + 1 } - 1 }$ . Liu marks the points $( 2 ^ { k } - 1 ) \delta { \mathrm { f o r } } k = 1 , 2 , \ldots , n$ . The stick is divided into $n + 1$ intervals

$$
I _ { k } = \left[ ( 2 ^ { k - 1 } - 1 ) \delta , \thinspace ( 2 ^ { k } - 1 ) \delta \right] , \qquad k = 1 , \ldots , n + 1 ,
$$

$$
l _ { k } = 2 ^ { k - 1 } \delta
$$

$$
\delta ( 2 ^ { n + 1 } - 1 ) = 1
$$

Xiang marks at most n further points. The stick is cut at every marked point, producing pieces. Let m be the number of positive pieces. Since there are at most 2n marks, m $\leq 2 n + 1$ Sort the pieces in non-increasing order and pad with zeros to obtain a list of length $2 n + 2$ (if $m < 2 n + 1$ add zeros; if $m = 2 n + 1$ add one zero). Pair the list as $( a _ { 1 } , b _ { 1 } ) , \dotsc , ( a _ { n + 1 } , b _ { n + 1 } )$ with $a _ { i } \geq b _ { i } \geq 0 .$ . Associate each piece with its parent interval $I _ { k } ;$ zero pieces are associated with a dummy vertex D.

Construct a directed multigraph $G \colon$ vertices are $I _ { 1 } , \ldots , I _ { n + 1 }$ and D. For each pair $( a _ { i } , b _ { i } )$ draw a directed edge from the parent of $a _ { i }$ to the parent of $b _ { i }$ . Thus G has $V = n { \overline { { + } } } 2$ vertices and $E = n + 1$ edges. Every $I _ { k }$ contains at least one piece, and D is incident to at least one zero piece (since exactly $2 n + 2 - m \geq 1$ zeros were added), so every vertex has degree $\geq 1$

Consider the undirected version of G. For any connected component $C ,$ we have $E _ { C } \ge V _ { C } - 1$ (a connected graph has at least $V _ { C } - 1$ edges). Summing over all components, $E \geq V - c ,$ where c is the number of components. Since $\overset { \vartriangle } { \boldsymbol { E } } = \boldsymbol { V } - \boldsymbol { 1 }$ , we get $V - 1 \geq \bar { V } - c ,$ hence $c \geq 1$ . If every component satisfied $E _ { C } ^ { \ \bullet } \geq V _ { C }$ , then $E \geq V$ , contradicting $E = V - 1$ . Therefore there exists at least one component $T$ with $E _ { T } = V _ { T } - 1$ . Because every vertex has degree $\geq 1 , T$ contains at least two vertices. A connected multigraph with $E _ { T } = \dot { V _ { T } } - 1$ has no cycles (any cycle requires at least $V _ { T }$ edges), so $T$ is a tree (acyclic, simple). Bipartition this tree $\dot { T }$ into $\dot { U } \cup \dot { W }$ . Both U and W are non-empty.

Define $P = \{ k : \bar { I _ { k } } \stackrel { . } { \in } U \}$ and $Q = \{ k : I _ { k } \in W \}$ . The tree has at least two vertices and only one can be the dummy D, so $P \cup Q$ is nonempty. For a directed edge $e = ( A , B )$ in $T$ (where A is the parent of $a _ { i } , B$ is the parent of $b _ { i } ) _ { \cdot }$ , define $\mathrm { s g n } ( e ) = + 1 \mathrm { i f } \mathbf { \bar { A } } \in U , \mathbf { \bar { B } } \in W$ and −1 if $A \in W , \mathbf { \bar { B } } \in U$ . Let $d _ { e } = a _ { i } - b _ { i } ( \mathbf { s o } ~ d _ { e } \geq 0 )$ . For a vertex v let $L ( v )$ be the total length of pieces from v (all pieces from v are incident to edges in $T ,$ because v belongs to T). We claim

$$
\sum _ { e \in T } \operatorname { s g n } ( e ) d _ { e } = \sum _ { v \in U } L ( v ) - \sum _ { v \in W } L ( v ) .
$$

Indeed, for an edge $e = ( A , B )$ the contribution is $\mathrm { s g n } ( e ) ( a _ { i } - b _ { i } )$ . If $A \in U , B \in W$ this is $a _ { i } - b _ { i } ; \mathrm { i f } A \in W , B \in U$ it is $b _ { i } - a _ { i }$ . In both cases the piece from the U-endpoint receives +1 and the piece from the W-endpoint receives −1. Summing over all edges in $\dot { T _ { ☉ } }$ each piece from $v \in U$ is counted +1 and each from $v \in W$ is counted −1. Hence the identity holds. Now $L ( I _ { k } ) = l _ { k } = 2 ^ { k - 1 } \delta \mathrm { f o r } k = 1 , \dots , n + 1$ and $L ( D ) = 0$ . The difference $\textstyle \sum _ { U } L - \sum _ { W } L$ is a sum of some $l _ { k }$ with coefficients ±1. Since the $l _ { k }$ are distinct powers of two times $\delta ,$ any such sum is a multiple of $\delta .$ . Moreover, $P$ and $Q$ are disjoint and ${ \bar { P } } \cup Q$ is nonempty. If $\begin{array} { r } { \sum _ { k \in P } ^ { } 2 ^ { k - 1 } = \sum _ { k \in Q } 2 ^ { \dot { k } - 1 } } \end{array}$ , uniqueness of binary expansion gives $P = Q ;$ disjointness then gives $P = Q = \emptyset$ , contradicting $P \cup Q$ nonempty. Therefore the signed sum is nonzero and its absolute value is at least δ.

Now $\begin{array} { r } { D = \sum _ { i = 1 } ^ { n + 1 } ( a _ { i } - b _ { i } ) = \sum _ { e \in E } d _ { e } } \end{array}$ . Since $d _ { e } \geq 0$ , we have

$$
D \geq \sum _ { e \in T } d _ { e } \geq \left| \sum _ { e \in T } \operatorname { s g n } ( e ) d _ { e } \right| = \left| \sum _ { U } L - \sum _ { W } L \right| \geq \delta .
$$

Thus Liu’s total is at least $\textstyle { \frac { 1 + \delta } { 2 } }$

## 2. UPPER BOUND – XIANG CAN FORCE $D \leq \delta$

If Liu marks fewer than n points, he creates at most n intervals. Xiang bisects each interval (using at most n cuts); then all pieces come in equal pairs, so $D = 0 \leq \delta .$ Assume Liu marks exactly n points, creating $n + 1$ intervals of lengths $a _ { 0 } , a _ { 1 } , \dots , a _ { n } > 0$ with $\textstyle \sum a _ { i } = 1$ Consider all subsets of $\{ 0 , 1 , \ldots , n \}$ . There are $2 ^ { n + 1 }$ subset sums (counted with multiplicity), all in [0, 1], including 0 and 1. Sort them: $s _ { 1 } \leq s _ { 2 } \leq \cdot \cdot \cdot \leq s _ { 2 ^ { n + 1 } }$ . The $2 ^ { n + 1 } - 1$ consecutive gaps are non-negative and telescope to 1, so one gap is at most ${ \frac { 1 } { 2 ^ { n + 1 } - 1 } } = \delta$ . The two entries bounding this gap come from two distinct subsets (if the gap is zero the subsets are still distinct). Hence there exist distinct subsets $X , Y \subseteq \{ 0 , \ldots , n \}$ with ${ \big | } \sum X - \sum Y { \big | } \leq \delta$ Among all such pairs, choose one that minimises $| X | + { \overline { { | Y | } } }$ . Let $A \ = \ X$ $B \ = \ Y$ with $\Sigma A \bar { \geq } \Sigma B$ (swap if necessary) and set $r = \textstyle \sum A - \sum B ;$ then $0 \leq r \leq \delta .$ Claim 1. ${ \overline { { . } } } A \cap B = \emptyset .$ . Proof. If $Z = A \cap B \neq { \overline { { \alpha } } }$ $B \neq \overline { { \varnothing } } , \operatorname { l e t } \bar { A ^ { \prime } } = A \setminus Z , \ B ^ { \prime } = B \setminus Z$ . Then $A ^ { \prime } , B ^ { \prime }$ are distinct, $| \sum A ^ { \prime } - \sum B ^ { \prime } | \stackrel { . . } { = } \dot { r } \leq \delta , \mathrm { a n d } | A ^ { \prime } | \stackrel { . . } { + } | B ^ { \prime } | = | A | + | B | ^ { - } - 2 | Z | < | A | ^ { + } + | B |$ , contradicting minimality. □

$$
p = | A | , q = | B |
$$

Case 1: $B = \varnothing .$ . Then A is non-empty and $\quad \sum A = r \leq \delta .$ . Minimality forces $| { \cal A } | = 1$ . Indeed, $\operatorname { i f } \left| A \right| \geq 2 ,$ choose a singleton $\{ i \} \subset A .$ . Then $\Sigma \{ i \} = a _ { i } \leq r \leq \delta$ and $| \{ i \} | + | \dot { \emptyset } | = 1 < | A | .$ contradicting minimality. Thus $\dot { A } = \{ k \}$ for some k and $r = a _ { k } \le \delta$ . Xiang leaves $I _ { k }$ uncut and bisects all other n intervals (using n cuts). The pieces are one piece of length r and 2n pieces forming n equal pairs. By Lemma 2, $D = r \leq \delta$

Case 2: $B \neq \varnothing .$ . Then $q \geq 1$ . We have A, B disjoint. Order the intervals in A by their positions on the stick: lengths $\alpha _ { 1 } , \ldots , \alpha _ { p } .$ , cumulative sums $\begin{array} { r } { A _ { 0 } = 0 , A _ { 1 } = \alpha _ { 1 } , \dots , A _ { p } = \sum A } \end{array}$ . Similarly for $B \colon \beta _ { 1 } , \ldots , \bar { \beta _ { q } } , B _ { 0 } = 0 , \ldots , B _ { q } = \textstyle \sum { B }$

$$
1 \leq k ^ { ^ { \prime } } < \bar { p } , A _ { k } \leq \sum B
$$

$$
\textstyle \sum B < A _ { k } < \sum A
$$

$$
k < p .
$$

$$
\{ \alpha _ { k + 1 } , \ldots , \alpha _ { p } \}
$$

$$
\textstyle \sum T = \sum A - A _ { k } < r \leq \delta
$$

$$
| T | = p - k < p + q
$$

$$
q \geq 1 ) .
$$

$$
\begin{array} { r } { | \widecheck { \sum } T - \widecheck { \sum } \emptyset | = \widecheck { \sum } T \leq \delta , } \end{array}
$$

$$
( T , \emptyset )
$$

$$
| T | + 0 = p - k < p + q ,
$$

$$
A _ { p - 1 } \leq \sum B \leq A _ { p }
$$

Now we construct Xiang’s marks. The cumulative numbers $A _ { j }$ and $B _ { j }$ are coordinates in two separate abstract concatenations, not positions on the original stick. Order the actual selected A-intervals left-to-right and translate them end-to-end, preserving lengths, to virtual $[ 0 , \sum A ]$ where actual interval $\alpha _ { j }$ represents virtual $[ A _ { j - 1 } , A _ { j } ]$ . Separately translate the actual B-intervals to virtual $[ 0 , \Sigma B ]$

Retain Claim 2 and the common sorted boundary union $\begin{array} { r } { 0 = t _ { 0 } < t _ { 1 } < \dots < t _ { s } = \sum B } \end{array}$ , where

$$
S = \{ 0 \} \cup \{ A _ { 1 } , \ldots , A _ { p - 1 } \} \cup \{ B _ { 1 } , \ldots , B _ { q } \} .
$$

(Recall $A _ { p - 1 } \leq \sum B$ , so all $A _ { j }$ for $j < p$ are $\leq \sum B$ , and $\boldsymbol { B } _ { \boldsymbol { q } } = \sum \boldsymbol { B } . )$ For each B-boundary $B _ { j } \ ( 1 \ \stackrel { \cdot } { \ } \ ) \ j \ \leq \ q )$ strictly inside a virtual A-cell $[ A _ { k - 1 } , A _ { k } ]$ , mark the corresponding actual A-interval at distance $B _ { j } - A _ { k - 1 }$ from its actual left endpoint. For each Aboundary $A _ { j } \bar { ( 1 \leq j \leq p - 1 ) }$ ) strictly inside a virtual B-cell $[ B _ { k - 1 } , B _ { k } ]$ , mark the corresponding actual B-interval at distance $A _ { j } \ : - \ : B _ { k - 1 }$ from its actual left endpoint. Equal boundaries need no cuts. Thus overlay cuts $\leq p + q - 1$ , and all marks are legal and distinct.

Because A and B are disjoint, the A-intervals and B-intervals are distinct intervals on the stick. The two abstract concatenations now have the same refinement boundary list up to $\sum B$ , so their resulting physical pieces have paired equal lengths $t _ { i } - t _ { i - 1 } ;$ they need not be at the same physical locations.

Because $A _ { p - 1 } \leq \sum B \leq A _ { p }$ , the virtual tail $[ \Sigma B , \Sigma A ]$ corresponds to one tail in the last actual A-interval, of length r; when $r = 0$ it is absent.   
Bisect all intervals not in A or $B ;$ there are $n + 1 - p - q$ of them, using $n + 1 - p - q$ cuts. Total cuts used is at most $( p + q - 1 ) + ( n + 1 - p - q ) = n$ . Hence Xiang can use at most n cuts.

The final pieces consist of: • equal pairs from the overlay construction, • equal pairs from the bisected intervals, • one extra piece of length $r \left( \mathrm { i f } r > 0 \right)$

By Lemma 2, the alternating sum D equals ${ \mathfrak { o r } } 0 { \mathrm { i f } } r = 0 )$ . Since $r \leq \delta ,$ , we have $D \leq \delta .$

## 3. CONCLUSION

Liu can guarantee at least $\textstyle { \frac { 1 + \delta } { 2 } }$ and Xiang can force at most $\textstyle { \frac { 1 + \delta } { 2 } }$ . Therefore the largest number c Liu Bang can guarantee is

$$
\boxed { \frac { 2 ^ { n } } { 2 ^ { n + 1 } - 1 } }
$$

$$
c = { \frac { 1 + \delta } { 2 } } = { \frac { 1 + { \frac { 1 } { 2 ^ { n + 1 } - 1 } } } { 2 } } = { \frac { 2 ^ { n } } { 2 ^ { n + 1 } - 1 } } .
$$

F.4.4 IMO PROBLEM 4  
Question   
Shan-Yu and Mulan are playing a game. Let θ be an angle satisfying   
$0 ^ { \circ } < \theta < 1 8 0 ^ { \circ } ,$   
known to both players. Initially, Shan-Yu constructs a paper triangle $T$ with measurements of   
his choice. They then repeatedly perform the following steps:   
1. If T has at least one angle measuring exactly $\theta ,$ the game ends and Mulan wins.   
2. Otherwise, Mulan chooses a point $P$ on the perimeter of T, distinct from its three   
vertices.   
3. She makes a straight cut from P to the opposite vertex of T, dividing $T$ into two   
triangles.   
4. Shan-Yu discards one of the two triangles, and the remaining triangle becomes the new   
T.   
For which real values of θ can Mulan guarantee victory in finitely many steps, regardless of how   
Shan-Yu plays?

K2-Horizon-7B generation   
Let θ be an angle with $0 ^ { \circ } \ : < \theta \ : < \ : 1 8 0 ^ { \circ }$ . A triangle is determined by its three positive angles   
A, B, C with ${ \bar { A } } + B + C = 1 8 0 ^ { \circ }$ . A cut at vertex $A$ (with angles $\bar { \boldsymbol { A } } , \boldsymbol { B } , \boldsymbol { C } )$ means choosing   
a point $P$ on side BC (different from $B , C )$ and drawing segment AP. This splits the triangle   
into two triangles with angles   
$( x , \ B , \ 1 8 0 ^ { \circ } - x - B ) \qquad { \mathrm { a n d } } \qquad ( A - x , \ C , \ 1 8 0 ^ { \circ } - ( A - x ) - C )$   
for some x with $0 < x < A .$ (The angles at $B$ and $C$ remain unchanged, and the new angles at   
A are x and $A - x ;$ the third angles are forced by the sum $1 8 0 ^ { \circ } .$ )   
We determine for which θ Mulan has a strategy that guarantees victory in finitely many steps,   
no matter how Shan-Yu plays.   
1. NECESSITY: IF θ DOES NOT DIVIDE $1 8 0 ^ { \circ } .$ , MULAN CANNOT FORCE A WIN.   
Assume $\theta \dag 1 8 0 ^ { \circ } ;$ then there is no integer k with $1 8 0 ^ { \circ } = k \theta .$   
Lemma 1. Let $T$ be a triangle with angles $A , B , C ,$ none of which is an integer multiple of   
θ. For any cut of $T ,$ at most one of the two resulting triangles has an angle that is an integer   
multiple of θ.   
Proof. Suppose we cut at vertex A, splitting A into x and $A - x \left( 0 < x < A \right)$ . The two triangles   
are   
$T _ { 1 } = ( x , \ B , \ 1 8 0 ^ { \circ } - x - B ) , \qquad T _ { 2 } = ( A - x , \ C , \ 1 8 0 ^ { \circ } - ( A - x ) - C ) .$   
Assume, for contradiction, that both $T _ { 1 }$ and $T _ { 2 }$ contain an angle that is an integer multiple of θ.   
We examine the possibilities.   
• If $T _ { 1 } \ ' _ { \mathrm { s } }$ multiple is B or $T _ { 2 } \mathrm { { ' } s }$ multiple is $C ,$ then $T$ itself has a multiple, contradicting the   
hypothesis. $\bullet \mathrm { I f } \ : T _ { 1 } \ : \mathrm { \overline { { s } } }$ multiple is x and $\bar { T } _ { 2 } ^ { \ , } \mathrm { s }$ multiple is $A - x ,$ , then $A = x + ( A - x )$ is a multiple   
of θ, contradiction. $\cdot \mathrm { H } T _ { 1 } \dot { ? }$ s multiple is x and ${ T _ { 2 } } ^ { j }$ s multiple is the third angle $1 8 0 ^ { \circ } - \left( A - x \right) - \mathrm { \dot { C } } ,$   
then   
$1 8 0 ^ { \circ } - ( A - x ) - C = B + x \qquad ( { \mathrm { s i n c e ~ } } A + B + C = 1 8 0 ^ { \circ } ) .$   
Hence $B + x$ is a multiple of $\theta ;$ because $x$ is a multiple of θ, B is also a multiple of $\theta ,$ contra  
diction. • If T ’s multiple is the third angle $1 8 0 ^ { \circ } - x { \stackrel { - } { - } } B$ and $T _ { 2 } \mathrm { { ' } s }$ multiple is ${ \bar { \boldsymbol { A } } } - { \boldsymbol { x } } ,$ then   
$1 8 0 ^ { \circ } - x - B = ( A - x ) + C \qquad { \mathrm { ( a g a i n u s i n g ~ } } A + B + C = 1 8 0 ^ { \circ } { \mathrm { ) } } ,$   
so $C$ is a multiple of θ, contradiction. • If both multiples are the third angles, then   
$( 1 8 0 ^ { \circ } - x - B ) + ( 1 8 0 ^ { \circ } - ( A - x ) - C ) = 3 6 0 ^ { \circ } - ( A + B + C ) = 1 8 0 ^ { \circ } ,$

which gives $( p + q ) \theta = 1 8 0 ^ { \circ }$ for some integers $p , q ;$ this contradicts $\theta \nmid 1 8 0 ^ { \circ }$   
All cases lead to a contradiction, so at most one of $T _ { 1 } , T _ { 2 }$ can have a multiple of $\theta .$ □ Shan-Yu’s strategy. Choose an initial triangle with no angle a multiple of $\theta ;$ for instance the equilateral triangle with angles $6 0 ^ { \circ } , 6 0 ^ { \circ } , 6 0 ^ { \circ } . \mathrm { ~ ( I f ~ 6 0 ^ { \circ } ~ }$ were a multiple of $\theta ,$ then $1 8 0 ^ { \circ } = 3 \cdot 6 0 ^ { \circ }$ would be a multiple of $\theta ,$ contradicting $\theta \ : \forall \ : 1 8 0 ^ { \circ } .$ )   
Whenever Mulan makes a cut, Shan-Yu discards the triangle that contains a multiple of θ (if such a triangle exists); otherwise he discards arbitrarily. By Lemma 1 the remaining triangle has no multiple of $\theta .$ Hence the game never produces an angle equal to $\theta ,$ and Mulan never wins. Thus Mulan cannot guarantee victory when $\theta \mathbin { \uparrow } 1 8 0 ^ { \circ }$

## 2. SUFFICIENCY: IF $\theta$ DIVIDES $1 8 0 ^ { \circ }$ , MULAN HAS A WINNING STRATEGY.

$$
1 8 0 ^ { \circ } = m \theta
$$

$$
m \geq 2
$$

$$
\theta < 1 8 0 ^ { \circ } \mathrm { g i v e s } m > 1 )
$$

$$
T
$$

$$
A , B , C .
$$

5. • If any angle equals θ, Mulan wins immediately. • Otherwise, we distinguish two cases. Case 1. $\Gamma ^ { \mathbf { \bar { \mathbf { \Lambda } } } }$ has an angle that is a positive integer multiple of θ, say $A = q \theta$ with $q \geq 2 .$ . Mulan cuts at vertex A, splitting it into $\bar { \theta }$ and $( q - 1 ) \overline { { \theta } }$ . The two resulting triangles are

$$
T _ { 1 } = ( \theta , \ B , \ A + C - \theta ) , \qquad T _ { 2 } = ( ( q - 1 ) \theta , \ C , \ B + \theta ) .
$$

(One checks that $A + C - \theta = 1 8 0 ^ { \circ } - B - \theta$ and $B + \theta = 1 8 0 ^ { \circ } - C - ( q - 1 ) \theta .$ , so the angles sum to $1 8 0 ^ { \circ }$ in each triangle.) $T _ { 1 }$ contains $\theta ,$ so Shan-Yu will discard $T _ { 1 }$ . The remaining triangle is $T _ { 2 } . \mathrm { ~ H ~ } q = 2 ,$ then $T _ { 2 }$ contains $\theta$ and Mulan wins. I $\mathrm { ~ f ~ } q > 2 ,$ , then $T _ { 2 }$ has an angle ${ \bar { ( q - 1 ) } } \theta$ which is a positive integer multiple of $\theta ;$ we are back to Case 1 with a smaller multiple. Hence after at most $q - 1$ moves Mulan wins.

Case 2. T has no angle that is a positive integer multiple of $\theta .$ Mulan performs a special cut to create two triangles that both have a multiple of $\theta .$ Lemma 2. There exist a vertex $i \in \{ 0 , 1 , 2 \}$ and a positive integer $\mathit { p }$ such that

$$
T ( i + 1 ) < p \theta < T ( i ) + T ( i + 1 ) \qquad \mathrm { ( i n d i c e s ~ m o d u l o ~ 3 ) } .
$$

Proof of Lemma 2. Suppose first that there is a vertex i with $T ( i ) > \theta .$ . Let $p$ be the smallest positive integer with $p \mathbf { \bar { \theta } } > T ( i + 1 )$ (such $p$ exists because $T ( i + 1 ) < 1 8 0 ^ { \circ } = m \theta )$ . Then $\begin{array} { r } { \dot { ( } p - 1 ) \theta \le \bar { T } ( i + 1 ) \dot { < } p \theta . } \end{array}$ . Since $T ( i ) > \theta ,$

$$
p \theta \leq T ( i + 1 ) + \theta < T ( i + 1 ) + T ( i ) ,
$$

$$
p \theta < T ( i ) + T ( i + 1 )
$$

$$
T ( i + 1 ) < p \theta
$$

$$
T ( i ) < \theta
$$

$$
i = 0
$$

$$
T ( i ) \leq \theta _ { : }
$$

$$
T ( 1 ) < \dot { \theta }
$$

$$
\theta
$$

$$
T ( 0 ) + T ( 1 ) = 1 8 0 ^ { \circ } - T ( 2 ) > 1 8 0 ^ { \circ } - \theta = ( m - 1 ) \theta \geq \theta \quad ( { \mathrm { s i n c e ~ } } m \geq 2 ) .
$$

Hence $T ( 1 ) < \theta < T ( 0 ) + T ( 1 )$ , and we may take $p = 1 .$ □ Now let i and $p$ be as in Lemma 2. Set $x = \overset { \cdot } { p } \theta - \overset { \cdot } { T } ( i + 1 )$ . Then $0 < x < T ( i )$ . Mulan cuts at vertex i, splitting the angle $T ( i )$ into x and $T ( i ) - x$ . The two resulting triangles are

$$
L = ( x , \ T ( i + 1 ) , \ ( m - p ) \theta ) , \qquad R = ( T ( i ) - x , \ T ( i + 2 ) , \ p \theta ) .
$$

(One verifies the third angles using $A + B + C = 1 8 0 ^ { \circ }$ and $1 8 0 ^ { \circ } = m \theta . )$   
Both L and R have an angle that is a positive integer multiple of $\theta .$ Shan-Yu discards one of them. The remaining triangle has an angle that is a positive integer multiple of θ. If that multiple is $\theta ,$ Mulan wins immediately. If it is larger, say $q \theta$ with $q \geq 2$ , Mulan proceeds to Case 1, which will finish in at most $q - 1$ moves. Because $p \leq m - 1$ and the remaining multiple is either p or $m - p ,$ we have $q \leq m - 1$ . Thus after the initial cut Mulan wins in at most $m - 1$ further moves, i.e. in at most m moves altogether.   
This completes the description of Mulan’s strategy.

## 3. CONCLUSION

$$
\boxed { \theta = \frac { 1 8 0 ^ { \circ } } { k } \quad \mathrm { f o r ~ s o m e ~ i n t e g e r ~ } k \geq 2 } .
$$

Mulan can guarantee her victory in finitely many steps if and only if θ divides $1 8 0 ^ { \circ }$ , i.e.

## F.4.5 IMO PROBLEM 5

## Question

Determine all functions $f : \mathbb { R } _ { > 0 }  \mathbb { R } _ { > 0 }$ such that $\sqrt { ( x ^ { 2 } + f ( y ) ^ { 2 } ) / 2 } \ge ( f ( x ) + y ) / 2 \ge$   
$\sqrt { x f ( y ) }$ for every $x , y \in \mathbb { R } _ { > 0 } .$   
K2-Horizon-7B generation   
Let $f : ( 0 , \infty ) \to ( 0 , \infty )$ satisfy   
$\sqrt { \frac { x ^ { 2 } + f ( y ) ^ { 2 } } { 2 } } \geq \frac { f ( x ) + y } { 2 } \geq \sqrt { x f ( y ) } ( \forall x , y > 0 ) .$   
All quantities are positive, so squaring preserves the order. The left inequality becomes   
${ \frac { x ^ { 2 } + f ( y ) ^ { 2 } } { 2 } } \geq { \Big ( } { \frac { f ( x ) + y } { 2 } } { \Big ) } ^ { 2 } \iff 2 x ^ { 2 } + 2 f ( y ) ^ { 2 } \geq ( f ( x ) + y ) ^ { 2 } ,$   
and the right inequality becomes   
$\Big ( \frac { f ( x ) + y } { 2 } \Big ) ^ { 2 } \geq x f ( y ) \iff ( f ( x ) + y ) ^ { 2 } \geq 4 x f ( y ) .$   
Thus for all $x , y > 0$   
$2 x ^ { 2 } + 2 f ( y ) ^ { 2 } \geq ( f ( x ) + y ) ^ { 2 } \geq 4 x f ( y ) .$ (1)   
1. A FUNCTIONAL EQUATION   
Put $x = f ( y )$ in (1). Because $f ( y ) > 0 ,$   
$2 f ( y ) ^ { 2 } + 2 f ( y ) ^ { 2 } \geq ( f ( f ( y ) ) + y ) ^ { 2 } \geq 4 f ( y ) ^ { 2 } ,$   
hence $( f ( f ( y ) ) + y ) ^ { 2 } = 4 f ( y ) ^ { 2 }$ . Since $f ( f ( y ) ) + y > 0 ,$   
$f ( f ( y ) ) + y = 2 f ( y ) \qquad ( \forall y > 0 ) ,$   
or   
f(f(y)) = 2f(y) − y. (2)   
2. INTRODUCING g   
Define $g ( y ) = f ( y ) - y$ for $y > 0 ;$ then $f ( y ) = y + g ( y )$ . From (2),   
g(f(y)) = f(f(y)) − f(y) = (2f(y) − y) − f(y) = f(y) − y = g(y). (3)   
3. ITERATION AND NON-NEGATIVITY OF g   
We prove by induction that for every $y > 0$ and every $n \in \mathbb { N } ,$   
g(y + n g(y)) = g(y) and y + n g(y) > 0. (4)   
Base $n = 0 \colon g ( y ) = g ( y )$ and $y > 0 .$ Inductive step: Assume (4) holds for some n. Then   
f(y + n g(y)) = (y + n g(y)) + g(y + n g(y)) = y + (n + 1)g(y).   
Because $y + n g ( y ) > 0$ , the hypothesis $f ( z ) > 0$ for $z > 0$ gives $f ( y + n g ( y ) ) > 0 ;$ hence   
$y + ( n + 1 ) g ( y ) > 0$ . Moreover,   
$g ( y + ( n + 1 ) g ( y ) ) = g { \big ( } f ( y + n g ( y ) ) { \big ) } = g ( y + n g ( y ) ) = g ( y ) ,$   
where we used (3) and the induction hypothesis. Thus (4) holds for $n + 1 .$

Now suppose $g ( y ) < 0$ for some $y > 0 .$ Then $- g ( y ) > 0$ . Choose $n \in \mathbb N$ with $n > \frac { y } { - g ( y ) } .$ Then

$$
y + n g ( y ) = y - n ( - g ( y ) ) < 0 ,
$$

contradicting the second part of (4). Therefore

$$
g ( y ) \geq 0 \qquad ( \forall y > 0 ) .\tag{5}
$$

Consequently $f ( y ) = y + g ( y ) \geq y$ for all $y > 0 .$

## 4. A KEY INEQUALITY

For arbitrary $x , y > 0$ set $\Delta = g ( x ) - g ( y )$ . Expanding the two parts of (1) gives

$$
2 x ^ { 2 } + 2 f ( y ) ^ { 2 } - ( f ( x ) + y ) ^ { 2 } = ( x - f ( y ) ) ^ { 2 } - 2 \Delta { \left( x + f ( y ) \right) } - \Delta ^ { 2 } ,
$$

$$
( f ( x ) + y ) ^ { 2 } - 4 x f ( y ) = ( x - f ( y ) ) ^ { 2 } + 2 \Delta \left( x + f ( y ) \right) + \Delta ^ { 2 } .
$$

Hence (1) yields

$$
( x - f ( y ) ) ^ { 2 } - 2 \Delta \left( x + f ( y ) \right) - \Delta ^ { 2 } \geq 0 ,\tag{6}
$$

$$
\begin{array} { r } { ( x - f ( y ) ) ^ { 2 } + 2 \Delta \left( x + f ( y ) \right) + \Delta ^ { 2 } \geq 0 . } \end{array}\tag{7}
$$

Now observe that

$$
4 f ( y ) \Delta + \left( ( x - f ( y ) ) + \Delta \right) ^ { 2 } = ( x - f ( y ) ) ^ { 2 } + 2 \Delta \left( x + f ( y ) \right) + \Delta ^ { 2 } .
$$

Thus (6) and (7) are exactly the two parts of

$$
0 \leq 4 f ( y ) { \big ( } g ( x ) - g ( y ) { \big ) } + { \big ( } ( x - f ( y ) ) + ( g ( x ) - g ( y ) ) { \big ) } ^ { 2 } \leq 2 ( x - f ( y ) ) ^ { 2 } .\tag{8}
$$

So (8) holds for all $x , y > 0 .$

## 5. BOUNDING THE DIFFERENCE OF g

Let $u , v > 0 .$ . We prove

$$
4 \operatorname* { m i n } ( u , v ) | g ( u ) - g ( v ) | \leq ( u - v ) ^ { 2 } .\tag{9}
$$

Assume first that $g ( v ) \leq g ( u )$ . Put $x = f ( u ) , y = v$ in (8). Using (3),

$$
\begin{array} { r l } & { 0 \leq 4 f ( v ) \big ( g ( f ( u ) ) - g ( v ) \big ) } \\ & { \qquad + \big ( ( f ( u ) - f ( v ) ) + ( g ( f ( u ) ) - g ( v ) ) \big ) ^ { 2 } } \\ & { \qquad \leq 2 ( f ( u ) - f ( v ) ) ^ { 2 } . } \end{array}
$$

Since $g ( f ( u ) ) = g ( u )$ and $f ( v ) = v + g ( v )$ , the right-hand inequality becomes

$$
\begin{array} { r l } & { 4 \big ( v + g ( v ) \big ) \big ( g ( u ) - g ( v ) \big ) } \\ & { \quad + \left( \big ( f ( u ) - f ( v ) \big ) + \big ( g ( u ) - g ( v ) \big ) \right) ^ { 2 } } \\ & { \quad \le 2 ( f ( u ) - f ( v ) ) ^ { 2 } . } \end{array}
$$

Now $f ( u ) = u + g ( u ) , f ( v ) = v + g ( v ) ,$ , so

$$
f ( u ) - f ( v ) = ( u - v ) + ( g ( u ) - g ( v ) ) .
$$

Let $\Delta = g ( u ) - g ( v ) \geq 0$ . Then

$$
f ( u ) - f ( v ) = ( u - v ) + \Delta , \qquad \left( f ( u ) - f ( v ) \right) + \Delta = ( u - v ) + 2 \Delta .
$$

Substituting these into the inequality and expanding gives

$$
4 v \Delta + 4 g ( v ) \Delta + ( u - v ) ^ { 2 } + 4 \Delta ( u - v ) + 4 \Delta ^ { 2 } \leq 2 ( u - v ) ^ { 2 } + 4 \Delta ( u - v ) + 2 \Delta ^ { 2 } ,
$$

which simplifies to

$$
4 v \Delta + 4 g ( v ) \Delta + 2 \Delta ^ { 2 } \leq ( u - v ) ^ { 2 } .
$$

Because $g ( v ) \geq 0$ and $\Delta \geq 0$ , we have

$$
4 v \Delta \leq 4 ( v + g ( v ) ) \Delta \leq ( u - v ) ^ { 2 } - 2 \Delta ^ { 2 } \leq ( u - v ) ^ { 2 } .
$$

Thus

$$
4 v ( g ( u ) - g ( v ) ) \leq ( u - v ) ^ { 2 } .\tag{10}
$$

If instead $g ( u ) \leq g ( v )$ , we interchange the roles of u and v (use $x = f ( v )$ , y = u in (8)) and obtain

$$
4 u ( g ( v ) - g ( u ) ) \leq ( u - v ) ^ { 2 } .\tag{11}
$$

In either case (9) follows.

## 6. g IS CONSTANT

Suppose there exist $p , q > 0$ with $g ( p ) \neq g ( q )$ . Without loss of generality, assume $p < q .$ . Put $E \stackrel {  } { = } | g ( q ) - g ( p ) | > 0 .$ . Choose a natural number n such that

$$
n > \frac { ( q - p ) ^ { 2 } } { 4 p E } .\tag{12}
$$

Set $d = { \frac { q - p } { n } } > 0$ and define $z _ { k } = p + k d$ for $k = 0 , 1 , \ldots , n$ . Then $z _ { n } = q ,$ and for each k   
we have $z _ { k } > 0$ and min $\begin{array} { r } { \ i ( z _ { k + 1 } , z _ { k } ) = z _ { k } \ge p . } \end{array}$   
Applying (9) to the pair $\left( z _ { k + 1 } , z _ { k } \right)$ yields

$$
4 z _ { k } | g ( z _ { k + 1 } ) - g ( z _ { k } ) | \leq ( z _ { k + 1 } - z _ { k } ) ^ { 2 } = d ^ { 2 } .
$$

Since $z _ { k } \geq p ,$

$$
4 p | g ( z _ { k + 1 } ) - g ( z _ { k } ) | \leq 4 z _ { k } | g ( z _ { k + 1 } ) - g ( z _ { k } ) | \leq d ^ { 2 } .
$$

Summing these inequalities for $k = 0 , 1 , \ldots , n - 1 \mathrm { g i v e s }$

$$
4 p \sum _ { k = 0 } ^ { n - 1 } | g ( z _ { k + 1 } ) - g ( z _ { k } ) | \leq n d ^ { 2 } = { \frac { ( q - p ) ^ { 2 } } { n } } .
$$

By the triangle inequality,

$$
\sum _ { k = 0 } ^ { n - 1 } | g ( z _ { k + 1 } ) - g ( z _ { k } ) | \geq | g ( z _ { n } ) - g ( z _ { 0 } ) | = | g ( q ) - g ( p ) | = E .
$$

Consequently

$$
4 p E \leq { \frac { ( q - p ) ^ { 2 } } { n } } .
$$

But (12) implies $\frac { ( q - p ) ^ { 2 } } { \ldots } < 4 p E .$ , a contradiction. Hence our assumption was false, and $g ( p ) =$ n   
$g ( q )$ for all $p , q > 0$ . Therefore g is constant.

## 7. DETERMINING f

Let $g ( y ) = c$ for all $y > 0$ . Then $f ( y ) = y + c . $ The condition $f ( y ) > 0$ for all $y > 0$ forces $c \geq 0$ . (If $c < 0 ,$ , then for $y = - c / 2 > 0$ we would have $f ( y ) = c / 2 < 0$ , impossible.) Conversely, if $c \geq 0$ and $f ( x ) = { \dot { x } } + c ,$ , then for any $x , y > 0$

$$
{ \sqrt { \frac { x ^ { 2 } + ( y + c ) ^ { 2 } } { 2 } } } \geq { \frac { x + y + c } { 2 } } \geq { \sqrt { x ( y + c ) } } .
$$

Indeed, the right inequality is equivalent to $( x + y + c ) ^ { 2 } \geq 4 x ( y + c )$ , which simplifies to $( x - y - c ) ^ { 2 } \overset { \cdot } { \geq } 0 ;$ the left inequality is equivalent to $2 ( x ^ { 2 } + ( y + \bar { c } ) ^ { 2 } ) \geq ( x + y + c ) ^ { 2 }$ , which simplifies to $( x - ( y + c ) ) ^ { 2 } \geq 0$ . Both are true.

## $f ( x ) = x + c$ for an arbitrary constant $c \geq 0$

## F.4.6 IMO PROBLEM 6

## Question

Let $a _ { 1 } , a _ { 2 } , a _ { 3 } , . . .$ . be an infinite sequence of positive integers greater than 1. Suppose that, for every positive integer $n ,$ the number $a _ { n + 1 }$ is the smallest integer greater than $a _ { n }$ such that gcd $( a _ { n + 1 } , a _ { i } ) > 1$ for every $1 \leq i \leq n$ . Prove that there exist positive integers T and L such that $a _ { n + T } = a _ { n } + L$ for every positive integer n.

<table><tr><td>K2-Horizon-7B generation Let  $( a _ { n } ) _ { n \geq 1 }$  be the given sequence of integers  $> 1$  . Define a positive integer  $x > 1$ </td></tr><tr><td> $\operatorname* { g c d } ( a _ { m } , a _ { m } ) = a _ { m } > 1$   $m < n ,$   $a _ { n }$   $> a _ { n - 1 }$  with  $\operatorname* { g c d } ( a _ { n } , a _ { i } ) > 1$  for all  $i \leq n - 1 )$  we have  $m \leq n - 1$  and hence  $\operatorname* { g c d } ( a _ { n } , a _ { m } ) > 1$  If  $m > n$  the same argument with roles reversed gives gco  $\mathrm { 1 } ( a _ { m } , a _ { n } ) > 1$  □ Consequently every term  $a _ { n }$  is good: for any  $i , \mathrm { i f } \ i < n$  then  $\operatorname* { g c d } ( a _ { n } , a _ { i } ) > 1$  by the above; if  $i > n$  then  $\operatorname* { g c d } ( a _ { i } , a _ { n } ) > 1$  ; and if  $i = n$  then  $a _ { n } > 1 \quad$  Lemma 2. The sequence is strictly increasing and  $a _ { n } \geq a _ { 1 } + n - 1$  for all  $n \geq 1$  . Proof. The hypothesis gives  $a _ { n + 1 } > a _ { n } . \ \mathbf { B } \mathbf { y }$  induction,  $a _ { n } \geq a _ { 1 } + n - 1$  □ Lemma 3. The set of good numbers that are  $\geq a _ { 1 }$  is exactly  $\{ a _ { n } : n \geq 1 \}$  . Proof. By Lemma 1 every  $a _ { n }$  is good and  $a _ { n } \geq a _ { 1 }$  Conversely, let x be good with  $x \geq a _ { 1 } . \operatorname { I f } x = a _ { 1 }$  we are done.  $\mathrm { I f } \ x > a _ { 1 }$  , then by Lemma 2 the sequence is unbounded, so there exists a smallest  $N \geq 2$  such that  $a _ { N } \geq x$  Then  $a _ { N - 1 } < x \leq a _ { N }$   $\operatorname { I f } x = a _ { N }$  we are done.  $\mathrm { I f } \ x < a _ { N }$  , then by the definition of  $a _ { N }$  there exists  $i \leq N - 1$  with  $\operatorname* { g c d } ( x , a _ { i } ) = 1$  , contradicting that x is good. Hence  $x = a _ { N }$  □ Lemma 4. Any two good numbers are not coprime. Proof. Let  $x , y$  be good. Set  $z = y ^ { a _ { 1 } + 1 }$  Because  $y > 1$  and  $a _ { 1 } \geq 2 ,$  ,we have  $z > a _ { 1 }$  . For any i, since  $y$  is good there is a prime  $p \mid y$  and  $p \mid a _ { i } ;$  then  $p \mid z .$  Thus  $\operatorname* { g c d } ( z , a _ { i } ) \geq p > 1$  for all  $i ,$  so z is good. By Lemma  $3 , z = a _ { N }$  for some N. Since x is good,  $\operatorname* { g c d } ( x , a _ { N } ) > 1$  , hence  $\operatorname* { g c d } ( x , z ) > 1$  . If  $\operatorname* { g c d } ( x , y ) = 1$  , then  $\operatorname* { g c d } ( x , y ^ { a _ { 1 } + 1 } ) = 1 \mathrm { . }$  a contradiction. Therefore  $\operatorname* { g c d } ( x , y ) > 1$  □ Lemma 5. Let  $B = a _ { 1 } ^ { a _ { 1 } + 1 } + a _ { 1 } + 2 .$  For every good number x and every  $i \geq 1$  there exists a prime  $p < B$  such that  $p \mid$  x and  $p \mid a _ { i } .$  Proof. We first establish some auxiliary facts. and thus then □</td></tr><tr><td>Witness Lemma.  $\mathrm { ~ I f ~ } k > a _ { 1 }$  is not good, then there exists an index  $j \geq 1$  with  $a _ { j } < k$  gcd  $( k , a _ { j } ) = 1$  . Proof. Since  $a _ { n }  \infty ,$  choose N with  $a _ { N } > k$  . Let n be the largest integer with  $a _ { n } ~ < ~ k$  (so  $a _ { n } < k < a _ { n + 1 }$  or  $k = a _ { n + 1 } )$  . Because  $k$  is not good,  $k \neq a _ { n + 1 } ;$   $a _ { n } < k < a _ { n + 1 }$  . By the definition of there is  $\AA . \leq n$  with  $a _ { n + 1 }$   $\operatorname* { g c d } ( k , a _ { i } ) = 1$  . Take  $j = i ;$   $a _ { j } \leq a _ { n } <$  k and  $\operatorname* { g c d } ( k , a _ { j } ) = 1$  Core numbers. A positive integer  $d > 1$  is called a core if d is good and for every good divisor f of d we have d  $\underline { { \bar { \cdot } } } \ f \ ( \mathrm { i . e . } .$  d has no proper good divisor). Every good number x possesses a core divisor: let d be the smallest good divisor of  $x .$  Then d is good,  $d \mid x ,$  and if  $f$  is a good divisor of  $d ,$  then  $f$  is a good divisor of  $x ,$  so by minimality  $d \leq f .$  Hence d is core. Claim. For every core  $d ,$  all prime factors of d are  $< B$  Proof by strong induction on d. Base case: d is prime. Since d is good, gcd  $( d , a _ { 1 } ) > 1 ;$  because d is prime,  $d \mid a _ { 1 } ,$  SO  $d \leq a _ { 1 } < B .$ </td></tr><tr><td>The only prime factor is d itself, which is  $< B .$  Inductive step: Assume d is composite and the claim holds for all core  $d ^ { \prime } < d .$  Let p be a prime divisor of d and write  $d = p e$  with  $e > 1 .$  Because d is core, e is not good (otherwise e would be a smaller good divisor of d). Also  $e < d .$  We distinguish two cases. Case 1:  $e \mathrm { ~ > ~ } a _ { 1 }$  . Since e is not good and  $e \mathrm { ~ > ~ } a _ { 1 }$  , the Witness Lemma gives an index  $j$  with  $a _ { j } <$  e and  $\operatorname* { g c d } ( e , a _ { j } ) = 1$  . Let c be the smallest good divisor of  $a _ { j } .$  Then c is good,  $c \mid a _ { j } ,$   $c \leq a _ { j } < e < d ,$  and c is core. Moreover  $\operatorname* { g c d } ( e , c ) = 1$  because  $c \mid a _ { j }$  and  $\operatorname* { g c d } ( e , a _ { j } ) = \mathrm { \widetilde { 1 } }$  Now c and d are both good, so by Lemma 4,  $\operatorname* { g c d } ( c , d ) > 1$  . Choose a prime  $q \mid \operatorname* { g c d } ( c , d )$  Then  $q \mid d = p e ,$  so  $q \mid p \mathrm { o r } q \mid$  e. Since  $\operatorname* { g c d } ( e , c ) = 1 , q \nmid e ;$  hence  $q \mid p .$  As p is prime, Thus  $q = p .$   $p \mid c .$  Now c is a core number with  $c < d .$  By the induction hypothesis, all prime factors of c are  $< B ;$  in particular  $p < B .$  Case  $\displaystyle 2 \colon \boldsymbol { e } ~ \le ~ a _ { 1 }$  Consider  $k = e ^ { a _ { 1 } + 1 }$  Because  $e > 1$  and  $a _ { 1 } \ \geq \ 2 ,$  we have  $k > a _ { 1 }$  and  $k \leq a _ { 1 } ^ { a _ { 1 } + 1 } < B .$  Also  $e \mid k .$  We claim k is not good. Indeed, if k were good then for every i,</td></tr><tr><td>a contradiction. Hence k is not good and  $k > a _ { 1 }$  . By the Witness Lemma there exists an index  $a _ { j } <$  k and  $\operatorname* { g c d } ( k , a _ { j } ) \stackrel { - } { = } 1$  . (Note that k is not a term, so the lemma applies.) Let c be the smallest good divisor of  $a _ { j }$  Then c is good,  $c \mid a _ { j } , c \leq a _ { j } < k ,$  and c is core. Moreover  $( e , c ) = 1$  because e | k and  $\operatorname* { s c d } ( k , a _ { j } ) { \dot { = } } 1 . \operatorname { A g a i n } .$  , c and d are good, so gcd q be a prime dividing  $\operatorname* { g c d } ( c , d )$  . Then  $q \mid p e ,$  and since  $\operatorname* { g c d } ( e , c ) = 1 , q \mid p ;$  thus Now  $c < k \leq a _ { 1 } ^ { a _ { 1 } + 1 } < B$  . Consequently  $p \leq c < B .$  In both cases we obtain  $p < B .$  This completes the induction, Now return to the original statement. Let x be good and  $i \geq 1$  . Take d to be the smallest good divisor of  $x ;$  then d is core. By the Claim, every prime factor of d is  $< B .$   $( d , a _ { i } ) > 1$  ; pick a prime  $p \mid \operatorname* { g c d } ( d , a _ { i } )$  . Then  $p \mid d \mid x$  and  $p \mid a _ { i }$  and  $p < B .$  Lemma 6. Let  $\bar { L } = B !$  (the factorial of B). For every  $x > 1$  we have good(x) . Proof. If good(x), then for any i Lemma 5 gives a prime  $p < B$  with Because p  $< B , p \mid { \cal L } ,$  SO  $p \mid x + L$  Hence gco  $\mathsf { l } ( x + L , a _ { i } ) \geq p > 1$  and  $\operatorname* { g o o d } ( x + L )$  . Conversely, suppose  $\operatorname* { g o o d } ( x + { \bar { L } } )$  . Then for any  $i ,$  Lemma 5 applied to gives a prime  $p < B$  with  $p \mid x + L$  and  $p \mid a _ { i }$  Since  $p \mid L ,$  we have  $p \mid ( x + L ) - L = x$  and  $p \mid a _ { i } .$  Also  $x > 1 \colon$  if  $x \leq 1$  then  $x = 1 ( { \mathrm { s i n c e } } x$  : is a positive integer), but then impossible. Therefore  $\operatorname* { g c d } ( x , a _ { i } ) > 1$  for all  $i ,$  so good(x). Construction of  $T$  and  $L .$  Since  $a _ { 1 }$  is good, Lemma 6 yields  $\operatorname { g o o d } ( a _ { 1 } + L )$   $a _ { N } = a _ { 1 } + L .$ </td></tr><tr><td colspan="3"> $a _ { 1 } + L \geq a _ { 1 }$  is a term of the sequence; let N be the index with and the sequence is strictly increasing,  $N > 1$  . Define</td></tr><tr><td> $T = N - 1 , \qquad L = B ! .$  Then  $T > 0$  and  $a _ { 1 + T } = a _ { 1 } + L .$  Lemma 7. For every  $n \geq 1$  we have  $a _ { n + T } = a _ { n } + L$ </td><td>Proof by induction on  $n .$  Base case Because the</td></tr><tr><td> $n = 1 \colon a _ { 1 + T } = a _ { 1 } + L$  sequence is strictly increasing,  $a _ { n + 1 }$ </td><td>by construction. Inductive step. Assume  $a _ { n + T } = a _ { n } + L .$  is good, so by Lemma  $6 , a _ { n + 1 } + L$  is good. Moreover  $a _ { n + T } = a _ { n } + L < a _ { n + 1 } + L .$ </td></tr><tr><td>Since  $a _ { n + T + 1 }$ </td><td>is the smallest good number larger than  $a _ { n + T }$  , we obtain</td></tr><tr><td></td><td> $a _ { n + T + 1 } \leq a _ { n + 1 } + L .$ </td></tr><tr><td>Suppose for contradiction that</td><td> $a _ { n + T + 1 } < a _ { n + 1 } + L . { \mathrm { S e t } } x = a _ { n + T + 1 } - L .$ </td></tr><tr><td></td><td></td></tr><tr><td></td><td> $a _ { n } < x < a _ { n + 1 } .$ </td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td>By the definition of</td><td>(or the contrapositive of its minimality) there exists</td></tr><tr><td> $a _ { n + 1 }$ </td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td> $\mathrm { g } \mathrm { \dot { c } d } ( x , a _ { i } ) = 1$ </td><td></td></tr><tr><td>. On the other hand,</td><td>is a term, hence good. Lemma 6 gives</td></tr><tr><td></td><td> $a _ { n + T + 1 }$ </td></tr><tr><td></td><td></td></tr><tr><td>because good  $( x + L )$  holds. Therefore</td><td></td></tr><tr><td></td><td> $\operatorname* { g c d } ( x , a _ { i } ) > 1$  for all  $i ,$  contradicting</td></tr><tr><td></td><td></td></tr><tr><td>Hence</td><td></td></tr><tr><td> $a _ { n + T + 1 } = a _ { n + 1 } + L$ </td><td> $a _ { ( n + 1 ) + T } = a _ { n + 1 } + L$ </td></tr><tr><td></td><td>, which is exactly</td></tr><tr><td></td><td></td></tr><tr><td>Conclusion. We have exhibited positive integèrs</td><td></td></tr><tr><td></td><td> $\acute { T }$  and  $L$  (depending only on</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td>for every</td><td></td></tr><tr><td> $a _ { n + T } = a _ { n } + L$   $n \geq 1$ </td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td>There exist positive integers T and L such that</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr></table>