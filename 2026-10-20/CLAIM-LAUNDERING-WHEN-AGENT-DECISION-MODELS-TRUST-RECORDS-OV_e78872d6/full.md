# CLAIM LAUNDERING: WHEN AGENT DECISION MODELS TRUST RECORDS OVER REALITY

Tina Khezresmaeilzadeh <sup>1</sup> Seyedarmin Azizi <sup>1</sup> Erfan Baghaei Potraghloo <sup>1</sup> Mehdi Kamal <sup>1</sup> Massoud Pedram <sup>1</sup>

## ABSTRACT

Agent decision models select actions in a single forward pass from inputs that combine environmental observations with the agent’s own record of previous actions. When an action fails silently, these sources can conflict: the record reports success, but the environment shows that the step did not take effect. We ask which source determines the next decision. Across Android screen traces, ALFWorld, and WebShop, we construct paired states that independently vary the observation, the logged action, and an explicit success claim. In all 12 configurations of four decision models and three domains, records have a larger average effect on commitment decisions than observations; after claimed-but-failed steps, the models still stop, advance, or purchase in 53–100% of cases. Yet the relevant evidence is often represented internally: linear probes on intermediate hidden states distinguish successful from failed steps on held-out apps, tasks, and goals (AUROC 0.77–1.00), and activation patching reveals increasing influence from records in later layers. We use this signal in EVIGATE, an intermediate-layer veto that requires no additional forward pass and adds 0.04–2.0% serving latency. Under injected silent failures, EVIGATE improves task success by 18.7 points in live ALFWorld agents, 5.0 in WebShop, and 5.1 in a phone simulator, and in WebShop and the simulator it also outperforms random vetoes at the same rate. Its effectiveness depends on representative calibration states: gates calibrated on offline counterfactual states do not transfer reliably to live agent trajectories.

## 1 INTRODUCTION

Agents spend most of their inference on small, repeated decisions: stop or continue, buy or keep looking, move on or retry. A new class of System-One decision models is built for exactly this workload (TypeSafe AI, 2026; Palmer, 2026; Marosi, 2026; Convai Innovations, 2026). Given a state and a typed question, such a model returns a probability for every predefined option in a single forward pass, at a fraction of the latency and cost of LLM generation. Jev is already served through OpenRouter (OpenRouter, 2026) and Cloudflare Workers AI and integrated with LangChain and Pydantic AI.

The problem: records can disagree with the world. A decision model does not watch the environment. It reads a state that the agent assembles: the goal, a record of what the agent did (LLM-written step summaries in T3A, ReAct thoughts in ALFWorld, logged clicks in WebShop) and the current observation. The record and the world can diverge. Actions fail silently: a tap misses, a click does not register, a command returns “Nothing happens.”. And records are written from the action’s intent, often before its effect can be checked. A 9B summarizer given T3A’s own prompt, which asks it to “be critical”, still reports success after 39% of actions that had no effect (§4). When the record and the observation disagree, the observation is right. A reliable decision layer must notice the mismatch before it takes an irreversible step. One that follows the record instead turns the agent’s claim into the decision; we call this claim laundering (Figure 1).

This matters in practice. Replacing an LLM action selector with Jev cut the cost of every decision but lowered endto-end success on AndroidWorld by 13–18 points, with premature completion among the leading causes (Ling et al., 2026b). Across LLM agents, 45–76% of failures on τ<sup>2</sup>- bench and AppWorld are false successes, where the agent reports completion the environment does not show (Advani, 2026). These studies describe the symptom. We ask where it comes from in the component that makes the decision, and how to detect it.

What we do. We answer four questions, each in its own section.

1. Do decision models detect record–state mismatches? (§4) We build paired states that change only the record or only the observation, in three domains, and test seven open decision models, hosted Jev and their base LLMs. They do not: the record moves the decision more than the observation for all four models tested in every domain, and for six of the seven open models on phones.

2. Does the model know better than it says? (§5) Yes. A linear probe on the hidden state detects whether the step worked, often where the model’s own output is near chance.

3. Why does the output ignore it? (§6) Activation patching shows the observation reaching the decision position by mid-depth, while record-dependent influence becomes dominant in the last layers.

4. Can we use this to detect mismatches? (§7–§8) EVI-GATE reads the middle layer and vetoes unsupported stop, proceed or buy decisions. We test it offline and in three live agents, against random vetoing at the same rate. We then ask whether simply fine-tuning the model would be enough (§9).

## Contributions.

• A failure and how to measure it. We define record– state mismatch for agent decision layers and measure it with counterfactual pairs in three domains. After a failed step whose record reports success, the models still stop, proceed or buy in 53–100% of cases. The bias is already present in the base LLM, and decision tuning does not remove it.

• Where it happens. The mismatch is encoded in the hidden state (probe AUROC 0.77–1.00 on held-out groups) and outweighed by record-dependent influence in the last layers.

• A detector that works in live agents. EVIGATE costs one dot product and raises task success in all three domains, provided it is calibrated on states from the deployed agent itself. Fine-tuning removes the bias too, but only in the domains it is trained on.

## 2 BACKGROUND

Decision models. A request contains a state (text or JSON) and typed questions, for example a choice among named options. The model reads state and questions in one sequence and returns a probability for every option in one forward pass. The open models differ in how they read the answer. decider (Marosi, 2026) fine-tunes Qwen3.5 base models (Qwen Team, 2026) (Gemma-4 for its 12B model) and reads option-letter logits at an answer position. Kev (Palmer, 2026) keeps a Qwen3.5 backbone frozen, adds a rank-16 LoRA and scores options with a pointer head. Laya (Convai Innovations, 2026) is a ModernBERT encoder (Warner et al., 2025). The hosted Jev model is closed-weight; all three open families serve the same API.

Goal: Turn on Wi-Fi   
Action history:   
Step 3: tapped ’Wi-Fi toggle’   
Agent note: successfully completed: Turn on Wi-Fi.   
Current screen (Settings > Network):   
Wi-Fi toggle: Off   
Text: Search settings ...  
Decision model output: p(stop) = 0.65 ⇒ declares done EVIGATE (layer 20): p(step worked) = 0.00 ⇒ veto  
Figure 1. A record–state mismatch (real instance from our diagnostic set). The agent’s record (red) says the task is done; the screen (blue) shows it is not. The decision model follows the record. A probe on its own hidden state, trained on other goals, reads the screen correctly.

Decision models in agents. Ling et al. (2026b) integrate Jev into two agents, and we reuse both request formats verbatim. In the Android agent T3A (Rawles et al., 2025), the state is T3A’s full prompt (goal, LLM-written step summaries, list of screen elements) and one question chooses among twelve operations, including complete. In Re-Act (Yao et al., 2023) on ALFWorld (Shridhar et al., 2021), the state is the ReAct transcript and the question chooses the next command. We apply the same pattern to WebShop (Yao et al., 2022).

## 3 MEASURING RECORD–STATE MISMATCH

Problem formulation. A decision model receives a goal, an agent-generated action record, the latest environmental observation, and a typed question. We write this input as $\boldsymbol { x } = ( g , r , o , q )$ and denote by $p _ { \theta } ( a ^ { \star } \mid x )$ its probability for the decision $a ^ { \star }$ that commits to the preceding step having succeeded. The observation o describes the current environment; the record r describes what the agent believes or says it did. These are not interchangeable. An attempted action may be logged even if it has no effect, and an agent may explicitly claim success despite an unchanged environment. We call a case a record–state mismatch when the record suggests an achieved outcome that the observation does not support. An explicit success claim creates a direct contradiction; a truthful log of an attempted action may instead become a misleading cue when the model treats the attempt as evidence of success.

We distinguish two parts of the record: a logged action, which states that an operation was attempted, and an explicit success claim, which states that its intended outcome occurred. We say a step took effect when the environment reached the intended post-action state. The decision $a ^ { \star }$ is complete on phones, advancing past a required subgoal in ALFWorld, and buy now in WebShop. These are consequential commitments, although advancing in ALFWorld is not literally an episode-terminal action.

Counterfactual pairs. To isolate the influence of each input, we construct contrast sets (Gardner et al., 2020; Kaushik et al., 2020) around decision points at which a step should complete a task or a required subgoal. Each set contains five variants of the same example, with the goal, question, and unrelated context held fixed:

• DONE: the action is logged and the observation shows its intended effect, without an explicit success claim.

• DONE+CLAIM: the same successful transition, with an explicit success claim.

• HONEST: the action has not been attempted, is not logged, and the observation shows the preceding state.

• LOGGED: the action is logged but has no effect; the observation still shows the preceding state.

• LOGGED+CLAIM: the same failed transition, with an explicit but incorrect success claim.

The variants support three controlled comparisons. Writing $L ( v ) = \log p _ { \theta } ( a ^ { \star } \mid x _ { v } )$ , we measure

$$
\Delta _ { \mathrm { o b s } } = L ( \mathrm { D O N E } ) - L ( \mathrm { L O G G E D } ) ,\tag{1}
$$

$$
\Delta _ { \mathrm { l o g } } = L \big ( \mathrm { L o G G E D } \big ) - L \big ( \mathrm { H o N E S T } \big ) ,\tag{2}
$$

$$
\Delta _ { \mathrm { c l a i m } } = L ( \mathrm { L O G G E D + C L A I M } ) - L ( \mathrm { L O G G E D } ) .\tag{3}
$$

The first changes the observed outcome while keeping the action log fixed; the second adds an attempted action without changing the observed outcome; the third adds a success assertion to a failed attempt. Thus a positive effect increases the model’s support for committing to completion. Effects are differences in log probabilities: for example, +0.7 corresponds to approximately twice the probability, not a 70- percentage-point increase. We also compare DONE+CLAIM with LOGGED+CLAIM when testing whether a detector distinguishes success from failure despite an identical success claim.

Three domains. Table 1 summarizes 6,175 states from 1,235 decision points. Phones: we use screen-element descriptions and action histories from real AndroidControl trajectories (Li et al., 2024), spanning 245 apps. The screen after a task-completing action supplies the successful observation (we assume, as the dataset intends, that the demonstrated final action completes the task); the preceding screen supplies the unchanged observation in the counterfactual failure. The record consists of step summaries, with an optional completion claim. The decision model reads the textual representation of a real screen, not raw image pixels. ALFWorld: at up to three state-changing steps along expert trajectories, we use the environment’s “Nothing happens.” feedback for unsuccessful attempts and vary whether the ReAct transcript contains the attempted step and a success thought. WebShop: the agent reaches the target product through environment interactions and attempts to select the required options; for failed variants, those clicks remain recorded although the current page shows the options unselected. The construction and domain-specific prompts are detailed in Appendix A.

Table 1. The three domains: real Android screens from Android-Control (Li et al., 2024) (245 apps), unseen ALFWorld tasks, and WebShop goals. Request formats follow Ling et al. (2026b).
<table><tr><td>Domain</td><td>Terminal decision Silent failure</td><td></td><td>States</td></tr><tr><td>Phones</td><td>complete</td><td>screen unchanged</td><td>3,200</td></tr><tr><td>ALFWorld proceed</td><td></td><td>&quot;Nothing happens.&quot;</td><td>1,475</td></tr><tr><td>WebShop</td><td>buy now</td><td>options not selected</td><td>1,500</td></tr></table>

These paired failures are controlled counterfactuals, not estimates of how often silent failures occur during unmodified deployment. This distinction allows us to measure which input drives a decision without confounding the result with the natural frequency of action failures.

Models and evaluation. Four models are evaluated in all three domains: decider-4b, Kev-4B, decider-12b, and hosted Jev (jev-1.13-20260917). On phones, we additionally evaluate decider-2b, Kev-0.8B, Kev-9B, and Laya, as well as the base and chat Qwen3.5-4B models underlying the 4B decision models. We preserve the agent request formats of Ling et al. (2026b); only the targeted input component changes within each comparison. We report the proportion of LOGGED+CLAIM cases in which the model chooses a<sup>⋆</sup>, alongside mean paired log-probability effects. Confidence intervals are obtained by bootstrapping over episodes or evaluation groups rather than treating variants of a single decision point as independent.

Hidden-state probes. For each open model, we extract the hidden state at the answer position every two layers and fit an ℓ<sub>2</sub>-regularized logistic probe to predict whether the step took effect. All variants of a decision point remain in the same split; evaluation holds out entire apps, tasks, or goals using GroupKFold. Layer selection takes place by cross-validation within training groups, not on test groups. We compare the probe’s AUROC with AUROC from the model’s own output score, both on all states and on the subset whose records claim success. The latter comparison tests whether the observation remains decodable when the claim itself provides no information about the label.

Table 2. Which input moves the decision. Wrong: how often the model still stops, proceeds or buys after a failed step whose record claims success (LOGGED+CLAIM). Obs., Logged, Claim: how much the observation, the logged step and the explicit claim raise log p of that option. Bold: the largest of the three. In every row the bold record effect exceeds the observation effect with a paired 95% CI above zero (Appendix Table 7).
<table><tr><td>Model</td><td>Wrong</td><td>Obs.</td><td>Logged</td><td>Claim</td></tr><tr><td>Phones</td><td></td><td></td><td></td><td></td></tr><tr><td>decider-4b</td><td>95%</td><td>+.05 +.09</td><td>+1.10</td><td>+.80 +.32</td></tr><tr><td>Kev-4B decider-12b</td><td>96%</td><td>+.21</td><td>+.20</td><td></td></tr><tr><td>hosted Jev</td><td>84% 88%</td><td>+.61</td><td>+1.10 +2.31</td><td>+.56 +.95</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Embodied (ALFWorld) decider-4b</td><td>90%</td><td>+1.19</td><td>+2.60</td><td>+1.28</td></tr><tr><td>Kev-4B</td><td>80%</td><td>+.28</td><td>+.25</td><td>+.32</td></tr><tr><td>decider-12b</td><td>97%</td><td>+.40</td><td>+1.20</td><td>+.90</td></tr><tr><td>hosted Jev</td><td>95%</td><td>+.98</td><td>+4.41</td><td>+1.16</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Web (WebShop)</td><td></td><td></td><td></td><td></td></tr><tr><td>decider-4b</td><td>100%</td><td>+.02</td><td>+3.71</td><td>+.07</td></tr><tr><td>Kev-4B</td><td>53%</td><td>+.11</td><td>+.31</td><td>+.23</td></tr><tr><td>decider-12b</td><td>99%</td><td>-.09</td><td>+1.20</td><td>+.98</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>hosted Jev</td><td>99%</td><td>+.20</td><td>+2.75</td><td>+.57</td></tr></table>

## 4 DECISION MODELS FAVOR RECORDSOVER OBSERVATIONS

Records dominate the controlled comparisons. Table 2 reports the four models evaluated in every domain. In all 12 model–domain configurations, the largest mean increase in the probability of the commitment decision comes from a record component (the logged action or explicit claim), not the environmental observation. This is a comparison of causal input edits within matched examples; it does not imply that every individual decision follows the record. The comparison is statistically reliable: in every configuration the larger record effect exceeds the observation effect with a paired 95% bootstrap interval above zero, including the closest case, Kev-4B in ALFWorld (claim minus observation +0.04 [+0.01, +0.07]). When the false claim is present, changing the observation from failed to successful moves the decision by at most +0.24, and by +0.035 or less for decider-4b and decider-12b on phones and WebShop (Appendix Table 7).

On phones, an attempted action recorded without its intended effect increases the log probability of complete by +0.20 to +2.31, whereas showing that the screen actually reached the target state contributes only +0.05 to +0.61. With both the unsuccessful action and a false success claim in the record, these models stop in 84–96% of cases. In ALFWorld, the observation explicitly says “Nothing happens.” Nevertheless, the models advance past the unfinished subgoal in 80–97% of LOGGED+CLAIM cases. In WebShop, the current page shows the required options unselected, but the logged clicks raise the log probability of purchasing by +0.31 to +3.71, compared with an observation effect between −0.09 and +0.20. The resulting wrong-purchase rate is 53–100% across models. Hosted Jev exhibits the same qualitative pattern as the open models, so it is not confined to one open-model architecture or answer head.

The effect persists with LLM-written records. The preceding explicit claims are controlled templates. To test whether a realistic summarization component produces misleading records, we run Qwen3.5-9B with T3A’s own step-summary prompt, providing the before- and after-step screen descriptions in the phone failure condition. Despite access to the unchanged observation, 39% of these generated summaries (78 of 200) report that the action succeeded (Appendix A). Conditional on such a false summary, decider-4b, Kev-4B, and hosted Jev choose complete in 77%, 94%, and 80% of cases, respectively; with an honest summary, the corresponding rates are 26%, 48%, and 27%. Thus an incorrect LLM-written record increases premature completion by 46–53 percentage points in these paired evaluations. The 39% figure measures summarizer behavior on our failedstep test cases, not the prevalence of failure in deployed Android agents. In the live ALFWorld and WebShop experiments (§8), the records are instead generated by the acting agents themselves.

Neither scale nor decision tuning consistently removes the bias. The additional phone models exhibit the same general vulnerability (Appendix Table 8). For models up to 4B, including the encoder Laya, the explicit claim has a greater mean effect than the observed screen transition. Kev-9B is a partial exception: the screen contributes +0.34 compared with +0.27 for the log and +0.24 for the claim, yet the model still completes after 65% of claimed failures. We also compare the decision-tuned models with their Qwen3.5-4B backbone (Appendix Table 11). The base model already responds more strongly to the claim (+0.51) than to the observed outcome (+0.02; not significant). In these comparisons, decision tuning does not eliminate that behavior: Kev’s logged-action effect changes from +0.15 to +0.21, while decider’s increases to +1.12. The results are consistent with a preference for action records that predates decision-specific tuning, although they do not by themselves identify its training-time cause.

The behavioral evidence establishes a failure of evidence use, rather than simply a lack of information: models are given an observation that contradicts the record, but their selected commitment is disproportionately driven by what the record says. We next test whether the contradiction is recoverable from the representations that produce those decisions.

Table 3. Detecting whether the step worked (AUROC on held-out apps, tasks or goals; 0.5 is chance). Head: the model’s own output. Probe: a linear probe on its hidden state. All: all test states. Claimed: only states whose record claims success, so that only the observation differs. Jev is closed-weight and cannot be probed.
<table><tr><td></td><td colspan="2">All</td><td colspan="2">Claimed</td></tr><tr><td>Model</td><td>Head</td><td>Probe</td><td>Head</td><td>Probe</td></tr><tr><td>Phones decider-4b Kev-4B</td><td>.63 .65</td><td>.77 .80</td><td>.51 .55</td><td>.74 .79</td></tr><tr><td>decider-12b hosted Jev</td><td>.69 .75</td><td>.80 一</td><td>.61 .68</td><td>.77 一</td></tr><tr><td>Embodied (ALFWorld) decider-4b</td><td>.92</td><td>1.00</td><td>.96</td><td></td></tr><tr><td>Kev-4B</td><td>.78</td><td>1.00</td><td>.80</td><td>1.00</td></tr><tr><td>decider-12b</td><td>.72</td><td>1.00</td><td></td><td>1.00</td></tr><tr><td>hosted Jev</td><td></td><td></td><td>.72</td><td>1.00</td></tr><tr><td></td><td>.83</td><td>一</td><td>.89</td><td>一</td></tr><tr><td>Web (WebShop)</td><td></td><td></td><td></td><td></td></tr><tr><td>decider-4b</td><td>.60</td><td>1.00</td><td>.54</td><td>1.00</td></tr><tr><td>Kev-4B</td><td>.72</td><td>1.00</td><td>.65</td><td></td></tr><tr><td>decider-12b</td><td></td><td></td><td></td><td>1.00</td></tr><tr><td></td><td>.55</td><td>1.00</td><td>.53</td><td>1.00</td></tr><tr><td>hosted Jev</td><td>.54</td><td></td><td>.52</td><td></td></tr><tr><td></td><td></td><td>一</td><td></td><td></td></tr></table>

## 5 HIDDEN STATES RETAIN EVIDENCE OF FAILURE

The hidden representation is more informative than the final score. Table 3 compares two ways to predict whether the last step took effect: the decision model’s own commitment score (Head) and a linear readout of its internal state (Probe). Both are evaluated on held-out apps, tasks, or goals. The most diagnostic setting is Claimed: successful and unsuccessful states contain the same success assertion, so a classifier cannot obtain the label simply by detecting optimistic record language. It must exploit information associated with the environment’s actual state.

On phones, where the action outcome must be inferred from screen-element descriptions, the open-model heads reach only 0.51–0.61 AUROC on claimed states. Linear probes reach 0.74–0.79, improving discrimination by 0.16–0.24 AUROC points. On WebShop, the heads achieve 0.53–0.65 while probes reach 1.00. In ALFWorld, the heads perform better (0.72–0.96), but the probes again reach 1.00. The hosted Jev outputs exhibit the same limitations in some settings (0.68 AUROC on claimed phone states and 0.52 on claimed WebShop states), although Jev’s hidden states are unavailable for probing. Across all states, open-model probe AUROC ranges from 0.77 to 1.00.

Separating useful representations from obvious textual cues. Perfect probing on ALFWorld and WebShop should not be mistaken for evidence of sophisticated hidden reasoning. Their failures are accompanied by distinctive, consistent textual traces; a classifier trained directly on the input text also reaches 1.00 AUROC. The important observation in those domains is narrower: the outcome is readily available, yet the decision head often fails to use it. Phones provide the more demanding test. Failure is implicit in longer screen descriptions, and the text-only classifier reaches 0.59 AU-ROC, compared with 0.77–0.80 for the hidden-state probes on all states. Thus the intermediate representations expose a more useful signal than this text baseline or the model’s own commitment score, including on held-out apps.

Where the signal appears and how well it transfers. In the 32-layer 4B models, phone outcome information becomes linearly readable around layer 16, with the strongest probes at roughly 55–75% of network depth (Appendix Figure 4). A probe trained on only ten labelled phone episodes already outperforms the model’s final score (Appendix Figure 5). But representation quality does not make calibration domain-independent: training on synthetic template states and evaluating on real phone-screen descriptions yields only 0.63 AUROC. This shortfall motivates fitting the eventual detector on labelled states representative of the target agent, rather than relying on mismatched synthetic examples.

These findings establish a decodability gap: relevant evidence about action outcomes is present in an intermediate representation but is not reliably reflected in the model’s output probability. Linear decodability alone does not show how the model uses that information or why its output favors the record. We therefore turn to activation patching (§6) to locate the layerwise contributions to the decision, before using an intermediate-layer readout as a gate (§7).

## 6 WHY THE OUTPUT IGNORES IT

To find where the record takes over, we use activation patching. We run a state, replace the hidden state at the decision position in one layer with the one from its paired state (e.g. LOGGED in place of LOGGED+CLAIM), and measure how much of the difference in log p this restores. Patching costs 19 forward passes per pair, so we run it on decider-4b over 100 decision points per domain (Figure 2) and repeat it on decider-12b (Appendix E). The pattern is the same in every domain:

• The observation arrives at mid-depth. At layer 24 of 32, patching restores 60% (phones) and 72% (embodied) of the observation’s effect.

• The record arrives late. At layer 24 the claim is only 25% (phones) and 21% (embodied) present, and a logged web click 12%. By layer 30 these rise to 62%, 46% and 61%.

![](images/97e1ec9438ab2ab411bfe49c0321de44ef09ec54bceaf4cb4194520d3d5450ae.jpg)  
Figure 2. Activation patching on decider-4b: the share of each input’s effect on log p of the terminal option that is restored by copying the decision-position state from the paired state at a single layer (mean ±95% CI). The observation is present by mid-depth; the logged step and the claim arrive in the last layers. We keep pairs whose log p differs by more than 0.2. In WebShop the page barely moves the decision (+0.02, Table 2), so there is no observation effect to locate and only the logged-click curve (n = 100) is shown.

• Nothing is decided before layer 16: patching earlier has no effect.

The larger decider-12b (48 layers) shows the same order in all three domains: by layer 32, patching the observation restores 46–66% of its effect, while the logged step restores 10–42% and the claim 22–38%; the record reaches 63–85% only by layer 42 (Appendix Table 14). Record-dependent influences thus become increasingly dominant in later layers, even though outcome information is already available in the intermediate representation at the decision position. The record does not travel alone, however. Removing the record’s own tokens from the last layers (and keeping the rest of the sequence) leaves the decision unchanged, because by then the record has spread into the tokens that follow it: later pages, the question and the options (Appendix D). This is why reading the middle layer works better than editing the late computation: on a synthetic diagnostic set, steering the answer position raised accuracy on claimed-but-unfinished states from 0.48 to 0.70 at best, while simply reading the middle-layer probe reached 0.89.

## 7 EVIGATE: DETECTING THE MISMATCH

The detector. Let $h _ { \ell }$ be the decision model’s hidden state at layer ℓ and at the position where it reads its answer; the forward pass computes it anyway. EVIGATE keeps a linear readout $\boldsymbol { s } ~ = ~ \sigma ( \boldsymbol { w } ^ { \top } \tilde { h } _ { \ell } + \boldsymbol { b } )$ on the standardised state, an estimate of the probability that the last step took effect. When the model proposes a terminal option and $s < \tau .$ , the gate vetoes it and the agent takes the best nonterminal option instead (Algorithm 1, Figure 3). It needs no extra forward pass and adds 0.04–2.0% latency in decider’s production engine and Kev’s serving path, against 56–98% for a second question or contrastive decoding (Appendix F).

![](images/573dab0454ba7397837825bdbf6250210669b53863700c53328ec572f3129552.jpg)  
Figure 3. EVIGATE on an illustrative phone step. The record says Wi-Fi is on; the screen shows it is off. The output follows the record and proposes stop. The gate reads the decision-position state at a middle layer ℓ of the same forward pass, before recorddependent influence dominates the last layers, and vetoes. It acts only on terminal options. Probabilities are illustrative.

Fitting it. The readout $( w , b , \ell )$ is fit on states labelled by whether the step took effect. The labels come free from the environment: an option is selected or not, an action returned “Nothing happens.” or not.

Choosing the threshold. τ trades two errors: letting a failed step through, and vetoing a step that worked. We choose the τ that maximises balanced accuracy on held-out calibration states.

## 7.1 Offline evaluation

Table 4 compares EVIGATE with the two detectors that need no training. Retuning the threshold on the model’s own score helps little, because the record has already corrupted its ranking. Asking a second yes/no question inherits the same bias on phones and web and works only with ALFWorld’s explicit failure phrase. EVIGATE is the best detector for every model and domain: it gains 0.10–0.12 balanced accuracy on phones and is near-perfect in ALFWorld and WebShop. On phones, the same holds for all seven models and for contrastive decoding (Shi et al., 2024) (Appendix Table 10). Simply deleting the record from the input is no substitute: the record carries progress information, so on phones the record-free model vetoes 93% of correct completions, and in ALFWorld and WebShop it still passes 6–10% of failed steps, against at most 0.5% for EVIGATE (Appendix D).

Algorithm 1 EVIGATE at one agent step   
Input: state x, question q, readout (w, b, ℓ), threshold τ   
p, h<sub>ℓ</sub> ← DECISIONMODEL(x, q) {one pass}   
a ← arg max<sub>o</sub> p(o)   
if a is terminal (stop, proceed, buy) then   
$\boldsymbol { s } \gets \sigma ( \boldsymbol { w } ^ { \top } \tilde { h } _ { \ell } + \boldsymbol { b } )$ {O(d)}   
if s < τ then   
a ← arg max<sub>o/∈terminal</sub> p(o) {veto}   
end if   
end if   
return a

Table 4. Gating terminal decisions on held-out groups: balanced accuracy (0.5 is chance). Own score: the model’s own probability of the terminal option, with a tuned threshold. Second q.: an added yes/no question asking whether the step worked. Wrong: failed steps that still end in a terminal decision, with the own score → with EVIGATE.
<table><tr><td>Model Own score Second q. EvIGATE Wrong</td></tr><tr><td>Phones</td></tr><tr><td>decider-4b .59 .60 .71 72% → 38%</td></tr><tr><td>Kev-4B .60 .61 .72 58% → 38% decider-12b .63 .62 .73 33% → 33%</td></tr><tr><td>hosted Jev .66 .69 一</td></tr><tr><td>Embodied (ALFWorld)</td></tr><tr><td>decider-4b .84 .97 1.00 18% → 0%</td></tr><tr><td>Kev-4B .69 .81 .99 21% → 0.5%</td></tr><tr><td>decider-12b .65 .75 1.00 36% → 0% hosted Jev .78 .99</td></tr><tr><td>一</td></tr><tr><td>Web (WebShop)</td></tr><tr><td>decider-4b .56 .52 1.00 68% → 0%</td></tr><tr><td>Kev-4B .68 .66 1.00 38% → 0.2%</td></tr><tr><td>decider-12b .54 .50 1.00 60% → 0%</td></tr><tr><td>hosted Jev .53 .76 一</td></tr></table>

## 8 DETECTING MISMATCHES IN LIVE AGENTS

Offline accuracy does not show that an agent finishes more tasks: a veto helps only if the agent then does something better. We therefore run full agents. Every condition uses the same tasks and random seeds, so differences are paired.

## Agents.

• Phones (simulator). Live AndroidWorld needs KVM emulation, which we did not have, so we use a text phone simulator with 24 tasks in 11 apps, 30 seeds per task. The decisive tap fails silently 40% of the time, and the simulator’s record generator still claims success after 70% of failed taps.

• ALFWorld and WebShop. As in the ReAct-J agent of Ling et al. (2026b), an LLM (Qwen3.5-9B in ALF-World, Qwen3.5-4B in WebShop) writes a thought before each action, and decider-4b chooses the action. The records are these thoughts. We use 134 unseen ALFWorld tasks and 200 WebShop goals, with one run per task (greedy thoughts and a fixed failure seed), so their confidence intervals reflect variation across tasks. The gate checks moving on after a state-changing action (ALFWorld; a veto retries the action) and buy now (WebShop).

To make silent failures frequent enough to measure, actions and option clicks fail silently 30% of the time, a stress test in the spirit of fault-injection benchmarks for web agents (Bai et al., 2026). Without injection, the mismatch still arises from the agent’s own optimistic records (§4).

Conditions (Table 5). No gate: the agent takes whatever the decision model chooses. Second question: before accepting a terminal decision, ask the decision model whether the last step worked (one extra call). Random veto: veto terminal decisions at random, as often as the gate it is compared with; any gain over it comes from which decisions a gate vetoes, not from vetoing more. EVIGATE is fit either on offline contrast sets, the paired states of §3, or on target-domain states: states recorded from the same agent on other tasks (ALFWorld: 148 states from 100 training games; WebShop: 480 states from 300 other goals).

## Results.

• Detection raises task success in every domain. EVI-GATE adds 5.1 points on the phone simulator, 18.7 in ALFWorld and 5.0 in WebShop. On phones and the web it also beats random vetoing at the same rate (+4.7, +9.0), so the gain comes from vetoing the right decisions. On the simulator it matches a 9B LLM judge without the judge’s extra calls (Appendix G).

• A second question is a strong baseline wherefailure is easy to see. In the simulator and in ALFWorld, whose failures are templated or stated explicitly, asking the decision model whether the step worked does as well as EVIGATE (83.2 vs. 82.4; 61.2 vs. 61.2), at the cost of an extra question per decision. In WebShop it fails (7.0, below no gate). EVIGATE is the only method that helps in all three domains.

• The gate must be fit on target-domain states. Fit on offline contrast sets, the gate fails in real agents: in

Table 5. Task success (%) of live agents; same tasks and seeds in every row. Random veto: vetoes at random, about as often as the EVIGATE row above it (rates in the note below). Bottom: paired differences with 95% CIs over tasks.
<table><tr><td>Condition</td><td>Phones*</td><td>ALFWorld</td><td>WebShop</td></tr><tr><td>No gate</td><td>77.2</td><td>42.5</td><td>8.5</td></tr><tr><td>Second question</td><td>83.2</td><td>61.2</td><td>7.0</td></tr><tr><td>EvIGATE, fit on offline contrast sets</td><td></td><td>58.2</td><td>2.5</td></tr><tr><td>EvIGATE, fit on target-domain states</td><td>82.4</td><td>61.2</td><td>13.5</td></tr><tr><td>random veto, matched rate</td><td>77.6</td><td>58.2</td><td>4.5</td></tr><tr><td colspan="4">EVIGATE – no gate</td></tr><tr><td></td><td>+5.1 [1.8, 8.9]</td><td>+18.7 [11.9, 26.1]</td><td>+5.0 [1.0, 9.5]</td></tr><tr><td colspan="4">EvIGATE — random veto</td></tr><tr><td></td><td>+4.7 [1.7, 8.2]</td><td>+3.0 [-1.5, 7.5]</td><td>+9.0 [4.5, 13.5]</td></tr></table>

<sup>∗</sup>Simulator. Target-domain states are simulator-generated template states from the other half of the tasks; in ALFWorld and WebShop they are states recorded from the same agent on separate training tasks. The gate fit on offline contrast sets was not run in the simulator. Random veto rates: phones, the gate’s measured rate (21.8% and 9.1% vs. 21.8% and 8.0% for EVIGATE on the two task halves); WebShop, the gate’s measured rate (63.6% vs. 67.1%); ALFWorld, the failure rate among gated steps on the training games (78.6% vs. 75.5%).

WebShop it vetoes 95% of purchases and success falls to 2.5%; in ALFWorld it is no better than random. Live records look different from offline ones (a thought at every step, other page sequences), and a few hundred states from the agent itself fix it.

• Recovery limits the gain. After a vetoed purchase the WebShop agent often wanders instead of selecting the missing option, so it buys less (46 vs. 78 purchases). In ALFWorld, 80% of gated steps follow a real failure, so retrying is almost always right and random vetoing recovers most of the gain. A detector is most valuable where vetoes are rare and must be precise, and only if the agent has a good next action.

The same readout can also pick the best of several attempts: in best-of-16 sampling on WebShop it chooses a successful attempt 12.0 points more often than the decision model’s own judgement (Appendix I).

## 9 WHY NOT FINE-TUNE THE MODEL?

If the model knows, could we simply train its output to use what it knows? This is the standard remedy when a model trusts the wrong input: counterfactual data augmentation (Kaushik et al., 2020), training on conflicting contexts so a model follows its context over its memory (Li et al., 2023), synthetic data against sycophancy (Wei et al., 2023), and adversarial negatives for reward models (Zhao et al., 2025). We apply it to decider-4b: a LoRA adapter (rank 16, 0.8% of the parameters) trained on paired states with one label, whether the step took effect. A second loss term keeps the model’s preferences among the non-terminal options unchanged (details in Appendix H). We train three adapters: on phones only (2,755 states from 205 of the 245 apps); on all three domains with 1,000 states each (from training apps, games and goals); and on all three domains with all 2,755 phone states. All are tested on held-out apps, games and goals.

Table 6. Fine-tuning decider-4b on paired states. Wrong: failed steps whose record claims success that still end in a terminal decision. AUROC: the output’s detection of the failure on claimed states. Tests use held-out apps, games and goals. Last rows: accuracy on two of decider’s held-out general tasks.
<table><tr><td rowspan="2">Test domain</td><td rowspan="2"></td><td rowspan="2"></td><td colspan="3">Fine-tuned on</td></tr><tr><td>Base phones all 3</td><td></td><td>all 3 1k each all phones</td></tr><tr><td>Phones</td><td>Wrong</td><td>91%</td><td>47%</td><td>89%</td><td>47%</td></tr><tr><td rowspan="2">ALFWorld</td><td>AURÓC</td><td>.50</td><td>.84</td><td>.67</td><td>.82</td></tr><tr><td>Wrong</td><td>90%</td><td>91%</td><td>0%</td><td>0.3%</td></tr><tr><td rowspan="2">WebShop</td><td>AURÓC</td><td>.91</td><td>.92</td><td>1.00</td><td>1.00</td></tr><tr><td>Wrong</td><td>100% .46</td><td>100%</td><td>0%</td><td>0%</td></tr><tr><td></td><td>AURÓC</td><td></td><td>.48</td><td>1.00</td><td>1.00</td></tr><tr><td>HWU64</td><td>Acc.</td><td>84.5</td><td>84.3</td><td>84.8</td><td>85.0</td></tr><tr><td>DBpedia</td><td>Acc.</td><td>77.5</td><td>78.3</td><td>78.3</td><td>78.5</td></tr></table>

Table 6 gives three findings.

• In-domain, fine-tuning works. Trained on phones, the model stops after 47% instead of 91% of claimed failures on unseen apps. The screen now drives the decision (its effect rises from +0.02 to +1.34) and the claim barely matters (+0.74 → +0.08). On claimed states the output’s AUROC (.84) even exceeds the base model’s probe (.74). General accuracy is unchanged.

• It does not transfer across domains. The phone-trained model is unchanged in ALFWorld and WebShop. What it learns is how failure looks on a phone screen, not a general rule to trust the observation over the record.

• With enough datafrom every domain, one modelfixes all three. Trained on 1,000 states from each domain, the model makes no wrong decisions in ALFWorld and WebShop but loses most of the phone gain (89%). With all 2,755 phone states plus the same ALFWorld and WebShop states, it matches the phone-only model on phones (47%, AUROC .82) while staying near-perfect in ALFWorld (0.3%) and WebShop (0%), with general accuracy unchanged. Training on several domains therefore need not cost accuracy, as long as each domain has enough paired states.

Fine-tuning and detection therefore need the same thing, labelled states from the target domain, but differ in what they change: fine-tuning edits a model shared across every task it serves and needs thousands of paired states per domain, while EVIGATE leaves the model untouched and needs a few hundred labelled states from the deployment. Where the model’s weights cannot be changed, as for hosted Jev, neither applies directly, which motivates exposing intermediate states in hosted decision APIs. We evaluate fine-tuning offline only; whether its gains carry over to live agents, as EVIGATE’s do, remains to be tested.

## 10 RELATED WORK

Decision models in agents. Decision models answer typed questions in a single forward pass, making them faster and cheaper than LLM generation (TypeSafe AI, 2026; Marosi, 2026; Palmer, 2026). Thousands of public projects use Jev for routing, scoring, and action selection (Ling et al., 2026a), and replacing LLM-based action selection with Jev reduces per-decision costs (Ling et al., 2026b). Prior work has identified several failure modes: decision models favor option names over their definitions (Azizi et al., 2026a; Sun et al., 2026), adversarially renamed options can bypass guardrails (Azizi et al., 2026b), and premature completion decisions reduce end-to-end task success (Ling et al., 2026b). We study a different failure mode that requires no attacker: a decision model trusts the agent’s record of its actions over conflicting evidence from the environment.

Unreliable agent records. LLM agents often report success without corresponding evidence in the environment (Advani, 2026; Wang et al., 2026). Such inaccurate records can mislead downstream components. For example, safety monitors may approve risky actions when an upstream summary omits an “unverified” marker (Hu, 2026). We examine how such discrepancies affect downstream decision models, and why these models favor the records.

Probing and gating agents. Prior work shows that probes trained on agent hidden states can predict task success (Mammen et al., 2026) and episode failure (Ruan et al., 2026). In agent safety, Fayyazi et al. (2026) find that false statements often precede unauthorized actions, while lies follow them. They use hidden-state probes to predict and gate these actions, and steering to reduce subsequent lies. Unlike these approaches, which examine the agent’s own hidden states, we study decision models that consume the agent’s records. Using activation patching, we identify where these models begin to favor agent records over conflicting environmental observations. We then show that evidence of the conflict remains accessible in intermediate layers and can be used to detect mismatches. Finally, we show that the detector must be calibrated on states produced by the agent itself.

## 11 LIMITATIONS

To obtain enough mismatches for evaluation, our live agents run with injected silent failures (30% of actions in ALF-World and WebShop; in the phone simulator, 40% of first attempts and 20% of retries). Our offline experiments show the same behavior without injecting failures into live execution, using real trajectories and LLM-written summaries (§4); how often such mismatches arise naturally in deployment remains unknown. Because Android emulation was unavailable, our live phone agent runs in a text simulator, and real phone screens are evaluated offline only. In ALF-World and WebShop, failed actions leave explicit textual traces that give reliable labels but make mismatches easier to detect; the phone domain, built from real screens where failures are implicit, covers the harder case. Finally, our mitigation methods require labelled states from the target agent and access to model internals: EVIGATE relies on hidden activations, while fine-tuning additionally requires model weights. The hosted Jev model shows the same bias but can only be measured.

## 12 CONCLUSION

Agents record what they did, and the fast decision models that act on those records trust them over the world: on phones, in embodied tasks and in web shopping, in open models from 0.8B to 12B parameters and in hosted Jev. The models are not blind: the observation is present in their middle layers, but record-dependent influences dominate the final layers. Reading the middle layer detects the mismatch with one dot product and raises task success in all three domains when calibrated on the agent’s own states. Finetuning fixes the output too, but only in the domains it is trained on. For builders of agents, the practical advice is to keep unverified records out of the decision state where possible, and otherwise to check terminal decisions against what the model already encodes.

## REFERENCES

Advani, L. From confident closing to silent failure: Characterizing false success in LLM agents. arXiv preprint arXiv:2606.09863, 2026.

Azizi, S., Baghaei Potraghloo, E., and Pedram, M. Labels override definitions in JEV-style typed decision models. arXiv preprint arXiv:2610.02586, 2026a.

Azizi, S., Baghaei Potraghloo, E., and Pedram, M. One word opens the gate: The option-channel attack on typed decision models as agent guardrails. arXiv preprint arXiv:2610.12292, 2026b.

Bai, H., Wang, D., Chen, L., Hao, B., Shao, P., Yang, Y., He, Y., and Zhuang, C. StressWeb: A diagnostic benchmark for web agent robustness under realistic interaction variability. arXiv preprint arXiv:2604.16385, 2026.

Convai Innovations. Laya: open-weights typed decision models. https://huggingface.co/ convaiinnovations/laya, 2026.

Fayyazi, A., Kamal, M., and Pedram, M. The agent knew: Internal states forecast unauthorized actions and stop them, and the lies that follow. arXiv preprint arXiv:2610.17301, 2026.

Gardner, M., Artzi, Y., Basmova, V., Berant, J., Bogin, B., Chen, S., Dasigi, P., Dua, D., Elazar, Y., Gottumukkala, A., Gupta, N., Hajishirzi, H., Ilharco, G., Khashabi, D., Lin, K., Liu, J., Liu, N. F., Mulcaire, P., Ning, Q., Singh, S., Smith, N. A., Subramanian, S., Tsarfaty, R., Wallace, E., Zhang, A., and Zhou, B. Evaluating models’ local decision boundaries via contrast sets. In Findings of the Association for Computational Linguistics: EMNLP 2020, pp. 1307–1323, 2020.

Hu, Y. Silence is endorsement: Verification-status laundering in LLM agent pipelines. arXiv preprint arXiv:2609.20211, 2026.

Kaushik, D., Hovy, E., and Lipton, Z. C. Learning the difference that makes a difference with counterfactuallyaugmented data. In International Conference on Learning Representations, 2020.

Li, D., Rawat, A. S., Zaheer, M., Wang, X., Lukasik, M., Veit, A., Yu, F., and Kumar, S. Large language models with controllable working memory. In Findings of the Association for Computational Linguistics: ACL 2023, 2023.

Li, W., Bishop, W., Li, A., Rawles, C., Campbell-Ajala, F., Tyamagundlu, D., and Riva, O. On the effects of data scale on UI control agents. In Advances in Neural Information Processing Systems, Datasets and Benchmarks Track, 2024.

Ling, G., Xue, M., and Ye, Z. Jev in the wild: A data-driven analysis of the jev model’s functionality, applications and ecosystem. arXiv preprint arXiv:2609.30216, 2026a.

Ling, Y., Yu, S., Chen, Z., and Fang, C. When faster decisions do not make better agents: An empirical study of Jev in agent workflows. arXiv preprint arXiv:2610.13092, 2026b.

Mammen, P. M., Joswin, E., and Medicherla, S. Do agents know when they succeed? calibrating agent confidence from internal representations. arXiv preprint arXiv:2609.09448, 2026.

Marosi, M. decider: one-pass typed decisions with calibrated probabilities. https://github.com/ Mapika/decider, 2026.

OpenRouter. Jev on OpenRouter. https:// openrouter.ai/docs/guides/community/ jev, 2026.

Palmer, J. Kev: open jev-like decision models. https: //github.com/jaredpalmer/kev, 2026.

Qwen Team. Qwen3.5 model family. https:// huggingface.co/Qwen, 2026.

Rawles, C., Clinckemaillie, S., Chang, Y., Waltz, J., Lau, G., Fair, M., Li, A., Bishop, W., Li, W., Campbell-Ajala, F., Toyama, D., Berry, R., Tyamagundlu, D., Lillicrap, T., and Riva, O. AndroidWorld: A dynamic benchmarking environment for autonomous agents. In International Conference on Learning Representations, 2025.

Ruan, K., Huang, Z., Zhou, Z., Wei, Q., Lin, J., Wang, X., and Sun, H. Doomed from the start: Early abort of LLM agent episodes via a recall-controlled probe cascade. arXiv preprint arXiv:2607.06503, 2026.

Shi, W., Han, X., Lewis, M., Tsvetkov, Y., Zettlemoyer, L., and Yih, W.-t. Trusting your evidence: Hallucinate less with context-aware decoding. In Proceedings of NAACL, 2024.

Shridhar, M., Yuan, X., Cotˆ e, M.-A., Bisk, Y., Trischler,´ A., and Hausknecht, M. ALFWorld: Aligning text and embodied environments for interactive learning. In International Conference on Learning Representations, 2021.

Sun, Y., Xu, J., Shi, J., and Yang, Z. Type-safe is not error-free: Typed decision models follow the option name, not the definition bound to it. arXiv preprint arXiv:2609.26758, 2026.

TypeSafe AI. Jev: a system one decision model. https:// docs.typesafe.ai/introduction, 2026. Released September 15, 2026.

Wang, B., Wang, Y., and Wu, Y. The unreliable progress bar: Can LLM agents reliably report task progress throughout execution? arXiv preprint arXiv:2609.08589, 2026.

Warner, B., Chaffin, A., Clavie, B., Weller, O., Hallstr´ om,¨ O., Taghadanian, S., Gallagher, A., Biswas, R., Ladhak, F., Aarsen, T., Cooper, N., Adams, G., Howard, J., and Poli, I. Smarter, better, faster, longer: A modern bidirectional encoder for fast, memory efficient, and long context finetuning and inference. In Proceedings ofthe Annual Meeting ofthe Associationfor Computational Linguistics, 2025.

Wei, J., Huang, D., Lu, Y., Zhou, D., and Le, Q. V. Simple synthetic data reduces sycophancy in large language models. arXiv preprint arXiv:2308.03958, 2023.

Yao, S., Chen, H., Yang, J., and Narasimhan, K. Web-Shop: Towards scalable real-world web interaction with grounded language agents. In Advances in Neural Information Processing Systems, 2022.

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., and Cao, Y. ReAct: Synergizing reasoning and acting in language models. In International Conference on Learning Representations, 2023.

Zhao, Y., Liu, H., Yu, D., Kung, S., Chen, M., Mi, H., and Yu, D. One token to fool LLM-as-a-judge. In Advances in Neural Information Processing Systems, 2025.

Appendix overview. Appendix A specifies data construction, model interfaces, probing, agent execution, and compute. Appendix C reports complete phone-domain results and additional baselines supporting §4 and §5. Appendix D tests whether removing record tokens from later layers changes decisions (§6). Appendix F reports the runtime overhead of EVIGATE (§7). Appendix G gives the complete phone-simulator comparison (§8); Appendix H details fine-tuning (§9); and Appendix I evaluates the same probe as a best-of-N verifier.

## A EXPERIMENTAL SETUP

Phone decision requests. We use the T3A decision interface of Ling et al. (2026b) unchanged. Each request has three parts.

• State: T3A’s action-selection prompt P, filled with the goal, the summaries of the earlier steps and the elements of the current screen, and wrapped as {"native t3a prompt": P}. The template is ACTION SELECTION PROMPT TEMPLATE in android world/agents/t3a.py.

• Question: a typed choice, “Following native t3a prompt, choose the next single action type to advance the goal.”

• Options: twelve action types, namely interaction (click, long press, input text, scroll, keyboard enter), navigation (open app, navigate back, navigate home), wait and answer, and the two terminal operations complete (“The requested goal is already achieved.”) and infeasible.

The second-question baseline adds one yes/no question to the same request: “Judging from the UI elements of the current screen in native t3a prompt, has the user goal already been achieved?”

LLM-written summaries. For each of the 200 episodes of the 1,000-state phone subset, Qwen3.5-9B rewrites the final step’s summary with T3A’s step-summary prompt (SUMMARIZATION PROMPT TEMPLATE in android world/agents/t3a.py), given the goal, the action, and the screen before and after the step; in the failure condition the after-screen equals the before-screen. Generation is greedy (no sampling, thinking disabled, at most 120 new tokens; the first line is kept). A summary counts as claiming success if the same model, asked “Does the summary state or imply that the action succeeded or that the goal is now accomplished? Answer yes or no.”, answers yes; 78 of the 200 failed-step summaries do. This automatic judgement was not verified by human annotation.

Phone screens and action records. We extract screen elements with T3A’s forest to ui elements and validity filter. Each element is represented as a JSON object containing its nonempty fields, including text, content description, hint, class, resource name, checked and selected states, and interaction flags (clickable, editable, scrollable). We exclude screen descriptions longer than 9,000 characters. From AndroidControl shards 0–1, we retain episodes containing at least two actions, with a final action that visibly changes the screen, is neither navigation nor wait, and does not answer a question-type goal. For a recorded action i, the logged-step template is “Step i: Verb[ text] in order to instruction.” An explicit success claim is added using one of four fixed sentences, such as “This completes the requested task.” The offline contrasts use actual before- and after-action screen descriptions, but the failed-action variant is constructed by pairing the action record with the unchanged earlier screen; it is not an independently observed failed Android execution.

ALFWorld and WebShop contrast sets. For ALFWorld, we run the environment’s hand-coded expert on all 134 valid unseen tasks; 117 are solved. We form five-state contrast sets at up to three state-changing expert steps per task. The claim is a ReAct-style thought: “think: Now I have done ‘a’. Next, I need to a<sup>′</sup>.” The candidate actions are the expert’s next command, a retry of the previous command, and two other valid commands. For WebShop, we use the 1,000-product catalogue with attribute-based goals and BM25 (rank bm25) search over WebShop’s indexed product text. For each goal, we open the target product and execute the required option selections in the environment. A failed variant retains the recorded clicks but presents the page with those options unselected. The four candidate operations are buy now, the first required option, description, and back to search.

Reading decision models. For decider, we take the answer-position hidden state (at “Answer: (”) and use the released scoring temperatures. For Kev, we use its <decide> representation and pointer head, with LoRA weights merged. For laya-multilingual, we average the [CLS] representations across input windows. We evaluate the Qwen3.5-4B base and chat models with decider’s prompts and option-letter scoring, without additional training.

Probes and uncertainty estimates. We extract representations from every second layer, standardize them using trainingfold statistics, and fit ℓ<sub>2</sub>-regularized logistic regression with C = 0.02. We use five-fold GroupKFold splits by app, ALFWorld task, or WebShop goal, so variants of the same evaluation group cannot appear in both training and testing. Within each training fold, a four-fold grouped cross-validation chooses the layer from the upper two-thirds of model depth. We estimate paired log-probability effects using 1,000 bootstrap resamples at the episode level; AUROC differences use 1,000 paired bootstrap resamples of apps. The phone text-only baseline is logistic regression on TF–IDF features from screen text, using word unigrams and bigrams.

Live ALFWorld agent. The state consists of a ReAct transcript with two in-context examples. Before each action, Qwen3.5-9B generates a thought greedily (at most 60 tokens), and decider-4b selects from valid commands using the ALFWorld instruction of Ling et al. (2026b). Each state-changing command independently has a 0.3 probability of injected failure: the environment does not execute the command and returns “Nothing happens.” The agent has at most 50 steps. We fit the gate on 148 labelled decision states collected from 100 separate ALFWorld train-split games, then evaluate on the unseen-task split.

Live WebShop agent. Qwen3.5-4B writes the search query and one thought for each page, while decider-4b chooses among clickable elements. Option clicks have a 0.3 probability of silently doing nothing; the episode ends after at most 15 steps. For gate training, we collect 480 states from the agent on WebShop goals 1000–1299; the label for purchase verification indicates whether all required options are selected. The gate acts at purchase decisions, not at every page visit. For both environments, the probe reads layer 20. The balanced thresholds are 0.158 for ALFWorld and 0.229 for WebShop. Random-veto controls in WebShop and the phone simulator veto at the measured rate of the gate they are compared with; in ALFWorld, at the failure rate among gated steps on the training games (79.7% targeted, 78.6% measured, vs. 75.5% for EVIGATE).

Phone simulator. We simulate 24 tasks spanning 11 apps. Each task starts at a home screen and requires opening the correct app and performing one state-changing action. The available options include two distractors, Navigate back, and Stop; screens contain four to six distractor elements, and option order is randomized. The decisive action does nothing with probability 0.4 on its first attempt and 0.2 on later attempts. After a failure, the simulated action record falsely states “goal – done.” with probability 0.7; otherwise it reports “The screen did not change.” The step limit is eight. This phone experiment is a text simulator with a stochastic record generator, not a live Android emulator or an LLM-written summary at each step. The gate reads layer 20 and is fitted on 384 template states from the other half of the task set.

Compute. Experiments use two NVIDIA RTX A6000 GPUs (48 GB each), bf16 computation, PyTorch 2.14, and Transformers 5.19. Approximate compute is 20 GPU-hours for offline measurements, 60 GPU-hours for live-agent experiments (including best-of-N), and 5 GPU-hours for fine-tuning.

## B UNCERTAINTY OF THE INPUT EFFECTS

Table 7 gives paired bootstrap confidence intervals (2,000 resamples of decision points) for the comparisons behind Table 2. Logged − Obs. and Claim − Obs. are within-point differences between a record effect and the observation effect; Obs. | claim is the effect of changing the observation from failed to successful while the false success claim stays in the record (DONE+CLAIM − LOGGED+CLAIM). In every configuration at least one record effect exceeds the observation effect with an interval above zero. With the claim present, the observation has little influence on the decision.

## C ADDITIONAL PHONE RESULTS

The main text compares the four models evaluated in all three domains. Here we report all seven open models evaluated on phones, varying model size (Kev-0.8B, 4B, and 9B; decider-2b, 4b, and 12b) and architecture (the encoder-based Laya). Unless otherwise marked, results use the complete set of 3,200 counterfactual states. Rows marked <sup>†</sup> use a 1,000-state subset.

Sensitivity to observations and records. Table 8 reports the effect of changing the observed outcome, the logged action, or the explicit success claim, each with other inputs fixed. The effect of adding a success claim exceeds the observation effect for every model up to 4B. This pattern does not hold uniformly at larger scales: Kev-9B gives greater weight to the

Table 7. Paired 95% bootstrap intervals for the effects in Table 2 (differences in log p of the terminal option). Decision points: phones 640 (decider-4b, Kev-4B) or 200 (decider-12b, Jev); ALFWorld 295; WebShop 300.
<table><tr><td>Model</td><td>Logged – Obs.</td><td>Claim – Obs.</td><td>Obs. | claim</td></tr><tr><td>Phones</td><td></td><td></td><td></td></tr><tr><td>decider-4b</td><td>+1.05 [0.94, 1.17]</td><td>+0.74 [0.69, 0.80]</td><td>+0.00 [−0.01, 0.01]</td></tr><tr><td>Kev-4B</td><td>+0.11 [0.09, 0.14]</td><td>+0.23 [0.22, 0.25]</td><td>+0.05 [0.04, 0.06]</td></tr><tr><td>decider-12b</td><td>+0.89 [0.68, 1.09]</td><td>+0.34 [0.23, 0.46]</td><td>+0.03 [−0.02, 0.09]</td></tr><tr><td>hosted Jev</td><td>+1.70 [1.33, 2.10]</td><td>+0.34 [0.22, 0.47]</td><td>+0.14 [0.11, 0.18]</td></tr><tr><td>ALFWorld</td><td></td><td></td><td></td></tr><tr><td>decider-4b</td><td>+1.41 [1.18, 1.65]</td><td>+0.09 [0.00, 0.19]</td><td>+0.24 [0.18, 0.29]</td></tr><tr><td>Kev-4B</td><td>-0.03 [−0.05, −0.00]</td><td>+0.04 [0.01, 0.07]</td><td>+0.23 [0.21, 0.25]</td></tr><tr><td>decider-12b</td><td>+0.80 [0.68, 0.93]</td><td>+0.50 [0.42, 0.58]</td><td>+0.12 [0.10, 0.13]</td></tr><tr><td>hosted Jev</td><td>+3.42 [3.06, 3.78]</td><td>+0.18 [0.09, 0.28]</td><td>+0.22 [0.19, 0.25]</td></tr><tr><td>WebShop</td><td></td><td></td><td></td></tr><tr><td>decider-4b</td><td>+3.69 [3.58, 3.81]</td><td>+0.05 [0.04, 0.06]</td><td>-0.00 [−0.00, −0.00]</td></tr><tr><td>Kev-4B</td><td>+0.20 [0.18, 0.22]</td><td>+0.12 [0.11, 0.14]</td><td>+0.06 [0.05, 0.07]</td></tr><tr><td>decider-12b</td><td>+1.29 [1.18, 1.39]</td><td>+1.07 [0.95, 1.18]</td><td>+0.01 [−0.01, 0.02]</td></tr><tr><td>hosted Jev</td><td>+2.54 [2.40, 2.69]</td><td>+0.37 [0.32, 0.42]</td><td>+0.02 [0.01, 0.02]</td></tr></table>

screen than to either record component considered separately, whereas decider-12b remains most sensitive to the action log.   
The evidence therefore supports a widespread, but not scale-invariant, preference for record cues.

Table 8. Paired effects on log p(complete) on phones, with 95% bootstrap confidence intervals. For models up to 4B, explicit claims exert a larger mean effect than screen evidence. The larger models differ: Kev-9B weights the observed outcome more than either record component individually, whereas decider-12b remains most sensitive to the logged action. Rows marked <sup>†</sup> use a 1,000-state subset.
<table><tr><td>Model</td><td>Observation</td><td>Logged step</td><td>Claim</td></tr><tr><td>decider-4b</td><td>+0.05 [.02,.08]</td><td>+1.10 [1.00,1.21]</td><td>+0.80 [.74,.85]</td></tr><tr><td>decider-2b†</td><td>+0.02 [−.02,.05]</td><td>+0.48 [.38,.58]</td><td>+0.79 [.72,.87]</td></tr><tr><td>Kev-4B</td><td>+0.09 [.07,.10]</td><td>+0.20 [.18,.22]</td><td>+0.32 [.31,.33]</td></tr><tr><td>Kev-0.8B†</td><td>+0.00 [.00,.01]</td><td>+0.02 [.01,.02]</td><td>+0.21 [.20,.22]</td></tr><tr><td>Laya†</td><td>-0.04 [−.13,.06]</td><td>-0.01 [−.06,.04]</td><td>+0.32 [.27,.38]</td></tr><tr><td>Kev-9B†</td><td>+0.34 [.28,.41]</td><td>+0.27 [.22,.32]</td><td>+0.24 [.22,.27]</td></tr><tr><td>decider-12b†</td><td>+0.21 [.11,.32]</td><td>+1.10 [.93,1.27]</td><td>+0.56 [.47,.66]</td></tr></table>

Output scores versus internal probes. bl i f d i h h h fi l h i k ff The Claimed columns contain both successful and failed states with success-claim language, requiring discrimination from other evidence. For decider-4b and Kev-4B, claimed-state AUROC rises from 0.51 and 0.55 at the output to 0.74 and 0.79 with a linear probe. The corresponding all-state gains are 0.14 and 0.15, with positive bootstrap confidence intervals. The smallest Kev model shows a smaller gain (0.04, 95% CI [−0.01, 0.09]), so improvement is not statistically established for every model. The input-text classifier reaches 0.59 AUROC, below the probe results for the 4B models.

Detection baselines. Table 10 compares the model’s original decision with thresholding its own score, asking a second yes/no question, contrastive decoding using a screen-removed pass, and EVIGATE. The gate has the highest balanced accuracy in each reported phone-model row. For decider-4b, the percentage of failed steps ending in complete decreases from 79% with the original choice to 38% with EVIGATE; for Kev-4B it decreases from 86% to 38%. These are offline decision-level measurements, not task-success rates.

Backbone comparison. Table 11 compares a Qwen3.5-4B backbone before and after decision-specific training on the same 1,000 phone states. Both backbone variants are more sensitive to an explicit claim than to the observation. Decision tuning changes the size of individual effects but does not consistently correct this imbalance. The comparison describes a shared pattern in the evaluated checkpoints; it does not establish which training stage originally caused the preference.

Table 9. Detecting whether the phone step worked (AUROC on held-out apps), as in Table 3. Gap: probe minus output on all states, with 95% CI. Text only: a classifier on the screen text alone.
<table><tr><td></td><td colspan="2">Output</td><td colspan="2">Probe</td><td></td><td></td></tr><tr><td>Model</td><td>All</td><td>Claimed</td><td>All</td><td>Claimed</td><td>Gap</td><td>Text only</td></tr><tr><td>decider-4b</td><td>.63</td><td>.51</td><td>.77</td><td>.74</td><td>+.14 [.11,.17]</td><td>.59</td></tr><tr><td>Kev-4B</td><td>.65</td><td>.55</td><td>.80</td><td>.79</td><td>+.15 [.12,.17]</td><td>.59</td></tr><tr><td> $\mathrm { d e c i d e r } { - 2 { \bf b } ^ { \dagger } }$ </td><td>.59</td><td>.50</td><td>.65</td><td>.61</td><td>+.06 [.02,.09]</td><td>.59</td></tr><tr><td> $\mathrm { K e v - 0 . 8 B ^ { \dag } }$ </td><td>.59</td><td>.50</td><td>.64</td><td>.59</td><td>+.04[−.01,.09]</td><td>.59</td></tr><tr><td> $\mathrm { L a y a ^ { \dagger } }$ </td><td>.52</td><td>.50</td><td>.62</td><td>.60</td><td>一</td><td>一</td></tr><tr><td> $\mathrm { K e v - } 9 \mathrm { B } ^ { \dag }$ </td><td>.74</td><td>.68</td><td>.83</td><td>.83</td><td>+.09 [.05,.12]</td><td>.59</td></tr><tr><td> $\mathrm { d e c i d e r } { - 1 2 \mathbf { b } } ^ { \dagger }$ </td><td>.69</td><td>.61</td><td>.80</td><td>.77</td><td>+.11 [.08,.14]</td><td>.59</td></tr></table>

Table 10. Gating the phone stop decision on held-out apps (balanced accuracy), as in Table 4, for all seven models and one more detector. Model’s choice: no gate. Own score and Second q.: as in Table 4. Contrastive: contrastive decoding against a pass without the screen (Sh et al., 2024). Bottom: failed steps that still end in complete, over both failed variants (LOGGED and LOGGED+CLAIM, 1,280 states); Table 2 counts LOGGED+CLAIM only, hence its higher rates.
<table><tr><td>Model</td><td>Model&#x27;s choice</td><td>Own score</td><td>Second q.</td><td>Contrastive</td><td>EVIGATE</td></tr><tr><td>decider-4b</td><td>.59</td><td>.59</td><td>.60</td><td>.60</td><td>.71</td></tr><tr><td>Kev-4B</td><td>.59</td><td>.60</td><td>.61</td><td>.61</td><td>.72</td></tr><tr><td>decider-2b</td><td>.57</td><td>.55</td><td>.59</td><td>.56</td><td>.64</td></tr><tr><td>Kev-0.8B</td><td>.53</td><td>.56</td><td>.58</td><td>.56</td><td>.61</td></tr><tr><td>Laya</td><td>.50</td><td>.53</td><td>.55</td><td>.49</td><td>.59</td></tr><tr><td>Kev-9B</td><td>.66</td><td>.66</td><td>.68</td><td>.65</td><td>.75</td></tr><tr><td>decider-12b</td><td>.60</td><td>.63</td><td>.62</td><td>.61</td><td>.73</td></tr></table>

Wrong stops, decider-4b: model’s choice 79% → own score 72% → EVIGATE 38% Wrong stops, Kev-4B: model’s choice 86% → own score 58% → EVIGATE 38%

Layer depth and label efficiency. Figure 4 reports AUROC as a function of representation depth. On phones, the outcome becomes more readily linearly decodable around the middle of the network. The very high ALFWorld and WebShop probe results should be interpreted alongside their fixed textual failure traces: a text-only classifier can also separate those outcomes. Figure 5 shows that a phone probe fitted on ten labelled episodes already outperforms the decision model’s output on the reported learning-curve setting. These results motivate a lightweight readout, without implying that it will transfer to new state distributions without recalibration.

## D REMOVING THE RECORD

Activation patching in §6 identifies where controlled input changes begin to affect the decision-position representation. We additionally test whether the record’s late influence can be prevented by removing its tokens from subsequent computation. For decider-4b, we preserve activations through layer ℓ − 1 and re-run layers $\ell , \ldots , 3 2$ after removing record tokens, while retaining the positions of the remaining tokens. Removed tokens are phone step summaries, ALFWorld action and thought entries, or WebShop action entries. We delete rather than mask these tokens because 24 of the 32 layers use recurrent linear attention, for which the standard attention mask would not implement the intended intervention. As a control, re-running with no token removal exactly reproduces the normal output.

Table 12 reports the resulting rate of premature commitment on failed steps in the evaluated ALFWorld and WebShop subsets. Removing record tokens beginning at layer 12 reduces premature advancement in ALFWorld, but removal beginning at layer 20 has no measurable effect in either domain. One plausible explanation is that the record’s influence has already propagated into retained representations by that depth. Because this intervention changes a partially computed sequence rather than cleanly isolating a single source of information, it does not by itself prove literal erasure or overwriting of observationa evidence. The direct mid-layer detector is more effective on these subsets.

Removing the record from the input. A simpler alternative is to delete the agent’s record before the decision mode sees it. We run decider-4b with all record tokens removed from the input (the same tokens as above, removed from layer

Table 11. The bias is already present in the base model: one Qwen3.5-4B backbone before (top) and after (bottom) decision tuning, 1,000 phone states. Left: paired effect on log p(complete). Right: AUROC for detecting the failure from the output and from a probe.
<table><tr><td>Model</td><td></td><td>Observation Logged step</td><td>Claim</td><td>Output Probe</td><td></td></tr><tr><td>Qwen3.5-4B-Base</td><td>+.02</td><td>+.15</td><td>+.51</td><td>.59</td><td>.65</td></tr><tr><td>Qwen3.5-4B (chat)</td><td>+.05</td><td>+.28</td><td>+.42</td><td>.61</td><td>.72</td></tr><tr><td>Kev-4B (LoRA)</td><td>+.09</td><td>+.21</td><td>+.32</td><td>.66</td><td>.77</td></tr><tr><td>decider-4b (full fine-tuning)</td><td>+.01</td><td>+1.12</td><td>+.78</td><td>.62</td><td>.75</td></tr></table>

![](images/3c8076f7debf852a04a68267fdf9061ef3700037290e654703748bcc26f2d1f6.jpg)  
Figure 4. Layerwise decoding of action outcomes. (a) Phone probe AUROC across relative depth, compared with the decision model’s output (grey band). (b) Model-output AUROC (hollow points) and probe AUROC (filled points) across domains; dashed grey indicates the input-text baseline. In ALFWorld and WebShop, standardized failure messages also allow a text-only classifier to reach AUROC 1.00.

1 on) and treat a terminal option as vetoed when the record-free model does not choose it. Table 13 compares this with EVIGATE on the paired states (phones: the 1,000-state subset). Without the record, the phone model loses track of what it has already done: it rarely chooses complete at all, vetoing 93% of correct completions, and its probability of completion no longer separates finished from failed steps (AUROC 0.53). In ALFWorld and WebShop, where the failure is visible in the observation, removing the record helps, but more failed steps still get through and more good steps are vetoed than with EVIGATE. The record carries progress information the model needs; the problem is that it also overrides the observation.

## E ACTIVATION PATCHING IN DECIDER-12B

We repeat the patching of Section 6 on decider-12b (Gemma-4, 48 layers) with the same procedure and the same filter (log p differs by more than 0.2), which leaves 48–94 pairs per curve. Table 14 shows the same order as in decider-4b: the observation reaches the decision position first and the record a few layers later. The gap is largest in WebShop and smallest on phones.

## F SERVING LATENCY

The decision-model forward pass already computes the hidden state used by EVIGATE. The gate extracts the answer-position representation at one intermediate layer and evaluates a linear readout, requiring approximately 2d floating-point operations for a dot product in d dimensions (about $5 \times 1 0 ^ { 3 }$ for $d = 2 5 6 0 )$ , versus approximately $2 . 4 \times 1 0 ^ { 1 \dot { 3 } }$ operations for a 3,000-token forward pass of the 4B model. No second model inference is required.

We integrate the readout into decider’s serving engine using torch.compile: CUDA graphs for inputs up to 2,048 tokens and a fused eager path for longer inputs. A hook at the selected layer exposes the probe score as an additional output column, without an additional kernel launch or synchronization in this implementation. Table 15 reports median latency (p50) on one RTX A6000 GPU using bf16, CUDA-event timing, and 30 iterations after warm-up. Inputs include actual T3A request states and shorter 280-token states. Added latency ranges from 0.04% to 2.01% in the reported decider conditions. By comparison, the second-question baseline requires 93–98% additional latency, and contrastive decoding requires 56–79%. In Kev’s serving path, median latency rises from 1207.8 to 1211.2 ms (0.29%). These are measurements of the specific implementations and hardware configurations tested, not a hardware-independent bound.

![](images/ba7be06f1abfcd10c9d5118e246ae76e0d26f6ea0df4907ad8907f6e591ac50c.jpg)  
Figure 5. How many labels the phone probe needs (held-out apps). ×: the output’s AUROC. Ten labelled episodes already beat the outpu for the decision-tuned models.

Table 12. Failed steps that still get a terminal decision when the record is removed from layer ℓ onward (decider-4b), compared with EVIGATE. Failed steps are both failed variants (LOGGED and LOGGED+CLAIM: ALFWorld 590, WebShop 600 states); Table 2 counts LOGGED+CLAIM only.
<table><tr><td></td><td>ALFWorld</td><td>WebShop</td></tr><tr><td>Normal pass</td><td>59%</td><td>99.7%</td></tr><tr><td>Record removed from layer 12</td><td>26%</td><td>93%</td></tr><tr><td>Record removed from layer 16</td><td>50%</td><td>99.5%</td></tr><tr><td>Record removed from layer 20</td><td>59%</td><td>99.7%</td></tr><tr><td>EVIGATE</td><td>0.5%</td><td>0%</td></tr></table>

## G COMPLETE PHONE-SIMULATOR RESULTS

The phone simulator contains 24 tasks, each evaluated with 30 matched random seeds (720 trials per condition). A task requires the agent to navigate to the appropriate app and perform its state-changing action; the record generator may incorrectly report that a failed action succeeded. The same task–seed pairs are used to compare conditions. Besides the main-text baselines, we consider contrastive decoding and a separate 9B LLM judge (Qwen3.5-9B) queried whenever the decision model proposes stopping.

Table 16 reports task success, paired success differences with 95% bootstrap intervals over tasks, premature-stop frequency, and inference calls. The balanced EVIGATE threshold increases success from 77.2% to 82.4% (+5.1 percentage points, 95% CI [1.8, 8.9]) while reducing false stops from 13.9% to 8.6%. The second question performs similarly in success (83.2%) but uses more calls per successful episode (6.10 versus 4.86 for EVIGATE). The separate 9B judge achieves 81.5% success, requiring 26% more calls and 29% more time per successful task than the ungated baseline in this implementation. Random vetoing offers little improvement. The simulator’s simplified task dynamics and template records limit how directly these results transfer to real phone interactions.

## H FINE-TUNING DETAILS

We fine-tune Mapika/decider-4b, a Qwen3.5-4B model with 32 layers (24 recurrent linear-attention layers and eight full-attention layers) and hidden dimension 2560. The base weights remain frozen. We train rank-16 LoRA adapters on all linear layers (scale 32, dropout 0.05), with 32.5 million trainable parameters, or approximately 0.8% of the model. We merge the adapter before evaluation.

Each training example is one of the five paired states and uses its domain’s ordinary decision question. We assign $y = 1$ to DONE and DONE+CLAIM and $y = 0$ to the other three variants. Let $p _ { T }$ denote the probability of the commitment option, and let q and $q _ { \mathrm { b a s e } }$ be the current and original distributions over non-commitment options. Training combines a binary

Table 13. Removing the record from the input vs. EVIGATE (decider-4b, paired states). Failed pass: failed steps that still end in a terminal decision. Good vetoed: steps that worked but whose terminal decision is blocked. Failed and good steps include both variants with and without the claim (phones: the 1,000-state subset, 400 failed and 400 good states; ALFWorld 590 and 590; WebShop 600 and 600).
<table><tr><td rowspan="2"></td><td colspan="2">Phones</td><td colspan="2">ALFWorld</td><td colspan="2">WebShop</td></tr><tr><td>Failed pass</td><td>Good vetoed</td><td>Failed pass</td><td>Good vetoed</td><td>Failed pass</td><td>Good vetoed</td></tr><tr><td>No gate</td><td>78%</td><td>0%</td><td>59%</td><td>0%</td><td>99.7%</td><td>0%</td></tr><tr><td>Record removed from the input</td><td>2%</td><td>93%</td><td>9.5%</td><td>13%</td><td>6.0%</td><td>7.5%</td></tr><tr><td>EVIGATE</td><td>25%</td><td>27%</td><td>0.5%</td><td>0%</td><td>0%</td><td>0.5%</td></tr></table>

Table 14. Activation patching in decider-12b (48 layers). Each cell is the share (%) of an input’s effect on the decision that is restored by patching the decision-position state at that layer. Obs = the screen or observation; logged = the logged step; claim = the false success claim. Before layer 26 every share is at most 2%.
<table><tr><td>Layer</td><td>Phones obs / logged / claim</td><td>ALFWorld obs / logged / claim</td><td>WebShop obs / logged / claim</td></tr><tr><td>28</td><td>23 / 6 / 5</td><td>7/4/2</td><td>5 /1/3</td></tr><tr><td>32</td><td>66 / 42 / 38</td><td>50 / 31 / 22</td><td>46 / 10 / 22</td></tr><tr><td>36</td><td>82 / 68 / 67</td><td>70 / 50 / 40</td><td>69 / 32 / 44</td></tr><tr><td>42</td><td>95 / 85 / 85</td><td>87 / 74 / 67</td><td>88 / 63 / 73</td></tr></table>

cross-entropy objective,

$$
\mathcal { L } _ { \mathrm { c o m m i t } } = - y \log p _ { T } - ( 1 - y ) \log ( 1 - p _ { T } ) ,
$$

with a distribution-preservation term,

$$
{ \mathcal { L } } _ { \mathrm { r e t a i n } } = \mathrm { K L } ( q _ { \mathrm { b a s e } } \parallel q ) .
$$

The second term discourages changes to preferences among non-commitment options while the first trains the commitment score to reflect the outcome.

We use AdamW with learning rate $1 0 ^ { - 4 } , \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 5$ , zero weight decay, 20 warm-up steps followed by cosine decay, batch size eight, one epoch, gradient clipping at 1.0, bf16, and random seed 0. Training uses one RTX A6000 GPU. In the phone-only condition, 2,755 states from 205 apps are used for training, and 445 states from other apps are held out. In the three-domain condition, training draws 1,000 states per domain from phone training apps, ALFWorld train-split games (excluding the 117 solved test games), and WebShop goals numbered 3000 and above (excluding test goals). A third condition uses all 2,755 phone training states with the same 1,000 ALFWorld and 1,000 WebShop states, with identical settings, and is evaluated on the same held-out apps, games and goals. To check for broad changes in performance, we also evaluate the first 400 examples from decider’s held-out HWU64 and DBpedia-L2 tasks.

## I USING THE PROBE AS A BEST-OF-N VERIFIER

To test whether the probe is useful beyond binary vetoing, we use it to rank complete WebShop attempts. For each of 50 goals, the ungated live agent samples 16 trajectories, generating thoughts at temperature 0.8. For each candidate attempt that reaches a purchase, we evaluate the decision model’s own answer to “is the purchased product correct?” and the probe score, which was trained on other goals to predict whether required options were selected. We compare their ability to select a successful attempt, along with random selection and an oracle that knows the evaluation outcome.

For each value of N, we average over 300 randomly selected subsets of N attempts per goal. Table 17 shows that the probe selects a successful attempt more often than the model’s own judgment, with the largest observed gap at N = 16: 40.0% versus 28.0%, a 12.0-point advantage (95% CI [2.0, 22.0]). The oracle reaches 44.0% at the same sampling budget. The probe’s training target is option-selection completion, not general product correctness; these results demonstrate useful ranking within the evaluated goal distribution, rather than a general-purpose guarantee of purchase correctness.

Table 15. Decision latency (p50, ms) of decider-4b in its production engine, with and without EVIGATE (30 iterations after warm-up). The p99 latency of the base engine is within 0.3–0.7% of p50 for T3A states and 0.7–4.5% for short states, so overheads below about 1% are within measurement variability.
<table><tr><td>States</td><td>Batch</td><td>Engine</td><td>+EVIGATE</td><td>Overhead</td></tr><tr><td>T3A (~2.7–3.4k tokens)</td><td>1</td><td>363.4</td><td>368.3</td><td>+1.35%</td></tr><tr><td rowspan="5">Short (~280 tokens)</td><td>8</td><td>3870.2</td><td>3873.8</td><td>+0.09%</td></tr><tr><td>32</td><td>19503.6</td><td>19512.3</td><td>+0.04%</td></tr><tr><td>1</td><td>39.7</td><td>40.5</td><td>+2.01%</td></tr><tr><td>8</td><td>214.9</td><td>217.8</td><td>+1.32%</td></tr><tr><td>32</td><td>829.3</td><td>831.4</td><td>+0.26%</td></tr></table>

Table 16. Phone simulator, all conditions (24 tasks × 30 seeds, decider-4b, EVIGATE with the balanced threshold). Gain: paired change in success vs. no gate, with 95% CI over tasks. False stops: episodes that stop before the goal is reached.
<table><tr><td>Condition</td><td>Success</td><td>Gain</td><td>False stops</td><td>Calls / episode</td><td>Calls / success</td></tr><tr><td>No gate</td><td>77.2</td><td></td><td>13.9</td><td>3.80</td><td>4.92</td></tr><tr><td>Random veto, second question&#x27;s rate</td><td>77.5</td><td>+0.3 [−0.3, 0.8]</td><td>13.3</td><td>3.99</td><td>5.14</td></tr><tr><td>Random veto, EvIGATE&#x27;s rate</td><td>77.6</td><td>+0.4 [0.0, 0.8]</td><td>13.3</td><td>3.98</td><td>5.12</td></tr><tr><td>Contrastive decoding</td><td>77.1</td><td>-0.1 [-9.7, 7.9]</td><td>8.2</td><td>8.93</td><td>11.58</td></tr><tr><td>Second question</td><td>83.2</td><td>+6.0 [1.7, 11.2]</td><td>7.8</td><td>5.07</td><td>6.10</td></tr><tr><td>LLM judge (9B)</td><td>81.5</td><td>+4.3 [1.7, 7.6]</td><td>9.3</td><td>5.04</td><td>6.18</td></tr><tr><td>EVIGATE</td><td>82.4</td><td>+5.1 [1.8, 8.9]</td><td>8.6</td><td>4.00</td><td>4.86</td></tr></table>

Table 17. Best-of-N success (%) on 50 WebShop goals, using 16 sampled agent trajectories per goal. Each non-oracle strategy selects among candidate attempts using the indicated ranking criterion; results average over 300 random subsets per goal. Strategies choose only among attempts that end in a purchase, and a goal counts as failed when none of its N attempts buys; this is why even random selection improves with N. Confidence intervals resample goals.
<table><tr><td>N</td><td>1</td><td>2</td><td>4</td><td>8</td><td>16</td></tr><tr><td>Oracle</td><td>9.8</td><td>16.9</td><td>25.9</td><td>35.8</td><td>44.0</td></tr><tr><td>Random attempt</td><td>9.8</td><td>14.1</td><td>17.2</td><td>19.8</td><td>21.2</td></tr><tr><td>Decision model&#x27;s own judgement</td><td>9.8</td><td>15.2</td><td>20.4</td><td>24.9</td><td>28.0</td></tr><tr><td>Probe on its hidden state</td><td>9.8</td><td>16.5</td><td>24.3</td><td>32.8</td><td>40.0</td></tr></table>