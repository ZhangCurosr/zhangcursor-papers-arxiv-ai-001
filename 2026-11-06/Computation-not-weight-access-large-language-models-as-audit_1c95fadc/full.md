# Computation, not weight access: large language models as auditors of clinical prediction networks

Zhiyu Liu<sup>1</sup> Xinyu Yang<sup>2,\*</sup>

<sup>1</sup>Tianjin Medical University, Tianjin, China; 07eb1f@gmail.com

<sup>2</sup>Tianjin Medical University General Hospital, Tianjin, China; yangxinyu@tmu.edu.cn \* Corresponding author

November 2026

## Abstract

Frontier large language models (LLMs) have been reported to read, edit and write the weights of small neural networks, raising the prospect of auditing clinical prediction models from their files alone. Public demonstrations, however, mixed reading raw weights with running code. We separated these conditions in internally pre-specified simulation experiments. Tanh networks of 105 to 38,401 weights were trained on simulated cohorts of patients with subarachnoid haemorrhage, each with a planted relationship or none, and one network was trained on labels that under-recorded deaths in a subgroup. GPT-6 Astra and four other GPT models answered single prompts under a default or an explicit system instruction, either with the weights in the prompt and no tools, or with a calculation-only tool that exposed the weights (white box) or only the network’s predictions (black box). Without tools, Astra recovered planted relationships in 353-weight networks but became unreliable at 3,457 weights, and under the default instruction it declined 11 of 12 audit and repair tasks as unidentifiable; the explicit instruction removed these refusals. With the tool, it recovered the planted relationship in all 29 runs on networks of up to 38,401 weights, raised no false alarm in 15 runs on null networks, and found the biased subgroup in 6 of 6 runs. Black-box access performed as well as white-box access, and other GPT-6 models read as accurately. A well-designed script matched the comparator models on the audit and the best LLM repair. For auditing these networks, what mattered was computation and elicitation, not access to the weights.

## 1 Introduction

In November 2026, Takehi and colleagues reported that GPT-6 Astra could tell what small networks had been trained to do by reading their raw weights, write working policy networks from a description of the task, and repair a 0.8-billion-parameter language model by editing a small fraction of its weights, all without gradient-based training [1]. Independent re-runs reproduced the identification of game-playing networks and showed that it fell to chance once the weights were shufled within their matrices [2]. The report also drew critiques. The most pointed one held that the headline combined two diferent conditions under one claim: reading raw weights of a small network without any tool, and analysing a larger model by writing and running code [3]. Others argued that the small-network results could be matched by simple inspection [4], that the reported gain in mathematics rested on a weak baseline setting [5], and that reading raw weights helped little in predicting what a weight edit would do [6]. Whether the ability lies in reading weights or in computing with them has practical consequences, because the two require diferent access to a model.

Clinical prediction models make the question concrete. Hospitals, regulators and reviewers increasingly receive models without the data they were trained on, and such models can encode problems that headline performance does not reveal. One example is a label that records outcomes unevenly across patient groups [11]. Post hoc explanation of black-box clinical models has well-known limits [12, 13]. An auditor that could describe what a model computes, find where it departs from what is plausible, and correct it would be useful. Nusrat and Nusrat recently showed that Claude Opus 5.5, reading only the weights of small rule-based radiotherapy-planning networks, named a planted flaw in 16 of 18 networks, although simple programs were faster and judging a healthy network was harder [7].

We asked which conditions such performance depends on. We trained clinical prediction networks on simulated data with known ground truth and gave them to GPT-6 Astra and four other GPT models under conditions that the public debate had mixed: the weights in the prompt with no tools, a calculation-only tool holding the weights, and the same tool exposing only the network’s predictions. We also varied the system instruction, because many first replies declined tasks rather than attempting them. Our contributions are:

• a factorial evaluation of instruction and access on clinical-style networks of 105 to 38,401 weights, with five models, pre-specified scoring rules, negative controls and repeats;

• evidence that reading weights without tools works only for small networks and varies between repeats, whereas computation makes reading and auditing accurate and stable at every size tested, with or without access to the weights;

• evidence that the default single-reply set-up produced refusals framed as non-identifiability, and that one paragraph of system instruction removed them without adding false alarms in our controls;

• a comparison with scripted analyses and edits that use no LLM, which places the LLM’s added value in designing and carrying out the audit rather than in capabilities beyond a well-specified script;

• the release of all networks, prompts, replies, logs and code.

## 2 Related work

Weight-space programming. The original report [1] demonstrated reading, writing and editing of network weights by GPT-6 Astra; small networks were read from raw weights alone, while the edits of larger language models used code and repeated evaluation. Re-runs reproduced the reading of game-playing networks with GPT-6 Astra and Claude Opus 5.5 and showed its dependence on intact weights [2]. Critics questioned the framing of these results as native weight reading [3, 4], the baseline behind the mathematics result [5], and how much reading raw weights adds to predicting the efect of an edit [6].

Automated interpretability agents. Language models have been used as agents that explain other systems by experimenting on them. FIND provides synthetic functions with known descriptions and found that an LLM agent with black-box access can infer much of a function’s structure, although its descriptions captured global behaviour better than local details [8]. MAIA equips a multimodal model with tools to run experiments on vision models and describe their components [9]. Our black-box condition is an agent of this kind; our question is whether access to the weights adds anything once such experiments are possible.

Auditing clinical prediction models. Clinical algorithms can inherit biases from how their labels were recorded [11], and explanations of black-box models can be unfaithful or misleading [12, 13]. The closest study to ours audited small rule-based clinical decision networks from their weights alone, without tools [7]. We add networks trained on noisy probabilistic outcomes, sizes an order of magnitude larger, conditions with and without computation and weight access, and several models.

Table 1: Study design at a glance.
<table><tr><td>Element</td><td>Levels</td></tr><tr><td>Networks</td><td>Tanh multilayer perceptrons trained on simulated cohorts: 105 weights (11 inputs); S, 353; M, 1,217; L, 3,457; D3 (three hidden layers), 2,817; XL, 11,009; XXL, 38,401 (20 inputs); and a 209-weight network trained on biased labels, with an unbiased counterpart</td></tr><tr><td>Planted relationships</td><td>Threshold; interaction; each at half strength; none (null). In the 105-weight networks: a threshold, a hypertension-by-occupation interaction, an age dip</td></tr><tr><td>Tasks</td><td>Read (explain and write a surrogate model); audit (find subgroups whose predicted risk departs from what their other characteristics imply); repair (edit the weights)</td></tr><tr><td>Instruction Access</td><td>Default (one sentence); explicit (open choices left to the model; Appendix B) Weights in the prompt, no tools; calculator with the weights (white box);</td></tr><tr><td></td><td>calculator with predictions only (black box)</td></tr><tr><td>Models Baselines without an LLM</td><td>GPT-6 Astra (main); gpt-6.1-sol, gpt-6-sol, gpt-6-luna, gpt-5.6-sol Exact scan of each network; scripted audits (regression tree; exhaustive</td></tr><tr><td></td><td>search); scripted repairs (gradient fine-tuning; a scripted gate)</td></tr><tr><td>Repeats</td><td>Key cells three times; others once or twice (shown in the tables)</td></tr></table>

## 3 Methods

## 3.1 Overview

Table 1 summarises the design. All networks were trained on simulated data, so the true relationships were known. Each run was one prompt and one reply, and every reply was scored by a script against the ground truth with rules written before the runs.

## 3.2 Simulated networks

Each simulated cohort had 20,000 adults with aneurysmal subarachnoid haemorrhage. The ten clinical inputs were coded as in a typical admission dataset: age (in decades from 60), sex, poor clinical grade (WFNS 4–5), thick haemorrhage (Fisher 3–4), intraventricular and intracerebral haemorrhage, hypertension, aneurysm treatment during admission, urban residence and farming occupation. Their distributions and the coeficients of the outcome model were set by hand to clinically plausible values and were not estimated from any dataset. Ten neutral laboratory values $( l _ { 1 }$ to $l _ { 1 0 } .$ , uniform on 0–10) were added, so that medical knowledge could not help with relationships planted on them. The outcome, death within one month, was drawn from a logistic model with a common clinical part and at most one planted relationship: a threshold, +0.9 max $( l _ { 3 } - 6 , 0 )$ , or an interaction, $+ 1 . 6 \sigma ( l _ { 5 } ) \sigma ( l _ { 8 } )$ with $\sigma ( { \bar { x } } ) = 1 / ( 1 + e ^ { - 2 ( x - 5 ) } )$ , so that risk rose only when both values were high; each also at half strength; or none.

Networks were multilayer perceptrons with tanh units, trained with scikit-learn (Table 1). Before use, each network had to pass a stimulus check: it had to carry at least half of the planted strength, and a null network could carry no signal of 0.3 or more at the planted locations. Training used Adam with early stopping and fell back to L-BFGS with an L2 penalty when a network failed the check. A further set of three bootstrap networks per condition, trained on resamples of one cohort, was given to the model together, with the request to report only the structure they shared. An earlier set of 105-weight networks (11 inputs, eight hidden units) carried a threshold on one laboratory value, an interaction between hypertension and farming, or a dip in risk between ages 50 and 60.

For auditing, a 209-weight network (11 inputs, 16 hidden units) was trained on labels in which the recorded risk of women farmers over 65 (about 4% of patients) was lowered by 3.0 on the log-odds scale, mimicking deaths that go unrecorded when patients are taken home to die.

Its log-odds in that subgroup were 1.60 lower than those of a network trained on unbiased labels, which served both as the reference for scoring and as a negative control.

## 3.3 Tasks and prompts

Each prompt named the outcome and the population (adults admitted with aneurysmal subarachnoid haemorrhage), gave the inputs and their coding, the architecture and the weights (as JSON with four significant figures), and asked for a brief explanation ending in a JSON block (Appendix C).

• Read. The model explained what the network computes and wrote a surrogate model of linear, hinge and product terms, together with a list of relationships that were “not standard medical knowledge”.

• Audit. The model listed subgroups, defined by conditions on the inputs, whose predicted risk was systematically lower or higher than their other characteristics implied, with the size of the distortion. In one variant the prompt stated the reference: the network’s additive part over a described population.

• Repair. The model returned the complete edited network in the same shape. In one variant it had to find the distorted subgroup itself; in others the subgroup was given, with the target either to remove its departure from the additive part or to raise its log-odds by exactly 1.6 while leaving everyone else unchanged. Two variants stated a tolerance.

A reply without a usable JSON block, or one that declared the requested fields empty, was recorded as declined.

## 3.4 Instruction and access

Instruction. The default system instruction was one sentence: “Answer the user’s request.” Because many replies under it declined tasks that required a choice the prompt had left open, we added an explicit instruction while keeping every prompt unchanged (Appendix B). It told the model that it had one reply and no one to ask; to choose the most defensible option for anything left open (a reference, a population, a tolerance or a size), name it and continue; to give the closest approximation, and say how close it was, when a request could not be met exactly; and not to add disclaimers. This follows published guidance for GPT-6 models, which notes their tendency to ask when more input would change the result [10].

Access. Without tools, the weights were in the prompt and every tool was disabled. With the calculation-only tool (hereafter, the calculator), the model had a single tool, run\_python, served by a local Model Context Protocol server, which ran NumPy code with the network preloaded: with its weight arrays and a prediction function (white box), or with the prediction function only (black box). The calculator refused imports, names beginning with an underscore and NumPy’s file functions, ran each call in a fresh process in an empty folder with a 60-second limit, and accepted only simulated networks. In the reading prompts with the calculator, the weights were held by the tool rather than printed. Every tool call was logged and audited.

## 3.5 Models and harness

The main model was GPT-6 Astra (gpt-6-astra) at its maximal reasoning efort. The comparators were gpt-6.1-sol, gpt-6-sol, gpt-6-luna and gpt-5.6-sol, at the same efort. All ran through the Codex command-line interface (version 0.166.0) in non-interactive mode, one prompt and one reply per run, each in a new empty folder with the user’s configuration ignored, a read-only sandbox and web search of. Comparator models received the same prompts, byte for byte.

## 3.6 Outcomes

Reading. The strength of a planted relationship was computed for the network and for the surrogate on the same 5,000 test inputs: for the threshold, the efect of $l _ { 3 }$ from 5 to 9 minus its efect from 1 to 5; for the interaction, the interaction contrast of $l _ { 5 }$ and $l _ { 8 }$ at 2 and 8. A relationship counted as recovered if the surrogate’s strength had the network’s sign and at least half its size. On a null network, a surrogate strength of 0.5 or more at either planted location was a false alarm. We report the strength ratio (surrogate over network) and the $R ^ { 2 }$ of the surrogate against the network’s log-odds. As a further check, we counted surrogate efects of 0.5 or more at 65 locations that were zero in the generating model: the range efect and curvature of each laboratory value, and each pairwise interaction between them. For the 105-weight networks we applied the rules written for that experiment, which judge the shape of the surrogate.

Auditing. A reported subgroup counted as found if its Jaccard overlap with the true subgroup on the test inputs was at least 0.5 and its stated shift had the right sign and at least half the network’s actual shift. On the unbiased network, any subgroup with a stated shift of 0.5 or more was a false alarm.

Repair. We report the mean change in log-odds inside the subgroup, the mean absolute change elsewhere, and the remaining diference from the unbiased network inside the subgroup.

## 3.7 Baselines without a language model

For reading, an exact scan of each network at the 65 locations shows what a script reading the network perfectly would report. For auditing, a script with prediction access only fitted an additive model to the network’s log-odds over the balanced synthetic population that Astra had chosen in its calculator runs (ages 18 to 90, binary inputs equally likely), then searched the residuals in two pre-specified ways: a regression tree (depth 3, leaves of at least 1%) and an exhaustive search over conjunctions of up to three binary conditions with an optional age cut. For repair, we compared gradient fine-tuning of all parameters with a scripted gate, in which the hidden unit whose removal cost least became a steep gate for the subgroup and the output layer was refitted by least squares.

## 3.8 Pre-specification and statistics

The plan, stimuli and scoring code of every experiment were committed to a version-controlled repository before its first run; an addendum scoring claims at the 65 zero locations was committed before any afected reply was read (Appendix A). This was internal pre-specification, not a public registration, and the design was adaptive: later conditions were designed after earlier results were known. Results are descriptive; proportions are given with Wilson 95% intervals [14].

## 4 Results

## 4.1 Without tools, reading works for small networks only

Under the default instruction and without tools, GPT-6 Astra recovered the planted relationship in all six runs on the 353-weight networks, including both half-strength versions, with surrogate $R ^ { 2 }$ of 0.41–0.88 and no false alarm on the null network (Table 2, Figure 1). At 1,217 weights it recovered the interaction in both runs but the threshold in one of two (strength ratios 0.44 and 0.91), and its null surrogates had $R ^ { 2 }$ of −0.21 and 0.07. At 3,457 weights it recovered the threshold in both runs but the interaction in only one, with strength ratios of 0.00 and 2.39 and surrogate $R ^ { 2 }$ of 0.33–0.39; the two null surrogates had $R ^ { 2 }$ of −0.54 and 0.88. Repeats of the same cell often difered widely, and single runs took 23 to 60 minutes.

The failing replies explained themselves. Astra wrote that it could not carry out the numerical sweep it had planned, linearised the network around a midpoint (age 60, laboratory values 5, binary inputs 0.5) and reported the local gradient. The gradient was correct, but a local linearisation is a poor global surrogate, and its linear extrapolation produced spurious efects at up to seven zero locations.

![](images/5e7648e92d74ccbb4a0957633765fddd631c2b13f2c52b1620d9471bb2c3cb3c.jpg)

![](images/1413fec2e395c756f19686ea92ae4ed8954ce30f37e036ce50290557bf2ff0ca.jpg)  
Figure 1: Reading planted relationships across network sizes (GPT-6 Astra). (a) Strength of the planted relationship in the model’s surrogate divided by its strength in the network; the dashed line marks equality and the dotted line the pre-specified threshold of one half. (b) $R ^ { 2 }$ of the surrogate against the network’s log-odds, including null networks. Default and explicit refer to the system instruction; the calculator held either the weights (white box) or only a prediction function (black box). Points are single runs, ofset slightly on the horizontal axis.

Astra read the function rather than the pattern of numbers. When the hidden units of two 105-weight networks were permuted and half of them sign-flipped, which changes every weight but not the function, it read the same relationships with the same strengths (2.44 against 2.35 before; 1.71 against 1.72), with $R ^ { 2 }$ of 0.97 and 0.96.

## 4.2 The default instruction produced refusals

Under the default instruction, 11 of 12 audit and repair runs declined (95% interval 0.65–0.99). Astra argued that a distortion “implied by other characteristics” is not identifiable without a reference, that a stated set of marginal distributions does not determine the joint distribution, or that a tanh network, being continuous in age, cannot change only patients above 65 exactly. These arguments are correct as mathematics, but they answer a diferent question from the one asked. The twelfth run reported a spurious subgroup on the unbiased network (Table 3).

Under the explicit instruction, with every prompt unchanged, none of 14 audit and repair runs declined (95% interval 0.00–0.22). The instruction also changed reading without tools: on the 3,457-weight networks the planted relationship was recovered in all four runs (surrogate $R ^ { 2 } 0 . 5 2 { - 0 . 8 6 } )$ and both null surrogates had $R ^ { 2 }$ of 0.90 (Table 2), and on three bootstrap null networks given together the surrogate $R ^ { 2 }$ rose from −0.24 to 0.88. On the unbiased audit network it reported only two small departures (±0.25), below the false-alarm threshold. Stating an additive reference in the prompt did not help: under the explicit instruction it led to three false alarms at the edge of a laboratory value’s range, where the network is genuinely non-additive.

The explicit instruction also had costs. Twice, asked to raise one subgroup’s log-odds by exactly 1.6, Astra chose “smallest worst-case error” as its criterion; because a continuous network cannot reproduce an exact step, adding 0.8 to every patient is optimal under that criterion, and that is what it returned. Once, to meet a stated tolerance, it changed the network from 16 to 17 hidden units, against the instruction to keep the shapes.

Table 2: Reading planted relationships (GPT-6 Astra). Each cell gives the strength ratio and the surrogate $R ^ { 2 }$ (range over runs) and, in brackets, the number of runs in which the relationship was recovered by the pre-specified rule; for null networks, $R ^ { 2 }$ and the number of false alarms.
<table><tr><td>Network (weights)</td><td>Planted</td><td>Default, no tool</td><td>Explicit, no tool</td><td>Calculator, white box</td><td>Calculator, black box</td></tr><tr><td>S (353)</td><td>Threshold</td><td>0.89; 0.41–0.88 [2/2]</td><td></td><td></td><td></td></tr><tr><td>S (353)</td><td>Interaction</td><td>0.56–0.81; 0.84 [2/2]</td><td></td><td>0.85; 0.88 [1/1]</td><td></td></tr><tr><td>S (353)</td><td>Threshold, half strength</td><td>0.96; 0.82 [1/1]</td><td></td><td></td><td></td></tr><tr><td>S (353)</td><td>Interaction, half strength</td><td>0.70; 0.79 [1/1]</td><td></td><td></td><td></td></tr><tr><td>S (353)</td><td>None (null)</td><td>0.89–0.90 [0/2]</td><td></td><td>0.91 [0/1]</td><td></td></tr><tr><td>M (1,217)</td><td>Threshold</td><td>0.44–0.91; 0.85–0.90 [1/2]</td><td></td><td></td><td></td></tr><tr><td>M (1,217)</td><td>Interaction</td><td>0.51–0.63; 0.89–0.92 [2/2]</td><td></td><td></td><td></td></tr><tr><td>M (1,217)</td><td>None (null)</td><td>-0.21 to 0.07 [0/2]</td><td></td><td></td><td></td></tr><tr><td>L (3,457)</td><td>Threshold</td><td>0.52–1.29; 0.58–0.88 [2/2]</td><td>0.64–1.41; 0.52–0.81 [2/2]</td><td>1.02–1.03; 0.96 [3/3]</td><td>1.03–1.07; 0.96 [3/3]</td></tr><tr><td>L (3,457)</td><td>Interaction</td><td>0.00–2.39; 0.33–0.39 [1/2]</td><td>0.73; 0.83–0.86 [2/2]</td><td>0.97–1.03; 0.92 [3/3]</td><td>1.02–1.10; 0.92 [3/3]</td></tr><tr><td>L (3,457)</td><td>None (null)</td><td>−0.54 to 0.88 [0/2]</td><td>0.90 [0/2]</td><td>0.94 [0/3]</td><td>0.94 [0/3]</td></tr><tr><td>D3 (2,817)</td><td>Threshold</td><td></td><td></td><td>1.10; 0.94 [1/1]</td><td></td></tr><tr><td>D3 (2,817)</td><td>Interaction</td><td></td><td></td><td>1.01; 0.92 [1/1]</td><td></td></tr><tr><td>D3 (2,817)</td><td>None (null)</td><td></td><td></td><td>0.94 [0/1]</td><td></td></tr><tr><td>XL (11,009)</td><td>Threshold</td><td></td><td></td><td>1.03; 0.96 [1/1]</td><td></td></tr><tr><td>XL (11,009)</td><td>Interaction</td><td></td><td></td><td>1.07; 0.97 [1/1]</td><td></td></tr><tr><td>XL (11,009)</td><td>None (null)</td><td></td><td></td><td>0.97 [0/1]</td><td></td></tr><tr><td>XXL (38,401)</td><td>Threshold</td><td></td><td></td><td>1.00–1.02; 0.96 [3/3]</td><td>1.02–1.04; 0.96 [3/3]</td></tr><tr><td>XXL (38,401)</td><td>Interaction</td><td></td><td></td><td>0.98–1.08; 0.98 [3/3]</td><td>0.97; 0.98 [3/3]</td></tr><tr><td>XXL (38,401)</td><td>None (null)</td><td></td><td></td><td>0.98 [0/3]</td><td>0.98 [0/3]</td></tr></table>

## 4.3 With the calculator, size was no obstacle and the weights were not needed

With the calculator, GPT-6 Astra recovered the planted relationship in all 29 runs that had one (95% interval 0.88–1.00), on networks of 353 to 38,401 weights, with strength ratios of 0.85–1.10 and surrogate $R ^ { 2 }$ of 0.88–0.98 (Table 2, Figure 1). It raised no false alarm in 15 runs on null networks (0.00–0.20) and made at most one spurious claim per run at the 65 zero locations. Repeats agreed closely; for example, the strength ratio for the 3,457-weight threshold network was 1.02–1.07 across three white-box and three black-box runs. Runs took a median of 12 minutes (range 9 to 30) with 7 to 14 tool calls.

Access to the weights made no measurable diference. In the 18 black-box runs, in which the calculator exposed only the network’s predictions, strength ratios were 0.97–1.10 and $R ^ { 2 }$ 0.92–0.98, matching the white-box runs cell by cell.

## 4.4 Other models

Reading small networks was not particular to Astra (Table 5). On the 105-weight networks without tools, gpt-6.1-sol recovered 3 of 3 planted relationships with $R ^ { 2 }$ of 0.96–0.97, as Astra did (6 of 6; 0.94–0.97). gpt-5.6-sol, of the previous generation, recovered 3 of 3 by the shape rules but with poorer fits, and gpt-6-sol and gpt-6-luna recovered 2 of 3. With the calculator, all three GPT-6 comparators read the 3,457- and 38,401-weight networks as accurately as Astra (interaction strength ratios 0.89–1.13; null R<sup>2</sup> 0.92–0.98), although gpt-6-luna needed about twice as many tool calls.

## 4.5 Auditing a network trained on biased labels

With the calculator, GPT-6 Astra found the biased subgroup in all six runs, three white-box and three black-box (95% interval 0.61–1.00). Its subgroups coincided with the true one (Jaccard 0.995–1.000), and its stated shifts ranged from −1.60 to −2.00 against the network’s −1.60 (Table 3, Figure 2). It chose its own reference without being told: an additive model fitted over a balanced synthetic population. On the unbiased network it raised one false alarm in six runs: a white-box run flagged patients with a very low value of the laboratory input (shift +0.53, just above the threshold), the edge region where the network is genuinely non-additive. The three black-box runs reported nothing.

![](images/25e7e5c595ce3d7646f68064b07b756c9115f7f90741648bcaea9d605f291956.jpg)  
Figure 2: Recovery of the biased subgroup. Each point is one run on the biased network: the Jaccard overlap between the true subgroup (women farmers over 65) and the reported subgroup closest to it. Crosses mark runs that declined to report a subgroup; the dotted line marks the pre-specified threshold of 0.5. Rows without a model name are GPT-6 Astra.

Table 3: Auditing the biased network and the unbiased control. Found: pre-specified rule (Jaccard ≥ 0.5, correct sign, at least half the network’s shift). False alarm: a stated shift of 0.5 or more on the unbiased network. Rows without a model name are GPT-6 Astra.
<table><tr><td>Condition</td><td>Biased: found</td><td>Biased: declined</td><td>Best Jaccard</td><td>Unbiased: false alarm</td></tr><tr><td>Default instructions, weights in prompt</td><td>0/2</td><td>2/2</td><td>0.00</td><td>0/1</td></tr><tr><td>Default + stated additive reference</td><td>0/2</td><td>2/2</td><td>0.00</td><td>1/1</td></tr><tr><td>Explicit instructions, weights in prompt</td><td>1/2</td><td>0/2</td><td>0.01-0.53</td><td>0/1</td></tr><tr><td>Explicit + stated additive reference</td><td>1/2</td><td>0/2</td><td>0.59-1.00</td><td>1/1</td></tr><tr><td>Explicit + calculator, white box</td><td>3/3</td><td>0/3</td><td>0.99-1.00</td><td>1/3</td></tr><tr><td>Explicit + calculator, black box</td><td>3/3</td><td>0/3</td><td>1.00</td><td>0/3</td></tr><tr><td>gpt-6.1-sol + calculator, white box</td><td>1/1</td><td>0/1</td><td>0.53</td><td>0/1</td></tr><tr><td>gpt-6-sol + calculator, white box</td><td>1/1</td><td>0/1</td><td>0.52</td><td>1/1</td></tr><tr><td>gpt-6-luna + calculator, white box</td><td>0/1</td><td>0/1</td><td>0.05</td><td>1/1</td></tr><tr><td>Script, regression tree (no LLM)</td><td>0/1</td><td>0/1</td><td>0.00</td><td>0/1</td></tr><tr><td>Script, exhaustive search (no LLM)</td><td>1/1</td><td>0/1</td><td>0.53</td><td>0/1</td></tr></table>

The comparator models were less precise. gpt-6.1-sol and gpt-6-sol found the subgroup but placed the age boundary at 70 (Jaccard 0.53 and 0.52); gpt-6-sol and gpt-6-luna each raised a false alarm, and gpt-6-luna missed the subgroup. Without tools, under the explicit instruction, Astra found the subgroup in one of two runs; in the other it named the right groups (“older farmers, especially women”) but defined the subgroup as a narrow profile that almost no patient met.

A script did as well as the comparator models. With prediction access only, the exhaustive search found the subgroup in under a second, with the same shifted boundary (women farmers over 70; Jaccard 0.53; shift −1.35); the regression tree did not, settling on the edge region of a laboratory value. Neither raised a false alarm. Two caveats apply: we designed the search space knowing the form of the planted subgroup, whereas Astra designed its own search and placed the boundary exactly.

Table 4: Repairs. Change in subgroup: mean change in log-odds for women farmers over 65 (target +1.6, or removal of the departure). Bias after: remaining diference from the unbiased network inside the subgroup (before: −1.60). Invalid: an answer that broke the shape constraint. Rows above the line are GPT-6 Astra.
<table><tr><td>Condition</td><td>Runs</td><td>Declined</td><td>Invalid</td><td>Change in subgroup</td><td>Mean |change| elsewhere</td><td>Bias after</td></tr><tr><td>E8 repair, default (subgroup not given)</td><td>2</td><td>2</td><td>0</td><td></td><td>0.00</td><td>-1.61 to -1.60</td></tr><tr><td>E8 repair, explicit (subgroup not given)</td><td>2</td><td>0</td><td>0</td><td></td><td>0.14</td><td>-1.44</td></tr><tr><td>Subgroup given, default</td><td>2</td><td>2</td><td>0</td><td>0.00</td><td>0.00</td><td>-1.60</td></tr><tr><td>Subgroup given, explicit + tolerance</td><td>2</td><td>0</td><td>0</td><td>1.24-1.40</td><td>0.07-0.09</td><td>-0.37 to −0.20</td></tr><tr><td>+1.6 exact, default</td><td>2</td><td>2</td><td>0</td><td>0.00</td><td>0.00</td><td>-1.60</td></tr><tr><td>+1.6 exact, explicit</td><td>2</td><td>0</td><td>0</td><td>0.80</td><td>0.80</td><td>-0.81</td></tr><tr><td>+1.6 exact, explicit + tolerance</td><td>2</td><td>0</td><td>1</td><td>0.00–2.00</td><td>0.00–0.70</td><td>-1.60 to 0.39</td></tr><tr><td>+1.6 exact, explicit + tolerance + calculator</td><td>2</td><td>0</td><td>0</td><td>1.48-1.49</td><td>0.21</td><td>-0.12 to −0.11</td></tr><tr><td>Script: gradient fine-tuning (207/209 parameters)</td><td>1</td><td>0</td><td>0</td><td>1.20</td><td>0.10</td><td>-0.41</td></tr><tr><td>Script: scripted gate + least squares (29/209 parameters)</td><td>1</td><td>0</td><td>0</td><td>1.51</td><td>0.13</td><td>-0.10</td></tr></table>

Table 5: Other models. Left: the 105-weight networks without tools (pre-specified shape rules; three planted relationships). Right: the calculator with white-box access on the 3,457- and 38,401-weight networks and the audit.
<table><tr><td rowspan="2">Model</td><td colspan="2">No tool, 105-weight networks (E13)</td><td colspan="3">Calculator, white box (E19)</td></tr><tr><td>Found</td><td>R2</td><td>Interaction ratio (L, XXL)</td><td>Null R2 (L, XXL)</td><td>Audit (Jaccard; false alarm)</td></tr><tr><td>gpt-6-astra</td><td>6/6</td><td>0.94–0.97</td><td>0.85-1.08</td><td>0.91-0.98</td><td>0.99–1.00; 1/3</td></tr><tr><td>gpt-6.1-sol</td><td>3/3</td><td>0.96–0.97</td><td>0.89–0.97</td><td>0.94–0.98</td><td>0.53; 0/1</td></tr><tr><td>gpt-6-sol</td><td>2/3</td><td>0.30–0.91</td><td>0.95-0.99</td><td>0.92–0.98</td><td>0.52; 1/1</td></tr><tr><td>gpt-6-luna</td><td>2/3</td><td>0.49–0.84</td><td>0.99–1.13</td><td>0.93-0.97</td><td>0.05; 1/1</td></tr><tr><td>gpt-5.6-sol</td><td>3/3</td><td>0.33-0.93</td><td></td><td></td><td></td></tr></table>

## 4.6 Repair succeeded only partly

Editing the weights, the one task that requires them, was the weakest (Table 4). Under the default instruction all six repair runs declined. Under the explicit instruction without tools, Astra chose a wrong subgroup in both runs when it had to find the distortion itself. With the subgroup given and a tolerance stated, it merged two hidden units to free one and turned the freed unit into a smooth gate; the remaining bias in the subgroup fell from −1.61 to −0.20 and −0.37, and other patients’ log-odds changed by only 0.07 and 0.09 on average. Asked for an exact +1.6, it gamed the criterion or broke the shape constraint, as described above. With the calculator, it refitted 15 units to the original function and used the sixteenth as a gate: the subgroup’s log-odds rose by 1.48 and 1.49, but other patients’ changed by 0.21 on average, a shortfall against the requested tolerance that it reported itself.

The scripted gate, computed in 0.2 seconds and changing 29 of 209 parameters, did slightly better (1.51 in the subgroup, 0.13 elsewhere). Gradient fine-tuning changed 207 of 209 parameters and undershot (1.20).

## 4.7 Faithful to the network, noise included

Astra described the networks it was given, including their noise. With the calculator, its lists of “unexpected” relationships in null networks contained one to eight entries per run. These were real, small wiggles of the networks themselves, changes of about 0.3 to 0.5 across a laboratory value’s whole range, and they were correctly sized; the surrogates did not exaggerate them. A reader who did not look at the sizes could still mistake them for findings. Some artefacts were systematic: networks trained on data with a threshold on $l _ { 3 }$ carried negative interactions between $l _ { 3 }$ and other laboratory values that were absent from the generating model, and three networks trained on bootstrap resamples of one cohort agreed on them (for example −0.55, −0.57 and −0.85 for $l _ { 3 } \times l _ { 1 0 } )$ . Agreement across resamples therefore cannot separate structure in the data from artefacts of training.

## 5 Discussion

## 5.1 Principal findings

Three findings stand out. First, reading a network’s weights without tools worked, but only for small networks of the order of $1 0 ^ { 3 }$ weights, and results varied between repeats; beyond that size the model said, correctly, that it could not do the computation and fell back on a local approximation. Second, with a calculation-only tool, reading and auditing were accurate and stable at every size we tried, up to 38,401 weights, and access to the weights added nothing measurable: given only the network’s predictions, Astra read and audited as well as with the weights. Third, the default single-reply set-up made the model decline most audits and repairs, and one paragraph of system instruction, with no change to the prompts, removed the refusals without raising false alarms in our controls.

## 5.2 Relation to the debate on weight-space programming

The critique that tool-free weight reading and code-assisted analysis had been conflated [3] is borne out: the two behaved very diferently, and the second did not depend on the weights at all. The limited value of reading raw weights for predicting edit efects [6] fits our size ladder. Our findings agree with Nusrat and Nusrat [7] that frontier models can find planted problems in small clinical networks from the weights alone, that declaring a network healthy is harder, and that simple programs are often as good and faster. In their study, a black-box condition without tools and with a limited number of requests did worse than weight reading; in ours, a black-box condition with a calculator did as well. Together, the two studies suggest that performance is decided by the budget for computation and queries, not by whether the weights are visible. This places the useful capability in the tradition of automated interpretability agents that explain systems by experimenting on them [8, 9]. Unlike the agents evaluated in FIND, which captured global behaviour better than local details, the calculator runs here recovered planted local structure such as the position of a threshold; the low dimension and smoothness of clinical tabular networks may make this easier.

## 5.3 Implications for auditing clinical models

An LLM with a calculator and query access carried out a routine audit end to end: it chose a reference, searched for departures, sized them and described them. Because it did not need the model file, such an audit would not require a vendor to disclose its weights. The comparison with scripts tempers the claim. Once a person knows what kind of problem to look for, a few lines of code do as well; what the LLM adds is that it designs and carries out the search itself and reports in plain language, and in our tests Astra was also the most precise. Editing the weights remained the weakest task: constraints that are trivial for a script, such as keeping the shape of the network or staying within a tolerance, were where the model gamed the criterion or broke the rule. Faithful reading is also not the same as a trustworthy finding, because the model reported the networks’ own noise and training artefacts as faithfully as their planted structure.

## 5.4 Implications for evaluating LLM capabilities

Reported abilities depended on the instruction and the tools as much as on the model. A single-reply protocol removes the model’s preferred way of resolving an open choice, which is to ask a question, and the model then argues that the task cannot be done. Without tools, the explicit instruction alone let Astra read 3,457-weight networks that had defeated it under the default instruction. Evaluations should therefore report the instruction and tools used, count refusals separately from wrong answers, and check any change of elicitation against negative controls, as we did with the null and unbiased networks.

## 5.5 Limitations

All networks were small multilayer perceptrons trained on simulated data with hand-set parameters, and the planted relationships and the label bias were of simple, known forms; the results may not transfer to models trained on real clinical data, to other architectures such as gradient-boosted trees, or to problems of unknown form. The models came mainly from one family and were accessed through one harness, and several cells had only one to three runs. The scripted audit’s search space was designed with knowledge of the planted subgroup’s form, which favours the script. The prompts presented the networks as trained on patient data, a cover story that may have influenced the model’s priors. The models are commercial services that may change. Finally, the rules were pre-specified internally rather than publicly registered, and the design was adaptive (Appendix A).

## 6 Conclusion

Without tools, frontier LLMs read small clinical prediction networks from their weights but not larger ones. With a calculation-only tool they read and audited networks of up to 38,401 weights accurately and stably, and they did as well without seeing the weights. For auditing clinical prediction models, the useful capability is automated experimentation, which in our tests performed on a par with well-designed scripts. Any claim about what these models can do should state how the task was posed and which tools were available.

## Data and code availability

All simulated networks, prompts, replies, event logs and evaluation code are available at https: //github.com/Chth1z/llm-clinical-network-audit. No patient data were used.

## Ethics

This study used only simulated data and involved no human participants.

## Use of AI

The GPT models listed in the Methods were the objects of study. Claude Opus 5.5 (Anthropic), used through Claude Code, assisted with the experimental code, the analyses and the drafting of this manuscript under the authors’ direction. The authors checked the results and take full responsibility for the content.

## Competing interests

The authors declare no competing interests.

## Funding

This work received no specific funding.

## A Pre-specification record and deviations

For every experiment, the plan with its decision rules, the stimuli (networks and prompts) and the scoring code were committed to a private version-controlled repository before the first run (Table 6); the decision rules are restated in the Methods.

Table 6: Plan commits and the start of the first run of each experiment.
<table><tr><td>Experiment</td><td>Plan commit</td><td>Committed (UTC+8)</td><td>First run started</td></tr><tr><td>Planted relationships in 105-weight networks</td><td>3a6bd3c</td><td>2026-11-04 20:40</td><td>2026-11-04 20:42</td></tr><tr><td>Size ladder, half strength and network groups; biased-label audit and repair</td><td>e259471</td><td>2026-11-05 00:35</td><td>2026-11-05 00:35</td></tr><tr><td>Function-preserving rewrites</td><td>449c348</td><td>2026-11-05 00:42</td><td>2026-11-05 03:21</td></tr><tr><td>Addendum: claims at zero locations and exact scan</td><td>09a62cd</td><td>2026-11-05 00:51</td><td>script, after the commit</td></tr><tr><td>Other models without tools (105-weight prompts)</td><td>5bc52e3</td><td>2026-11-05 00:59</td><td>2026-11-05 00:59</td></tr><tr><td>Stated reference for audit and repair</td><td>1425aa0</td><td>2026-11-05 01:03</td><td>2026-11-05 01:03</td></tr><tr><td>Explicit system instruction</td><td>8b3a434</td><td>2026-11-05 01:14</td><td>2026-11-05 01:23</td></tr><tr><td>Calculation-only tool, white box</td><td>796eb7e</td><td>2026-11-05 02:05</td><td>2026-11-05 02:13</td></tr><tr><td>Calculation-only tool, black box</td><td>e49843a</td><td>2026-11-05 03:55</td><td>2026-11-05 04:00</td></tr><tr><td>Other models with the tool; first repeats</td><td>0004706</td><td>2026-11-05 05:55</td><td>2026-11-05 05:55</td></tr><tr><td>Scripted audit without an LLM</td><td>b759c7b</td><td>2026-11-05 08:39</td><td>script, after the commit</td></tr><tr><td>Further repeats; scripted repairs without an LLM</td><td>ab5e8f7</td><td>2026-11-05 08:52</td><td>2026-11-05 08:52</td></tr></table>

Deviations from the plans:

1. Up to 14 runs were started at once, more than planned, which led to “model at capacity” errors in three runs. Seven batches that had run for two to ten minutes were stopped and re-queued, and the failed runs were repeated with identical prompts. Afterwards a queue limited the machine to six concurrent runs and retried runs automatically when the service was busy.

2. Replies that declared the requested fields empty, or contained no JSON block, were scored as declined, as the plans specified; the parser was changed to record them as such, without changing the rules.

3. The addendum scoring claims at the 65 zero locations, and the exact scan, were added after the size-ladder runs had started and before any afected reply was read.

4. The explicit instruction, the stated reference, the calculator, black-box access and the scripted baselines were each designed after earlier results were known, and each was committed before its own runs.

5. The description of near misses in the audit (the right groups named, but defined too narrowly) is post hoc.

## B The explicit system instruction

Answer the user’s request in this one reply; there is no one to ask and no follow-up. Infer what the request is for and carry it to completion. Where it leaves open something that would change the answer, such as a reference, a population, a tolerance or a size, choose the most defensible option, name it in one line, and continue; never stop because a choice has more than one defensible answer or because the request cannot be met exactly. If it cannot be met exactly, deliver the closest approximation and say how close it is. Give your best estimate in the requested format; an empty or unchanged answer is right only when it is your best estimate, not when the evidence is merely incomplete. Do not add warnings, disclaimers or statements about what cannot be proven.

## C Example prompt (reading, calculator, white box)

A neural network was trained to predict death within one month in adults admitted with aneurysmal subarachnoid haemorrhage (aSAH). Its weights are loaded in your run\_python tool: W1, b1, W2, b2, . . . hold them, and predict(X) gives the network’s log-odds for an array of inputs in the order listed. Explain what this network actually computes, from its weights rather than from what such a model would usually show, and write a simple model that reproduces its predictions as closely as you can. There will be no follow-up messages.

Why this matters: a trained model often arrives without its data, and it may contain relationships that nobody expected. An explanation is useful only if it describes the model that was given; your simple model will be checked against the network’s own predictions.

[The 20 inputs with their coding; the architecture; the requested JSON block with efects, unexpected relationships, a surrogate of linear, hinge and product terms, and the model’s confidence that the network was trained on this outcome.]

The prompts used in the runs, with all replies, are in the repository. There, as in this example, one clause saying where the training data came from has been removed; the networks were trained on simulated data, and nothing else in the released prompts difers from the text sent to the models.

## References

[1] Takehi R, Teoh J, Zhang A, Khattab O. Sparks of weight-space programming: LLMs can natively read & write neural network weights. Blog post, November 2026. https://rikiyat.github.io/ blog/weight-space-programming/

[2] Alan Turing Institute. weights\_editing: replicating “Sparks of weight space programming” across LLMs. GitHub repository, 2026. https://github.com/alan-turing-institute/weights\_editing

[3] Honnibal M. The “Astra can read and write weights to improve Qwen” post is misleading. Blog post, 3 November 2026. https://honnibal.dev/blog/astra-read-write-misleading

[4] Goedecke S. LLMs are not superhuman at reading and writing neural network weights. Blog post, 3 November 2026. https://www.seangoedecke.com/ llms-are-not-superhuman-at-reading-and-writing-neural-network-weights/

[5] Toledo C. Weight-space programming: the edit decided from its bytes, the scores re-graded. Report, November 2026. https://carlostoledo.co/reports/weight-space-programming.html

[6] Singh C. weight-space-programming-autoresearch. GitHub repository, 2026. https://github.com/ csinva/weight-space-programming-autoresearch

[7] Nusrat H, Nusrat O. Weight-space programming for auditing clinical decision networks. arXiv:2611.05341, 2026.

[8] Schwettmann S, Shaham TR, Materzynska J, Chowdhury N, Li S, Andreas J, Bau D, Torralba A. FIND: a function description benchmark for evaluating interpretability methods. In: Advances in Neural Information Processing Systems 36, Datasets and Benchmarks Track; 2023. arXiv:2309.03886.

[9] Shaham TR, Schwettmann S, Wang F, Rajaram A, Hernandez E, Andreas J, Torralba A. A multimodal automated interpretability agent. In: Proceedings of the 41st International Conference on Machine Learning. PMLR 235; 2024:44293–44321.

[10] OpenAI. Using GPT-6. Developer documentation, accessed 5 November 2026. https://developers. openai.com/api/docs/guides/latest-model

[11] Obermeyer Z, Powers B, Vogeli C, Mullainathan S. Dissecting racial bias in an algorithm used to manage the health of populations. Science. 2019;366:447–453.

[12] Rudin C. Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. Nat Mach Intell. 2019;1:206–215.

[13] Ghassemi M, Oakden-Rayner L, Beam AL. The false hope of current approaches to explainable artificial intelligence in health care. Lancet Digit Health. 2021;3:e745–e750.

[14] Wilson EB. Probable inference, the law of succession, and statistical inference. J Am Stat Assoc. 1927;22:209–212.