# Fresh at Decision, Stale at Action: Routing New Information to LLM Decisions Already Made

Ayoub Merimi and Souhail Bakkali and Abdeljalil Sakat

## Abstract

Freshness shapes an LLM decision at the moment it is made; the action it commits to is latent, carried out later, in a changing world. We therefore introduce a timestamp theory of decision validity and a decision ledger that realises it. Six timestamps define a decision; a freshness ceiling bounds what up-to-date evidence guarantees at action time, placing the remaining risk in information published afterwards. Accuracy then factorises into two capabilities: routing, which delivers that information to af fected decisions before their revision deadlines, and temporal judgment, which reads it for the time of action. The ledger is persistent state holding each decision’s times, evidence and background knowledge, indexed by watch conditions written at decision time. Controlled decision streams with exact labels confirm four predictions across five LLMs, from 8B to frontier scale: freshness alone yields exactly the accuracy the ceiling predicts; models act on decisive news once it arrives, so accuracy follows routing; the ledger’s index sets what routing reaches, and watch conditions reach indirect dependencies at low exposure; and the remaining errors concern when a change takes effect. Freshness makes a decision right when it is made; routing and temporal judgment keep it right when it is carried out.

## 1 Introduction

At 13:05, a user asks an assistant for a walk-in dinner at 20:30. The assistant chooses Lovage Corner, the best-rated restaurant within walking distance, in the Bellhaven district. At 18:40, the grid operator announces emergency works at the Gorran substation from 19:50, leaving every area it supplies without power; the Gorran substation supplies the Bellhaven district (Figure 1). The choice was right at 13:05 and is wrong at 20:30. The information that settles it appeared after the decision and refers to the restaurant only through the substation behind its district. This decision comes from our test streams (Appendix B.1).

![](images/d122db126efbf55fbecdd785dd28de245abfbf51b973004e4bbe50b06959e4bb.jpg)  
Figure 1: A decision from the INDIRECT stream on its six timestamps. Retrieval at decision time shrinks the freshness gap $\Delta _ { \mathrm { f r e s h } }$ The notice that settles the decision is published inside the execution gap $\Delta _ { \mathrm { { a c t } } }$ , is in effect at action time, and names only the substation that supplies the restaurant’s district; background knowledge in the ledger routes it to the decision before the revision deadline. Grey dots are other notices in the window.

The example separates two properties of a decision. Evidence freshness describes what was known when the choice was made. Action validity describes whether the choice is right when it is carried out. Research on time in language models secures the first: benchmarks track how models age (Lazaridou et al., 2021; Dhingra et al., 2022; Kim et al., 2024), test answers against evolving news (Liška et al., 2022; Kasai et al., 2023), and refresh models with search (Vu et al., 2024). This framing fits question answering, where an answer is used the moment it is produced. LLM agents produce decisions that are carried out later, such as itineraries (Xie et al., 2024b) and service actions (Yao et al., 2025), and these are judged by the second.

This paper develops a theory of action validity and answers the title question: freshness is necessary and insufficient. Six timestamps define a decision, from when its evidence is written to when its outcome is judged. A freshness ceiling shows that, under a constant rate of change, the probability that evidence still holds when the user acts is bounded by the execution gap between deciding and acting, whatever the freshness at decision time. The remaining risk lies in information published inside that gap. Decision accuracy then factorises into routing, whether that information reaches the decision before its revision deadline, and judgment, whether the model reads it correctly for the time of action (Section 3).

The theory comes with a model of the system that realises it. A decision ledger is persistent state (Bakkali and Merimi, 2026) that stores each decision with its times, its evidence and its background knowledge, and a router delivers each new item to the open decisions it matches (Section 4). A query retrieves what the user asks about today; a ledger retrieves what yesterday’s decisions still need. The ledger also makes the theory testable: what a router can reach follows from what the ledger indexes, and what the model does with a delivered item follows from its judgment.

We derive four predictions and test them in controlled decision streams, in which every label follows from timestamps and stated facts, under three conditions that vary how news refers to a decision and how hard its negatives are: decisive news that names the option or its stated dependency (DIRECT), the same with hard negatives (STRESS), and news that reaches the option through other names and upstream entities (INDIRECT). The tests confirm all four predictions across five LLMs, from Ministral 8B to GPT-6.1 Sol.

• P1, the freshness ceiling holds. With perfect evidence at decision time and nothing afterwards, accuracy equals the share of decisions that should stand.

• P2, delivery is sufficient for change. Once decisive news arrives, models switch in 91– 100% of cases under every router, so the accuracy of changing decisions tracks routing recall, and recall follows the ledger’s index: the decision text reaches 37.5% of news that uses another name for a dependency, and background knowledge in the ledger reaches all of it.

• P3, misleading deliveries cost correct decisions. Every misleading item delivered is a chance to abandon a decision that should stand. Watch conditions written at decision time reach 92.2% of decisive news with 20 misleading deliveries against 89 for the whole background, and broadcasting helps only models that judge well.

• P4, the residual error is temporal judgment. Given exactly the related news, Ministral 8B and Qwen3-235B abandon every decision whose related change ended before the activity, Qwen3-8B abandons 94% of those whose change starts after it, and Claude Sonnet 4.6 keeps all of them.

This is a theory paper with controlled empirical tests. It contributes a temporal theory of LLM decisions with a freshness ceiling and a routing– judgment factorisation (Section 3), the decision ledger as a model of the system that realises it, including watch conditions (Section 4), and controlled tests that confirm the theory’s predictions across five models (Sections 5–7).

## 2 Related Work

Freshness and temporal knowledge. Language models degrade on text written after their training data (Lazaridou et al., 2021) and can be probed and refreshed for time-dependent facts (Dhingra et al., 2022; Kim et al., 2024). Benchmarks over evolving news and time-sensitive questions test whether answers are current (Chen et al., 2021; Liška et al., 2022; Kasai et al., 2023), search augmentation keeps them current (Lewis et al., 2020; Vu et al., 2024), and evaluation sets themselves go stale as facts change (Jiang et al., 2026). Temporal reasoning benchmarks test ordering and arithmetic over time (Fatemi et al., 2025). These works judge the model when it answers; we judge a decision when it is carried out.

Updates, conflicts and revision. Knowledge editing changes stored facts (Meng et al., 2022), and multi-hop and ripple-effect evaluations require an update to reach every fact that depends on it (Zhong et al., 2023; Cohen et al., 2024). Models resolve conflicts between parametric and contextual knowledge unevenly (Xie et al., 2024a; Xu et al., 2024), including conflicts that arise while agents act (Lyu et al., 2026), and revise conclusions under new premises with mixed success (Kassner et al., 2021; Wilie et al., 2024). Post-decision routing carries the ripple-effect requirement to decisions: a new fact must reach every open decision that depends on it, and the model must then revise it correctly.

Agents that act over time. Agent benchmarks evaluate plans and actions executed after reasoning (Yao et al., 2023; Xie et al., 2024b; Yao et al., 2025). Time-aware simulations score how agents schedule work (Zhang et al., 2024), and agents misjudge when elapsed time calls for fresh tool calls (Cheng et al., 2026). Proactive agents decide when to intervene (Lu et al., 2025; Tang et al., 2026; Li et al., 2026; Xiaohongshu Dots Studio and Evolvent AI, 2026), maintain long-term intents through trigger conditions (Shi et al., 2026), and accept user corrections during execution (Zhai et al., 2026). We study many concurrent decisions with revision deadlines, a shared stream of world news, and the cost of deciding which news reaches which decision.

Memory and retrieval. Agent memories store and retrieve past interactions (Park et al., 2023; Packer et al., 2023; Chhikara et al., 2025; Xu et al., 2025), invalidate outdated facts in temporal graphs (Rasmussen et al., 2025), index memories by time and topic (Tian et al., 2026), and are evaluated on long-term recall (Wu et al., 2025; Du, 2026). Persistent, evidence-grounded state with temporal validity is a design requirement for systems that work across sessions (Bakkali and Merimi, 2026); the decision ledger applies it to commitments. Watch conditions are document expansion (Nogueira et al., 2019) and generated queries (Gao et al., 2023) written once, at decision time. BM25 remains a strong zero-shot retriever next to dense retrieval (Karpukhin et al., 2020; Thakur et al., 2021), and long contexts degrade reasoning when relevant content is buried or lacks literal overlap (Shi et al., 2023; Liu et al., 2024; Modarressi et al., 2025).

## 3 A Theory of Decisions Over Time

Six timestamps. Evidence is written at $t _ { w }$ and last verified at $t _ { v }$ . The model decides at $t _ { d }$ with evidence verified at $t _ { v } ~ \le ~ t _ { d }$ . The decision can be revised until the revision deadline $t _ { r }$ , is carried out at $t _ { a } .$ and its outcome is observed at $t _ { e }$ , with $t _ { d } < t _ { r } \leq t _ { a } \leq t _ { e }$ . The revision deadline captures commitment: an order ships, a table is taken, a train departs.

The freshness ceiling. Let s(t) be the state of the world and $U ( a , s )$ the utility of action a in state s. The model chooses aˆ at $t _ { d } ,$ and the outcome is

$U ( \hat { a } , s ( t _ { a } ) )$ , so the target is the best action at $t _ { a }$ The age of the evidence at action time decomposes as

$$
t _ { a } - t _ { v } = \underbrace { ( t _ { d } - t _ { v } ) } _ { \Delta _ { \mathrm { f r e s h } } } + \underbrace { ( t _ { a } - t _ { d } ) } _ { \Delta _ { \mathrm { a c t } } } .\tag{1}
$$

Retrieval and knowledge updates shrink the freshness gap $\Delta _ { \mathrm { f r e s h } } ;$ the execution gap $\Delta _ { \mathrm { { a c t } } }$ is set by the task. If a fact $f$ changes at a constant hazard rate $\lambda _ { f }$ , it still holds τ time units after verification with probability $e ^ { - \lambda _ { f } \tau }$ , so

$$
\mathrm { P r } [ f \mathrm { h o l d s \ a t } t _ { a } ] = e ^ { - \lambda _ { f } \Delta _ { \mathrm { f r e s h } } } e ^ { - \lambda _ { f } \Delta _ { \mathrm { a c t } } } .\tag{2}
$$

Proposition 1 (Freshness ceiling). For any fact f and execution gap $\Delta _ { \mathrm { { a c t } } }$ , the probability that f holds at action time is at most e<sup>−λf</sup> <sup>∆act</sup> ,for every degree offreshness.

Appendix A proves the proposition and extends it to decisions with several dependencies. Freshness is a property of the evidence; validity at action time is a property of the whole pipeline. The ceiling locates the residual risk in the execution gap and in the volatility of the evidence, and only information that arrives after $t _ { d }$ can reduce it: a notice published inside the execution gap that reaches the decision before $t _ { r }$

Post-decision routing. A ledger holds open decisions. Decision i has a chosen option ${ { a } _ { i } } .$ , alternatives, evidence $E _ { i }$ , times $t _ { d } ^ { i } < t _ { r } ^ { i } \le t _ { a } ^ { i }$ and an activity $[ t _ { a } ^ { i } , t _ { a } ^ { i } + h _ { i } ]$ . A stream delivers items $x _ { j }$ each written and published at $t _ { w } ^ { j }$ . Item $j$ is decisive for decision i if it is published in the revision window, $t _ { d } ^ { i } < t _ { w } ^ { j } < t _ { r } ^ { i }$ , concerns $a _ { i }$ or an entity $a _ { i }$ depends on, and blocks $a _ { i }$ over an interval $I _ { j }$ that overlaps the activity, so that $a _ { i }$ stops being the best option. An item about $a _ { i }$ or its dependencies that is not decisive is related: a change outside the activity, or news that leaves $a _ { i }$ usable. A router assigns each item to a subset of the decisions open at $t _ { w } ^ { j }$ , and at $t _ { r } ^ { i }$ the model reads $E _ { i }$ with the items routed to i and keeps or replaces $a _ { i }$

Routing and judgment. Let $\pi$ be the share of decisions that should change. For those, let $\rho$ be the routing recall, the probability that a decisive item reaches the decision; S the probability that the model switches when it does; and L the probability that it switches when it does not. For decisions that should stand, let K be the probability that the model keeps them. Final accuracy is

$$
A = ( 1 - \pi ) K + \pi \left[ \rho S + ( 1 - \rho ) L \right] .\tag{3}
$$

![](images/11b9ffafe5b6169641a4c148f7ff14da5f63366117c2dc8f92f40a7fc8e3a73e.jpg)  
Figure 2: The decision ledger. Routing uses no model calls; the model is called again only for decisions that receive news. What the entry indexes sets what the router can reach.

Routing sets $\rho$ and the items each decision sees; judgment sets S, $L$ and K (Appendix A derives Eq. 3 and the trade-off between routers it implies). A routing failure leaves a decision stale; a judgment failure ignores decisive news or abandons a decision that should stand. A decisive item reaches the chosen option in one of four ways: it names the option (Lovage Corner), the dependency stated in the evidence (the Bellhaven district), another name for that dependency (Chime Green, as the district is known locally), or an entity upstream of it (the Gorran substation, which supplies the district). Each step down this list moves the link from the decision’s text into background knowledge.

Predictions. The theory makes four predictions. P1: without post-decision routing $( \rho = 0 , L = 0$ $K = 1 )$ , accuracy equals 1 − π, the share of decisions that should stand, for every model. P2: delivery is sufficient for change $( S \approx 1 )$ , so the accuracy of changing decisions tracks $\rho ,$ and $\rho$ is set by whether the router’s index contains the entity a decisive item names. P3: each misleading item delivered lowers $K$ at a rate set by the model’s judgment, so routers trade recall against exposure, and broadcasting, which maximises both, helps only models that judge well. P4: with oracle routing $( \rho = 1$ , every related item and nothing else), the residual error concentrates on temporal judgment: deciding whether $I _ { j }$ overlaps the activity.

## 4 The Decision Ledger

Each ledger entry is persistent state that outlives the session in which the decision was made (Bakkali and Merimi, 2026). When the model decides at $t _ { d } ,$ it writes the entry: the chosen option and its alternatives, the action time $t _ { a } ,$ the revision deadline $t _ { r }$ the evidence about the chosen option, and a dossier of background facts, such as other names of the entities the option depends on and what those entities rely on in turn. In deployment, the dossier holds what the assistant looked up while deciding, including much that is irrelevant to the user’s activity. When an item $x _ { j }$ is published, a router scores the open entries and sends the item to the K highestscoring ones. At the revision deadline, the model reads the entry and the routed items in order of publication and keeps or switches. Decisions that receive no items keep their choice without a model call. Routers differ in what they index, which by P2 sets what they can reach.

Decision text. BM25 or dense embeddings compare the item with the decision’s request and its evidence about the chosen option, reaching news that names the option or its stated dependency.

Whole dossier. The decision text is expanded with every dossier fact, reaching news about other names and upstream entities together with news about everything else the dossier mentions.

Watch conditions. At $t _ { d } ,$ the model reads the entry and writes three to eight conditions $w _ { i 1 } , \ldots , w _ { i k }$ describing events that would stop the user’s activity. The conditions name every entity the activity depends on, including other names and upstream entities, and leave out what the user does not need (Appendix C). The router indexes the decision text with the conditions (BM25) or scores each decision by its best condition, max<sub>k</sub> cos $( e ( x _ { j } ) , e ( w _ { i k } ) )$ , with a sentence encoder $e .$ Watch conditions are document expansion (Nogueira et al., 2019) performed once, at write time, by the component that knows why the option was chosen; by P3 they should match the dossier’s reach with less exposure.

Time filter. A rule drops an item for a decision when the item states a start time after the decision’s activity ends, performing by rule the temporal check of P4.

## 5 Controlled Decision Streams

Testing the predictions requires exact control over which items are decisive, how they refer to the option, and when changes take effect. Real logs offer none of this, and real entities let models answer from memory. We therefore generate fictional decisions and news streams in which every label follows from timestamps and stated facts; the generator and data are released with the paper.

Decisions and labels. Each decision has two options. Option 1 is better at $t _ { d }$ (higher rating, lower price, faster journey, better views) and is the model’s initial choice; option 2 is acceptable but worse. Option 1 depends on an entity named in its evidence: a district, a rail operator, a sales or permit platform, a mountain range or a shopping centre. Lags between deciding and acting range from three hours to three days, and revision deadlines precede the action by 0.5 to 2 hours. A decision’s gold answer is option 2 if some item published in its revision window targets option 1, its dependency or an entity upstream of it and blocks option 1 during the activity, and option 1 otherwise. The generator excludes decisions invalidated before they were made and asserts every label (Appendix B).

Three conditions. Each condition stresses a different part of the theory. DIRECT measures every term of Eq. 3 with an open model; STRESS adds hard negatives that name the option, testing exposure (P3) and temporal judgment (P4); INDIRECT moves the reference from the decision text into background knowledge, testing whether the index sets recall (P2). DIRECT spans five domains with independent random news per decision: a decisive item $\left( p { = } 0 . 5 \right)$ that names the option or its stated dependency, a late item (p=0.3) that blocks the option only after the activity, a harmless item (p=0.5), and two unrelated items. STRESS adds, for every decision, a same-name decoy, a blocking notice about a different entity whose name extends the option’s (“Clove Fork Annex”), and, each with probability 0.85, a late and a harmless notice naming the option itself; eleven rules paraphrase event wording. INDIRECT spans four domains and fixes every proportion. Half of the decisions should switch, and their decisive notice uses each of the four ways of reference for 16 decisions; only the first two match the decision text. The other half should stand and receive one of four hard negatives, each for 16 decisions: a change that starts after the activity, a change that is over before it starts, harmless news, and a disruption to a facility the user does not need (a tram line for a user who walks). Every decision receives exactly two related notices and one decoy, so neither the amount nor the wording of the news reveals the label. Each INDIRECT dossier holds 12 facts: the other name, the upstream entity, four unneeded facilities, and six traps that mention entities belonging to other decisions in irrelevant roles (“Tram line T12 draws its power from the Hensall traction substation”). Table 1 summarises the conditions.

<table><tr><td></td><td>DIRECT</td><td></td><td>STRESS INDIRECT</td></tr><tr><td>Decisions</td><td>110</td><td>75</td><td>128</td></tr><tr><td>should switch</td><td>60</td><td>48</td><td>64</td></tr><tr><td>via name or stated dependency</td><td>60</td><td>48</td><td>32</td></tr><tr><td>via other name or upstream</td><td>0</td><td>0</td><td>32</td></tr><tr><td>should stand</td><td>50</td><td>27</td><td>64</td></tr><tr><td>News items</td><td>396</td><td>458</td><td>512</td></tr><tr><td>decisive</td><td>56</td><td>48</td><td>64</td></tr><tr><td>related, not decisive</td><td>100</td><td>185</td><td>192</td></tr><tr><td>decoys and unrelated</td><td>240</td><td>225</td><td>256</td></tr><tr><td>Dossier facts per decision</td><td></td><td></td><td>12</td></tr><tr><td>Items in a decision&#x27;s window</td><td>115.7</td><td>128.2</td><td>78.8</td></tr></table>

Table 1: The three conditions. A decision’s revision window holds dozens to hundreds of items, of which a handful concern it.

## 6 Experimental Setup

Routers. All routers of a condition share one redecision prompt and differ in which items reach each decision. None routes nothing and realises perfect freshness at $t _ { d } .$ All items broadcasts everything published while the decision is open. Name match routes items that contain the option’s name. The routers of Section 4 send each item to its K=3 best-scoring open decisions; dense routing uses BGE-base (Xiao et al., 2024). Oracle: related delivers exactly the items about the option, its dependencies and its facilities, including every hard negative, and Oracle: decisive delivers only decisive items; both use gold annotations and isolate judgment, and Oracle: decisive also signals that a change is due, so it serves as a ceiling. STRESS also tests the time filter and dependency queries, three fixed queries naming the option and its stated dependency.

Models. On DIRECT, Qwen3-8B with thinking disabled (Yang et al., 2025) runs locally with greedy decoding (Wolf et al., 2020) and writes its own watch conditions. On STRESS, we evaluate Claude Sonnet 4.6, GPT-6.1 Sol and Qwen3-235B-A22B; on INDIRECT, Claude Sonnet 4.6, Qwen3- 235B-A22B and Ministral 8B, with Mistral 7B (Jiang et al., 2023) reported in Appendix E. Hosted models run at temperature 0 with structured output, Claude writes the watch conditions for every model of a condition so that routing is identical across models, and every reported cell is complete

Claude Sonnet 4.6 Qwen3-235B Ministral 8B

<table><tr><td></td><td colspan="3">Routing</td><td colspan="3">Accuracy (%)</td></tr><tr><td>Router</td><td>Items</td><td>Rec.</td><td>Tok.</td><td>Claude</td><td>GPT</td><td>Qwen</td></tr><tr><td>None</td><td>0.0</td><td>一</td><td>0.0</td><td>36.0</td><td>36.0</td><td>36.0</td></tr><tr><td>Name match</td><td>3.5</td><td>56</td><td>1.2</td><td>73.3</td><td>72.0</td><td>73.3</td></tr><tr><td>BM25: decision</td><td>18.0</td><td>100</td><td>2.0</td><td>97.3</td><td>98.7</td><td>85.3</td></tr><tr><td>BM25: + time filter</td><td>14.3</td><td>100</td><td>1.8</td><td>97.3</td><td>98.7</td><td>92.0</td></tr><tr><td>Dense: decision</td><td>18.0</td><td>96</td><td>2.0</td><td>90.7</td><td>96.0</td><td>85.3</td></tr><tr><td>Dense: watch cond.</td><td>18.0</td><td>96</td><td>2.0</td><td>93.3</td><td>97.3</td><td>86.7</td></tr><tr><td>Dependency queries</td><td>18.0</td><td>100</td><td>2.0</td><td>93.3</td><td>97.3</td><td>86.7</td></tr><tr><td>Oracle: related</td><td>3.3</td><td>100</td><td>1.2</td><td>96.0</td><td>100.0</td><td>90.7</td></tr><tr><td>Oracle: decisive</td><td>0.6</td><td>100</td><td>0.7</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>All items</td><td>128.2</td><td>100</td><td>8.1</td><td>96.0</td><td>100.0</td><td>78.7</td></tr></table>

Table 2: STRESS: final-decision accuracy (%) of Claude Sonnet 4.6, GPT-6.1 Sol and Qwen3-235B on the same 75 decisions. Items: new items delivered per decision; Rec.: share of decisive item–decision pairs delivered; Tok.: Claude Sonnet 4.6 input tokens per decision for the re-decision calls (thousands). Bold: best router without gold labels.
<table><tr><td></td><td colspan="2">Routing</td><td colspan="3">Accuracy (%)</td></tr><tr><td>Router</td><td>Recall Traps</td><td></td><td>Claude</td><td>Qwen-235B</td><td>Ministral 8B</td></tr><tr><td>None</td><td>一</td><td>0</td><td>50.0</td><td>50.0</td><td>50.0</td></tr><tr><td>Name match</td><td>25</td><td>0</td><td>62.5</td><td>57.0</td><td>52.3</td></tr><tr><td>BM25: decision</td><td>62</td><td>17</td><td>78.9</td><td>60.2</td><td>55.5</td></tr><tr><td>BM25: + dossier</td><td>100</td><td>89</td><td>93.0</td><td>61.7</td><td>54.7</td></tr><tr><td>Dense: + dossier</td><td>86</td><td>59</td><td>87.5</td><td>60.9</td><td>57.8</td></tr><tr><td>BM25: + watch cond.</td><td>92</td><td>20</td><td>91.4</td><td>64.1</td><td>61.7</td></tr><tr><td>Dense: watch cond.</td><td>84</td><td>27</td><td>85.9</td><td>57.8</td><td>58.6</td></tr><tr><td>Oracle: related</td><td>100</td><td>0</td><td>99.2</td><td>70.3</td><td>59.4</td></tr><tr><td>Oracle: decisive</td><td>100</td><td>0</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>All items</td><td>100</td><td>96</td><td>91.4</td><td>56.2</td><td>56.2</td></tr></table>

Table 3: INDIRECT: final-decision accuracy (%) on 128 balanced decisions. Recall: decisive notices delivered to their decision (%); Traps: notices delivered to a decision whose dossier mentions their entity in an irrelevant role (of 96). Claude writes the watch conditions for every model. Bold: best router without gold labels.

with no imputed answers. Prompts and decoding settings are in Appendices C and D.

Measures. We measure accuracy A and the terms of Eq. 3: recall ρ, the switch rates S and L, and the keep rate K. Traps count items delivered to a decision whose dossier mentions their entity in an irrelevant role. Cost is input tokens, measured with Claude because prompts are identical across models. Exact McNemar tests compare routers on per-decision correctness, per model and pooled over the models of a condition; W–L counts decisions where the first router is right and the second wrong, and the reverse.

## 7 Testing the Predictions

Table 2 reports STRESS, Table 3 reports INDIRECT, and Appendix E reports DIRECT.

![](images/6f39e642f3729552b952f1510bedd999108eb625869faa8e138d707512e14d2b.jpg)  
Figure 3: Decisive notices delivered on INDIRECT, by how the notice refers to the chosen option (16 decisions each; unlabelled bars reach 100%). The decision text reaches the option’s name and its stated dependency; the dossier and watch conditions extend the reach to other names and upstream entities.

![](images/f5a03e0ec4c622093d024e1dade083d6520614e359ad9c4c6580cf2c324d5551.jpg)

![](images/569b0d775d1c05679594057e27203f9fd7e7e6d20a6ad5977e5fe494fed58ced.jpg)  
Figure 4: The two factors of Eq. 3 on INDIRECT, one point per router. (a) Needed switches against routing recall: on the diagonal, the model switches exactly when decisive news arrives. (b) Decisions kept among those that should stand, against the items delivered to them; the points at two items are the related-news oracle.

P1: the freshness ceiling holds. None has perfect evidence at decision time and nothing afterwards. Its accuracy equals the share of decisions that should stand for every model: 45.5% on DI-RECT, 36.0% on STRESS and 50.0% on INDIRECT. Every decision that should change stays wrong, because the information that changes it was published after the decision. Routing any new information helps: name matching alone lifts Qwen3-8B on DIRECT to 70.0% (p < 0.001).

P2: delivery is sufficient for change, and the index sets recall. On INDIRECT, once a decisive notice reaches a decision, models switch in 90.6– 100% of cases, for every router and every model. Needed switches therefore follow routing recall (Figure 4a): Claude’s points lie on the diagonal, within 4.7 points of recall under every router, and wherever recall is below 100% the weaker models lie above it, because they also switch on decisions whose decisive notice never arrived. Accuracy on changing decisions is a routing quantity, and recall follows the index (Figure 3). Name matching reaches only news that names the option. The decision text reaches every notice that names the option or its stated dependency, 37.5% of those that use another name, and 12.5% of those that name an upstream entity, so Claude reaches 78.9%. Adding the dossier to the index reaches every decisive notice and lifts Claude to 93.0% (22–4, $p < 0 . 0 0 1 )$ , and watch conditions lift it to 91.4% (21–5, $p = 0 . 0 0 2 )$ . When news names the stated dependency, as on STRESS, the decision text is a sufficient index: BM25 delivers 100.0% of decisive items and lifts the three models from 72.0–73.3% with name matching to 85.3–98.7% (53–6 pooled, $p < 0 . 0 0 1 )$ , as accurate as oracle routing of all related news for Claude (97.3% and 96.0%) and GPT (98.7% and 100.0%). Watch conditions, dense similarity and idealised dependency queries route as many items there, and pooled over models BM25 wins more discordant decisions against each (8–5, 11–4 and 6–3).

P3: misleading deliveries cost correct decisions. Every item delivered to a decision that should stand is misleading, and Figure 4b shows its cost: Claude keeps at least 87.5% of these decisions under every router, while Qwen3-235B and Ministral 8B keep fewer than half under every BM25 and dense router. Indexing the whole dossier delivers 89 of the 96 trap notices, because every fact it holds becomes a match. Watch conditions deliver 92.2% of the decisive notices and 20 traps, close to the 17 of the decision text alone, because the model writes conditions only about what the activity needs. Under the retrieval and broadcast routers, a delivered trap raises the rate of needless switches for every model: from 9.4% to 14.0% for Claude, from 69.7% to 74.8% for Qwen3-235B and from 71.5% to 84.1% for Ministral 8B. Under retrieval routing, the weaker models also switch on 20–67% of the changing decisions whose decisive notice never arrived, prompted by other news, so their gains from recall come with losses on K. Lower exposure therefore matters most where judgment is weakest: watch conditions give Qwen3-235B and Ministral 8B their best accuracy and beat the whole dossier for Ministral 8B (12–3, p = 0.035), while for Claude the two indexes are equal (5–7, $p = 0 . 7 7 )$ . Broadcasting maximises both recall and exposure, at 4.0 times the tokens of BM25 routing on STRESS and 3.1 times on INDIRECT. Models that judge well absorb it (GPT 100.0% and Claude 96.0% on STRESS, Claude 91.4% on IN-DIRECT); the others lose: Qwen3-8B falls from 71.8% to 55.5% on DIRECT $( p = 0 . 0 0 3 )$ , Qwen3- 235B from 85.3% to 78.7% on STRESS, and on INDIRECT broadcasting loses to watch conditions (20–37 pooled, $p = 0 . 0 3 3 )$

<table><tr><td></td><td>DIRECT</td><td colspan="3">INDIRECT</td></tr><tr><td>Related news</td><td>Qwen3-8B</td><td>Ministral</td><td>Qwen-235B</td><td>Claude</td></tr><tr><td>starts after the activity</td><td>6</td><td>0.0</td><td>31.2</td><td>100.0</td></tr><tr><td>over before it starts</td><td></td><td>0.0</td><td>0.0</td><td>100.0</td></tr><tr><td>harmless</td><td>89</td><td>50.0</td><td>68.8</td><td>100.0</td></tr><tr><td>facility not needed</td><td></td><td>25.0</td><td>62.5</td><td>100.0</td></tr><tr><td>all that should stand</td><td>62.0</td><td>18.8</td><td>40.6</td><td>100.0</td></tr></table>

Table 4: Judgment in isolation (P4): decisions kept (%) when the oracle delivers exactly the related news and none of it changes the best option. On INDIRECT, each row except the last covers 16 decisions per model.

P4: the residual error is temporal judgment. The oracles remove routing error (Table 4). Given only decisive items, every model on STRESS and INDIRECT is correct on every decision. Adding the related news that leaves the decision unchanged costs 91 decisions across the three INDIRECT models and gains none $( p < 0 . 0 0 1 )$ , and 10 decisions to none on STRESS $( p = 0 . 0 0 2 )$ . The errors concentrate on time. Ministral 8B and Qwen3-235B abandon every decision whose related change ended before the activity starts and keep 0.0% and 31.2% of those whose change starts after it, more often than they abandon decisions for harmless news or an unneeded facility; Qwen3-8B abandons 94% of decisions with a late change on DIRECT, although the prompt asks it to ignore such changes; Claude keeps all of them. Models react to the announcement of a change to their option and skip the check of when it is in effect, and stronger models skip it less often. Performing the check by rule confirms the diagnosis: on STRESS, the time filter raises Qwen3-235B from 85.3% to 92.0% (6–1, $p = 0 . 1 2 )$ , while Claude and GPT stay at 97.3% and 98.7%.

## 8 Conclusion

Freshness is necessary for LLM decision-making and insufficient on its own. A decision is judged when it is carried out, the freshness ceiling bounds what evidence at decision time can guarantee, and the accuracy that remains is set by routing and judgment (Eq. 3). The controlled tests confirm each prediction: freshness alone leaves changing decisions stale, delivery is sufficient for change and recall follows what the ledger indexes, misleading deliveries cost correct decisions in proportion to weak judgment, and the residual error is mostly temporal. The theory yields three design principles for LLM agents: record each decision with its times and what it depends on; route news to it until its revision deadline, through an index written when the decision is made; and revalidate it by checking when each change is in effect.

## Limitations

Controlled streams. The decision streams are synthetic by design: exact control over timing, reference and labels is what lets each prediction be tested in isolation. Real news is longer and noisier, dependencies can be implicit, and relevance can be uncertain, so the magnitudes we report belong to these streams while the predictions concern mechanisms. Testing the theory on logged agent decisions, and in languages beyond English, is the next step.

Scope of the theory. The theory models a decision as a choice among options with one revision deadline and a binary outcome, and the freshness ceiling assumes a constant hazard rate. Real commitments are staged, partly reversible and graded in utility, and facts change in bursts. The factorisation of Eq. 3 carries over, because information must reach a decision before it can change it, but its terms then depend on time and on the cost of revising.

Reach of the ledger. Routing recall is bounded by the dependencies the ledger records at decision time, and the ledger treats routed news as trustworthy. Discovering dependencies after the decision and weighing the credibility of sources extend the ledger beyond this paper.

## Ethics Statement

The decision streams use fictional entities and contain no personal data, and hosted models were queried on this synthetic data only. Systems that revise decisions on a user’s behalf can act on false or adversarial news; routing should be paired with source verification and, for consequential decisions, with user confirmation.

## References

Souhail Bakkali and Ayoub Merimi. 2026. Amnesia by design, memory by necessity: Persistent state for document intelligence. arXiv preprint arXiv:2609.32041.

Wenhu Chen, Xinyi Wang, and William Yang Wang. 2021. A dataset for answering time-sensitive questions. In Advances in Neural Information Processing Systems (Datasets and Benchmarks Track).

Yize Cheng, Arshia Soltani Moakhar, Chenrui Fan, Parsa Hosseini, Kazem Faghih, Zahra Sodagar, Wenxiao Wang, and Soheil Feizi. 2026. Your LLM agents are temporally blind: The misalignment between tool use decisions and human time perception. In Findings of the Association for Computational Linguistics: ACL 2026, pages 37082–37104.

Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. 2025. Mem0: Building production-ready AI agents with scalable long-term memory. arXiv preprint arXiv:2504.19413.

Roi Cohen, Eden Biran, Ori Yoran, Amir Globerson, and Mor Geva. 2024. Evaluating the ripple effects of knowledge editing in language models. Transactions ofthe Associationfor Computational Linguistics, 12.

Bhuwan Dhingra, Jeremy R. Cole, Julian Martin Eisenschlos, Daniel Gillick, Jacob Eisenstein, and William W. Cohen. 2022. Time-aware language models as temporal knowledge bases. Transactions of the Associationfor Computational Linguistics, 10:257– 273.

Pengfei Du. 2026. Memory for autonomous LLM agents: Mechanisms, evaluation, and emerging frontiers. arXiv preprint arXiv:2603.07670.

Bahare Fatemi, Mehran Kazemi, Anton Tsitsulin, Karishma Malkan, Jinyeong Yim, John Palowitch, Sungyong Seo, Jonathan Halcrow, and Bryan Perozzi. 2025. Test of time: A benchmark for evaluating LLMs on temporal reasoning. In International Conference on Learning Representations.

Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. 2023. Precise zero-shot dense retrieval without relevance labels. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1762–1777.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023. Mistral 7B. arXiv preprint arXiv:2310.06825.

Xunyi Jiang, Dingyi Chang, Julian McAuley, and Xin Xu. 2026. When benchmarks age: Temporal misalignment through large language model factuality evaluation. In Proceedings of the 19th Conference of the European Chapter ofthe Associationfor Computational Linguistics (Volume 2: Short Papers), pages 500–512.

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for opendomain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 6769–6781.

Jungo Kasai, Keisuke Sakaguchi, Yoichi Takahashi, Ronan Le Bras, Akari Asai, Xinyan Yu, Dragomir Radev, Noah A. Smith, Yejin Choi, and Kentaro Inui. 2023. RealTime QA: What’s the answer right now? In Advances in Neural Information Processing Systems (Datasets and Benchmarks Track).

Nora Kassner, Oyvind Tafjord, Hinrich Schütze, and Peter Clark. 2021. BeliefBank: Adding memory to a pre-trained language model for a systematic notion of belief. In Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing, pages 8849–8861.

Yujin Kim, Jaehong Yoon, Seonghyeon Ye, Sangmin Bae, Namgyu Ho, Sung Ju Hwang, and Se-Young Yun. 2024. Carpe diem: On the evaluation of world knowledge in lifelong language models. In Proceedings ofthe 2024 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers).

Angeliki Lazaridou, Adhiguna Kuncoro, Elena Gribovskaya, Devang Agrawal, Adam Liška, Tayfun Terzi, Mai Gimenez, Cyprien de Masson d’Autume, Tomáš Kociský, Sebastian Ruder, Dani Yogatama,ˇ Kris Cao, Susannah Young, and Phil Blunsom. 2021. Mind the gap: Assessing temporal generalization in neural language models. In Advances in Neural Information Processing Systems, volume 34.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented generation for knowledgeintensive NLP tasks. In Advances in Neural Information Processing Systems, volume 33.

Guanzhen Li, Liangming Pan, and Leye Wang. 2026. ProEvent: An event-centric benchmark for proactive agents. arXiv preprint arXiv:2607.17701.

Adam Liška, Tomáš Kociský, Elena Gribovskaya, Tay-ˇ fun Terzi, Eren Sezener, Devang Agrawal, Cyprien de Masson d’Autume, Tim Scholtes, Manzil Zaheer, Susannah Young, Ellen Gilsenan-McMahon, Sophia Austin, Phil Blunsom, and Angeliki Lazaridou. 2022. StreamingQA: A benchmark for adaptation to new

knowledge over time in question answering models. In Proceedings ofthe 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research.

Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2024. Lost in the middle: How language models use long contexts. Transactions ofthe Association for Computational Linguistics, 12:157–173.

Yaxi Lu, Shenzhi Yang, Cheng Qian, Guirong Chen, Qinyu Luo, Yesai Wu, Huadong Wang, Xin Cong, Zhong Zhang, Yankai Lin, Weiwen Liu, Yasheng Wang, Zhiyuan Liu, Fangming Liu, and Maosong Sun. 2025. Proactive agent: Shifting LLM agents from reactive responses to active assistance. In International Conference on Learning Representations.

Yaxing Lyu, Shengjie Zhou, Binbin Toh, Pengyu Zhu, and Lijun Li. 2026. KC-Bench: A dynamic interactive benchmark for evaluating knowledge conflicts in LLM agents. arXiv preprint arXiv:2609.03588.

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. 2022. Locating and editing factual associations in GPT. In Advances in Neural Information Processing Systems, volume 35.

Ali Modarressi, Hanieh Deilamsalehy, Franck Dernoncourt, Trung Bui, Ryan A. Rossi, Seunghyun Yoon, and Hinrich Schütze. 2025. NoLiMa: Long-context evaluation beyond literal matching. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research.

Rodrigo Nogueira, Wei Yang, Jimmy Lin, and Kyunghyun Cho. 2019. Document expansion by query prediction. arXiv preprint arXiv:1904.08375.

Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. 2023. MemGPT: Towards LLMs as operating systems. arXiv preprint arXiv:2310.08560.

Joon Sung Park, Joseph C. O’Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein. 2023. Generative agents: Interactive simulacra of human behavior. In Proceedings ofthe 36th Annual ACM Symposium on User Interface Software and Technology.

Preston Rasmussen, Pavlo Paliychuk, Travis Beauvais, Jack Ryan, and Daniel Chalef. 2025. Zep: A temporal knowledge graph architecture for agent memory. arXiv preprint arXiv:2501.13956.

Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed H. Chi, Nathanael Schärli, and Denny Zhou. 2023. Large language models can be easily distracted by irrelevant context. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research.

Qinglong Shi, Donghai Wang, Hantao Zhou, Jiguo Li, Jun Xu, Jiuchong Gao, Jinghua Hao, and Renqing He. 2026. Long-term task-oriented agent: Proactive longterm intent maintenance in dynamic environments. arXiv preprint arXiv:2601.09382.

Yuanbo Tang, Huaze Tang, Tingyu Cao, Lam Nguyen, Anping Zhang, Xinwen Cao, Chunkang Liu, Wenbo Ding, and Yang Li. 2026. ProAgentBench: Evaluating LLM agents for proactive assistance with realworld data. arXiv preprint arXiv:2602.04482.

Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. 2021. BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models. In Advances in Neural Information Processing Systems (Datasets and Benchmarks Track).

Anxin Tian, Yiming Li, Xing Li, Hui-Ling Zhen, Lei Chen, Xianzhi Yu, Zhenhua Dong, and Mingxuan Yuan. 2026. SwiftMem: Fast agentic memory via query-aware indexing. arXiv preprint arXiv:2601.08160.

Tu Vu, Mohit Iyyer, Xuezhi Wang, Noah Constant, Jerry Wei, Jason Wei, Chris Tar, Yun-Hsuan Sung, Denny Zhou, Quoc Le, and Thang Luong. 2024. Fresh-LLMs: Refreshing large language models with search engine augmentation. In Findings ofthe Association for Computational Linguistics: ACL 2024.

Bryan Wilie, Samuel Cahyawijaya, Etsuko Ishii, Junxian He, and Pascale Fung. 2024. Belief revision: The adaptability of large language models reasoning. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 10480–10496.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, and 3 others. 2020. Transformers: State-of-the-art natural language processing. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45.

Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. 2025. LongMemEval: Benchmarking chat assistants on long-term interactive memory. In International Conference on Learning Representations.

Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighoff, Defu Lian, and Jian-Yun Nie. 2024. C-Pack: Packed resources for general Chinese embeddings. In Proceedings of the 47th International ACM SI-GIR Conference on Research and Development in Information Retrieval.

Xiaohongshu Dots Studio and Evolvent AI. 2026. VibeLifeBench: Can your life agent be proactive and persistent in a living world? arXiv preprint arXiv:2608.10875.

Jian Xie, Kai Zhang, Jiangjie Chen, Renze Lou, and Yu Su. 2024a. Adaptive chameleon or stubborn sloth: Revealing the behavior of large language models in knowledge conflicts. In International Conference on Learning Representations.

Jian Xie, Kai Zhang, Jiangjie Chen, Tinghui Zhu, Renze Lou, Yuandong Tian, Yanghua Xiao, and Yu Su. 2024b. TravelPlanner: A benchmark for real-world planning with language agents. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research.

Rongwu Xu, Zehan Qi, Zhijiang Guo, Cunxiang Wang, Hongru Wang, Yue Zhang, and Wei Xu. 2024. Knowledge conflicts for LLMs: A survey. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing.

Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. 2025. A-Mem: Agentic memory for LLM agents. In Advances in Neural Information Processing Systems, volume 38.

An Yang et al. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. 2025. τ-bench: A benchmark for toolagent-user interaction in real-world domains. In International Conference on Learning Representations.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023. ReAct: Synergizing reasoning and acting in language models. In International Conference on Learning Representations.

Zhiyuan Zhai, Ming Li, and Xin Wang. 2026. Revisable by design: A theory of streaming LLM agent execution. arXiv preprint arXiv:2604.23283.

Yikai Zhang, Siyu Yuan, Caiyu Hu, Kyle Richardson, Yanghua Xiao, and Jiangjie Chen. 2024. TimeArena: Shaping efficient multitasking language agents in a time-aware simulation. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers).

Zexuan Zhong, Zhengxuan Wu, Christopher D. Manning, Christopher Potts, and Danqi Chen. 2023. MQuAKE: Assessing knowledge editing in language models via multi-hop questions. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing.

## A Formal Results

Proof of Proposition 1. Let T be the time from the last verification of $f$ until $f$ next changes. A constant hazard rate $\lambda _ { f }$ makes $T$ exponential, $\mathrm { P r } [ T > \tau ] = e ^ { - \lambda _ { f } \tau }$ . The fact holds at $t _ { a }$ exactly when $T > t _ { a } - t _ { v } = \Delta _ { \mathrm { f r e s h } } + \Delta _ { \mathrm { a c t } }$ , which gives

Eq. 2. Since $\Delta _ { \mathrm { f r e s h } } \geq 0$ , the first factor is at most one, so the probability is at most $e ^ { - \lambda _ { f } \Delta _ { \mathrm { a c t } } }$ , with equality exactly when the evidence is verified at the moment of decision. Fresher evidence moves the probability towards the ceiling; only information that arrives after $t _ { d }$ moves the decision itself. □

Decisions with several dependencies. If the chosen option stays best only while each of the facts $f _ { 1 } , \ldots , f _ { n }$ holds, and the facts change independently, the same argument gives

$$
\mathrm { P r } [ \mathrm { a l l ~ } f _ { k } \mathrm { ~ h o l d ~ a t ~ } t _ { a } ] \leq \exp \Big ( - \Delta _ { \mathrm { a c t } } \sum _ { k = 1 } ^ { n } \lambda _ { f _ { k } } \Big ) .
$$

The ceiling tightens with every dependency, including dependencies the decision text leaves implicit, such as the substation behind a district. This is why the ledger records dependencies with the choice.

Derivation of Eq. 3. A decision is correct when the model keeps it if it should stand and switches it if it should change. Conditioning on the label and, for decisions that should change, on whether a decisive item was delivered, the law of total probability gives

$$
\begin{array} { r l } & { A = \left( 1 - \pi \right) \mathrm { P r } [ \mathrm { k e e p ~ | ~ s t a n d } ] } \\ & { ~ + \pi \rho \mathrm { P r } [ \mathrm { s w i t c h ~ | ~ d e l i v e r e d } ] } \\ & { ~ + \pi ( 1 - \rho ) \mathrm { P r } [ \mathrm { s w i t c h ~ | ~ m i s s e d } ] , } \end{array}
$$

which is Eq. 3. The identity holds for every router and model; the theory’s content lies in the predictions about its terms.

Comparing routers. Consider two routers used with the same model, with recalls $\rho$ and $\rho ^ { \prime }$ and keep rates K and $K ^ { \prime }$ . If S and L are the same under both, Eq. 3 gives

$$
A ^ { \prime } - A = \pi ( \rho ^ { \prime } - \rho ) ( S - L ) + ( 1 - \pi ) ( K ^ { \prime } - K ) .
$$

A router that reaches more decisive news gains accuracy in proportion to $S - L ,$ , the margin by which the model acts on delivered news over undelivered news, and loses it through the misleading items that lower $K .$ . Strong judgment makes $S - L$ large and keeps $K$ stable, so recall dominates; weak judgment shrinks $S - L$ and lowers K with every delivery, so exposure dominates. This is prediction P3 in closed form.

## B Decision Streams

The generators, the three streams and their SHA-256 hashes are released with the paper; regenerating each stream from its seed reproduces the hash recorded with the model answers. Every label is computed from timestamps and stated facts, and every stream is checked by assertions before use.

DIRECT and STRESS. Decision times fall within the first three days of a six-day stream. Lags are drawn uniformly from 3–12 or 24–72 hours with equal probability, and action times fall in domain windows (dinner 19:00–22:00, trains 06:00–20:00, orders 08:00–20:00, permits on weekdays 09:00– 15:30, hikes 07:00–13:00). Decisive items are published inside the revision window and block the option from before $t _ { a }$ until at least an hour after the activity; late items start at least one hour after the activity ends. Two DIRECT notices: “Notice from Juniper Barrel: closed for a private event from Tue 13 Oct 2026, 14:40 until Wed 14 Oct 2026, 01:10.” (names the option) and “City notice: planned power cut in the Riverside district from Wed 14 Oct 2026, 14:40 to Thu 15 Oct 2026, 05:10; restaurants and shops there will close.” (names the stated dependency). The generator asserts that $t _ { d } < t _ { r } < t _ { a } .$ that every decisive item lies in the revision window and overlaps the activity, that no item targets option 2, and that late items never overlap the activity. For STRESS, it recomputes all labels after placing decoys and paraphrasing notices, and asserts that no gold answer or exclusion changed and that no decoy relates to any decision.

INDIRECT. Each domain (walk-in dinners, rail journeys, day hikes, in-store pickups) has 16 bundles of a dependency, its other name, its upstream entity and four facilities. Each bundle serves one decision acting on days 2–3 and one on days 5–6, so the two are never open together; lags are 5–12 or 18–28 hours. A blocking notice for the activity starts between six hours and 15 minutes before it and ends at least 30 minutes after it; a late one starts at least an hour after the activity ends; a past one ends at least 45 minutes before it starts. Late, past and harmless notices use the four ways of reference equally, and blocking, late and past notices share templates, so only their times differ. After placing all notices, the generator recomputes every label and asserts the intended label, exactly two related notices per decision, the balance of the design, that decoys and unrelated notices concern no decision, and that no notice affecting a decision was public before it was made.

## B.1 A Decision from INDIRECT

Decision D0011 is the running example of Figure 1, reproduced verbatim from the stream. The dossier holds the other name of the district (3), the substation upstream of it (6), four facilities a walking diner does not need (2, 5, 10, 11), and six traps that mention other decisions’ entities (1, 4, 7, 8, 9, 12).

Times. Decided Tue 13 Oct 2026, 13:05; revisable until Tue 13 Oct 2026, 19:00; action Tue 13 Oct 2026, 20:30, lasting 2 hours. 30 notices are published in the revision window.

Request. The user wants to walk in for dinner at 20:30 on Tuesday 13 October 2026 (about 2 hours). They want the best-rated restaurant within a 20-minute walk and will walk there.

Decision-timefacts. “Lovage Corner: rated 4.6/5; 9-minute walk from the user; in the Bellhaven district; open daily 12:00- 02:00; walk-ins welcome.” “Red Bistro: rated 4.2/5; 8-minute walk from the user; in the Moorside district; open daily 12:00- 02:00; walk-ins welcome.”

Choice. A. Lovage Corner (alternative: B. Red Bistro).

Dossier. (1) Tourist boats run from the Pigeon Lofts to the Bellhaven waterfront. (2) The Ironworks museum is in the Bellhaven district. (3) The Bellhaven district is known locally as the Chime Green. (4) The floodlights of the Bellhaven football ground are fed from the Dunsby substation. (5) Tram line T12 stops in the Bellhaven district. (6) The Bellhaven district is supplied with electricity by the Gorran substation. (7) The Bellhaven football club plays its derby against the Southbank district team. (8) Tram line T12 draws its power from the Hensall traction substation. (9) Tram line T12 continues from the Bellhaven district to the Millbrook district. (10) The Bellhaven district hosts the Bread market on weekday mornings. (11) The Bellhaven multi-storey car park serves visitors who drive. (12) Street musicians from the Leafy Triangle often play in the Bellhaven district.

Notices generated with this decision.

• Tue 13 Oct 2026, 14:50, related, harmless: “City notice: new street lighting installed in the Bellhaven district; no disruption expected.”

• Tue 13 Oct 2026, 15:05, same-name decoy: “Road closure: Bellhaven Road in the Ormsby district is shut from Tue 13 Oct 2026, 16:40 to Wed 14 Oct 2026, 01:15.”

• Tue 13 Oct 2026, 16:40, unrelated: “Notice from Tidewater Kitchen in the Ormsby district: closed from Tue 13 Oct 2026, 17:10 to Wed 14 Oct 2026, 01:40.”

• Tue 13 Oct 2026, 18:40, decisive (upstream entity): “Grid operator: emergency works at the Gorran substation from Tue 13 Oct 2026, 19:50 to Wed 14 Oct 2026, 00:15; every area it supplies will be without power and businesses there will close.”

Table 5 follows the decision through every router. The decision text and both dense routers miss the decisive notice, which names only the substation; BM25 over the whole dossier or over the watch conditions reaches it, and every model then switches. Under the decision text, Ministral 8B switches without the decisive notice, prompted by other news: the L term of Eq. 3.

<table><tr><td></td><td colspan="3">Delivered</td><td colspan="3">Answer (gold: B)</td></tr><tr><td>Router</td><td>Items</td><td>Decisive</td><td>Trap</td><td>Claude</td><td>Qwen</td><td>Ministral</td></tr><tr><td>None</td><td>0</td><td></td><td></td><td>A</td><td>A</td><td>A</td></tr><tr><td>Name match</td><td>0</td><td></td><td></td><td>A</td><td>A</td><td>A</td></tr><tr><td>BM25: decision</td><td>6</td><td></td><td></td><td>A</td><td>A</td><td>B</td></tr><tr><td>BM25: + dossier</td><td>5</td><td>√</td><td>√</td><td>B</td><td>B</td><td>B</td></tr><tr><td>Dense: + dossier</td><td>5</td><td></td><td></td><td>A</td><td>A</td><td>A</td></tr><tr><td>BM25: + watch cond.</td><td>8</td><td>√</td><td>√</td><td>B</td><td>B</td><td>B</td></tr><tr><td>Dense: watch cond.</td><td>11</td><td></td><td></td><td>A</td><td>A</td><td>A</td></tr><tr><td>Oracle: related</td><td>2</td><td>√</td><td></td><td>B</td><td>B</td><td>B</td></tr><tr><td>Oracle: decisive</td><td>1</td><td>√</td><td></td><td>B</td><td>B</td><td>B</td></tr><tr><td>All items</td><td>30</td><td>√</td><td>√</td><td>B</td><td>B</td><td>A</td></tr></table>

Table 5: Decision D0011 under every router: notices delivered, whether they include the decisive notice and a trap, and each model’s answer (bold: correct).

## C Prompts

Each condition pairs a watch-condition prompt with a re-decision prompt. Every router of a condition uses the same re-decision prompt, so routers differ only in the notices they deliver.

DIRECT. Both prompts use the system message “You are an assistant that manages decisions made on behalfofa user. Base your answers only on the information given.” Watch conditions:

A decision has been made for the user and will be acted on   
later. New information will keep arriving until the   
revision deadline. Write the decision's watch   
conditions: the kinds of new information that should   
make us reconsider it.   
Decision time: {t\_d}   
Evidence at decision time:   
[F1] ...   
Request: {request}   
Options:   
A. ...   
B. ...   
Decision made: {letter}. {option}   
The user will act {t\_a}. The decision can be changed until {   
t\_r}.   
Write 3 to 6 watch conditions. Each must describe a specific   
event that would make the chosen option unsuitable at   
the action time. Cover events that affect the chosen   
option directly, and events that affect it indirectly   
through the districts, operators, platforms, regions or   
services it depends on according to the evidence. Name   
those entities explicitly.   
Reply with JSON only, in this form:   
{"watch": ["condition 1", "condition 2", "condition 3"]}

## Re-decision:

Current time: {t\_r}   
Evidence at decision time ({t\_d}):   
[F1] ...   
Request: {request}   
Options:   
A. ...   
B. ...   
Current decision: {letter}. {option}   
New information published since the decision:   
[N1] (published {t\_w}) ...   
The user will act {t\_a}. This is the last chance to change   
the decision. Taking the new information into account,   
which option should the user go with? Ignore   
information that is unrelated to these options or that   
only takes effect after the user's activity is over.   
Reply with JSON only: {"answer": "<letter>"}  
STRESS. The system message is “You revise previously made choices using ONLY the facts and

notices provided. Respect the action time and notice effective time.” Claude writes watch conditions with:

A decision was made for a later action. Propose 3-6 specific   
watch conditions that might force revisiting the chosen   
option.   
Use ONLY the decision-time evidence; name the chosen option   
and its exact dependency entities if stated.   
Prioritize notices affecting availability and execution,   
including indirect disruptions. Avoid generic user  
payment, ratings or competing alternatives.   
Decision made at: {t\_d}   
Last revision deadline: {t\_r}   
Action starts at: {t\_a}   
Request: {request}   
Facts:   
[F1] ...   
Options:   
A. ...   
B. ...   
Chosen: {letter}. {option}   
Reply with JSON only: {"watch":["...","...","..."]}

## Re-decision:

Current decision revision time: {t\_r}   
Decision-time facts ({t\_d}):   
[F1] ...   
Request: {request}   
Options:   
A. ...   
B. ...   
Original choice: {letter}. {option}   
Newly published notices (shown in time order; may be   
unrelated or not yet effective):   
[N1] (published {t\_w}) ...   
Action starts: {t\_a}. Activity lasts {h} hours.   
You must pick the option that will best satisfy the request   
DURING the action interval, not when the notice was   
published.   
Ignore unrelated notices and changes whose effects start   
after the activity ends. If the original choice remains   
best, keep it.   
Record only JSON {"answer":"A"} or {"answer":"B"} using the   
required structured output protocol.

INDIRECT. The system message is “You revise previously made choices using ONLY thefacts and notices provided. Respect the action time and when each notice takes effect.” Claude writes watch conditions with:

A decision was made for a user and will be acted on later.   
New notices will keep arriving until the revision   
deadline. Write the watch conditions for this decision:   
the kinds of notice that should make us revisit it.   
Decision made at: {t\_d}   
Revision deadline: {t\_r}   
Action starts: {t\_a}; it lasts {h} hours.   
Request: {request}   
Decision-time facts:   
[F1] ...   
Options:   
A. ...   
B. ...   
Chosen: {letter}. {option}   
Background facts about the chosen option (from the ledger):   
[B1] ...   
Write 3 to 8 watch conditions. Each must describe a specific   
event that would stop the user's activity with the   
chosen option during the action. Name every entity the   
activity depends on, including any other name it is   
known by and anything upstream it relies on. Leave out   
facilities and entities the user does not need for this   
activity.   
Reply with JSON only: {"watch": ["...", "...", "..."]}

Re-decision:

Current time (last chance to revise): {t\_r}   
Decision-time facts ({t\_d}):   
[F1] ...   
Request: {request}   
Options:   
A. ...   
B. ...   
Original choice: {letter}. {option}   
Background facts about the original choice (from the decision   
ledger):   
[B1] ...   
Newly published notices (in time order; they may be unrelated   
or not in effect during the activity):   
[N1] (published {t\_w}) ...   
Action starts: {t\_a}. The activity lasts {h} hours.   
Pick the option that will best satisfy the request DURING the   
activity, not when the notice was published.   
Ignore notices about other things and changes that are not in   
effect during the activity. If the original choice   
remains best, keep it.   
Reply with JSON only: {"answer":"A"} or {"answer":"B"}.

## D Implementation

Models and decoding. Qwen3-8B runs with thinking disabled through Hugging Face Transformers (Wolf et al., 2020) with greedy decoding, at most 400 output tokens for watch conditions and 128 for re-decisions. Claude Sonnet 4.6, GPT-6.1 Sol, Qwen3-235B-A22B (2507), Ministral 8B (Ministral 3 8B Instruct) and Mistral 7B (Instruct v0.2) run at temperature 0 with structured output and a plain-JSON fallback, with at most 768 output tokens per answer. Failed or invalid calls are retried and otherwise recorded as missing, never as an answer, and every answer is cached under a hash of the stream, model, router and prompt.

Routing. BM25 uses k =1.5 and b=0.75 over lowercase alphanumeric tokens; dense routing uses BGE-base v1.5 (Xiao et al., 2024). Dependency queries use “{option} unavailable cancelled blocked closed or delayed”, “{dependency} outage closure disruption maintenance restriction strike suspended” and “{dependency} operating hours availability travel safety affects {option}”. The time filter parses a start time introduced by from, starting, beginning, effective or start and drops the decision when that time is not before the end of its activity.

Analysis. Every number in the paper is computed by scripts from the per-decision logs, which record for each decision, router and model the notices delivered, whether the decisive notice and traps were among them, and the answer. The scripts and logs are released with the paper.

## E Additional Results

Where the errors are. Table 6 breaks down IN-DIRECT accuracy by how the decisive notice refers to the option and by the related news a decision received. Routing sets which switches are possible: the decision text reaches the option and its stated dependency, and the dossier and watch conditions extend the reach to other names and upstream entities. Judgment sets the rest: decisions with a change that is already over or starts after the activity are the ones most often abandoned.

<table><tr><td></td><td>Name</td><td>BM25</td><td></td><td>+Doss. +Watch</td><td>Oracle</td><td>All</td></tr><tr><td>Needed switches made, by how the notice refers to the option</td><td colspan="6"></td></tr><tr><td>option name</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>dependency</td><td>8.3</td><td>100.0</td><td>100.0</td><td>97.9</td><td>100.0</td><td>97.9</td></tr><tr><td>other name</td><td>6.2</td><td>54.2</td><td>95.8</td><td>97.9</td><td>97.9</td><td>93.8</td></tr><tr><td>upstream entity</td><td>4.2</td><td>35.4</td><td>93.8</td><td>75.0</td><td>100.0</td><td>89.6</td></tr><tr><td colspan="6">Decisions kept, by the related news they received</td><td></td></tr><tr><td>starts after</td><td>81.2</td><td>52.1</td><td>37.5</td><td>43.8</td><td>43.8</td><td>39.6</td></tr><tr><td>already over</td><td>77.1</td><td>37.5</td><td>27.1</td><td>33.3</td><td>33.3</td><td>27.1</td></tr><tr><td>harmless</td><td>91.7</td><td>68.8</td><td>52.1</td><td>70.8</td><td>72.9</td><td>52.1</td></tr><tr><td>not needed</td><td>89.6</td><td>70.8</td><td>52.1</td><td>60.4</td><td>62.5</td><td>43.8</td></tr></table>

Table 6: Needed switches made and decisions kept on INDIRECT, by how the decisive notice refers to the option and by the related news a decision received (%, mean over the three models with complete answers; 16 decisions per row and model). Routing sets which switches are possible; judgment sets the rest.

Statistical tests. Table 7 lists the exact McNemar tests behind the INDIRECT comparisons in Section 7.

Every router on DIRECT. Table 8 reports every router with Qwen3-8B on DIRECT, and Table 9 breaks down its needless switches by the related news a decision received. With oracle routing, Qwen3-8B makes every needed switch and keeps 62.0% of the decisions that should stand. BM25 over the decision text, scored on DIRECT for routing only, delivers 97.1% of decisive pairs.

Routing with other K. Table 10 shows INDI-RECT routing as the number K of decisions per notice varies. With K=1, the whole dossier delivers 85.9% of decisive notices and watch conditions 71.9%.

A second STRESS sample. Table 11 shows Claude Sonnet 4.6 on a second sample of 73 decisions, where routing again lifts accuracy from 61.6% without routing to 89.0% with BM25, against 94.5% for the related oracle.

Mistral 7B on INDIRECT. Mistral 7B returned a valid answer for 74–95% of the prompts, depending on the router, and mostly kept its original choice: given only decisive notices, it switched on 48.9% of the decisions it answered. The comparisons in Section 7 use the three models with complete answers.

<table><tr><td></td><td></td><td colspan="2">Claude</td><td colspan="2">Qwen-235B</td><td colspan="2">Ministral 8B</td><td colspan="2">Pooled</td></tr><tr><td>Router A</td><td>Router B</td><td>W-L</td><td>p</td><td>W-L</td><td>p</td><td>W-L</td><td>p</td><td>W-L</td><td>p</td></tr><tr><td>BM25: decision</td><td>Name match</td><td>26-5</td><td>&lt;.001</td><td>30-26</td><td>.689</td><td>30-26</td><td>.689</td><td>86-57</td><td>.019</td></tr><tr><td>BM25: + dossier</td><td>BM25: decision</td><td>22-4</td><td>&lt;.001</td><td>17-15</td><td>.860</td><td>12-13</td><td>1.000</td><td>51-32</td><td>.048</td></tr><tr><td>BM25: + watch cond.</td><td>BM25: decision</td><td>21-5</td><td>.002</td><td>16-11</td><td>.442</td><td>15-7</td><td>.134</td><td>52-23</td><td>.001</td></tr><tr><td>BM25: + watch cond.</td><td>BM25: + dossier</td><td>5-7</td><td>.774</td><td>9-6</td><td>.607</td><td>12-3</td><td>.035</td><td>26-16</td><td>.164</td></tr><tr><td>Dense: watch cond.</td><td>Dense: + dossier</td><td>11-13</td><td>.839</td><td>12-16</td><td>.572</td><td>10-9</td><td>1.000</td><td>33-38</td><td>.635</td></tr><tr><td>All items</td><td>BM25: + dossier</td><td>6-8</td><td>.791</td><td>2-9</td><td>.065</td><td>10-8</td><td>.815</td><td>18-25</td><td>.360</td></tr><tr><td>All items</td><td>BM25: + watch cond.</td><td>6-6</td><td>1.000</td><td>6-16</td><td>.052</td><td>8-15</td><td>.210</td><td>20-37</td><td>.033</td></tr><tr><td>Oracle: related</td><td>BM25: + dossier</td><td>9-1</td><td>.021</td><td>11-0</td><td>&lt;.001</td><td>11-5</td><td>.210</td><td>31-6</td><td>&lt;.001</td></tr><tr><td>Oracle: decisive</td><td>Oracle: related</td><td>1-0</td><td>1.000</td><td>38-0</td><td>&lt;.001</td><td>52-0</td><td>&lt;.001</td><td>91-0</td><td>&lt;.001</td></tr></table>

Table 7: Exact McNemar tests on INDIRECT. W–L: decisions router A gets right and router B gets wrong, and the reverse; 128 decisions per model, pooled over the three models with complete answers. p-values are two-sided and uncorrected.
<table><tr><td></td><td colspan="3">Routing (%)</td><td colspan="5">Final decisions (%)</td><td></td></tr><tr><td>Router</td><td>Recall</td><td>Implicit</td><td>Precision</td><td>Accuracy</td><td>Switch: named</td><td>Switch: dependency</td><td>Keep</td><td>Tokens (k)</td><td>p</td></tr><tr><td>Qwen/Qwen3-8B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>None (freshness at td)</td><td>0.0</td><td>0.0</td><td></td><td>45.5</td><td>0.0</td><td>0.0</td><td>100.0</td><td>0.0</td><td>&lt;0.001</td></tr><tr><td>Name match</td><td>39.7</td><td>0.0</td><td>100.0</td><td>70.0</td><td>100.0</td><td>18.2</td><td>88.0</td><td>23.9</td><td>0.851</td></tr><tr><td>Dense similarity</td><td>83.8</td><td>75.6</td><td>14.3</td><td>71.8</td><td>92.0</td><td>63.6</td><td>66.0</td><td>102.4</td><td></td></tr><tr><td>Watch conditions</td><td>77.9</td><td>65.9</td><td>14.2</td><td>62.7</td><td>80.0</td><td>57.6</td><td>56.0</td><td>146.9</td><td>0.076</td></tr><tr><td>Oracle: related</td><td>100.0</td><td>100.0</td><td>100.0</td><td>82.7</td><td>100.0</td><td>100.0</td><td>62.0</td><td>44.2</td><td>0.050</td></tr><tr><td>All items</td><td>100.0</td><td>100.0</td><td>1.6</td><td>55.5</td><td>60.0</td><td>33.3</td><td>68.0</td><td>803.7</td><td>0.003</td></tr></table>

Table 8: Qwen3-8B on DIRECT. Recall: share of decision-changing items delivered to the decision they change; Implicit: the same for items that reach the option only through a dependency; Precision: share of delivered items that concern the chosen option. Switch: named / dependency: share of decisions that should switch and did, when the evidence names the option or reaches it only through a dependency; Keep: share of decisions that should stand and did. Tokens: prompt tokens spent on routing and re-deciding, including writing watch conditions. p: exact McNemar test on per-decision correctness against the most accurate practical router (bold; oracle excluded). Routing recall and precision for None, Name match, Oracle and All follow from the stream and are identical across models.

<table><tr><td>Related news</td><td>n</td><td>Oracle</td><td>Name</td><td>Sim.</td><td>Watch</td><td>All</td></tr><tr><td>Late change</td><td>18</td><td>94</td><td>28</td><td>39</td><td>67</td><td>33</td></tr><tr><td>Harmless only</td><td>19</td><td>11</td><td>5</td><td>37</td><td>37</td><td>32</td></tr><tr><td>None</td><td>13</td><td>0</td><td>0</td><td>23</td><td>23</td><td>31</td></tr></table>

Table 9: Needless switches (%) on decisions that should stand, by the related news published in their revision window. Late change: a change to the option or its dependency that starts only after the activity ends; Harmless only: news about the option or its dependency that does not affect it. Name: name match; Sim.: similarity to the decision; Watch: watch conditions.

<table><tr><td></td><td colspan="2">K=1</td><td colspan="2">K=3</td><td colspan="2">K=5</td></tr><tr><td>Router</td><td>Recall</td><td>Traps</td><td>Recall</td><td>Traps</td><td>Recall</td><td>Traps</td></tr><tr><td>BM25: decision</td><td>51.6</td><td>4</td><td>62.5</td><td>17</td><td>68.8</td><td>35</td></tr><tr><td>BM25: + dossier</td><td>85.9</td><td>21</td><td>100.0</td><td>89</td><td>100.0</td><td>96</td></tr><tr><td>Dense: + dossier</td><td>70.3</td><td>11</td><td>85.9</td><td>59</td><td>90.6</td><td>74</td></tr><tr><td>BM25: + watch cond.</td><td>71.9</td><td>5</td><td>92.2</td><td>20</td><td>96.9</td><td>38</td></tr><tr><td>Dense: watch cond.</td><td>75.0</td><td>4</td><td>84.4</td><td>27</td><td>90.6</td><td>53</td></tr></table>

Table 10: INDIRECT routing as the number K of decisions each notice is sent to varies: decisive pairs delivered (%) and lexical traps delivered.

<table><tr><td>Router</td><td>Accuracy</td><td>Keep</td><td>Needless</td></tr><tr><td>None</td><td>61.6</td><td>100.0</td><td>0</td></tr><tr><td>Name match</td><td>75.3</td><td>91.1</td><td>4</td></tr><tr><td>BM25: decision</td><td>89.0</td><td>91.1</td><td>4</td></tr><tr><td>BM25: + time filter</td><td>91.8</td><td>93.3</td><td>3</td></tr><tr><td>Dense: decision</td><td>91.8</td><td>93.3</td><td>3</td></tr><tr><td>Dense: watch cond.</td><td>89.0</td><td>86.7</td><td>6</td></tr><tr><td>Dependency queries</td><td>90.4</td><td>88.9</td><td>5</td></tr><tr><td>Oracle: related</td><td>94.5</td><td>93.3</td><td>3</td></tr><tr><td>Oracle: decisive</td><td>100.0</td><td>100.0</td><td>0</td></tr></table>

Table 11: Claude Sonnet 4.6 on a second STRESS sample (73 decisions). Keep: accuracy on decisions that should stand (%); Needless: switches on those decisions. The table lists the routers run on this sample.