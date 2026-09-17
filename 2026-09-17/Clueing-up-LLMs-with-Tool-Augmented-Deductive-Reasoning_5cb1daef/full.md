# Clueing up LLMs with Tool-Augmented Deductive Reasoning

Rebecca Ansell<sup>1</sup> and Autumn Toney-Wails<sup>2</sup>,<sup>3</sup>

<sup>1</sup> Georgetown University

<sup>2</sup> Syntheos, Corp

<sup>3</sup> UNU-MERIT

rja80@georgetown.edu, toney@merit.unu.edu

## Abstract

Despite recent advances in large language models (LLMs), performing logically consistent deductive reasoning over extended interactions remains challenging. Tasks that require integrating evidence across multiple reasoning steps, maintaining consistency with prior inferences, and updating beliefs under new constraints can surface limitations in current models while providing a useful testbed for evaluating reasoning enhancements. In this paper, we implement a text-based, multi-agent version of the classic board game Clue as an environment to evaluate multi-step, agentic deductive reasoning. In this setting, agents must infer hidden information from a sequence of observations, maintain consistency across turns, and reason over an evolving set of logical constraints. We instantiate six LLM-based agents (GPT-4o-mini and Gemini-2.5- Flash) as players that engage in turn-based gameplay; using three agents per model family, we establish baseline performance across repeated games. We then introduce a tool-augmented approach in which a structured possibility matrix converts implicit game state from generated reasoning logs into an explicit representation of remaining possibilities. The possibility matrix encodes extended-turn memory and deductive constraints, offloading these tasks from the agent. We compare this approach against the baseline to evaluate how tool augmentation supports reasoning quality and task success for autonomous agents in a strategic reasoning environment.

## 1 Introduction

Game environments have proven to be dynamic testbeds for researchers to evaluate autonomous agents’ abilities to reason and strategize, as these environments extend beyond static benchmarks and evolve with the actions and behaviors of multiple players [Huang et al., 2025]. Specifically, analyzing generated reasoning outputs of large language model (LLM)- based agents throughout gameplay provides insights into their decision-making processes, strategic planning, and interaction dynamics [Chalamalasetti et al., 2023; Lin et al., 2025;

![](images/3b55e0dd3e609567e843b665974389e49aebae20c88a34ab50c11d33141e086c.jpg)  
Figure 1: Example turn in Clue game environment with the possibility matrix tool highlighted; the matrix records Yes, No, and Maybe values for possible cards in other players’ hands.

Gevers and Daelemans, 2025; Hu et al., 2025]. However, LLM-based agents often underperform in game environments requiring extended memory and multi-step reasoning, motivating the development of prompt engineering, fine-tuning, and tool-augmentation techniques to support agentic reasoning [Trencsenyi et al., 2025].

Recent work has highlighted tool augmentation as a promising approach to improving the reasoning capabilities of LLMs [Chen et al., 2023; Shim et al., 2025]. Toolaugmented language models (TALMs) provide LLMs with access to external tools that can be invoked via APIs and support information retrieval, reasoning, and decision-making [Parisi et al., 2022; Qu et al., 2025]. In this way, tool augmentation (rather than prompt engineering and fine-tuning) offloads complex retrieval and reasoning tasks from the agent’s internal representations to external tool interactions, improving performance on benchmarks [Hao et al., 2023; Das et al., 2024; Ma et al., 2024]. Despite these advantages, effectively incorporating tools into LLM-based agents remains challenging, since tool-use capabilities are not uniformly supported across models and agents often struggle to reliably determine when and how to invoke tools in complex, interactive settings.

In this work, we design and implement a tool augmentation approach tailored to interactive game environments by introducing a structured possibility matrix that represents the current game state (shown in Figure 1). Our external tool programmatically extracts and maintains each player’s belief state throughout gameplay, and is invoked at each turn to dynamically update the agent’s prompt. By externalizing state tracking in this way, we offload the burden of extendedmemory reasoning from the agent and instead provide a consistent and explicit representation of the evolving game state and deductive constraints. Our proposed approach mitigates the degradation of reasoning quality that can arise from long reasoning histories, enabling agents to operate over a cleaner and more reliable contextual grounding when making decisions. Furthermore, our approach is applicable to LLMs that do not natively support tool-use functionalities.

We implement a text-based, multi-agent version of the classic board game Clue following Ansell and Toney [2026], which instantiates six agentic players derived from GPT-4omini and Gemini-2.5-Flash (three players per model family). Clue provides a deductive reasoning game environment, where game success is achieved through maintaining and updating beliefs over partially observed information to correctly infer the hidden combination of suspect, weapon, and location. To evaluate our tool augmentation approach, we run six games with the baseline models, six games with the players all using the possibility matrix tool, and six games where there are two baseline players and one tool user.

Across these 18 game instances, we investigate three research questions:

• (RQ1) To what extent does externalized belief state tracking via tool augmentation improve agent reasoning and gameplay performance?

• (RQ2) How does the presence of tool-augmented agents influence the behavior and performance of non-tool agents in mixed-agent environments?

• (RQ3) How does structured reasoning support affect agents’ decision-making strategies, particularly with respect to risk-taking and the timing of final decisions?

We find that tool augmentation leads to near-perfect gameplay for tool-augmented agents, improves the performance of non-tool agents in mixed settings, and results in less delay for accusations without a reduction in accusation accuracy.

## 2 Related Work

## 2.1 Deductive Reasoning in LLMs

Chain-of-thought prompting has achieved measurable improvements on arithmetic, commonsense QA, and symbolic reasoning tasks [Chu et al., 2024], however these gains degrade as inference chains grow longer, with the generated reasoning steps themselves often containing logical inconsistencies [Patel et al., 2024]. Recent analyses further show that models can arrive at the correct answers through unfaithful intermediate steps, so the surface reasoning does not reliably reflect the process that produced the final answer [Xu et al., 2026; Zheng et al., 2025]. Widely used reasoning benchmarks, covering mathematical problem solving [Zeng et al., 2024; Hendrycks et al., 2021], commonsense inference [Talmor et al., 2019], and formal logic [Liu et al., 2023; Yu et al., 2020], share a structural limitation relevant to our setting: events are evaluated in isolation, so they cannot measure whether a model sustains a coherent belief state across multiple interdependent inferences.

## 2.2 Games as Deductive Reasoning Testbeds

Interactive games have become a common way to probe agentic reasoning, but existing environments rarely isolate deduction as the target capability. Social deduction games like Werewolf, Avalon, and Among Us [Xu et al., 2025; Chi et al., 2024] test persuasion and intent modeling, while Diplomacy [FAIR et al., 2022], poker [Zhuang et al., 2025], and broader game suites [Lin et al., 2025] test reasoning under uncertainty. This research gap leaves extended-memory deductive reasoning as an open challenge in game environments. Ansell and Toney [2026] take a step toward addressing this gap with a text-based implementation of Clue, but finds that models continue to struggle in this setting, even with text-based fine-tuning on related logic puzzle tasks.

## 2.3 Tool-Augmented Reasoning

A growing research area is in Tool-Augmented Language Models (TALMs), which provides LLMs with external tools via API calls [Parisi et al., 2022]. This tool-augmented capability shifts components of the information retrieval and reasoning workload from the model’s internal representations to structured external retrieval and computation [Qu et al., 2025]. Tools are traditionally invoked through LLM API functionality, but can also be encoded in natural language as shown by [Hao et al., 2023]. TALMs have been shown to improve performance across a range of complex reasoning tasks [Chen et al., 2023; Das et al., 2024; Ma et al., 2024] as well as in multi-turn, interactive dialogue settings [Arcadinho et al., 2024; Shim et al., 2025; Jung et al., 2025]. However, effective tool use remains an open challenge, as models do not always invoke tools appropriately or reliably integrate tool outputs into downstream reasoning [Patil et al., 2024; Chen et al., 2024; Kwak et al., 2025].

Our work addresses these open challenges by implementing a tool-augmentation approach for agentic gameplay that maintains a structured belief state and injects a corresponding possibility matrix into agent prompts. To this end, we evaluate whether offloading extended-memory belief tracking can improve the deductive reasoning capabilities of autonomous agents, without requiring them to learn when or how to call tools effectively.

## 3 Experimental Design

## 3.1 Clue Game Environment

The murder mystery, board game Clue was adapted into a text-based environment by [Ansell and Toney, 2026], in which the rules, turn-taking, and game state transitions are conducted through the inputs (prompts) and outputs (generated reasoning logs) of six autonomous agent players. Following the original board game setup, 21 cards (six suspects, six weapons, and nine rooms) are shuffled, and one card from each category is randomly selected and placed in an envelope, forming the hidden solution unknown to all players. The remaining cards are dealt evenly, with each player receiving a private hand of three cards.

The objective of the game is to correctly deduce the hidden combination of suspect, weapon, and room by making a final accusation. Players take turns making suggestions (i.e., strategic, publicly stated hypotheses) about the solution. If another player holds one or more of the suggested cards, they must privately reveal one of those cards to the suggesting player, with the option to choose which card to show when multiple apply. While other players observe that a card was revealed, they do not see its content. A player will only see at most one card during their turn.

This game interaction structure encourages strategic reasoning and actions, as each turn reveals only partial information; suggestions are publicly expressed through dialogue, while the resulting card reveals remain private to the suggesting player (who may also hold cards included in their own suggestion to potentially mislead other players). In the textbased setting, agents are prompted to verbalize their reasoning at each turn, and their generated reasoning is included in each turn prompt so they can process their previous turns.

In the board game, play ends when the first player makes a correct accusation; however, in our implementation, after a player has won the remaining agentic players are all given a chance to arrive at the final solution and the cards remain hidden until the end. Each game is limited to a maximum of 20 rounds, but ends sooner if all players have made final accusations in less rounds.

## 3.2 LLM Agents

We instantiate agentic players using two language models: OpenAI’s GPT-4o-mini [OpenAI, 2024] and Google’s Gemini-2.5-Flash [Comanici et al., 2025]. Ansell and Toney [2026] selected these models to balance efficiency and performance, as they are lightweight and cost-effective while still achieving strong reasoning performance. Each agent is initialized with default parameters and receives prompts with identical structure and content across all players. We do not assign distinct personas or role-specific behaviors; all agents operate under the same configuration to ensure consistency and comparability of behavior across model families.

## 3.3 Possibility Matrix Tool

To support belief state tracking beyond verbose agentic reasoning logs, we design a tool that maintains a structured possibility matrix for each agentic player. The tool has three components it maintains for robust player support: envelope candidates, known cards by player, and game matrix. The tool output represents each agent’s current belief state over all 21 game cards across every possible holder (i.e., all six players and the hidden solution envelope). Each cell in the game matrix stores one of three values: YES, NO, or MAYBE, indicating whether a given card is known to be held by a particular player (or the envelope), known to not be held, or remains uncertain.

A player’s possibility matrix is instantiated before its first turn action, recording the corresponding hand cards in the known cards by player component and as “YES” in the game matrix; the remaining cards are populated with “MAYBE”, for example the GPT4o MINI 3 player’s known cards by player:

"GPT4o\_MINI\_3": [   
"Miss Scarlet",   
"Mrs. Peacock",   
"Rope"   
]   
and a component of game matrix:

```json
"Rope": {
"ENVELOPE": "NO",
"GEMINI_FLASH_1": "NO",
"GEMINI_FLASH_2": "NO",
"GEMINI_FLASH_3": "NO",
"GPT4o_MINI_1": "NO",
"GPT4o_MINI_2": "NO",
"GPT4o_MINI_3": "YES"
}
```

The matrix is then updated incrementally over the course of play based on both private observations and public game events (generated reasoning output and card reveals). Private observations include cards directly revealed to the agent, while public updates are derived from suggestions and disproof behavior. For example, if a player passes on disproving a suggestion, the matrix records that the player does not hold any of the suggested cards (i.e., the matrix records “NO” for that player on all three cards). If a player disproves a suggestion but the exact card is not observed, the matrix stores a constraint indicating that the player must hold at least one of the suggested cards. If no player disproves a suggestion, the matrix rules out all non-suggesting players as holders of the suggested cards.

The final reasoning support mechanism of the possibility matrix tool is its evaluation of the remaining envelope candidates (possible solution candidates). If there is exactly one candidate left for each card category (suspect, weapon, and room) in the envelope candidate component, the tool includes an accusation ready cell that records the solution. The cell remains null until the candidate requirement is satisfied. Thus, agentic players receive explicit information about whether their accumulated knowledge supports making a final accusation.

At each turn, the tool is called and its output is injected into the current agentic player’s turn prompt with the prefix “Use this matrix state as the authoritative belief state for this turn”. Then the tool is called after the turn actions have ended to update with any cards revealed; this process (and comparison to the baseline implementation) is shown in Figure 2. The tool output summary includes current envelope candidates, known card assignments for other players, unresolved disproof constraints, and an indication of whether the agent has sufficient information to make an accusation. In this way, the possibility matrix offloads extended-memory deductive reasoning into a structured intermediate representation. Rather than requiring agents to reconstruct and maintain belief states solely from natural language reasoning history, the tool provides a clean and updated summary of the evolving game state. As a result, agents are able to reason over explicit deductive constraints at each turn.

![](images/6c01369657ab7cec6f826f9f2883eee576e36f65925f322b06c827add5a0c52c.jpg)  
Figure 2: Comparison between the baseline agentic player’s turn and the tool-augmented agentic player’s turn.

## 3.4 Game Implementation

We run three versions of the Clue game: BASELINE (all baseline players), MATRIX (all tool-augmented players), and MIXED (both baseline and tool-augmented players). Each game has six agentic players (3 per model family), with the MIXED game having one tool-augmented player and two baseline players per model family. We run each game version six times, shuffling the player order. Across the 18 games, we record all elements of gameplay (game log, game state, player’s hands, player’s cards seen, player’s prompts, and player’s reasoning) for analysis.

To support answering our three research questions, we evaluate gameplay performance using four metrics: (1) accusation accuracy, (2) deduction quality, (3) knowledge accumulation, and (4) final accusation delay. Accusation accuracy represents the fraction of cards the agentic player correctly used in their accusation (3/3 being perfect). Deduction quality tracks the ratio of correct and incorrect deductions made by players. We label deductions programmatically by evaluating each inferred card assignment against the ground-truth game state, including all player hands and the envelope contents. A deduction is assigned the correct label if it matches the true card assignment and incorrect otherwise. For example, if a player deduces that player x holds the Rope card when player x does not, that deduction is classified as incorrect. Knowledge accumulation represents the number of cards learned through play per round. Final accusation delay counts how many individual player turns took place from the first suggestion made that had no disproofs (and was the game solution) to the first accusation.

## 4 Results

We summarize the results of the 18 game runs in Table 1, providing details on agentic player performance across game versions and model families. We report outcome metrics (wins, mean finishing rank, and accusation accuracy) alongside reasoning metrics (mean correct and incorrect deductions per

player-game).

We find that GPT-4o-mini players win the majority of games (13/18) over Gemini-2.5-Flash players. However, all tool-augmented Gemini-2.5-Flash players achieve perfect accusation accuracy, outperforming GPT-4o-mini players.

## 4.1 Accusation Accuracy

We display player-level game accuracy findings in Figure 3. In the BASELINE game, accuracy was consistently low and no player was able to reach the final solution. Less than half of players (47%) identified only one card correctly, while 36% identified none. In contrast, in the MATRIX games 94.5% of all players identified the correct solution, with only two instances of 2/3 accuracy outcomes across the six game observations (from GPT-4o-mini). Gemini-2.5-Flash achieved perfect accuracy once the matrix tool was introduced.

In the MIXED run, both player types substantially outperformed the BASELINE players. Agentic players without tool-augmentation still achieved a mean accuracy of 2.71/3, compared to 0.81/3 in the BASELINE runs. Tool-augmented agentic players in MIXED games achieved a mean accusation value of 2.67/3, comparable to 2.94 from the MATRIX game version.

## 4.2 Deduction Quality

Parsing the player logs, we programmatically evaluate if deductions made are viable (i.e., did a player make a deduction that is contradictory to the actions and observations in the game) and denote viable deductions as correct. Figure 4 shows the mean correct and incorrect deductions per player broken down by game version and model family. In the BASELINE version (A), both models produced the highest raw counts of correct deductions (GPT-4o-mini: 9.3, Gemini-2.5-Flash: 8.0), but also the highest incorrect deduction counts (2.4 each).

The MATRIX game reduced the number of incorrect deductions for both models, most noticeably for Gemini-2.5-Flash, which dropped to 0.1 incorrect deductions per game while making 4.7 correct deductions. GPT-4o-mini also improved, reaching 6.8 correct and 1.7 incorrect deductions. The overall reduction in both correct and incorrect counts relative to Baseline is consistent with games resolving faster when a tool-augmented agent is playing, leaving fewer rounds for inference.

Table 1: Performance summary across 6 games per condition (36 player-game observations per condition). Outcome: Games won (out of 6), mean finishing position (Rank; 1 = best, 6 = worst), and normalised accusation accuracy (Acc.; 0–1). Reasoning: Average correct and incorrect deductions per game.
<table><tr><td rowspan="2">Version</td><td rowspan="2">Model</td><td colspan="3">Outcome</td><td colspan="2">Reasoning</td></tr><tr><td>Wins</td><td>Rank</td><td>Acc.</td><td>Ded. Correct</td><td>Ded. Incorrect</td></tr><tr><td rowspan="2">BASELINE</td><td>GPT-4o-mini</td><td>4/6</td><td>3.33</td><td>0.24</td><td>9.33</td><td>2.39</td></tr><tr><td>Gemini-2.5-Flash</td><td>2/6</td><td>3.67</td><td>0.30</td><td>8.00</td><td>2.39</td></tr><tr><td rowspan="2">MATRIX</td><td>GPT-4o-mini</td><td>3/6</td><td>3.39</td><td>0.96</td><td>6.83</td><td>1.67</td></tr><tr><td>Gemini-2.5-Flash</td><td>3/6</td><td>3.61</td><td>1.00</td><td>4.67</td><td>0.11</td></tr><tr><td rowspan="4">MIXED</td><td>GPT-4o-mini (baseline)</td><td>4/6</td><td>3.25</td><td>0.89</td><td>7.42</td><td>1.92</td></tr><tr><td>GPT-4o-mini (matrix)</td><td>2/6</td><td>3.17</td><td>0.78</td><td>6.00</td><td>2.17</td></tr><tr><td>Gemini-2.5-Flash (baseline)</td><td>0/6</td><td>4.08</td><td>0.92</td><td>7.00</td><td>1.83</td></tr><tr><td>Gemini-2.5-Flash (matrix)</td><td>0/6</td><td>3.17</td><td>1.00</td><td>6.67</td><td>0.17</td></tr></table>

![](images/771e6f5d311c0010ca3ff6d4546a48d32fb73bff9050fb53cb553095bd9ddffb.jpg)

![](images/5c7dc19509a4d05cd35006f46510769de8bd1a3fea8e66030fafa164cc59d814.jpg)

![](images/36ca349c9dfa4c63683527ccd82867a58a2841ac1e199344a4fa6c163a1d5612.jpg)  
Figure 3: Per-game accusation accuracy for each player, where each cell represents how many solution cards were correctly identified.

In the MIXED condition, the deduction quality pattern differs by player type. Baseline players (C) produced correct deduction rates slightly lower than the BASELINE game. Matrix players (D) showed similar profiles to the (B) MATRIX game, with GPT-4o-mini performing slightly worse, and Gemini Flash performing slightly better.

## 4.3 Knowledge Accumulation

We compute the number of new cards discovered per game round by player to evaluate knowledge accumulation. Figure 5 shows total cards known per round (player’s hand plus cards seen through play) averaged across games, with a maximum of 18 known cards. Across the three game versions, we find that tool-augmented players accumulate knowledge more slowly per round, but translate what they learn into correct accusations more efficiently, as seen in Section 4.1, while Baseline players rely on longer games with extended play to build a near complete picture before solving.

Baseline players (A) accumulate knowledge rapidly in the first five rounds and approach the knowledge ceiling by round

17–18, utilizing the full 20-round game length to observe suggestions and build up a complete picture. In contrast, toolaugmented players (B) end their games much earlier (maximum of 11 rounds observed), truncating the knowledge curve around 14 cards. These players never reach the same total knowledge as the Baseline players, but they solve the game far more accurately. The MIXED game (C) shows the contrast between the two methods side by side. The baseline players gather information slightly quicker, but plateau around 14–15 cards, while the Matrix players diverge sharply after round 10, reaching the ceiling of 18 cards by round 15.

## 4.4 Solution Suggested to Final Accusation

By measuring the number of turns between a correct suggestion (i.e., the suggestion is the game solution) and the subsequent correct accusation, we estimate how effectively players leverage accumulated information to form strong deductions and reach a final decision. Additionally, this analysis provides insights into how “risky” players are based on their game state beliefs.

Figure 6 plots the turn where the correct solution was first publicly confirmed (all three solution cards suggested, no player able to disprove) and the turn where the final correct accusation ended the game.

In the MATRIX game, the first confirmed suggestion occurred between turns 18 and 23, and the accusation followed

(A) Baseline Only  
![](images/5041a01d3a8e1d1ee20cbcd4ebaf317fa01b3d8ebcc35072ae7fd954614a1692.jpg)

(B) Matrix Only  
![](images/c67369d8fa31811fdbfeec3a6079656cf9f30cf1d4018a05f3dc875c91e676c5.jpg)

(C) Mixed (Baseline)  
(D) Mixed (Matrix)  
![](images/416e2b7f022c739ab4a2533cd1a3b11109b9e2056c24541d6a00d6adeecbb493.jpg)  
Figure 4: Mean correct (green) and incorrect (red) deductions by player, model family and game version. The top row shows the two homogeneous conditions, Baseline Only (A) and Matrix Only (B), while the bottom row separates Mixed-condition agents by their prompt type: Baseline-prompted (C) and Matrix-prompted (D).

7–19 turns later. Only 2 of the 6 games were won by the player who first suggested the solution. In the MIXED game, the first confirmed suggestion occurred between turns 13 and 35, and the gap to accusation was 7–31. The first suggester won 3 out of 6 games.

To understand the delay at the player level from knowing the solution to final accusation, we examined when each tool-augmented player first received accusation ready from the possibility matrix, indicating that there was only one candidate per card category remaining. In the MA-TRIX game, 33 out of 36 players had accusation ready set before the game ended. The mean gap from this point to the game-ending accusation was 11.1 turns for GPT-4omini and 8.2 turns for Gemini-2.5-Flash. In the MIXED game version (only tool-augmented players received the signal) their mean gaps were 18.0 turns (GPT-4o-mini) and 15.3 (Gemini-2.5-Flash). Inspection of the player-turns where the accusation ready was set but no accusation was made shows a recurring pattern: agents do not exhibit risky play. All agentic players gather more information despite knowing with certainty what the final solution was.

## 5 Discussion

Here, we address the answers to our research questions drawing evidence from our presented results.

(RQ1) To what extent does externalized belief state tracking via tool augmentation improve agent reasoning and gameplay performance? Comparing the accusation accuracies across game versions, we find that tool augmentation has a clear effect: games with tool-augmented players consistently achieve near-perfect accuracy. This observed game play improvement cannot be attributed to increased information exposure (highlighted in Figure 5). Baseline agents approached the knowledge ceiling of cards by the final rounds and produced the highest raw deduction counts of any game version; however, they failed to convert this evidence into correct accusations. Tool-augmented agents accumulated substantially less total knowledge, but they made accusations with near perfect accuracy.

The response of the two models to the possibility matrix tool differs slightly. Gemini-2.5-Flash reduces incorrect deductions to near zero and achieves perfect accusation accuracy, suggesting that it treats the matrix as a hard constraint. GPT-4o-mini retains a similar rate of incorrect deductions across both conditions, indicating that structured output does not fully remove unsupported reasoning for all models. Despite the differences in deduction precision, both models achieve comparable win counts, suggesting that the MATRIX game outcomes are more balanced.

(RQ2) How does the presence of tool-augmented agents influence the behavior and performance of non-tool agents in mixed-agent environments? Baselineprompted players in the MIXED games achieved substantially higher accusation accuracy than in the BASELINE games, de spite no change to their prompt. Their deduction profiles across the two experiments are fairly similar, and so their accusation accuracy improvement reflects a possible game environment effect: tool-augmented players producing more informative suggestions raises the quality of shared evidence available to all players regardless of prompt type.

More specifically, this finding suggests that the benefits of structured belief states are not confined to the agents that use it directly. Even a minority of tool-augmented agents is sufficient to improve group-level accusation accuracy close to that of a fully tool-augmented player game. At the same time, the win distribution in the MIXED games reveals an asymmetry among the model families, as all wins were taken by GPT-4o-mini players regardless of prompt type, while Gemini-2.5- Flash players won none in either role. This observed pattern suggests that win outcomes in MIXED games may be more affected by model behavior (e.g., willingness to accuse) than by whether or not a player used the possibility matrix tool.

(RQ3) How does structured reasoning support affect decision-making, particularly with respect to risk-taking and the timing of final decisions? In our experiments, we identified a notable turn gap between the solution knowledge and accusation. The accusation accuracy analysis available with the tool-augmented players suggests that agents were not strictly using the structured belief representation for their action selection. This observed behavior indicates that agentic players retain autonomy in their decisionmaking, as we did not require that they accuse when they received the accusation accuracy signal, only that they use the information in the possibility matrix to make their turn. Adjusting the timing of final accusations would likely require a prompt-based intervention; these results are consistent with prior work highlighting tool-use challenges of autonomous agents.

![](images/2423b6c552f8371a73d438703fa7386829fb64b2d46828ba9b6c5793e3d7c14a.jpg)

![](images/40593a7225c01fc1be13892b02bc5c6eed8714beb6d65aff3247e0d9d29d6238.jpg)

![](images/405af8cf8052ecde47872f6499aba874a915e968d03745ddd7ff81a81f93352d.jpg)

Figure 5: Mean cards learned through play per round, averaged across all players within each simulation. Shaded bands show the standard error mean across games. The dotted line at 18 marks the the ceiling (18 non-solution cards minus the 3 dealt to each player.)  
![](images/291e1c353412bb4ee6e39752c150b3f38376084c9b3a22045ac69c161aa4937c.jpg)  
Figure 6: Per-game solution timing in the Matrix (A) and Mixed (B) conditions. Each row is one game. The orange circle marks the first turn on which the complete solution was suggested and went undisproved; the green diamond marks the turn of the correct accusation. Gap labels (+N) show the number of turns (6 turns per round). The side column indicates whether the same player who first identified the solution also made the final accusation (green) or a different player did (red).

Our hypothesized bottleneck in unstructured game play was the ability to maintain a logically consistent belief state over sequential observations, as Ansell and Toney [2026] found that text-based fine-tuning on a related task worsened agentic play. We found that our tool-augmented approach enables the models to access a structured representation of the game state, while preventing the accumulation of unsupported inferences that are characteristic of the Baseline players’ reasoning. This effect is highlighted in the reduction of incorrect deductions across both models with the introduction of the possibility matrix tool.

In summary, our findings suggest that tool augmentation targeting the offloading of extended-memory reasoning and belief tracking leads to substantial improvements in tasks requiring sustained deductive inference over sequential, long interactions. However, tool-augmentation (as we implemented it) does not necessarily directly affect the autonomy of agents in their decision-making.

## 6 Limitations and Future Work

While our findings provide evidence that structured belief tracking can support deductive reasoning, we outline several limitations as follows:

First, our proposed possibility matrix is a task-specific representation designed around the deductive structure of the game Clue. The matrix explicitly encodes card ownership constraints, maintains a structured belief state, and provides an accusation ready signal when only a single solution remains. As a result, the tool is designed for a specific environment as opposed to a generalizable solution. Future work should investigate whether similar representations generalize beyond Clue to other reasoning domains, including tasks requiring inductive or abductive reasoning, as well as other deductive environments with different state structures and constraints.

Second, the experimental evaluation is limited in scale. We evaluate two model families (Gemini-2.5-Flash and GPT-4omini) across 18 total games, with six games per experimental condition. Although this setup produces consistent qualitative trends, the number of models and independent game runs is small-scale for a stochastic, multi-agent environment. Our experiments provide preliminary results suggesting that external belief tracking tools are a promising approach for improved agentic deductive reasoning. Future work should conduct larger-scale evaluations across additional models and game simulations to assess the robustness and generality of these results. In particular, comparisons involving larger reasoning-oriented models may help clarify whether the observed benefits arise primarily from compensating for limitations in lightweight models or represent a broader advantage of structured belief tracking.

Lastly, our baseline comparisons focus on agents that reason from natural-language game histories. While this establishes a clear contrast between unstructured and structured state representations, it does not isolate which aspects of the possibility matrix are responsible for the observed improvements. For example, deduction and gameplay improvement may arise from reduced context length, explicit constraint representation, improved memory retention, or some combination of these factors. Future work could compare against intermediate baselines, such as summarized histories, modelgenerated tables, or alternative memory-augmentation strategies.

## 7 Conclusion

Achieving strong performance in extended-memory deductive reasoning tasks remains challenging for autonomous agents. In this work, we investigate whether offloading belief state tracking (previously maintained through natural language logs) to an external tool can improve agent performance. We evaluate this approach within a text-based implementation of the board game Clue, a dynamic multi-agent environment in which outcomes depend on the interactions, strategies, and information exchange between players, rather than static question-answering tasks with fixed ground truth (e.g., “What is the capital of France”).

Our results show that tool augmentation substantially improves gameplay performance, achieving near-perfect accuracy for tool-augmented agents while also benefiting baseline agents in mixed settings. More broadly, these results suggest that when belief states are presented to an agent in a more succinct and structured representation (over natural language reasoning logs) lightweight agents can perform nearoptimally in complex, interactive settings.

## References

[Ansell and Toney, 2026] Rebecca Ansell and Autumn Toney. How clued up are LLMs? evaluating multi-step deductive reasoning in a text-based game environment. In ICLR 2026 Workshop on Logical Reasoning of Large Language Models, 2026.

[Arcadinho et al., 2024] Samuel Arcadinho, David Oliveira Aparicio, and Mariana SC Almeida. Automated test generation to evaluate tool-augmented llms as conversational ai agents. In Proceedings of the 2nd GenBench Workshop on Generalisation (Benchmarking) in NLP, pages 54–68, 2024.

[Chalamalasetti et al., 2023] Kranti Chalamalasetti, Jana Gotze, Sherzod Hakimov, Brielen Madureira, Philipp ¨ Sadler, and David Schlangen. clembench: Using game play to evaluate chat-optimized language models as conversational agents. In Proceedings ofthe 2023 conference

on empirical methods in natural language processing, pages 11174–11219, 2023.

[Chen et al., 2023] Zhipeng Chen, Kun Zhou, Beichen Zhang, Zheng Gong, Wayne Xin Zhao, and Ji-Rong Wen. Chatcot: Tool-augmented chain-of-thought reasoning on chat-based large language models. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 14777–14790, 2023.

[Chen et al., 2024] Zehui Chen, Weihua Du, Wenwei Zhang, Kuikun Liu, Jiangning Liu, Miao Zheng, Jingming Zhuo, Songyang Zhang, Dahua Lin, Kai Chen, et al. T-eval: Evaluating the tool utilization capability of large language models step by step. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9510–9529, 2024.

[Chi et al., 2024] Yizhou Chi, Lingjun Mao, and Zineng Tang. Amongagents: Evaluating large language models in the interactive text-based social deduction game. In Proceedings of the 4th Wordplay: When Language meets Games Workshop. Association for Computational Linguistics, 2024.

[Chu et al., 2024] Zheng Chu, Jingchang Chen, Qianglong Chen, Weijiang Yu, Tao He, Haotian Wang, Weihua Peng, Ming Liu, Bing Qin, and Ting Liu. Navigate through enigmatic labyrinth a survey of chain of thought reasoning: Advances, frontiers and future. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1173– 1203, Bangkok, Thailand, August 2024. Association for Computational Linguistics.

[Comanici et al., 2025] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

[Das et al., 2024] Debrup Das, Debopriyo Banerjee, Somak Aditya, and Ashish Kulkarni. Mathsensei: a toolaugmented large language model for mathematical reasoning. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 942–966, 2024.

[FAIR et al., 2022] Meta Fundamental AI Research Diplomacy Team FAIR, Anton Bakhtin, Noam Brown, Emily Dinan, Gabriele Farina, Colin Flaherty, Daniel Fried, Andrew Goff, Jonathan Gray, Hengyuan Hu, Athul Paul Jacob, Mojtaba Komeili, Karthik Konath, Minae Kwon, Adam Lerer, Mike Lewis, Alexander H. Miller, Sasha Mitts, Adithya Renduchintala, Stephen Roller, Dirk Rowe, Weiyan Shi, Joe Spisak, Alexander Wei, David Wu, Hugh Zhang, and Markus Zijlstra. Human-level play in the game of ¡i¿diplomacy¡/i¿ by combining language models with strategic reasoning. Science, 378(6624):1067–1074, 2022.

[Gevers and Daelemans, 2025] Ine Gevers and Walter Daelemans. Do you get the hint? benchmarking llms on the board game concept. arXiv preprint arXiv:2510.13271, 2025.

[Hao et al., 2023] Shibo Hao, Tianyang Liu, Zhen Wang, and Zhiting Hu. Toolkengpt: Augmenting frozen language models with massive tools via tool embeddings. Advances in neural information processing systems, 36:45870– 45894, 2023.

[Hendrycks et al., 2021] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset, 2021.

[Hu et al., 2025] Lanxiang Hu, Qiyu Li, Anze Xie, Nan Jiang, Ion Stoica, Haojian Jin, and Hao Zhang. Gamearena: Evaluating LLM reasoning through live computer games. In The Thirteenth International Conference on Learning Representations, 2025.

[Huang et al., 2025] Jen-tse Huang, Eric John Li, Man Ho Lam, Tian Liang, Wenxuan Wang, Youliang Yuan, Wenxiang Jiao, Xing Wang, Zhaopeng Tu, and Michael Lyu. Competing large language models in multi-agent gaming environments. In The Thirteenth International Conference on Learning Representations, 2025.

[Jung et al., 2025] Sunghee Jung, Donghun Lee, Shinbok Lee, Gaeun Seo, Daniel Lee, Byeongil Ko, Junrae Cho, Kihyun Kim, Eunggyun Kim, and Myeongcheol Shin. Diatool-dpo: Multi-turn direct preference optimization for tool-augmented large language models. In Proceedings of the 26th Annual Meeting of the Special Interest Group on Discourse and Dialogue, pages 397–416, 2025.

[Kwak et al., 2025] Beong-woo Kwak, Minju Kim, Dongha Lim, Hyungjoo Chae, Dongjin Kang, Sunghwan Kim, Dongil Yang, and Jinyoung Yeo. ToolHaystack: Stresstesting tool-augmented language models in realistic longterm interactions. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng, editors, Findings of the Association for Computational Linguistics: EMNLP 2025, pages 24696–24727, Suzhou, China, November 2025. Association for Computational Linguistics.

[Lin et al., 2025] Wenye Lin, Jonathan Roberts, Yunhan Yang, Samuel Albanie, Zongqing Lu, and Kai Han. GAMEBoT: Transparent assessment of LLM reasoning in games. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors, Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7656–7682, Vienna, Austria, July 2025. Association for Computational Linguistics.

[Liu et al., 2023] Hanmeng Liu, Jian Liu, Leyang Cui, Zhiyang Teng, Nan Duan, Ming Zhou, and Yue Zhang. Logiqa 2.0—an improved dataset for logical reasoning in natural language understanding. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 31:2947– 2962, 2023.

[Ma et al., 2024] Yubo Ma, Zhibin Gou, Junheng Hao, Ruochen Xu, Shuohang Wang, Liangming Pan, Yujiu Yang, Yixin Cao, and Aixin Sun. Sciagent: Toolaugmented language models for scientific reasoning. In Proceedings of the 2024 conference on empirical methods in natural language processing, pages 15701–15736, 2024.

[OpenAI, 2024] OpenAI. Gpt-4o system card, 2024.

[Parisi et al., 2022] Aaron Parisi, Yao Zhao, and Noah Fiedel. Talm: Tool augmented language models. arXiv preprint arXiv:2205.12255, 2022.

[Patel et al., 2024] Nisarg Patel, Mohith Kulkarni, Mihir Parmar, Aashna Budhiraja, Mutsumi Nakamura, Neeraj Varshney, and Chitta Baral. Multi-LogiEval: Towards evaluating multi-step logical reasoning ability of large language models, November 2024.

[Patil et al., 2024] Shishir G Patil, Tianjun Zhang, Xin Wang, and Joseph E Gonzalez. Gorilla: Large language model connected with massive apis. Advances in Neural Information Processing Systems, 37:126544–126565, 2024.

[Qu et al., 2025] Changle Qu, Sunhao Dai, Xiaochi Wei, Hengyi Cai, Shuaiqiang Wang, Dawei Yin, Jun Xu, and Ji-Rong Wen. Tool learning with large language models: A survey. Frontiers of Computer Science, 19(8):198343, 2025.

[Shim et al., 2025] Jeonghoon Shim, Gyuhyeon Seo, Cheongsu Lim, and Yohan Jo. Tooldial: Multi-turn dialogue generation method for tool-augmented language models. In The Thirteenth International Conference on Learning Representations, 2025.

[Talmor et al., 2019] Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. CommonsenseQA: A question answering challenge targeting commonsense knowledge. In Jill Burstein, Christy Doran, and Thamar Solorio, editors, Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4149–4158, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics.

[Trencsenyi et al., 2025] Vince Trencsenyi, Agnieszka Mensfelt, and Kostas Stathis. Approximating human strategic reasoning with llm-enhanced recursive reasoners leveraging multi-agent hypergames. In International Workshop on Multi-Agent Systems and Agent-Based Simulation, pages 15–27. Springer, 2025.

[Xu et al., 2025] Zelai Xu, Chao Yu, Fei Fang, Yu Wang, and Yi Wu. Language agents with reinforcement learning for strategic play in the werewolf game, 2025.

[Xu et al., 2026] Zhichao Xu, Zongyu Wu, Yun Zhou, Aosong Feng, Kang Zhou, Sangmin Woo, Kiran Ramnath, Yijun Tian, Xuan Qi, Weikang Qiu, Lin Lee Cheong, and Haibo Ding. Beyond correctness: Rewarding faithful reasoning in retrieval-augmented generation, 2026.

[Yu et al., 2020] Weihao Yu, Zihang Jiang, Yanfei Dong, and Jiashi Feng. Reclor: A reading comprehension dataset requiring logical reasoning, 2020.

[Zeng et al., 2024] Zhongshen Zeng, Pengguang Chen, Shu Liu, Haiyun Jiang, and Jiaya Jia. Mr-gsm8k: A metareasoning benchmark for large language model evaluation, 2024.

[Zheng et al., 2025] Tianshi Zheng, Yixiang Chen, Chengxi Li, Chunyang Li, Qing Zong, Haochen Shi, Baixuan Xu, Yangqiu Song, Ginny Y. Wong, and Simon See. The curse of cot: On the limitations of chain-of-thought in in-context learning, 2025.

[Zhuang et al., 2025] Richard Zhuang, Akshat Gupta, Richard Yang, Aniket Rahane, Zhengyu Li, and Gopala Anumanchipalli. Pokerbench: Training large language models to become professional poker players, Apr. 2025.