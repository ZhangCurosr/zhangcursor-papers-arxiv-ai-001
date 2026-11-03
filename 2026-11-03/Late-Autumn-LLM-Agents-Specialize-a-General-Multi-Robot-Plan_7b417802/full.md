# Late Autumn: LLM Agents Specialize a General Multi-Robot Planner for One Warehouse

Yimin Tang<sup>1</sup>, Han Zhang<sup>3</sup>, Shao-Hung Chan<sup>3</sup>, Erdem Bıyık<sup>1</sup>, Sven Koenig<sup>2</sup>, Jingkai Chen<sup>3</sup>

<sup>1</sup>Thomas Lord Department of Computer Science, University of Southern California

<sup>2</sup>Department of Computer Science, University of California Irvine <sup>3</sup>Symbotic

{yimintan,biyik}@usc.edu, svenk@uci.edu, {hanzhang,shchan,jichen}@symbotic.com

## Abstract

No algorithm is best on all problems, but a real warehouse is a narrow problem: its layout and processes rarely change. Operators still run one general multi-robot planner at many sites, because experts cannot tune it for each one. In this work, which we call Late Au tumn, we explore whether LLM coding agents running autonomous research can specialize a general planner for one site, trading generality for throughput, much as post-training adapts a pretrained LLM to one task. We study six campaigns on a lifelong planner that combines task assignment with multi-agent path finding, on a floor modeled on an existing automated warehouse. We find that (1) the agents raise throughput by 22–60%, and the gains hold on unseen seeds; (2) they rediscover known MAPF mechanisms, such as corridor-exit priorities, lo cal guidance, and rollouts, and independent campaigns find the same bottleneck; (3) the gains largely carry over to another layout, a larger fleet, and other service times, shrink with a smaller fleet, and mostly vanish under an outbound-heavy demand mix; and (4) each final patch (0.7k–2.4k lines) is a chain of scored commits with a median step of 27 lines, so experts can review it, and review finds problems that the score misses; some patches multiply planner compute by over 100×. LLM agents can therefore turn each site’s bias into extra throughput at little engineering cost, with reviewable patches; we expect experts to shift toward designing insightful metrics and fast simulators that mirror the real problem.

## 1 Introduction

Automated warehouses move goods with fleets of robots that shuttle cases between storage aisles and stations. Commercial systems take many forms (Figure 1): drive units that carry shelves to pickers (Amazon Robotics, Geek+), bots that carry cases through multi-level aisles (Symbotic), robots on a grid above stacked bins (AutoStore), rack-climbing tote robots (Hai Robotics), and robots that work beside human pickers (Locus Robotics). The coordination layer must repeatedly assign tasks to robots and plan collision-free paths for all of them, a lifelong variant of Multi-Agent Path Finding (MAPF) with target assignment (Ma et al., 2017; Li et al., 2021; Ma and Koenig, 2016). MAPF is NP-hard to solve optimally (Yu and LaValle, 2013), so large fleets call for fast suboptimal planners such as PIBT (Okumura et al., 2022) and LaCAM (Okumura, 2023b). We use a lifelong version of ITA-LaCAM (Tang et al., 2026), which couples target assignment with LaCAM search. As a concrete example, we model the floor of one existing automated warehouse; since MAPF models any AGV warehouse as a graph with tasks, the approach is not tied to it.

![](images/2dcf579c81eabdff54c1fd1b99b4d98dfd9b6608d213e05ab91fdc7692397ec3.jpg)  
Figure 1: Commercial warehouse robot systems.

This planner is a general algorithm, designed for a broad class of layouts and workloads. By the No Free Lunch theorems, averaged over all possible problems, every search algorithm performs equally well (Wolpert and Macready, 1997). Real sites are far from this average: warehouses differ in aisle geometry, fleet size, carry capacity, demand mix, and service times, but each one runs the same layout and largely fixed processes every day. A planner can therefore trade generality that one site never needs for throughput, much as post-training specializes a pretrained LLM. Until now, only experts could exploit this bias, by inspecting traces and changing cost or priority rules, and no company can do this for its growing number of sites. An agent that can would give each site extra throughput at almost no engineering cost.

![](images/102d4f068fa1c685052fc3d10d05f7654bc942d66ba5d5443afbc202645e9e01.jpg)  
(a) auto-research loop

![](images/053958c7cdba7e61728d10d63c99f4093f0e833598e52df58490a0935915b6f7.jpg)  
(b) tuned site  
Figure 2: (a) The agent edits a general planner for one site, self-tests on public seeds with traces, and is scored only on hidden seeds; experts review the diffs. (b) The tuned site: aisles and station lanes are one-wide corridors, and one cell links the two halves.

LLM-based coding agents now perform parts of this loop (Novikov et al., 2025; Qu et al., 2026), but most of this evidence comes from short programs with cheap, deterministic scores. A warehouse planner differs in three ways. First, the agent edits a complete planner (assignment, search, and simulation) rather than a single heuristic function. Second, the score comes from a stochastic simulation in which many changes are smaller than the noise, so an agent can “improve” by chance. Third, a change reaches physical robots only after an engineer understands it and trusts it. We therefore ask:

RQ1 After autonomous research on one site configuration, does the planner’s throughput really improve?

RQ2 What methods do the agents propose?

RQ3 Are these methods site-specific, or do they work for other warehouse configurations?

RQ4 How large is each improving change, and can an expert review it to confirm, from first principles, that it is correct?

In this work, which we call Late Autumn, we explore this idea with an existing auto-research system and answer these questions with a case study. An agent receives the planner repository and a task contract that fixes one site configuration. It may change any planner logic, but it is scored only on hidden seeds by an isolated host, and it must commit and evaluate every candidate. We ran six campaigns (three model families, three knowledge settings; 600 official evaluations), re-evaluated each best patch on fresh seeds and on seven held-out configurations, and analyzed every diff that raised the best score. Our results show that every campaign’s best patch beats the baseline on all 20 fresh seeds, by 22–63%, and that the patches keep a significant gain in 40 of 42 held-out configurations. Our contributions are:

1. A case study of agent-driven specialization of a lifelong MAPF planner to one site, run with hidden-seed scoring, a mandatory commit-andevaluate lifecycle, and compute accounting.

2. Evidence that agents raise throughput by 22– 60% (RQ1).

3. A catalog of the discovered mechanisms (RQ2), a transfer study (RQ3), and a code-level analysis (RQ4): improving steps are small (median 27 lines), so even the final patches can be reviewed commit by commit, and review finds problems that the score misses.

## 2 Problem Definition and Baseline

Lifelong TAPF with multi-carry robots. Let $G = ( V , E )$ be a 4-connected grid graph whose vertices are free cells; self-loops represent waiting. Agents $I = \{ 1 , \ldots , N \}$ start at distinct vertices and each can carry at most K tasks. Tasks are released online; a task $\boldsymbol { \tau } = \left( \boldsymbol { s } _ { \tau } , \boldsymbol { G } _ { \tau } \right)$ has a pickup vertex s<sub>τ</sub> and a goal set $G _ { \tau } \subseteq V$ , any vertex of which completes the delivery. Let $x _ { i } ( t ) \in V$ be the location of agent i at timestep t. A plan is valid if, for all agents $i \neq j$ and timesteps t: 1) $x _ { i } ( t + 1 ) { = } x _ { i } ( t )$

or $( x _ { i } ( t ) , x _ { i } ( t { + } 1 ) ) \in E ; 2 ) x _ { i } ( t ) \neq x _ { j } ( t )$ (no vertex conflict); 3) agents never swap vertices (no edge conflict); and 4) agent i picks up τ only by staying at $s _ { \tau }$ for $d _ { p }$ consecutive timesteps while it carries fewer than K tasks, and delivers τ only by staying at some $g \in G _ { \tau }$ for $d _ { d }$ timesteps. The objective is throughput: tasks completed within horizon H, divided by H. Each planner call has a 1 s wall-clock limit.

Baseline planner. Our planner, a lifelong version of ITA-LaCAM (Tang et al., 2026), replans after every pickup or delivery or when its plan runs out. A Hungarian matching assigns each agent one goal option (a pickup, a goal cell of a carried case, or a costly wait), costed by distance plus service time, with at most one agent per pickup or drop cell; as in ITA-CBS (Tang et al., 2023), the matching is repaired incrementally at each search node. LaCAM (Okumura, 2023b) with FOCAL node selection then generates configurations with PIBT (Okumura et al., 2022) until some agent completes a service. The planner has 9.0k lines of C++ and needs 4.8 ms per call at the tuned site (§4): its anytime refinement exits after the first solution (the stopping test compares quantities on different scales), so over 99% of the budget is unused. It reaches a hidden-seed throughput of 1.23 (pseudocode in Appendix A).

## 3 Related Work

MAPF in warehouses. PIBT and LaCAM plan for hundreds of agents in milliseconds (Okumura et al., 2022; Okumura, 2023b,a), unlike optimal CBS (Sharon et al., 2015; Stern et al., 2019). TAPF couples target assignment with path finding (Ma and Koenig, 2016; Hönig et al., 2018; Tang et al., 2023, 2024; Kumagai and Okumura, 2026), and lifelong variants model warehouses with continuous task arrival (Ma et al., 2017; Li et al., 2021; Chan et al., 2024b). Guidance-graph optimization (Zhang et al., 2024; Zang et al., 2024) and imitation learning (Jiang et al., 2024) raise throughput; learned planners and fleet foundation models serve many maps (Tang et al., 2025; Agaskar et al., 2025). These methods are designed by researchers for a class of instances; we study whether an agent can do the remaining site-specific engineering.

LLM agents for algorithm engineering. LLMdriven program search discovered new constructions and heuristics (Romera-Paredes et al., 2024;

Novikov et al., 2025; Lange et al., 2025; Liu et al., 2024; Ye et al., 2024), agentic systems run research loops over code (Jiang et al., 2025; Lu et al., 2024; Qu et al., 2026), and benchmarks evaluate them (Chan et al., 2024a; Huang et al., 2023; Jimenez et al., 2024; Press et al., 2025; Imajuku et al., 2025). There, programs are small and scores are cheap. LLMs fail when they plan MAPF paths directly (Chen et al., 2024); we let them write planner code instead, so the planner remains a verifiable search algorithm.

## 4 Auto-Research Setup

In this section, we describe the auto-research setup: the agent’s tools, the task contract it must follow, and how we re-evaluate its results. We use CORAL (Qu et al., 2026) to run one autonomous coding agent per campaign in its own git worktree (Figure 2a). The agent can read and edit all code, build, test, inspect traces, and request evaluations. A natural-language task contract (Appendix C) sets the rules below.

Site and workload. The site the agents tune (Figure 2b) follows one existing warehouse floor: a 35×39 grid with 691 free cells, 19 one-wide storage aisles of 20 cells, and 8 one-wide station lanes; a wall splits it into two halves joined by a single cell. An outbound task (probability $p _ { \mathrm { o u t } } { = } 0 . 5 )$ starts at a storage cell and may be delivered to any of 3 cells of one station lane; an inbound task starts at a station cell and may be delivered to any of 3 storage cells. New tasks are released so that 2N tasks stay unfinished, which keeps the system saturated. We use N=100, K=2, H=400, and $d _ { p } { = } d _ { d } { = } 4$

Fixed site, free algorithm. The contract fixes this site and workload. The agent may change any planner, assignment, or goal-selection logic and any algorithm-side default. It may not modify the map, the task process, or the benchmark runner (the evaluation host overwrites them), hard-code seeds or map-specific shortcuts, or fabricate outputs; the grader recomputes throughput from the completed tasks.

Isolated evaluation. Throughput depends on how much search fits into 1 s, so builds and benchmarks run on a separate host with pinned 10-core slots. A self-test runs public seeds 0–9 and returns per-seed traces. An official evaluation runs 10 hidden seeds and returns only aggregates. The same binary is reproducible on one slot, but contention can move the 10-seed mean by about 0.025; six evaluations of the unmodified baseline spanned 1.198–1.249. The contract therefore treats mean changes below 0.03 as noise.

Commit-and-evaluate lifecycle. Every candidate that changes behavior, builds, and yields valid output must be committed and officially evaluated before the agent keeps, reverts, or abandons it. This rule prevents silent cherry-picking and gives us the complete record. CORAL adds periodic prompts to reflect (every 2 evaluations), consolidate (every 8), and pivot (on plateaus). A campaign stops after 100 official evaluations.

Safety property. The simulator validates every executed joint move for vertex and edge conflicts. As long as a patch leaves this validator unchanged, which review confirmed for all 102 new-best steps, a patch can lower throughput but cannot produce a collision in an executed plan.

Knowledge settings and campaigns. In the paper-grounded setting, every idea must cite one of 137 briefs of MAPF-related papers and competition reports. In the open setting, any source is allowed and must be named; open+web also enables web search. We ran six campaigns from the same commit at high reasoning effort, with four models in three agent harnesses (Table 1).

Re-evaluation and transfer. A campaign’s best hidden score is a maximum over noisy evaluations. We therefore rebuilt the baseline and each best commit and ran them on 20 fresh seeds for the tuned site and for seven held-out configurations, each changing one factor: an alternative layout (Appendix B), 60 or 150 robots, K=1, an 80/20 outbound/inbound mix, and service durations of 2 or 8, a total of 1,120 episodes (7 builds, 8 configurations, 20 seeds). Gain is the paired throughput difference to the baseline on the same seeds, with a 95% CI over seeds; a new-best step is an official evaluation that raises a campaign’s best hidden score; its size is the number of changed lines of algorithm code, excluding tests, tools, and build files. All builds use GCC 10.5 -O3; the re-evaluation runs used a 32-core AMD EPYC 9R14 host with 64 GB RAM.

## 5 Results

In our experiments, we evaluate the best patch of each campaign against the unmodified baseline and answer the four research questions in turn.

![](images/e5c2af7ef54466ca2a10671a8197133c3d24f0f5894345c02522ecdfb87b8cc0.jpg)  
Figure 3: (a) Best-so-far hidden-seed throughput (dots: single evaluations, those below 1.0 on the bottom edge; gray band: six baseline evaluations). (b) Size of the best commit so far (changed lines of algorithm code); it drops when a new best comes from a branch with less code. Solid: paper-grounded; dashed: open.

## 5.1 RQ1: Does throughput really improve?

Yes, by 22–60%. Every campaign raised hiddenseed throughput far beyond the noise floor (Table 1, Figure 3). The best campaign (Opus, papergrounded) reached 1.973 tasks per timestep, 60% above the baseline mean of 1.230. The four Claudefamily campaigns gained 43–60%, and both GPT campaigns plateaued near 1.50 (+22%).

The gains hold on fresh seeds. On 20 seeds that no agent had seen, every best patch scores within 0.001–0.024 of its hidden best (Table 1), so selection bias is small, and beats the baseline on all 20 seeds; for example, the Opus/paper patch reaches 1.968 (+63%).

Gains come early, partly from compute. The Claude-family campaigns reached 65–86% of their final gain within 20 evaluations (0.6–1.9 hours); only 27 of the 102 new-best steps exceed the 0.03 noise floor. Relative to their baseline evaluation, the paper-grounded Claude patches use 4–10× more planner time and the open-setting patches 53–91× (mean calls of 0.3–0.6 s against the 1 s limit). GPT-6.1 Sol’s last new best (+0.008) raised its planner time 87-fold (112× its baseline).

<table><tr><td>Model</td><td>Harness</td><td>Ideas</td><td>Evals</td><td>Hours</td><td>Hidden best TP</td><td>gain</td><td>Fresh seeds TP</td><td>gain</td><td>Evals to 90% gain</td><td>Planner s / episode</td></tr><tr><td>Unmodified baseline (6 evals)</td><td></td><td></td><td></td><td>一</td><td>1.230</td><td></td><td>1.209</td><td></td><td></td><td>1.8-3.8</td></tr><tr><td>Opus 5.5</td><td>Claude Code</td><td>paper</td><td>100</td><td>7</td><td>1.973</td><td>+60%</td><td>1.968</td><td>+63%</td><td>24</td><td>21.2</td></tr><tr><td>Opus 5.5</td><td>Claude Code</td><td>open</td><td>100</td><td>12</td><td>1.875</td><td>+52%</td><td>1.869</td><td>+55%</td><td>73</td><td>189.9</td></tr><tr><td>Fable 5.1</td><td>Kiro</td><td>paper</td><td>100</td><td>8</td><td>1.884</td><td>+53%</td><td>1.861</td><td>+54%</td><td>67</td><td>11.3</td></tr><tr><td>Fable 5.1</td><td>Kiro</td><td>open+web†</td><td>100</td><td>9</td><td>1.758</td><td>+43%</td><td>1.746</td><td>+44%</td><td>27</td><td>94.1</td></tr><tr><td>GPT-6 Luna</td><td>Codex</td><td>paper</td><td>100</td><td>22</td><td>1.500</td><td>+22%</td><td>1.476</td><td>+22%</td><td>83</td><td>2.9</td></tr><tr><td>GPT-6.1 Sol</td><td>Codex</td><td>paper</td><td>100</td><td>34</td><td>1.498</td><td>+22%</td><td>1.497</td><td>+24%</td><td>49</td><td>342.1</td></tr></table>

Table 1: The six campaigns. Hidden-best gain: relative to the baseline mean 1.230. Fresh seeds: best commit on 20 unseen seeds. Hours: active run time. Planner s: per 400-step episode (baseline: range over six evaluations). <sup>†</sup>Web search enabled but never used.

![](images/a157a58928087f366961eb0512ef8851a7483925c6e662ff59fc82bafcffe1da.jpg)

![](images/aeb2ccb120716de82d6c09a623654c15a85df1321a2bde896d0b5cd1af759e40.jpg)  
Figure 4: (a) Hidden-score gain per campaign by mechanism group; each new-best step counts toward its main change, and retunes toward the mechanism they tune. (b) Size of each new-best step versus its increment; filled marks exceed the 0.03 noise floor. Sizes count algorithm code only.

<table><tr><td>Mechanism (pipeline stage)</td><td>OP 00 FP FW</td><td>GL</td><td>GS</td></tr><tr><td>Corridor-exit priority (PIBT)</td><td>.10 .14 .09</td><td>.03* 一</td><td></td></tr><tr><td>Corridor-blocking cost (assign.) Local guidance (PIBT order)</td><td>.12 .28 .35 .04</td><td>.20 X</td><td>一 一</td></tr><tr><td>Flow-aware distances / routes</td><td>.00 .13</td><td>X .03</td><td>X .17</td></tr><tr><td>Time-aware assignment</td><td>.13</td><td>一</td><td></td></tr><tr><td>Rollouts / Monte-Carlo PIBT</td><td>.00 .12</td><td>.16 .51</td><td>X .02</td></tr><tr><td>K=2 insertion / pairing</td><td>.03 一</td><td>.03</td><td>.11 .04</td></tr><tr><td>Search-level changes</td><td>.05 一</td><td>一 一 一</td><td>.02 .01</td></tr></table>

Table 2: Gain of the new-best steps that introduced or tuned each recurring mechanism (×: tried, never a new best). OP/OO: Opus paper/open; FP/FW: Fable paper/open+web; GL/GS: GPT-6 Luna/6.1 Sol. <sup>∗</sup>Around articulation cells only.

## 5.2 RQ2: What did the agents propose?

We read all 102 diffs that raised the best score, together with the agents’ notes (Table 2, Figure 4a).

Independent campaigns found the same bottleneck. On this floor, 470 of the 691 free cells lie in one-wide corridors, and a servicing robot blocks everyone behind it. Three Claude campaigns found this, from traces (“corridor entrances are occupied 89% of the time”) or from a cited paper, and converged on two rules. A corridor-exit priority gives a PIBT priority boost to a robot inside a corridor whose goal lies outside it, so that priority inheritance pushes entering robots back instead of pushing the leaving robot deeper (+0.09 to +0.14; cf. PIBTTP (Fujitani et al., 2022)). A corridor-blocking cost makes pickup and drop cells behind occupied corridor cells more expensive in the matching (+0.12 to +0.28; cf. idle-time-aware assignment (Kou et al., 2019)). All three detected corridors generically, as components of degree-2 vertices. These rules provide 28–67% of those campaigns’ gains.

Agents spent the idle budget. Agents ported local guidance (Arita and Okumura, 2025, 2026) (+0.35 in Opus/paper), added flow-dependent edge weights (Zhang et al., 2024; Zang et al., 2024) (+0.13 in Fable/paper), and built rollout planners; Fable/open+web replaced LaCAM’s search with the best of many noisy PIBT rollouts (+0.38 in one step).

Sources and failures. Paper-grounded campaigns cited a brief in 97–99% of evaluations but used only 18–44 of 137 briefs; 79% of Opus/open candidates came from its own diagnosis; the web was never used. Comparing lineages, we find two reasons why the GPT campaigns plateaued near 1.50. First, they turned to the library instead of the traces and never built corridor rules for the aisles; the same local-guidance paper gave Opus +0.24 but GPT 0.98–1.30, because GPT used a 5-step window and no warm start. Second, they followed CORAL’s “hardest idea, three evaluations” prompt literally, with median attempts of 66 and 290 added lines (Claude: 4–16).

<table><tr><td rowspan=2 colspan=1></td><td rowspan=1 colspan=2>tuņed new</td><td rowspan=1 colspan=2>robots</td><td rowspan=1 colspan=1>carry8</td><td rowspan=1 colspan=1>0/20</td><td rowspan=1 colspan=2>service</td></tr><tr><td rowspan=1 colspan=1>site l</td><td rowspan=1 colspan=1>ayou</td><td rowspan=1 colspan=1>t60</td><td rowspan=1 colspan=1>150</td><td rowspan=1 colspan=1>K=1</td><td rowspan=1 colspan=1>mix</td><td rowspan=1 colspan=2>d=2 d=8</td></tr><tr><td rowspan=1 colspan=1>Opus / paper</td><td rowspan=1 colspan=1>+63</td><td rowspan=1 colspan=1>+38</td><td rowspan=1 colspan=1>+28</td><td rowspan=1 colspan=1>+68</td><td rowspan=1 colspan=1>+54</td><td rowspan=1 colspan=1>+10</td><td rowspan=1 colspan=1>+55</td><td rowspan=1 colspan=1>+54</td></tr><tr><td rowspan=1 colspan=1>Opus / open</td><td rowspan=1 colspan=1>+55</td><td rowspan=1 colspan=1>+32</td><td rowspan=1 colspan=1>+26</td><td rowspan=1 colspan=1>+55</td><td rowspan=1 colspan=1>+50</td><td rowspan=1 colspan=1>+8</td><td rowspan=1 colspan=1>+48</td><td rowspan=1 colspan=1>+49</td></tr><tr><td rowspan=1 colspan=1>Fable / paper</td><td rowspan=1 colspan=1>+54</td><td rowspan=1 colspan=1>+36</td><td rowspan=1 colspan=1>+27</td><td rowspan=1 colspan=1>+39</td><td rowspan=1 colspan=1>+45</td><td rowspan=1 colspan=1>+3</td><td rowspan=1 colspan=1>+48</td><td rowspan=1 colspan=1>+44</td></tr><tr><td rowspan=1 colspan=1>Fable / open+web</td><td rowspan=1 colspan=1>+44</td><td rowspan=1 colspan=1>+18</td><td rowspan=1 colspan=1>+16</td><td rowspan=1 colspan=1>+44</td><td rowspan=1 colspan=1>+46</td><td rowspan=1 colspan=1>+5</td><td rowspan=1 colspan=1>+38</td><td rowspan=1 colspan=1>+39</td></tr><tr><td rowspan=1 colspan=1>GPT-6 Luna</td><td rowspan=1 colspan=1>+22</td><td rowspan=1 colspan=1>+14</td><td rowspan=1 colspan=1>+9</td><td rowspan=1 colspan=1>+19</td><td rowspan=1 colspan=1>+19</td><td rowspan=1 colspan=1>-6</td><td rowspan=1 colspan=1>+19</td><td rowspan=1 colspan=1>+13</td></tr><tr><td rowspan=1 colspan=1>GPT-6.1 Sol</td><td rowspan=1 colspan=1>+24</td><td rowspan=1 colspan=1>+11</td><td rowspan=1 colspan=1>+7</td><td rowspan=1 colspan=1>+15</td><td rowspan=1 colspan=1>+23</td><td rowspan=1 colspan=1>-6</td><td rowspan=1 colspan=1>+20</td><td rowspan=1 colspan=1>+18</td></tr><tr><td rowspan=1 colspan=1>baselin</td><td rowspan=1 colspan=1>e1.21</td><td rowspan=1 colspan=1>1.66</td><td rowspan=1 colspan=1>1.06</td><td rowspan=1 colspan=1>1.22</td><td rowspan=1 colspan=1>1.07</td><td rowspan=1 colspan=1>1.20</td><td rowspan=1 colspan=1>1.50</td><td rowspan=1 colspan=1>0.92</td></tr></table>

Figure 5: Paired gain (%) of each unchanged best patch over the baseline, 20 fresh seeds per cell; all 95% CIs exclude zero. Bottom row: baseline throughput.

## 5.3 RQ3: Specific or general?

Code-level generality. We graded each frontier step as G0 (generic features only), G1 (generic code, layout-tuned constants), or G2 (scenario facts). Of 77 Claude-family steps, 12 are G0, 59 G1, and 6 G2; none hard-codes seeds, coordinates, or the map.

Transfer. Applied unchanged to seven held-out configurations (Figure 5), the patches gain significantly in 40 of 42 cells. On the alternative layout, the patches keep 57–91% of their absolute gain, because its aisles are still one-wide. We observe a clear trend with congestion: with 60 robots only 25–43% of the absolute gain remains, whereas with 150 robots 64–110% remains. This is because the discovered mechanisms target blocking in onewide corridors, which grows with fleet size. Relative gains also stay close to the tuned ones for single-carry robots and service durations of 2 and 8. The exception is the 80/20 outbound-heavy mix, which sends most deliveries to the 8 station lanes: the Claude patches keep only +3 to +10%, and both GPT patches lose 6%. Site-specific campaigns thus produced mostly general improvements, but a new demand mix calls for a new campaign.

## 5.4 RQ4: How large are the changes, and can humans review them?

Steps and patches are small. The median improving step changes 27 lines, 27 of the 102 steps only retune a constant, and the largest gains came from 36–383-line mechanisms (Figures 3b, 4b; Appendix E). The final patches (0.7k–1.5k lines for Claude, 1.3k–2.4k for GPT) are modest next to the 9k-line planner, and each is a chain of 13–49 officially scored commits (median 4–51 lines) that an expert can review one at a time; this also exposes dead code (26% of the lines added by Opus/paper).

Review is feasible and necessary. Of the 24 frontier steps of Opus/paper, 7 are constant retunes, 10 are local cost or priority rules that an expert can verify from first principles, and 7 need careful review. Review found problems that the score cannot reveal: (i) the Opus/open rollout stops at 880 ms and then starts the main solve with afresh 1 s deadline, so the per-call limit holds only empirically; (ii) a Monte-Carlo step redraws PIBT’s random tie-breaks, so the executed configuration can differ from the scored one; and (iii) dwell-sized constants and K=2 formulas are hard-coded.

## 6 Lessons and Conclusion

LLM agents can specialize a general algorithm to one site for extra performance: campaigns raised throughput by 22–60% with trace-diagnosed corridor rules and with guidance and rollouts that use idle compute, and the gains mostly transferred. As a site’s layout and processes rarely change, this turns site bias into nearly free throughput and lower logistics costs. The final patches can be reviewed commit by commit. Since agents optimize exactly what the score and simulator measure, we expect experts to shift from hand-tuning algorithms toward designing insightful metrics and fast simulators that mirror the real problem. We recommend short campaigns (about 20 evaluations), unseen-seed checks, and a graded call-time limit. Next, we plan to deploy reviewed patches on physical robots.

## Limitations

This is a case study of one planner and one family of sites. All results come from simulation; we have not yet deployed an agent-generated patch on physical robots. We ran one campaign per (model, harness, knowledge setting) combination, so differences between campaigns combine the effects of the model, the agent harness (Claude Code, Kiro, Codex), the knowledge setting, and run-to-run variance; the comparison is indicative, not causal. The best hidden score of each campaign is selected from up to 100 noisy evaluations and is therefore optimistic; we report fresh-seed re-evaluations to correct for this. Our reviewability analysis is based on diff statistics and on an LLM-assisted code reading whose key findings we checked against the source. Finally, the agents optimized throughput only. They could trade other operational metrics, such as task latency, energy use, or robot wear, which we did not measure.

## Ethical Considerations

Agent-generated changes to software that controls physical robots carry safety risk. In our protocol, agents never act on hardware; they edit planner code that is scored in simulation, and the planner’s collision checks remain the final safety layer. We argue that every patch should pass expert review and staged testing before deployment. The intended use is to let a small engineering team support many sites; this may change the work of operations engineers, and we recommend keeping humans responsible for deployment decisions. Higher throughput from the same robots can lower the cost of warehouse logistics for operators and their customers; this benefit should be weighed against the extra planner compute that some patches need. No personal data was used.

## References

Ameya Agaskar, Sriram Siva, William Pickering, Kyle O’Brien, Charles Kekeh, et al. 2025. DeepFleet: Multi-agent foundation models for mobile robots. arXiv preprint arXiv:2508.08574.

Tomoki Arita and Keisuke Okumura. 2025. Local guid ance for configuration-based multi-agent pathfinding. arXiv preprint arXiv:2510.19072.

Tomoki Arita and Keisuke Okumura. 2026. Lifelong La-CAM with local guidance for lifelong MAPF. arXiv preprint arXiv:2605.16855.

Jun Shern Chan, Neil Chowdhury, Oliver Jaffe, James Aung, Dane Sherburn, Evan Mays, Giulio Starace, Kevin Liu, Leon Maksin, Tejal Patwardhan, Lilian Weng, and Aleksander M ˛adry. 2024a. MLEbench: Evaluating machine learning agents on machine learning engineering. arXiv preprint arXiv:2410.07095.

Shao-Hung Chan, Zhe Chen, Teng Guo, Han Zhang, Yue Zhang, Daniel Harabor, Sven Koenig, Cathy Wu, and Jingjin Yu. 2024b. The league of robot runners competition: Goals, designs, and implementation. In ICAPS 2024 System Demonstrations Track.

Weizhe Chen, Sven Koenig, and Bistra Dilkina. 2024. Why solving multi-agent path finding with large language model has not succeeded yet. arXiv preprint arXiv:2401.03630.

Yukita Fujitani, Tomoki Yamauchi, Yuki Miyashita, and Toshiharu Sugawara. 2022. Deadlock-free method for multi-agent pickup and delivery problem using priority inheritance with temporary priority. arXiv preprint arXiv:2205.12504.

Wolfgang Hönig, Scott Kiesel, Andrew Tinka, Joseph W. Durham, and Nora Ayanian. 2018. Conflict-based search with optimal task assignment. In Proceedings ofthe International Conference on Autonomous Agents and Multiagent Systems (AAMAS).

Qian Huang, Jian Vora, Percy Liang, and Jure Leskovec. 2023. MLAgentBench: Evaluating language agents on machine learning experimentation. arXiv preprint arXiv:2310.03302.

Yuki Imajuku, Kohki Horie, Yoichi Iwata, Kensho Aoki, Naohiro Takahashi, and Takuya Akiba. 2025. ALE-Bench: A benchmark for long-horizon objectivedriven algorithm engineering. arXiv preprint arXiv:2506.09050.

He Jiang, Yutong Wang, Rishi Veerapaneni, Tanishq Duhan, Guillaume Sartoretti, and Jiaoyang Li. 2024. Deploying ten thousand robots: Scalable imitation learning for lifelong multi-agent path finding. arXiv preprint arXiv:2410.21415.

Zhengyao Jiang, Dominik Schmidt, Dhruv Srikanth, Dixing Xu, Ian Kaplan, Deniss Jacenko, and Yuxiang Wu. 2025. AIDE: AI-driven exploration in the space of code. arXiv preprint arXiv:2502.13138.

Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. 2024. SWE-bench: Can language models resolve real-world GitHub issues? In International Conference on Learning Representations (ICLR).

Ngai Meng Kou, Cheng Peng, Hang Ma, T. K. Satish Kumar, and Sven Koenig. 2019. Idle time optimization for target assignment and path finding in sortation centers. arXiv preprint arXiv:1912.00253.

Yu Kumagai and Keisuke Okumura. 2026. Alternating target-path planning for scalable multi-agent coordination. arXiv preprint arXiv:2605.07744.

Robert Tjarko Lange, Yuki Imajuku, and Edoardo Cetin. 2025. ShinkaEvolve: Towards open-ended and sample-efficient program evolution. arXiv preprint arXiv:2509.19349.

Jiaoyang Li, Andrew Tinka, Scott Kiesel, Joseph W. Durham, T. K. Satish Kumar, and Sven Koenig. 2021. Lifelong multi-agent path finding in large-scale warehouses. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI).

Fei Liu, Xialiang Tong, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu, and Qingfu Zhang. 2024. Evolution of heuristics: Towards efficient automatic algorithm design using large language model. arXiv preprint arXiv:2401.02051.

Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Foerster, Jeff Clune, and David Ha. 2024. The AI scientist: Towards fully automated open-ended scientific discovery. arXiv preprint arXiv:2408.06292.

Hang Ma and Sven Koenig. 2016. Optimal target assignment and path finding for teams of agents. arXiv preprint arXiv:1612.05693.

Hang Ma, Jiaoyang Li, T. K. Satish Kumar, and Sven Koenig. 2017. Lifelong multi-agent path finding for online pickup and delivery tasks. In Proceedings of the International Conference on Autonomous Agents and Multiagent Systems (AAMAS).

Alexander Novikov, Ngân Vu, Marvin Eisenberger, Em-˜ ilien Dupont, Po-Sen Huang, Adam Zsolt Wagner, Sergey Shirobokov, Borislav Kozlovskii, Francisco J. R. Ruiz, Abbas Mehrabian, et al. 2025. AlphaEvolve: A coding agent for scientific and algorithmic discovery. arXiv preprint arXiv:2506.13131.

Keisuke Okumura. 2023a. Engineering LaCAM<sup>∗</sup>: Towards real-time, large-scale, and near-optimal multiagent pathfinding. arXiv preprint arXiv:2308.04292.

Keisuke Okumura. 2023b. LaCAM: Search-based algorithm for quick multi-agent pathfinding. In Proceedings ofthe AAAI Conference on Artificial Intelligence (AAAI).

Keisuke Okumura, Manao Machida, Xavier Défago, and Yasumasa Tamura. 2022. Priority inheritance with backtracking for iterative multi-agent path finding. Artificial Intelligence, 310.

Ori Press, Brandon Amos, Haoyu Zhao, Yikai Wu, Samuel K. Ainsworth, Dominik Krupke, et al. 2025. AlgoTune: Can language models speed up generalpurpose numerical programs? arXiv preprint arXiv:2507.15887.

Ao Qu, Han Zheng, Zijian Zhou, Yihao Yan, Yihong Tang, Shao Yong Ong, Fenglu Hong, Kaichen Zhou, Chonghe Jiang, Minwei Kong, et al. 2026. CORAL: Towards autonomous multi-agent evolution for openended discovery. arXiv preprint arXiv:2604.01658.

Bernardino Romera-Paredes, Mohammadamin Barekatain, Alexander Novikov, Matej Balog, M. Pawan Kumar, Emilien Dupont, Francisco J. R. Ruiz, Jordan S. Ellenberg, Pengming Wang, Omar Fawzi, Pushmeet Kohli, and Alhussein Fawzi. 2024. Mathematical discoveries from program search with large language models. Nature, 625:468–475.

Guni Sharon, Roni Stern, Ariel Felner, and Nathan R. Sturtevant. 2015. Conflict-based search for optimal multi-agent pathfinding. Artificial Intelligence, 219:40–66.

Roni Stern, Nathan R. Sturtevant, Ariel Felner, Sven Koenig, Hang Ma, Thayne T. Walker, Jiaoyang Li, Dor Atzmon, Liron Cohen, T. K. Satish Kumar, Roman Barták, and Eli Boyarski. 2019. Multi-agent pathfinding: Definitions, variants, and benchmarks. In Proceedings of the International Symposium on Combinatorial Search (SoCS).

Yimin Tang, Sven Koenig, and Jiaoyang Li. 2024. ITA-ECBS: A bounded-suboptimal algorithm for combined target-assignment and path-finding problem. In Proceedings of the International Symposium on Combinatorial Search (SoCS), pages 134–142.

Yimin Tang, Zhongqiang Ren, Jiaoyang Li, and Katia Sycara. 2023. Solving multi-agent target assignment and path finding with a single constraint tree. In Proceedings of the International Symposium on Multi-Robot and Multi-Agent Systems (MRS), pages 8–14.

Yimin Tang, Xiao Xiong, Jingyi Xi, Jiaoyang Li, Erdem Bıyık, and Sven Koenig. 2025. RAILGUN: A unified convolutional policy for multi-agent path finding across different environments and tasks. arXiv preprint arXiv:2503.02992.

Yimin Tang, Han Zhang, Shao-Hung Chan, Junsoo Kim, Erdem Bıyık, Sven Koenig, and Jingkai Chen. 2026. ITA-LaCAM: A complete and scalable TAPF solver via assignment-aware configuration-space search. arXiv preprint arXiv:2609.14208.

David H. Wolpert and William G. Macready. 1997. No free lunch theorems for optimization. IEEE Transactions on Evolutionary Computation, 1(1):67–82.

Haoran Ye, Jiarui Wang, Zhiguang Cao, Federico Berto, Chuanbo Hua, Haeyeon Kim, Jinkyoo Park, and Guojie Song. 2024. ReEvo: Large language models as hyper-heuristics with reflective evolution. arXiv preprint arXiv:2402.01145.

Jingjin Yu and Steven M. LaValle. 2013. Structure and intractability of optimal multi-robot path planning on graphs. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI).

Hongzhi Zang, Yulun Zhang, He Jiang, Zhe Chen, Daniel Harabor, Peter J. Stuckey, and Jiaoyang Li. 2024. Online guidance graph optimization for lifelong multi-agent path finding. arXiv preprint arXiv:2411.16506.

Yulun Zhang, He Jiang, Varun Bhatt, Stefanos Nikolaidis, and Jiaoyang Li. 2024. Guidance graph optimization for lifelong multi-agent path finding. arXiv preprint arXiv:2402.01446.

Input: grid G, N agents, capacity K, durations $d _ { p } , d _ { d } ,$ horizon H   
1 x ← distinct random cells; release $2 N$ tasks; plan ← ∅   
2 for t = 0, . . . , H − 1 do   
3 if t=0, a service finished, or plan is empty then   
4 reset assigned-but-unpicked tasks to pending   
5 $\Omega _ { i }  \mathrm { p i c k u p s } ( \mathrm { i f } \ | P _ { i } | < K )$ , goals of carried $P _ { i } ,$ wait   
6 $c _ { i \omega } \gets \left( \mathrm { d i s t } ( x _ { i } , \omega ) + d _ { \omega } \right) \sigma _ { i \omega } + \mathrm { o f f s e t } _ { i \omega }$   
7 A ← Hungarian(c), one agent per pickup/drop cell   
8 $\mathrm { p l a n }  \mathrm { L a C A M - F O C A L } ( x , A _ { 0 } )$ with PIBT successors and   
incremental assignment repair, until a service ends   
9 x ← next configuration of plan (no vertex/swap conflicts)   
10 update dwell counters; pick up or deliver when counter $\geq d$   
11 release tasks to keep 2N unfinished   
12 return #completed/H

Figure 6: Baseline lifelong ITA-LaCAM loop.  
![](images/f54ffc37850d982d2bde597cfc7baa43bf1a69cbb3a6c29aa5827d414ff16ad4.jpg)  
Figure 7: The alternative layout used in the transfer study: 37×39 cells, 835 free cells, an open top corridor, no dividing wall, and 13 station lanes; colors as in Figure 2b.

## A Baseline Planner Pseudo-Code

## B Held-Out Layout

## C Task Contract Excerpt

The contract was given to the agents in Chinese; we translate its key rules. “You may generally change planner, assignment, or goal-candidate behavior. You may not change thefixed official target, hardcode seed/task/map shortcuts,fabricate CSV, trace, or schedule outputs, or optimize only the output format.” “A mean change smaller than 0.03 on public seeds is not a signal; prefer trace-derived behavior signals.” “Once a behavior-changing candidate is finalized, builds, and produces valid output, you must commit it and run an official evaluation, whether your self-test shows improvement, no change, or regression.” “Iteration time is also a resource . . . only accept a clearly higher runtime cost when the throughput evidence is strong (about 0.1 or more).”

## D Per-Campaign Frontier Details

Table 3 summarizes the frontier of each campaign: the number of new-best steps, how many exceed the noise floor, the number of single-constant retunes, the generality grades, and the final patch size. The full per-step table (campaign, evaluation index, commit, hidden score, increment, lines added and deleted, mechanism group, mechanism, and the agent’s evaluation message) is the ancillary file frontier\_steps.csv.

<table><tr><td>Campaign</td><td>Steps</td><td>&gt;.03</td><td>Retunes</td><td>G0/G1/G2</td><td>Patch</td></tr><tr><td>Opus / paper</td><td>24</td><td>6</td><td>7</td><td>5/15/4</td><td>+1454/-10</td></tr><tr><td>Opus / open</td><td>20</td><td>8</td><td>10</td><td>3/17/0</td><td>+635/-22</td></tr><tr><td>Fable / paper</td><td>21</td><td>5</td><td>4</td><td>3/16/2</td><td>+824/-20</td></tr><tr><td>Fable / open+web</td><td>12</td><td>2</td><td>6</td><td>1/11/0</td><td>+558/-101</td></tr><tr><td>GPT-6 Luna / paper</td><td>14</td><td>4</td><td>0</td><td>12/2/0</td><td>+1264/-28</td></tr><tr><td>GPT-6.1 Sol / paper</td><td>11</td><td>2</td><td>0</td><td>11/0/0</td><td>+2372/-57</td></tr></table>

Table 3: Frontier statistics per campaign. Retunes: steps that only change constants. Patch: lines of algorithm code added/deleted from the baseline to the best commit (tests, tools, and build files excluded).

## E Example of an Improving Step

Figure 8 shows the C++ code added by the second evaluation of the Opus/open campaign (commit 3400f25f): corridor detection and the corridorexit priority. The step adds 45 lines of algorithm code, removes 2, and raised the hidden score from 1.238 to 1.380. A reviewer can check its two claims, that corridors are connected components of vertices with degree at most 2 and that only unsatisfied agents whose goal lies outside their corridor are boosted, directly from the code.

## F Photo Credits

Figure 1: AutoStore robots, photo by Euku, Wikimedia Commons, CC BY-SA 4.0; Amazon Robotics, Symbotic, Hai Robotics, Geek+, and Locus Robotics, product images from the companies’ websites. Images were cropped.

```cpp
// (in the TAPFPlanner constructor)
auto next_component = 0;
for (auto v : ins->G.V) {
if (v->neighbor.size() > 2 || corridor_component[v->id] >= 0) continue;
auto stack = std::vector<Vertex*>{v};
corridor_component[v->id] = next_component;
while (!stack.empty()) {
auto u = stack.back(); stack.pop_back();
for (auto w : u->neighbor) {
if (w->neighbor.size() > 2 || corridor_component[w->id] >= 0) continue;
corridor_component[w->id] = next_component; stack.push_back(w);
}
}
++next_component;
}
// (at the LaCAM root)
// An agent inside a single-lane corridor whose goal lies outside it is
// leaving; give it precedence so agents entering the corridor do not push
// it deeper.
constexpr float kCorridorExitPriority = 10.0f;
auto boosted = false;
for (auto i = 0; i < N; ++i) {
if (S_init->satisfied[i] || S_init->service_assignment[i] >= 0) continue;
const auto comp = corridor_component[S_init->C[i]->id];
if (comp < 0) continue;
const auto goal = assigned_goal(S_init->assignment, i);
if (goal == nullptr || corridor_component[goal->id] == comp) continue;
S_init->priorities[i] += kCorridorExitPriority; boosted = true;
}
if (boosted) {
std::sort(S_init->order.begin(), S_init->order.end(), [&](int a, int b) {
return S_init->priorities[a] > S_init->priorities[b]; });
}
```  
Figure 8: An agent-written improving step (corridor-exit priority, +0.142 hidden throughput), from the diff of commit 3400f25f. Code and comments are verbatim; we merged some short lines, dropped enclosing braces, and added the two location notes in parentheses.