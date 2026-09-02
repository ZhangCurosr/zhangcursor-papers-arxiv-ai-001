# Agentic Empirical Asset Pricing: Methodological Foundations<sup>∗</sup>

Yingjian Pan Advanced Financial Technologies Laboratory Management Science & Engineering Stanford University pyj@stanford.edu

Xiaowei Ding School of Information Management Nanjing University dingxiaowei@nju.edu.cn

Kay Giesecke Artificial Intelligence & Quantitative Finance Hasso Plattner Institute kay.giesecke@hpi.de

This draft: July, 2026

## Abstract

Recent advances in LLM agents enable a new paradigm for asset pricing, which we call Agentic Empirical Asset Pricing (AEAP): systems that autonomously conduct the scientific discovery process itself. We define AEAP and identify its core building blocks. Existing evaluation practices backtest only the outputs—factors or trades—not the autonomous discovery system that produced them. We focus on factor discovery, contributing a reference architecture, a rigorous evaluation standard for discovered factors, and a method for out-of-sample backtesting the discovery system. As a concrete instance of that architecture, we evaluate SEADS against five re-implemented baselines on two US equity panels using this standard: no single metric ranks the systems consistently, motivating evaluation on multiple axes at once. A separate rolling re-execution then asks the complementary question of whether the discovery process itself, not one static output, is reliable. We also report negative findings and limitations that surface further evaluation pitfalls for future AEAP systems.

## 1 Introduction

Empirical asset-pricing research has traditionally relied on humans to design and execute the discovery process, whether through theory-guided construction or data-driven machine learning [Gu et al., 2020, Kozak et al., 2020]. LLM agents that autonomously generate hypotheses, synthesize them into executable code, and statistically validate the outcome change this division of labor—we call the resulting paradigm Agentic Empirical Asset Pricing (AEAP) (§2).

The paradigm is not hypothetical: in the last two years, at least a dozen systems have implemented a hypothesize–code–backtest–refine loop for formulaic factor mining [Tang et al., 2025, Shi et al.,

2025, Li et al., 2025c, Han et al., 2026, Huang and Fan, 2026, Shi et al., 2026b], multi-agent teams reach trading decisions through debate or structured pipelines [Xiao et al., 2025, Miyazaki et al., 2026, Chen et al., 2026b], and LLM agents are wired into structural pricing pipelines built around an explanatory asset-pricing model [Koijen and Levy, 2026, Cheng and Chin, 2025]. The resemblance across these is real but the vocabulary is not shared: each names its own architecture, states its own notion of validity, and is evaluated against its own idiosyncratic metric.

That last point is what this paper is built around: a shared vocabulary’s absence is not just inconvenient, it hides a specific failure mode. Every AEAP system we know of is evaluated the way a classical, human-discovered factor is: hold out a test window, backtest the output—the factor or trade—and report its performance. This is necessary but not sufficient. An autonomous discovery system is itself a research process with its own way of getting lucky: a gate admitting only a small fraction of one run’s candidates can be well-calibrated, can be leaking information through an unnoticed bug, or can simply have drawn a favorable in-sample window—nothing about that run’s own reported Sharpe distinguishes the three. None of this is visible from backtesting the output alone— it requires backtesting the system that produced it, re-run under the information constraints a real deployment would face. To our knowledge, no existing AEAP evaluation does this. The problem is not specific to finance: any autonomous discovery agent that revises its own hypotheses over time faces the same question, and asset pricing simply gives it an unusually clean testbed—clear temporal ordering, measurable future outcomes, and severe adaptive-overfitting risk.

That gap is a foundations problem before it is a better-system problem. Without a shared definition of what counts as an AEAP system, an account of the objectives it pursues, and a standard for scoring the discovery process itself rather than only its output, results across systems cannot be compared, replicated, or built upon—each new entrant restarts the vocabulary from scratch. This paper supplies those foundations: broadly, for the paradigm as a whole (§2); in full technical detail, for factor discovery—EAP’s most central objective, hence most agentic work’s focus (§3).

## Contributions. We make three.

(C1) A definition of AEAP and its building blocks. We define AEAP precisely enough to distinguish it from both classical ML asset pricing and LLM copilots that leave a human in the loop, and identify five building blocks common to any such system (§2).

(C2) For factor discovery: an evaluation methodology. We factor out of five re-implemented systems (§2.2) a reference architecture, then consolidate a multi-part evaluation standard around three jointly-necessary properties of a discovered pool—productivity, performance, and novelty, no one meaningful without the other two—for scoring what a discovery system reports. We further propose re-executing the entire discovery loop at a sequence of historical decision dates: measuring whether a discovery process adapts over time rather than getting lucky once, a distinction no static backtest can draw (§3).

(C3) SEADS: a concrete agentic factor-discovery system. We build SEADS (Stanford Engine for Agentic Discovery of Signals, pronounced “seeds”—an echo of the Planner’s own seed records, §3.3.1) as an instance of (C2)’s reference architecture and apply (C2)’s standard to it alongside five re-implemented baselines [Tang et al., 2025, Shi et al., 2025, Li et al., 2025c, Han et al., 2026, Huang and Fan, 2026] on two comprehensive US equity panels: the US subset of JKP’s global-characteristics panel [Jensen et al., 2023], and a CRSP/Compustat panel of primitive fea tures novelty-checked against the 45-characteristic reference panel of Bryzgalova et al. [2025]. No single metric ranks all six systems consistently, which is itself the point of (C2)’s standard (§5.1). We also surface negative findings and limitations that are pitfalls for any discovery system, not only ours (§6).

## 2 Agentic Empirical Asset Pricing

## 2.1 Definition and Core Building Blocks

Empirical Asset Pricing (EAP) is the branch of financial economics seeking scientific understanding of asset prices through data. Its objectives recur across the field, e.g. testing asset-pricing theories (CAPM, ICAPM, consumption-based models), discovering and evaluating risk factors [Jensen et al., 2023], explaining the cross-section of expected returns, forecasting returns, studying anomalies, evaluating portfolio strategies, and estimating risk premia. These objectives were traditionally pursued through parametric econometrics; over the last decade, a second wave used data-driven machine learning [Gu et al., 2020, Kozak et al., 2020]. We propose a third wave: Agentic Empirical Asset Pricing (AEAP), in which autonomous LLM-based agents pursue these objectives with minimal human intervention—not displacing econometrics and ML but calling on them within an agent’s own loop. We propose not a new objective, but a new researcher.

Formally, we define an AEAP system as one in which LLM-based agents autonomously execute at least one full cycle of the scientific-discovery process—hypothesis, formalization, evaluation—over asset-pricing data, without a human performing any of those three steps. This has two consequences for what counts. A machine-learning return predictor [Gu et al., 2020, Kozak et al., 2020] is not an AEAP system: the researcher, not the model, designs the hypothesis space and decides what counts as validation. A human proposing an idea for an LLM copilot to formalize and backtest [Wang et al., 2023] is likewise not fully autonomous. What qualifies is a system whose loop—propose, formalize, check—closes without a human decision, even if a human designed it and reviews its output; a single cycle already qualifies, and persisting state across cycles (B5) is common, not required.

Reading across the systems we survey (§2.2, Appendix I), five building blocks recur regardless of output: (B1) hypothesis generation—an LLM proposes a candidate mechanism (a factor’s rationale, a trade thesis, an anomaly’s explanatory channel), not a bare parameter draw; (B2) formalization into a checkable artifact—code for a factor, a discrete action for a trade, a fitted model for a structural claim; (B3) execution or estimation against historical or live data—simulated for a factor or trade, statistically fit for a structural model; (B4) an evaluation gate, statistical or LLMjudged, deciding what survives or, for debate-structured systems, what the group commits to; and (B5) memory and iteration: persistent state—including the discovered set and its reference pool, not only model weights—that lets the system improve across cycles instead of restarting cold. B1– B2 generate; B4 judges; keeping them separate makes B4’s verdict evidence, not the generator’s own opinion of itself, whether B4 is a statistical test or an independent LLM critic. These map onto standard LLM-agent components from the broader agentic-AI literature [Wang et al., 2024a]: B1–B2 to planning and action-formulation, B3 to tool use, B4 to a critic model, B5 to memory—grounding AEAP in agentic AI generally, not only in finance. B3 and B4 are stated here in their most literal, factor-discovery-native form.

## 2.2 Examples

AEAP is not specific to one EAP objective: discovering factors is scored as a cross-sectional signal, evaluating a trading policy as a trajectory of decisions, testing a structural model by estimation. The systems below span these objectives; not all are AEAP systems.

Factor discovery. AlphaAgent [Tang et al., 2025] gates on AST-originality and hypothesis-factor consistency; RD-Agent(Q) [Li et al., 2025c] co-optimizes factors and models via bandit-scheduled feedback and a backtest-metric admission unit; QuantaAlpha [Han et al., 2026] evolves mining trajectories under LLM-judged semantic consistency plus algorithmic complexity/redundancy constraints; Beyond Prompting [Huang and Fan, 2026] is, on its own account, this group’s rigor fron tier, importing the Harvey et al. [2016] t > 3.0 hurdle; and AlphaForge [Shi et al., 2025]—non-LLM, and not itself an AEAP system by §2.1’s definition, but included as the field’s most-cited non-agentic combination baseline—mines via a generative-predictive network and combines survivors dynamically. None combines more than two independent statistical criteria into one decision, and none re-executes its own discovery loop at more than one historical decision date (Table 3, Appendix B). AlphaForge’s mining stage instead searches combinatorially, filtering ex post: more candidates only raise the bar a genuine discovery must clear.

Other systems. Xiao et al. [2025], Miyazaki et al. [2026], Chen et al. [2026b] reach trading decisions through multi-agent debate or structured pipelines, and Koijen and Levy [2026], Cheng and Chin [2025] estimate an explanatory asset-pricing model end-to-end—genuine AEAP systems (§2.1), pursuing a different objective, not directly comparable to ours. Further removed, and not meeting the AEAP standard themselves: non-agentic predecessors [Yu et al., 2023, Cui et al., 2021] lack the agentic loop, and general-purpose “AI-scientist” engines [Lu et al., 2024, Yamada et al., 2025, Novikov et al., 2025] lack the finance-specific objective, though both are architecturally foundational; Appendix I extends this survey substantially. We further situate this work within the ML asset-pricing lineage [Gu et al., 2020, Kozak et al., 2020] and the replication debate [Jensen et al., 2023].

## 2.3 Evaluation Principles

Having surveyed these systems, we now ask how any of them—or SEADS—should be evaluated. Classical backtesting reduces to one discipline: use only information available at each decision date. When the researcher is itself a model, this discipline binds two nested loops, not one. An inner loop is the discovery cycle itself, hypothesis through evaluation; an outer loop re-runs it across historical decision dates rather than scoring output on just one split. A methodologically sound AEAP evaluation needs three principles: (P1) point-in-time data—no feature observable only after decision date t enters the inner loop; (P2) point-in-time process—the outer loop actually exists: the inner loop is re-executable and scoreable at each historical date, not just run once; (P3) outof-sample-only results—headline numbers come from a split neither loop ever touched. Classical backtests satisfy P1 and P3 by construction, and never needed an outer loop either: with a human, not an autonomous inner loop, doing the discovering, there was no comparably systematic process to rerun. P2 is the property no existing AEAP system we survey satisfies (Table 3, Appendix B)—an AEAP system’s own reliability cannot be assessed without it (§3.5). We omit a fourth, backbonevintage principle; §6 argues its residual risk is task-dependent—soft for ours.

We focus the rest of this paper on factor discovery, the objective with the most existing systems and the best-defined statistical target for §3’s backtesting-the-system question.

## 3 Factor Discovery

§2.3’s P1–P3 principles are not specific to factor discovery; they generalize to any AEAP objective. We narrow to factor discovery here because it is the objective most tractable to formalize fully, end to end.

## 3.1 Problem Formulation

We work on a panel of asset-periods (i, t) with features $x _ { i , t } \in \mathbb { R } ^ { p }$ observable at the close of period t and next-period excess return ${ r } _ { i , t }$ over (t, t+1]; our own panels (§4) take the period to be a month, but nothing in the formulation requires it. A candidate factor is a scoring function $f _ { \theta } ~ : ~ x _ { i , t } \mapsto s _ { i , t }$ , typically ending in a within-period percentile rank, where θ denotes a symbolic (Python) recipe rather than a learned parameter vector; its period-t information coefficient is $\rho _ { t } = \mathrm { S p e a r m a n } ( s . , . , r . , t )$ across stocks—the quantity §3.3.3’s Rank-IC check is built on.

The discovery target is a factor set $\mathcal { G } = \{ f _ { \theta _ { 1 } } , \ldots , f _ { \theta _ { q } } \}$ that is simultaneously out-of-sample valid and non-redundant; no single metric captures both (§3.4). Non-redundancy is with respect to an explicit, disclosed reference set Z, chosen per panel rather than drawn from a single run’s own candidates, which would be circular (§4 gives each panel’s specific choice); $f _ { \theta }$ is admissible only $\mathrm { i f } \ \lvert \mathrm { c o r r } ( s . , t , z . , t ) \rvert$ | stays below a threshold for every $z \in \mathcal { Z } \left( \ S 3 . 3 . 3 \right)$ . G and $\mathcal { Z }$ are both fixed here at one decision date; §3.5 asks what changes once both are allowed to evolve across a rolling history instead.

## 3.2 A Factor-Discovery Reference Architecture

Table 3’s (Appendix B) five baselines share no common vocabulary. Reading across all five reimplementations (four agentic, one not; §2.2), each instantiates §2.1’s building blocks as a generator (B1–B2), a sandboxed executor against point-in-time data (B3), a validation gate (B4), and, in the more developed systems, a persistence mechanism accumulating admitted candidates into a library (B5). A well-designed B3 is ideally schema-agnostic, so the same executor and gate generalize across feature panels without redesign—the property §3.3.2 details for SEADS’s own implementation. This is also why we exclude CogAlpha [Liu et al., 2026b] from direct comparison: its executor is built around daily OHLCV series, not a schema-agnostic characteristic panel, so porting it would mean reimplementing its core mechanism, not just retargeting its gate. Figure 1 diagrams SEADS’s own architecture, one concrete instantiation of this pattern; no specific module is mandated—any can be swapped for another implementation while the system remains an instance of it—but Table 3 shows every baseline instantiates only a strict subset (§2.2).

![](images/6211909c0697a355e3d711caae34f21ce92ed214d05109065675d88c3f3672f4.jpg)  
Figure 1: Diagram for SEADS’s architecture. One Planner call plans an entire round (one seed per slot); each seed is then expanded by its own Proposer instance (stacked boxes). Near-passes are mutated by self-evolution and re-run through the identical sequence. The dashed navy loop is the discovery-round unit every system in Table 3 has; the rolling-context banner above it is SEADSspecific (none of the five baselines re-executes this loop, Table 4), so it is a light annotation, not a nested box. Figure 5 (Appendix F) details the repropose/refit mechanics; Algorithm 1 (Appendix C) gives it as pseudocode. The gate box lists its nine checks; §3.3.3 gives the full order and thresholds.

## 3.3 SEADS: An Instantiation

SEADS instantiates §3.2’s factor-discovery architecture concretely (Figure 1); it is the system we build and evaluate, and Algorithm 1 (Appendix C) states its full discovery loop as pseudocode. Data construction for both panels it runs on is in §4 and Appendix A.

## 3.3.1 Generation: Planner and Proposer

Each round, a single Planner call emits one seed per candidate slot—a structured record (exploreor-exploit type, theme, variables, mechanism, complexity tier) rather than a finished formula. Seeds split between exploit (a bounded variation of a prior successful recipe) and explore (a hypothesis from a different signal family, default 70% of slots); two deterministic backstops correct observed Planner failure modes (Appendix E). Planner and Proposer are deliberately separate roles, not one call issuing finished proposals directly: the Planner controls breadth—how slots split across themes and complexity tiers—while each seed’s own Proposer instance controls depth. The Proposer draws on a narrower context scoped to prior attempts sharing at least one underlying variable, not the Planner’s full history (Appendix E), independently expanding its seed into a complete specification and formula, which then proceeds to sanity checks (§3.3.2).

## 3.3.2 Sanity Checks

A deterministic structural pre-check first rejects malformed specs before any code executes (Appendix E). Point-in-time data (P1, §2.3) is enforced next: the Implementation Agent runs a surviving spec’s formula in a sandboxed namespace restricted to exactly the columns it statically reads, checks it against a look-ahead firewall, and attempts up to two LLM-assisted repairs on failure. The firewall is a deterministic AST walk enforcing seven rule classes plus a taxonomy whitelist (Appendix D); nothing in the sandbox, prompts, or gate hardcodes column names, so the identical pipeline runs unmodified over JKP’s ∼400 characteristics or CRSP/Compustat’s 87 variables alike (§4). An Alignment Checker—an LLM call—separately verifies the formula implements what its rationale describes via a required line-by-line trace, rejecting only at high confidence.

## 3.3.3 Validation Gate

Once a candidate clears the sanity checks above, it faces the statistical validation gate, admission requiring all checks to pass, in order. Two structural checks come first: a duplicate fingerprint, and a self-collapse check rejecting a candidate whose final formula algebraically collapses to a plain rank of one of its own declared inputs. Three core statistical checks follow: coverage, Rank-IC t-stat, and out-of-decade regime-consistency. A final robustness trio closes the gate—novelty against the refer ence set Z (§3.1), partial-IC against the top-K most-correlated controls, and Fama–MacBeth [Fama and MacBeth, 1973] t against the same controls—with sign agreement required among Rank-IC, partial-IC, and Fama–MacBeth beta (full order and thresholds in Appendix D). The out-of-decade split deliberately tests cross-regime generalization, not same-period consistency. Each threshold is set from discipline convention, an established statistical procedure (the Rank-IC bar’s Bonferroni correction), or a simple ex-ante default, fixed before any experimental run and never adjusted afterward. No baseline combines this many independent checks into one admission decision (§2.2).

## 3.3.4 Self-Evolution

Self-evolution mutates a near-pass—one that cleared a minimum progress floor or failed at a specific gate stage—through the identical gate again, for a bounded number of rounds, stopping early once a mutation clears the Rank-IC bar or shows no further improvement across attempts; candidates falling far short of any bar are discarded rather than mutated. Feedback names the exact failure stage rather than a scalar fitness, and a progress-weighted score rewards reaching a later stage over clearing an earlier one by a larger margin (Appendix E).

## 3.3.5 Persistent State and Combination

Three memory tiers persist across rounds: baseline characteristic statistics the Planner uses for new seeds; an append-only log of every candidate’s outcome—admission, failure mode, or mutation— consulted to avoid repeating dead ends; and capped natural-language lessons from a reflector, feeding seeds. Admitted factors accumulate in a persistent library; a combinator fits a Ridge regression (α = 1) on the library’s rank-normalized values to rank stocks monthly into a long-short portfolio— the deployed portfolio construction Table 1’s performance column scores. This is simpler by design than alternatives like XGBoost or LightGBM: an ex-ante, reproducibility-motivated default, fixed pre-run (Appendix E) and distinct from §5.2’s joint regression, a separate external incremental-value check.

## 3.4 Evaluating Discovered Factors

However a system’s output was produced, how should it be scored? A pool of discovered factors earns its place in a fund only if it clears three bars jointly: productivity, performance, and novelty—no two substitute for the third. Performance without novelty rewards redundant repackaging of what a fund already trades (§5.1’s account of Beyond Prompting’s larger, markedly less-novel admitted set). Performance without a productivity expectation rewards finding one good signal by any means—a bar so low that AlphaForge’s non-agentic search, with only 5 and 1 total admissions on our two panels (Table 1), already clears it, and an agentic system that cannot out-produce that has no case for its own expense. Novelty without performance rewards noise, novelty’s cheapest source. §3.1’s two discovery desiderata defined what makes any one factor good: out-of-sample validity and non-redundancy. Scoring a pool carries both forward—validity becomes performance, non-redundancy becomes novelty—and adds productivity as the genuinely new question a set, not a single factor, raises. Each of the three admits more than one reasonable measurement—performance alone could be scored by mean or median per-factor Sharpe, by information coefficient, or by postadmission survival, to name a few—so the metrics below are our chosen operationalization, not the only defensible one.

We score what a factor-discovery system reports—SEADS and the five baselines alike—on four metrics (Table 1): productivity, total admissions out of the shared 300-candidate budget—a function of both native gate strictness and discovery quality, which is why we never score it alone; performance, mean per-factor out-of-sample (OOS) Sharpe across the admitted set; and novelty, each admitted factor’s maximum correlation against the union of the system’s own other admits and the panel’s reference set, averaged across the admitted set rather than worst-cased so one highly-correlated factor cannot sink an otherwise-diversified system’s score—one metric per jointly-necessary property. A fourth check tests performance and novelty jointly: a spanning test, a Ridge regression of the system’s admitted factors against the reference set: the two resulting OOS long-short return series are regressed against each other, and we report the t-statistic on the alpha. Larger positive values are read as suggestive, not confirmatory, evidence of value added beyond the reference set. We score SEADS and all five baselines identically.

## 3.5 Backtesting the Discovery System

§3.4 scores whatever a system reports, however its output was produced—but productivity, performance, and novelty are all properties of one decision date’s output, and factor discovery is not, in practice, a one-off exercise: markets shift, and a deployed system must keep finding and refreshing signals as they do. A fourth dimension—call it adaptivity—asks whether the discovery process itself both holds up across changing conditions and improves by carrying forward what it has already learned, neither of which a static snapshot, however many axes it scores, can show by construction. SEADS’s own gate already tests a static proxy for the first half—requiring a candidate’s Rank-IC to hold out-of-decade, not only within-period (§3.3.3)—but one regime-robust admission is still one decision at one point in time, saying nothing about whether the system uses what it already found to discover better next time. Every row of Table 3 (Appendix B), ours included up to this point, backtests only that output: discover a factor set once, hold out a test window, report what that one set did. This licenses a claim about the factor set, not the process that produced it: a process can get lucky once on a fixed split, invisibly to the output alone. It is also a different evaluation target than §3.4’s: there, each admitted factor is scored on its own information coefficient and Sharpe; here, we score the pool—the combinator’s portfolio over the full admitted library, what a deployed system actually trades, not any one signal alone.

We propose backtesting the discovery system itself—supplying the point-in-time process principle (P2, §2.3) no system we survey satisfies: re-executing the entire loop of Figure 1 at a sequence of historical decision dates, each conditioning only on a sliding in-sample window ending at that date. Concretely, this means running a rolling protocol: starting from an identical library and gate, the system additionally reproposes—new seeds, new proposals, a re-run gate, growing the library—every k months, with the combinator refit on a finer, independent cadence every j months, j dividing k so a newly admitted factor is reflected in the portfolio by the next refit rather than sitting un-weighted until the next repropose. A static backtest—discover once, thereafter only re-fit the combinator on the frozen library as new months arrive—is what every system in Table 3 runs by default, and all a system with no re-executable loop can ever produce: sufficient for backtesting an output, not for backtesting the system that produced it.

Re-executing the loop is necessary but not sufficient: a rolling arm that discarded all state and reproposed cold at every event would just be several independent static runs stitched together. What makes rolling re-discovery a test of the system’s adaptivity, not resampling luck, is that SEADS carries state across repropose events—the growing library becomes tomorrow’s building blocks (Figure 1), the novelty reference set updates to include newly admitted factors, and lessons persist via the reflector (§3.3.5). A system with no carry-forward can still be rolled, but only memorylessly, which tests a strictly weaker claim.

## 4 Experimental Setup

Two comprehensive US equity panels. We evaluate on two panels posing different discovery questions. Panel A (JKP) is a large, already heavily-mined characteristics panel—the US subset of JKP’s globally-constructed database [Jensen et al., 2023] (ex-microcap)—checking novelty against that same curated zoo. Panel B (CRSP/Compustat base features) instead gives the system primitive, uncurated accounting and market variables, checking novelty against an external published reference [Bryzgalova et al., 2025]. Both use an in-sample window of 2010–2019; Panel A’s out-ofsample window runs to December 2025 (n ≈ 71 months) and Panel B’s to November 2024 (n ≈ 59 months), reflecting each source’s data availability (§6). Appendix A gives full construction details.

<table><tr><td>System</td><td>Productivity Performance (A/B)</td><td>(A/B)</td><td>Novelty (A/B)</td><td>Spanning (A/B)</td></tr><tr><td>AlphaForge</td><td>5.0/1.8</td><td>0.34/-0.08</td><td>0.96/0.74</td><td>-0.13/0.56</td></tr><tr><td>RD-Agent(Q)</td><td>39.0/47.0</td><td>0.19/-0.01</td><td>0.79/0.64</td><td>1.27/0.88</td></tr><tr><td>AlphaAgent</td><td>31.0/32.6</td><td>0.12/0.09</td><td>0.82/0.67</td><td>-0.76/0.34</td></tr><tr><td>QuantaAlpha</td><td>60.8/34.6</td><td>-0.01/0.04</td><td>0.65/0.61</td><td>0.37/1.46</td></tr><tr><td>Beyond Prompting</td><td>96.4/85.0</td><td>0.41/0.11</td><td>0.90/0.81</td><td>2.04/1.01</td></tr><tr><td>SEADS (ours)</td><td>14.0/13.8</td><td>0.25/0.16</td><td>0.65/0.46</td><td>1.21/1.28</td></tr></table>

Table 1: Matched-budget, out-of-sample-only comparison (P3, §2.3); identical scoring on both panels (§4); A = Panel A (JKP), B = Panel B (CRSP/Compustat); mean across 5 replicates; per-entry cross-replicate SEs in Table 11 (Appendix K). Productivity, Performance, Novelty, and Spanning Test are defined in §3.4; higher is better for all but Novelty, where lower means more novel. Bold marks each column’s best value per panel.

Five re-implemented baselines. We re-implement AlphaAgent, AlphaForge, RD-Agent(Q), QuantaAlpha, and Beyond Prompting [Tang et al., 2025, Shi et al., 2025, Li et al., 2025c, Han et al., 2026, Huang and Fan, 2026] rather than cite each system’s own number on its native market, which would not isolate method from data. Porting fidelity is not uniform, so results below are our reimplementations, not the named systems’ own published numbers: AlphaForge and QuantaAlpha are verified against released source; RD-Agent(Q) against the official package; AlphaAgent and Beyond Prompting are close reconstructions from paper text, the latter releasing no code. Every system proposes on the same in-sample window at a matched 300-candidate budget, scored out-of-sample by identical code, using the same current LLM backbone (§6); Appendix G details each port.

Validation gate and backtesting-the-system protocol. SEADS uses one locked internal gate across both panels—coverage, Rank-IC, out-of-decade regime, novelty, partial-IC, and Fama– MacBeth thresholds, in §3.3.3’s order and at the values given in Appendix K. Each threshold is either a fixed, discipline-conventional constant (e.g. novelty’s 0.7 cutoff) or governed by a fixed rule as a function of cumulative trial count (the Bonferroni-dynamic Rank-IC bar, Appendix K)—the rule itself set ex-ante and never adjusted after observing a run’s OOS performance. Each baseline retains its own native admission logic (Appendix G); every system’s factors are then scored identically by §3.4’s external standard, not SEADS’s gate. For §3.5’s protocol, we evaluate an annual re-discovery cadence with a 4-month combinator refit, against an identical-library static baseline; this cadence and §5.3’s ablation conditions were likewise fixed pre-run from in-sample evidence alone.

## 5 Results

## 5.1 Main Out-of-Sample Comparison

Table 1 reports productivity, performance, novelty, and a spanning test for all six systems, each entry a mean over 5 replicates. Three rankings disagree, and the winner shifts by panel: by performance, Beyond Prompting leads Panel A (0.41, AlphaForge second at 0.34, SEADS third at 0.25) while SEADS leads Panel B (0.16 vs. 0.11, AlphaForge last at −0.08); by novelty, SEADS and QuantaAlpha are tied atop Panel A (0.65 each, AlphaForge worst of all six at 0.96), while SEADS leads Panel B outright (0.46 vs. QuantaAlpha’s next-best 0.61); by productivity, Beyond Prompting leads both panels (96.4/85.0 vs. SEADS’s 14.0/13.8). AlphaForge’s own split—strong performance, least novel of any system—repeats §3.4’s productivity argument on the novelty axis: performance is cheap without novelty. Only on Panel B does one system, SEADS, rank top two on both novelty and performance; on Panel A none does, since the two performance leaders (Beyond Prompting, AlphaForge) are also the two least novel. Beyond Prompting’s productivity and performance both partly reflect a looser gate and predominantly redundant formulas (Appendix H), not just superior discovery. No single static ranking is decisive: §3.5 asks a different question of the same systems, not a fourth static one.

Rolling OOS equity curve: incremental value over raw-400 (admitted factors + raw 400, jointly regressed via LR)  
![](images/f0777684a29fc7c2880827ce039e1ec31207acd28e3467ed3fca1f0e9e0a8c67.jpg)  
Figure 2: §3.5’s protocol applied on Panel A (JKP): cumulative OOS growth of \$1 under SEADS and the four other agentic factor-discovery baselines. Each system’s admitted factors are jointly regressed with the full raw ∼400-characteristic panel, not combined alone, so an admitted-only Sharpe edge from a large, redundant set (Appendix H) cannot be mistaken for value the raw panel already supplies; Raw 400 only is the resulting no-discovery floor: the same raw panel, refit identically, with no admitted factors, fully deterministic (no LLM step, no replicate variance). SEADS (static) isolates repropose’s contribution: its own starting library and refit schedule, never re-discovering. The line shown for each of the other six is one of 3 replicates; the legend’s Sharpe is instead the mean ± SE across all 3, each computed independently before averaging, so a line’s equity path need not track its own legend ranking—Sharpe rewards steadier growth, not a higher endpoint. AlphaForge is omitted (too few admits).

Panel B’s looser reference set makes every system more novel (Table 1), while its weaker raw features lower mean performance—least for SEADS, whose admits are mostly multiplicative interactions rather than the additive spreads RD-Agent(Q) and Beyond Prompting favor (Appendix H).

## 5.2 A Rolling Comparison

Figure 2’s clearest pattern is that all five discovery systems clear the no-discovery floor by a wide margin (mean Sharpe 0.82–0.96 vs. Raw 400 only’s 0.62): whatever each system’s admitted factors contribute on top of the raw panel, it substantially exceeds what a plain linear model already extracts from that panel alone. Among the five, SEADS nominally leads (0.96±0.04) but is not reliably distinguishable from Beyond Prompting (0.91±0.02) or QuantaAlpha (0.90±0.02) given cross-replicate noise; it is more clearly separated from RD-Agent(Q) and AlphaAgent (0.82±0.03 each). This compares each system’s best rolling configuration, not an isolated persistence effect— SEADS’s carry-forward arm and each baseline’s own rolling arm still differ in algorithm, gate, and reconstruction fidelity, not persistence alone. SEADS’s static arm—identical starting library, never re-discovering—reaches only 0.71±0.05, closer to the no-discovery floor than to any rolling system: most of SEADS’s rolling advantage is repropose itself, not the initial library.

<table><tr><td>Configuration</td><td></td><td>Productivity Performance</td><td>Novelty</td></tr><tr><td>Full pipeline</td><td> $1 4 . 0 \pm 1 . 4$ </td><td> $0 . 2 5 4 \pm 0 . 0 3$ </td><td> $0 . 6 5 3 \pm 0 . 0 1$ </td></tr><tr><td>Planner off</td><td> $3 7 . 2 \pm 2 . 5$ </td><td> $0 . 1 0 1 \pm 0 . 0 4$ </td><td> $0 . 6 7 6 \pm 0 . 0 1$ </td></tr><tr><td>Align Check off</td><td> $1 7 . 6 \pm 1 . 2$ </td><td> $0 . 2 1 5 \pm 0 . 0 3$ </td><td> $0 . 6 6 2 \pm 0 . 0 1$ </td></tr><tr><td>Mem (Lessons) off</td><td> $1 7 . 2 \pm 1 . 0$ </td><td> $0 . 2 0 3 \pm 0 . 0 3$ </td><td> $0 . 6 5 0 \pm 0 . 0 1$ </td></tr><tr><td>Mem (Attempts) off</td><td> $1 0 . 0 \pm 0 . 6$ </td><td> $0 . 0 9 6 \pm 0 . 0 2$ </td><td> $0 . 6 2 7 \pm 0 . 0 1$ </td></tr><tr><td>Self-evolution off</td><td> $7 . 8 \pm 0 . 6$ </td><td> $0 . 0 8 0 \pm 0 . 0 2$ </td><td> $0 . 6 3 0 \pm 0 . 0 2$ </td></tr></table>

Table 2: Panel A mechanism ablations, testing whether each of $\overline { { { \mathrm { S E A D S } } { \mathrm { ? } } { \mathrm { s } } } }$ mechanisms is loadbearing: one mechanism removed at a time relative to the full pipeline, mean ± SE across 5 replicates. “Memory (lessons/attempts) off” isolates the reflector’s natural-language lessons from the append-only attempt log (§3.3.5). Higher is better for productivity and performance; lower is more novel.

## 5.3 Ablations

We run a battery of ablations (Table 2), each isolating one mechanism relative to the full pipeline. Self-evolution is the single most load-bearing mechanism: removing it shrinks the admitted set itself (7.8 vs. 14.0 admits) and cuts Sharpe by 68% (0.08 vs. 0.25), the largest drop in the battery. Removing the Planner instead inflates admissions nearly threefold (37.2 vs. 14.0) while Sharpe still collapses (0.10)—a different failure mode, too many weak factors rather than too few. Removing the Alignment Checker or either memory tier alone costs 15–62% of Sharpe, while novelty moves comparatively little across the whole battery. The validation gate and look-ahead firewall are audited differently: turning either off changes what counts as an admission at all, not just how many, so neither is ablated outright.

## 6 Limitations

Small samples: admitted-factor counts run in the tens per replicate, OOS windows 59–71 months, so most cross-system differences are not reliably distinguishable. Replicate counts (5 static, 3 rolling) reflect this study’s API budget, not a deliberate power limit.

Independent admission does not guarantee a coherent combinedportfolio: the gate certifies a factor’s marginal in-sample signal, not its joint, non-monotone out-of-sample contribution alongside later admits (Figure 3, Appendix E).

Achievable Sharpe is dataset-dependent: the panels differ in alpha headroom, so absolute Sharpe alone is not informative—§5.1 compares systems within a panel instead.

Backbone information leakage is soft, not absent: SEADS uses one current LLM (GPT-5.2) for every role, not a release-dated model per historical date. Proposing from feature descriptions rather than realized returns keeps this close to the safe end, but a soft residual remains: general knowledge of which factor families worked could still bias proposed hypotheses [Li et al., 2025b, Glasserman and Lin, 2024, Sarkar and Vafa, 2024, Lopez-Lira et al., 2025], more so across the rolling protocol’s several decision dates than the static comparison’s single cutoff. A release-dated backbone (e.g. GPT-4o-mini, October 2023) is the natural alternative, but leaves only ∼25 clean OOS months—too few here, and confounded with a weaker, older model’s reasoning capacity.

Our look-aheadfirewall may be overly conservative: static pattern-matching can reject safe formulas with unsafe ones; a less conservative alternative exists but postdates these results.

Candidate counts are matched, compute is not: the 300-candidate budget matches initial proposals, but self-evolution’s mutation retries and repairs add LLM calls beyond one proposal per slot— removing it alone collapses Sharpe by 68% (Table 2)—so realized compute per admission is not matched the way candidate counts are.

Reported Sharpe is gross of trading costs: we model neither transaction costs nor turnover, so frequent reranking or more complex formulas could erode realized performance disproportionately, a gap no number reported here reflects.

## 7 Conclusion

AEAP has neither a shared definition nor a shared evaluation discipline for systems that discover their own hypotheses rather than test a researcher’s. Our three contributions (§1) give a definition and its building blocks (C1); an output-facing standard scoring a discovered set on multiple axes jointly, plus a method for backtesting the discovery system across historical decision dates rather than just one run’s output (C2); and SEADS, instantiating both against five re-implemented baselines (C3).

Across both panels, no system excels on every standard §3.4 scores; even a multi-axis standard can not settle the best system from one decision date alone. §3.5 asks what static scoring cannot: whether the discovery process holds up when re-executed under historical, point-in-time constraints. Our two negative findings—non-monotone combination, a soft backbone-leakage residual—are evidence an unsettled field should expect, not a failure of one system.

## Ethical Statement

This work concerns automated discovery of trading signals, a dual-use capability whose primary risk is overstated performance claims rather than any single dangerous capability. Our evaluation standard is deliberately deflationary: scoring a discovery system out-of-sample and reporting negative findings reduces, rather than inflates, the field’s headline claims. Automated discovery at scale can contribute to crowding if deployed identically by many actors; our panels include small-cap names with capacity limits our construction does not model. Nothing here is investment advice.

## References

Svetlana Bryzgalova, Sven Lerner, Martin Lettau, and Markus Pelger. Missing financial data. Review of Financial Studies, 38(3):803–882, 2025.

Lang Cao, Zekun Xi, Long Liao, Ziwei Yang, and Zheng Cao. Chain-of-alpha: Unleashing the power of large language models for alpha mining in quantitative trading, 2025. both v1 and v2 removed by arXiv administrators for a licensing-rights issue at submission; cited only to acknowledge the design, not as a verifiable result.

Binqi Chen, Hongjun Ding, Ning Shen, Jinsheng Huang, Taian Guo, Luchen Liu, and Ming Zhang. AlphaSAGE: Structure-aware alpha mining via GFlowNets for robust exploration, 2026a. ICLR 2026.

Xiwen Chen, Wenhui Zhu, Songzhu Zheng, Kashif Rasul, Yueyue Deng, and Huayu Li. SHARP: A selfevolving human-auditable rubric policy for financial trading agents, 2026b. Working paper.

Junyan Cheng and Peter Chin. Empirical asset pricing with large language model agents, 2025. ICLR 2025 Workshop on Advances in Financial AI.

Yuhan Cheng and Ke Tang. GPT’s idea of stock factors. Quantitative Finance, 24(9):1301–1326, 2024.

Chanyeol Choi, Yoon Kim, Yu Yu, Young Cha, V. Zach Golkhou, Igor Halperin, Georgios Papaioannou, Minkyu Kim, Zhangyang Wang, Jihoon Kwon, Minjae Kim, Alejandro Lopez-Lira, and Yongjae Lee. From text to alpha: Can LLMs track evolving signals in corporate disclosures?, 2025. Working paper.

Can Cui, Wei Wang, Meihui Zhang, Gang Chen, Zhaojing Luo, and Beng Chin Ooi. AlphaEvolve: A learning framework to discover novel alphas in quantitative investment. In Proceedings of the 2021 International Conference on Management of Data (SIGMOD), pages 2208–2216. ACM, 2021.

Han Ding, Yinheng Li, Junhao Wang, Hang Chen, Doudou Guo, and Yunbai Zhang. Large language model agent in financial trading: A survey, 2024. International Conference on Computers in Management and Business (ICCMB) 2026.

Hongjun Ding, Binqi Chen, Jinsheng Huang, Taian Guo, Zhengyang Mao, Guoyi Shao, Lutong Zou, Luchen Liu, and Ming Zhang. AlphaEval: A comprehensive and efficient evaluation framework for formula alpha mining, 2026. KDD 2026.

Yifei Dong, Fengyi Wu, Kunlin Zhang, Yilong Dai, Sanjian Zhang, Wanghao Ye, Sihan Chen, and Zhi-Qi Cheng. Large language model agents in finance: A survey bridging research, practice, and real-world deployment. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2025, pages 17889–17907, 2025.

Yitong Duan, Chuheng Zhang, and Jian Li. FactorMAD: A multi-agent debate framework based on large language models for interpretable stock alpha factor mining. In Proceedings of the 6th ACM International Conference on AI in Finance (ICAIF). ACM, 2025.

Eugene F. Fama and James D. MacBeth. Risk, return, and equilibrium: Empirical tests. Journal of Political Economy, 81(3):607–636, 1973.

Paul Glasserman and Caden Lin. Assessing look-ahead bias in stock return predictions generated by GPT sentiment analysis, 2024. Journal of Financial Data Science, vol. 6, no. 1, pp. 25–42.

Shihao Gu, Bryan Kelly, and Dacheng Xiu. Empirical asset pricing via machine learning. Review of Financial Studies, 33(5):2223–2273, 2020.

Taian Guo, Haiyang Shen, Junyu Luo, Binqi Chen, Hongjun Ding, Jinsheng Huang, Luchen Liu, Yun Ma, and Ming Zhang. AlphaPROBE: Alpha mining via principled retrieval and on-graph biased evolution, 2026. Working paper.

Jun Han, Shuo Zhang, Wei Li, Yifan Dong, Tu Hu, Yumo Zhu, Xiaomin Yu, Xin Guo, Zhaowei Liu, Kunyi Wang, Jingping Liu, Tianyi Jiang, Ruichuan An, Sen Hu, Zhi Yang, Ronghao Chen, and Huacan Wang. QuantaAlpha: An evolutionary framework for LLM-driven alpha mining, 2026. Working paper.

Campbell R. Harvey, Yan Liu, and Heqing Zhu. . . . and the cross-section of expected returns. Review of Financial Studies, 29(1):5–68, 2016.

Allen Yikuan Huang and Zheqi Fan. Beyond prompting: An autonomous framework for systematic factor investing via agentic AI, 2026. Working paper (rev. Apr 2026).

Yikuan Huang, Zheqi Fan, Kaiqi Hu, and Yifan Ye. From hypotheses to factors: Constrained LLM agents in cryptocurrency markets, 2026. Working paper.

Theis Ingerslev Jensen, Bryan Kelly, and Lasse Heje Pedersen. Is there a replication crisis in finance? Journal ofFinance, 78(5):2465–2518, 2023.

Zuoyou Jiang, Li Zhao, Rui Sun, Ruohan Sun, Zhongjian Li, Jing Li, Daxin Jiang, Zuo Bai, and Cheng Hua. Alpha-R1: Alpha screening with LLM reasoning via reinforcement learning, 2025. Working paper.

Ralph S. J. Koijen and Bradford Levy. Assessing the benefits of optimized agentic AI systems for asset pricing. SSRN 6474601, 2026. Working paper.

Zhizhuo Kou, Holam Yu, Junyu Luo, Jingshu Peng, Xujia Li, Chengzhong Liu, Juntao Dai, Lei Chen, Sirui Han, and Yike Guo. Automate strategy finding with LLM in quant investment. In Findings of EMNLP 2025, 2025.

Serhiy Kozak, Stefan Nagel, and Shrihari Santosh. Shrinking the cross-section. Journal of Financial Economics, 135(2):271–292, 2020.

Haohang Li, Yupeng Cao, Yangyang Yu, Shashidhar Reddy Javaji, Zhiyang Deng, Yueru He, Yuechen Jiang, Zining Zhu, K. P. Subbalakshmi, Jimin Huang, Lingfei Qian, Xueqing Peng, Jordan W. Suchow, and Qianqian Xie. InvestorBench: A benchmark for financial decision-making tasks with LLM-based agent. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2509–2525, 2025a.

Xiangyu Li, Yawen Zeng, Xiaofen Xing, Jin Xu, and Xiangmin Xu. Profit mirage: Revisiting information leakage in LLM-based financial agents, 2025b. Working paper.

Yuante Li, Xu Yang, Xiao Yang, Minrui Xu, Xisen Wang, Weiqing Liu, and Jiang Bian. R&D-Agent-Quant: A multi-agent framework for data-centric factors and model joint optimization, 2025c. NeurIPS 2025.

Terence Lim, Kumar Muthuraman, and Michael Sury. QRAFTI: An agentic framework for empirical research in quantitative finance, 2026. Working paper.

Qinhong Lin, Ruitao Feng, Yinglun Feng, Zhenxin Huang, Yukun Chen, Zhongliang Yang, Linna Zhou, Binjie Fei, Jiaqi Liu, and Yu Li. FactorEngine: A program-level knowledge-infused factor mining framework for quantitative investment, 2026. Working paper.

Fengyuan Liu, Yuchen Fu, Yuqi Wang, and Qi Liu. XALPHA: A memory-driven AI quant researcher for hypothesis-to-code alpha discovery, 2026a. Working paper.

Fengyuan Liu, Yi Huang, Sichun Luo, Yuqi Wang, Yazheng Yang, Xinye Li, Zefa Hu, Junlan Feng, and Qi Liu. Cognitive alpha mining via LLM-driven code-based evolution, 2026b. ACL 2026.

Andrew W. Lo. The statistics of Sharpe ratios. Financial Analysts Journal, 58(4):36–52, 2002.

Alejandro Lopez-Lira, Yuehua Tang, and Mingyin Zhu. The memorization problem: Can we trust LLMs economic forecasts?, 2025. Working paper.

Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Foerster, Jeff Clune, and David Ha. The AI scientist: Towards fully automated open-ended scientific discovery, 2024. Working paper.

Haochen Luo, Yuan Zhang, and Chen Liu. EFS: Evolutionary factor searching for sparse portfolio optimization using large language models, 2025. Working paper.

Kunihiro Miyazaki, Takanobu Kawahara, Stephen Roberts, and Stefan Zohren. Toward expert investment teams: A multi-agent LLM system with fine-grained trading tasks, 2026. Working paper.

Alexander Novikov, Ngan Vˆ u, Marvin Eisenberger, Emilien Dupont, Po-Sen Huang, Adam Zsolt Wagner,˜ Sergey Shirobokov, Borislav Kozlovskii, Francisco J. R. Ruiz, Abbas Mehrabian, M. Pawan Kumar, Abigail See, Swarat Chaudhuri, George Holland, Alex Davies, Sebastian Nowozin, Pushmeet Kohli, and Matej Balog. AlphaEvolve: A coding agent for scientific and algorithmic discovery, 2025. Working paper, Google DeepMind.

Bo Qu and Mingguang Chen. CLQT: A closed-loop, cost-aware, strategy-consistent benchmark for diagnostic evaluation of LLM portfolio-management agents, 2026. Working paper.

Suproteem K. Sarkar and Keyon Vafa. Lookahead bias in pretrained language models. SSRN 4754678, 2024. Working paper; poster, ICML 2025 DIG-BUGS Workshop (non-archival).

Hao Shi, Weili Song, Xinting Zhang, Jiahe Shi, Cuicui Luo, Xiang Ao, Hamid Arian, and Luis Seco. AlphaForge: A framework to mine and dynamically combine formulaic alpha factors, 2025. AAAI 2025.

Runze Shi, Shengyu Yan, Yuecheng Cai, and Chengxi Lv. Hubble: An LLM-driven agentic framework for safe, diverse, and reproducible alpha factor discovery, 2026a. Working paper.

Yu Shi, Yitong Duan, and Jian Li. Navigating the alpha jungle: An LLM-powered MCTS framework for formulaic alpha factor mining, 2026b. AAAI 2026.

Ziyi Tang, Zechuan Chen, Jiarui Yang, Jiayao Mai, Yongsen Zheng, Keze Wang, Jinrui Chen, and Liang Lin. AlphaAgent: LLM-driven alpha mining with regularized exploration to counteract alpha decay, 2025. KDD 2025.

Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, Wayne Xin Zhao, Zhewei Wei, and Ji-Rong Wen. A survey on large language model based autonomous agents. Frontiers ofComputer Science, 18(6):186345, 2024a.

Meiyun Wang, Kiyoshi Izumi, and Hiroki Sakaji. LLMFactor: Extracting profitable factors through prompts for explainable stock movement prediction, 2024b. Findings of ACL 2024.

Saizhuo Wang, Hang Yuan, Leon Zhou, Lionel M. Ni, Heung-Yeung Shum, and Jian Guo. Alpha-GPT: Humanai interactive alpha mining for quantitative investment, 2023. EMNLP 2025 System Demonstration Track.

Yanlong Wang, Jian Xu, Hongkang Zhang, Shao-Lun Huang, Danny Dongning Sun, and Xiao-Ping Zhang. FactorMiner: A self-evolving agent with skills and experience memory for financial alpha discovery, 2026. Working paper.

Yining Wang, Jinman Zhao, and Yuri Lawryshyn. GPT-Signal: Generative AI for semi-automated feature engineering in the alpha research process, 2024c. FinNLP 2024.

Zhangyuhua Weng, Shengli Zhang, Taotao Wang, and Yihan Xia. AlphaLogics: A market logic-driven multiagent system for scalable and interpretable alpha factor generation, 2026. Working paper.

Yijia Xiao, Edward Sun, Di Luo, and Wei Wang. TradingAgents: Multi-agents LLM financial trading framework, 2025. AAAI 2025 Workshop on Multi-Agent AI in the Real World.

Yutaro Yamada, Robert Tjarko Lange, Cong Lu, Shengran Hu, Chris Lu, Jakob Foerster, Jeff Clune, and David Ha. The AI scientist-v2: Workshop-level automated scientific discovery via agentic tree search, 2025. Working paper.

Shuo Yu, Hongyan Xue, Xiang Ao, Feiyang Pan, Jia He, Dandan Tu, and Qing He. Generating synergistic formulaic alpha collections via reinforcement learning, 2023. KDD 2023.

Yangyang Yu, Haohang Li, Zhi Chen, Yuechen Jiang, Yang Li, Denghui Zhang, Rong Liu, Jordan W. Suchow, and Khaldoun Khashanah. FinMem: A performance-enhanced LLM trading agent with layered memory and character design, 2024. AAAI Spring Symposium Series 2024, vol. 3, no. 1, pp. 595–597.

Lingzhe Zhang, Tong Jia, Yunpeng Zhai, Zixuan Xie, Chiming Duan, Minghua He, Philip S. Yu, and Ying Li. From feedback loops to policy updates: Reinforcement fine-tuning for LLM-based alpha factor discovery, 2026. Working paper.

Tianjiao Zhao, Jingrao Lyu, Stokes Jones, Harrison Garber, Stefano Pasquali, and Dhagash Mehta. AlphaA gents: Large language model based multi-agents for equity portfolio constructions, 2025. Working paper.

## Appendix

This appendix gives full methodology and results supporting the main paper, which cites specific tables/figures below where relevant.

## A Data Construction Detail

Panel A: JKP characteristics (ex-microcap). Built from the JKP global factor characteristics data [Jensen et al., 2023], restricted to US-listed stocks (excntry = USA). Four size-filtered variants of this panel exist in our data pipeline; the one used for every result in this paper drops rows whose size grp label (recomputed every month from that month’s NYSE market-cap percentile breakpoints, so the cut itself carries no look-ahead) is micro or nano, retaining small, large, and mega. This yields 1,936,944 stock-months across 21,052 unique stocks, 1925–2025, and ∼400 taxonomy-declared characteristics (representative examples: oneyear asset growth, book-to-market, operating-cash-flow-to-assets, gross profitability, sales growth, capex intensity, the Piotroski F-score, the Ohlson O-score, the Altman Z-score, 60-month beta, 252-day idiosyncratic volatility, 126-day turnover, and 12-1 momentum). Novelty on Panel A is checked against this same panel, so a candidate must be distinct from the already-curated JKP zoo itself, not merely from a smaller reference. The in-sample window used for every result in this paper is 2010–2019; the out-of-sample window runs through December 2025 (n ≈ 71 months).

Panel B: CRSP/Compustat base features. Two further panels support Panel B. The base-features panel is built directly from CRSP and Compustat as 87 comparatively primitive accounting and market variables, organized into 13 categories (trading frictions, market price, size, operating assets, liquid assets, intangibles, leverage, equity structure, revenue and costs, cash flow, earnings, dividends, and capital actions; category sizes range from 3 to 13 variables): 1,213,574 stock-months, 5,907 unique stocks, 1975–2024. These are deliberately not pre-curated characteristics—ratios and transforms an existing factor library would already apply are left for the discovery system itself to construct.

Panel B’s novelty reference. The reference panel for novelty on Panel B is the 45-characteristic panel of Bryzgalova et al. [2025] (“Missing Financial Data”), built by that paper’s own local cross-sectional/backwardcross-sectional imputation method without look-ahead, restricted to the top 80% of stocks by the previous month’s market equity: 2,402,623 stock-months, 22,336 unique stocks, 1977–2024, spanning six categories (value, profitability, investment, trading frictions, past returns, and a single size/intangibles variable; category sizes 1–13). We check Panel B’s discovered factors against this established reference rather than against Panel B’s own 87 raw inputs, since the latter would reward literal repackaging of a single input as “novel.” The in sample window matches Panel A (2010–2019); the out-of-sample window runs through this panel’s own data availability, November 2024 (n ≈ 59 months)—shorter than Panel A’s, correspondingly lowering this panel’s statistical power (§6).

Point-in-time convention. Every feature source in our data layer is contractually required to store its return column as a lead-1-month return—ret(t) realized over (t, t+1]—documented in each source’s taxonomy entry rather than left to each track’s own convention. This is a real invariant, not a formality: a per-track lag parameter was tried and removed in favor of the uniform contract, since a single wrong return-column convention on any one panel silently corrupts every factor built on top of it.

## B Reference Architecture Component Audit

Table 3 summarizes what each of the five re-implemented systems’ own paper reports about itself (§4 reimplements and re-scores all six under one shared protocol instead). Table 4 then audits which components of Figure 1 each system instantiates, expanding Table 3’s summary columns into four of §2.1’s five building blocks (B3, execution against point-in-time data, is omitted as uniform across every factor-discovery system surveyed) plus backtesting-the-system.

## C The SEADS Discovery Loop: Pseudocode

Algorithm 1 states §3.3’s five named components (Figure 1) as one closed loop, at the level of abstraction Table 4 compares systems at. It elides engineering detail—concurrency, name-collision handling, exact prompt content—covered instead in Appendix E; GATE is Table 6’s full ordered sequence, not a single test, and SELF-EVOLVE re-enters at Step 2 for the same slot rather than starting a new one. The rolling protocol (Appendix F) re-invokes this whole procedure at each repropose event as walk-forward time advances, rather than nesting that outer loop here.

<table><tr><td>System</td><td>Validation signal</td><td>Nov. check? # gates</td><td></td></tr><tr><td>AlphaAgent</td><td>AST-originality + LLM judge</td><td>√</td><td>2</td></tr><tr><td>AlphaForge</td><td>fitness pools</td><td>x</td><td>1</td></tr><tr><td>RD-Agent(Q)</td><td>backtest-metric unit</td><td>x</td><td>1</td></tr><tr><td>QuantaAlpha</td><td>LLM-judged semantic + complexity</td><td>partial</td><td>2</td></tr><tr><td>Beyond Prompting</td><td>pre-spec. stat. gates</td><td>x</td><td>2</td></tr><tr><td>SEADS (ours)</td><td>multi-stage stat. gate</td><td>√</td><td>9</td></tr></table>

Table 3: Per-system comparison, based on what each cited system’s own paper reports about itself. Whether the discovery loop is itself re-executed at more than one historical decision date (§3.5) is audited separately in Table 4. “# gates”: count of independent admission criteria, statistical or heuristic.

<table><tr><td>System</td><td>B1 Hyp.</td><td>B2 Formal.</td><td>B4 Gate</td><td>B5 Persist.</td><td>Backtests</td></tr><tr><td>AlphaAgent</td><td>LLM idea agent</td><td>symbolic factor</td><td>2 criteria</td><td>AST-orig. memory</td><td>x</td></tr><tr><td>AlphaForge</td><td>none (mined)</td><td>symbolic factor</td><td>fitness pool</td><td>factor zoo</td><td>x</td></tr><tr><td>RD-Agent(Q)</td><td>LLM hypothesis</td><td>free-form code</td><td>1 unit</td><td>knowledge forest</td><td>x</td></tr><tr><td>QuantaAlpha</td><td>div. planning</td><td>symbolic factor</td><td>2 criteria</td><td>trajectory pop.</td><td>x</td></tr><tr><td>Beyond Prompting</td><td>closed-loop</td><td>symbolic factor</td><td>2 (pre-spec.)</td><td>strategy update</td><td>x</td></tr><tr><td>SEADS</td><td>agent Planner + Proposer</td><td>Python/pandas</td><td>9 criteria</td><td>tax.+log+lessons+lib.</td><td>√</td></tr></table>

Table 4: Reference-architecture component audit (§3.2), based on our re-implementation of each system (§4, Appendix G). AlphaForge’s “none (mined)” B1 cell reflects that it is not itself an AEAP system by §2.1’s definition (no LLM-driven hypothesis step)—we include it as the field’s standard non-agentic comparison baseline, not as evidence that the building blocks are optional within the AEAP systems we survey.

NEARPASS holds when θ cleared a minimum progress floor or failed specifically at the rank-IC or partial-IC stage (Appendix E); a candidate failing far short of any bar is discarded rather than mutated. Every system in Table 3 instantiates Steps 1–3 in some form, and Step 5’s cross-round persistence appears in every system too (Table 4’s B5 column), though SEADS’s own mechanism is the most elaborate (four components vs. one each); Step 4—mutating a near-miss rather than discarding it—is not something Table 4 audits for any system.

## D Validation Gate and Firewall Detail

Table 5 gives the look-ahead firewall’s rule classes; Table 6 gives the validation gate’s exact current execution order, expanding §3.3.3. The gate is deterministic throughout; admission is the logical AND of every configured check, short-circuiting at the first failure, followed by the sign backstop.

## E Pipeline Internals

Concurrency and naming. Within one run, shared mutable state (the control panel, the trial counter, the name registry) is mutated strictly sequentially or under a lock, so concurrent candidate dispatch cannot corrupt an admission decision; a name collision bumps an existing version suffix rather than stacking a new one.

Self-evolution. A candidate is mutated only if its rank-IC t-statistic cleared a minimum progress floor of 2.0 or it failed specifically at the rank-IC or partial-IC stage; one failing far short of any bar is discarded rather than mutated. A progress-weighted score ranks a mutation strictly by the latest gate stage it reached (rank-IC < regime-consistency < novelty < partial-IC < Fama–MacBeth, each a full tier above the last), with magnitude breaking ties only within a tier, so that reaching further into the gate always outranks a larger margin earlier in it. Its mutations are correlated draws from the same parent candidate rather than fresh independent trials; whether the rank-IC gate’s live Bonferroni correction (Appendix K) should count each mutation separately toward the cumulative candidate count or discount correlated attempts is left for future work.

## Algorithm 1: SEADS Closed-Loop Factor Discovery

Input: Point-in-time panel X; reference set Z; gate thresholds Θ = {τ<sub>cov</sub>, τ<sub>ic</sub>, τ<sub>regime</sub>, τ<sub>nov</sub>, τ<sub>pic</sub>, τ<sub>fm</sub>};   
rounds K; slots per round S, K×S=300; max self-evolution attempts E   
Output: Admitted factor library G; combinator weights w   
1: G ← ∅; memory (T<sub>1</sub>, T<sub>2</sub>, T<sub>3</sub>) ← (∅, ∅, ∅)   
2: for round = 1 to K do   
3: τ ← Bonferroni-dynamic update from trials so far ▷ Appendix K   
Step 1: Generation (Planner, Proposer) ▷ §3.3.1   
4: {seed<sub>1</sub>, . . . , seed<sub>S</sub>} ← PLANNER(T<sub>1</sub>, T<sub>2</sub>, T<sub>3</sub>, G) ▷ one call plans the round   
5: for slot = 1 to S (independent; dispatched concurrently) do   
6: θ ← PROPOSER(seed<sub>slot</sub>, T<sub>1</sub>, T<sub>2</sub>, T<sub>3</sub>, G) ▷ rationale + symbolic formula   
7: for attempt = 0 to E do   
8: if attempt > 0 then   
9: θ ← SELFEVOLVE(θ, stage, T ) ▷ Step 4 (§3.3.4): mutate the near-pass   
10: end if   
Step 2: Sanity Checks ▷ §3.3.2   
11: if not STRUCTURALLYVALID(θ) then   
12: break ▷ malformed spec; no code executes   
13: end if   
14: code ← IMPLEMENT(θ, X) ▷ sandboxed; firewall + ≤ 2 LLM repairs   
15: if firewall fails or not ALIGNMENTCHECK(θ, code) then   
16: break   
17: end if   
Step 3: Validation Gate ▷ §3.3.3   
18: decision, stage ← GATE(f<sub>θ</sub>, X, G, Z, Θ) ▷ sequential AND, Table 6   
19: if decision = admit then   
20: G ← G ∪ {f<sub>θ</sub>}; break ▷ §3.3.5   
21: else if not NEARPASS(stage) or attempt = E then   
22: break ▷ far short of any bar, or evolution budget spent   
23: end if   
24: end for   
25: end for   
Step 5: Persistent State ▷ §3.3.5   
26: update $T _ { 1 }$ (characteristic stats), T<sub>2</sub> (every attempt and its stage); T<sub>3</sub> ← REFLECTOR(T<sub>2</sub>)   
27: end for   
28: w ← COMBINATOR(G) ▷ Ridge, α=1, rank-normalized library values   
29: return G, w

Planner and Proposer context scope. The Planner sees the full history of past attempts when planning a new round; each seed’s Proposer instance instead sees only prior attempts sharing at least one of that seed’s underlying variables, not the Planner’s full history. This scoping matches each role’s job: the Planner needs breadth across themes and variable families, not formula-construction guidance, since it emits a seed rather than a formula; the Proposer needs depth on a narrow, variable-specific slice of prior attempts, and correspondingly more explicit formula-construction guidance than the Planner requires.

Planner backstops and structural pre-check. Two deterministic backstops correct observed Planner failure modes: one forces complexity-tier rebalancing when the Planner’s own “aim for a mix” instruction alone leaves it over-concentrated on a single tier (simple, moderate, or rich); the other substitutes fresh variables into an explore seed whose variable pair was already tried repeatedly, since the Planner otherwise kept reselecting the same pair despite full visibility into past attempts. A deterministic structural pre-check separately rejects malformed specs before any code executes: naming convention, minimum rationale length, a non-empty formula, and a valid category.

Prompt design. Of SEADS’s six LLM-prompted roles (Table 9), four—Planner, Proposer, Self-Evolve, Alignment Checker—share one structural convention: a GENERAL INFO block states the role’s job and previews the prompt’s own section order (e.g. the Planner’s own preamble: “RULES you must follow, then three memory tiers. . . then FINAL DIRECTION with the exact output format”), so a long, many-section prompt is self-documenting before the model reads past the first paragraph. RULES follow next; the Proposer and Self-Evolve additionally interleave PANEL STRUCTURE & LOOK-AHEAD RULES and FACTOR QUALITY GUIDANCE here, since panel conventions and what makes a formula worth proposing are each several distinct points, not one rule among others—the Planner has neither, since it emits a seed rather than a formula. The three memory tiers—Building Blocks (T ), Past Attempts (T ), Lessons Learned (T ) in Algorithm 1’s notation—come after;

<table><tr><td></td><td># Rule class</td><td>What it blocks (static AST)</td></tr><tr><td></td><td>1 Forbidden</td><td>any read of return/target columns</td></tr><tr><td></td><td>columns 2 Dynamic</td><td>via subscript, attribute, or . get getattr, eval, exec, .eval(),</td></tr><tr><td></td><td>access</td><td>.query()</td></tr><tr><td></td><td>3 Negative shift</td><td>.shift/.diff/.pct_change with</td></tr><tr><td></td><td>4 Centered</td><td>a negative literal .rolling(center=True)</td></tr><tr><td></td><td>window 5 Bare rank</td><td>.rank() not scoped to</td></tr><tr><td></td><td></td><td>groupby(&#x27;date&#x27;)</td></tr><tr><td>7 Full-</td><td>6 Per-stock transform</td><td>groupby(&#x27;permno&#x27;) .transform(...) over full history panel-wide mean/std/... or</td></tr></table>

Table 5: The seven rule classes of the deterministic AST look-ahead firewall; a whitelist additionally enforces that every column read is taxonomy-declared, and every read is further scoped to a sandboxed namespace exposing only the formula’s statically-read columns.

<table><tr><td>#</td><td>Check</td><td>Threshold key</td><td>Value</td><td>Runs if</td></tr><tr><td>Oa</td><td>duplicate fingerprint</td><td></td><td></td><td>always</td></tr><tr><td>0b</td><td>self-collapse</td><td>fixed</td><td>0.95</td><td>always</td></tr><tr><td>1</td><td>coverage</td><td>Min_Coverage</td><td>0.5</td><td>always</td></tr><tr><td>2</td><td>Rank-IC t-stat</td><td>Min_RANKIC_T</td><td>max(3.0, Bonf.-dyn.)</td><td>coverage ok</td></tr><tr><td>3</td><td>regime consist.</td><td>Regime_Lookback_Years</td><td>10</td><td>rank-IC ok</td></tr><tr><td>3.5/4</td><td>novelty</td><td>Max_Corr</td><td>0.7</td><td>2, 3 ok</td></tr><tr><td>5</td><td>partial-IC</td><td>Min_Partial_IC</td><td>1.0 (top-5)</td><td>2,3,4 ok</td></tr><tr><td>6</td><td>Fama- MacBeth</td><td>Min_FM</td><td>1.0 (top-5)</td><td>2, 3, 4, 5 ok</td></tr><tr><td>7</td><td>sign alignment</td><td>determ.</td><td></td><td>alpha-pass</td></tr></table>

Table 6: Validation-gate execution order and threshold values (§3.3.3), used for every result in this paper. The self-collapse check (0b) rejects a candidate whose final formula correlates ≥ 0.95 with a plain rank of one of its own declared input variables. The regime-consistency check’s (3) window reanchors to the in-sample start, sliding automatically under §3.5’s rolling protocol. The sign check (7) requires the realized rank-IC, partial-IC, and Fama–MacBeth beta to agree with each other in sign. Novelty runs after rank-IC/regime rather than before because the shared control-panel correlation matrix used by checks 4–6 is computed once, unconditionally, the moment rank-IC passes, making novelty a cheap lookup by the time it runs, while it would otherwise be the single slowest check once the reference panel is large (measured at over 100 seconds per candidate against a ∼400-column reference, vectorized). The candidate budget (300) is a separate, global experimental parameter, not a per-check threshold.

the Alignment Checker has none, since it judges one already-written formula in isolation rather than proposing a new one. A mandatory pre-submission self-check and the output template both sit last, immediately before generation, on the same closest-instruction-wins logic. Implementation Agent and the reflector use simpler, single-shot prompts scoped to their narrower tasks (fixing a formula; summarizing a round into a lesson), not this multi-section structure.

Alignment Checker grounding. The Alignment Checker’s prompt requires a formula trace—a mechanical, line-by-line restatement of what each step computes, filled in before the aligned/not-aligned verdict— for two purposes. First, grounding: the checker once rejected a formula by citing a .shift(1) call that did not appear anywhere in the actual code, a fabrication that cost a pipeline slot on a formula with no real defect, so every look-ahead rejection now must quote an exact substring present in that item’s own formula. Second, al gebraic cancellation, motivated by a second real incident: a formula computing (shrout\*cfacshr)/shrout was live and admitted before this check existed, algebraically just cfacshr—shrout cancels exactly—even though its rationale described a genuine two-variable interaction between share count and an adjustment factor; the trace must now catch a formula that reduces to fewer effective variables than variables used claims, since that misdescribes what the formula actually computes regardless of how genuine the stated interaction sounds. Rank direction and sign convention are explicitly out of scope for this judge for a separate, sign-specific reason: it had repeatedly rejected correctly-signed formulas for “not inverting the rank” when the rationale had already established the raw variable as higher-is-better, so sign checking was moved out of the LLM judge entirely and made a deterministic post-hoc comparison against the realized Fama–MacBeth coefficient instead (§3.3.3, check 7).

Combinators. We use a Ridge regression (α = 1) fit on the admitted library’s rank-normalized values alone, an ex-ante default chosen independently of any comparative performance check. A more expressive learner—XGBoost or LightGBM, say—could also convert the admitted library into scores; we use linear/Ridge instead because it keeps the combinator stable, comparable across systems and panels, and exactly reproducible, properties this evaluation needs more than whatever incremental predictive power a more flexible fit might add. This is not merely a stability preference: every admitted factor already cleared the gate’s Rank-IC and partial-IC checks (§3.3.3), both rank/linear-correlation measures, so the library a combinator scores is pre-selected for linear predictive power—a nonlinear combinator has comparatively little nonlinearity left to exploit, and would mainly add tuning variance the evaluation does not need. This deployed Ridge combinator, Figure 3’s single-OLS illustration, and the rolling comparison’s raw-panel joint regression (Figure 2) are three distinct procedures answering three different questions about the same admitted library. Figure 3 and Figure 4 give two further empirical patterns behind §6’s claims, and Table 7 maps every mechanism named above to the module that implements it.

Individual vs. combined OOS Sharpe  
![](images/326c65a1a341b385a2ab371b12dfb637b4b8747eb53eb7a19859571e3c4f1ff4.jpg)  
Figure 3: Mean per-factor out-of-sample Sharpe vs. combined-portfolio out-of-sample Sharpe, three replicates per system, Panel A (JKP). Combined SR is from a single OLS regression of next-month return on the replicate’s admitted factors, fit once over the in-sample window and applied out-ofsample to score and rank stocks into a long-short portfolio each month; we report that portfolio’s annualized Sharpe. 14 of 15 replicates fall below the dashed diagonal—combining admitted factors typically does not preserve, let alone improve on, their average individual quality (§6).

Real IS vs. OOS rank-IC, n=83 admitted factors  
![](images/16f21ddea0f148f1616c6bfe478d0bd96fba0b853ab0e4b3da347135db773f51.jpg)

Figure 4: In-sample vs. out-of-sample mean rank-IC across admitted factors from a few sample runs of SEADS: sign-agreement most of the time, no systematic shrinkage in mean |IC|, but far from a tight relationship (§6).
<table><tr><td>Mechanism</td><td>Module / function</td></tr><tr><td>Planner (seeds)</td><td>planner_agent.plan_run</td></tr><tr><td>Proposer</td><td>hypothesis_agent. propose_factor</td></tr><tr><td>Look-ahead firewall</td><td>implementation_agent.</td></tr><tr><td>Sandboxed execution</td><td>check_formula_firewall implementation_agent.</td></tr><tr><td>Alignment / sign</td><td>compute_factor rationale_gate</td></tr><tr><td>Validation gate</td><td>validation_engine. validate_factor</td></tr><tr><td>Self-evolution</td><td>self_evolve_agent</td></tr><tr><td>Memory</td><td>memory manager</td></tr><tr><td>Rolling/backtesting-</td><td>rolling-engine</td></tr><tr><td>the-system Combination</td><td></td></tr></table>

Table 7: Map from named mechanisms to the modules that realize them.

## F Backtesting-the-System: Full Walk-Forward Protocol

We use “rolling” and “walk-forward” interchangeably for §3.5’s protocol, since it is the same sliding-window re-evaluation discipline standard in trading-strategy validation, applied here to the discovery process itself rather than to a fixed model’s parameters. The engine supports two factor-update modes at each re-discovery event: revalidate (the default; re-checks every existing specification’s statistics on the slid in-sample window with zero new LLM calls) and repropose (a full fresh discovery run—new Planner seeds, new proposals, a re-run gate—in the event’s own subfolder). §5.2’s cross-system rolling comparison uses repropose throughout, since revalidate alone cannot add genuinely new candidates and would not test whether re-discovery itself has value. The in-sample window is a fixed-length sliding window, not an expanding one: at re-discovery event with outof-sample month m, the in-sample end date is the month-end immediately preceding m, and the in-sample start date is exactly window length years (10 years) earlier, so a decision at any point in the walk-forward never conditions on more history than a fixed-length lookback allows, however far the walk-forward has progressed. Repropose and combinator-refit cadences are independent config parameters (we evaluate an annual repropose cadence with a 4-month refit cadence, §4); between two repropose events, the admitted library is held fixed and only the combinator is refit. Two flags independently control what a repropose event inherits from the immediately preceding one: whether the natural-language lesson log carries forward, and whether the admittedfactor library and its registry do. A separate mechanism, used only to isolate the effect of re-discovery itself from ordinary run-to-run LLM variance, lets a rolling arm’s very first re-discovery event clone an alreadycompleted static run’s initial library wholesale rather than independently rediscovering it, so the two arms trajectories are identical until the rolling arm’s own first library change.

![](images/c5a7fd15f71ac392a1701c54bdc70272cdd25a00926b48c32fda9ff85003e3a6.jpg)  
Figure 5: Illustrative repropose/refit timeline (k = 12, j = 4; §3.5), not the actual OOS schedule of §4. Top: SEADS’s carry-forward rolling—library, novelty reference set, and lessons persist across every repropose event. Bottom: baseline rolling—each repropose event re-invokes the baseline’s own driver at the new decision date, without SEADS’s specific cross-refresh mechanisms (library, novelty reference set, reflector lessons). Both rows share the identical repropose/refit tick structure; only what crosses between ticks differs.

## G Per-Baseline Fidelity Notes

We disclose porting fidelity per system rather than assert uniform faithfulness. Table 1’s Productivity metric is each baseline’s own admission count—native-gate, except where a baseline’s own evaluator has no look-ahead check of its own (RD-Agent(Q)), in which case it is the firewall-clean count instead.

AlphaAgent [Tang et al., 2025] proposes a market hypothesis and a factor expression together, constrained at generation time by a complexity cap on the formula’s size, then admitted on two independent criteria (largest-common-subtree novelty against the admitted zoo, and an LLM-judged hypothesis-formula consistency score) with no separate backtest-significance gate, matching the paper’s stated objective; its continuous regularization weights are never disclosed in the paper, so our port uses documented, reasonable substitutes rather than a literal reconstruction.

AlphaForge [Shi et al., 2025] is verified against the authors’ own released repository; its generativepredictive mining stage (a generator trained against a frozen predictor, not an adversarial/GAN setup) is replaced with random search over the same, math-verified operator grammar (consistent with the paper’s own random-initialization bootstrap, Algorithm 1), while its dynamic-combination stage keeps the paper’s real filterthen-regress structure: a rolling-window rank-IC >0.02 and rank-ICIR >0.2 threshold, falling back to the single best factor if none clears it, before the regression step. It makes zero LLM calls.

RD-Agent(Q) [Li et al., 2025c] is verified against the actual installed official package; its evaluator has no temporal-leakage check of its own, only execution success and near-duplicate detection, so our port runs our look-ahead firewall as a post-hoc audit rather than a pre-execution gate, to avoid misrepresenting what the original system actually checks, and reports native, firewall-audited, and firewall-clean views separately: in one matched-budget (300-candidate) run, its native evaluator admitted 57 candidates, of which the firewall subsequently flagged 18 (31.6%) for temporal look-ahead or cross-stock rank pooling, leaving 39 firewallclean—the view Table 1 reports.

QuantaAlpha [Han et al., 2026] is verified against its real released repository; its complexity/redundancy thresholds are taken directly from its source, a higher-fidelity tier than AlphaAgent’s necessarily-substituted ones, though its paper’s stated pool-admission rule is verified to not match what its own released code implements—we port the paper’s stated rule rather than the code’s weaker one, so QuantaAlpha remains a distinct comparison point in our results.

Beyond Prompting [Huang and Fan, 2026] is a close reconstruction from its paper’s Algorithm 1, since the paper releases no official code; its gate thresholds, including the Harvey–Liu–Zhu t>3.0 hurdle it imports [Harvey et al., 2016], are kept as published even where they differ from our own gate’s conventions, to stay faithful to what the paper actually specifies rather than to our own house style.

## H Formula-Complexity Case Study: Interactions vs. Additive Spreads

A single-run comparison surfaced a structural difference in what kind of formula each system tends to admit, independent of any performance number: SEADS’s admitted factors are overwhelmingly multiplicative interactions or ratios of two-to-three ranked characteristics, while RD-Agent(Q) and Beyond Prompting admit mostly flat additive/subtractive spreads of exactly two raw characteristics. Table 8 quantifies this over admitted factors pooled across SEADS’s locked-gate flagship replicates (5 runs each, Panel A and Panel B, deduplicated by name), against RD-Agent(Q)’s and Beyond Prompting’s own natively-admitted and promoted factors from matched-protocol comparison runs (§5.1). Classification is by static analysis of each admitted formula’s Python: the top-level operator combining the final factor assignment, tracing through intermediate variable assignments and stripping postfix method chains (.astype(...), .fillna(...)) to reach the underlying expression, not a manual read.

<table><tr><td>Structural type</td><td>SEADS</td><td>RD-Agent(Q)</td><td>Beyond Prompting</td></tr><tr><td>Multiplicative / gate</td><td>83.8%</td><td>21.1%</td><td>14.5%</td></tr><tr><td>Ratio</td><td>6.3%</td><td>11.8%</td><td>3.4%</td></tr><tr><td>Additive / subtractive</td><td>9.0%</td><td>50.0%</td><td>78.7%</td></tr><tr><td>Mixed / other</td><td>0.9%</td><td>17.1%</td><td>3.4%</td></tr></table>

Table 8: Structural type of every admitted factor, by system, static-analysis classification of the final formula assignment (Appendix H). SEADS’s percentages pool its locked-gate flagship replicates (5 runs each, Panel A and Panel B), deduplicated by name.

Representative formulas. SEADS examples below are real admitted factors, reduced to their underlying expression (stripping the groupby(’date’)...rank(pct=True) scaffolding each is actually stored and executed as, per the tracing procedure above); RD-Agent(Q) and Beyond Prompting store formulas natively as expressions, so theirs are shown as-is.

• SEADS (cheapness conditioned on low accruals): rank(sale bev)\*(1-rank(taccruals at))

• SEADS (long momentum conditioned on stable turnover): rank(ret 48 12)\*(1- rank(turnover var 126d))

• SEADS (three-variable example): rank(sale bev)\*(1-rank(spi at))\*(1-rank(ret 1 0))

• RD-Agent(Q): rank(cash at) - rank(debt at); gp gr1a - gp gr3a (unranked).

• Beyond Prompting: rank(gp at) - rank(at gr1); rank(fcf me) - rank(capx at).

Every SEADS example is a product of ranked legs that must each score well simultaneously; every baseline example assumes its two ingredients contribute independently and linearly.

Interaction terms over flat spreads. A flat difference of two raw characteristics is, for a linear downstream model, exactly spanned by including both characteristics separately — discovering it as its own factor adds nothing a linear combinator would not already learn on its own. SEADS’s TRIVIALITY RULE—a deterministic check applied at every complexity tier the Planner assigns a seed (§3.3.1)—is built on this reasoning and rejects plain sums/averages for the same reason; the Proposer’s rationale requirement additionally asks th model to state why combining these variables is stronger than either alone, which a flat spread does not naturally answer. We are optimizing for a different, arguably harder target than standalone statistical significance: incremental value to a portfolio that already holds the full raw characteristic panel, which is exactly what the novelty gate (§3.3.3) exists to protect.

Empirical illustration. Beyond Prompting’s admits illustrate the failure mode this guards against concretely: many hold one ingredient (a cash-flow-yield characteristic) fixed and vary the second across a long list of unrelated characteristics (fcf yield minus capex intensity, fcf yield minus asset growth, fcf yield minus interest burden, . . . ), producing a set with high within-admission correlation (mean pairwise |ρ|=0.281, effective independent count 3.46 of 94 nominal admits by a standard $n / ( 1 + ( n - 1 ) \bar { \rho } )$ diversification measure) and correlation with the raw reference panel itself, not because each is individually invalid but because the family is one idea restated many times. SEADS’s admitted sets have lower within admission correlation across its 5 Panel A flagship replicates (mean pairwise |ρ|=0.167–0.229, effective count 3.51–4.49 of 10–18 nominal admits).

## I Broader Related-Work Landscape

Factor-discovery lineage. Beyond the five re-implemented systems, earlier and adjacent factor-discovery work includes human–AI interactive mining [Wang et al., 2023], reinforcement-learning formula search [Yu et al., 2023], pre-LLM evolutionary mining [Cui et al., 2021], prompted factor conceptualization [Cheng and Tang, 2024], and semi-automated feature engineering [Wang et al., 2024c].

Adjacent LLM miners. A larger population of LLM-based miners each contributes one specific search or optimization mechanism: chain-based generation [Cao et al., 2025] (removed by arXiv administrators for a licensing-rights issue; we note its design but do not treat any of its reported numbers as verifiable), an MCTS miner with frequent-subtree avoidance to resist formulaic homogenization [Shi et al., 2026b], GFlowNet-based search [Chen et al., 2026a], on-graph evolution [Guo et al., 2026], RL screening [Jiang et al., 2025], evolutionary sparse portfolios [Luo et al., 2025], and reinforcement fine-tuning [Zhang et al., 2026]. A second group contributes an agentic or strategic mechanism instead: multi-agent debate [Duan et al., 2025], strategy-finding agents [Kou et al., 2025], a sandboxed execution framework [Shi et al., 2026a], market-logic generation [Weng et al., 2026], and agentic finance research more broadly [Lim et al., 2026].

Memory-based precedents. Skills-and-experience memory with failure constraints [Wang et al., 2026] and memory-driven hypothesis-to-code discovery [Liu et al., 2026a] are the closest precedents to our own memory tiers; §3.3.5 differentiates on the granularity and pipeline-stage specificity of what feeds back, not on the presence of memory itself.

Text-adjacent precedents. Some systems use price data with text as an idea seed only [Lin et al., 2026, Liu et al., 2026a], use news text directly [Wang et al., 2024b], use disclosures to predict firm-level returns rather than register a reusable, named characteristic [Choi et al., 2025], or use multi-agent debate for discretionary portfolio construction [Zhao et al., 2025]—distinct from our own two panels, both structured numerical data rather than text.

Evaluation infrastructure and realism. Evaluation infrastructure exists for formulaic alphas [Ding et al., 2026] and for trading-decision agents more broadly [Li et al., 2025a]; a survey reporting a median test window of only 1.3 years across agent papers surveyed [Ding et al., 2024] motivates the realism argument behind §3.5. Huang et al. [2026], evaluated on cryptocurrency rather than equity markets, is OOS-disciplined on a single static split with one modern backbone—the closest prior instance of §3.4’s standard we are aware of, absent the system-level backtesting of §3.5.

Broader agentic-AI context. General-purpose AI-scientist engines [Lu et al., 2024, Yamada et al., 2025, Novikov et al., 2025] and the broader LLM-agents-in-finance survey literature [Dong et al., 2025] situate AEAP within the wider landscape of autonomous research agents. Within AEAP but outside factor discovery, Koijen and Levy [2026], Cheng and Chin [2025] pursue agentic structural asset pricing and Xiao et al. [2025], Chen et al. [2026b], Miyazaki et al. [2026], Yu et al. [2024] pursue portfolio strategy, with Qu and Chen [2026] as complementary evaluation infrastructure rather than a portfolio-strategy system itself.

<table><tr><td>Role</td><td>Temp. Why</td><td></td></tr><tr><td>Planner, Proposer</td><td>0.7</td><td>breadth across seeds/hypotheses</td></tr><tr><td>Self-evolution</td><td>0.5</td><td>moderate variation on mutations</td></tr><tr><td>Implementation</td><td>0.2</td><td>reliable</td></tr><tr><td rowspan="3">Agent, reflector Alignment Checker</td><td rowspan="3">0.1</td><td></td></tr><tr><td>code/summaries</td></tr><tr><td>consistent judgments over</td></tr></table>

Table 9: Per-role LLM temperature; all other API parameters are identical across roles.

## J Reproducibility Notes

Leakage prevention and determinism. Headline results are drawn only from out-of-sample windows never touched by discovery or selection; the out-of-sample split is excluded from every gate decision in code, and the validation engine is a pure function of a candidate series, the in-sample window, and a threshold dictionary that never reads the out-of-sample split. The firewall, all statistical checks, duplicate/self-collapse fingerprinting, and the sign backstop are deterministic and seed-independent; only the LLM roles are stochastic.

Backbone. A single current LLM (§6) is used for every agentic role in every reported configuration. The engine separately supports resolving a release-dated backbone per historical decision date for a chronologicallyconsistent alternative protocol; no result in this paper uses that path, consistent with §6’s scoping decision.

Shared scoring and replicate reporting. All systems are scored under one shared portfolio-construction and metric implementation so that “Sharpe,” “admission rate,” and “OOS survival” are defined identically for every system; SEADS and ablation numbers are reported as means ± standard error over independent replicates (Table 1, Table 2), not single runs.

Engineering notes. Cost asymmetry motivates gate ordering (rank-IC rejects most candidates in milliseconds; partial-IC and Fama–MacBeth each run a per-month regression loop and are placed after the cheaper novelty lookup); the discovery loop supports resumable, checkpointed runs.

LLM configuration. Every role calls GPT-5.2 through the standard chat-completions API; no role has tool use, function calling, or web access exposed to it. Temperature is set per role rather than fixed globally, matching each role’s own need for diversity versus consistency (Table 9).

Code release. All source code required to conduct and analyze the experiments in this paper will be made publicly available upon acceptance, under a license permitting free use for research purposes.

Computing environment. Python 3.14, pandas 2.2, numpy 2.4, scikit-learn 1.9, scipy 1.17, on macOS; no GPU is used, since all LLM roles call a hosted API rather than running locally, and local computation is limited to lightweight data processing and statistics.

## K Metric Definitions

Rank-IC / ICIR. Per month, the cross-sectional Spearman correlation $\rho _ { m }$ between factor and next-month excess return over stocks with $\geq ~ 5$ valid pairs (months with $< ~ 1 2$ valid $\rho _ { m }$ are undefined, not zero); $\begin{array} { r } { \overline { { \rho } } ~ = ~ \frac { 1 } { M } \sum _ { m } \rho _ { m } , \mathrm { I C I R } ~ = ~ \overline { { \rho } } / \mathrm { s t d } ( \rho _ { m } ) \cdot \sqrt { 1 2 } , } \end{array}$ , and the rank-IC t-statistic $t ~ = ~ \overline { { \rho } } / ( \mathrm { s t d } ( \rho _ { m } ) / \sqrt { M } )$ . The locked gate thresholds on t alone, floored at 3.0 and rising above that floor via a live Bonferroni bound $t = \Phi ^ { - 1 } ( 1 - 0 . 0 2 5 / N ) ( \alpha = 0 . 0 5$ , family-wise error rate control) computed from the cumulative candidate count N so far—the only statistic in the gate this correction is applied to, since the Fama–MacBeth bar below is a fixed $t \geq 1 . 0$ regardless of N. Coverage is the fraction of in-sample months with a non-null candidate value (≥Min Coverage); regime consistency recomputes mean rank-IC on the Regime Lookback Years immediately preceding the in-sample window and requires sign agreement with it.

<table><tr><td>n (months)</td><td> $\widehat { S R } { = } 0 . 3$ </td><td>0.5</td><td>0.7</td><td>1.0</td></tr><tr><td>23</td><td>0.213</td><td>0.221</td><td>0.233</td><td>0.255</td></tr><tr><td>35</td><td>0.173</td><td>0.179</td><td>0.189</td><td>0.207</td></tr><tr><td>59</td><td>0.133</td><td>0.138</td><td>0.145</td><td>0.159</td></tr><tr><td>71</td><td>0.121</td><td>0.126</td><td>0.132</td><td>0.145</td></tr><tr><td>120</td><td>0.093</td><td>0.097</td><td>0.102</td><td>0.112</td></tr></table>

Table 10: Analytic Lo-(2002) SE(SRc ) by window length, including Panel B’s $n { = } 5 9$

<table><tr><td>System</td><td>(A/B)</td><td>Productivity Performance (A/B)</td><td>Novelty (A/B)</td><td>Spanning (A/B)</td></tr><tr><td>AlphaForge</td><td>0.45/0.20</td><td>.036/.033</td><td>.025/.028</td><td>3.325/.342</td></tr><tr><td>RD-Agent(Q)</td><td>2.55/2.81</td><td>.025/.028</td><td></td><td>.017/.024.364/.295</td></tr><tr><td>AlphaAgent</td><td>2.66/2.54</td><td>.017/.023</td><td></td><td>.021/.025.260/.271</td></tr><tr><td>QuantaAlpha</td><td>3.38/2.35</td><td>.022/.027</td><td></td><td>.015/.019.290/.313</td></tr><tr><td>Beyond Prompting</td><td>3.91/3.30</td><td>.017/.014</td><td></td><td>.023/.025.343/.368</td></tr><tr><td>SEADS (ours)</td><td>1.43/1.26</td><td>.025/.021</td><td></td><td>.014/.017.257/.317</td></tr></table>

Table 11: Per-entry cross-replicate standard errors for Table 1, same systems, metrics, and panels.

Fama–MacBeth [Fama and MacBeth, 1973] t. Per month, $\begin{array} { r } { r _ { i } = a + \sum _ { k } b _ { k } c _ { k , i } + b _ { \mathrm { c a n d } } x _ { i } + \varepsilon _ { i } } \end{array}$ over the top-5 most-correlated controls plus the candidate (winsorized 1/99%, $\ge \operatorname* { m a x } ( \ddot { 3 } 0 , 3 \cdot 7 ) = 3 0$ stocks/month); $t = \mathrm { m e a n } ( b _ { \mathrm { c a n d } } ) / ( \mathrm { s t d } ( b _ { \mathrm { c a n d } } ) / \sqrt { T } )$

Partial-IC. The candidate is residualized each month on its top-5 most-correlated admitted-library columns via per-month OLS; the residual’s monthly-Spearman rank-IC t-statistic is gated. This top-5 set is computed once from a single shared correlation matrix and reused for novelty, partial-IC, and Fama–MacBeth.

Novelty, duplicate, and self-collapse. Novelty gates on mean-monthly Spearman < 0.7 against a designated reference panel plus the admitted library. Duplicate detection uses an exact fingerprint—(count, mean, std, 10th/90th percentile, a hash of the full rounded value array)—so it catches statistically identical candidates without relying on summary statistics alone, which rank-normalized series can share regardless of the underlying variable. Self-collapse flags a final factor correlating ≥ 0.95 with a plain rank of one of its own declared inputs.

Sharpe and its SE. Monthly decile long-short returns annualized by ${ \sqrt { 1 2 } } ;$ reported gross of costs, with the Lo-(2002) standard error $\mathrm { S E } ( \widehat { S R } ) \approx \sqrt { ( 1 + \widehat { S R } ^ { 2 } / 2 ) / n [ \mathrm { L o } }$ , 2002] and a 95% interval (Table 10 tabulates this by window length). OOS survival is the fraction of admitted factors with $\operatorname { O O S } | t | \geq 1 . 9 6$ and the same sign as their IS t. As a separate diagnostic, not a headline adjustment, we also compute a cost-adjusted return assuming 50bps per side, round-trip, scaled by each factor’s own realized monthly rank turnover.

Validation thresholds. Table 6 (Appendix D) gives the gate values used for every result in this paper, alongside each check’s execution order: coverage ≥ 0.5, Rank-IC t ≥ max(3.0, Bonferroni-dynamic), out-ofdecade regime sign-agreement, novelty $< 0 . 7$ , partial-IC $t \geq 1 . 0 ,$ , and Fama–MacBeth $t \geq 1 . 0 .$

Table 1’s cross-replicate standard errors. Table 11 gives the per-entry standard error (SD across 5 replicates, divided by $\sqrt { 5 } )$ for each of Table 1’s four metrics, same systems and panels.

## L Example Admits

Three admitted SEADS factors from the Panel A locked-gate flagship replicates (§5.1), each chosen to illustrate one property: strong in-sample/out-of-sample agreement, low correlation against the reference panel, and the multiplicative-interaction structure typical of SEADS’s admits (Appendix H). All three cleared every check in Table 6 (Appendix D); IS stats are from the in-sample window, OOS from the held-out window (§4), both from Table 12. These three are hand-selected to each demonstrate one property, not randomly drawn or representative of the admitted pool’s average statistics—Table 1’s Panel A mean per-factor OOS Sharpe is 0.25, well below any shown here.

<table><tr><td></td><td>Strong IS/OOS</td><td>Low corr.</td><td>Represent.</td></tr><tr><td>Coverage</td><td>97.8%</td><td>89.3%</td><td>85.0%</td></tr><tr><td>Rank-IC (ICIR)</td><td>0.024 (1.25)</td><td>0.031 (1.52)</td><td>0.016 (1.35)</td></tr><tr><td>Partial-IC t</td><td>2.45</td><td>4.77</td><td>3.25</td></tr><tr><td>FM t</td><td>2.65</td><td>3.36</td><td>2.61</td></tr><tr><td>OOS Sharpe</td><td>0.89</td><td>0.52</td><td>0.88</td></tr><tr><td>OOS t</td><td>2.16</td><td>1.27</td><td>2.14</td></tr><tr><td>OOS ann. ret.</td><td>13.1%</td><td>7.1%</td><td>7.3%</td></tr><tr><td>Max corr</td><td>0.662</td><td>0.449</td><td>0.697</td></tr></table>

Table 12: Gate and out-of-sample statistics for the three example admits below, one column each; Coverage, Rank-IC (ICIR), Partial-IC t, FM t, and Max corr are in-sample, computed at admission time, while the three OOS-labeled rows are out-of-sample.

Strong IS/OOS agreement: stable liquidity efficiency gate (replicate a, rich tier). Formula: (1-rank(bidaskhl 21d)) × (1-rank(zero trades 126d)) × Formula: (1-rank(bidaskhl\_21d))×(1-rank(zero\_trades\_126d))×

(1-rank(dolvol var 126d/(1+trail12m mean))). Tight spreads and low trading inactivity indicate low trading frictions and faster price discovery, lowering required returns; downweighting names with an unstable dollar-volume regime isolates the structural, not episodic, component. Correlates most with dolvol var 126d (Table 12).

Low correlation: distress amplified perf mispricing (replicate d, simple tier). Formula: rank(mispricing perf) × rank(o score). Performance-based mispricing is more likely sustained by limits-to-arbitrage and forced trading among financially weak (high O-Score) firms, so the subsequent correction should be stronger when both signals coincide. Correlates most with debt at, at 0.449 the lowest of any admit pooled across the 5 replicates (Table 12).

Representative structure: cash op profit x low tax payable growth (replicate b, simple tier). Formula: rank(cop bev) × (1-rank(txp gr1a)). Cash operating profitability is more informative when low growth in taxes payable reflects a persistent cash-tax advantage rather than a temporary accounting effect. A two-variable rank product, the structural form 83.8% of SEADS’s admits share (Appendix H); correlates most with txp gr1a (Table 12).

## M AI Acknowledgement

Language models are this paper’s experimental subject, not merely a tool used alongside it: every reported result comes from an LLM-driven system, SEADS and all five baselines alike, which call a large language model at every proposal, evaluation, and evolution step (§2, Appendix E). Separately, generative AI tools (large language models) were also used as an assistant in this project’s own process—for part of the supporting code development, and for portions of this manuscript’s preparation and editing. All experimental results, data, and claims were verified by the authors against the underlying code and run logs; the authors take full responsibility for the paper’s content.