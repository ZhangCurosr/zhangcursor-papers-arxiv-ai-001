# KAPPA: Checking the Proofs in Statistics Papers with Lean 4 and Language Model Agents

Zihua She Xiao Wang

Department of Statistics, Purdue University, West Lafayette, IN 47907, USA she8@purdue.edu, wangxiao@purdue.edu

## Abstract

Papers in statistical theory rest on long proofs that few readers check line by line. We present KAPPA (Kit of Agents for Proof-checking Papers with a proof Assistant), open-source software that organizes language model agents, run by Claude Code, to translate the proofs of a L<sup>A</sup>T X paper into the proof assistant Lean 4 with its library Mathlib and check them there, result by result. Each numbered result receives one of four verdicts: verified unconditionally; verified conditionally, with every cited or unproved input stated as a named hypothesis; issue recorded, with its place in the paper and the side at fault; or unfinished, with the reason. The kit follows the paper’s own proof step by step, rather than finding a new one. Separate agents write and review, each statement is compared with the paper before it is proved and fingerprinted afterwards, and every verdict rests on evidence that can be rerun without a language model. The kit runs unattended for hours, waits out usage limits, re-checks only what a revision changed and reports its cost by role, model and result. We describe the design, walk through a complete run, and apply the kit to four example papers with answer keys and to three published statistical papers, checked blind in their versions from before public corrections. In all three, the kit found the published error at the step that the correction names, together with further problems that we confirmed by hand. Given the same papers directly, the same language model working with Lean classified fewer results correctly on the longest paper and reported some statements narrower than the paper’s as proved; no verdict of the kit claimed more than its Lean evidence showed.

Keywords: proof checking, formal verification, Lean, language models, statistical theory, reproducibility.

## 1 Introduction

A theorem in a statistics paper usually rests on a proof in an appendix, often several pages long, that combines a few cited results with many routine steps. Readers and referees follow the main line of the argument, and errors in the routine steps can go unnoticed. Journals publish corrections when such errors come to light (three recent ones are Henzi and Ziegel 2022a; Arab, Lando, and Oliveira 2026; Livingstone, Betancourt, Byrne, and Girolami 2026). Two of these three leave the theorem true and repair its proof, one corrects a false statement, and in each the error sits in a single step that a careful reader could have checked.

A specialist can formalize such a result in a proof assistant such as Lean 4 (de Moura and Ullrich 2021) with the library Mathlib (The mathlib Community 2020), and a small trusted kernel then checks every step; formalizing one statistics paper in this way can take weeks of expert work. A language model reads a proof in minutes, but its answer is another informal argument that someone has to check.

KAPPA (She and Wang 2026), a kit of agents for proof-checking papers with a proof assistant, combines the two. For each numbered result of a LAT<sub>E</sub>X paper, agents translate the statement and the paper’s own proof into Lean, other agents review the translation, and the Lean kernel checks the proof. The outcome is one of four verdicts (Section 3). A result is verified when the paper’s proof compiles step by step. A cited lemma enters as a named hypothesis of the theorem. A problem is recorded once a second agent has confirmed it, preferably with a counterexample in Lean. Where the paper’s argument fails, the kit reports the failing step and leaves the repair to the authors. The report also says what a person still has to check, and where. For a verified result, this is the statement, which the report prints beside its place in the paper; for a conditional result, also the named inputs. An issue comes with the failing step, its class and its evidence, so that the authors know where a repair has to start.

Three features of the task shape the design. A Lean proof can be correct for a statement that difers from the printed one, so every statement is compared with the paper before it is proved and fingerprinted, which makes any later change visible (Section 5.2). Language model output difers from run to run, so every verdict is tied to evidence that stays the same: compiled files, axiom lists and dependency walks that can be rerun ofline (Section 8). Checking a research paper takes hours of agent work, so the kit runs unattended, survives interruptions and re-checks only what a revision changed (Section 5.6).

The software consists of agent definitions, rules and Bash and Python scripts for Claude Code (Anthropic 2026b), together with Lean commands for auditing. In spirit it is close to tools that check the reliability of statistical computations, such as accuracy (Altman, Gill, and McDonald 2007), which find problems and make the evidence inspectable. As infrastructure it resembles workflow tools such as BatchJobs (Bischl, Lang, Mersmann, Rahnenführer, and Weihs 2015) and Benchpress (Rios, Mofa, and Kuipers 2025), which run many jobs unattended and keep their state in files.

As an example, in the paper on Hamiltonian Monte Carlo of Section 6.3, a corollary is called “a direct consequence” of a theorem whose conditions include a bound on the growth of the gradient. The kit found that this bound fails for part of the family that the corollary covers. It proved in Lean that the bound fails exactly for $2 < \beta < \log _ { 2 } { \theta }$ and holds for larger β, and it recorded the gap at its place in the paper. The authors’ corrigendum, posted seven years after the paper and not available to the run, describes the same range.

Section 2 compares the kit with formalization by experts, with language model provers and with language model reviews of papers. Section 3 defines the verdicts and what each of them lets a reader conclude, and Section 4 follows a complete run on a short paper. Section 5 describes the design, Section 6 the case studies, and Section 7 compares the kit with a direct use of a language model and evaluates its cost, its fact check and its revisions. Section 8 explains what can be reproduced and how, and Section 9 states the limitations.

The kit is available at https://github.com/ziwaa-se/kappa under the MIT license. This paper describes version 1.3.1, with which all runs were made. Version 1.3.4 adds fixes for problems that these runs revealed, and the current release, 1.4.0, gives the kit its present name; earlier versions were released as the TeX Proof-Check Kit. It needs Claude Code, Lean 4.19.0 and the Mathlib revision pinned in the repository, Python 3.6 or later and a Bash shell. The replication materials are described in Section 8.

## 2 Related work

Formalization by experts. Large results have been formalized completely, among them the four color theorem (Gonthier 2008), the odd order theorem (Gonthier et al. 2013), the Kepler conjecture (Hales et al. 2017) and, in Lean, a central theorem of condensed mathematics (Scholze 2022). In probability, the central limit theorem has been formalized in Isabelle/HOL (Avigad, Hölzl, and Serafin 2017), and Mathlib now contains measure-theoretic probability with conditional expectation, independence and martingales. Projects of this kind are often organized by a blueprint (Massot 2025), a LAT X document whose numbered results are linked to Lean declarations, with a dependency graph that shows what has been proved. The kit uses the same organization, with one group of declarations per numbered result and the dependencies recorded. Its aim is narrower. A formalization project wants a general proof that a library can reuse, and may change the argument to get one. The kit asks whether the paper’s own argument works, and uses whatever definitions encode the paper faithfully, even if they are too special to reuse.

Language model provers. Systems that search for formal proofs with language models are evaluated mostly on library theorems and on competition and textbook problems, stated in Lean or other proof assistants (Polu and Sutskever 2020; Zheng, Han, and Polu 2022; Yang et al. 2023; Xin et al. 2024). Autoformalization translates informal statements into formal ones (Wu et al. 2022; Azerbayev et al. 2023), and informal proofs can guide the search for formal ones (Jiang et al. 2023). The kit uses the same methods but asks a diferent question. A prover asked for some proof of a statement may succeed with an argument unrelated to the paper’s, and a failed search tells the reader little about the paper. The kit holds the prover to the paper’s route, so that a step that does not go through becomes a finding about the paper, to be confirmed and classified. It treats the translation of the statement, the main risk of autoformalization, as the step that most needs review: every statement is compared with the printed text before any proving and fingerprinted afterwards. A general-purpose agent asked to check a paper with Lean already reports failing steps. Section 7.1 shows which guarantees it still lacks. Two recent agent systems formalize research papers: MerLean for papers in quantum computation (Ren, Li, and Qi 2026), and Theo for papers in theoretical computer science and discrete mathematics (Soltani Moakhar, Gholami, Springer, JafariRaviz, and Hajiaghayi 2026), which also reported a false lemma in one of its papers. Both admit cited results as Lean axioms. The kit states every such input as a named hypothesis of the theorem that uses it, so that a verdict names its conditions and the axiom check confirms that nothing else was assumed.

Language models reading papers. Language models can comment usefully on manuscripts (Liang et al. 2024), and a model asked to check a proof often finds a real error. But a reader has to redo the check to tell a real finding from a confident mistake. In the kit, a suspected error is recorded as an issue only after a second agent has confirmed it, preferably with a counterexample that Lean accepts, and a verified result comes with a proof that anyone can recompile.

Reliable computation and reproducible workflows. In statistical computing, accuracy (Altman et al. 2007) diagnoses numerical problems that can invalidate statistical analyses, and the reproducibility literature asks that every number in a paper can be regenerated from archived code and data (Gentleman and Temple Lang 2007; Peng 2011). The kit applies this standard to proofs by leaving evidence that others can rerun. Its driver, which runs many short sessions unattended and keeps all state in files, belongs with workflow tools such as BatchJobs (Bischl et al. 2015), GNU Make (Baker 2020) and Benchpress (Rios et al. 2025).

Table 1 summarizes the comparison. A verified result of the kit rests on the Lean kernel, as an expert formalization does. Its statement is reviewed by agents, the review is on record, and the report lists the statements of the main results so that the authors can read them against the paper.

<table><tr><td></td><td>Expert formalization</td><td>Language model prover</td><td>Language model review</td><td>This kit</td></tr><tr><td>What is checked</td><td>the theorem, by any proof</td><td>a given formal statement, by any proof</td><td>the text of the paper</td><td>the paper&#x27;s statements, by the paper&#x27;s proofs</td></tr><tr><td>Who writes the formal statements</td><td>experts</td><td>people who set the problem</td><td>nobody</td><td>agents, reviewed by other agents before proving</td></tr><tr><td>Evidence</td><td>Lean kernel</td><td>Lean kernel</td><td>an informal opinion</td><td>Lean kernel and recorded reviews</td></tr><tr><td>When a proof fails</td><td>the expert repairs it</td><td>little information</td><td>a comment that needs checking</td><td>a located and classified issue, or the step where the</td></tr><tr><td>Cited results</td><td>formalized or assumed</td><td>not applicable</td><td>read</td><td>check stopped explicit hypotheses, listed for a person</td></tr></table>

Table 1: Four ways of checking the proofs in a paper.
<table><tr><td>Verdict</td><td>Requirements</td></tr><tr><td>Verified unconditionally</td><td>The statement passed the review before proving; the whole statement is proved along the paper&#x27;s route; the gated statement is unchanged; the build is clean, only the three standard axioms of Lean are used and</td></tr><tr><td>Verified</td><td>no sorry is reachable; no external parameter or recorded debt remains; the final audit passed. As above, but the proof takes explicit external parameters (a cited</td></tr><tr><td>conditionally</td><td>lemma, a fact the paper asserts without proof, a clause of another result), each listed with the step that uses it.</td></tr><tr><td>Issue recorded</td><td>A problem in this result&#x27;s own statement or proof, confirmed independently: by a reviewed Lean counterexample, or by a second agent&#x27;s review of the reasoning. The record names the side at fault</td></tr><tr><td>Unfinished</td><td>(Table 3). No conclusion yet, with the reason (for example, a missing library interface); not counted against the paper.</td></tr></table>

Table 2: The four verdicts.

## 3 Verdicts and what they mean

Every numbered result of the paper (theorem, proposition, lemma, corollary) receives exactly one of the verdicts in Table 2. The rules are written out in the kit’s VERDICTS.md, and a validator checks every recorded verdict against the evidence that must back it (Section 5.4).

Issues are classified by who is at fault (Table 3), because authors need to know what to change. An ERROR means that the printed conclusion is false. A PAPER\_ARGUMENT\_GAP means that the justification given for a step fails while the conclusion may well be true. The formalization-side classes concern only the Lean encoding. Steps that the paper leaves implicit and that any careful reader would fill in the same way are recorded as FAIR\_IMPLICIT\_STEP and proved.

Inputs that the paper cites or uses without proof are written as explicit Lean propositions, def

<table><tr><td>Class</td><td>Meaning</td></tr><tr><td colspan="2">The paper</td></tr><tr><td>ERROR</td><td>The printed conclusion is false; a counterexample exists.</td></tr><tr><td>PAPER_ARGUMENT_GAP</td><td>The justification given for a step does not hold; the conclusion may still be true.</td></tr><tr><td>PAPER ASSUMPTION GAP</td><td>A step needs a hypothesis that the paper does not state and that cannot be derived.</td></tr><tr><td>EXPOSITION_GAP</td><td>The conclusion is true, but some range or case is not covered.</td></tr><tr><td>DOMAIN_CLARIFICATION</td><td>The objects are ambiguous, and the conclusion depends on the reading.</td></tr><tr><td colspan="2">The formalization (not counted against the paper)</td></tr><tr><td>ENCODING_ERROR</td><td>The drafted Lean statement does not say what the paper says.</td></tr><tr><td>TRANSLATION BLOCKED</td><td>A needed interface is missing from the pinned library.</td></tr></table>

Table 3: Issue classes.

Tool : Prop, and appear as hypotheses of the verified theorem, never as axioms. Each is labeled with its provenance: (i) cited by the paper, (ii) asserted without proof, (iii) used but never stated, (iv) a clause of one of the paper’s own results whose proof has a recorded issue, (v) a standard fact with no interface in the pinned library, and (p) a clause of a predecessor result that is not proved yet, which is reconnected once it is.

The reports state what each verdict lets a reader conclude. A verified result means that the Lean statement, which reviewers compared with the paper, is proved from the listed hypotheses. The faithfulness of the translation rests on that review, and the report lets a reader repeat it. A conditional result is as sound as its listed hypotheses, which the worksheet asks a person to check. The verdict stays conditional even after a person has checked the hypotheses and the check is recorded. An unfinished result records where the check stopped and why.

## 4 A first run

We follow a two-lemma note, distributed with the kit as examples/two-lemmas-demo, through a complete run. Its first lemma states $a ^ { 2 } + b ^ { 2 } \geq 2 a b$ . Its second states that $x ^ { 2 } \geq x$ for every real x, with the proof “the two factors x and x − 1 always have the same sign”.

## 4.1 Setup

The commands below install the kit and a project for one paper. Here and below \$ is the shell prompt.

\$ git clone https://github.com/ziwaa-se/kappa

\$ kappa/install.sh mypaper-check --pinned

\$ cd mypaper-check && lake exe cache get

\$ scripts/pv\_lean.sh build PVTools

The third command downloads Lean and the compiled Mathlib once, and several projects can share one installation. In a guided setup, a Claude Code chat runs these steps and explains them. The paper’s LAT<sub>E</sub>X files, its PDF and its .aux file, which gives the exact numbers of the results, go into

source/. Unattended runs also need the project’s permission file. It allows the kit’s own commands and forbids network access and edits to the paper (Section 5.6).

\$ cp .claude/settings.example.json .claude/settings.json

## 4.2 Running the check

The check can run in a Claude Code chat, started with /pv-setup and /pv-run, or unattended from a terminal:

\$ scripts/pv\_drive.sh --setup source/demo.tex --auto-resume --report

The driver starts a fresh agent session for each batch of at most three results, records everything in files, and continues by itself after a usage limit. When it stops, the progress table shows the verdicts:

\*\*2 numbered results: 1 verified (unconditional) · 0 verified (conditional)   
· 1 issue recorded · 0 unfinished.\*\*   
| No. | ID | Verdict | Conditions / issues   
| Lemma 1 | L01 | Verified (unconditional) | ... no external parameters ...   
| Lemma 2 | L02 | Issue recorded | ... false on 0<x<1, witness   
x=1/2, confirmed by an   
independently reviewed Lean   
counterexample (B1).

Columns are shortened here. Lemma 1 is proved along the paper’s route, with the two steps of the paper as two obligations:

```lisp
theorem L01_main (a b : <sup>R</sup>) : a ^ 2 + b ^ 2 ≥ 2 * a * b := by
have h1 : a ^ 2 + b ^ 2 - 2 * a * b = (a - b) ^ 2 := L01_S1 a b
have h2 : (a - b) ^ 2 ≥ 0 := L01_S2 a b
have h3 : a ^ 2 + b ^ 2 - 2 * a * b ≥ 0 := by rw [h1]; exact h2
linarith
```

For Lemma 2 the prover could not justify the “same sign” step and stopped at it, as the rules require. The issue checker then encoded the printed statement and refuted it in Lean:

def PrintedLemma2 : Prop := ∀ x : <sup>R</sup>, x ^ 2 ≥ x

theorem not\_printedLemma2 : ¬ PrintedLemma2 := by   
intro h   
have h2 := h (1 / 2 : <sup>R</sup>)   
norm\_num at h2

A second agent checked that PrintedLemma2 is the printed claim, no stronger, and that x = 1/2 satisfies every hypothesis of the paper. The issue was recorded after this check.

## 4.3 What the run produces

The project then holds the records listed in Table 4. The report, written for the authors, contains a summary table, the verdict of every result with its evidence, every external input with what a person should check, the issues with their places in the paper, and a list of what remains for a person to check. Before it is delivered, an independent agent checks every statement in it against the records. In this run it found nine problems in the first draft, among them a suggested fix for Lemma 2 that no review had made (Section 7.3).

<table><tr><td>File</td><td>Contents</td></tr><tr><td>verification/RESULTS.json</td><td>The state of every result: stage, verdict, conditions, evidence paths.</td></tr><tr><td>verification/audits/</td><td>Gate, final-audit, mechanical-check and issue reports, one per review.</td></tr><tr><td>pv_work/*.lean</td><td>Statement files, proofs, counterexamples; each compiles at the pinned versions.</td></tr><tr><td>verification/BLOCKERS.md</td><td>Recorded issues, with places in the paper and classes.</td></tr><tr><td>verification/report/</td><td>The author report, a list of files to check, and a worksheet for the external inputs.</td></tr></table>

Table 4: Main records of a project.
<table><tr><td>Mechanism</td><td>Guarantee</td><td>Section</td></tr><tr><td>Statement review before proving; fingerprints</td><td>The statement proved has been compared with 5.2 the paper, and it cannot change unnoticed</td><td></td></tr><tr><td>Proof along the paper&#x27;s route; failing steps kept as gaps</td><td>A verdict is about the paper&#x27;s argument, not about some other proof</td><td>t 5.2</td></tr><tr><td>Inputs as named hypotheses, with provenance</td><td>Nothing is assumed silently; a conditional verdict says what remains to check</td><td>3</td></tr><tr><td>Separate writers and reviewers; issues confirmed by a second agent, with a counterexample when possible</td><td>No result is passed, and no problem recorded, 5.1 on one agent&#x27;s word</td><td></td></tr><tr><td>Mechanical audit and validator</td><td>The verdict agrees with its evidence: axioms, 5.3, 5.4 closure, hypotheses used, conditions</td><td></td></tr><tr><td>All state in files; offline replay</td><td>Every verdict can be rechecked without a language model</td><td>5.4, 8</td></tr><tr><td>Definitions by identifier; cascade decided by fingerprints</td><td>A revision redoes only what changed, and an issue is closed only by a new audit</td><td>5.5</td></tr><tr><td>Independent fact check of the report</td><td>Every claim in the report is checked against the records</td><td>7.3</td></tr></table>

Table 5: The mechanisms of the kit and what they guarantee.

## 5 Design and implementation

Table 5 lists the mechanisms of the kit and what each of them guarantees about a verdict. The subsections below describe them, and Section 7.1 shows what a direct use of a language model lacks without them.

## 5.1 Roles and independence

The main Claude Code session acts as coordinator. It assigns work, checks outputs, keeps the records and writes the reports, and it leaves all proving and reviewing to subagents, each with its own context (Figure 1 and Table 6). Writers and reviewers are always diferent agents, so no agent reviews its own work. The reviewing roles run on the stronger of two models, the writing roles on the cheaper one, and a proof that gets stuck is retried on the stronger model.

![](images/f345e39839e25f216c5e41ec814c46b74a5d5436a1a07b6570149269744fd252.jpg)

Figure 1: The three stages of a check and the agents of each. Shaded boxes are reviews by agents other than the writer. The architect lists the results; every result then passes through drafter, gate, prover and auditor, with the issue checker called when a step of the paper is not justified; the compiler and the fact check produce the report.
<table><tr><td>Role</td><td>Model</td><td>Writes</td></tr><tr><td>architect</td><td>Sonnet</td><td>inventory of results, dependencies, citations</td></tr><tr><td>drafter</td><td>Sonnet</td><td>Lean statement and step obligations</td></tr><tr><td>gate</td><td>Opus</td><td>review of the statement, before proving</td></tr><tr><td>prover</td><td></td><td>Sonnet, then Opus the proof, along the paper&#x27;s route</td></tr><tr><td>auditor</td><td>Opus</td><td>final audit; counterexample and reasoning reviews</td></tr><tr><td>issue checker</td><td>: Opus</td><td>ruling on a suspected issue; counterexamples</td></tr><tr><td>compiler</td><td>Sonnet</td><td>material for the report (read-only)</td></tr><tr><td>fact-check</td><td>Opus</td><td>check of the report against the records (read-only)</td></tr></table>

Table 6: Roles. The coordinator runs on Sonnet.

Each role is defined in a Markdown file that carries a short rule card, the excerpts of the rules that the role needs, each tagged with its source section, so that an agent reads only the rules it uses. A script checks that every cited section still exists. Rule cards and the model choices together cut the cost of a run by about three quarters without changing a verdict (Section 7.2).

## 5.2 The pipeline for one result

Figure 2 shows the stages a result passes through. The drafter writes two files: a statement file with the definitions that the statement needs, and a draft with the main theorem and one obligation per step of the paper’s proof, each with sorry in place of a proof. The gate then compares the draft with the paper: hypotheses, quantifiers, domains, constants, and whether every step corresponds to a sentence of the proof. A statement that fails goes back to the drafter.

![](images/578242e27b2379a15f6bd71a3731536086e08de65dd6ed3009659a6f46263cf0.jpg)  
Figure 2: The stages of one result. Shaded stages are reviews by agents other than the writer; the gate records fingerprints of the statement, which the final audit compares. A result whose predecessors are not yet proved uses their clauses as parameters and is reconnected later (not shown).

When the gate passes a statement, it records a fingerprint of every declaration: a hash of the type, a hash of the value for definitions, and the binder annotations. The prover then proves the obligations along the paper’s route. Where the paper gives no valid justification for a step, the prover leaves that step as a gap, recorded separately, and goes on to check the remaining steps. After at most two proof sessions per pass, the final auditor recompiles everything privately, compares the fingerprints with the gate’s, and runs the mechanical checks of Section 5.3. A suspected problem in the paper goes to the issue checker, whose ruling, with a counterexample when one can be built, is reviewed by an auditor before it is recorded.

A result that depends on another result not yet proved can start as soon as the other result’s statement has passed its gate. The clause it uses enters as a predecessor parameter (p), and once the predecessor is proved, a reconnection replaces the parameter by the predecessor’s theorem. The reconnection is audited like a proof.

## 5.3 Mechanical evidence

The checks that decide a verdict are deterministic. A Lean file of the kit, PVTools.lean, provides commands that list the axioms a declaration depends on, walk its dependency closure, report which hypotheses its proof uses, and print fingerprints. The auditor runs all of them with one script, pv\_audit\_mech.py, which writes a report of twelve numbered items: the compile check, a private rebuild with identical output, the integrity of the audit tool itself, the fingerprints against the gate snapshot, the axioms, the absence of sorry in the closure, the project constants reached with the audit record that covers each, the helper declarations to review, hypothesis usage, the conditions traced from the gated theorems, and who last wrote the audited files.

A condition listed in the records need not be used by the final proof, because a hypothesis can stay in a signature after the step that needed it was rewritten. The script therefore walks each gated theorem and reads its hypothesis usage, and it treats an external input as a condition only when a hypothesis that the proof uses mentions it. Inputs mentioned only by unused hypotheses are listed as formal, and inputs registered for the result but never reached are flagged. The auditor confirms the list, and the final audit ends with a machine-readable line such as CONDITIONS: P01\_NesbittInequality, from which the records are written.

## 5.4 Records and validation

All state is in files, so that any session can stop at any time and a new one can continue. The central file, RESULTS.json, holds every result’s stage, verdict, conditions and evidence paths. A validator checks every verdict against what must back it. A verified verdict needs a gate report and a final audit whose last lines say PASS, a main theorem that exists in the file named, no gap, and conditions that agree with the audit’s CONDITIONS line. An issue needs a recorded problem of this result, with a class on the paper’s side and a confirming review. The coordinator writes records only through commands of pv\_status.py, which fill in evidence paths from the latest reports and write the audit log, so that records are never copied by hand.

## 5.5 Revisions

When the authors correct the paper, the kit compares the new version with the copy it kept and redoes only what the change afects. A result is redone when its statement, its proof or a definition it uses changed. Definitions are compared by identifier: every shared definition has an identifier and an exact line range, and every result lists the definitions it uses, so that an unchanged formula next to a changed one does not afect its users.

The results that use a redone result need more care. A result consumes only the statements of the results it imports. If a redone result passes its gate with the same statement as before, every result that imports it still consumes exactly what it consumed before, so its proof remains valid and only needs to be compiled against the new module. The kit therefore marks the direct dependents “re-check pending” and decides how to treat them once the redone result has passed its gate again. If the fingerprints of its definitions and main theorem are unchanged, the dependents are rebuilt and checked mechanically by an auditor. If they changed, the dependents are redone, and their own dependents are marked in turn. An earlier issue is resolved only by a new final audit of the revised result. The report then shows, for each re-checked result, the verdict before and after.

## 5.6 Unattended operation

A long coordinator session becomes expensive, because each step re-reads the whole history. The driver pv\_drive.sh therefore runs the check in short headless sessions of at most three results each, passing all state through the records. It stops when every result has a verdict or a recorded reason, or when a session makes no progress, and its last line and a status file name the reason. With –auto-resume it reads the reset time from the account’s usage-limit message, sleeps until then and continues. With –report it writes the report at the end.

Unattended agents need permissions that a person cannot grant on the spot. The project’s permission file allows the kit’s own scripts, in every form in which the agents call them, and refuses everything else, including network access and edits to the paper and to the rules. A hook checks every write and every shell command against the role that makes it: nobody writes outside the project, the coordinator writes no proof and no review, and reviewers change nothing they review. The hook also logs each write with its role, which is how the mechanical report knows who wrote the audited files. After each session the driver counts the permission refusals in the transcripts and stops if commands that should have run were refused.

## 5.7 Cost accounting

A script, pv\_usage.py, reads the transcripts of all sessions and agents and sums their token use by role, model and result, at list prices. On a subscription plan the figures measure how fast the usage limit is consumed. For the driver’s sessions, the script compares its estimate with the cost that Claude Code itself reports, per model. The two agree when the price table is current.

## 6 Case studies

The case studies test whether the kit reaches the right verdicts and whether it finds errors known to exist in published work. Four short example papers were written for this purpose, each with an answer key, and together their keys call for every verdict and every issue class that concerns the paper; Table 7 shows where each verdict, class and mechanism of Section 5 occurred in the runs. Three published papers with public corrections test the kit on real proofs whose errors other people found.

## 6.1 How the case studies were run

Answer keys first. For each example paper, and for each published case, we wrote down before the run what a correct check should report for every result: the verdict, the issue class or the provenance of the inputs, the outcomes that are also acceptable and those that are not. Where the classification is ambiguous (for instance, a step that both needs an unstated hypothesis and asserts something false), the key accepts both classes and says why.

Blind runs. Each run took place in its own project folder that contained only the paper. The answer keys, the published corrections and the repository of the kit were kept elsewhere, the project’s permission file denied the agents access to them, and network access was denied as for every run. After each run, a script (examples/tools/blind\_audit.py) scanned every transcript, of the coordinator and of all subagents, for access to those locations, mentions of an answer file and network tools. Over 443 transcripts with 11 341 tool calls it found none.

Released version. All runs used version 1.3.1 as released (commit 338b9d9), with the same Lean environment, the complete compiled Mathlib, the unattended driver with –auto-resume –report, and the models of Section 7.2. A pilot run of the first example on the pre-release version is reported separately.

Scoring. A result agrees with the key when its verdict and classification are the expected ones or among those the key accepts. It is found with a diferent class when the kit reports the problem under another class, and missed or a false alarm otherwise. We checked by hand every finding that the key did not anticipate.

Ofline confirmation. For every run, the replay script of Section 8 rebuilt the archived Lean files at the pinned versions and reran the mechanical audit, without a language model. Every verdict whose evidence is a Lean file was confirmed this way, in two to thirteen minutes per run on the machine described in the Computational details. Two issues rest on a reviewed argument, and the replay has no file to rebuild for them.

## 6.2 Example papers with known answers

Table 7 lists the verdicts, classes and mechanisms that the runs of this section and the next exercised, and where. The four papers were written to exercise them, and the table records the paths that the runs actually took.

<table><tr><td>Feature</td><td>Where it occurred</td></tr><tr><td>Verified unconditionally</td><td>E1 2.1, 2.2, 2.4; E2 2.1, 2.3, 2.5; E3 2.1, 2.4; E4 2.1; B 2.5, 4.12, 5.3; C 5.15</td></tr><tr><td>Conditional on a cited result, (i)</td><td>E1 2.3; E4 2.2, 2.3; A 3.3</td></tr><tr><td>Conditional on an asserted fact, (ii)</td><td>A 3.4; C 5.14</td></tr><tr><td>Conditional on a clause of a result with an E3 2.3; B 3.3; C 5.17, 5.14, 2.2 issue, (iv)</td><td></td></tr><tr><td>Conditional on a standard fact missing from the library, (v)</td><td>C 5.14, 2.2</td></tr><tr><td>ERROR with a Lean counterexample</td><td>E1 2.5; E2 2.4; A 3.5; B 2.8, 4.8</td></tr><tr><td>PAPER_ARGUMENT_GAP</td><td>E1 2.6; E2 2.6; E3 2.2; A 3.1, 3.2; B 3.1, 3.2; C 5.16, 2.3</td></tr><tr><td>PAPER_ASSUMPTION_GAP</td><td>B 2.3, 4.6, 4.10, 5.2</td></tr><tr><td>DOMAIN_CLARIFICATION</td><td>E2 2.2; B 3.2</td></tr><tr><td>Fair implicit step filled in, no issue</td><td>E1 2.2, 2.4; E2 2.3; E4 2.3</td></tr><tr><td>Unfinished, BUDGET_EXHAUSTED, then a second pass</td><td>E4 2.3</td></tr><tr><td>Gate rejects an unfaithful draft</td><td>B 4.6</td></tr><tr><td>Gate requires amendments before proving</td><td>E2 2.3; B 2.5, 3.2, 4.12; A, C several</td></tr><tr><td>Review rejects an issue ruling, which is then revised</td><td>B 3.2</td></tr><tr><td>Traced hypothesis usage removes a formal condition</td><td>E3 2.4</td></tr><tr><td>Revision: reset of changed results only, definitions compared by identifier, issue resolved by a new audit</td><td>E3, A, B, C, second round</td></tr><tr><td>Reconnection of a clause after its predecessor is verified</td><td>B 3.3, C 5.17 (second round)</td></tr><tr><td>Formalization-side ENCODING_ERROR, stop as TRANSLATION_BLOCKED, nothing counted against the paper</td><td>C 2.2 (second round)</td></tr><tr><td>Unattended driver waiting out a usage limit</td><td>all runs</td></tr><tr><td>Independent fact check of the report</td><td>all runs</td></tr><tr><td>Not exercised</td><td>EXPOSITION_GAP as a final verdict; (iii); a suspected issue left unconfirmed</td></tr></table>

Table 7: Where each verdict, class and mechanism occurred in the case studies (E1–E4: example papers; A–C: published cases; first round unless stated).

<table><tr><td></td><td>No. Answer key</td><td>Run</td><td>Score</td></tr><tr><td>2.1</td><td>unconditional</td><td>unconditional</td><td>agrees</td></tr><tr><td>2.2</td><td>unconditional; the expansion is a fair implicit step</td><td>unconditional; the expansion recorded agrees as a fair implicit step</td><td></td></tr><tr><td>2.3</td><td>conditional on the cited inequality, (i)</td><td>conditional on Popoviciu&#x27;s inequality, (i)</td><td>agrees</td></tr><tr><td>2.4</td><td>exposition gap (case s = 0), or unconditional with that case filled in</td><td>unconditional; the case s = 0 ruled a fair implicit step and filled in</td><td>accepted</td></tr><tr><td>2.5</td><td>ERROR, with a counterexample</td><td>ERROR; counterexample n = 2, x = (1, −1)</td><td>agrees</td></tr><tr><td>2.6</td><td>conditional on the clause of 2.5, (iv)</td><td>PAPER_ARGUMENT_GAP in its own proof, differs with 2.5&#x27;s issue upstream</td><td></td></tr></table>

Table 8: The first example paper: the answer key, written before the run, and the verdicts of the run. The run took 82 minutes and cost \$14.54.

## 6.2.1 Sample variance: Errors, gaps and inherited conditions

The first paper collects six elementary facts about the sample mean x¯ and the sample variance $s ^ { 2 }$ of n observations. Two are identities. One bounds $s ^ { 2 }$ for data in an interval and rests on a cited inequality (Popoviciu’s). Samuelson’s inequality $| x _ { i } - \bar { x } | \leq ( n - 1 ) s / \sqrt { n }$ is proved by dividing by s, without the case $s = 0$ . The lemma $\begin{array} { r } { s ^ { 2 } \leq \frac { 1 } { n } \sum _ { i } x _ { i } ^ { 2 } } \end{array}$ is false, because its proof divides by n where it should divide by n − 1, and the last result, $s ^ { 2 } \leq 2$ when all $| x _ { i } | \le 1$ , is true but proved from the false lemma. Table 8 compares the run with the answer key.

The Lean file that refutes Lemma 2.5 states the printed clause, the instance and the failure of the conclusion. Abridged, it reads

def PrintedClause : Prop :=   
∀ {n : <sup>N</sup>} (\_hn : 2 ≤ n) (x : Sample n), Conclusion x   
noncomputable def x0 : Sample 2 := ![1, -1]   
theorem x0\_var : sampleVar x0 = 2 := by   
simp [sampleVar, sampleMean, x0, Fin.sum\_univ\_two]; norm\_num   
theorem counterexample : ¬ PrintedClause := by   
intro h   
exact x0\_not\_conclusion (h (by norm\_num) x0)

where Conclusion x is the gated statement $s ^ { 2 } \leq \textstyle { \frac { 1 } { n } } \sum _ { i } x _ { i } ^ { 2 }$ and x0\_not\_conclusion shows that it fails at x0, where the right-hand side is 1. The same file also refutes the weaker version of the lemma that Corollary 2.6 consumes, with the extra hypothesis $| x _ { i } | \le 1$ , because $x _ { 0 }$ satisfies it.

Five verdicts agree with the key or with an outcome it allows. For 2.4 the kit chose the more lenient of the two allowed outcomes: the missing case is trivial, both sides being zero, and the kit filled it in. For 2.6 the kit went further than the key. The proof of the corollary reads “By Lemma 2.5, $\begin{array} { r } { s ^ { 2 } \leq \frac { 1 } { n } \sum _ { i } x _ { i } ^ { 2 } \leq 1 \leq 2 " } \end{array}$ . The final audit verified the corollary conditionally on the clause of 2.5, as the key expects. While compiling the report, the compiler noticed that the audit had asked for a ruling on the intermediate claim $s ^ { 2 } \leq 1$ and that none had been made. The coordinator sent the claim to the issue checker, which refuted it at the same $x _ { 0 }$ and argued that the issue belongs to the corollary itself: the corrected lemma gives $\textstyle s ^ { 2 } \leq { \frac { n } { n - 1 } }$ , so no correction of Lemma 2.5 makes the printed chain true. We checked this by hand and agree with the kit. The key had excluded an issue because the corollary’s proof seemed fine apart from the cited lemma. That was too generous, but the score stays “difers” because keys are not changed after a run. In the pilot run on the pre-release version the corollary received the key’s verdict, so the classification of this borderline step varied between runs. Both reports tell the authors the same thing: the conclusion is not shown false, but the paper’s proof does not establish it.

## 6.2.2 Proportions and sums: Ambiguity, unproved facts and a quantifier trap

The second paper studies $V _ { n } ( p ) = p ( 1 - p ) / n$ , the variance of a sample proportion, for $p$ in “the unit interval” I, and the sample variance of a sum. Its six results raise the questions that a careful referee asks about wording. The paper never says whether I contains its endpoints. Every result holds under both readings except Lemma 2.2, $V _ { n } ( p ) > 0$ , which fails at $p = 0$ and $p = 1$ . Proposition 2.3, midpoint concavity of ${ \sqrt { V _ { n } } } ,$ rests on a fact that the paper calls well known and leaves unproved, so the expected verdict is conditional on an input of provenance (ii). Lemma 2.4, $V _ { n } ( p ) \leq p / ( 4 n )$ is false $( p = 1 / 2 )$ . Proposition 2.5 asserts a constant C with $V _ { n } ( p ) \leq C / n$ for all n. A draft that lets C depend on n would be easy to prove and wrong, which tests the gate on the order of the quantifiers. The proof of Proposition $2 . 6 , s ^ { 2 } ( x + y ) \leq 2 s ^ { 2 } ( x ) + 2 s ^ { 2 } ( y )$ , starts from an equality that holds only when the sample covariance of x and y vanishes, a hypothesis the paper does not state. The conclusion itself is true.

Five of the six verdicts agree with the key (Table 9). For Lemma 2.2 the drafter chose the open reading, because the proof itself says that “both $p$ and $1 - p$ are positive”, and noted that the lemma is false under the closed one. The gate agreed, with the reason that the kit’s “representation choices must not manufacture a paper error”, and asked for the ambiguity to be recorded. The open reading was proved, the issue checker refuted the closed reading at $p = 0$ , and the issue was recorded as a DOMAIN\_CLARIFICATION, with the note that the authors must say which interval they mean. The false Lemma 2.4 was refuted at $n = 1 , p = 1 / 1 0$ , and the issue record adds what the authors need in order to repair it: the bound holds on (0, 1) exactly when $p \geq 3 / 4$ . For Proposition 2.5 the gated statement puts the constant first,

def Conclusion : Prop :=

$$
\exists \texttt { C } : \mathbb { R } , \ \forall \texttt { n } : \mathbb { N } , \ 1 \leq \texttt { n } \to \forall \texttt { p } : \mathbb { R } , \ \texttt { V n p } \leq \texttt { C } / \ ( \texttt { n } : \mathbb { R } )
$$

and the gate’s report says why: “uniformity in n and $p$ is the content of the proposition”. The drafter got the order right at the first attempt, so the gate had nothing to reject here. Case B shows a rejection (Section 6.3). For Proposition 2.6 the kit refuted the first step of the proof, $s ^ { 2 } ( x + y ) = s ^ { 2 } ( x ) + s ^ { 2 } ( y )$ , at $x = y = ( 0 , 1 )$ , recorded a PAPER\_ARGUMENT\_GAP, and said explicitly that the conclusion itself is not refuted.

The one disagreement comes from a rule that version 1.3.1 left open. The drafter first encoded the “well known” concavity as an input of provenance (ii), as the key expects. The gate then derived the fact from Mathlib in a probe and recommended proving it, and the final audit classified the sentence as a fair implicit step. Version 1.3.1 allowed both readings. We prefer the key’s, because a reader of the report should see that the paper asserts the fact without an argument, and version 1.3.2 keeps an explicitly asserted fact as an input even when the library proves it. The verdict itself is correct, since the concavity is true and its Lean proof is checked, and the report’s table of boundary cases records the decision. Only the summary table shows the result without conditions.

<table><tr><td></td><td>No. Answer key</td><td>Run</td><td>Score</td></tr><tr><td>2.1</td><td>unconditional</td><td>unconditional</td><td>agrees</td></tr><tr><td>2.2</td><td>DOMAIN_CLARIFICATION</td><td>DOMAIN_CLARIFICATION; open reading proved, closed reading refuted</td><td>agrees</td></tr><tr><td></td><td>2.3 conditional on the asserted concavity, (ii)</td><td>unconditional; the concavity proved from Mathlib</td><td>differs</td></tr><tr><td>2.4</td><td>ERROR</td><td>ERROR; counterexample n = 1,  $p = 1 / 1 0$ </td><td>agrees</td></tr><tr><td>2.5</td><td>unconditional, constant before n</td><td>unconditional, constant before n</td><td>agrees</td></tr><tr><td></td><td>2.6 PAPER_ASSUMPTION_GAP or PAPER_ARGUMENT_GAP</td><td>PAPER_ARGUMENT_GAP; the first step refuted</td><td>agrees</td></tr></table>

Table 9: The second example paper. The run took 75 minutes and cost \$14.74.

## 6.2.3 Linear shrinkage: A revision, followed step by step

The third paper takes the estimator cX of a mean $\theta \neq 0$ from one observation X with variance $\sigma ^ { 2 }$ , whose risk is $R ( c ) = c ^ { 2 } \sigma ^ { 2 } + ( 1 - c ) ^ { 2 } \theta ^ { 2 }$ , and shows that the oracle factor $c ^ { * } = \theta ^ { 2 } / ( \theta ^ { 2 } + \sigma ^ { 2 } )$ attains the minimal risk $\rho = \theta ^ { 2 } \sigma ^ { 2 } / ( \theta ^ { 2 } + \sigma ^ { 2 } )$ . Lemma 2.1 completes the square. Lemma 2.2 states $\rho \le \operatorname* { m i n } \{ \theta ^ { 2 } , \sigma ^ { 2 } \}$ , which is true, but its proof says that a harmonic mean is at most the smaller of its two arguments, and derives the false intermediate bound $\rho \le { \frac { 1 } { 2 } } \operatorname* { m i n } \{ \theta ^ { 2 } , \sigma ^ { 2 } \}$ ; with $\theta ^ { 2 } = 1$ and $\sigma ^ { 2 } = 3 , \rho = 3 / 4 > 1 / 2$ . Theorem 2.3 uses Lemma 2.2 for one of its clauses and should therefore be verified only conditionally, on that clause. Corollary 2.4 cites Lemma 2.2 in its proof without needing it, and the kit should find that the inherited condition is unused and drop it.

This paper is checked twice. After the first check we revised it as an author would: the proofs of 2.1 and 2.2 were rewritten with their statements unchanged, and a sentence was added inside the definition of $R , \ : c ^ { * }$ and $\rho ,$ which moves every later line without changing a definition. The revision should then redo only 2.1 and 2.2, rebuild 2.3 and 2.4 against the new modules without redoing them, close the issue of 2.2 with a reference to its new audit, and reconnect the clause of 2.2 in 2.3, which then becomes unconditional.

The first check agreed with the key on all four results. Lemma 2.2 was recorded as a PAPER\_ARGUMENT\_GAP: the issue checker refuted both false assertions of its proof in Lean (the harmonic mean of 1 and 4 is $8 / 5 > 1 ;$ at $\theta = 1 , \ : \sigma = 2 , \ : \rho = 4 / 5 > 1 / 2 )$ , proved moreover that the intermediate bound holds only when $\theta ^ { 2 } = \sigma ^ { 2 }$ , and recorded that the lemma’s conclusion is true, so that the class is not ERROR. Theorem 2.3 was verified conditionally on the clause of 2.2 that it uses. For Corollary 2.4 the drafter kept the clause of 2.2 that the paper cites, but the trace of hypothesis usage showed that the proof of the corollary’s statement does not reach it, and the final audit recorded it as a remark. The corollary was verified unconditionally. The run took 71 minutes and cost \$9.85.

The revision was then handled as the kit’s guide prescribes: the files in source/ were replaced by the new version and one unattended session followed the revision instructions. Its record of what changed reads, in part,

\- Changes: new sentence main.tex:34 (remark in Definition 1.1; D01-D04 UNCHANGED per defs-changed); L01 proof rewritten (43-52); L02 proof rewritten (58-61); T01, C01 text unchanged, only line shifts

The four shared definitions were compared by identifier and found unchanged, although every line after the new sentence had moved, so no result was reset because of a definition. Only the two results whose proofs changed were reset. The other two kept their verdicts, marked as waiting for a re-check. The driver then re-checked the reset results, and the follow-up table of the new report summarizes the outcome:

```typescript
| No. | Before (revision) | Now | Status |
| 2.1 | Verified unconditionally | Verified unconditionally | — |
| 2.2 | Recorded issue | Verified unconditionally | B1: resolved |
| 2.3 | Verified conditionally | Verified unconditionally | — |
| 2.4 | Verified unconditionally | Verified unconditionally | — |
```

The issue of 2.2 was closed by its new final audit, and Theorem 2.3 lost its condition because it now imports the verified lemma. The end state is the one the key describes, reached by a longer route. The key expected 2.3 and 2.4 to be only rebuilt and checked mechanically, since the statements of 2.1 and 2.2 did not change. At the new gate of 2.2, however, the statement file had lost two auxiliary definitions that only the old, wrong proof had used (the harmonic mean and the false claim about it). Version 1.3.1 compared every definition of the statement file, saw a changed statement and redid 2.3 and 2.4 in full; version 1.3.2 compares only what the dependents use. The revision took 23 minutes and cost \$7.40, against 71 minutes and \$9.85 for the first check.

## 6.2.4 Empirical distribution functions: The limits of the library

The fourth paper tests the edge of the library. Lemma 2.1, that the empirical distribution function $F _ { n }$ is nondecreasing with $n F _ { n } ( t ) \in \{ 0 , \ldots , n \}$ , is elementary. Proposition 2.2 is the Glivenko– Cantelli theorem, sup $| F _ { n } ( t ) - F ( t ) | \to |$ 0 almost surely, whose proof is the single sentence “This is the Glivenko–Cantelli theorem”: the theorem is named, but neither cited nor proved. The statement can be written with Mathlib’s independence and almost-sure convergence. The paper gives no proof to check and the pinned library lacks the theorem, so the expected verdict is conditional on an input of provenance (v). Theorem 2.3 is the Kolmogorov–Smirnov limit law. Its statement needs the Brownian bridge, which the pinned Mathlib does not provide, so the expected outcome is an unfinished check, TRANSLATION\_BLOCKED, that names what is missing: a gap of the library, which the report keeps apart from the paper’s issues.

The run took a diferent and more ambitious route than the key expected, one the key allows (Table 10). For Theorem 2.3 the drafter defined a Brownian bridge by its properties, without constructing one,

```javascript
def IsBrownianBridge {Ω' : Type*} [MeasurableSpace Ω'] (Q : Measure Ω')
(B : <sup>R</sup> → Ω' → <sup>R</sup>) : Prop :=
(∀ t, Measurable (B t)) ∧
(∀ ω, ContinuousOn (fun t => B t ω) (Icc (0:<sup>R</sup>) 1)) ∧
∀ (m : <sup>N</sup>) (u a : Fin m → <sup>R</sup>), (∀ i, u i ∈ Icc (0:<sup>R</sup>) 1) →
```

<table><tr><td>No.</td><td>Answer key</td><td>Run</td><td>Score</td></tr><tr><td>2.1</td><td>unconditional</td><td>unconditional</td><td>agrees</td></tr><tr><td>2.2</td><td>conditional on Glivenko–Cantelli, (v) or (ii); (i) acceptable</td><td>conditional on Glivenko-Cantelli, (i) accepted</td><td></td></tr><tr><td>2.3</td><td>unfinished, TRANSLATION_BLOCKED; conditional with the Brownian bridge as an explicit parameter</td><td>conditional on Donsker&#x27;s theorem and Kolmogorov&#x27;s formula, the Brownian bridge defined by its properties</td><td>accepted</td></tr></table>

Table 10: The fourth example paper. The run took 81 minutes and cost \$14.32.

Q.map (fun $\omega \ = > \ \sum \ \mathrm { ~ i ~ } , \texttt { a i * B ( u i ) } \ \omega ) \ =$   
ProbabilityTheory.gaussianReal 0   
(Real.toNNReal   
(P i, P j, a i \* a j \* (min (u i) (u j) - u i \* u j)))

and stated the theorem for every such process on every probability space, with Donsker’s theorem and Kolmogorov’s formula for the law of the supremum as cited inputs. The gate accepted the encoding. The first proof pass stopped on the measurability of $D _ { n }$ , a step the paper takes for granted, and the result was recorded as unfinished, BUDGET\_EXHAUSTED, the one stop reason that the example papers could not be written to trigger. The driver’s second pass, on the stronger model, proved ${ \mathrm { i t } } ,$ and the theorem became verified conditionally on the two cited results. The report then asks a person to check what the Lean evidence cannot show: that the encoded property describes the standard Brownian bridge, and that such a process exists, since a statement quantified over all Brownian bridges would hold vacuously if the encoded property could not be satisfied. Proposition 2.2 was verified conditionally on the Glivenko–Cantelli theorem, which the kit labeled as cited (i) rather than as missing from the library (v); the key accepts this, since the paper names the theorem.

## 6.3 Published results with public corrections

We searched for published statistical results whose proofs were later corrected in public, and chose three in which the corrected error lies in a proof (not in a typo or a numerical experiment), the afected results are few enough to check, and Mathlib covers most of what the proofs need. For each, the kit checked the version from before the correction, blind. We then compared its report with the correction and checked the corrected version through the revision workflow.

Case A: Sequential e-values for forecast comparison. Henzi and Ziegel (2022b) construct e-values for testing whether one probability forecast dominates another, first for a single period and then sequentially for forecasts made h steps ahead. Their Section 3 has five numbered results. The last (Proposition 3.5 in the arXiv version we checked, Proposition 2 in the journal) states that the averaged lag-h e-value $e _ { \tau }$ satisfies $\mathbb { E } ( e _ { \tau } ) \le 1$ at every almost surely finite stopping time $\tau .$ The authors’ correction (Henzi and Ziegel 2022a) shows by an example with h = 2 that this is false for $h > 1$ , and proves the bound for $e _ { \tau + h - 1 }$ instead. The kit checked the whole paper, version 2 on arXiv, which predates the correction.

The kit found the published error. Proposition 3.5 was recorded as an ERROR. The issue checker located the failing step, $\begin{array} { r } { ^ { \mathfrak { c } } \mathbb { E } ( e _ { \tau } ) = \sum _ { t } \mathbb { E } \{ \mathbb { E } ( e _ { t } \mid \mathcal { F } _ { t } ) 1 ( \tau = t ) \} \le 1 ^ { \mathfrak { , } } } \end{array}$ , and explained it: Proposition 3.4 gives only $\mathbb { E } ( e _ { t } ) \le 1$ , and since $e _ { t }$ is $\mathcal { F } _ { t } .$ -measurable, $\mathbb { E } ( e _ { t } \mid \mathcal { F } _ { t } ) \leq 1$ would mean $e _ { t } \leq 1$ . It then built a counterexample in Lean on a probability space of two points, a fair coin: the Brier score, forecasts $p = 1 / 4$ and $q = 3 / 4$ , lag $h = 2$ , both outcomes $Y _ { 3 }$ and $Y _ { 4 }$ equal to the coin, a betting fraction of one at time 2 only, and the stopping time that stops at 3 if $Y _ { 3 } = 0$ and at 4 if $Y _ { 3 } = 1$

## def tau0 : Bool → <sup>N</sup> := fun ω => if ω = true then 4 else 3

for which $\begin{array} { r } { \mathbb { E } ( e _ { \tau } ) = \frac { 1 } { 2 } ( \frac { 3 } { 2 } + 1 ) = \frac { 5 } { 4 } } \end{array}$ . An auditor confirmed that the instance satisfies every printed hypothesis, including the conclusion of Proposition 3.4 that the proof relies on. The report adds that for $h = 1$ the conclusion is believed true but not shown by the printed proof, and records the issue checker’s suggestion, marked as not checked in Lean: restrict the proposition to $h = 1$ , and for $h \geq 2$ state the version that the paper uses later, for the e-value stopped at $\tau + h - 1$ . This is the correction the authors published.

The run also recorded two issues that the key did not anticipate, both in proofs of true theorems and both at the edge of a parameter range. The proof of Theorem 3.1 concludes that $E ( 0 ) \in ( 0 , 1 )$ although the theorem allows $\lambda = 1$ , where $E ( 0 ) = 0$ . The proof of Theorem 3.2 claims that $\begin{array} { r } { \operatorname* { l i m } _ { \lambda \to 1 } g ( \lambda ) = - \infty } \end{array}$ , which fails at $\pi _ { 1 } = 1$ , an alternative the theorem allows. We checked both by hand. They are real, they leave the theorems intact, and the authors’ corrected version keeps both proofs as they were. Under the scoring rule they count as difering from the key; both are correct minor findings. Proposition 3.4 was verified conditionally, and Theorem 3.3 conditionally on the cited theorem of Grünwald and coauthors, as the key expects. The run took 130 minutes and cost \$36.36.

We then gave the kit the authors’ corrected version, version 3 on arXiv, as a revision. It reset only Propositions 3.4 and 3.5, the two results whose text had changed, and kept the issues of Theorems 3.1 and 3.2, whose proofs version 3 leaves as they were. The corrected Proposition 3.5, $\mathbb { E } ( e _ { \tau + h - 1 } ) \le 1$ , was verified conditionally: the case $h = 1$ without any input, and the case $h > 1$ through Proposition 3.4, on the clause of Theorem 3.1 that it uses and on the mixture representation of the scoring function that the paper asserts. The earlier issue of Proposition 3.5 was closed by the new final audit. The second round took 83 minutes and cost \$24.08.

Case B: Stochastic dominance of convex combinations. Arab, Lando, and Oliveira (2025) introduce a class of heavy-tailed distributions and prove, in their main result (Theorem 3.1), that a variable X of this class is stochastically smaller than every convex combination $\theta _ { 1 } X _ { 1 } + . . . + \theta _ { n } X _ { n }$ of independent copies. The proof is an induction on n. Their corrigendum (Arab et al. 2026) explains that the induction step is wrong: after a chain of correct inequalities, it uses the sign of a bracketed term in the wrong direction. The statement is true, and the corrigendum gives a new induction step. The kit checked the whole paper, version 2 on arXiv, which predates the corrigendum.

The kit found the published error. Its issue on Theorem 3.1 is placed at the last display of the induction step: the first inequality there, $P ( \theta _ { 1 } X _ { 1 } + \ldots + \theta _ { n } X _ { n } > x ) \ge \bar { F } _ { X } ( x / \theta _ { n } ) + \bar { F } _ { X } ( x / ( 1 - \theta _ { n } ) )$ is false, because the term dropped from the previous line is nonpositive, so dropping it gives an upper bound and not the claimed lower bound. A Lean file refutes the intermediate claim (it fails at $x = 0$ for every law of the class, and for a Pareto law at $x = 1 / 2$ with equal weights $1 / 3$ its right-hand side is $5 / 3 )$ , and the issue was recorded as a PAPER\_ARGUMENT\_GAP, with the theorem’s conclusion not threatened. The issue checker added, as its own suggestion and marked as not in the paper, that the final bound follows directly from the previous line and the defining inequality of the class. We checked this by hand: with $a = F _ { X } ( x ) , b = F _ { X } ( x / \theta _ { n } )$ and $c = F _ { X } ( x / ( 1 - \theta _ { n } ) )$ ), the

previous line minus ${ \bar { F } } _ { X } ( x )$ equals $a ( 1 + a - b - c ) $ , which is nonnegative by the defining inequality.   
The repair is correct, and shorter than the new induction step of the corrigendum.

Beyond the published error, the run recorded issues on eight of the twelve registered results, against one in the key, and we checked every one by hand (Table 11). Three are the same small omission. Definition 2.2 and Definition 4.9 require $F _ { X } ( 0 ) = 0$ , but Lemma 2.3, Lemma 4.10 and Proposition 5.2 state their characterizations without it, so their $ { \mathbf { \hat { \theta } } } _ { \mathrm { 1 f } } ^ { \prime \prime }$ directions fail for $X = 0$ almost surely. The journal version adds the condition to Lemma 2.3 and Proposition 5.2. Proposition 4.6 similarly needs X to be non-degenerate, and the proof of Proposition 3.2 takes square roots of a variable that is not assumed nonnegative. Theorem 2.8 fails for a star-shaped function that vanishes near zero, such as $h ( x ) = \operatorname* { m a x } ( x - 1 , 0 )$ , since $h ( X )$ has an atom at zero whenever $P ( X \leq 1 ) > 0$ as for the InvSub law $F _ { X } ( x ) = x / ( 1 + x )$ . For example, take Theorem 4.8, “if X is nonnegative super-Pareto then X is InvSub”. The kit’s counterexample is $X = Z + 1$ , where Z has the Pareto law $F _ { Z } ( z ) = 1 - 1 / z \ \mathrm { o n } \ [ 1 , \infty )$ : X is super-Pareto, being a convex increasing function of $Z ,$ , and has no atom at zero, but $G ( s ) = 1 - F _ { X } ( 1 / s )$ satisfies $G ( 1 / 2 ) = 1 > G ( 1 / 4 ) + G ( 1 / 4 ) = 2 / 3$ so it is not subadditive. The step that fails is the proof’s appeal to shift invariance, which lets it assume $h ( 0 ) = 0$ . The statement presumably intends laws whose support starts at zero with no atom there, which is the repair the run suggests. None of these issues is in the corrigendum, and none afects the main result of the paper. Apart from Lemma 2.3 and Proposition 5.2, all of them are still in the journal version, where Theorem 2.8 and Lemma 4.10 are numbered 2.9 and 4.11. Table 11 also shows that Theorem 4.2, which the paper quotes from another paper without proof, was not registered as a result but as a cited input. The key had listed it, so the scoring rule counts it as missing.

In this run the gate also rejected a first draft. The first draft of Proposition 4.6 was failed outright, for three reasons recorded in the gate’s report: the drafted quantile function contradicted the paper’s definition of the generalized inverse; a drafted external input was false as stated, which would have made the main theorem vacuous; and one step was false only because of how the drafter had chosen the domain, “a formalization-side manufactured counterexample, not a paper error”. The redrafted statement passed, and the issue finally recorded for 4.6 is the paper’s own.

For the second round we replaced the induction step of Theorem 3.1 by the corrigendum’s and changed nothing else. The kit reset Theorem 3.1 alone, marked Proposition 3.3, which uses it, for a re-check, and left the other issues standing, since the corrigendum does not touch them. The new induction step was found “correct as written”, and the theorem was verified conditionally on one clause: the “only $\mathrm { i f } ^ { \dag }$ direction of Lemma 2.3, which is true (only the $ { \mathbf { \hat { \theta } } } _ { \mathrm { 1 f } } ^ { \prime \prime }$ direction fails) but belongs to a lemma with a recorded issue. The earlier issue of Theorem 3.1 was closed, and in Proposition 3.3 the clause of Theorem 3.1 was reconnected to the newly verified theorem. The second round took 54 minutes and cost \$15.24.

Case C: Geometric ergodicity of Hamiltonian Monte Carlo. Livingstone, Betancourt, Byrne, and Girolami (2019) give conditions under which Hamiltonian Monte Carlo is, or is not, geometrically ergodic. Their Corollary 2.3(ii) states that for the exponential family $\mathcal { E } ( \beta , \alpha )$ the method is not geometrically ergodic when $\beta < 1$ or $\beta > 2$ , as a consequence of Theorem 2.2, whose light-tailed part rests on Theorem 5.14. In September 2026 the authors added a corrigendum to the arXiv version (Livingstone et al. 2026). A condition of Theorem 5.14 compares gradients at points whose norms difer by a factor of two, and for $\mathcal { E } ( \beta , \alpha )$ it holds only when $\beta \geq 1 + \log _ { 2 } 3 \approx 2 . 5 8 5$ so the proof of the corollary has a gap for $2 < \beta < 1 + \log _ { 2 } 3$ . With a factor of three the condition holds for every $\beta > 2$ , and the corollary stands. The corrigendum also notes that the proof of Lemma 5.16 uses that condition although the lemma does not assume it. The paper is long, so we restricted the check to Theorem 2.2, Corollary 2.3(ii), Theorem 5.14 and Lemmas 5.15–5.17. Other results of the paper that these proofs use entered as inputs, stated by the paper and not checked. The scope includes results with no known error; because we chose it with the correction in view, this case is a weaker test than the other two.

<table><tr><td>No.</td><td>Key</td><td>Run</td><td>Our check</td></tr><tr><td>2.3</td><td>verified</td><td>PAPER_ASSUMPTION_GAP: Fx (0) = 0 missing</td><td>correct, minor</td></tr><tr><td>2.5</td><td>verified</td><td>unconditional</td><td></td></tr><tr><td>2.8</td><td>verified</td><td>ERROR: fails when h vanishes near 0</td><td>correct, minor</td></tr><tr><td>3.1</td><td>PAPER_ARGUMENT_GAP</td><td>PAPER_ARGUMENT_GAP in the induction step</td><td>the published error</td></tr><tr><td>3.2</td><td>verified</td><td>PAPER_ARGUMENT_GAP  $( { \sqrt { X } }$  for signed X) and DOMAIN_CLARIFICATION</td><td>correct, minor</td></tr><tr><td>3.3</td><td>conditional, (iv)</td><td>conditional on the clause of 3.1</td><td></td></tr><tr><td>4.2</td><td>conditional</td><td>not registered; a cited input</td><td>quoted without proof</td></tr><tr><td>4.6</td><td>verified</td><td>PAPER_ASSUMPTION_GAP:</td><td>correct, minor</td></tr><tr><td>4.8</td><td>verified</td><td>non-degeneracy missing ERROR: counterexample  $Z + 1$ </td><td>correct; the statement needs a</td></tr><tr><td></td><td>4.10 verified</td><td>PAPER ASSUMPTION GAP:</td><td>support starting at 0 correct, minor</td></tr><tr><td>4.12</td><td>verified</td><td> $F _ { X } \mathrm { ( 0 ) = 0 ~ m i s s i n g }$  unconditional</td><td></td></tr><tr><td>5.2</td><td>verified</td><td>PAPER_ASSUMPTION_GAP: Fx (0) = 0 missing</td><td>correct, minor</td></tr><tr><td>5.3</td><td>verified</td><td>unconditional</td><td></td></tr></table>

Table 11: Case B, version 2 on arXiv. “Key” is the answer key written from the corrigendum before the run; “our check” is the hand check of every finding that the key did not anticipate. The run took 171 minutes and cost \$65.24.

The kit found both points of the corrigendum. For Corollary 2.3(ii) the drafter and the gate noticed that the proof, “a direct consequence of Theorem 2.2”, needs condition (18) for the exponential family, and the issue checker settled the question in Lean in both directions: condition (18) fails for every member of $\mathcal { E } ( \beta , \alpha )$ with $2 < \beta < \log _ { 2 } 6$ and holds for $\beta \geq \log _ { 2 } 6$

theorem osc\_fails {β α : <sup>R</sup>} {U : E 1 → <sup>R</sup>} {gU : E 1 → E 1}   
(hU : InExpFam β α U) (hgrad : IsGradField U gU)   
(hβ : 2 < β) (hβ6 : β < Real.logb 2 6) :   
¬ HypOscillations gU   
theorem osc\_holds $\{ \beta ~ \alpha ~ : ~ \mathbb { R } \} ~ \{ \mathrm { U } ~ : ~ \mathrm { ~ E ~ } ~ 1 ~  ~ \mathbb { R } \} ~ \{ \mathrm { g U } ~ : ~ \mathrm { ~ E ~ } ~ 1 ~  ~ \mathrm { E ~ } ~ 1 \}$   
(hU : InExpFam β α U) (hgrad : IsGradField U gU)

(hβ1 : 1 < β) (hβ6 : Real.logb $2 \ 6 \ \leq \ \beta )$   
HypOscillations gU

Since $\log _ { 2 } 6 = 1 + \log _ { 2 } 3$ , this is exactly the range of the corrigendum. The issue was recorded as a PAPER\_ARGUMENT\_GAP, with the remark that the conclusion is not shown false on that range but is not established by the paper either; among the recorded options for a repair is to replace the factor 2 in (18) by a larger fixed constant, which is what the authors did, with 3. For Lemma 5.16 the kit recorded that the proof uses condition (18), which the lemma does not assume, and added that the conclusion holds under (17) alone, which an issue checker’s probe proved in Lean; the corrigendum makes the same two observations. The class difers from the key’s first choice (PAPER\_ASSUMPTION\_GAP) but is the alternative the key accepts. Theorems 2.2 and 5.14 and Lemma 5.17 were verified conditionally, with the clause of Lemma 5.16 among their conditions, and Lemma 5.15 unconditionally, all as the key expects. The run took 154 minutes and cost \$36.37.

The corrected version, with the corrigendum’s changes carried into the text, changes the statements and proofs of all four results around Theorem 5.14 and the proof of the corollary, so the revision redid them. Lemmas 5.15–5.17 were all verified unconditionally: the issue of Lemma 5.16 was closed by its new audit, and Lemma 5.17 lost its condition when the clause of Lemma 5.16 that it uses was reconnected to the verified lemma. Theorem 5.14 was verified conditionally on the proposition on rejection probabilities that the paper uses, on standard Gaussian facts, and on the extension to a random number of leapfrog steps that the paper asserts. Corollary 2.3(ii) was verified conditionally on the clause of Theorem 2.2 that it uses, for a fixed number of leapfrog steps, with the earlier gap gone because the corrected condition (18) holds for every $\beta > 2$ . Theorem 2.2 itself ended unfinished, for a reason on the formalization side: the drafted version of Theorem 5.14 for a random number of steps carried an antecedent that the paper does not state, which the issue checker ruled an ENCODING\_ERROR, and the run stopped that part as TRANSLATION\_BLOCKED rather than redraft and re-gate the statement. The report files it on the formalization side. The second round took 88 minutes and cost \$31.28.

What the three cases show. In each case the kit found the published error, placed it at the step the correction names, and classified it as the correction implies: a false statement with a counterexample in case A, a gap in a true theorem in cases B and C. In case C it also found the second, minor point of the corrigendum, and it determined in Lean the exact range of the parameter for which the printed argument fails. Each report ended with a repair close to the authors’ own: the version stopped at $\tau + h - 1$ in case A, a larger factor in condition (18) in case C, and in case B a repair that is simpler than the published one. The kit also reported problems that no correction mentions, two in case A and seven in case B, all of which we confirmed by hand; most are missing conditions at the edge of a definition, and two, in case B, are theorems that are false as stated (Theorems 2.8 and 4.8). We found no false alarm in the three cases, and no published error was missed.

Selection and training data. We chose the three papers knowing their corrections, among those whose proofs Mathlib can mostly express. Cases A and B were checked in full, case C around the corrected results. The models may have seen some of these corrections during training: the correction of case A appeared in 2022 and that of case B in the spring of 2026, both before the training-data cutof of June 2026 of the models we used (Anthropic 2026a), while that of case C was posted in September 2026, after it. No transcript of any of the three runs, of the coordinator or of any agent, mentions a corrigendum or a published correction. The findings also rest on more than the agents’ word: every published error is backed by a reviewed Lean file, a counterexample in case A and a refuted step in cases B and C, which anyone can rebuild without a language model, and the findings that no correction mentions go beyond what the corrections contain, although the models may have seen the journal versions, which in case B already repair two of them. The published run records contain the Lean files and the reports, and the papers themselves are available from arXiv.

<table><tr><td></td><td>Paper (results) Correctly classified Faithful in Lean Hidden or without Lean evidence</td></tr><tr><td>A (5)</td><td></td></tr><tr><td> $5 ~ / ~ 5 ~ / ~ 5$ </td><td> $3 ~ / ~ 3 ~ / ~ 0 ~$   $\mathrm { ~ 0 ~ / ~ 2 ~ / ~ 3 ~ }$   $~ 0 ~ / ~ 4 ~ / ~ 8$ </td></tr><tr><td>B (12) C (6)</td><td> $1 2 \mathrm { ~ / ~ 9 ~ / ~ 1 0 ~ }$   $1 0 ~ / ~ 5 ~ / ~ 0$ </td></tr><tr><td> $2 ~ / ~ 4 ~ / ~ 0$   $~ 0 ~ / ~ 0 ~ / ~ 2$ </td><td> $6 ~ / ~ 6 ~ / ~ 6$ </td></tr><tr><td>23 / 20 / 21  $1 5 / 1 2 / 0$ </td><td></td></tr></table>

Table 12: The kit, the direct agent and the direct chat on the three published papers (entries kit / agent / chat). “Faithful in Lean”: verdicts backed, without conditions, by a compiled, sorry-free theorem stating what the paper states. The kit’s other verdicts are conditional, with every input a named hypothesis of the theorem. “Hidden or without Lean evidence”: verdicts that rest on an input assumed without saying so, on a statement narrower than the paper’s presented as the paper’s, or on no Lean file at all.

## 7 Evaluation

## 7.1 What a direct use of a language model does not guarantee

A statistician can also give a paper directly to a strong model and ask it to check the proofs in Lean. We did this for the papers of Section 6, with the same model, claude-opus-5-5, in two ways. As an agent, it worked in Claude Code without the kit, in a folder with the paper and the same Lean environment, told how to compile, without network access, and asked:

Please use Lean to check the paper: for each numbered result (theorem, proposition, lemma, corollary), tell me whether its statement and its proof are correct, prove in Lean what you can, give a counterexample in Lean when something is false, and point out any errors in the paper.

As a chat, it received the LAT X source in one message with the same request and no tools. The prompts and the grading rules were fixed before the runs. Each answer was graded against the answer keys and our hand checks: a verdict counts as backed by Lean only if the cited theorem compiles, uses only the standard axioms, has no sorry and states what the paper states. The grading was done by Claude agents following these written rules, with every judgment recorded with its reason, and checked by us. The four example papers, written to illustrate the kit, are short enough that all three uses found the planted errors and reached essentially the same verdicts on them; the informative comparison is on the three research papers (Table 12), where every use also found the published error.

The diferences come from the mechanisms of Section 5, which the paragraphs below take in turn.

The statement is compared with the paper. In the kit, a second agent compares every statement with the printed one before it is proved, and the fingerprints recorded at that point stop it from changing afterwards. A direct agent chooses its own statements, and nothing marks the diference when they drift. In case A it reported Theorems 3.2 and 3.3 as proved in Lean, but its files prove Theorem 3.2 only for $p < q ,$ leaving out the case $p > q$ that the paper calls analogous, and Theorem 3.3 only for $p < q$ and $\pi _ { 1 } < 1$ ; its summary mentions neither restriction. Finding them took a reading of every file.

Every input is visible. The kit turns each cited result, each fact asserted without proof and each clause of a result with an issue into a named hypothesis of the Lean theorem, and the verdict lists them for a person to check. In none of the 23 results did a kit verdict rest on an input it did not name. The direct agent rested three verdicts on a narrower statement or an input it did not mention (in case A, Proposition 3.4 assumes the fact that the paper asserts at the start of the section), and in three more gave no Lean evidence for steps it called correct; the chat, which cannot compile, gave none for twelve verdicts and hid an input in one more.

Every result of the paper is checked. The kit takes each numbered result through the same pipeline, and a suspected problem is recorded only after a second agent confirms it. On the longest paper, case B with twelve results, the kit classified all twelve correctly, the agent nine and the chat ten. Both missed conditions absent from the statements of Lemma 4.10 and Proposition 4.6, and the agent also that of Proposition 5.2; the kit’s reviewers had found and confirmed all three, and so had our hand check.

The verdict and its evidence agree. The mechanical audit ties each verdict to the theorem it names, checks its axioms, its closure and the hypotheses its proof uses, and the validator refuses a record that disagrees with the audits; the evidence can be rebuilt ofline. A direct answer is text, and its claims about its own files can be checked only by reading those files.

The direct agent sometimes went further than the paper, proving in Lean inputs that the kit kept as conditions: four of the six in case C, and in case A the cited theorem of Grünwald and coauthors for the case it needed. The kit keeps such inputs as conditions by design, so that a verdict on the paper rests only on the proof that the paper contains. These mechanisms make a check with the kit more expensive than a direct use of the model, since every statement, proof and finding passes through several agents and a mechanical audit. In return, every statement is checked against the paper before it is proved, and every verdict can be verified afterwards from files that rebuild without a model.

## 7.2 Cost: An earlier design against version 1.3

The repository contains a benchmark paper with three results: an easy lemma, a proposition that rests on a cited inequality, and a lemma with a planted error that is false for $x < - 1$ . Table 13 compares two runs of an earlier design of the kit, which used the stronger model for every role, with two runs of version 1.3, which introduced the per-role models, the rule cards, the audit script and the driver. All four runs used the same prompt, the same Lean and Mathlib versions and the same machine, and all four gave the verdicts of the answer key: the lemma verified unconditionally, the proposition conditionally on the cited inequality, and the planted error recorded as an ERROR with a reviewed counterexample.

Part of the saving comes from cheaper models for the writing roles, whose work is reviewed afterwards. Part comes from structure: the reviewing tasks ran on the stronger model in both designs and still became 2.4 to 2.7 times cheaper, because of the rule cards, the audit script and shorter outputs. On research papers proving and auditing dominate the cost, which this three-result benchmark hardly exercises.

Version 1.3.1 changed robustness rather than eficiency. Two runs on the same paper, with the models available in October 2026 (claude-sonnet-5-5 and claude-opus-5-5), again gave the three verdicts of the answer key, for \$4.21 and \$3.78 in 14 and 16 minutes of wall-clock time. In both, the auditor ran the mechanical checks itself, and the cost estimate from the transcripts agreed with the figure reported by Claude Code to the cent. Since the models had changed as well, we also ran version 1.3 twice with the newer models. The first run cost \$5.52 and gave the three verdicts of the key. The second ended after ten minutes with two results unfinished. The gate had found the statement of the easy lemma faithful, but the permission system refused the way it had written its compile command, so it failed the statement twice for want of evidence, and the proposition that depends on the lemma never started. Version 1.3.1 allows every form of the kit’s commands, tells each agent the one to use, and stops the run with an explanation when commands in their standard form are refused (Section 5.6). With the newer models, complete runs of either version cost \$3.78 to \$5.52, but only version 1.3.1 completed both runs.

<table><tr><td></td><td>Earlier design</td><td>Version 1.3</td><td>Change</td></tr><tr><td>Cost (list price)</td><td>$71.15, $65.04</td><td>$17.73, $13.68</td><td>-77%</td></tr><tr><td>Tokens processed (millions)</td><td>73.2, 64.1</td><td>31.2, 21.9</td><td>-61%</td></tr><tr><td>Time, sum of sessions (minutes)</td><td>84, 83</td><td>51, 30</td><td>-51%</td></tr><tr><td>Verdicts as in the answer key</td><td>3/3, 3/3</td><td>3/3, 3/3</td><td></td></tr><tr><td>False alarms; unfinished</td><td>0; 0</td><td>0; 0</td><td></td></tr></table>

Table 13: Two runs of the earlier design and two of version 1.3 on the three-result benchmark paper. Changes are between the means.

## 7.3 The independent fact check of reports

Reports written by the coordinator contained errors in every run, and the fact-check role found them before delivery (Table 14). The major ones were of one kind: statements that the records did not support, such as a suggested correction that no review had proposed, or evidence attributed to the wrong agent. In the proportions example, for instance, the draft report advised the authors to add an argument for the “well known” concavity; the fact check found that no review had made that suggestion and that the compiler’s own notes said so, and the sentence was replaced by what the reviews had actually ruled. The other problems were mostly wrong line numbers, a fair implicit step listed as a typo, or a claim that the report had already passed the fact check before the corrections were made.

## 7.4 A revision

In the full-workflow example, the authors’ correction of the benchmark’s false lemma took 17 minutes and \$5.06 to re-check: only that lemma was redone, it became verified unconditionally, and the earlier issue was closed with a reference to the new final audit. In the shrinkage example (Section 6.2.3) a revision that rewrote two proofs took 23 minutes and \$7.40, a third of the first check’s time, and closed the one issue; the comparison of definitions by identifier kept a moved but unchanged definition from resetting anything, and the cascade redid two dependents that only needed a rebuild, which version 1.3.2 now rebuilds. The three published cases add three revisions of research papers. Each reset exactly the results whose text the correction had changed (two in case A, one in case B, five in case C), closed the published issue only with a new final audit, kept the issues that the correction does not address, and reconnected the clauses of the newly verified results in their dependents. They took 54 to 88 minutes and cost \$15 to \$31, between 23 and 86 percent of the cost of the first check, depending on how much the correction changed.

<table><tr><td>Run</td><td>Major</td><td>Moderate</td><td>Minor</td><td>Total</td></tr><tr><td>Two-lemma example</td><td>1</td><td>3</td><td>5</td><td>9</td></tr><tr><td>Full-workflow example, first report</td><td>1</td><td>7</td><td>6</td><td>14</td></tr><tr><td>Full-workflow example, report after revision</td><td>1</td><td>5</td><td>8</td><td>14</td></tr><tr><td>Sample variance (E1)</td><td>1</td><td>3</td><td>9</td><td>13</td></tr><tr><td>Proportions and sums (E2)</td><td>1</td><td>3</td><td>7</td><td>11</td></tr><tr><td>Linear shrinkage (E3), first report</td><td>1</td><td>5</td><td>10</td><td>16</td></tr><tr><td>Linear shrinkage (E3), report after revision</td><td>1</td><td>4</td><td>9</td><td>14</td></tr><tr><td>Empirical distribution function (E4)</td><td>0</td><td>2</td><td>8</td><td>10</td></tr><tr><td>Case A, first report</td><td>0</td><td>7</td><td>11</td><td>18</td></tr><tr><td>Case B, first report</td><td>1</td><td>9</td><td>13</td><td>23</td></tr><tr><td>Case C, first report</td><td>2</td><td>6</td><td>9</td><td>17</td></tr><tr><td>Case A, report after revision</td><td>2</td><td>8</td><td>13</td><td>23</td></tr><tr><td>Case B, report after revision</td><td>1</td><td>3</td><td>13</td><td>17</td></tr><tr><td>Case C, report after revision</td><td>1</td><td>12</td><td>12</td><td>25</td></tr></table>

Table 14: Problems found by the fact check in the first draft of each report.

## 8 Reproducibility

The agents’ output difers from run to run, and a new run needs a Claude Code account. Everything the verdicts rest on, and every table of results in this paper, can be reproduced exactly from the archived records without a language model. The records of every run of Sections 6 and 7 are part of the replication materials, with the answer keys, the hand checks and the time and checksum of each key. For the four example papers the materials also contain the papers, and their records are in the repository’s examples folder from version 1.3.2 on; the three published papers are available from arXiv. The replication materials are in the folder replication of the repository, https://github.com/ziwaa-se/kappa.

1. Ofline, exactly. With the pinned Lean and Mathlib, every archived Lean file of every run recompiles, and the mechanical checks (axioms, sorry walk, hypothesis usage, fingerprints) give the archived results. A script regenerates the tables of results (verdicts against the keys, costs, times and fact-check counts) from the archived records. None of this needs a language model. Replaying one case study takes two to thirteen minutes on the machine described in the Computational details, after a one-time download of the compiled Mathlib (about 5 GB).

2. Archived, as records. The text written by agents (proofs, reviews, reports) is archived as it was produced, cleaned of local paths and user names, together with a summary of every session (cost, turns, models, times) and the driver’s log. The full transcripts remain with the authors, who ran the blind audit on them.

3. The comparison. The prompts, the answers, the Lean files and the gradings of the comparison in Section 7.1 are part of the materials, together with the protocol fixed before the runs.

4. New runs, similar results. The benchmark runner repeats a run. Verdicts should agree with the answer key, while text, cost and time vary, as the pairs of runs in Table 13 show.

The kit’s scripts have an ofline test suite, run on every change on Linux and macOS, and integration tests that compile Lean files.

## 9 Limitations

The kernel checks the proof of the Lean statement; whether that statement says what the paper says rests on the review by the gate and the auditor. The report lists the statements of the main results with their places in the paper, so that a person can confirm this review. A conditional verdict is only as reliable as its listed inputs, which a person has to check. A cited result that Mathlib happens to contain is still kept as an input. Results that need theory missing from the pinned Mathlib stop as unfinished, with the missing interface named. The agents run in Claude Code, a paid service, although everything the verdicts rest on can be rechecked without it (Section 8). The case studies cover 42 results in seven papers, each checked once, which is too few to estimate error rates. The permission rules and the hook guard against mistakes by agents and are not a security boundary.

## 10 Summary

KAPPA turns the proofs of a LAT<sub>E</sub>X paper into a Lean check of that paper’s own argument and reports one of four verdicts for every numbered result. Its design responds to the facts that a proof that compiles may prove the wrong statement, that language model output varies and needs checking, and that checking a paper takes hours. The kit therefore reviews and fingerprints every statement before it is proved, separates the agents that write from those that review, rests each verdict on files that can be rebuilt without a model, and runs unattended in short sessions that keep all state on disk.

On four example papers with answer keys written in advance, 17 of 19 verdicts agreed with the key or with an outcome the key allowed. Of the other two, one was a finding that our hand check showed to be better than the key, and one revealed an ambiguity in the kit’s rules that the current release resolves. On three published papers checked before their corrections, the kit found each published error at the step the correction names, with a Lean counterexample or a refuted step, and suggested repairs close to the authors’. It also reported problems that the corrections do not mention, which we checked by hand and found correct. Most are conditions missing at the edge of a definition. The same model, given the papers directly as a single agent with Lean, also found the published errors, but it missed problems on the longest paper, reported statements narrower than the paper’s as proved, and assumed inputs without saying so. The mechanisms of the kit are designed to prevent each of these (Section 7.1). A first check with the kit cost between \$10 and \$65 at list prices and took one to three hours of unattended running, and a revision cost between 23 and 86 percent of that.

A verified result of the kit is backed by a Lean proof of a statement that was reviewed against the printed text before it was proved, and a conditional result names every input that a person still has to check. Within the reach of the pinned Mathlib, the kit gives authors and referees a list of which steps of a proof hold and which need repair, each located in the paper, classified and backed by files that can be rechecked.

## Computational details

The runs in this paper used KAPPA 1.3.1 (commit 338b9d9, released as the TeX Proof-Check Kit), Claude Code 2.1.296 with the models claude-sonnet-5-5 and claude-opus-5-5, Lean 4.19.0 and Mathlib revision c44e0c8e, on a Mac mini with an Apple M4 processor and 16 GB of memory running macOS 26.6.2. Up to seven runs shared the machine at a time. Costs are those that

Claude Code reported for each session, at list prices. The estimates that pv\_usage.py computes from the transcripts agree with them to within five cents per run. Times are active running times, without the waits for usage limits. The comparison of Section 7.1 used claude-opus-5-5 through the same Claude Code version. The benchmark runs of the earlier design (version 1.2.1 in the kit’s changelog) and of version 1.3 used the models claude-sonnet-5 and claude-opus-5; their costs are list-price estimates computed from the transcripts.

## Acknowledgments

Xiao Wang acknowledges support from the National Science Foundation under Grant No. SES-2316428 and from Anthropic through its AI for Science program.

Xiao Wang conceived the project and supervised it. Zihua She designed and implemented the software, wrote the example papers and their answer keys, ran the case studies, checked the findings by hand and wrote the first draft of the manuscript. Both authors chose the published cases and revised the manuscript.

Claude Code helped develop the kit and revise this manuscript; the authors checked all of it and are responsible for the content.

## References

Altman M, Gill J, McDonald MP (2007). “accuracy: Tools for Accurate and Reliable Statistical Computing.” Journal of Statistical Software, 21(1), 1–30. doi:10.18637/jss.v021.i01.

Anthropic (2026a). Models Overview. Claude Platform Docs, accessed 2026-10-23, URL https: //platform.claude.com/docs/en/models/overview.

Anthropic (2026b). Claude Code. Version 2.1.296, URL https://code.claude.com/docs/en/ overview.

Arab I, Lando T, Oliveira PE (2025). “Convex Combinations of Random Variables Stochastically Dominate the Parent for a New Class of Heavy Tailed Distributions.” Electronic Communications in Probability, 30, 1–11. doi:10.1214/25-ECP714.

Arab I, Lando T, Oliveira PE (2026). “Corrigendum to “Convex Combinations of Random Variables Stochastically Dominate the Parent for a New Class of Heavy Tailed Distributions”.” Electronic Communications in Probability, 31, 1–3. doi:10.1214/26-ECP778.

Avigad J, Hölzl J, Serafin L (2017). “A Formally Verified Proof of the Central Limit Theorem.” Journal of Automated Reasoning, 59(4), 389–423. doi:10.1007/s10817-017-9404-x.

Azerbayev Z, Piotrowski B, Schoelkopf H, Ayers EW, Radev D, Avigad J (2023). “ProofNet: Autoformalizing and Formally Proving Undergraduate-Level Mathematics.” arXiv preprint 2302.12433. doi:10.48550/arXiv.2302.12433.

Baker P (2020). “Using GNU Make to Manage the Workflow of Data Analysis Projects.” Journal of Statistical Software, Code Snippets, 94(1), 1–46. doi:10.18637/jss.v094.c01.

Bischl B, Lang M, Mersmann O, Rahnenführer J, Weihs C (2015). “BatchJobs and BatchExperiments: Abstraction Mechanisms for Using R in Batch Environments.” Journal of Statistical Software, 64(11), 1–25. doi:10.18637/jss.v064.i11.

de Moura L, Ullrich S (2021). “The Lean 4 Theorem Prover and Programming Language.” In A Platzer, G Sutclife (eds.), Automated Deduction – CADE 28, volume 12699 of Lecture Notes in Computer Science, pp. 625–635. Springer-Verlag, Cham. doi:10.1007/978-3-030-79876-5\_37.

Gentleman R, Temple Lang D (2007). “Statistical Analyses and Reproducible Research.” Journal of Computational and Graphical Statistics, 16(1), 1–23. doi:10.1198/106186007X178663.

Gonthier G (2008). “Formal Proof—The Four-Color Theorem.” Notices of the American Mathematical Society, 55(11), 1382–1393.

Gonthier G, Asperti A, Avigad J, Bertot Y, Cohen C, Garillot F, Le Roux S, Mahboubi A, O’Connor R, Ould Biha S, Pasca I, Rideau L, Solovyev A, Tassi E, Théry L (2013). “A Machine-Checked Proof of the Odd Order Theorem.” In S Blazy, C Paulin-Mohring, D Pichardie (eds.), Interactive Theorem Proving – ITP 2013, volume 7998 of Lecture Notes in Computer Science, pp. 163–179. Springer-Verlag, Berlin, Heidelberg. doi:10.1007/978-3-642-39634-2\_14.

Hales T, Adams M, Bauer G, Dang TD, Harrison J, Hoang LT, Kaliszyk C, Magron V, McLaughlin S, Nguyen TT, Nguyen QT, Nipkow T, Obua S, Pleso J, Rute J, Solovyev A, Ta THA, Tran NT, Trieu TD, Urban J, Vu K, Zumkeller R (2017). “A Formal Proof of the Kepler Conjecture.” Forum of Mathematics, Pi, 5, e2. doi:10.1017/fmp.2017.1.

Henzi A, Ziegel JF (2022a). “Correction to: ‘Valid Sequential Inference on Probability Forecast Performance’.” Biometrika, 109(4), 1181–1182. doi:10.1093/biomet/asac043.

Henzi A, Ziegel JF (2022b). “Valid Sequential Inference on Probability Forecast Performance.” Biometrika, 109(3), 647–663. doi:10.1093/biomet/asab047.

Jiang AQ, Welleck S, Zhou JP, Lacroix T, Liu J, Li W, Jamnik M, Lample G, Wu Y (2023). “Draft, Sketch, and Prove: Guiding Formal Theorem Provers with Informal Proofs.” In International Conference on Learning Representations (ICLR). URL https://openreview.net/forum?id= SMa9EAovKMC.

Liang W, Zhang Y, Cao H, Wang B, Ding DY, Yang X, Vodrahalli K, He S, Smith DS, Yin Y, McFarland DA, Zou J (2024). “Can Large Language Models Provide Useful Feedback on Research Papers? A Large-Scale Empirical Analysis.” NEJM AI, 1(8), AIoa2400196. doi: 10.1056/AIoa2400196.

Livingstone S, Betancourt M, Byrne S, Girolami M (2019). “On the Geometric Ergodicity of Hamiltonian Monte Carlo.” Bernoulli, 25(4A), 3109–3138. doi:10.3150/18-BEJ1083.

Livingstone S, Betancourt M, Byrne S, Girolami M (2026). “On the Geometric Ergodicity of Hamiltonian Monte Carlo.” arXiv preprint 1601.08057v5, with a corrigendum in Appendix D. doi:10.48550/arXiv.1601.08057.

Massot P (2025). leanblueprint: A plasTeX Plugin to Build Formalization Blueprints. Python package version 0.0.20, URL https://github.com/PatrickMassot/leanblueprint.

Peng RD (2011). “Reproducible Research in Computational Science.” Science, 334(6060), 1226– 1227. doi:10.1126/science.1213847.

Polu S, Sutskever I (2020). “Generative Language Modeling for Automated Theorem Proving.” arXiv preprint 2009.03393. doi:10.48550/arXiv.2009.03393.

Ren Y, Li J, Qi Y (2026). “MerLean: An Agentic Framework for Autoformalization in Quantum Computation.” arXiv preprint 2602.16554. doi:10.48550/arXiv.2602.16554.

Rios FL, Mofa G, Kuipers J (2025). “Benchpress: A Versatile Platform for Structure Learning in Causal and Probabilistic Graphical Models.” Journal of Statistical Software, 114(12), 1–43. doi:10.18637/jss.v114.i12.

Scholze P (2022). “Liquid Tensor Experiment.” Experimental Mathematics, 31(2), 349–354. doi: 10.1080/10586458.2021.1926016.

She Z, Wang X (2026). KAPPA: Kit of Agents for Proof-Checking Papers with a Proof Assistant. Version 1.4.0, URL https://github.com/ziwaa-se/kappa.

Soltani Moakhar A, Gholami I, Springer M, JafariRaviz M, Hajiaghayi M (2026). “Beyond the Library: An Agentic Framework for Autoformalizing Research Mathematics.” arXiv preprint 2606.31134. doi:10.48550/arXiv.2606.31134.

The mathlib Community (2020). “The Lean Mathematical Library.” In Proceedings of the 9th ACM SIGPLAN International Conference on Certified Programs and Proofs, pp. 367–381. ACM, New York, NY, USA. doi:10.1145/3372885.3373824.

Wu Y, Jiang AQ, Li W, Rabe M, Staats C, Jamnik M, Szegedy C (2022). “Autoformalization with Large Language Models.” In S Koyejo, S Mohamed, A Agarwal, D Belgrave, K Cho, A Oh (eds.), Advances in Neural Information Processing Systems 35, pp. 32353–32368. Curran Associates, Inc. doi:10.52202/068431-2344.

Xin H, Guo D, Shao Z, Ren Z, Zhu Q, Liu B, Ruan C, Li W, Liang X (2024). “DeepSeek-Prover: Advancing Theorem Proving in LLMs through Large-Scale Synthetic Data.” arXiv preprint 2405.14333. doi:10.48550/arXiv.2405.14333.

Yang K, Swope A, Gu A, Chalamala R, Song P, Yu S, Godil S, Prenger RJ, Anandkumar A (2023). “LeanDojo: Theorem Proving with Retrieval-Augmented Language Models.” In A Oh, T Naumann, A Globerson, K Saenko, M Hardt, S Levine (eds.), Advances in Neural Information Processing Systems 36, Datasets and Benchmarks Track, pp. 21573–21612. Curran Associates, Inc. doi:10.52202/075280-0944.

Zheng K, Han JM, Polu S (2022). “miniF2F: A Cross-System Benchmark for Formal Olympiad-Level Mathematics.” In International Conference on Learning Representations (ICLR). URL https://openreview.net/forum?id=9ZPegFuFTFv.

## A Command reference

Most users never type these commands: in the guided mode, a Claude Code chat started in the kit’s folder runs them on requests such as “I have a new paper; call it mypaper” or “Continue mypaper”. They are listed here because they are the interface that the agents use and that a reader of the records may want to rerun. Table 15 lists the chat commands, which are Claude Code skills, and Table 16 the scripts; every script prints its options when called without arguments.

<table><tr><td>Command</td><td>What it does</td></tr><tr><td>/pv-setup</td><td>checks the Lean environment, has the architect list every numbered result with its place in the paper, its dependencies and its citations, and writes the records</td></tr><tr><td> $/ \mathsf { p v } \mathtt { - r u n }$ </td><td>the coordinator's loop: takes each result through draft, gate, proof and final audit, sends suspected problems to the issue checker, and records verdicts, until every result has one</td></tr><tr><td></td><td>/pv-resume continues after an interruption: reconciles half-finished agent outputs with the records and resumes the loop</td></tr><tr><td></td><td>/pv-revise handles a new version of the paper: compares it with the kept copy, resets only the affected results and marks their dependents for re-checking</td></tr><tr><td></td><td>/pv-report writes the author report, the worksheet of external inputs and the index of files to check, and has them fact-checked</td></tr><tr><td>/pv-status prints the progress table</td><td></td></tr><tr><td>Script</td><td>What it does</td></tr><tr><td>install.sh</td><td>creates a project, or upgrades one in place, keeping its records</td></tr><tr><td>scripts/pv_drive.sh</td><td>the unattended driver: short headless sessions, -auto-resume, -report</td></tr><tr><td>scripts/pv_lean.sh</td><td>compiles one file (check) or a module (build) with the pinned versions; refuses files with input-output actions</td></tr><tr><td>scripts/pv_status.py</td><td>all writes to the records (record, add-tool, reset, human-check, set-deps), the validator (validate), what can start now (next), what a revision changed (defs-changed, followup), and the progress table (table)</td></tr><tr><td>scripts/pv_audit_mech.py</td><td>the mechanical part of a final audit (Section 5.3)</td></tr><tr><td>scripts/pv_snapshot.py</td><td>saves and restores checkpoints of the records, and keeps the copy of the paper that revisions are compared with</td></tr><tr><td>scripts/pv_usage.py</td><td>token use and cost by role, model and result</td></tr><tr><td>scripts/pv_guard.py</td><td>the hook that checks writes and shell commands against the role making them</td></tr><tr><td>scripts/pv_settings.py</td><td>writes the permission file for the project's paths and checks an existing one</td></tr><tr><td>scripts/pv_cards_check.py</td><td>checks that every rule cited in an agent's card still exists</td></tr><tr><td>examples/tools/replay.py</td><td>rebuilds an archived run offline and reruns its mechanical audits</td></tr><tr><td>examples/tools/score.py</td><td>compares a run with an answer key</td></tr><tr><td></td><td>examples/tools/blind_audit.py scans a run's transcripts for forbidden reads and network access</td></tr><tr><td>examples/tools/sanitize.py</td><td>copies a run for publication without local paths, user names or e-mail addresses</td></tr></table>

Table 15: Chat commands.

Table 16: Scripts.