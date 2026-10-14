# A dataset of formalised theorem statements from the Annals of Mathematics

Thomas Browning Department of Mathematics Imperial College London t.l.browning@imperial.ac.uk

David Ledvinka<sup>∗</sup> Department of Mathematics Imperial College London d.ledvinka@imperial.ac.uk

Hang Lu Su Department of Mathematics Imperial College London hanglu.su@pm.me

Jack McKoen   
Department of Philosophy   
Carnegie Mellon University mckoen@cmu.edu

Katerina Hristova<sup>∗</sup> Department of Mathematics Imperial College London k.hristova@imperial.ac.uk

Justus Springer<sup>∗</sup> Department of Mathematics Imperial College London j.springer@imperial.ac.uk

Kevin Buzzard<sup>†</sup> Department of Mathematics Imperial College London k.buzzard@imperial.ac.uk

## Abstract

Recent progress in autoformalisation of mathematics using AI has been impressive. Measuring this progress requires datasets of formal statements that are both hard and trustworthy. We present ANNALSCHALLENGE: 50 formalised theorem statements written by hand in Lean 4 using Mathlib. Each is the main result of a paper published in the Annals of Mathematics since 2020. Every formalisation was manually reviewed and then audited by AI. We release them without formalised proofs.

## 1 Introduction

ANNALSCHALLENGE is a set of formalised theorem statements written in Lean 4 [Moura and Ullrich, 2021], using Mathlib [The mathlib Community, 2020], its mathematics library. These theorem statements are the main results of 50 papers published in volumes 191–203 (covering the period from 2020 to mid-2026) of the Annals ofMathematics, one of the top pure mathematics journals. They constitute a quarter of the 207 papers published in this period.

The purpose of ANNALSCHALLENGE is three-fold: to analyse the state of Mathlib and its readiness for the formalisation of contemporary pure mathematics research, to enrich Mathlib with missing definitions and theorems, and to gauge the progress of AI autoformalisation applied to existing proofs.

For these reasons, ANNALSCHALLENGE differs from other existing datasets and is the first formalisation challenge of its kind for contemporary pure mathematics research. Competition-based datasets include miniF2F (Olympiad-style problems from the AMC, AIME, and IMO) [Zheng et al., 2022], FIMO (IMO shortlisted problems) [Liu et al., 2023], and PutnamBench (from the William Lowell Putnam Competition) [Tsoukalas et al., 2024]. Other datasets include FormalMATH, spanning high-school to undergraduate-level mathematics [Yu et al., 2025], and ProofNet, targeted at undergraduate level mathematics [Azerbayev et al., 2023]. Domain-specific benchmarks like CombiBench (combinatorics) [Liu et al., 2025] and FATE (algebra) [Jiang et al., 2026] focus on a single area of mathematics rather than reaching across a wide spectrum of modern research mathematics.

Table 1: Distribution of papers by estimated difficulty of formalising their main theorem statements.
<table><tr><td>Difficulty</td><td>Sample</td><td>Formalised</td></tr><tr><td>1</td><td>40</td><td>36</td></tr><tr><td>2</td><td>37</td><td>11</td></tr><tr><td>3</td><td>56</td><td>3</td></tr><tr><td>4</td><td>44</td><td>0</td></tr><tr><td>5</td><td>30</td><td>0</td></tr><tr><td>Total</td><td>207</td><td>50</td></tr></table>

Research-level datasets spanning multiple areas of mathematics differ in their sources and construction. ArXivLean [Gehrunger et al., 2026, Dekoninck et al., 2026] automatically extracts and formalises statements from recent paper abstracts, followed by manual review. Its formalisation pipeline is instructed to exclude candidates requiring definitions not already present in Mathlib. FormalConjectures [Firsching et al., 2026] focuses on open problems without known informal proofs, although it also includes solved research problems for proof autoformalisation. SorryDB [Letson et al., 2026] draws tasks from existing Lean formalisation projects, asking models to supply missing proofs or fill gaps in partially completed proofs.

ANNALSCHALLENGE was released on GitHub [Browning et al., 2026], as well as on LEANEVAL, a public benchmark and leaderboard for autoformalisation of mathematics in Lean. On LEANEVAL, submitted proofs are checked for correctness without additional axioms in a sandboxed environment [Lean FRO, 2026].

## 2 Methodology

ANNALSCHALLENGE has the following key design principles:

Trust. We use Mathlib whenever possible, where definitions have been reviewed by the community.   
When we introduce local definitions, they are carefully reviewed by humans and AI.

Faithfulness. The formalised statement should be as close as possible to the informal source.

Correctness. When the informal statement has gaps or is incorrect, we correct the statement to match the author’s presumed intent. When the intended statement was unclear, we asked the authors of the paper.

## 2.1 Statement selection process

The first step of our selection process was a triage of papers according to the difficulty of formalising their main theorems (statements, not proofs) in volumes 191 to 203. Excluding retractions, errata and corrigenda, there are 207 papers published in this period. We read each paper and identified the main result(s). We then assigned a difficulty rating from 1 to 5, reflecting how hard we believe it would be to formalise the statement given the current state of Mathlib. A rating of 1 means the statement can be formalised directly with all relevant definitions already present in Mathlib. A rating of 5 means the statement relies on definitions which would be a major research project to formalise. A low rating does not imply that the theorem is easy to prove. For example, our dataset contains several theorems in finite group theory whose statements are elementary, but whose proof depends on the classification of finite simple groups. The proof of the classification spans over ten thousand pages and has not been formalised to date. Table 1 shows the distribution of papers along our difficulty scale.

We started with formalising the easiest statements while simultaneously aiming for a wide subject area coverage.

Table 2: Subject-area coverage of the 207 papers considered and the 50 papers whose main theorem statements were formalised. Papers may belong to more than one subject area.
<table><tr><td>Subject area (MSC)</td><td>Papers</td><td>Formalised</td><td>%</td></tr><tr><td>Algebraic geometry</td><td>54</td><td>8</td><td>14.8%</td></tr><tr><td>Number theory</td><td>50</td><td>20</td><td>40.0%</td></tr><tr><td>Partial differential equations</td><td>29</td><td>2</td><td>6.9%</td></tr><tr><td>Differential geometry</td><td>27</td><td>3</td><td>11.1%</td></tr><tr><td>Dynamical systems and ergodic theory</td><td>21</td><td>4</td><td>19.0%</td></tr><tr><td>Group theory and generalizations</td><td>20</td><td>8</td><td>40.0%</td></tr><tr><td>Topological groups, Lie groups</td><td>16</td><td>0</td><td>0.0%</td></tr><tr><td>Several complex variables and analytic spaces</td><td>15</td><td>2</td><td>13.3%</td></tr><tr><td>Probability theory and stochastic processes</td><td>14</td><td>4</td><td>28.6%</td></tr><tr><td>Combinatorics</td><td>13</td><td>8</td><td>61.5%</td></tr><tr><td>Convex and discrete geometry</td><td>10</td><td>6</td><td>60.0%</td></tr><tr><td>Manifolds and cell complexes</td><td>10</td><td>2</td><td>20.0%</td></tr><tr><td>Other</td><td>17</td><td>5</td><td>29.4%</td></tr><tr><td>Total</td><td>207</td><td>50</td><td>24.2%</td></tr></table>

## 2.2 Dataset statistics

We took the subject classification of each paper from the Annals ofMathematics website. Where it lists none, we fell back to MathSciNet. We count both primary and secondary subjects, so one paper can count towards several different areas. The rows in Table 2 therefore sum to more than the totals, which count each paper once.

## 2.3 Formalisation philosophy

We formalised the statements manually over the course of ten months. The formalisations are built upon definitions and theorems in Mathlib, our only dependency. Wherever new definitions were needed, we either added them to Mathlib in the course of the project or, in the rare cases where a definition was paper-specific and of no broader interest, we developed it locally. Every formalised statement is kept as close to its natural language form as possible. We also preserved the generality of the original papers and followed their naming conventions and notation.

## 2.4 Manual review

After each statement was formalised, we performed an in-depth manual review checking for various types of issues: incorrect edge cases, inconsistencies with the paper, missing hypotheses, or errors in our interpretation of the paper. Our team included mathematicians with expertise across a range of fields and prior experience with Lean formalisation. On a few occasions, we found issues with the statement in the paper, typically missing implicit hypotheses. In such cases, we consulted the authors to confirm the intended interpretation and that the further hypotheses neither altered the meaning of their intended result nor its correctness.

## 2.5 AI audit

After the 50 statements were manually formalised and reviewed, we ran multiple AI audits on each file. We first used Claude Fable 5 (Max Effort), then GPT 5.6 Sol Ultra, and then Claude Fable 5 (Max Effort) again. The prompt, provided in Appendix A, was written to search for misformalisations. For 32 of the papers, the AI audit found no issues of mathematical significance. For 7 of the papers, the AI audit revealed missing (mostly trivial) assumptions in the paper whose absence made the theorem false (see Appendix B for more details). For the remaining 11 papers, the AI audit revealed mistakes in our formalisation. We reviewed the AI findings manually and fixed all identified issues before public release. Since release, no further misformalisations have been found.

Table 3: Accepted community proof submissions on LEANEVAL as of 2 October 2026.
<table><tr><td>Model/Harness</td><td>Proofs formalised / 50</td><td>%</td></tr><tr><td>Axiom Prover (Axiom Math)</td><td>40/50</td><td>80.0%</td></tr><tr><td>Humanifa + GPT 5.6 sol</td><td>34/50</td><td>68.0%</td></tr><tr><td>NEAR AI w/ DeepSeek V4</td><td>33/50</td><td>66.0%</td></tr><tr><td>Tau (caj.al)</td><td>23/50</td><td>46.0%</td></tr><tr><td>Midas</td><td>20/50</td><td>40.0%</td></tr><tr><td>Total</td><td>40/50</td><td>80.0%</td></tr></table>

Misformalisation is a known difficulty in constructing datasets of formalised statements. For example, as of 2 October 2026, the FormalConjectures repository contains 3,604 formalised statements, and has had 511 misformalisation fixes merged.

## 3 Lessons from the formalisation process

To our surprise, we learned that mathematical practice in some areas of mathematics makes the work less amenable to direct formalisation. Authors sometimes write assuming conventions that may be clear to the experts in the field but are not explicitly stated. For example, implicitly assuming regularity criterion of certain objects such as measurability of functions or sets, or using definitions such as the Fourier transform or solution to a PDE which can have different meaning depending on the context - the definition of the Fourier transform could be an explicit function or distribution valued, and the definition of the solution of a PDE could be a classical solution or various notions of weak solutions. Which definition is intended is not always clear from the definitions given explicitly in the paper and the main theorem.

Going forward, a natural next goal would be to formalise the statements of the remaining 157 papers. This is not as easy as it sounds; Mathlib is still missing many fundamental definitions and theories. For example theorems about symplectic geometry, Riemannian geometry, Kähler geometry, Lie theory, intersection theory, infinity categories, moduli stacks, automorphic representations, and Shimura varieties are often essentially impossible to even state right now. Getting the relevant mathematics into Mathlib will be a long process.

## 4 Autoformalisation results

ANNALSCHALLENGE has been live on LEANEVAL since 17 August 2026. Table 3 shows the model standings as of 2 October 2026. Accepted formal proofs had been submitted for 40 of the 50 challenges. These results come from independent community submissions rather than a controlled evaluation: compute budgets, numbers of attempts, human involvement, and access to the source papers were not standardised across submitters.

Autoformalisation was successful on 28 of the 36 difficulty level 1 papers, on all 11 of the difficulty level 2 papers, and on one of the three difficulty level 3 papers (see Section 2.1 for our difficulty scale). Note that the difficulty range of the proofs is quite big, and may be uncorrelated with the difficulty of formalising the statement. Some papers use rather elementary proof techniques, while some others rely on large amounts of deep modern mathematics.

## 5 Conclusion and future work

## 5.1 Conclusion

In this work, we presented ANNALSCHALLENGE, a novel dataset of theorem statements in Lean 4. Unlike previous datasets, we cover a broad range of subject areas from modern research level mathematics.

Mathlib is beginning to be ready for mathematics at this level: we were able to state fifty recent main results on top of it, even if the difficulty distribution in Section 2.1 shows how much is still out of

reach. Within weeks of release, more than half of these statements had a formalised proof submitted to LEANEVAL, which we read as a sign of strong interest in the dataset, and as an indication that autoformalisation will play an important role in how mathematics is reviewed in the future.

## 5.2 Future work

We see three directions. The first is to state more theorems, especially in the areas underrepresented in Section 2.2. Several of these areas are barely covered by Mathlib, so this would require building out the library first. The second is to scale up with AI, for instance, by relying on definitions from Tau Ceti [Tau Ceti Project, 2026], an AI-managed library. Every definition would still need careful review, or the dataset would lose its trustworthiness. The third is to cover other journals, whose subject balance differs from that of the Annals ofMathematics.

## 6 Acknowledgements

This project is funded by Renaissance Philanthropy’s 2025 AI for Math Fund. We thank Renaissance Philanthropy and XTX Markets, the founding donor of the AI for Math Fund, for their support of this project.

We would like to thank Bhavik Mehta, Heather Macbeth and Mathias Stout for their help with formalising and reviewing the statements. We thank Kim Morrison for including our dataset into LEANEVAL. Finally, we thank the authors who answered questions about their work: Alex Cohen, Tuan Ngo Dac, Andrei Jaikin, Victor Lie, Andrew Lobb, Ezequiel Maderna, James Maynard, Bhargav Narayanan, Huy Pham, Anthony Poëls, Alex Smith, and Andrea Venturelli.

## References

Zhangir Azerbayev, Bartosz Piotrowski, Hailey Schoelkopf, Edward W. Ayers, Dragomir Radev, and Jeremy Avigad. Proofnet: Autoformalizing and formally proving undergraduate-level mathematics, 2023. URL https://arxiv.org/abs/2302.12433.

Manjul Bhargava. Galois groups of random integer polynomials and van der Waerden’s conjecture. Ann. of Math. (2), 201(2):339–377, 2025. ISSN 0003-486X,1939-8980. doi: 10.4007/annals.2025. 201.2.1. URL https://doi.org/10.4007/annals.2025.201.2.1.

Petter Brändén and June Huh. Lorentzian polynomials. Ann. ofMath. (2), 192(3):821–891, 2020. ISSN 0003-486X,1939-8980. doi: 10.4007/annals.2020.192.3.4. URL https://doi.org/10. 4007/annals.2020.192.3.4.

Thomas Browning, Katerina Hristova, David Ledvinka, Justus Springer, Hang Lu Su, Jack McKoen, and Kevin Buzzard. AnnalsChallenge. https://github.com/ImperialCollegeLondon/ AnnalsChallenge, 2026.

Alex Cohen. Fractal uncertainty in higher dimensions. Ann. ofMath. (2), 202(1):265–307, 2025. ISSN 0003-486X,1939-8980. doi: 10.4007/annals.2025.202.1.4. URL https://doi.org/10. 4007/annals.2025.202.1.4.

Jasper Dekoninck, Nikola Jovanovic, Tim Gehrunger, Kári Rögnvaldsson, Ivo Petrov, Chenhao Sun,´ and Martin Vechev. Beyond benchmarks: MathArena as an evaluation platform for mathematics with LLMs. In ICML 2026 AI4Math Workshop, 2026. URL https://arxiv.org/abs/2605. 00674.

Javier Fernández de Bobadilla and Tomasz Peł ka. Symplectic monodromy at radius zero and equimultiplicity of µ-constant families. Ann. ofMath. (2), 200(1):153–299, 2024. ISSN 0003-486X,1939- 8980. doi: 10.4007/annals.2024.200.1.4. URL https://doi.org/10.4007/annals.2024. 200.1.4.

Moritz Firsching, Paul Lezeau, Salvatore Mercuri, Miklós Z. Horváth, Yaël Dillies, Calle Sönne, Eric Wieser, Fred Zhang, Thomas Hubert, Blaise Agüera y Arcas, and Pushmeet Kohli. Formal Conjectures: An open and evolving benchmark for verified discovery in mathematics. arXiv preprint arXiv:2605.13171, 2026. URL https://arxiv.org/abs/2605.13171.

Keith Frankston, Jeff Kahn, Bhargav Narayanan, and Jinyoung Park. Thresholds versus fractional expectation-thresholds. Ann. ofMath. (2), 194(2):475–495, 2021. ISSN 0003-486X,1939-8980. doi: 10.4007/annals.2021.194.2.2. URL https://doi.org/10.4007/annals.2021.194.2.2.

Tim Gehrunger, Jasper Dekoninck, and Martin Vechev. ArXivLean: How well can LLMs formally prove research math? MathArena, April 2026. URL https://matharena.ai/arxivlean/.

W. T. Gowers, Ben Green, Freddie Manners, and Terence Tao. On a conjecture of Marton. Ann. of Math. (2), 201(2):515–549, 2025. ISSN 0003-486X,1939-8980. doi: 10.4007/annals.2025.201.2.5. URL https://doi.org/10.4007/annals.2025.201.2.5.

Larry Guth and James Maynard. New large value estimates for Dirichlet polynomials. Ann. ofMath. (2), 203(2):623–675, 2026. ISSN 0003-486X,1939-8980. doi: 10.4007/annals.2026.203.2.6. URL https://doi.org/10.4007/annals.2026.203.2.6.

Jiedong Jiang, Wanyi He, Wang Yuefeng, Guoxiong Gao, Yongle Hu, Jingting Wang, Nailin Guan, Peihao Wu, Bryan Dai, Liang Xiao, and Bin Dong. FATE: A formal benchmark series for frontier algebra of multiple difficulty levels. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=3bD19r4jqh.

Lean FRO. LeanEval, 2026. URL https://github.com/leanprover/lean-eval. Accessed 1 September 2026.

Austin Letson, Leopoldo Sarra, Auguste Poiroux, Oliver Dressler, Paul Lezeau, Dhyan Aranha, Frederick Pu, Aaron Hill, Miguel Corredera Hidalgo, Julian Berman, George Tsoukalas, and Lenny Taelman. SorryDB: Can AI provers complete real-world Lean theorems? In Proceedings ofthe 43rd International Conference on Machine Learning, volume 306 of Proceedings ofMachine Learning Research, pages 65604–65627. PMLR, 2026. URL https://proceedings.mlr.press/v306/ letson26a.html.

Chengwu Liu, Jianhao Shen, Huajian Xin, Zhengying Liu, Ye Yuan, Haiming Wang, Wei Ju, Chuanyang Zheng, Yichun Yin, Lin Li, Ming Zhang, and Qun Liu. Fimo: A challenge formal dataset for automated theorem proving, 2023. URL https://arxiv.org/abs/2309.04295.

Junqi Liu, Xiaohan Lin, Jonas Bayer, Yael Dillies, Weijie Jiang, Xiaodan Liang, Roman Soletskyi, Haiming Wang, Yunzhou Xie, Beibei Xiong, Zhengfeng Yang, Jujian Zhang, Lihong Zhi, Jia Li, and Zhengying Liu. Combibench: Benchmarking llm capability for combinatorial mathematics, 2025. URL https://arxiv.org/abs/2505.03171.

Leonardo de Moura and Sebastian Ullrich. The lean 4 theorem prover and programming language. In André Platzer and Geoff Sutcliffe, editors, Automated Deduction – CADE 28, pages 625–635, Cham, 2021. Springer International Publishing. ISBN 978-3-030-79876-5.

Bryce Orloski, Naser Talebizadeh Sardari, and Alexander Smith. New lower bounds for the Schur-Siegel-Smyth trace problem. Math. Comp., 94(354):2005–2040, 7 2025. ISSN 0025-5718. doi: 10.1090/mcom/4004. URL https://doi.org/10.1090/mcom/4004.

Jinyoung Park and Huy Tuan Pham. On a conjecture of Talagrand on selector processes and a consequence on positive empirical processes. Ann. ofMath. (2), 199(3):1293–1321, 2024. ISSN 0003-486X,1939-8980. doi: 10.4007/annals.2024.199.3.6. URL https://doi.org/10.4007/ annals.2024.199.3.6.

Alexander Smith. Algebraic integers with conjugates in a prescribed distribution. Ann. of Math. (2), 200(1):71–122, 2024. ISSN 0003-486X,1939-8980. doi: 10.4007/annals.2024.200.1.2. URL https://doi.org/10.4007/annals.2024.200.1.2.

Tau Ceti Project. Tau Ceti, 2026. URL https://taucetiproject.github.io/TauCeti/. Accessed 1 September 2026.

The mathlib Community. The Lean Mathematical Library. In Proceedings ofthe 9th ACM SIGPLAN International Conference on Certified Programs and Proofs, CPP 2020, New Orleans, LA, USA, January 2020. ACM. doi: 10.1145/3372885.3373824. URL https://doi.org/10.1145/ 3372885.3373824.

George Tsoukalas, Jasper Lee, John Jennings, Jimmy Xin, Michelle Ding, Michael Jennings, Amitayush Thakur, and Swarat Chaudhuri. Putnambench: Evaluating neural theorem-provers on the putnam mathematical competition, 2024. URL https://arxiv.org/abs/2407.11214.

Zhouliang Yu, Ruotian Peng, Keyi Ding, Yizhe Li, Zhongyuan Peng, Minghao Liu, Yifan Zhang, Zheng Yuan, Huajian Xin, Wenhao Huang, Yandong Wen, Ge Zhang, and Weiyang Liu. Formalmath: Benchmarking formal mathematical reasoning of large language models, 2025. URL https://arxiv.org/abs/2505.02735.

Kunhao Zheng, Jesse Michael Han, and Stanislas Polu. Minif2f: a cross-system benchmark for formal olympiad-level mathematics, 2022. URL https://arxiv.org/abs/2109.00110.

## A AI audit prompt

The prompt for the AI audit was as follows:

We are working on a project to formalise (in Lean) the statement of the main theorems from various Annals of Mathematics papers. We are focussing on just one paper here. The pdf of the paper is in the pdf directory, and the corresponding formal statement is proved in the file <FILE NAME>. The statements of the main sorried result or results in Lean should be equivalent to the corresponding statements in the paper, and as faithful to the statement in the paper as is reasonably possible. Please review the formalisation and docstring for any errors. In particular you should look for subtle errors in the theorem statement (including any definitions that they rely on) that make them wrong or unfaithful to the statement in the paper. You may want to run tests by checking special cases of definitions, or by proving or disproving special cases of the theorems. Please do not attempt to fill in the sorry following the paper – that would be a major project! But if you can find a simpler proof or disproof of one of the sorries, thus indicating a misformalization, then feel free – this would be a wonderful outcome. We are just auditing the formalization, not filling in the sorries from scratch. Please do not ask any questions when working, but instead make commits to the <BRANCH NAME> branch directly. Humans will review the suggestions afterwards. Do not push anything!

## B Missing or implicit hypotheses in papers

We record the (essentially all trivial) slips in the published versions of the theorems we formalised.

1. Lorentzian polynomials [Brändén and Huh, 2020], Theorem 2.25: The paper allows n and d to be nonnegative, but the theorem is false for $n = 0$ and $d = 2$ , where

$\mathring { L } _ { 0 } ^ { 2 } = \{ f \in P _ { 0 } ^ { 2 } \ | \ \mathcal { H } _ { f }$ is nonsingular and has exactly one positive eigenvalue} $\} = \emptyset$

since the $0 \times 0$ Hessian matrix of $f$ cannot have a positive eigenvalue, but $L _ { 0 } ^ { 2 } = S _ { 0 } ^ { 2 }$ contains the zero polynomial, since the zero polynomial is homogeneous of degree 2.

2. Thresholds versusfractional expectation-thresholds [Frankston et al., 2021], Theorem 1.1: The assumption that $\ell ( \mathcal { F } ) \geq 2$ is not stated explicitly in the paper, but it is necessary: if $\ell ( \mathcal { F } ) = 1$ , then $\log ( \ell ( \dot { \mathcal { F } } ) ) = 0 .$ , whereas $p _ { c } ( \mathcal { F } ) \in ( 0 , 1 )$ ), so the inequality cannot hold.

3. On a conjecture ofTalagrand on selector processes and a consequence on positive empirical processes [Park and Pham, 2024], Theorem 1.2: The assumption that

$$
0 < \mathbb { E } \operatorname* { s u p } _ { \lambda \in \Lambda } \sum _ { i \in X _ { p } } \lambda _ { i } < \infty
$$

is not explicitly stated in the theorem but used in the last step of the proof where the inequality $\mathbb { E } \geq ( L / L ^ { \prime } ) \mathbb { E }$ is a contradiction for $L > L ^ { \prime }$

4. On a conjecture of Talagrand on selector processes and a consequence on positive empirical processes [Park and Pham, 2024], Theorem 1.3: The requirement that the collection C be finite is not explicitly stated in the paper. However, C must at least be assumed to be countable in order for the sum to make sense, and the collections C produced by the proof are indeed finite.

5. Algebraic integers with conjugates in a prescribed distribution [Smith, 2024]: The definition of λ<sub>SSS</sub> contains a typo that was caught during human review. The paper said “least real number”, but this is clearly a typo, as this set does not have a lower bound. Defining it as the greatest real number is consistent with how this constant is defined in another paper by the same author [Orloski et al., 2025].

6. Symplectic monodromy at radius zero and equimultiplicity of µ-constant families [Fernández de Bobadilla and Peł ka, 2024], Theorem 1.1: We need to require the constant coefficients to vanish. This is equivalent to saying that $f _ { t }$ is a germ $( \bar { \mathbb { C } } ^ { n } , 0 ) \to ( \mathbb { C } , 0 )$ i.e. that it is a family of isolated hypersurface singularities through the origin, which is mentioned both in the abstract and in Corollary 1.3, but not explicitly in Theorem 1.1.

7. On a conjecture of Marton [Gowers et al., 2025]: The set A must be nonempty, but this is left implicit in the paper.

8. Galois groups ofrandom integer polynomials and van der Waerden’s Conjecture [Bhargava, 2025], Theorem 1: We must impose $n \geq 3 ,$ , as the result is false as stated for $n = 2 .$

9. Fractal uncertainty in higher dimensions [Cohen, 2025], Theorem 1.1: The sets X and Y are intended to be measurable, but this is not explicitly stated. This was caught during human review and clarified with the authors.

10. New large value estimates for Dirichlet polynomials [Guth and Maynard, 2026], Theorem 1.1: The inequality may be false at $T = \dot { 1 }$ , so we must impose $T > 1$