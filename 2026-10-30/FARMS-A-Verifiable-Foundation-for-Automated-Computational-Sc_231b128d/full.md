# FARMS: A Verifiable Foundation for Automated Computational Science

Di Wu<sup>1</sup>, Ling Liang<sup>2</sup> and Haizhao Yang<sup>1,3,∗</sup>

<sup>1</sup>Department of Mathematics, University of Maryland, College Park, USA

<sup>2</sup>Department of Mathematics, The University of Tennessee, Knoxville, USA

<sup>3</sup>Department of Computer Science, University of Maryland, College Park, USA

<sup>∗</sup>Corresponding author: hzyang@umd.edu

## Abstract

Automated computational science could accelerate discovery by selecting, adapting and developing numerical methods. This prospect requires mathematical knowledge that AI systems can interpret and reuse with explicit guarantees. Here we present FARMS, a faithful and eficient foundation that transforms computational mathematics into AI-ready computing knowledge. FARMS connects formal statements to source context, algorithmic meaning and quantitative guarantees through source preparation, reusable representations and semantic review. Across 171 targets in randomized numerical linear algebra and numerical partial diferential equations, 157 were accepted by the pipeline, with mean estimated costs of US \$0.85–2.29 per target across sources. Lean-checked implications from explicit source premises connect selected statements to computational skills, illustrated by a heat-flow solve and wave-response compression. FARMS therefore supports an afordable, reusable knowledge foundation for trustworthy automated computational science.

§ GitHub repo: https://github.com/Woody2357/FARMS-Computational-Science

Project page: https://woody2357.github.io/FARMS-Computational-Science/

## 1 Introduction

Artificial intelligence (AI) is expanding the ways in which scientific knowledge can be discovered and used<sup>1–3</sup>. One opportunity is to develop the computational methods that make new investigations possible. Advances in automated algorithm discovery <sup>4</sup>, from matrix multiplication<sup>5</sup> to program search<sup>6</sup>, show that AI can contribute to the machinery of computation itself. Extending this capability across computational science calls for a mathematical foundation that AI systems can use to select methods, combine their guarantees and reason about proposed improvements. Such a foundation would allow automated discovery to build on the accumulated knowledge of computational mathematics, with the conditions supporting each computational decision available for inspection and checking.

Randomized numerical linear algebra and numerical partial diferential equations (PDEs) are central to this opportunity. Randomized methods make large matrix problems tractable through sampling, sketching and low-rank approximation, supporting eficient analysis and computation at scale <sup>7;8</sup>. Numerical PDE methods connect mathematical models of physical systems to simulations through discretization, stability and error analysis <sup>9</sup>. Randomized sketching also yields preconditioners that accelerate the iterative solution of large least-squares and linear systems<sup>10;11</sup>, while low-rank representations compress families of simulated responses for repeated use<sup>12</sup>, connecting linear algebra to the computation and reuse of simulation results. Making their mathematical knowledge accessible to AI would provide reusable components for develop ing computational methods across a broad range of scientific problems.

Much of this knowledge is expressed through relationships among theorems, algorithms and surrounding explanations. A randomized approximation guarantee depends on how samples are drawn, which outputs the algorithm can construct and how success is measured. A PDE estimate may depend on boundary conditions, regularity and a particular error norm. These relationships determine whether a guarantee can support a computation. An AI-ready representation must therefore be faithful to the computational meaning of the source, preserving the conditions that connect a method to its guarantees. Constructing such representations must also be eficient if researchers are to extend them across the growing literature and share them as a community resource.

Formal mathematics provides a foundation for explicit representations and machine-checked reasoning<sup>13–16</sup>. Advances in neural theorem proving<sup>17–20</sup> and autoformalization<sup>21;22</sup> have broadened access to this foundation, supported by shared mathematical benchmarks <sup>23;24</sup>. These efforts increasingly encompass textbook libraries and applied mathematics <sup>25–28</sup>. Recent work in numerical analysis <sup>29</sup> and physics reasoning <sup>30</sup> report that compilation-based metrics overstate formalization quality and audits translations for semantic correctness and library reuse. Research on statement evaluation draws the same distinction between successful compilation and faithful representation of a source claim<sup>31;32</sup>. For computational science, these developments create an opportunity to connect mathematical literature with the practical reasoning needed to use numerical methods. This requires representations that retain algorithmic operations, quantitative guarantees and the dependencies needed to combine them in scientific workflows.

Here we present FARMS (Faithful Alignment and Representation of Mathematical Statements), a faithful and eficient foundation for automated computational science. FARMS turns mathematical literature into AI-ready computing knowledge by connecting formal statements to their source context and computational interpretation. Source preparation recovers the algorithms and definitions needed to interpret each claim; reusable domain representations support formalization; and compilation, semantic review and bounded repair assess the resulting state ments. Selected statements can then be combined through Lean-checked implications to form computational skills with explicit applicability conditions. This organization connects knowledge construction to its subsequent use in numerical reasoning and computation.

We study 171 targets from four sources in randomized numerical linear algebra and numerical PDEs<sup>7–9;33</sup>. The cumulative collection contains 157 pipeline-accepted formalizations, with mean estimated costs of US \$0.85–2.29 per target across sources. On six randomly selected targets, FARMS costs an estimated US \$2.84 per target, compared with US \$58.02 for our statementlevel adaptation of AutoformBot. We further demonstrate knowledge reuse in a steady heatflow solve and compression of wave-response libraries, where mathematical guarantees guide accuracy decisions. Together, these results support a practical route to a verifiable foundation for automated computational science: faithful computing knowledge, checked connections to applications and eficient construction that can lower the cost of community participation in trustworthy automated scientific discovery.

## 2 Results

## 2.1 Building AI-ready computing knowledge

FARMS builds AI-ready computing knowledge by linking each formal statement to the source material and computational interpretation needed to use it (Figure 1). Each representation makes the mathematical objects, applicability conditions and guarantees explicit, while retaining the evidence behind its interpretation. This organization provides a basis for checking how a computational claim is represented and how it can be connected to other claims.

Preparation. This stage assembles the source evidence needed to interpret each computational statement (Figure 1a). PDF or LaTeX sources are first converted into text that preserves mathematical expressions, algorithm descriptions and source locations. The Source auditor assesses whether a statement is accompanied by the definitions, assumptions and algorithmic context needed for formalization. When context is missing, the Retriever searches for relevant passages using deterministic retrieval routines guided by the identified gap. Here the target is the source text that fixes a statement’s meaning, so the retrieved object is evidence rather than a proof step<sup>34</sup>. The Evidence reviewer evaluates whether the retrieved material addresses that gap and returns supporting evidence to the Source auditor for re-auditing. This bounded loop links each statement to the source context needed to preserve its computational meaning, including algorithmic steps stated separately from its guarantee.

Formalization. The Formalizer uses the prepared evidence and shared domain library to construct a Lean statement and record its interpretation of the source. The library supplies reusable representations of mathematical objects and computational constructions, reducing the need to reconstruct common infrastructure for each target. The Lean checker compiles each candidate in the selected formal environment. Compilation establishes that a statement is well formed in the formal environment. The Semantic referee then examines whether the candidate preserves the source’s objects, assumptions, quantifiers, algorithmic constructions and quantitative guarantees. Semantic review assesses alignment with the source. A type or meaning mismatch is referred to the Repairer, after which the revised candidate returns for compilation and semantic review. The Consistency reviewer provides a final check on the disposition of findings raised during formalization and review. Review disagreements return to the Semantic referee for resolution before acceptance.

a Formalization pipeline  
![](images/fa713b1ba8f2eb6fcf69030b72bc1a8b9ba4355abd8904118a36335b5a92c695.jpg)  
b Corpus composition

![](images/56f8053ab04517f5515bc2dab4b44942331ecc6a346e1e0aa51065287895faa2.jpg)  
Figure 1: FARMS builds AI-ready computing knowledge for automated computational science. a, Source preparation links mathematical statements to supporting definitions, assumptions and algorithm descriptions. Formalization combines this evidence with reusable domain representations, followed by compilation, semantic review and bounded repair. Dashed paths show source re-auditing, repair and rechecking, resolution of review disagreement, and justified abstention. b, The collection comprises 171 targets, shown by domain (inner ring), source (middle) and statement type (outer): 87 from randomized numerical linear algebra and 84 from numerical PDEs. c, Computational structures represented in the collection, including algorithmic constructions and quantitative guarantees.

Statements that meet the acceptance criteria become reusable formal statements, with documented source warnings retained where applicable. A separate branch addresses persistent gaps reported by the Formalizer: the Abstention referee assesses whether abstention is justified, and the Explainer records the reason when it is upheld. These branches distinguish repairable formalization problems from cases in which the available evidence does not support completion.

Figure 1b summarizes the 171-target collection by domain, source and statement type. Figure 1c organizes its computational content into mathematical objects, conditions of validity and mathematical guarantees. Together, source-linked evidence and reusable formal representations make these relationships explicit for subsequent computational use.

## 2.2 Preserving computational meaning

Figure 2 illustrates how FARMS coordinates formalization, compilation and semantic review to preserve the mathematical meaning of textbook statements. The two examples follow distinct paths: recovering conditions omitted during formalization and identifying a defect in a source claim.

Figure 2a presents the formalization of the randomized low-rank approximation guarantee <sup>33</sup>. Under the source assumptions, the target requires that, with probability at least 0.85, a valid output $\tilde { U } _ { k }$ exists and every valid output satisfies $\| A - \tilde { U } _ { k } \tilde { U } _ { k } ^ { \mathsf { T } } A \| _ { F } \leq ( 1 + \varepsilon ) \| A - A _ { k } \| _ { F }$ , where $A _ { k }$ is a best rank-k approximation, ε is the error tolerance, and $\lVert \cdot \rVert _ { F }$ denotes the Frobenius norm. The success event represents an algorithm that can actually produce an admissible approximation. The first message from the Lean checker confirms that the candidate compiles, after which the Semantic referee examines whether its assumptions preserve the intended guarantee. The referee identifies two unintended cases admitted by the candidate. First, $A = 0$ and $k = 1$ leave no valid output, revealing an omitted rank restriction. Second, when the number of columns of A is $n = 1$ , the quantity $\alpha = k \log ( n ) / \varepsilon ^ { 2 }$ vanishes. The source expression log(α) is then undefined, whereas Lean’s definition assigns $\log ( 0 ) = 0$ . The Formalizer restores $k \leq \mathrm { r a n k } ( A )$ and $\alpha > 0$ using the domain of the source formula. The revised statement compiles, the Semantic referee confirms that the intended domain has been recovered, and the Consistency reviewer verifies that both findings have been resolved. With both conditions restored, the statement asserts a guarantee only where the algorithm can produce an admissible output and where the source’s own sampling formula is defined.

Figure 2b examines the optimality clause of the randomized matrix multiplication guarantee. The source claims that the normalized distribution $p _ { i } ^ { * } \propto \| A _ { : i } \| _ { 2 } \| B _ { i : } \| _ { 2 }$ minimizes the expected squared Frobenius error of the sampled approximation CR to AB, where $A _ { : i }$ and $B _ { i : }$ denote the ith column of A and row of $B ,$ respectively. Writing this error as $E ( q ) = \mathbb { E } _ { q } [ \| A B - C R \| _ { F } ^ { 2 } ]$ , the claim is $E ( p ^ { * } ) \leq E ( q )$ for every admissible distribution satisfying $q _ { i } \geq 0$ and $\textstyle \sum _ { i } q _ { i } = 1$ . The Formalizer preserves this stated domain, including zero probabilities, and the candidate passes compilation. The Semantic referee then constructs an explicit counterexample. For $A = [ 1 , 1 ]$ $B = [ 1 , - 1 ] ^ { \intercal }$ and one sample $( c = 1 )$ , the exact product is $A B = 0$ . The norm-based distribution $p ^ { * } = ( 1 / 2 , 1 / 2 )$ produces $C R = 2 { \mathrm { ~ o r ~ } } C R = - 2$ , each with probability $1 / 2$ , giving $E ( p ^ { * } ) = 4$ The admissible alternative $q = ( 1 , 0 )$ produces $C R = 1$ deterministically and gives $E ( q ) = 1$

![](images/b8839987c626db0d567f3f832e40096b33ac6733d0ef6b115e5d885b716043b3.jpg)  
<sup>Accepted</sup> <sup>outputs</sup> <sup>are</sup> <sup>formal</sup> <sup>statements</sup> <sup>with</sup> <sup>proof</sup> <sup>placeholders.</sup>Figure 2: FARMS refines formalization results and identifies defects in textbook claims. a, Agent interaction during the formalization of a randomized low-rank approximation guarantee. Although the initial candidate compiles, the Semantic referee identifies omitted rank and logarithm-domain restrictions that change the intended claim. Using evidence from the textbook context, the Formalizer restores these conditions. Subsequent compilation and seman tic review confirm the refinement, yielding an accepted formal statement. b, Agent interaction during the examination of the sampling optimality claim in randomized matrix multiplication guarantee. The candidate preserves the probability domain stated in the textbook and passes compilation. The Semantic referee nevertheless constructs a counterexample, and calls an independent source review to confirm a missing condition requiring positive sampling probability for every nonzero contribution, and this finding is kept as an explicit source warning. Together, the examples illustrate how FARMS uses source evidence and mathematical evaluation to distinguish repairable formalization errors from defects in the underlying claims.

Thus, the claimed optimality would require $4 \leq 1$ . An independent source review is called by the Semantic referee to confirm that the unbiasedness argument requires positive sampling probability for every nonzero contribution, $q _ { i } > 0$ whenever $A _ { : i } B _ { i : } \neq 0 ,$ a restriction absent from the source. The pipeline preserves the source claim and records the missing condition as an explicit warning, and the run consequently remains unaccepted.

These examples show why faithful computing knowledge must retain both the intended algorithmic conditions and any unresolved problems in the source. Recovering a missing condition can restore the meaning of a formalization and identifying a source defect makes a limitation explicit before the claim is reused.

## 2.3 From faithful statements to scientific computing

To examine how this knowledge can support scientific computation, we connected selected RNLA and PDE statements through Lean-checked implications to form reusable agent skills and used them in two numerical applications (Figure 3). Both applications require an accuracy decision: when an iterative solve is suficiently accurate, and when a compressed response library can be reused within a prescribed error tolerance.

The connection from statements to skills is shown in Fig. 3a. After establishing the reusable formal statement collection, we assembled existing statements and connecting proofs into procedures that an agent can apply to new cases. Each skill identifies the conditions to check, the method to execute, the guarantee it yields and the evidence supporting that guarantee. Task-specific data and numerical parameters are set at application time, while restrictions of the underlying proofs are fixed. Source statements remain explicit premises: each connecting proof establishes an implication from selected source propositions and application assumptions to a task-specific computational guarantee.

Figure 3b shows the mathematical connections behind these two decisions. The connecting proofs combine statements from our collection with one another or with results already available in Mathlib. Gaussian restricted singular-value bounds establish uniform energy comparisons, while Lax–Milgram and residual estimates connect them to solution accuracy. Wave energy conservation connects an error in the initial state to its error at every later time, while linear algebra supplies rank and residual bounds for compression. The two applications below use the energy-comparison and energy-preservation methods, respectively, with numerical constructions and checks implemented at application time.

The first application solves steady heat flow through a heterogeneous one-dimensional medium with conductivity contrasts of 100 and 1,000 (Figure 3c, left). The discretization has 512 unknown coeficients and a weighted gradient matrix with 8,192 rows. Combining Gaussian singular-value bounds with energy and residual estimates <sup>7;9</sup>, the skill selects a 2,048-row sketch to meet a prescribed conditioning requirement under the source and ideal-sampling assumptions. Its QR factor supplies a preconditioner, while the original equation’s residual determines when the requested relative discrete energy error of $1 0 ^ { - 6 }$ has been reached. Gaussian-preconditioned condition numbers were 8.83–8.95, compared with 99.6 and 964.0 for the constant-conductivity reference preconditioner. All six runs, using three prescribed seeds per contrast, passed the stopping criterion after 22 updates. The sketch thus supplies an auxiliary operator for a checked accuracy decision without reducing the unknown dimension.

The second application compresses a precomputed response library for a string fixed at both ends (Figure 3c, right). The prescribed family contains 384 initial displacement or velocity excitations, represented by an energy-coordinate matrix $F \in \mathbb { R } ^ { 2 5 6 \times 3 8 4 }$ . Queries are combina-

![](images/3b02aa64c9c1bdd786f90a43e78195f6ed79ce5cba897bedc9ef2d52bcd665f6.jpg)  
Conductivity contrast: 100 / 1,000 8,192 intervals · 512 coefficients Relative energy-norm error: 10⁻  
Preserved: objects · assumptions · quantifiers · norms · randomness · guarantees  
Steady heat flow: solve accurately

![](images/1be9cbc46825cb8859c596323dc14d61013d9885e205afd9744a58a0cf63b200.jpg)

![](images/191774165fbad932a042bb63997622dbc3c240e0a08bbddb62ab34529c123c21.jpg)  
128 spatial modes · 256 coordinates Input amplitude norm ≤ 1 Absolute energy-norm error: 0.01

a Build reusable skills  
![](images/10defea93a8e9c64b45bd462aa63f00c0c2f7e2c9f6b2aec18c5aebe70485d8d.jpg)  
b Connect the statements

## c Apply to two computational tasks

![](images/0f098695ea0cef7ecb5b446c419631b19bd14b269b5c6871b5fbb7ee6b69679b.jpg)

![](images/759be899c20317437ccbfb8930f5b18f550a1139b6685a207d92b41b9fb46e77.jpg)  
Figure 3: From faithful formal statements to numerical agent skills. a, Formal statements and connecting proofs form reusable skills that contain applicability conditions and link results to supporting evidence. b, Gaussian bounds and Lax–Milgram support an energy comparison that controls solve accuracy; wave energy conservation and a matrix residual bound support response compression. c, Representative applications of the skills. Left, a Gaussian sketch assists the steady heat-flow solve. All six runs meet the 10<sup>−6</sup> relative discrete energynorm target in a small number of updates. Right, 128 basis responses support wave predictions for 384 prescribed initial disturbances. The response library shrinks from 24 to 8.375 MiB while meeting the 0.01 energy-norm error target. The skills support accurate PDE solving and reusable response models with checkable error bounds.

tions Fα with $\| \alpha \| _ { 2 } \le 1$ . Gaussian sampling supplies candidate subspaces Q with coeficients $C = Q ^ { \mathsf { T } } F$ , and a candidate is accepted only when an exact residual check on the saved factors passes. For the ideal wave propagator $W _ { t }$ , energy conservation<sup>9</sup> and the matrix residual bound imply $\| W _ { t } F \alpha - ( W _ { t } Q ) C \alpha \| _ { E } \leq \| F - Q C \| _ { F } \| \alpha \| _ { 2 }$ . Thus one initial-factor check supports an absolute energy-error bound of 0.01 for all allowed queries and nonnegative times, under the stated source premise and exact-model assumptions. All three prescribed seeds selected 128 basis responses, with residual bounds of 0.00877–0.00938. Storing these responses and their combination coeficients reduced the 32-time response library from 24 to 8.375 MiB. This demonstrates accuracy-controlled reuse within a prescribed excitation family.

These applications demonstrate two ways in which formalized computing knowledge can guide a numerical method by forming mathematical skills. In the heat-flow problem, the connected guarantees support parameter selection and an accuracy-based stopping decision. In the wave problem, a bound on the initial representation error combines with energy conservation to control the error of subsequent responses. The collection thus provides reusable mathematical components for reasoning about computations across domains.

## 2.4 Quantitative results

We evaluated knowledge construction on 171 targets from four mathematical sources. The cumulative selected results contained 157 acceptances (91.8%), comprising 154 without source warnings and three with documented warnings (Figure 4a). A source warning records an unresolved issue in the source claim or its interpretation.

We measured resource use over each selected logical run, including source preparation, generation, review, repair and any continuation needed to complete that run. Mean estimated costs per planned target were \$1.92, \$0.85, \$2.29 and \$2.08 for the four sources, respectively (Figure 4b). The corresponding mean token totals were 0.256, 0.115, 0.311 and 0.311 million. For comparison, we adapted AutoformBot<sup>26</sup> to statement-level formalization where we restricted its scope to the selected targets and allowed target and auxiliary theorem proofs to be omitted, while retaining its native source reading, planning, auxiliary-definition construction, compilation, review and repair workflow. We evaluated this adapted version on six randomly selected targets. In total, FARMS used \$17.06 and 2.275 million tokens, whereas AutoformBot used \$348.09 and 63.071 million tokens. These totals correspond to \$2.84 versus \$58.02 and 0.379 versus 10.512 million tokens per target. AutoformBot’s recorded cost and token use were therefore 20.4 and 27.7 times those of FARMS in these configurations.

FARMS is organized around the computational content of mathematical statements. Its source preparation and semantic review connect mathematical guarantees to algorithmic operations, the conditions under which outputs can be constructed, and their accuracy and probability bounds. General-purpose autoformalization workflows pursue diferent objectives: building textbook-scale libraries with proofs <sup>26</sup>, formalizing competition problems at scale <sup>35</sup>, or improving terminological and syntactic consistency so that generated declarations fit an existing library <sup>36</sup>. For the computational guarantee, validity requires both output existence and the stated error bound. This emphasis is related to the explicit constructions evaluated by AMBER and the text-

![](images/9e075720f1dcd2f499ea99fa20d996ae699c5bf930d525b3be60dceb8e149ce0.jpg)  
Required output: a result built by the algorithm with the requested number of independent directions.  
For one sample, does the statement count a success?

![](images/82ef88b8445c49014f47ca95ea577e36f1d2e1d4eeff32e63ad0337e6d48d5eb.jpg)

![](images/4239b8b914e6600c129b953c1d1432d1497aa0c093c2603e3610947a2079ecc6.jpg)

## c Capturing algorithmic success

Randomized low-rank matrix approximation

Represent a matrix with fewer independent directions while controlling approximation error.

<table><tr><td>Possible outcome</td><td>FARMS</td><td>AutoformBot</td></tr><tr><td>Results exist and all meet the error limit.</td><td>Yes</td><td>Yes</td></tr><tr><td>A result exceeds the error limit.</td><td>No</td><td>No</td></tr><tr><td>No required approximation can be formed.</td><td>No</td><td>Yes</td></tr></table>

Figure 4: Knowledge-construction acceptance, resource use and algorithmic semantics. a, Cumulative selected outcomes for 171 targets; the 157 reported acceptances include three source-warning results (hatched). b, Mean cost and tokens per planned target, including source preparation, generation, review, repair and continuations within each selected logical run. Source and selected-set plots use separate scales. The six-target comparison uses our statementlevel adaptation of AutoformBot. c, Randomized low-rank matrix approximation illustrates FARMS’s computing-oriented design. The table contrasts output-existence and accuracy conditions in the two formal statements for the same sample.

book dependencies recovered by CAM-Bench<sup>27;28</sup>. Figure 4c illustrates this computing-oriented design through randomized low-rank matrix approximation. The task is to represent a matrix using fewer independent directions while controlling approximation error. The required output is a result constructed by the prescribed algorithm with the requested number of independent directions. For a fixed sample, FARMS’s formal statement requires such a result to exist and every allowed result to satisfy the error limit. The compared statement-level AutoformBot formalization retains the error condition but does not require such a result to exist. If no required result can be formed, there is no output that violates the error condition. FARMS explicitly excludes this case from the event counted as successful. The example makes the design emphasis concrete: an algorithmic success guarantee must include the conditions for constructing its output, together with the mathematical error bound.

Together with the numerical applications, these results support FARMS as a promising, faithful and eficient foundation for automated computational science. Its treatment of computational content makes algorithmic requirements explicit, while its low per-target cost can make the construction and extension of AI-ready computing knowledge more accessible. This combi nation provides a practical basis for broader community participation in trustworthy automated scientific discovery.

## 3 Discussion

FARMS connects computational mathematics to the decisions required in scientific computing. Its contribution is a reusable form of mathematical knowledge in which methods remain linked to their applicability conditions, quantitative guarantees and source evidence. The heat-flow and wave-response examples show how these links can support parameter selection, accuracy assessment and reuse of numerical results. Shared representations and source-guided review provide an organization for constructing this knowledge, with low per-target costs in the evaluated collection. These results make the expansion of an AI-ready mathematical foundation a practical direction for automated computational science.

The present collection covers four sources in two areas of computational mathematics. Extending its coverage will require handling diferences in exposition as well as mathematical content. Missing context and ambiguous formulations remain obstacles, alongside limitations of generation and semantic review. The numerical applications establish that selected statements can support useful computational decisions. Extending these connections to broader problems will require additional mathematical representations and application-specific checks.

A first direction is to integrate knowledge across textbooks. Organizing statements by source preserves provenance, but can leave equivalent results represented several times in diferent notation or at diferent levels of generality. A shared collection should identify these relationships while retaining distinctions in assumptions and guarantees. Formal equivalences and implications could link source-specific formulations to common representations, allowing users to find the version appropriate to a task <sup>37</sup>. This would reduce duplication and make results from different areas easier to combine without losing their original context. Related work addresses dependency-aware statement translation<sup>38</sup>, diagnosis and correction of semantic failures<sup>39</sup>, and autoformalization of scientific reasoning <sup>30</sup>. These approaches ofer complementary ways to extend the range of sources and strengthen the review of computational claims.

A second direction is to improve how the framework adapts to diferent styles of mathematical exposition. Some sources present computational results through prose, derivations or algorithm descriptions without a separate theorem statement. Feedback from source auditing, compilation and semantic review could help revise extraction and retrieval strategies for such material. Iterative refinement with language-model feedback provides a relevant precedent <sup>40</sup>. Faithfulness also requires preserving the distinction between a source claim and a repaired claim. FaithformBench evaluates both validity and invalidity preservation and reports that autoformalizers can silently turn invalid inputs into provable statements <sup>41</sup>. This motivates retaining source warnings and making any proposed correction explicit when constructing reusable mathematical knowledge. The objective is to recover the conditions needed for faithful representation across a wider range of sources while keeping the review efort manageable.

A third direction is to incorporate recent research articles. Textbooks provide established foundations, but new computational methods and sharper guarantees often appear first in the research literature. Retrieval-augmented generation <sup>42</sup> ofers a way to bring relevant documents into the construction process. Its value here would be access to new statements and supporting context. The retrieved material would still require source auditing, formal representation and review. Linking new results to the conditions and guarantees already in the collection could make advances easier to compare and reuse, while source and version records would support subsequent correction and extension.

The longer-term opportunity is to connect this evolving knowledge base to agents that propose, combine and evaluate computational methods <sup>4</sup>. The present skills demonstrate useful connections that such agents could draw on. Constructing new connections and discovering new methods remain open research tasks. A community-built collection of faithful statements, checked implications and executable procedures could make accumulated mathematical knowledge a more accessible resource for trustworthy automated scientific discovery.

## 4 Methods

## 4.1 Source collection

We studied statement formalization on 171 targets from four mathematical sources. The randomized numerical linear algebra collection comprised 32 targets from Drineas–Mahoney, 35 from Halko–Martinsson–Tropp and 20 from Martinsson–Tropp. The numerical partial diferential equation collection comprised 84 numbered theorems and lemmas from Larsson–Thom´ee.

## 4.2 FARMS workflow

FARMS first assembled a source package for each target. Preparation preserved the statement and the surrounding definitions, assumptions and algorithm descriptions needed to interpret it. Deterministic retrieval identified candidate source passages, and source auditing and evidence review assessed whether they supplied the required context. Unresolved questions triggered bounded retrieval and re-auditing. Retrieval serves diferent roles in formal mathematics: LeanDojo retrieves premises for proof generation, and LeanSearch retrieves Mathlib statements from natural-language queries<sup>34;43</sup>. In FARMS, the Retriever instead recovers source passages that determine a target statement’s meaning, including definitions and algorithm descriptions. The package retained the evidence supporting each interpretation, including dependencies between a mathematical guarantee and an algorithm described elsewhere in the source. A shared domain library supplied reusable formal representations of mathematical objects and computational constructions. The formalizer generated Lean declarations together with source-linked interpretations. Lean compilation checked whether the declarations were well formed in the selected environment. A separate semantic review assessed whether the candidate preserved the source objects, assumptions, quantifiers, algorithmic constructions and guarantees. Compilation failures and semantic findings entered bounded repair cycles, followed by renewed checks. Persistent context gaps were referred for source re-auditing and, when unresolved, abstention review. A consistency review checked the disposition of issues raised during the run. We defined acceptance as a final pipeline outcome of either “VERIFIED PASS” or “COMPILED WITH SOURCE WARNING”.

## 4.3 Models and execution environments

The principal model configuration used Anthropic’s Claude through the Claude Code command line interface, with the model selector claude-opus-5-5. Under the role-specific policy, the Source auditor, Formalizer, Semantic referee, Repairer, Consistency reviewer and Abstention referee used high efort. Initial evidence review also used high efort. Context-retrieval planning and quotation-selection calls used medium efort, as did the Explainer. The Retriever combined these planning calls with deterministic document searches and the Lean checker invoked the compiler directly. These tool operations had no language-model efort setting. Our statementlevel AutoformBot adaptation used the same Claude selector with high efort.

## 4.4 Connecting formal statements to computational skills

The downstream skills combined selected source propositions with additional Lean-checked implications and numerical procedures. Each skill specified applicability conditions, an executable method and the evidence required to support its conclusion. Source propositions remained explicit mathematical premises of the connecting proofs. The new proofs establish implications from those premises to task-specific guarantees. Dependency checks for the reported connecting proofs excluded “sorryAx” and retained only the standard logical axioms. The two applications were developed and executed locally using these skills. For each application, we distinguish the mathematical implication checked in Lean, the concrete assumptions supplied by the application, and the numerical checks performed outside Lean.

Reusable formal components already support several forms of mathematical reasoning, including verified convergence guarantees for optimization algorithms <sup>44</sup>. LEGO-Prover grows a library of verified lemmas for proof construction <sup>45</sup>, and Divide and Abstract learns abstractions for statement autoformalization<sup>46</sup>. FARMS uses shared domain representations to formalize source claims and connects selected claims through Lean-checked implications to numerical accuracy decisions.

## 4.4.1 Steady heat-flow application

Given a distributed source $f ( x ) = \sin ( \pi x ) + 0 . 2 \sin ( 5 \pi x ) + 0 . 1 \cos ( 3 \pi x )$ and zero temperature at both ends, the task is to solve

$$
- ( a _ { \eta } u ^ { \prime } ) ^ { \prime } = f \quad \mathrm { o n ~ } ( 0 , 1 ) , \qquad u ( 0 ) = u ( 1 ) = 0 ,
$$

with $a _ { \eta } ( x ) = \eta + ( 1 - \eta ) \sin ^ { 2 } ( 2 \pi x ) , \eta \in \{ 0 . 0 1 , 0 . 0 0 1 \}$ , giving conductivity contrasts, max $a _ { \eta } /$ min ${ { a } _ { \eta } } ,$ of 100 and 1,000. The temperature is approximated by 512 trial functions, constructed by linearly interpolating sampled sine modes on 8,192 mesh intervals. Thus, there are 512 unknown coeficients. The mesh creates the gradient and load data. The weighted gradient matrix $G \in \mathbb { R } ^ { 8 1 9 2 \times 5 1 2 }$ defines the discrete system $A c = b $ , where $A = G ^ { \mathsf { T } } G$ is $5 1 2 \times 5 1 2$ . The requested relative energy accuracy is $1 0 ^ { - 6 }$ for this discrete system. The spectral-equivalence skill turns the PDE data into a preconditioning and verification procedure. Using the Gaussian source statement, the application searches candidate sizes against a $1 \%$ failure budget and a requested condition-number upper bound of 30. It selects 2,048 sketch rows, with a suficient bound of approximately 15.3 under the source and ideal Gaussian sampling assumptions. The resulting sketch compresses the gradient data from 8,192 to 2,048 rows. A QR factorization of SG provides the preconditioner $B = ( S G ) ^ { \mathsf { T } } ( S G )$ , implemented by triangular solves. The unknown dimension remains 512. Combining the energy comparison with Lax–Milgram and residual estimates gives an accuracy test for the original equation. For contrasts of 100 and 1,000, a diagonal preconditioner from a constant conductivity reference gave condition numbers of 99.6 and 964.0, respectively. Gaussian preconditioning gave condition numbers of 8.83–8.95 across the two contrasts and three prescribed seeds per contrast. In these examples, the sketch provides an auxiliary energy that tracked the heterogeneous operator, maintaining similar preconditioned condition as the conductivity contrast increased. The skill connected this approximate operator to a verifiable accuracy target for the original discrete equation.

## 4.4.2 Wave-response compression application

The displacement satisfies $u _ { t t } = u _ { x x }$ on (0, 1), with zero boundary values. The task specifies localized initial displacement or velocity profiles at 64 positions and three widths, giving 384 excitation channels. Each profile is represented by 128 sine modes. Displacement and velocity together require 256 state coordinates. In coordinates whose Euclidean norm equals the physical energy norm, the columns of $F \in \mathbb { R } ^ { 2 5 6 \times 3 8 4 }$ describe these initial states. A query gives amplitudes α with $\| \alpha \| _ { 2 } \le 1$ , and asks for the complete response to $F \alpha$ . The target is an absolute energy-norm error of 0.01 over this amplitude class. The isometry skill makes this a problem of approximating the excitation map once. The application forms Gaussian combinations of its columns, orthogonalizes them and adds eight basis directions at a time. For each candidate basis $Q .$ , the coeficients $C = Q ^ { \mathsf { T } } F$ express the original excitations in that basis. The procedure accepts a factorization only after its residual meets the target, including an exact check of the saved factors. If $W _ { t }$ denotes wave propagation, the resulting prediction is $( W _ { t } Q ) C \alpha$ . The application propagates the basis states and reuses their responses for diferent queries. Wave energy conservation and the matrix residual bound ensure that checking the initial factorization supports every later time. All three prescribed seeds selected 128 basis directions, with checked Frobenius residuals of approximately 0.00877–0.00938. Each selected model was evaluated on the 384 individual excitations, 128 random amplitude combinations and a worst residual direction at six times. Replacing 384 excitation responses with 128 basis responses reduced float64 array storage from 24 to 8.375 MiB. The outcome is a smaller response library with an error budget that applies to all allowed queries and times. In this example, the learned basis provided a shared representation of the prescribed excitation family, reducing the storage required for its response library. The skill connected this compact representation to a verifiable error bound that remained valid for every allowed excitation combination throughout wave evolution.

## 4.5 Statement-level AutoformBot comparison and resource accounting

We compared FARMS with a statement-level adaptation of AutoformBot<sup>26</sup> on selected RNLA targets. Starting from the archived upstream revision, we restricted the task to the supplied target list and permitted ‘by sorry‘ in target and auxiliary theorem proofs. The adaptation retained the native source reading, retrieval, planning, auxiliary-definition construction, compi lation, review, repair, retry and merge workflow, with the execution and model-interface changes required by our environment. Definitions retained the original requirements. AutoformBot received the same complete source documents. For each target, we measured the resources of the selected logical run, including source preparation, generation, review, repair and recorded continuations. Mean resources were computed by dividing the total by the number of planned targets, including targets that were not accepted.

## 5 Acknowledgements

H.Y. was partially supported by the US National Science Foundation under awards IIS2520978, GEO/RISE-5239902, the DARPA D24AP00325, and the 2026 AIM Seed Grant from the AI Interdisciplinary Institute at Maryland (AIM). Approved for public release; distribution is unlimited.

## References

[1] Hanchen Wang, Tianfan Fu, Yuanqi Du, Wenhao Gao, Kexin Huang, Ziming Liu, Payal Chandak, Shengchao Liu, Peter Van Katwyk, Andreea Deac, et al. Scientific discovery in the age of artificial intelligence. Nature, 620(7972):47–60, 2023.

[2] Daniil A Boiko, Robert MacKnight, Ben Kline, and Gabe Gomes. Autonomous chemical research with large language models. Nature, 624(7992):570–578, 2023.

[3] Alex Davies, Petar Veliˇckovi´c, Lars Buesing, Sam Blackwell, Daniel Zheng, Nenad Tomaˇsev, Richard Tanburn, Peter Battaglia, Charles Blundell, Andr´as Juh´asz, et al. Advancing mathematics by guiding human intuition with ai. Nature, 600(7887):70–74, 2021.

[4] Alexander Novikov, Ngˆan V˜u, Marvin Eisenberger, Emilien Dupont, Po-Sen Huang, Adam Zsolt Wagner, Sergey Shirobokov, Borislav Kozlovskii, Francisco JR Ruiz, Abbas Mehrabian, et al. Alphaevolve: A coding agent for scientific and algorithmic discovery. arXiv preprint arXiv:2506.13131, 2025.

[5] Alhussein Fawzi, Matej Balog, Aja Huang, Thomas Hubert, Bernardino Romera-Paredes, Mohammadamin Barekatain, Alexander Novikov, Francisco J R. Ruiz, Julian Schrittwieser, Grzegorz Swirszcz, et al. Discovering faster matrix multiplication algorithms with reinforcement learning. Nature, 610(7930):47–53, 2022.

[6] Bernardino Romera-Paredes, Mohammadamin Barekatain, Alexander Novikov, Matej Balog, M Pawan Kumar, Emilien Dupont, Francisco JR Ruiz, Jordan S Ellenberg, Pengming Wang, Omar Fawzi, et al. Mathematical discoveries from program search with large language models. Nature, 625(7995):468–475, 2024.

[7] Per-Gunnar Martinsson and Joel A Tropp. Randomized numerical linear algebra: Foundations and algorithms. Acta Numerica, 29:403–572, 2020.

[8] Nathan Halko, Per-Gunnar Martinsson, and Joel A Tropp. Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. SIAM Review, 53(2):217–288, 2011.

[9] Stig Larsson and Vidar Thom´ee. Partial diferential equations with numerical methods. Springer, 2003.

[10] Vladimir Rokhlin and Mark Tygert. A fast randomized algorithm for overdetermined linear least-squares regression. Proceedings of the National Academy of Sciences, 105(36):13212– 13217, 2008.

[11] Xiangrui Meng, Michael A Saunders, and Michael W Mahoney. Lsrn: A parallel iterative solver for strongly over-or underdetermined systems. SIAM Journal on Scientific Computing, 36(2):C95–C118, 2014.

[12] Peter Benner, Serkan Gugercin, and Karen Willcox. A survey of projection-based model reduction methods for parametric dynamical systems. SIAM Review, 57(4):483–531, 2015.

[13] Jeremy Avigad. Mathematics and the formal turn. Bulletin of the American Mathematical Society, 61(2):225–240, 2024.

[14] Leonardo de Moura and Sebastian Ullrich. The Lean 4 theorem prover and programming language. In International Conference on Automated Deduction, pages 625–635. Springer, 2021.

[15] The Mathlib Community. The Lean mathematical library. In Proceedings of the 9th ACM SIGPLAN International Conference on Certified Programs and Proofs. ACM, 2020.

[16] Thomas Hales, Mark Adams, Gertrud Bauer, Tat Dat Dang, John Harrison, Le Truong Hoang, Cezary Kaliszyk, Victor Magron, Sean McLaughlin, Tat Thang Nguyen, et al. A formal proof of the kepler conjecture. In Forum of mathematics, Pi, volume 5, page e2. Cambridge University Press, 2017.

[17] Stanislas Polu and Ilya Sutskever. Generative language modeling for automated theorem proving. arXiv preprint arXiv:2009.03393, 2020.

[18] Trieu H Trinh, Yuhuai Wu, Quoc V Le, He He, and Thang Luong. Solving olympiad geometry without human demonstrations. Nature, 625(7995):476–482, 2024.

[19] ZZ Ren, Zhihong Shao, Junxiao Song, Huajian Xin, Haocheng Wang, Wanjia Zhao, Liyue Zhang, Zhe Fu, Qihao Zhu, Dejian Yang, et al. Deepseek-prover-v2: Advancing formal mathematical reasoning via reinforcement learning for subgoal decomposition. arXiv preprint arXiv:2504.21801, 2025.

[20] Thomas Hubert, Rishi Mehta, Laurent Sartran, Mikl´os Z Horv´ath, Goran Zuˇzi´c, Eric<sup>ˇ</sup> Wieser, Aja Huang, Julian Schrittwieser, Yannick Schroecker, Hussain Masoom, et al. Olympiad-level formal mathematical reasoning with reinforcement learning. Nature, 651 (8106):607–613, 2026.

[21] Yuhuai Wu, Albert Qiaochu Jiang, Wenda Li, Markus Rabe, Charles Staats, Mateja Jamnik, and Christian Szegedy. Autoformalization with large language models. Advances in Neural Information Processing Systems, 35:32353–32368, 2022.

[22] Albert Qiaochu Jiang, Sean Welleck, Jin Peng Zhou, Timothee Lacroix, Jiacheng Liu, Wenda Li, Mateja Jamnik, Guillaume Lample, and Yuhuai Wu. Draft, sketch, and prove: Guiding formal theorem provers with informal proofs. In The Eleventh International Conference on Learning Representations, 2023.

[23] Kunhao Zheng, Jesse Michael Han, and Stanislas Polu. MiniF2F: a cross-system benchmark for formal Olympiad-level mathematics. In International Conference on Learning Representations, 2022.

[24] Zhangir Azerbayev, Bartosz Piotrowski, Hailey Schoelkopf, Edward W Ayers, Dragomir Radev, and Jeremy Avigad. Proofnet: Autoformalizing and formally proving undergraduate-level mathematics. arXiv preprint arXiv:2302.12433, 2023.

[25] Fabian Gloeckle, Ahmad Rammal, Charles Arnal, Remi Munos, Vivien Cabannes, Gabriel Synnaeve, and Amaury Hayat. Automatic textbook formalization. arXiv preprint arXiv:2604.03071, 2026.

[26] Ahmad Rammal, Niket Patel, Fabian Gloeckle, Amaury Hayat, Julia Kempe, Remi Munos, Charles Arnal, and Vivien Cabannes. Formalizing mathematics at scale. arXiv preprint arXiv:2605.29955, 2026.

[27] Wentao Long, Yunfei Zhang, Chenyi Li, Li Zhou, Chumin Sun, and Zaiwen Wen. CAM-Bench: A benchmark for computational and applied mathematics in Lean. arXiv preprint arXiv:2605.17255, 2026.

[28] Bowen Yang, Yi Yuan, Chenyi Li, Ziyu Wang, Liangqi Li, Bo Zhang, Zhe Li, and Zaiwen Wen. Construction–verification: A benchmark for formalizing applied mathematics in Lean 4. arXiv preprint arXiv:2602.01291, 2026.

[29] Theodore Meek, Siyuan Ge, Di Qiu Xiang, Simon Chess, and Vasily Ilin. Formalizing numerical analysis: An agent pipeline and quality audit beyond kernel acceptance. arXiv preprint arXiv:2606.14000, 2026.

[30] Jordan Meadows, Lan Zhang, and Andr´e Freitas. Formalscience: Scalable human-in-theloop autoformalisation of science with agentic code generation in lean. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 23059–23077, 2026.

[31] Auguste Poiroux, Gail Weiss, Viktor Kunˇcak, and Antoine Bosselut. Reliable evaluation and benchmarks for statement autoformalization. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 17958–17980, 2025.

[32] Vasily Ilin and Brian Nugent. Sorries are not the hard part: An expert-review case study of a semi-autonomous formalization. arXiv preprint arXiv:2606.13925, 2026.

[33] Petros Drineas and Michael W Mahoney. Lectures on randomized numerical linear algebra. arXiv preprint arXiv:1712.08880, 2017.

[34] Kaiyu Yang, Aidan Swope, Alex Gu, Rahul Chalamala, Peiyang Song, Shixing Yu, Saad Godil, Ryan J Prenger, and Animashree Anandkumar. Leandojo: Theorem proving with retrieval-augmented language models. Advances in Neural Information Processing Systems, 36:21573–21612, 2023.

[35] Huaiyuan Ying, Zijian Wu, Yihan Geng, Jiayu Wang, Dahua Lin, and Kai Chen. Lean workbook: A large-scale lean problem set formalized from natural language math problems. Advances in Neural Information Processing Systems, 37:105848–105863, 2024.

[36] Lan Zhang, Xin Quan, and Andre Freitas. Consistent autoformalization for constructing mathematical libraries. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 4020–4033, 2024.

[37] Thibault Gauthier and Cezary Kaliszyk. Aligning concepts across proof assistant libraries. Journal of Symbolic Computation, 90:89–123, 2019.

[38] Hanyu Wang, Ruohan Xie, Yutong Wang, Guoxiong Gao, Xintao Yu, and Bin Dong. Aria: an agent for retrieval and iterative auto-formalization via dependency graph. In International Conference on Learning Representations, 2026.

[39] Haocheng Wang, Baiyu Huang, Yingjia Wan, Xiao Zhu, Xiaoyang Liu, Yinya Huang, and Zhijiang Guo. Formalrx: Rectify and examine semantic failures in autoformalization. In Forty-third International Conference on Machine Learning, 2026.

[40] Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegrefe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement with self-feedback. Advances in Neural Information Processing Systems, 36: 46534–46594, 2023.

[41] Rob Cornish, Iacopo Ghinassi, Po-Hung Yeh, Shuqi Liu, Qiyuan Xu, Haoxuan Yin, Dominik Wagner, Wenda Li, Yee Whye Teh, and Luke Ong. FaithformBench: Benchmarking faithfulness of mathematical chain-of-thought autoformalisation. arXiv preprint arXiv:2608.10916, 2026.

[42] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich K¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt¨aschel, et al. Retrievalaugmented generation for knowledge-intensive nlp tasks. Advances in Neural Information Processing Systems, 33:9459–9474, 2020.

[43] Guoxiong Gao, Haocheng Ju, Jiedong Jiang, Zihan Qin, and Bin Dong. A semantic search engine for Mathlib4. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 8001–8013. Association for Computational Linguistics, 2024.

[44] Chenyi Li, Ziyu Wang, Wanyi He, Yuxuan Wu, Shengyang Xu, and Zaiwen Wen. Formalization of convergence rates of four first-order algorithms for convex optimization. Journal of Automated Reasoning, 69:28, 2025.

[45] Haiming Wang, Huajian Xin, Chuanyang Zheng, Zhengying Liu, Qingxing Cao, Yinya Huang, Jing Xiong, Han Shi, Enze Xie, Jian Yin, Zhenguo Li, and Xiaodan Liang. LEGO-Prover: Neural theorem proving with growing libraries. In International Conference on Learning Representations, 2024.

[46] Marcus Min, Yeqi Gao, Wilson Sy, Zhaoyu Li, Xujie Si, and Osbert Bastani. Divide and abstract: Autoformalization via decomposition and abstraction learning. In International Conference on Learning Representations, 2026.