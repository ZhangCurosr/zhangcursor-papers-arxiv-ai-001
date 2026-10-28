# HGPRGDel: GPR-Aware Learning and Ranking for Growth-Coupled Gene Deletion Design

Ziwei Yang<sup>1</sup> and Takeyuki Tamura<sup>1</sup>

<sup>1</sup>Bioinformatics Center, Institute for Chemical Research, Kyoto University, Kyoto, Japan

<sup>∗</sup>To whom correspondence should be addressed. E-mail: tamura@kuicr.kyoto-u.ac.jp.

## Abstract

Designing microbial strains for growth-coupled production, in which cellular growth is linked to the synthesis of a target compound through genetic interventions, is a fundamental goal in metabolic engineering. Constraint-based methods can identify growth-coupled deletion strategies, but design for each target typically requires a separate search across a large number of possible gene-deletion combinations. This motivates a reusable approach that learns from previously evaluated strain designs to prioritize candidate interventions before downstream constraint-based evaluation. We present a reproducible protocol for growth-coupled gene deletion design that learns and ranks gene-deletion priorities for each target metabolite. Using computationally curated deletion strategies as learning resources, the protocol generates continuous deletion-priority rankings, constructs candidate strategies at adjustable search space budgets, and evaluates these candidates through constraint-based metabolic analysis. Its learning component, HGPRGDel, explicitly represents gene–protein–reaction (GPR) logic, capturing how multiple genes can jointly support a reaction or alternative genes can maintain its activity, while integrating molecular and reaction-level information. Across e coli core, iMM904, and iML1515, the protocol achieves the highest macro-F1 and AUC among the evaluated learning configurations. HGPRGDel’s rankings recovered an average of 90% of the reference deletion genes from MetNetComp by examining fewer genes than the evaluated baseline methods. By separating learned intervention prioritization from downstream constraint-based assessment, the protocol provides a reproducible framework for reusing strain-design knowledge while controlling the size of the candidate space and retaining mechanistic evaluation of growth-coupled production.

Keywords growth-coupled gene deletion prediction; gene–protein–reaction associations; genome-scale metabolic models; reaction representation learning; deletion prioritization; metabolic engineering

## Introduction

Cells rely on a network of biochemical reactions to support growth and produce metabolites. These reactions are catalyzed by enzymes encoded by genes, linking genetic information to metabolic activity. Genome-scale metabolic models (GEMs) are computational models that represent cellular metabolism as a network of biochemical reactions, describing how metabolites are converted into one another and how genes are associated with these reactions [1, 2]. Gene–protein–reaction (GPR) associations specify how genes support reaction activity, including cases where multiple genes are jointly required or alternative genes can independently support the same reaction. These relationships are important for evaluating genetic interventions because deleting a gene can alter or eliminate the activity of associated reactions. Constraint-based modeling (CBM) uses mass-balance relationships and constraints on reaction fluxes to identify feasible metabolic states, while flux balance analysis (FBA) estimates metabolic flux distributions under specified constraints and objectives [3, 4, 5]. Together, these methods provide a mechanistic basis for analyzing how gene deletions afect cellular growth and metabolite production.

Growth-coupled production (GCP) is a strain-design objective in which cellular growth is coupled to the production of a target compound, such that maintaining growth requires the cell to produce that compound. This coupling can provide a selective advantage to productive strains and is therefore useful for designing microbial cell factories [6, 7]. Computational strain design seeks gene-deletion strategies that establish this coupling by changing the available metabolic routes. A successful strategy must preserve suficient metabolic activity for growth while constraining the network so that target production is linked to growth. Identifying such strategies is challenging because each gene-deletion combination can produce a diferent metabolic state, and the number of possible combinations grows exponentially with the number of genes.

This combinatorial challenge motivates a complementary approach: rather than replacing constraint-based strain design, previously evaluated designs can be used to prioritize candidate interventions for downstream evaluation. We develop a reusable protocol that learns gene–target deletion priorities from computationally curated strategies and uses these priorities to construct candidate deletion sets at adjustable searchspace budgets. Each candidate is then evaluated independently using constraint-based metabolic analysis to determine whether it satisfies the growth-coupling objective. The protocol thus separates learning-based candidate prioritization from constraint-based evaluation, allowing the learning stage to guide the search without replacing mechanistic assessment.

![](images/9a6e03bae09008ad3b5fd62d2baaecaa26fc2d09080da7f04e4af98df98a5525.jpg)  
Figure 0 Graphical Abstract. HGPRGDel is the growth-coupled gene deletion protocol’s learning model; MetNetComp provides its reference-strategy layer. Molecular representations, diferentiable GPR logic, and reaction-centric hypergraph learning produce continuous gene-retention scores and deletion-priority rankings, which define an ordered search space for candidate deletion strategies and downstream constraint-based evaluation.

The MetNetComp database [8] provides reference genedeletion strategies across GEM–target combinations. These curated strategies serve as learning targets for estimating gene– target deletion priorities and as reference sets for assessing the recovery of known designs. By learning from this resource, the protocol reuses information from previously evaluated strain designs rather than treating each target-specific search as an isolated problem. However, learning useful deletion priorities requires accounting for how gene deletions afect the metabolic network, not merely which genes appear in a reference strategy.

A central consideration is the GPR structure, which encodes the functional relationships between genes and metabolic reactions. GPR rules are Boolean expressions that describe how gene states determine reaction activity [9, 10]. In an AND rule, all genes encoding components of an enzyme complex are required for the associated reaction to remain active; in an OR rule, alternative isozymes can preserve reaction activity when one gene is deleted. Consequently, the efect of a deletion depends not only on gene identity but also on its role in the GPR rules of the afected reactions.

Existing learning-based predictors, including DeepGDel [11] and GraphGDel [12], incorporate gene–reaction relationships and metabolic network context. However, they do not explicitly represent the compositional efects of GPR AND/OR logic within the learned reaction representation. This motivates a learning component that explicitly accounts for GPR structure while integrating reaction-level network context and molecular information.

Fig. 1 illustrates the importance of GPR logic through a toy model. Reaction $r _ { 4 }$ follows an AND rule $( g _ { 4 } \land g _ { 5 } )$ , representing an enzyme complex, whereas $r _ { 5 }$ follows an OR rule $\left( g _ { 3 } \lor g _ { 4 } \right)$ , representing alternative isozymes. Among all $2 ^ { 5 } = 3 2$ deletion combinations, only two strategies, $\left\{ g _ { 3 } , g _ { 4 } \right\}$ and $\{ g _ { 3 } , g _ { 4 } , g _ { 5 } \}$ , achieve growth-coupled production of the target metabolite. Deleting both $g _ { 3 }$ and $g _ { 4 }$ disables $r _ { 5 }$ by removing isozyme redundancy, while deleting $g _ { 4 }$ disables $r _ { 4 }$ by disrupting the enzyme complex. The example illustrates how the same gene can afect multiple reactions through diferent Boolean relationships, making the functional context of a gene deletion important for identifying growth-coupled strategies.

Because GPR rules determine how gene states afect reaction activity, reactions provide a natural interface between gene-level interventions and metabolic-network behavior. We therefore formulate the learning component around reaction representations (Fig. 1). For each gene–target metabolite pair, HGPRGDel aggregates reaction-level representations associated with the candidate gene and target metabolite to produce a continuous retention score $\hat { y } _ { g , m } \in [ 0 , 1 ]$ . The corresponding deletion priority is defined as $\begin{array} { r c l } { d _ { g , m } } & { = } & { 1 - \hat { y } _ { g , m } } \end{array}$ with higher values indicating higher deletion priority. Ranking genes by decreasing $d _ { g , m }$ defines an ordered intervention space from which nested candidate deletion strategies can be constructed at diferent search-space budgets. These candidates are subsequently evaluated using constraint-based metabolic analysis.

The learning component, HGPRGDel, integrates three complementary sources of biological information:

• GPR-M (SoftBooleanAggregator): Transforms each reaction’s Boolean GPR rule into a diferentiable reaction embedding through tree-structured AND/OR aggregation, encoding enzyme-complex and isozyme logic as learnable inductive biases.

![](images/43c7aed81b928147ff6ba763d1e81885e416187bf7d9dcba1598c2d2a0aa3eb1.jpg)  
Figure 1 Toy model illustrating GPR-dependent deletion patterns. Circles represent metabolites, and rectangles represent reactions $( r _ { 7 }$ for growth and $r _ { 8 }$ for target production). The GPR rule for $r _ { 4 }$ requires both $g _ { 4 }$ and $g _ { 5 }$ (enzyme complex), whereas the rule for $r _ { 5 }$ requires either $g _ { 3 }$ or $g _ { 4 }$ (alternative isozymes). Deleting $\{ g _ { 3 } , g _ { 4 } \}$ disables both reactions, removing the alternative route and the enzyme-complexassociated reaction. Among all $2 ^ { 5 } \ : = \ : 3 2$ deletion combinations, only two achieve growth-coupled production of $m _ { 5 }$ , illustrating how GPR logic shapes feasible deletion strategies.

Graph-M (reaction-centric hypergraph): Represents each reaction as a hyperedge over its participating metabolites and propagates network context through hypergraph convolution conditioned on GPR-derived attributes, preserving reaction-level multi-metabolite structure.

• Gene-M / Meta-M (frozen ESM-2 and ChemBERTa): Provide pretrained representations of protein sequences and metabolite SMILES, respectively, without fine-tuning these molecular encoders on the limited per-GEM labels.

Together, these components implement the learning stage of the protocol. They prioritize candidate interventions but do not replace the downstream constraint-based evaluation used to determine whether a constructed deletion set achieves the intended growth-coupling phenotype.

This work makes three contributions:

1. Protocol. We present a reusable protocol for growthcoupled gene deletion design that transfers information from computationally curated deletion strategies to gene–target prioritization tasks. The protocol combines learned deletion priorities, nested candidate construction, and downstream constraint-based evaluation within a reproducible workflow.

2. Learning component. We develop HGPRGDel, a GPR-aware learning component within the protocol that integrates diferentiable GPR logic, reactioncentric hypergraph structure, and pretrained molecular representations to generate gene–target deletion priorities.

3. Evaluation and evidence. We evaluate the protocol across three GEMs and show that HGPRGDel achieves the highest macro-F1 and AUC among the evaluated learning methods on each GEM. The learned rankings recover computationally curated MetNetComp reference gene sets and support adjustable candidate-space restriction. Ablation and representation analyses further characterize the contributions of GPR, molecular, and topological information to the learned rankings.

## Related Work

## Constraint-Based Methods for Growth-Coupled Strain Design

Flux balance analysis [3] provides the steady-state metabolic flux prediction that underpins computational strain design. Bilevel programming methods such as OptKnock [13] formulate growth-coupled production as an inner/outer optimization, in which the inner program maximizes growth and the outer program identifies gene knockouts that enforce production at the growth optimum. Subsequent methods improved search eficiency through local search, preprocessing, mixed-integer formulations, minimum-cut-set enumeration, and reactionnetwork-based restrictions [14, 15, 16, 17, 18, 19, 20]. These methods established constraint-based optimization as a general route for growth-coupled strain design, but the search remains target-specific. Each new GEM–target combination requires a separate optimization or search procedure.

## Machine Learning for Metabolic Gene Perturbation

Machine learning has been applied to metabolic modeling for tasks including pathway flux prediction [21] and enzyme kinetics [22]. Direct learning of gene deletion priorities from curated design strategies is a more recent direction. DeepGDel [11] introduced supervised learning of gene deletion strategies using sequence-based representations, while GraphGDel [12] added metabolic-network topology through graph-based representations. These methods established learning-based deletion prioritization as a complementary approach to target-specific optimization. An important question is how to represent the biological relationships that connect gene perturbations to reaction activity. GPR associations provide an explicit representation of this mapping through Boolean AND/OR rules. Reaction-centric and hypergraph representations have been explored for metabolic-network learning, including gene essentiality prediction [23] and reaction prediction in genome-scale models [24]. However, these approaches do not formulate deletion prioritization around an explicit diferentiable representation of GPR logic. This motivates a reaction-centric formulation in which GPR structure is treated as a biological prior within the learned reaction representation, while metabolic topology provides additional context.

## Foundation Models in Systems Biology

Protein and molecular language models provide pretrained representations of sequence and chemical information. ESM-2 [25] derives protein representations from large-scale sequence pretraining, while ChemBERTa [26] provides molecular representations from SMILES strings. Given the limited supervision available within individual GEMs, freezing these pretrained backbones and training lightweight task-specific projections provides a practical way to use these representations without introducing a large number of trainable parameters.

Together, these developments suggest a progression from targetspecific constraint-based search toward reusable learning-based design, and from generic learned representations toward models that explicitly encode biological structure. Our framework follows this progression by combining curated computational design knowledge, GPR logic, metabolic topology, and pretrained molecular representations within a reusable learning protocol for growth-coupled gene deletion design.

## The Growth-Coupled Gene Deletion Protocol Protocol Overview

Drawing on previous work on growth-coupled gene deletion design, we present a reproducible protocol for learning, ranking, constructing, and evaluating candidate gene-deletion strategies for growth-coupled metabolite production (Fig. 2). The protocol builds on the MetNetComp reference database [8] and previous methods, including DeepGDel [11], GraphGDel [12], and the DBgDel optimization framework [19]. The protocol takes a GEM, a target metabolite, and molecular sequence information as inputs. Computationally curated reference deletion strategies from MetNetComp serve as supervised learning targets and reference sets for evaluating gene-set recovery (Section 3.2). The protocol comprises five stages: (i) target-specific problem definition, (ii) HGPRGDel scoring, (iii) deletion-priority ranking, (iv) candidate strategy construction, and (v) downstream constraint-based assessment. Throughout this section, protocol refers to the complete workflow, whereas HGPRGDel denotes its GPR-aware learning component. Within this workflow, HGPRGDel integrates molecular and metabolic information into a shared GPR-aware reaction representation (Sections 3.3–3.5) and produces a continuous gene-retention score for each gene–target pair (Section 3.6). These scores are converted into deletion priorities to define an ordered search space, from which nested candidate deletion sets are constructed (Section 4.7). The candidate sets are then assessed independently using downstream constraint-based metabolic analysis (Supplementary Section S3.3).

## Problem Definition and Reference Strategies

A genome-scale metabolic model (GEM) C encodes cellular metabolism as a stoichiometric network with metabolite, reaction, and gene sets M, R, and G, together with GPR associations that link gene presence to reaction activity [9]. A deletion strategy is considered growth-coupled for a target metabolite when it satisfies the weak growth-coupling criterion used in our downstream FBA evaluation: biomass is first maximized and fixed at its optimum, target production is then maximized, and both growth and production must reach at least 0.001 (Supplementary Section S2.2). The supervised reference labels are the deletion gene sets reported by MetNetComp for each target metabolite [8]; their label construction and encoding are described in Supplementary Section S1.2.

Learning task. HGPRGDel outputs a continuous retention score $\hat { y } _ { g , m } \in [ 0 , 1 ]$ for each gene–target pair, with lower values indicating higher deletion priority. Defining $d _ { g , m } = 1 - \hat { y } _ { g , m } .$ ranking genes by decreasing $d _ { g , m }$ produces a complete deletionpriority order. This ranking defines an ordered search space from which candidate deletion sets of diferent sizes can be derived: the candidate set of size k consists of the k genes with highest deletion priority, so candidate sets at increasing k are nested. A threshold θ may be applied to select one operating point from this ranking; the choice of operating point is not part of the model definition. The learning task is to estimate and rank deletion priorities; exhaustive enumeration of feasible knockout sets is outside its scope.

Within the learning layer, HGPRGDel is organized around three complementary biological priors introduced sequentially into a shared reaction representation (Fig. 2): Molecular prior encodes sequence-derived information via frozen pretrained language models; GPR prior transforms Boolean GPR rules into diferentiable reaction embeddings; Topology prior propagates metabolite context through a reaction-centric hypergraph. Five modules implement the learning component (Gene-M, Meta-M, GPR-M, Graph-M, and Pred-M); their interactions are detailed in the following subsections.

## Encoding Molecular Information

Within HGPRGDel, compound and gene features are extracted from frozen pretrained language models to introduce the molecular prior into reaction representation learning, without fine-tuning on limited per-GEM labels. The molecular prior module (Meta-M) encodes metabolite SMILES with frozen ChemBERTa [26] (DeepChem/ChemBERTa-77M-MLM); the gene prior module (Gene-M) encodes protein sequences with frozen ESM-2 [25] (esm2 t6 8M UR50D). For each input, mean-pooled backbone outputs are mapped by a trained linear projection into the corresponding module’s representation space, yielding compound features $\mathbf { X } \in \mathbb { R } ^ { d _ { m } }$ (stacked over GEM metabolites) and gene embeddings $\mathbf { Z } _ { \mathrm { g e n e } } ^ { ( g ) } \in \mathbb { R } ^ { d _ { g } }$ The two projections are independent, so $d _ { g }$ and $d _ { m }$ need not coincide $( d _ { g } = 3 2 0$ $d _ { m } = 2 5 6 ;$ Supplementary Table S5). Only the projection layers are trained: this limits trainable parameters under the available per-GEM supervision and enables controlled comparisons with non-sequence baselines. Molecular priors are thus available as inputs to reaction representation learning without supervised fine-tuning of the backbone encoders.

## Diferentiable GPR Logic Modeling

## Biological Rationale

GPR rules determine how gene deletions propagate to reaction activity. Enzyme complexes, encoded as AND nodes, require all subunits; deleting any single subunit is suficient to block the reaction. Isozymes, encoded as OR nodes, allow any alternative to sustain the reaction; blocking requires deleting all alternatives. Because these Boolean distinctions must be represented diferentiably for gradient-based learning, they require a tailored aggregation scheme rather than standard mean pooling.

## GPR-Specific Diferentiable Aggregator

The GPRRuleParser converts COBRA-style Boolean rule strings into AND/OR parse trees with gene leaves; nested rules are handled recursively. The SoftBooleanAggregator evaluates each tree diferentiably (Algorithm 1).

![](images/6d73f5774bcf1520f6c926de8063d1268539f097f04f57b7784858cb32377980.jpg)  
Figure 2 Overview of the growth-coupled gene deletion protocol. The protocol takes a GEM, a target metabolite, and molecular sequence information as input and uses computationally curated MetNetComp reference strategies for supervised learning. HGPRGDel serves as the learning component: GPR-M converts Boolean AND/OR GPR structure into reaction embeddings; Graph-M propagates metabolic context through a reaction-centric hypergraph; Gene-M and Meta-M provide frozen molecular representations; and Pred-M converts the resulting gene, metabolite, and reaction representations into a continuous retention score $\hat { y } _ { g , m } .$ The resulting gene–target scores define a deletion-priority ranking, which provides an ordered search space for constructing nested candidate deletion sets and downstream constraint-based assessment.

For a conjunction (enzyme complex) with child embeddings $\{ { \mathbf { u } } _ { i } \} _ { i = 1 } ^ { k }$ ,

$$
\mathbf { z } _ { \mathrm { a n d } } = \phi _ { \Lambda } \bigl ( [ \mathbf { m } | | \mathbf { s } ] \bigr ) , \quad \mathbf { m } = \frac { 1 } { K } \sum _ { i = 1 } ^ { K } \mathbf { u } _ { i } , \quad \mathbf { s } = - \log \sum _ { i = 1 } ^ { K } \exp ( - \mathbf { u } _ { i } / \tau ) ,\tag{1}
$$

where K is the number of children of the node, s is a soft bottleneck over the weakest-contributing subunit, and $\tau > 0$ is the efective soft-AND temperature, obtained from a single learnable scalar ˆτ (Supplementary Section S7.1). The combiners $\phi _ { \wedge }$ and $\phi _ {  \vee }$ are two-layer multilayer perceptrons (MLPs) with ReLU activation (hidden width 256; Supplementary Section S7.1). For a disjunction (isozymes),

$$
\mathbf { z } _ { \mathrm { o r } } = \phi _ { \vee } \left( [ \mathbf { a } | | \mathbf { l } ] \right) , \quad \mathbf { a } = \mathrm { M H A } ( \mathbf { q } , \mathbf { U } , \mathbf { U } ) , \quad \mathbf { l } = \log \sum _ { i = 1 } ^ { K } \exp ( \mathbf { u } _ { i } ) ,\tag{2}
$$

where MHA denotes multi-head attention, $\begin{array} { r } { { \bf q } = \frac { 1 } { K } \sum _ { i } { \bf u } _ { i } , } \end{array}$ and $\mathbf { U } = [ \mathbf { u } _ { 1 } ^ { \top } ; \cdots ; \mathbf { u } _ { K } ^ { \top } ]$ . The attention weights capture diferences among isozyme alternatives, while log-sum-exp encodes the any-suficient principle. Both combiners map $\mathbb { R } ^ { 2 d _ { g } } \ \to \ \mathbb { R } ^ { d _ { g } }$ , so GPR-M maps each reaction to a reaction embedding $\mathbf { Z } _ { \mathrm { r x n } } ^ { ( r ) } ~ \in ~ \mathbb { R } ^ { d _ { g } }$ that preserves the gene-embedding dimension. The reduction to the smaller reaction-context dimension $D _ { \mathrm { r x n } }$ happens downstream, in Pred-M (Section 3.6). Gradient expressions, temperature constraints, and numerical stability details are given in Supplementary Section S7. The GPR prior is thus embedded into reaction representations, with Boolean AND/OR logic made diferentiable and available for downstream topology-level learning. GPR-M operates on gene embeddings derived from protein sequence alone; deletion labels are used as supervised targets, and their relationship to the resulting reaction representations is learned downstream by Pred-M (Section 3.6).

## Reaction-Centric Hypergraph Learning Hypergraph Formulation

We represent the metabolic network as a hypergraph $\mathcal { H } = ( \nu , \mathcal { E } )$ where compound nodes V are metabolites and each reaction $r \in R$ is a hyperedge connecting all its substrates and products. The incidence matrix $\textbf { H } \in \ \{ 0 , 1 \} ^ { | \mathcal { V } | \times | R | }$ marks metabolites that participate in each reaction. Currency metabolites (ATP, NADH, $_ \mathrm { H _ { 2 } O _ { 3 } }$ , etc.) are filtered to suppress topological hubs that would otherwise dominate message passing. Unlike pairwise metabolite graphs, each reaction remains an explicit computational unit throughout message passing, preserving reaction identity and multi-metabolite structure rather than decomposing multi-participant reactions into pairwise edges.

## GPR-Conditioned Message Passing (Graph-M)

Graph-M takes compound features X from Meta-M, the GEM incidence matrix H (fixed across folds; Supplementary Section S2.6.1), and GPR-derived reaction embeddings $\mathbf { Z } _ { \mathrm { r x n } } ^ { ( r ) }$ Each hyperedge attribute combines aggregated compound features with the GPR projection:

$$
\mathbf { A } _ { \mathrm { h e } } ^ { \left( r \right) } = \mathbf { W } _ { \mathrm { h e } } \left[ \left( \mathbf { H } ^ { \top } \mathbf { X } \right) _ { r } \Big \Vert \mathbf { W } _ { \mathrm { g p r } } \mathbf { Z } _ { \mathrm { r x n } } ^ { \left( r \right) } \right] .\tag{3}
$$

```latex
Algorithm 1 Diferentiable evaluation of GPR logic for reaction
representation learning.
1: Input: parse tree T; gene embeddings $\{ \mathbf { z } _ { g } \}$ ; temperature $\tau > 0 ;$
MLPs $\phi _ { \Lambda } , \phi _ { \vee } ;$ multi-head attention MHA.
2: Output: reaction embedding $\mathbf { z } _ { \mathrm { { r x n } } } \in \mathbb { R } ^ { d _ { g } } \ ( d _ { g } = 3 2 0 ;$ the combiners
preserve the gene-embedding dimension).
3: procedure GPR-Eval(T)
4: if T is empty then return 0
5: end if
6: if T is leaf gene g then return $\mathbf { z } _ { g }$ (or 0 if missing)
7: end if
8: i $\textbf { f } T = T _ { 1 }$ and · · · and $T _ { K }$ then
9: for i = 1 to K do
10: u ← GPR-Eval(T )
11: end for
12: r $\begin{array} { r l r } { \mathrm { ~  ~ { ~ \sigma ~ } ~ } } & { { }  } & { \frac { 1 } { K } \sum _ { i } \mathbf { u } _ { i } ; \mathrm { ~  ~ { ~ \sigma ~ } ~ } { \bf s }  - \log \sum _ { i } \exp ( - { \bf u } _ { i } / \tau ) } \end{array}$ return
ϕ<sub>∧</sub>([m∥s])
13: end if
14: i $\textbf { f } T = T _ { 1 }$ or · · · or $T _ { K }$ then
15: for $i = 1$ to K do
16: u ← GPR-Eval(T )
17: end for
18: $\begin{array} { r } { \mathbf { U } \gets [ \mathbf { u } _ { 1 } ^ { \top } ; \cdot \cdot \cdot ; \mathbf { u } _ { K } ^ { \top } ] ; \quad \mathbf { q } \gets \frac { 1 } { K } \sum _ { i } } \end{array}$ u
19: a ← MHA(q, U, U); l ← log P exp(u<sub>i</sub>) return
ϕ<sub>∨</sub>([a∥l])
20: end if
```

Two stacked HypergraphConv layers [27] (torch geometric) with node-level attention produce graph-refined compound representations $\mathbf { Z } _ { \mathrm { m e t a G } } ;$ the attention weights $\alpha _ { v , r } ^ { ( \ell ) }$ and $\beta _ { v , r } ^ { ( \ell ) }$ follow this layer’s built-in attention mechanism. At each layer $\ell ~ \in ~ \{ 0 , 1 \}$ , message passing proceeds in two steps. First, each hyperedge (reaction) aggregates its incident node features, conditioned on the GPR-derived attribute $\mathbf { A } _ { \mathrm { h e } } ^ { ( r ) }$ :

$$
\mathbf { m } ^ { ( r , \ell ) } = \sigma \Bigl ( \mathbf { W } _ { 1 } ^ { ( \ell ) } \sum _ { v \in \mathcal { N } ( r ) } \alpha _ { v , r } ^ { ( \ell ) } \mathbf { h } _ { v } ^ { ( \ell ) } + \mathbf { A } _ { \mathrm { h e } } ^ { ( r ) } \Bigr ) ,\tag{4}
$$

where $\mathbf { h } _ { v } ^ { ( 0 ) } = \mathbf { X } _ { \tau }$ (Meta-M compound features), $\mathcal { N } ( r )$ denotes the metabolites participating in reaction $\begin{array} { r l } { r } & { { } ( \mathrm { i . e . } } \end{array}$ the non-zero rows of column r of H), and $\alpha _ { v , r } ^ { ( \ell ) }$ are learned node-level attention weights. Second, each node (metabolite) updates its representation by aggregating messages from all reactions it participates in:

$$
\mathbf { h } _ { v } ^ { ( \ell + 1 ) } = \sigma \Big ( \mathbf { W } _ { 2 } ^ { ( \ell ) } \sum _ { r \in \mathcal { R } ( v ) } \beta _ { v , r } ^ { ( \ell ) } \mathbf { m } ^ { ( r , \ell ) } \Big ) ,\tag{5}
$$

where $\mathcal { R } ( v )$ is the set of reactions in which metabolite v participates and $\beta _ { v , r } ^ { ( \ell ) }$ are a second set of learned attention weights. The final node representations after two layers are collected as $\mathbf { Z } _ { \mathrm { m e t a G } } ~ = ~ \{ { \bf h } _ { v } ^ { ( 2 ) } \} _ { v \in \mathcal { V } }$ . GPR-derived hyperedge attributes make gene-level Boolean structure available during message passing, conditioning metabolite context on the GPRderived reaction attributes of the reactions in which each metabolite participates.

## Deletion-Priority Ranking: Continuous Gene-Retention Prediction and Training

For gene–target pair $( g , m )$ , Pred-M pools GPR-M reaction embeddings over $R _ { g \cup m } \ = \ R _ { g } \cup R _ { m } ,$ where $R _ { g }$ denotes the set of reactions associated with gene g and $R _ { m }$ denotes the set of reactions involving target metabolite $m ;$ this union pools reaction-level evidence relevant to either the candidate gene or the target metabolite, so Pred-M retains context even when the two do not co-occur in a shared reaction:

$$
\mathbf { c } _ { \mathrm { r x n } } ^ { ( g , m ) } = \frac { 1 } { | R _ { g \cup m } | } \sum _ { r \in R _ { g \cup m } } \mathbf { Z } _ { \mathrm { r x n } } ^ { ( r ) } .\tag{6}
$$

The pooled context $\begin{array} { r l r } { { \bf c } _ { \bf r x n } ^ { ( g , m ) } } & { { } \in } & { \mathbb { R } ^ { d _ { g } } } \end{array}$ is reduced by the association MLP $\begin{array} { r l r } { \mathbb { R } ^ { d _ { g } } } & { { } \to \mathbb { R } ^ { D _ { \mathrm { r x n } } } } & { ( D _ { \mathrm { r x n } } = 1 2 8 ; } \end{array}$ Supplementary Table S5). The reduced reaction context is concatenated with the gene and target-metabolite representations to form the joint representation $\mathbf { Z } ^ { ( g , m ) } \ = \ [ \mathbf { Z } _ { \mathrm { g e n e } } ^ { ( g ) } \rVert \mathbf { h } _ { m } ^ { ( 2 ) } \rVert \mathbf { M } \mathbf { L } \mathbf { P } ( \mathbf { c } _ { \mathrm { r x n } } ^ { ( g , m ) } ) ] \ \in$ $\mathbb { R } ^ { d _ { g } + d _ { m } + D _ { \mathrm { r x n } } }$ , where $\mathbf { h } _ { m } ^ { ( 2 ) } \in \mathbf { Z } _ { \mathrm { m e t a G } }$ is the embedding of the compound node corresponding to target metabolite m. An MLP followed by a sigmoid maps this joint representation to the generetention score $\hat { y } _ { g , m } \in [ 0 , 1 ]$ , where lower values indicate higher deletion priority. Model selection and operating-point selection are decoupled: training is early-stopped on the threshold-free AUC of an inner validation split (20% of each fold’s training partition), and only afterwards is θ chosen on that same inner split by maximizing macro-F1 over an 81-point grid on [0.1, 0.9] (θ is distinct from the learnable GPR-M temperature τ). The ranking induced by $\hat { y } _ { g , m }$ is retained independent of threshold selection.

Training minimizes a per-gene entropy-weighted binary crossentropy over supervised pairs:

$$
\begin{array} { r } { \mathcal { L } ( \mathcal { B } ) = \frac { \sum _ { ( g , m ) \in \mathcal { B } } w _ { g } \ell \left( y _ { g , m } , \hat { y } _ { g , m } \right) } { \sum _ { ( g , m ) \in \mathcal { B } } w _ { g } } , \qquad w _ { g } = \epsilon + ( 1 - \epsilon ) H _ { g } ^ { \gamma } , } \end{array}\tag{7}
$$

where $H _ { g }$ is the binary Shannon entropy of gene $g ^ { \prime } \mathbf { s }$ label distribution across supervised targets $( \gamma = 1 . 0 , ~ \epsilon = 0 . 0 5 ;$ Supplementary Section S2.5). This weighting reduces the influence of genes with nearly constant labels across targets and emphasizes genes whose deletion status varies with the target metabolite. Molecular encoders (ESM-2, ChemBERTa) remain frozen while all other modules are trained jointly. Full implementation details and hyperparameters are listed in Supplementary Table S5. The molecular, GPR-derived, and topology-conditioned representations are thus jointly translated into gene-retention scores, completing the learning layer of the protocol and yielding the continuous deletion-priority ranking used in downstream candidate construction.

## Results Experimental Setup

Datasets. We evaluate HGPRGDel on three GEMs: e coli core [28] (95 reactions, compact E. coli); iMM904 [29] (1,577 reactions, genome-scale S. cerevisiae); and iML1515 [30] (2,712 reactions, large E. coli reconstruction). Growth-coupled deletion labels are obtained from MetNetComp [8], using the curated maximal deletion strategy available for each target metabolite. Gene amino acid sequences are from KEGG [31]; metabolite SMILES from MetaNetX [32]. Table 1 summarizes model and dataset statistics.

Evaluation protocol. We use metabolite-based five-fold cross-validation (seed 42): the split unit is the target metabolite, so all gene labels for a metabolite remain in the same fold. Genes with near-constant labels across metabolites are excluded from the primary reported metrics because their labels provide limited discrimination for target-dependent ranking. The remaining target-dependent genes, whose deletion labels vary across supervised targets, number 66 on e coli core, 55 on iMM904, and 33 on iML1515 (Table 1). This filtering is performed independently of model predictions and applied identically to all methods and baselines under the same CV folds. A fullgene supplementary evaluation and an all-target extension of the top-10 recovery diagnostic are provided in Supplementary Sections S3.4 and S3.5. Tables 2 and 3 report macro-F1 and area under the ROC curve (AUC) as mean ± s.d. over five held-out test folds on target-dependent pairs.

Model selection and thresholding. Each outer fold reserves 20% of its training partition as an inner validation split, used for model selection and threshold selection only; neither the inner validation metabolites nor the outer test metabolites contribute to the loss. For each fold, the model state is selected by early stopping on the inner-validation AUC, a thresholdfree criterion over target-dependent gene–metabolite pairs (full early-stopping protocol in Supplementary Section S2.5). Only after the model is fixed is the decision threshold θ determined, by maximizing unweighted macro-F1 on the same inner validation split over a grid of 81 values on [0.1, 0.9]; θ is then applied unchanged to the held-out test metabolites.

Baselines. We compare with: DNN (sequence encoders, no graph); DeepGDel [11] (LSTM-based encoders with supervised prediction); NMA [12] (fixed one-hop metabolite neighborhood aggregation); and GraphGDel [12] (learnable GCN on pairwise metabolite graph). All baselines are trained and evaluated under identical CV folds, graph inputs, and threshold selection procedure. Thresholded scores are evaluated as candidate deletion sets against the curated reference gene sets, following the convention that genes outside the reference are labeled retained; the continuous ranking is evaluated separately as a deletion-priority order (Section 4.5).

Table 1 Metabolic model and dataset statistics. “#GCP target metabolites”: growth-coupled production (GCP) annotated targets; “#Deletion strategies”: targets with a usable curated maximal deletion set. “#Target-dependent genes” and “#Target-dependent pairs” exclude pre-determined genes (constant labels for ≥ 90% of targets; defined above), and, on iML1515, gene–target slots without an explicit supervised label in the curated reference (Section S1.2).
<table><tr><td colspan="2">e_coli_core</td><td>iMM904</td><td>iML1515</td></tr><tr><td>#Genes</td><td></td><td>905</td><td>1516</td></tr><tr><td>#Pre-determined genes</td><td>137 71</td><td>850</td><td>1483</td></tr><tr><td>#Target-dependent genes</td><td>66</td><td>55</td><td>33</td></tr><tr><td>#Reactions</td><td>95</td><td>1577</td><td>2712</td></tr><tr><td>#GCP target metabolites</td><td>48</td><td>782</td><td></td></tr><tr><td></td><td></td><td></td><td>1085</td></tr><tr><td>#Deletion strategies #Target-dependent pairs</td><td>41 2706</td><td>105 5775</td><td>443 13464</td></tr></table>

## The Core Learning Component Performs Consistently Across Three GEMs

Across the three GEMs, HGPRGDel achieves higher macro-F1 than all four evaluated baselines and outperforms GraphGDel on every individual fold (Table 2; Fig. 3A; Supplementary Section S3.2). The evaluated methods show incremental gains from sequence-based encoders (DNN, DeepGDel) to graph-augmented representations (NMA, GraphGDel), while HGPRGDel additionally models GPR logic at the reaction level.

Relative to GraphGDel, HGPRGDel combines GPRaware reaction embeddings with reaction-centric hypergraph propagation, with macro-F1 gains of 14.9, 7.1, and 18.4 percentage points on e coli core, iMM904, and iML1515, respectively. The margin is largest on the genome-scale iML1515 reconstruction, which has the largest reaction count and the largest absolute number of multi-gene (AND/OR) rules. Crossfold spread is lowest on the genome-scale iML1515 (90.38±0.74) and widest on e coli core (89.11±3.81), the GEM with the fewest supervised pairs per fold.

HGPRGDel also shows a consistently wider AUC–macro-F1 gap than any baseline (7.3, 7.6, and 6.7 percentage points across the three GEMs, versus 0.3–3.7 for the baselines). Because the operating threshold is selected within each training fold, fold-level macro-F1 reflects both ranking quality and the foldspecific operating point; we therefore use the continuous-score and ranking metrics below to assess prioritization independently of any fixed threshold.

Performance gains persist across GEM scales. HGPRGDel was applied to three GEMs spanning a 30-fold range in reaction count without GEM-specific hyperparameter tuning. The largest gain appears on iML1515, which contains the largest absolute number of multi-gene (flat AND/OR or nested) reactions (24.0% flat OR, 7.6% flat AND, 3.9% nested over 2,712 reactions; Supplementary Table S4). Performance on the eukaryotic iMM904 reconstruction is comparable to the genomescale E. coli model iML1515 (88.61% vs. 90.38% macro-F1). Cross-GEM absolute scores should be interpreted cautiously because target-dependent-pair numbers and label distributions difer across GEMs; comparisons therefore focus on relative improvements over matched baselines under identical CV folds.

Supplementary Table S4 summarizes GPR rule distributions across GEMs, showing that iML1515 contains substantially more AND/OR-associated reactions in absolute numbers. The same continuous scores also support gene-set recovery on held-out targets (Section 4.5); the resulting rankings require fewer genes than the evaluated baselines to reach 90% mean cumulative recall on each GEM.

## Molecular, GPR, and Topology Priors Provide Complementary Signal

Table 3 and Fig. 3B examine the contribution of each biological prior within HGPRGDel. We next asked whether the three representation sources contribute complementary information or whether performance is dominated by a single component. Variant definitions are given in Supplementary Section S4.1.

Molecular prior. Removing the pretrained weights (No pretrained; Supplementary Section S2.3) costs ≈6.6–12.6 pp macro-F1 across GEMs. This variant retains the ESM-2 and

<table><tr><td colspan="3"></td></tr><tr><td>Method</td><td>Macro-F1 (%)</td><td>AUC (%)</td></tr><tr><td></td><td></td><td> $6 2 . 1 8 \pm 3 . 9 6$ </td></tr><tr><td>DNN DeepGDel</td><td> $5 8 . 9 4 \pm 3 . 5 3 $ </td><td></td></tr><tr><td></td><td> $7 0 . 1 6 \pm 3 . 7 3$ </td><td> $7 0 . 4 7 \pm 3 . 9 0$ </td></tr><tr><td>NMA</td><td> $7 0 . 9 1 \pm 3 . 7 8$ </td><td> $7 1 . 2 5 \pm 3 . 9 4$ </td></tr><tr><td>GraphGDel</td><td> $7 4 . 2 2 \pm 3 . 2 7$ </td><td> $7 6 . 3 3 \pm 3 . 3 4$ </td></tr><tr><td>HGPRGDel</td><td> $\mathbf { 8 9 . 1 1 \pm 3 . 8 1 }$ </td><td> ${ \bf 9 6 . 4 2 \pm 1 . 5 7 }$ </td></tr></table>

Table 2 Benchmark performance (mean ± s.d. over five heldout metabolite test folds on target-dependent pairs). Bold: best per panel. The HGPRGDel rows report the five-fold protocol of Section 4.1, in which each fold uses its own threshold chosen on that fold’s inner validation split. The same scores, extended to all five metrics, are tabulated in Supplementary Table S7.  
(A) e coli core

$$
6 5 . 4 3 \pm 1 . 6 3
$$

$$
7 5 . 9 8 \pm 1 . 5 1
$$

$$
6 7 . 0 2 \pm 1 . 5 8
$$

$$
7 6 . 8 3 \pm 1 . 5 5
$$

$$
7 9 . 7 1 \pm 1 . 8 0
$$

$$
8 0 . 4 6 \pm 1 . 8 6
$$

$$
8 1 . 5 4 \pm 1 . 3 8
$$

$$
8 4 . 0 9 \pm 1 . 3 6
$$

$$
\mathbf { 8 8 . 6 1 \pm 1 . 4 2 }
$$

$$
{ \bf 9 6 . 2 2 \pm 1 . 1 2 }
$$

$$
5 8 . 0 2 \pm 1 . 2 1
$$

$$
5 9 . 5 3 \pm 1 . 1 8
$$

$$
6 2 . 7 4 \pm 1 . 3 8
$$

$$
6 3 . 6 2 \pm 1 . 4 0
$$

$$
6 5 . 0 8 \pm 1 . 2 0
$$

$$
7 2 . 0 3 \pm 1 . 0 5
$$

$$
6 5 . 8 9 \pm 1 . 1 4
$$

$$
{ \bf 9 0 . 3 8 \pm 0 . 7 4 }
$$

$$
7 3 . 0 2 \pm 1 . 0 9
$$

$$
\mathbf { 9 7 . 1 2 \pm 1 . 2 0 }
$$

Table 3 Component ablation (mean ± s.d. over five held-out folds). Bold: best per panel. The Full rows report the five-fold protocol of Section 4.1, in which each fold uses its own threshold chosen on that fold’s inner validation split.  
(A) e coli core
<table><tr><td colspan="3"></td></tr><tr><td>Configuration</td><td>Macro-F1 (%)</td><td>AUC (%)</td></tr><tr><td colspan="3"></td></tr><tr><td>Full</td><td> $\mathbf { 8 9 . 1 1 \pm 3 . 8 1 }$ </td><td> ${ \bf 9 6 . 4 2 \pm 1 . 5 7 }$ </td></tr><tr><td>Mean Pooling</td><td> $7 8 . 8 0 \pm 3 . 2 1$ </td><td> $8 8 . 1 0 \pm 3 . 2 4$ </td></tr><tr><td>No Graph-M</td><td> $7 7 . 5 8 \pm 3 . 2 9$ </td><td> $8 6 . 5 5 \pm 3 . 3 2$ </td></tr><tr><td>No pretrained</td><td> $7 6 . 5 0 \pm 3 . 3 0$ </td><td> $8 7 . 0 0 \pm 3 . 2 0 $ </td></tr><tr><td>No GPR-M</td><td> $7 5 . 3 0 \pm 3 . 2 9$ </td><td> $8 4 . 6 0 \pm 3 . 3 4$ </td></tr><tr><td colspan="3">(B) iMM904</td></tr><tr><td>Configuration</td><td>Macro-F1 (%)</td><td>AUC (%)</td></tr><tr><td colspan="3"></td></tr><tr><td>Full</td><td> $\mathbf { 8 8 . 6 1 \pm 1 . 4 2 }$ </td><td> ${ \bf 9 6 . 2 2 \pm 1 . 1 2 }$ </td></tr><tr><td>Mean Pooling</td><td> $8 5 . 3 0 \pm 1 . 1 9$ </td><td> $9 3 . 5 0 \pm 1 . 1 1$ </td></tr><tr><td>No Graph-M</td><td> $8 3 . 3 0 \pm 1 . 2 5$ </td><td> $9 2 . 1 0 \pm 1 . 1 4$ </td></tr><tr><td>No pretrained</td><td> $8 2 . 0 0 \pm 1 . 2 0 $ </td><td> $9 1 . 0 0 \pm 1 . 1 0 $ </td></tr><tr><td>No GPR-M</td><td> $8 1 . 3 0 \pm 1 . 2 7$ </td><td> $8 9 . 5 0 \pm 1 . 1 5$ </td></tr><tr><td colspan="3">(C) iML1515</td></tr><tr><td>Configuration</td><td>Macro-F1 (%)</td><td>AUC (%)</td></tr><tr><td colspan="3"></td></tr><tr><td>Full</td><td> ${ \bf 9 0 . 3 8 \pm 0 . 7 4 }$ </td><td> $\mathbf { 9 7 . 1 2 \pm 1 . 2 0 }$ </td></tr><tr><td>Mean Pooling</td><td> $8 6 . 4 0 \pm 0 . 8 2$ </td><td> $9 5 . 2 0 \pm 1 . 2 9$ </td></tr><tr><td>No Graph-M</td><td> $8 4 . 4 0 \pm 0 . 9 2$ </td><td> $9 1 . 5 0 \pm 1 . 3 6$ </td></tr><tr><td>No pretrained</td><td> $8 3 . 3 5 \pm 0 . 8 0$ </td><td> $9 2 . 5 0 \pm 1 . 2 0$ </td></tr><tr><td>No GPR-M</td><td> $8 2 . 4 0 \pm 0 . 9 6$ </td><td> $8 9 . 3 0 \pm 1 . 4 1 $ </td></tr></table>

ChemBERTa architectures and trains them end-to-end from random initialization, so the drop isolates the contribution of the pretrained weights themselves rather than of model capacity or input information. Its cost is the most GEM-dependent of the four ablations (12.6, 6.6, and 7.0 pp on e coli core, iMM904, and iML1515, respectively), and is largest on the smallest GEM, where supervision per fold is scarcest.

GPR prior. The ordering Full > Mean Pooling > No GPR-M decomposes the GPR-M benefit: Mean Pooling preserves gene–reaction connectivity but collapses AND/OR distinctions, while No GPR-M removes both. On e coli core, the AND/OR distinction contributes ≈10.3 pp and connectivity ≈3.5 pp of the total ≈13.8 pp GPR-M gain; disabling GPR-M entirely removes the largest single-component contribution in all three GEMs (≈7.3–13.8 pp macro-F1). To further dissect the SoftBooleanAggregator design, we compared three ANDnode variants on e coli core: using only the mean branch (m), only the soft-min branch (s), or both combined (default). Isolating either branch alone drops macro-F1 by approximately 20 percentage points relative to the combined configuration (89.11% vs. 68.64% / 69.38%), supporting complementary roles for the two branches: m captures overall complex representation, while s emphasizes the lower-valued subunit representations. Detailed AND-branch ablation results are provided in Supplementary Table S17.

Topology prior. Disabling Graph-M drops performance by ≈5–12 pp across GEMs. To further justify the hypergraph design, we evaluated three Graph-M variants on e coli core while holding GPR-M and Pred-M fixed (Supplementary Table S18): replacing the hypergraph with a reaction–metabolite bipartite network reduces macro-F1 by 9.7 pp (79.44%); a pairwise metabolite GCN reduces it by 13.1 pp (75.99%), approaching the GraphGDel baseline (74.22%) that uses no GPR information. The hypergraph variant retains reaction-level groupings of all participating metabolites and accepts GPR-derived hyperedge attributes without structural approximation, consistent with its stronger performance than both alternatives.

## Learned Reaction Embeddings Recover GPR Logic and Metabolic Organization

The architecture explicitly models GPR logic and metabolic topology at the reaction level. To dissect whether the learned reaction embeddings capture these structures and are associated with downstream deletion prioritization, we performed a sixpanel mechanistic analysis (Fig. 4): Panels A–B assess operator fidelity; C and D quantify GPR-related and biological properties of the learned representations; E and F examine downstream deletion behavior.

GPR-M reaction embeddings (Fig. 4; Supplementary Section S8) encode complementary structure without explicit pathway supervision. First, SoftBoolean AND and OR aggregations track hard-min / hard-max scalars with Spearman $r _ { s } ~ \ge ~ 0 . 8 6$ across GEMs (Fig. 4A,B), indicating that the diferentiable gates preserve the expected ordering properties of Boolean aggregation. Second, AND/OR type is linearly decodable at near-ceiling probe accuracy and nearest-neighbor subsystem enrichment strengthens with GEM scale (z=7.7– 104.8; Fig. 4D), indicating that subsystem organization is retained without explicit pathway supervision; GPR-rule representation separation is consistently weaker for singlegene reactions than for AND/OR reactions across all three GEMs (mean silhouette 0.05–0.25 versus 0.47–0.83; Fig. 4C), suggesting that AND/OR combinatorial structure is more distinctly represented than single-gene structure. Third, deletion F1 varies systematically by exclusive gene–GPR role (AND / OR / SINGLE; Fig. 4E). On iML1515, the OR-exclusive stratum has the highest deletion F1 (88.8%), while ANDand SINGLE-exclusive genes are closer (81.1% and 79.6%), so AND-exclusive genes are not uniformly the easiest stratum. The continuous deletion-priority score discriminates referenceset membership consistently across GPR roles and GEMs (AUC 0.84–0.99; Fig. 4F), indicating that this representation– output association is not confined to a single GPR role. Together, these analyses provide multi-level evidence that the learned reaction representations retain GPR-related structure and that this structure is reflected in downstream deletion prioritization. Reaction-type UMAP and OR-branch attention entropy (Supplementary Fig. S3) provide additional supporting evidence: partial GPR-type co-localization in embedding space and non-uniform isozyme attention weighting, respectively. Coarse-category UMAP is provided in Supplementary Fig. S4.

![](images/ac67cd31c5915a79533144238ab4309000ebe6f8eb5f80667d5f089a064f8bd8.jpg)

![](images/1d1ff42e90c76057d5c8a142a589b719de50bbe64987d4c71b997ae74ed9834a.jpg)  
Figure 3 Benchmark and component ablation across three GEMs. (A) Macro-F1 across evaluated methods. (B) Macro-F1 for HGPRGDel and its ablated variants. Values are mean ± s.d. over five held-out metabolite folds.

## Continuous Retention Scores Recover Curated Reference Gene Sets

We next systematically evaluated the model’s primary output, the continuous retention score, across held-out targets and GEMs, asking whether the continuous retention scores recover curated reference gene sets and whether the complete ranking provides prioritization value beyond a single threshold. Because the intended downstream use is deletion prioritization, ranking metrics $( \mathrm { A P } ,$ cumulative recall, k ) provide the primary evidence for ordering quality, while threshold-dependent classification and reference-set recovery provide complementary diagnostics. These analyses evaluate recovery of known metabolic-design patterns rather than claim de novo discovery, since the MetNetComp reference provides the supervised annotation [8]. Method-level predictive comparisons (DNN, DeepGDel, NMA, GraphGDel) are reported in Table 2; here we focus on reference gene-set recovery and the additional value of ranked gene prioritization.

The continuous retention score encodes deletion priority at every operating point, enabling both candidate-set recovery through thresholding and a complete ranking that retains prioritization information beyond any single operating point.

Recovery. Average precision evaluates the ordering of reference genes independently of any fixed operating threshold, and therefore matches the model’s primary output. Deletion rankings measured by mean average precision exceed a random prevalence baseline (Fig. 5A), with the largest margin on e coli core (mean AP 94.7% vs. 49% prevalence); genomescale AP values (99.5–100%) should be interpreted alongside the elevated prevalence baseline (74–80%) and read together with the cumulative-recall curves rather than in isolation. At one operating point, thresholded HGPRGDel predictions additionally achieve high gene-set overlap with the curated reference across all three GEMs (recall 92.0–99.0%; precision 92.5–99.3%; F1 92.0–99.2%; Table 4).

Ranking. HGPRGDel produces a continuous retention score for every gene–target pair, providing a complete deletionpriority ordering. The same held-out metabolites, gene roster, and curated labels used for HGPRGDel were scored with GraphGDel, NMA, DeepGDel, and DNN checkpoints. Ordering genes by deletion priority $( 1 - \hat { y } _ { g , m } )$ concentrates curated deletions toward the top-ranked fraction (Fig. 5B).

Across all three GEMs, HGPRGDel yields the lowest k among evaluated methods, defined as the fraction of genes that must be examined to reach ≈90% mean cumulative recall (Fig. 5C). On e coli core, $k _ { 9 0 } = 5 3 \%$ vs. DeepGDel 57%, GraphGDel/DNN 73%, NMA 84%, and random 90%. On genome-scale GEMs, the margin widens: $k _ { 9 0 } = 6 7 \%$ on iMM904 and 72% on iML1515, compared with DeepGDel (74% / 79%), GraphGDel (82% / 84%), DNN (83% / 85%), and NMA (88% / 89%). Notably, GraphGDel, despite strong macro-F1 (Table 2), produces genome-scale rankings only modestly better than a random order $( k _ { 9 0 } { \approx } 8 2 – 8 4 \%$ against 90% for random), showing that binary classification strength does not necessarily transfer to gene prioritization.

A  
![](images/c35345d1505eb5aaec6217d2954da7ce6d79612e5811be7077cc2083356cc0e0.jpg)

B  
![](images/b3824a40529dd07d026c3e1206593f20014ebbf3724501d51e9595de497d569a.jpg)

C  
![](images/fa8b243a633622dd03ad70b9b17066088d510f0514111ad3ab546b18d68d5fb9.jpg)

![](images/8ad31e12a7d2720b5c87927f2ae18db7eb3e8ea306a2bf7ef904213a813e59e1.jpg)

![](images/ca437a5d9f5d043698b45db0af46ddc0faf366cd0b282240f602d32acd65533d.jpg)

![](images/2174ad52c7a45e362c878fca3f55d1631bb8a04f6e6fc1d51bf5aded782983ad.jpg)  
Figure 4 Mechanistic analysis of GPR-aware reaction representations and their association with deletion prioritization. (A) SoftBoolean AND fidelity: soft-min versus hard-min leaf $L _ { 2 }$ scalars across GEMs (linear axes; dashed $y = x ;$ black line, pooled LOWESS trend). Inset shows the distribution of soft-min−hard-min residuals; mean $\pm \mathsf { S D }$ is annotated. Spearman $r _ { s }$ and n are reported for each GEM. (B) SoftBoolean OR fidelity: log-sum-exp versus hard-max leaf $L _ { 2 }$ scalars (log–log axes; dashed y=x). Grey hexbin density summarizes the pooled distribution, with GEM-specific points overlaid. Spearman $r _ { s }$ and n are reported for each GEM. (C) GPR-rule representation separation: mean silhouette score (cosine distance) for AND, OR, and single-gene reactions, computed jointly with empty reactions in the reference representation space. The number of reactions (n) is annotated for each GEM. (D) Embedding diagnostics: nearest-neighbor subsystem-enrichment z-scores across three GEMs, shown as bars; linear-probe accuracy for AND/OR rule classification (%) is annotated for each GEM. (E) Deletion F1 on target-dependent gene–metabolite pairs, stratified by genes whose associated reactions are exclusively AND, OR, or single-gene (MIXED-rule genes excluded). The number of genes (n) is annotated for each GEM and GPR role. (F) Representation–output association: AUC of the continuous deletion-priority score $( 1 - { \hat { y } } )$ for distinguishing reference-set membership, stratified by exclusive GPR role. The number of target-dependent gene–metabolite pairs (n) is annotated for each GEM and GPR role. UMAP embeddings and OR-branch attention entropy are shown in Supplementary Fig. S3.

## A Target-Level Case Study of GPR-Aware Deletion Prioritization

To illustrate how the learned ranking behaves at the target level, we examined succinate (succ c) in e coli core as a representative case study, among target-dependent genes $\left( \mathrm { F i g . ~ 6 } \right)$ . Succinate was selected because it represents a central-carbon target with diverse GPR configurations, including AND-type enzyme complexes and OR-type isozymes. Among 66 target-dependent genes, the curated reference annotates 44 deletions; the continuous ranking attains average precision 0.91 with all ten of the lowest-retention calls matching the curated reference (Supplementary Table S19). This AP is a case-study metric computed on the 66 target-dependent genes of a single target and is not directly comparable to the full-roster mean AP of Table 4 (Supplementary Section S5). The ranking places genes associated with distinct GPR roles at diferent priorities, illustrating how reaction-level structure can shape the predicted deletion order.

Among the lowest-retention predicted deletions, OR-type isozyme calls (e.g., b1773 on FBA; b1276 on ACONTa) co-occur with AND-type complex subunits (e.g., CYTBD, ATPS4r, AKGDH, FRD7) and a single-gene transport call (AKGt2r), illustrating diferent deletion priorities associated with AND, OR, and single-gene

![](images/4065e4144aec230f934f310ce24ac0400563f8966d146c5036c79ff276a0441e.jpg)

B  
![](images/889cef8d53b18c5dc27f847a8cb8d9faaa79718329882aab3f152184ec554567.jpg)  
C

![](images/4562de651ff151aa8067aeed02642649d07bdb22a90a66e75417d0cdee91c343.jpg)

![](images/fb0fa30089f1d3890559e02b495cec04e2d8550c1c618eae2b6c5c17d4c1a87e.jpg)  
Figure 5 Deletion-priority ranking recovers curated reference gene sets. All methods are evaluated on the same held-out metabolites, gene roster, and curated labels; scores are continuous retention probabilities ranked by $1 - \hat { y } _ { g , m }$ (higher = earlier deletion). (A) Mean average precision. (B) Cumulative recall as the top fraction of the ranked gene list increases, for HGPRGDel, GraphGDel, NMA, DeepGDel, DNN, and a random ranking. Circles mark each method’s $k _ { 9 0 }$ . (C) k , the fraction of genes required to recover 90% of reference genes; lower is better. Reference gene sets are computationally curated MetNetComp annotations and are not experimentally validated.

Table 4 Recovery of curated MetNetComp reference gene sets. Gene-level recovery metrics are computed for targets with explicit supervised labels; definitions are provided in Methods and Supplementary Section S1.2.
<table><tr><td>GEM</td><td>#Held-out</td><td>#Scored</td><td>Avg. |GT|</td><td>Avg. |Pred|</td><td>Rec. (%)</td><td>Prec. (%)</td><td>F1 (%)</td><td>AP (%)</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>92.03</td><td>92.53</td><td>91.98</td><td></td></tr><tr><td>e_coli_core iMM904</td><td>41 105</td><td>41 105</td><td>67.1 670.4</td><td>66.4 672.5</td><td>98.91</td><td>98.62</td><td>98.76</td><td>94.71 99.97</td></tr><tr><td>iML1515</td><td>443</td><td>408</td><td>1205.3</td><td>1201.6</td><td>99.02</td><td>99.34</td><td>99.18</td><td>99.53</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

#Held-out: number of held-out target metabolites. #Scored: number of targets with explicit supervised labels in the curated reference; on iML1515, 35 of the 443 held-out targets have no label rows in the curated reference (Section S1.2), and all metric columns and the mean AP are computed over the #Scored targets only. ${ \ " } \mathsf { A v g } .$ |GT|”: mean number of genes deleted in the reference gene set; “Avg. |Pred|”: mean number of genes predicted deleted at one operating point. Both are counted over the full curated maximal gene roster (including pre-determined genes), distinct from the target-dependent-gene filter used in CV metrics. Recall / Precision / F1: gene-set overlap with the reference under thresholded HGPRGDe predictions; mean AP: deletion-ranking quality of HGPRGDel on the same #Scored targets, with baseline rankings compared in ${ \mathsf { F i g . } }$ 5. Set-level recovery metrics and ranking quality (mean AP) are reported as complementary views of the same retention scores, and recovery denotes overlap with reference gene sets rather than experimental validation.

GPR structures. No false positives appear among the ten lowestretention calls. $\mathrm { C y }$ tosolic α-ketoglutarate (akg c) and fumarate (fum c), two additional TCA-neighbor targets with reordered GPR-typed priorities, are reported in Supplementary Section S5.

## Deletion-Priority Rankings Provide an Ordered Search Space for Candidate Strategy Construction

The complete deletion-priority ranking provides a natural way to construct candidate deletion sets without fixing a common strategy size across targets. For a given target, progressively larger candidate sets are obtained by taking the first q genes in the ranking, so strategy size is an operating-point choice applied to the ranking rather than a model hyperparameter.

The curated reference sets also vary substantially in size across GEMs, with mean sizes of 67.1, 670.4, and 1205.3 genes, respectively (Table 4).

These candidate sets are intended for downstream constraintbased evaluation, and growth-coupling feasibility is not determined by the ranking alone. To connect the ranking to this assessment, genes were added to a candidate deletion set one at a time in decreasing deletion priority, and the weak growth-coupling criterion was applied after each addition (Supplementary Section S3.3). This localized a weakly growthcoupled deletion set for 26 of 41 held-out e coli core targets, or 63.4%, and for 44 of 105 iMM904 targets, or 41.9%. These values report whether the ranked candidate sets contained a deletion set that satisfied the downstream criterion. They are not prediction rates, because the ranking does not determine growthcoupling feasibility. Every unsolved target ended at a lethal prefix before a feasible one was reached, whereas placing the curated reference genes first reached weak growth coupling for all 41 and 105 targets, so these failures do not reflect infeasibility of the reference sets. The reference-informed ordering is therefore used as a coverage control rather than as a predictive baseline.

A  
![](images/acdabca059974a679dcb7331c483d4514d00f044acfb3ebf71afc5cbce9dde16.jpg)  
B

![](images/67648f443bae0bc44c9c40b3d34482fe21946ea310746e7d6993271443c36a74.jpg)

![](images/877ccd9c9fec34739d7ece43486463decaafb0675c43cdd676dc8b99832d03e6.jpg)

![](images/1e6e42ced68f93e7c353178c2c21c31371c81b7a2cb74c1a153017b05906d1bd.jpg)

C  
![](images/f7d78b346c4b79f1926a1508cd52df661a1c5922360bd74652266de22be4edb8.jpg)  
Figure 6 Target-level case study of GPR-aware deletion prioritization for succinate (succ c) on e coli core (target-dependent genes; Supplementary Section S1). (A) Central-carbon schematic highlighting reactions hit by the ten lowest-retention predicted deletions. Colors denote GPR type: blue, AND complex; gold, OR isozymes; purple, SINGLE. Labels show reaction, lowest-retention gene(s), and retention score (lower = higher deletion priority); bold labels mark genes in the curated reference. (B) Ten highest- and ten lowest-retention scores; filled/open markers denote GT deleted/retained. (C) Retention heatmap for top-ranked genes across succ c, akg c, and fum c; ∗ denotes GT deleted. akg c (α-ketoglutarate) and fum c (fumarate) are TCA-neighbor targets detailed in Supplementary Section S5.

A separate controlled analysis on e coli core examined how much of the ranked gene space the downstream search needs. When gDel-minRN was restricted to the top-ranked fraction of genes, the fraction of targets with a feasible growth-coupled solution rose from 4.9% at 5% and 10% of genes to 9.8% at 20%, 24.4% at 30%, and 78.0% at 50%, and reached 100% when all genes were available (Supplementary Section S3.3.3, which also compares alternative rankings). The candidate-space budget therefore trades the scope of the downstream search against coverage of feasible interventions, and targets missed at a smaller budget were recovered by the unrestricted search.

## Discussion

## Reusing Computational Design Knowledge

The central contribution of this work is a reusable formulation for growth-coupled gene deletion design. Constraint-based strain-design methods can identify feasible deletion strategies, but related combinatorial searches must often be repeated across gene–target combinations. Here, previously obtained design strategies are used as a learning resource to derive gene–target deletion priorities, while constraint-based analysis remains responsible for evaluating the resulting interventions.

This separation gives learning a specific role in strain design: rather than replacing mechanistic optimization, it narrows and structures the intervention space explored downstream. The learned ranking can therefore be reused across targets while the candidate-space budget remains an explicit design parameter.

## Structured Biological Knowledge in Learning

GPR associations provide a natural biological prior because they determine how gene perturbations propagate to reaction activity. Incorporating this structure into the learning representation allows the model to capture dependencies arising from enzyme complexes and alternative isozymes, while molecular and network information provide complementary context. The ablation and representation analyses support the contribution of these information sources (Section 4.3; Fig. 4).

More broadly, this illustrates a useful division of roles between biological knowledge and learning: mechanistic structure can shape the learning problem without requiring the learning component to reproduce the full constraint-based model.

## Search-Space Control and Scope

The learned ranking is most naturally interpreted as an ordered intervention space rather than a final strain design. Candidate sets can therefore be constructed at diferent budgets and evaluated independently by constraint-based analysis. Not every target reached a feasible solution at the evaluated budgets because the corresponding feasible strategy may lie outside the selected candidate space; increasing the budget progressively restored access to feasible strategies, with the unrestricted search recovering all evaluated e coli core targets. The referenceinformed ordering provides a coverage control under the same mechanistic criterion rather than a predictive baseline.

The framework is suited to settings where prior computational design knowledge can be reused to guide subsequent search. Its scope is limited by the reference strategies used for supervision, the target-dependent gene definition, and the validation design.

## Limitations and Future Work

The reference strategies do not exhaustively represent all feasible or experimentally observed deletion strategies. Moreover, the available evaluation is limited by the target-dependent gene definition and the limited number of cross-validation folds (Supplementary Sections S3.2, S3.6, and S3.7). The reference strategies and downstream assessment are both based on constraint-based metabolic modeling, so this study evaluates the reuse and prioritization of computational design knowledge rather than independent biological ground truth.

Future work can incorporate multiple feasible strategies per target and experimental strain-engineering outcomes, and can integrate learned priorities more adaptively with constraintbased search. The same framework could also be extended to reaction-, pathway-, or metabolic-state interventions.

## Conclusion

This work establishes a reusable approach for growth-coupled gene deletion design in which computational design knowledge is converted into gene–target priorities and used to structure downstream constraint-based search. By separating intervention prioritization from mechanistic assessment, the framework retains the strengths of constraint-based modeling while making previously obtained design knowledge reusable across related design problems.

## Key Points

• We present a reproducible framework that reuses previously obtained computational design knowledge to prioritize gene deletions for growth-coupled production across gene–target combinations.

The framework formulates deletion design as a gene–target learning-and-ranking problem, with HGPRGDel providing a GPR-aware learning component that integrates GPR logic, molecular information, and metabolic-network context.

• Across three GEMs, HGPRGDel achieves the highest macro-F1 and AUC among the evaluated methods, with ablation and representation analyses supporting the contribution of these biological information sources.

• The learned deletion-priority rankings recover computationally curated MetNetComp reference gene sets and reach 90% mean cumulative recall after examining fewer genes than the evaluated baseline methods, providing an ordered intervention space for downstream constraint-based assessment.

• By separating intervention prioritization from mechanistic feasibility assessment, the framework makes previously obtained design knowledge reusable while retaining explicit control over the downstream candidate space.

## Data and Code Availability

Processed MetNetComp gene-deletion labels, COBRA-style GEM JSON models, metabolite SMILES tables, amino acid sequences, source code, and pretrained checkpoints are available at https://github.com/MetNetComp/HGPRGDel.

## Funding

This work was supported by JSPS, KAKENHI under Grant 26K02972.

## Conflict of Interest

The authors declare that they have no competing interests.

## Author Contributions

Z.Y. and T.T. conceived and designed the study. Z.Y. implemented the model and performed the computational experiments. T.T. supervised the project and acquired funding. Z.Y. drafted the manuscript, and both authors revised and approved the final version.

## Acknowledgements

The authors thank colleagues at BIC, Kyoto University for helpful discussions and computational support.

## References

1. Jennifer L Reed, Thuy D Vo, Christophe H Schilling, and Bernhard O Palsson. An expanded genome-scale model of Escherichia coli K-12 (iJR904 GSM/GPR). Genome Biology, 4(9):R54, 2003.

2. Ines Thiele and Bernhard Ø Palsson. A protocol for generating a high-quality genome-scale metabolic reconstruction. Nature Protocols, 5(1):93–121, 2010.

3. Jefrey D Orth, Ines Thiele, and Bernhard Ø Palsson. What is flux balance analysis? Nature biotechnology, 28(3):245–248, 2010.

4. Evangelos Simeonidis and Nathan D Price. Genomescale modeling for metabolic engineering. Journal of Industrial Microbiology and Biotechnology, 42(3):327– 338, 2015.

5. Laurent Heirendt, Sylvain Arreckx, Thomas Pfau, Sebasti´an N Mendoza, Anne Richelle, Almut Heinken, Hulda S Haraldsd´ottir, Jacek Wachowiak, Sarah M Keating, Vanja Vlasov, et al. Creation and analysis of biochemical constraint-based models using the cobra toolbox v. 3.0. Nature protocols, 14(3):639–702, 2019.

6. Cong T Trinh, Pornkamol Unrean, and Friedrich Srienc. Minimal escherichia coli cell for the most eficient production of ethanol from hexoses and pentoses.

Applied and environmental microbiology, 74(12):3634– 3643, 2008.

7. Deepanwita Banerjee, Thomas Eng, Andrew K Lau, Yusuke Sasaki, Brenda Wang, Yan Chen, Jan-Philip Prahl, Vasanth R Singan, Robin A Herbert, Yuzhong Liu, et al. Genome-scale metabolic rewiring improves titers rates and yields of the non-native product indigoidine at scale. Nature communications, 11(1):5385, 2020.

8. Takeyuki Tamura. Metnetcomp: Database for minimal and maximal gene-deletion strategies for growthcoupled production of genome-scale metabolic networks. IEEE/ACM Transactions on Computational Biology and Bioinformatics, 2023.

9. Markus W Covert, Christophe H Schilling, and Bernhard Palsson. Regulation of gene expression in flux balance models of metabolism. Journal of Theoretical Biology, 213(1):73–88, 2001.

10. Markus W Covert and Bernhard O Palsson. Constraints-based models: regulation of gene expression reduces the steady-state solution space. Journal of Theoretical Biology, 221(3):309–325, 2003.

11. Ziwei Yang and Takeyuki Tamura. DeepGDel: Deep learning-based gene deletion prediction framework for growth-coupled production in genome-scale metabolic models. IEEE Transactions on Computational Biology and Bioinformatics, 22(5):2252–2266, sep 2025.

12. Ziwei Yang and Takeyuki Tamura. GraphGDel: Constructing and learning graph representations of genome-scale metabolic models for growth-coupled gene deletion prediction. arXiv preprint arXiv:2504.06316, 2025.

13. Anthony P Burgard, Priti Pharkya, and Costas D Maranas. Optknock: a bilevel programming framework for identifying gene knockout strategies for microbial strain optimization. Biotechnology and bioengineering, 84(6):647–657, 2003.

14. Laurence Yang, William R Cluett, and Radhakrishnan Mahadevan. Emilio: a fast algorithm for genome-scale strain design. Metabolic engineering, 13(3):272–281, 2011.

15. Satoshi Ohno, Hiroshi Shimizu, and Chikara Furusawa. Fastpros: screening of reaction knockout strategies for metabolic engineering. Bioinformatics, 30(7):981–987, 2014.

16. Deqing Gu, Cheng Zhang, Shengguo Zhou, Liujing Wei, and Qiang Hua. Idealknock: a framework for eficiently identifying knockout strategies leading to targeted overproduction. Computational biology and chemistry, 61:229–237, 2016.

17. Philipp Schneider, Axel von Kamp, and Stefen Klamt. An extended and generalized framework for the calculation of metabolic intervention strategies based on minimal cut sets. PLoS computational biology, 16(7):e1008110, 2020.

18. Takeyuki Tamura, Ai Muto-Fujita, Yukako Tohsato, and Tomoyuki Kosaka. Gene deletion algorithms for minimum reaction network design by mixedinteger linear programming for metabolite production in constraint-based models: gdel minrn. Journal of Computational Biology, 30(5):553–568, 2023.

19. Ziwei Yang and Takeyuki Tamura. DBgDel: Database-enhanced gene deletion framework for growthcoupled production in genome-scale metabolic models. IEEE Transactions on Computational Biology and Bioinformatics, 22(4):1415–1427, jul 2025.

20. Takeyuki Tamura. Trimming gene deletion strategies for growth-coupled production in constraint-based metabolic networks: Trimgdel. IEEE/ACM Transactions on Computational Biology and Bioinformatics, 20(2):1540–1549, 2023.

21. Zak Costello and Hector Garcia Martin. A machine learning approach to predict metabolic pathway dynamics from time-series multiomics data. NPJ systems biology and applications, 4(1):1–14, 2018.

22. Feiran Li, Le Yuan, Hongzhong Lu, Gang Li, Yu Chen, Martin KM Engqvist, Eduard J Kerkhoven, and Jens Nielsen. Deep learning-based k cat prediction enables improved enzyme-constrained model reconstruction. Nature Catalysis, 5(8):662–672, 2022.

23. Ramin Hasibi, Tom Michoel, and Diego A Oyarz´un. Integration of graph neural networks and genome-scale metabolic models for predicting gene essentiality. npj Systems Biology and Applications, 10:24, 2024.

24. Kai Wang, Jiajun Qu, Fei Liu, Xiaoli Luan, and Jingwen Zhou. Hypergraph learning with multidimensional metabolite feature extractions and static– dynamic attention mechanisms to fill missing reactions in metabolic networks. Briefings in Bioinformatics, 27(3):bbag314, 2026.

25. Zeming Lin, Halil Akin, Roshan Rao, Brian Hie, Zhongkai Zhu, Wenting Lu, Nikita Smetanin, Robert Verkuil, Ori Kabeli, Yair Shmueli, et al. Evolutionaryscale prediction of atomic-level protein structure with a language model. Science, 379(6637):1123–1130, 2023.

26. Seyone Chithrananda, Gabriel Grand, and Bharath Ramsundar. Chemberta: Large-scale self-supervised pretraining for molecular property prediction. arXiv preprint arXiv:2010.09885, 2020.

27. Yifan Feng, Haoxuan You, Zizhao Zhang, Rongrong Ji, and Yue Gao. Hypergraph neural networks. In Proceedings of the AAAI conference on artificial intelligence, volume 33, pages 3558–3565, 2019.

28. Jefrey D Orth, Ronan MT Fleming, and Bernhard Ø Palsson. Reconstruction and use of microbial metabolic networks: the core escherichia coli metabolic model as an educational guide. EcoSal plus, 4(1):10–1128, 2010.

29. Monica L Mo, Bernhard Ø Palsson, and Markus J Herrg˚ard. Connecting extracellular metabolomic measurements to intracellular flux states in yeast. BMC systems biology, 3:1–17, 2009.

30. Jonathan M Monk, Colton J Lloyd, Elizabeth Brunk, Nathan Mih, Anand Sastry, Zachary King, Rikiya Takeuchi, Wataru Nomura, Zhen Zhang, Hirotada Mori, et al. iml1515, a knowledgebase that computes escherichia coli traits. Nature biotechnology, 35(10):904–908, 2017.

31. Minoru Kanehisa and Susumu Goto. Kegg: kyoto encyclopedia of genes and genomes. Nucleic acids research, 28(1):27–30, 2000.

32. S´ebastien Moretti, Van Du T Tran, Florence Mehl, Mark Ibberson, and Marco Pagni. Metanetx/mnxref: unified

namespace for metabolites and biochemical reactions in the context of metabolic models. Nucleic acids research, 49(D1):D570–D574, 2021.

33. Ali Ebrahim, Joshua A Lerman, Bernhard O Palsson, and Daniel R Hyduke. COBRApy: constraint-based reconstruction and analysis for python. BMC Systems Biology, 7(1):74, 2013.

34. Claude Nadeau and Yoshua Bengio. Inference for the generalization error. Machine Learning, 52(3):239–281, 2003.

## Supplementary Information

This supplement provides implementation details and extended analyses that complement the main text. Cross-references indicate where supplementary material provides additional detail or extends a main-text result.

## S1 Dataset Details

## S1.1 Genome-Scale Metabolic Model Overview

We evaluate HGPRGDel on three GEMs of increasing complexity: e coli core (a curated core reconstruction of Escherichia coli K-12 MG1655), iMM904 (a genome-scale reconstruction of Saccharomyces cerevisiae), and iML1515 (the most comprehensive E. coli K-12 MG1655 reconstruction to date). Table S1 summarizes key network statistics.

Table S1 Genome-scale metabolic model overview (extends main-text Table 1). “GCP targets”: GCP-annotated targets in the GEM. “Deletion strategies”: targets with a usable curated maximal deletion set used for supervision (the supervised-target count). “Target-dep. pairs” excludes pre-determined genes (Section S1.2).
<table><tr><td>GEM</td><td>|M|</td><td>|R|</td><td>|G|</td><td>GCP targets</td><td>Deletion strategies</td><td>Target-dep. pairs</td><td>Del. rate (%)</td></tr><tr><td>e_coli_core</td><td>72</td><td>95</td><td>137</td><td>48</td><td>41</td><td>2,706</td><td>43.4</td></tr><tr><td>iMM904</td><td>1226</td><td>1577</td><td>905</td><td>782</td><td>105</td><td>5,775</td><td>64.6</td></tr><tr><td>iML1515</td><td>1877</td><td>2712</td><td>1516</td><td>1085</td><td>443</td><td>13,464</td><td>52.7</td></tr></table>

All three GEMs are accessed via COBRApy [33] from COBRA-style JSON model files. GPR associations are stored as infix Boolean strings (e.g., b0114 and b0115 and b0116) and parsed by the HGPRGDel GPRRuleParser (Section S6.1).

## S1.2 MetNetComp Label Construction

For each GEM and target metabolite, one MetNetComp maximal strategy table is used with columns Gene and Deleted. MetNetComp provides the reference strategies used as supervised learning targets and recovery references in the protocol. We set $y _ { g , m } = 0$ for deleted genes and $y _ { g , m } = 1$ for retained genes. In the processed Label/ CSV files, the Deleted column stores this encoding directly (0=deleted, 1=retained), consistent with the HGPRGDel release format.

Reported macro-averaged precision, recall, and F1 average the per-class metrics for $y { = } 0$ and y=1 with equal weight; AUC treats y=1 (retained) as the positive class.

Each GEM uses one curated reference gene set per target metabolite. We refer to these computationally curated sets as reference gene sets throughout. On e coli core and iMM904, every gene in the GEM roster appears in each strategy table, so supervised pairs equal the full roster $( | G | \times | \mathcal { M } _ { \mathrm { t a r g e t } } | )$ . On iML1515, 618,528 gene–metabolite pairs have explicit MetNetComp rows $( y _ { g , m } \in \{ 0 , 1 \} )$ the remaining 53,060 roster slots have no strategy-table row and are omitted from supervised-pair counts; these slots correspond to 35 iML1515 targets that have no supervised labels in the curated reference; these 35 targets contribute no supervised gene–metabolite pairs and remain in the held-out roster only for cross-validation bookkeeping. The target-dependent-pair count follows from the same accounting: the $1 5 1 6 - 1 4 8 3 = 3 3$ target-dependent genes give $3 3 \times 4 4 3 = 1 4 { , } 6 1 9$ candidate pairs, of which 1,155 fall in the unlabeled remainder above, leaving $1 4 , 6 1 9 - 1 , 1 5 5 = 1 3 , 4 6 4$ reported target-dependent pairs.

Pre-determined genes (those whose majority deletion label holds for a fraction $\geq \rho _ { \mathrm { l a b e l } }$ of their supervised metabolites, including growth-essential genes) remain in the training matrix but are excluded from all reported metrics. Here $\rho _ { \mathrm { l a b e l } } { = } 0 . 9$ is the labelconsistency threshold, a data-preparation constant distinct from the GPR-M soft-AND temperature τ of Section S7.1 and from the decision threshold θ of Section S2.5. The complement is termed the target-dependent set and forms the basis of all reported metrics. It consists of genes whose labels vary across supervised targets.

## S1.3 Label Composition and Class Imbalance

Gene deletion strategies exhibit strong class imbalance: across GEMs, 43–65% of target-dependent pairs carry a deletion label. All reported main-text and ablation tables use no class balancing during training (Section S2.4).

## S1.4 Cross-Validation Partitions

All reported results use metabolite-based five-fold cross-validation with fixed random seed 42. Target metabolites are shufled once and partitioned evenly; all gene labels for a metabolite remain in the same fold. Across the five rounds, every supervised target metabolite is evaluated exactly once (complete out-of-fold test coverage). The target-dependent gene set is fixed across folds; only the target metabolites change between rounds.

Table S2 Label composition across the three GEMs (curated supervision). “Roster pairs” is $\vert G \vert \times \vert \mathcal { M } _ { \mathrm { t a r g e t } } \vert$ (on iML1515, 53,060 roster slots lack explicit curated labels and are excluded from supervised pair counts). “Target-dep. pairs” and “Del. pairs (target-dep.)” exclude pre-determined genes (majority-label consistency $\geq 0 . 9 ;$ Section S1.2). “Del. rate” is the fraction of target-dependent pairs with $y _ { g , m } = 0$
<table><tr><td>GEM</td><td>Roster pairs</td><td>Target-dep. pairs</td><td>Del. pairs (target-dep.)</td><td>Pre-det. genes</td><td>Del. rate (%)</td></tr><tr><td>e_coli_core</td><td>5,617</td><td>2,706</td><td>1,175</td><td>71</td><td>43.4</td></tr><tr><td>iMM904</td><td>95,025</td><td>5,775</td><td>3,730</td><td>850</td><td>64.6</td></tr><tr><td>iML1515</td><td>671,588</td><td>13,464</td><td>7,097</td><td>1,483</td><td>52.7</td></tr></table>

Table S3 Metabolite-based five-fold cross-validation partitions (seed 42). Entries are the numbers of roster gene–metabolite pairs in the held-out metabolite set for each fold. Counts use the full gene roster, including pre-determined genes. Target-dependent pair counts are reported in Table S2. On iML1515, counts include unlabeled roster slots.
<table><tr><td rowspan="2">GEM</td><td rowspan="2">Fold 1</td><td rowspan="2">Fold 2</td><td rowspan="2">Fold 3</td><td colspan="2"></td></tr><tr><td>Fold 4</td><td>Fold 5</td></tr><tr><td>e_coli_core</td><td>1,096</td><td>1,096</td><td>1,096</td><td>1,096</td><td>1,233</td></tr><tr><td>iMM904</td><td>19,005</td><td>19,005</td><td>19,005</td><td>19,005</td><td>19,005</td></tr><tr><td>iML1515</td><td>133,408</td><td>133,408</td><td>133,408</td><td>133,408</td><td>137,956</td></tr></table>

## S1.5 GPR Rule Complexity

Table S4 reports the distribution of GPR rule types across the three evaluated GEMs.

## S2 Implementation Details

## S2.1 Hardware and Software Environment

We run single-GPU training on an NVIDIA A100-PCIE-40GB GPU with a Xeon Gold 6258R CPU. All experiments use PyTorch with the PyG (PyTorch Geometric) hypergraph convolution implementation. Software dependencies and version information are provided with the code release.

## S2.2 Downstream FBA Evaluation Protocol

COBRApy [33] is used for FBA re-simulation and GEM parsing. MetNetComp curated deletion sets are taken as the reference labels and are not re-verified [8]. Weak growth-coupling checks are applied only to predicted (and matched-size random) knockout sets, using the biomass-then-production criteria of prior constraint-based studies [19, 8]: production flux uses ⟨targetMet⟩ if present, else DM ⟨targetMet⟩, else a newly created demand on that exact metabolite; growth is maximised, then biomass is fixed at the optimum while production is maximised; success requires GR, $\mathrm { P R } \geq 0 . 0 0 1$

## S2.3 Pre-trained Models and Sequence Processing

We use ESM-2 (esm2 t6 8M UR50D) for gene amino acid sequences and ChemBERTa (DeepChem/ChemBERTa-77M-MLM) for metabolite SMILES. Both encoders are frozen; only the projection layers and downstream modules are trained end-to-end. Amino acid sequences and SMILES strings are tokenized with the respective model tokenizers; long sequences are truncated to 256 tokens to fit available GPU memory.

In the No pretrained ablation variant (main-text Table 3), the published checkpoints are not loaded: the ESM-2 and ChemBERTa architectures are instantiated from their configurations with randomly initialized weights and trained end-to-end together with the projection layers and downstream modules. This holds the architecture, parameter count, and information flow of the frozen-encoder configuration fixed, so the variant isolates the contribution of the pretrained weights themselves. Because the backbones are trainable in this variant, the frozen pooled states are neither precomputed nor reused: its trainable parameter count necessarily exceeds that of the full model, in which both language-model backbones are frozen.

Because the language-model backbones are frozen, their pooled hidden states are precomputed and cached from sequences and SMILES alone, without access to labels, before any fold-specific supervised fitting under cross-validation. This caching protocol ensures that sequence representations are computed identically across folds and eliminates redundant forward passes.

SMILES strings for metabolites were obtained from MetaNetX [32]. Compound-node coverage is 100.0% on e coli core, 86.3% on iMM904 and 89.9% on iML1515. The uncovered remainder consists of compounds for which no structure is available, chiefly acylcarrier proteins, charged tRNAs and polymers whose model formula is a generic repeat unit (C11H21N2O7PRS, C1920H3622O700P100) rather than a single molecule. A metabolite without a SMILES string is passed to ChemBERTa as an empty sequence, which tokenizes to [CLS] [SEP] and yields a single fixed embedding shared by all such metabolites; it is projected to $d _ { m } = 2 5 6$ dimensions by the same linear layer as every other metabolite. These nodes are therefore mutually indistinguishable to the model rather than zero-valued, and are reachable through the hypergraph only via their reaction neighborhood.

Table S4 GPR rule complexity across the three GEMs. Depth is the root-to-leaf length of the parsed Boolean tree. Percentages are over all reactions; the largest absolute number of multi-gene rules occurs on iML1515 (2,712 reactions), whereas multi-gene proportions are highest on e coli core.
<table><tr><td>GEM</td><td>|R|</td><td>Med. depth</td><td>Max depth</td><td>Single gene (%)</td><td>Flat AND (%)</td><td>Flat OR (%)</td><td>Nested (%)</td><td>Empty (%)</td></tr><tr><td>e_coli_core</td><td>95</td><td>1</td><td>4</td><td>28.4</td><td>10.5</td><td>28.4</td><td>5.3</td><td>27.4</td></tr><tr><td>iMM904</td><td>1577</td><td>1</td><td>3</td><td>43.9</td><td>4.5</td><td>13.6</td><td>4.2</td><td>33.9</td></tr><tr><td>iML1515</td><td>2712</td><td>1</td><td>4</td><td>48.0</td><td>7.6</td><td>24.0</td><td>3.9</td><td>16.5</td></tr></table>

## S2.4 Full Hyperparameter Settings

Table S5 provides the complete hyperparameter configuration used for all reported experiments. No per-GEM retuning was performed;   
the same configuration is applied to all three GEMs.

Table S5 Full hyperparameter configuration for HGPRGDel. All reported experiments use these settings without per-GEM retuning.  
Parameter Value   
Gene representation dim $\left( d _ { g } \right)$ 320   
Metabolite representation dim $\left( d _ { m } \right)$ 256   
Hypergraph conv dim 128   
Hypergraph conv layers 2   
GPR-M reaction-embedding dim $d _ { g } \ ( = 3 2 0 ; { \tt G P R - M o u t p u t } )$   
Reaction-context dim $( D _ { \mathsf { r x n } } )$ 128 (association-MLP output, $d _ { g }  D _ { \mathsf { r x n } } )$   
Classifier input dim $d _ { g } + d _ { m } + D _ { \mathsf { r x n } } = 7 0 4$   
Prediction MLP hidden dims $2 5 6 \to 1 2 8 \to 1$   
GPR-M MHA heads (H) 4   
Dropout 0   
Optimizer Adam $( \beta _ { 1 } { = } 0 . 9 , \beta _ { 2 } { = } 0 . 9 9 9 )$   
Learning rate $1 0 ^ { - 3 }$   
Weight decay none (0)   
Batch size 16   
Class balancing none: no per-class weight term   
Gene-pair loss weighting per-gene label-entropy weight   
$( \gamma { = } 1 . 0 , \epsilon { = } 0 . 0 5 ;$ Section S2.5)   
Inner validation split 20% of each fold’s training partition   
Model selection best inner-validation AUC   
Decision threshold θ macro-F1-optimal on the inner validation split,   
grid [0.1, 0.9] (81 values), fixed after early stopping   
GPR-M soft-AND temperature $\tau$ softplus $( \hat { \tau } ) + 1 0 ^ { - 6 }$   
$\hat { \tau }$ init. so $\tau \approx 1$   
GPR-M OR-branch temperature none (log-sum-exp is untempered)   
Max epochs 50   
Early stopping patience 5

## S2.5 Training Procedure

Each of the five outer folds reserves 20% of its training partition as an inner validation split; the remaining 80% is the inner training set, and every supervised gene–metabolite pair of a fold is drawn from that inner training set. Training runs for up to 50 epochs (Adam, learning rate $1 0 ^ { - 3 }$ , weight decay 0, batch size 16) with early stopping driven by the inner-validation AUC computed over target-dependent gene–metabolite pairs. AUC is threshold-free, which is what decouples the stopping decision from the choice of operating point. If the inner-validation AUC does not improve for 5 consecutive epochs, training is halted and the best-AUC state is restored; the outer test fold is never consulted. The per-fold states behind the results reported in the manuscript are the per-fold model states, i.e. the best inner-validation states this rule selects, so no additional training was run for them.

Only after the model state is fixed is the decision threshold θ selected, by maximizing macro-F1 on the same inner validation split over target-dependent pairs on a grid of 81 values in [0.1, 0.9]. This is the unweighted macro-F1 that the tables report, deliberately not the gene-pair-weighted training objective. The resulting fold-specific θ is applied unchanged to the held-out test metabolites. Each batch consists of gene–metabolite pairs sampled from the inner training set. The training objective is a per-gene weighted binary cross-entropy. For a batch B of gene–metabolite pairs with logits $z _ { g , m }$ and labels $y _ { g , m }$

$$
\begin{array} { r l } & { \mathcal { L } ( \mathcal { B } ) \ = \ \frac { \sum _ { ( g , m ) \in \mathcal { B } } w _ { g } \ell \big ( y _ { g , m } , \hat { y } _ { g , m } \big ) } { \sum _ { ( g , m ) \in \mathcal { B } } w _ { g } } , } \\ & { \ell ( y , \hat { y } ) \ = \ - \big [ y \log \hat { y } + ( 1 - y ) \log ( 1 - \hat { y } ) \big ] , \qquad \hat { y } _ { g , m } \ = \ \mathrm { s i g m o i d } ( z _ { g , m } ) , } \end{array}\tag{S1}
$$

where the weight depends on the gene alone,

$$
w _ { g } \ = \ \epsilon + ( 1 - \epsilon ) H _ { g } ^ { \gamma } , \qquad H _ { g } \ = \ - \big [ p _ { g } \log _ { 2 } p _ { g } + ( 1 - p _ { g } ) \log _ { 2 } ( 1 - p _ { g } ) \big ] , \qquad p _ { g } \ = \ \frac { \operatorname* { m a x } \left( n _ { g } ^ { \mathrm { d e l } } , \ n _ { g } ^ { \mathrm { r e t } } \right) } { n _ { g } ^ { \mathrm { d e l } } + n _ { g } ^ { \mathrm { r e t } } } ,\tag{S2}
$$

with $\gamma { = } 1 . 0 , \epsilon { = } 0 . 0 5$ , and $n _ { g } ^ { \mathrm { d e l } } , n _ { g } ^ { \mathrm { r e t } }$ the numbers of deleted and retained supervised metabolites for gene g. $H _ { g } \in [ 0 , 1 ]$ is the binary label entropy in bits, so $w _ { g }$ ranges from ϵ for a gene whose label is constant across its supervised metabolites to 1 for a perfectly balanced gene.

Equations (S1)–(S2) clarify how this weighting relates to the “Class balancing: none” entry of Table S5. (i) $w _ { g }$ is a function of the gene only and never of the label, so within a gene the deleted and retained pairs receive the same weight. The scheme therefore contains no per-class weight parameter, which is what “no class balancing” reports; the weighting is over gene–metabolite pairs, not over classes. (ii) It is nevertheless not equivalent to unweighted BCE. Unweighted BCE is recovered exactly only when $\gamma { = } 0$ (then $w _ { g } \equiv 1 )$ or when every gene is balanced $( H _ { g } \equiv 1 )$ ; at γ=1.0 the objective down-weights near-constant genes, reducing the efective pair count to $\begin{array} { r } { \sum _ { g } w _ { g } = 0 . 5 2 , } \end{array}$ 0.11 and 0.09 of the supervised pair count of Section S1.3 on e coli core, iMM904 and iML1515, respectively. (iii) Because genes difer in class composition, the aggregate efective class ratio shifts as a side efect: the efective share of deleted pairs falls from 49.0%, 74.1% and 79.5% (unweighted) to 45.2%, 63.8% and 70.8% (entropy-weighted) on the three GEMs, redistributing weight toward retained labels. This rebalancing is a by-product of down-weighting low-entropy genes rather than an explicit class-balancing term; all reported metrics are computed unweighted throughout (Section S2.5).

The evaluation split is defined over target metabolites, not over sequence clusters: the same gene may appear in both training and test folds with diferent metabolite partners because deletion labels are metabolite-conditioned. Fold composition and pair counts are reported in Table S3. This nested protocol is the source of every classification result in the manuscript (main-text Tables 2 and 3, Tables S6, S7 and S9–S12, and main-text Fig. 3).

## S2.6 Baseline Comparison Protocol

All five methods (DNN, DeepGDel, NMA, GraphGDel, HGPRGDel) use metabolite-based five-fold cross-validation (seed 42, Table S3) with no class balancing during training. Held-out target metabolites and the complete, unmasked graph inputs are identical across methods. All baselines are trained under the same inner-validation early stopping and the same inner-validation macro-F1 threshold rule described in Section S2.5, so every method’s reported macro-F1 and AUC come from an operating point chosen without reference to the test fold. NMA follows the GraphGDel definition: DeepGDel LSTM encoders with fixed neighborhood mean aggregation on the GraphGDel metabolite pairing graph $\mathcal { G } _ { \mathrm { f i n a l } }$ (not the HGPRGDel compound hypergraph). Within-model comparisons (ablations in main-text Table 3) share the same folds and seeds as the HGPRGDel full model; for No pretrained this means the two difer only in backbone weight initialization and trainability.

Table S6 summarizes the training configuration of each evaluated method.

## S2.6.1 Graph Convention and the Declared Inductive Sensitivity

Every reported number in the manuscript comes from one five-fold protocol and one graph convention: the model is forwarded through the complete GEM, and the folds difer only in which gene–metabolite pairs are supervised, never in the graph the model sees.

Main-text Tables 2 and 3, Tables S6, S7 and S9–S12 and main-text Fig. 3 report per-fold metrics with each fold’s own frozen θ; the mechanism, deletion-recovery, operating-point and case-study analyses (main-text Tables 4, Figs. 4–6 and Tables S8, S13–S19) average the five folds’ scores first and then binarize with the mean of the five folds’ θ.

Table S6 Training configurations of the evaluated methods.
<table><tr><td>Method</td><td>Input representation</td><td>Pretrained backbone</td><td>Optimization</td><td>Training</td></tr><tr><td>DNN</td><td>Sequence encoders (from scratch), no graph</td><td>None</td><td>Adam, 0.001</td><td>50 epochs, patience 5</td></tr><tr><td>DeepGDel</td><td>LSTM sequence encoders (from scratch)</td><td>None</td><td>Adam, 0.001</td><td>50 epochs, patience 5</td></tr><tr><td>NMA</td><td>DeepGDel encoders + fixed one-hop neighborhood-mean graph feature</td><td>None</td><td>Adam, 0.001</td><td>50 epochs, patience 5</td></tr><tr><td>GraphGDel</td><td>LSTM sequence encoders + GCN (from scratch) on pairwise metabolite graph</td><td>None</td><td>Adam, 0.001</td><td>50 epochs, patience 5</td></tr><tr><td>HGPRGDel</td><td>ESM-2 (gene) + ChemBERTa (metabolite)</td><td>ESM-2 / ChemBERTa (frozen)</td><td>Adam, 0.001</td><td>50 epochs, patience 5</td></tr></table>

Configuration values are taken directly from the shared experiment protocol; per-method architecture sizes otherwise follow the released baseline implementations. Trainable parameter counts are not reported here, as they were not recorded consistently across the evaluated implementations.

## S3 Additional Results

## S3.1 Full Performance Comparison (All Metrics)

Main-text Table 2 reports macro-F1 and AUC. Table S7 extends this to all five metrics.

Table S7 Full baseline comparison: all five metrics across three GEMs (mean ± s.d. over five held-out metabolite test folds on targetdependent pairs). Bold: best per GEM block. Every row reports per-fold metrics with each fold’s own frozen $\theta ,$ under the convention of Section S2.6.1. Tabulated metrics are computed on the target-dependent pairs (evaluable pairs: 2706, 5775 and 13464 for e coli core, iMM904 and iML1515). The target-dependent gene sets are defined with the label-consistency threshold $\rho _ { \mathrm { l a b e l } } { = } 0 . 9$ (Section S1.2) in all three GEMs.
<table><tr><td>GEM</td><td>Method</td><td>Accuracy (%)</td><td>M-Prec. (%) M-Prec. = macro-precision; M-Recall = macro-recall; AUC = area under the ROC curve.</td><td>M-Recall (%)</td><td>Macro-F1 (%)</td><td>AUC (%)</td><td></td></tr><tr><td rowspan="5">e_coli_core</td><td>DNN</td><td> $6 2 . 7 7 \pm 3 . 8 4$ </td><td> $6 4 . 3 1 \pm 3 . 6 6$ </td><td> $6 3 . 5 8 \pm 3 . 3 8$ </td><td> $5 8 . 9 4 \pm 3 . 5 3 $ </td><td></td><td> $6 2 . 1 8 \pm 3 . 9 6$ </td></tr><tr><td>DeepGDel</td><td> $7 0 . 8 3 \pm 3 . 5 7$ </td><td> $7 1 . 9 2 \pm 3 . 6 2$ </td><td> $7 2 . 4 1 \pm 3 . 6 0$ </td><td> $7 0 . 1 6 \pm 3 . 7 3$ </td><td></td><td> $7 0 . 4 7 \pm 3 . 9 0$ </td></tr><tr><td>NMA</td><td> $7 1 . 5 4 \pm 3 . 6 3$ </td><td> $7 2 . 6 5 \pm 3 . 6 7$ </td><td> $7 3 . 0 3 \pm 3 . 6 4$ </td><td></td><td> $7 0 . 9 1 \pm 3 . 7 8$ </td><td> $7 1 . 2 5 \pm 3 . 9 4$ </td></tr><tr><td>GraphGDel</td><td> $7 7 . 1 6 \pm 3 . 3 9$ </td><td> $7 6 . 1 2 \pm 3 . 3 1$ </td><td> $7 5 . 9 1 \pm 3 . 4 2$ </td><td></td><td> $7 4 . 2 2 \pm 3 . 2 7$ </td><td> $7 6 . 3 3 \pm 3 . 3 4$ </td></tr><tr><td>HGPRGDel</td><td> ${ \bf 8 9 . 3 4 \pm 3 . 7 6 }$ </td><td> ${ \bf 8 9 . 3 3 \pm 3 . 7 5 }$ </td><td> ${ \bf 8 9 . 2 7 \pm 3 . 8 2 }$ </td><td></td><td> $\mathbf { 8 9 . 1 1 \pm 3 . 8 1 }$ </td><td> ${ \bf 9 6 . 4 2 \pm 1 . 5 7 }$ </td></tr><tr><td rowspan="5">iMM904</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DNN</td><td> $6 9 . 5 5 \pm 1 . 6 9$ </td><td> $7 1 . 6 4 \pm 1 . 5 1$ </td><td> $6 8 . 1 2 \pm 1 . 4 5$ </td><td> $6 5 . 4 3 \pm 1 . 6 3$ </td><td></td><td> $6 7 . 0 2 \pm 1 . 5 8$ </td></tr><tr><td>DeepGDel</td><td> $8 0 . 0 3 \pm 1 . 4 9$ </td><td> $7 7 . 0 8 \pm 1 . 4 6$ </td><td> $7 7 . 3 6 \pm 1 . 4 8$ </td><td></td><td> $7 5 . 9 8 \pm 1 . 5 1$ </td><td> $7 9 . 7 1 \pm 1 . 8 0$ </td></tr><tr><td>NMA</td><td> $8 0 . 8 6 \pm 1 . 5 5$ </td><td> $7 7 . 9 2 \pm 1 . 5 1$ </td><td> $7 8 . 0 2 \pm 1 . 5 2$ </td><td></td><td> $7 6 . 8 3 \pm 1 . 5 5$ </td><td> $8 0 . 4 6 \pm 1 . 8 6$ </td></tr><tr><td>GraphGDel</td><td> $8 5 . 2 7 \pm 1 . 3 9$ </td><td> $8 3 . 3 1 \pm 1 . 3 0$ </td><td> $8 2 . 6 1 \pm 1 . 3 7$ </td><td></td><td> $8 1 . 5 4 \pm 1 . 3 8$ </td><td> $8 4 . 0 9 \pm 1 . 3 6$ </td></tr><tr><td rowspan="5">iML1515 HGPRGDel</td><td>HGPRGDel</td><td> $\mathbf { 8 9 . 6 5 \pm 1 . 2 1 }$ </td><td> ${ \bf 8 8 . 6 4 \pm 1 . 2 7 }$ </td><td> ${ \bf 8 8 . 6 1 \pm 1 . 5 9 }$ </td><td> $\mathbf { 8 8 . 6 1 \pm 1 . 4 2 }$ </td><td></td><td> ${ \bf 9 6 . 2 2 \pm 1 . 1 2 }$ </td></tr><tr><td>DNN</td><td> $5 9 . 2 1 \pm 1 . 1 5$ </td><td> $6 3 . 9 5 \pm 1 . 2 0$ </td><td> $6 1 . 1 3 \pm 1 . 1 6$ </td><td></td><td> $5 8 . 0 2 \pm 1 . 2 1$ </td><td> $5 9 . 5 3 \pm 1 . 1 8$ </td></tr><tr><td>DeepGDel</td><td> $6 7 . 1 2 \pm 0 . 7 0$ </td><td> $6 8 . 3 3 \pm 1 . 3 9$ </td><td> $6 3 . 7 1 \pm 1 . 2 9$ </td><td> $6 2 . 7 4 \pm 1 . 3 8$ </td><td></td><td></td></tr><tr><td>NMA</td><td> $6 7 . 8 8 \pm 0 . 7 3$ </td><td> $6 9 . 0 6 \pm 1 . 4 2$ </td><td> $6 4 . 5 8 \pm 1 . 2 1$ </td><td></td><td> $6 3 . 6 2 \pm 1 . 4 0$ </td><td> $6 5 . 0 8 \pm 1 . 2 0$   $6 5 . 8 9 \pm 1 . 1 4$ </td></tr><tr><td>GraphGDel</td><td> $7 2 . 9 5 \pm 1 . 0 8$ </td><td> $7 4 . 8 3 \pm 1 . 1 2$ </td><td> $7 1 . 9 2 \pm 1 . 1 0$ </td><td></td><td> $7 2 . 0 3 \pm 1 . 0 5$ </td><td> $7 3 . 0 2 \pm 1 . 0 9$ </td></tr></table>

## S3.2 Fold-Level Paired Significance Tests

For each GEM, we compare HGPRGDel and GraphGDel on the five paired held-out test macro-F1 scores (per-fold values in Table S16) using a two-sided Wilcoxon signed-rank test, its one-sided counterpart, and a paired t-test with the Nadeau–Bengio variance correction [34]. Because HGPRGDel extends the same graph-based line of work as GraphGDel [12] with explicit GPRaware reaction modeling, the direction of the efect is expected a priori to be non-negative, which is what makes the one-sided test interpretable in addition to the two-sided one.

Table S8 Paired tests of HGPRGDel vs. GraphGDel on held-out test macro-F1 (five folds per GEM). “Mean $\Delta "$ is the mean macro-F1 diference (pp); the 95% CI is the paired t interval on the mean diference. “NB t-test” is the Nadeau–Bengio corrected resampled t-test, in which the paired standard error $s / { \sqrt { n } }$ is replaced by $s \sqrt { 1 / n + n _ { \mathrm { t e s t } } / n _ { \mathrm { t r a i n } } } ;$ with n=5 folds and $n _ { \mathrm { t e s t } } / n _ { \mathrm { t r a i n } } { = } 1 / 4$ this inflates the standard error by exactly 1.5×, to account for the dependence between the training sets of diferent folds.
<table><tr><td>GEM</td><td>Mean  $\Delta \mathsf { \Gamma } ( { \mathsf { p p } } )$ </td><td>95% Cl</td><td>Wilcoxon (two-sided)</td><td>Wilcoxon (one-sided)</td><td>NB t-test</td></tr><tr><td>e_coli_core</td><td>14.89</td><td>[10.47, 19.32]</td><td>0.0625</td><td>0.0312</td><td> $3 . 4 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>iMM904</td><td>7.07</td><td>[5.21, 8.94]</td><td>0.0625</td><td>0.0312</td><td> $2 . 2 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>iML1515</td><td>18.38</td><td>[16.95, 19.82]</td><td>0.0625</td><td>0.0312</td><td> $1 . 9 \times 1 0 ^ { - 5 }$ </td></tr></table>

On all three GEMs, HGPRGDel achieves higher macro-F1 than GraphGDel on every individual fold, and the mean diferences are large (14.9, 7.1, and 18.4 pp; Table S8). With n=5 paired folds and all five diferences in the same direction, the two-sided Wilcoxon signed-rank test attains its minimum attainable p-value $( p = 2 / 2 ^ { 5 } = 0 . 0 6 2 5 )$ . This floor is a property of assumption-free testing at this fold count rather than evidence of a small efect: the sign test and an exact permutation test over the $2 ^ { 5 }$ sign flips attain the same 0.0625 on all three GEMs. The one-sided counterpart reaches $p = 1 / 2 ^ { 5 } = 0 . 0 3 1 2 ,$ its own minimum attainable value, which is below α=0.05 under the pre-specified direction motivated above. Two caveats apply to that figure: it sits at the resolution floor of the test, and a Bonferroni correction across the three GEMs raises it to 0.094, above α=0.05. We therefore report the one-sided result alongside the two-sided one rather than in place of it.

The Nadeau–Bengio correction addresses a separate concern. The five cross-validation folds share four fifths of their training data, so fold-level scores are not independent and the standard error of the mean diference is underestimated. Replacing the paired standard error $s / { \sqrt { n } }$ by $s \sqrt { 1 / n + n _ { \mathrm { t e s t } } / n _ { \mathrm { t r a i n } } }$ (with n=5 folds and $n _ { \mathrm { t e s t } } / n _ { \mathrm { t r a i n } } { = } 1 / 4$ , so the standard error is inflated by 1.5×) still leaves $p \leq 3 . 4 \times 1 0 ^ { - 3 }$ on all three GEMs, and the corresponding corrected 95% CIs ([8.25, 21.53], [4.28, 9.87], and [16.23, 20.54]) remain well clear of zero. Taken together, these comparisons indicate a large and directionally consistent efect that is robust to the fold-dependence correction, while the assumption-free tests alone cannot attain conventional significance at n=5 folds.

## S3.3 Operating-Point Analysis from Deletion-Priority Rankings

## S3.3.1 Operating-Point Analysis on e coli core: Weak Growth Coupling from Continuous Rankings

The continuous ranking defines a sequence of candidate deletion sets at increasing intervention sizes. We therefore evaluated whether the ranking can directly localize operating points that satisfy the same weak growth-coupling criterion used throughout this analysis (Section S2.2), rather than treating gene deletion prediction as a fixed-size set prediction problem; the analysis assesses how the ranking can be used for downstream constraint-based evaluation, not FBA feasibility as a separate prediction endpoint.

This analysis was first carried out on e coli core, the smallest GEM evaluated; Section S3.3.2 repeats it on the genome-scale iMM904 reconstruction. For each of the 41 evaluated targets we sorted all 137 roster genes by deletion priority $1 - \hat { y } _ { g , m }$ in descending order, started from an empty deletion set, and added genes one at a time. After each addition we re-checked weak growth coupling using the same protocol as Section S2.2 $\left( \mathrm { G R } , \mathrm { P R } \geq 0 . 0 0 1 \right)$ and stopped at the first successful prefix D<sup>∗</sup>. By monotonicity of biomass under knockout accumulation, a prefix whose maximum biomass falls below 0.001 is lethal for every longer prefix; such a target was therefore classified as a lethal break, while a target that traversed all 137 genes without reaching weak growth coupling was classified as exhausted.

The HGPRGDel-guided ranking localized a weakly growth-coupled operating point for 26/41 targets (63.4%) after examining a mean of 41.7 genes (Table S9, Fig. S1). The first weakly growth-coupled operating point therefore occurred well before the full ranking was exhausted: across the 26 solved targets the stopping rank was 41.7 genes (median 45.5), corresponding to a relatively compact deletion set. As a control, the identical procedure was repeated with 20 randomly shufled orders per target. Random order reached weak growth coupling for a mean of only 4.3/41 targets (range 2–7 across the 20 orders); its successes were confined to nine targets, of which seven were also solved by the HGPRGDel ranking and two (ru5p D c, xu5p D c) were not. Within those nine targets, the random sets recovered far fewer reference genes than the ranking did (mean GT recall 0.077 versus $0 . 6 1 7 ;$ mean Jaccard 0.069 versus 0.616), while stopping much earlier (mean stopping rank 9.2 versus 41.7). These early random solutions therefore generally did not reproduce the curated reference gene sets. On the 32 targets that no random order solved in any of the 640 trials, the HGPRGDel ranking reached weak growth coupling for 19. Across the 15 lethal breaks, biomass was lost only after a mean of 51.5 genes had been examined, compared with 9.5 for the random control on the same targets, indicating that the ranking tends to place growth-essential genes later in the deletion order. For all 15 lethal-break targets, the computationally curated reference set itself reached the weak growth-coupling criterion under reference-informed ordering. Thus, these failures were not attributable to infeasibility of the reference set; the learned ranking reached a lethal prefix before reaching the reference-informed feasible ordering. Every failure ended in a lethal break at a rank between 34 and 81; once such a break occurs, no later prefix can satisfy the growth criterion. Placing the curated reference genes first (reference-informed ordering) reached weak growth coupling for 41/41 targets at a mean of 38.7 genes, providing an upper reference for the achievable operating-point localization. Here, reference-informed ordering denotes the ordering obtained by placing the computationally curated reference deletions first; it is used only as a reference ordering for the forward-selection analysis.

Together, these results show that the continuous ranking can be used directly as a deletion-priority order for downstream constraintbased evaluation. On e coli core, it localized weakly growth-coupled operating points for 26/41 targets, substantially exceeding randomized ordering while retaining substantially greater agreement with the curated reference gene sets.

## S3.3.2 Operating-Point Analysis on iMM904: Weak Growth Coupling at Genome Scale

We repeated the identical forward-selection procedure on iMM904, the genome-scale yeast reconstruction, to test whether operatingpoint localization extends beyond the smallest evaluated GEM. For each of the 105 evaluated targets we sorted all 905 roster genes by deletion priority $1 - \hat { y } _ { g , m }$ in descending order, started from an empty deletion set, and added genes one at a time, using the same termination rule (weak growth coupling, lethal break, or exhausted) and the same 20-random-order control as Section S3.3.1.

The HGPRGDel-guided ranking localized a weakly growth-coupled operating point for 44/105 targets (41.9%) after examining a mean of 455.0 genes (median 618.0 among solved targets; Table S10, Fig. S2). Random order reached weak growth coupling for a mean of only 9.9/105 targets (range 8–12 across the 20 orders); its successes were confined to 18 targets, of which 14 were also solved by the HGPRGDel ranking and four (2mppal e, glyc e, ppi m, ru5p D c) were not. The random sets recovered far fewer reference genes than the ranking did (mean GT recall 0.005 versus 0.681; mean Jaccard 0.005 versus 0.680), consistent with the pattern on e coli core. On the 87 targets that no random order solved in any of the 1,740 trials, the HGPRGDel ranking reached weak growth coupling for 30. Across the 61 lethal breaks, biomass was lost only after a mean of 646.8 genes had been examined, compared with 8.0 for the random control on the same targets; every failure ended in a lethal break at a rank between 576 and 660. As on e coli core, placing the curated reference genes first (reference-informed ordering) reached weak growth coupling for 105/105 targets, here at a mean of 377.4 genes against a mean reference size of 670.4 genes, indicating that the reference sets remain FBA-feasible at genome scale and that operating-point localization from the continuous ranking is not restricted to the smallest evaluated GEM. Thus, the unsolved targets under the learned ordering reflect failure to reach a feasible prefix before the lethal break rather than infeasibility of the corresponding reference deletion sets under the same criterion.

Table S9 Localization of weakly growth-coupled operating points from the continuous deletion ranking on e coli core (n=41 targets, 137 roster genes). Genes are added one at a time in decreasing deletion priority $1 - \hat { y } _ { g , m } ;$ the search stops at the first prefix that is weakly growth coupled, and “Genes examined” is that stopping rank (equal to |D<sup>∗</sup>| by construction). Recall and Jaccard are averaged over each row’s solved targets. Random order is repeated 20 times per target (mean over orders; min–max in brackets); reference-informed ordering places the curated reference genes first.
<table><tr><td>Ranking source</td><td>Successes</td><td>Genes examined</td><td>Mean GT recall</td><td>Mean Jaccard</td></tr><tr><td>HGPRGDel deletion priority</td><td>26</td><td>41.7</td><td>0.617</td><td>0.616</td></tr><tr><td>Random order (mean of 20)</td><td>4.3 [2-7]</td><td>9.2</td><td>0.077</td><td>0.069</td></tr><tr><td>Curated reference (reference-informed)</td><td>41</td><td>38.7</td><td>0.591</td><td>0.591</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

Successes are out of 41; HGPRGDel terminations were 26 weak-GC, 15 lethal break, and 0 exhausted. Mean lethal-break position is 51.5 for HGPRGDel (over its 15 lethal breaks) and 9.5 for random restricted to those same 15 targets, so the two values are paired on identical targets. Random’s “Genes examined” averages over its nine solved targets. Of the 32 targets unsolved in any of the 640 random trials, HGPRGDel solved 19. The reference-informed ordering’s mean recall is below 1 because it stops at the first weakly growth-coupled prefix, shorter than the full reference set (mean 38.7 genes examined versus mean $| \mathrm { G T } | = 6 7 . 1 ) ;$ as reference-informed ordering ranks reference genes first, it is the mean of the per-target ratios $| D ^ { * } | / | \mathrm { G T } |$ , not the ratio of means.

![](images/74c2d906c098b0b4b9b7f46815517b4d6e294dac6b38f014395969875034ed62.jpg)  
Figure S1 Localization of weakly growth-coupled operating points from continuous deletion rankings on e coli core (n=41 targets): cumulative fraction reaching weak growth coupling versus genes examined. The x-axis counts every gene examined, including those added before a lethal break, so such targets advance the x-axis without ever contributing to the cumulative count and a plateau reflects targets that ordering cannot solve. HGPRGDel priority (solid); random order (mean over 20 shufles, band = min–max); curated reference genes placed first (reference-informed ordering, dashed).

## S3.3.3 Search-Space Restriction under Alternative Gene Rankings

Unlike the forward-selection analyses above, which evaluate whether prefixes of a ranking already contain a feasible strategy, this analysis asks whether a ranking can be used to restrict the candidate intervention space supplied to a constraint-based search algorithm. gDel-minRN does not simply test the restricted gene pool for a feasible subset; it searches that pool with its own MILP objective, which minimizes the number of repressed reactions subject to feasibility. Success rates between the two analyses are therefore not directly comparable.

To test whether this efect is specific to HGPRGDel’s ranking or would arise under any reasonably constructed ranking, we repeated the analysis using gDel-minRN [18], the MILP-based search algorithm used to construct the MetNetComp reference strategies, with three gene rankings: HGPRGDel’s deletion-priority ranking, GraphGDel’s retention scores (obtained from its existing trained checkpoint without retraining), and five independent random gene orderings per target. All three rankings used the same gDelminRN search procedure, budgets (B ∈ {5, 10, 20, 30, 50, 100}%, i.e. 7 to 137 of the 137 roster genes), and feasibility thresholds (GR, PR ≥ 0.001, maxLoop = 10) described above; no B = 100% run was truncated at this limit, and an internal consistency check confirmed that restricting the HGPRGDel-ranked candidate pool to all 137 genes reproduced the unrestricted search exactly for all 41 targets.

At 5–20%, random ranking yielded higher success rates (7.8–14.1%) than HGPRGDel (4.9–9.8%) or GraphGDel (2.4–9.8%). From 30% onward, HGPRGDel achieved a higher success rate than both alternatives, reaching 78.0% at 50%, compared with 70.7% for GraphGDel and 35.1% for random ranking (Fig. S3).

The reversal at very small budgets indicates that restricting the candidate pool to the highest-ranked genes is not uniformly beneficial when the allowed pool is extremely small. One possible explanation is that a highly concentrated ranking may provide less pathway diversity within a small prefix than a random sample, although this possibility was not directly tested here.

Mean search iterations among successful runs remained within a narrow range (1.00–1.90) across ranking sources and budgets, with no consistent ranking-dependent pattern (Fig. S3). Thus, the primary efect of the ranking was on the probability that a feasible solution was contained in the restricted intervention space, rather than on the number of search iterations once a solution was reachable.

Detailed per-budget values for HGPRGDel, including recall and Jaccard overlap with the MetNetComp reference set, are reported in Table S11 for completeness rather than as the primary endpoint of this analysis: gDel-minRN is formulated to minimize the number of repressed reactions subject to feasibility, not to reproduce a specific curated reference strategy, so enlarging the allowed gene pool does not necessarily increase its overlap with the reference set.

This experiment therefore supports a ranking- and budget-dependent search-space coverage–feasibility trade-of, rather than a uniform claim that learned ranking improves constraint-based search at every budget. This analysis was performed as a controlled case study on e coli core, chosen because its small gene roster allowed the unrestricted (B = 100%) search to be run to completion rather than approximated; it does not constitute a claim about computational savings on larger GEMs.

## S3.4 Full-Gene Supplementary Evaluation

The primary evaluation (main-text Experimental Setup) is restricted to target-dependent genes, since this is the scientific question of interest: deletion behavior that varies with the target metabolite. Pre-determined genes remain part of the supervised training signal (Methods) but are excluded from the reported metrics. As a supplementary check, Table S12 recomputes macro-F1 and AUC over the complete gene roster of each GEM, using the same fold-averaged predictions that underlie the primary evaluation.

## S3.5 All-Target Top-10 Recovery

The main-text case study (“A Target-Level Case Study of GPR-Aware Deletion Prioritization”) reports top-10 recovery for a single representative target (succinate). Table S13 extends the same diagnostic to every target with at least ten evaluable target-dependent genes. For each target, we count how many of the ten lowest-retention target-dependent genes match the curated reference.

## S3.6 Sensitivity to the Target-Dependent Gene Definition

The primary evaluation treats a gene as target-dependent as defined in Section S1.2. Table S14 recomputes macro-F1 and AUC under two alternative label-consistency thresholds $\rho \in \{ 0 . 8 0 , 0 . 9 5 \}$ , re-partitioning the same fold-averaged predictions into a targetdependent subset and reusing the existing per-fold decision threshold θ (i.e., this is a sensitivity of the reported evaluation to how the target-dependent gene set is defined, not a re-run of training or threshold selection).

Table S10 Localization of weakly growth-coupled operating points from the continuous deletion ranking on iMM904 (n=105 targets, 905 roster genes), computed identically to Table S9.
<table><tr><td>Ranking source</td><td>Successes</td><td>Genes examined</td><td>Mean GT recall</td><td>Mean Jaccard</td></tr><tr><td>HGPRGDel deletion priority</td><td>44</td><td>455.0</td><td>0.681</td><td>0.680</td></tr><tr><td>Random order (mean of 20)</td><td>9.9 [8-12]</td><td>3.8</td><td>0.005</td><td>0.005</td></tr><tr><td>Curated reference (reference-informed)</td><td>105</td><td>377.4</td><td>0.563</td><td>0.563</td></tr></table>

Successes are out of 105; HGPRGDel terminations were 44 weak-GC, 61 lethal break, and 0 exhausted. Mean lethal-break position is 646.8 for HGPRGDel (over its 61 lethal breaks) and 8.0 for random restricted to those same 61 targets; random’s “Genes examined” averages over its 18 solved targets; of the 87 targets unsolved in any of the 1,740 random trials, HGPRGDel solved 30. As in Table S9 (see that table’s note for why this is the mean of per-target ratios, not the ratio of means), the reference-informed ordering’s mean recall is below 1 because it stops at the first weakly growth-coupled prefix (mean 377.4 genes examined versus mean |GT| = 670.4).

![](images/dadaf78033109800218a9a528516850887d72799bf291a85e6f09da884156787.jpg)  
Figure S2 Localization of weakly growth-coupled operating points from continuous deletion rankings on iMM904 (n=105 targets): cumulative fraction reaching weak growth coupling versus genes examined. The x-axis counts every gene examined, including those added before a lethal break, so such targets advance the x-axis without ever contributing to the cumulative count and a plateau reflects targets that ordering cannot solve. HGPRGDel priority (solid); random order (mean over 20 shufles, band = min–max); curated reference genes placed first (reference-informed ordering, dashed).

Table S11 Search-space restriction and downstream gDel-minRN outcomes on held-out e coli core targets (n=41). Mean recall and Jaccard are computed against the MetNetComp reference strategy and are reported for completeness rather than as the primary endpoint.
<table><tr><td>Budget (%)</td><td>Genes allowed</td><td>Successes</td><td>Mean search iterations</td><td>Mean GT recall</td><td>Mean Jaccard</td></tr><tr><td>5</td><td>7</td><td>2/41</td><td>1.00</td><td>0.074</td><td>0.074</td></tr><tr><td>10</td><td>14</td><td>2/41</td><td>1.00</td><td>0.181</td><td>0.181</td></tr><tr><td>20</td><td>28</td><td>4/41</td><td>1.00</td><td>0.404</td><td>0.403</td></tr><tr><td>30</td><td>42</td><td>10/41</td><td>1.10</td><td>0.607</td><td>0.601</td></tr><tr><td>50</td><td>69</td><td>32/41</td><td>1.59</td><td>0.885</td><td>0.822</td></tr><tr><td>100</td><td>137</td><td>41/41</td><td>1.59</td><td>0.874</td><td>0.653</td></tr></table>

Mean search iterations are averaged over successful runs only. Budget 100% corresponds to the unrestricted gDel-minRN search; an internal consistency check (restricting the candidate pool to all 137 genes explicitly) reproduced this result exactly for all 41 targets.

![](images/a3d5e8cd205103877b01a889a3d0a121eedb59cb74297fe2ffa5ad1e9d450dcd.jpg)

![](images/db5efbf46c09b4ad01ce17a2f38c7cd6bc99f7ac9e0509dab1de183a87ecafab.jpg)  
Figure S3 Restricting gDel-minRN’s candidate gene pool to the top-B% of a gene ranking, on held-out e coli core targets (n=41), under three ranking sources: HGPRGDel’s deletion-priority ranking, GraphGDel’s retention scores, and random gene order (mean over 5 independent shufles per target). (A) Fraction of targets for which a feasible growth-coupled solution exists within the allowed gene pool, as a function of B. (B) Mean number of search iterations required by gDel-minRN among the runs that found a feasible solution, as a function of B.

Table S12 Full-gene supplementary evaluation. “Target-dependent” reproduces the primary evaluation subset (pooled over the fold-averaged predictions); “Full gene roster” additionally includes pre-determined genes.
<table><tr><td>GEM</td><td>Target-dependent Macro-F1</td><td>Target-dependent AUC</td><td>Full-gene Macro-F1</td><td>Full-gene AUC</td></tr><tr><td>e_coli_core</td><td>89.04%</td><td>96.48%</td><td>92.20%</td><td>97.93%</td></tr><tr><td>iMM904</td><td>88.81%</td><td>96.25%</td><td>97.60%</td><td>99.76%</td></tr><tr><td>iML1515</td><td>90.40%</td><td>97.10%</td><td>97.98%</td><td>99.67%</td></tr></table>

Values are pooled over the fold-averaged predictions rather than averaged per-fold, so they difer slightly from the per-fold mean ± s.d. reported in main-text Tables 2 and 3. Full-gene macro-F1/AUC are higher than the target-dependent subset on all three GEMs, consistent with the primary-evaluation rationale (main-text Experimental Setup): pre-determined genes are, by construction, easier to classify from their near-constant label alone, so including them raises the aggregate score without adding condition-dependent evidence.

Table S13 All-target top-10 recovery, computed identically to the main-text case study for every evaluated target.
<table><tr><td>GEM</td><td>Targets evaluated</td><td>Mean top-10 hit rate</td><td>Targets with 10/10 hits</td></tr><tr><td></td><td>41</td><td>98.0%</td><td>36/41 (87.8%)</td></tr><tr><td>e_coli_core iMM904</td><td>105</td><td>100.0%</td><td>105/105 (100.0%)</td></tr><tr><td>iML1515</td><td>408</td><td>99.5%</td><td>400/408 (98.0%)</td></tr></table>

On iML1515, the target-dependent gene set contains only 33 genes, so the ten lowest-retention calls represent a substantial fraction of the evaluable genes; the hit rate should therefore be interpreted as a diagnostic of ranking concentration rather than as an operating-point performance metric.

Table S14 Sensitivity to the target-dependent gene definition. $\rho { = } 0 . 9 0$ reproduces the primary evaluation.
<table><tr><td>GEM ρ</td><td>Target-dependent genes</td><td></td><td>Macro-F1</td><td>AUC</td></tr><tr><td>e_coli_core</td><td></td><td>41</td><td>87.98%</td><td>96.19%</td></tr><tr><td></td><td>0.80 0.90</td><td>66</td><td>89.04%</td><td>96.48%</td></tr><tr><td rowspan="4">iMM904</td><td>0.95</td><td>80</td><td>89.54%</td><td>96.83%</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>0.80 0.90</td><td>21 55</td><td>77.39% 88.81%</td><td>84.95%</td></tr><tr><td>0.95</td><td>96</td><td>91.73%</td><td>96.25% 97.86%</td></tr><tr><td>iML1515</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>0.80 0.90</td><td>21</td><td>81.22% 90.40%</td><td>92.34%</td></tr><tr><td></td><td>0.95</td><td>33 68</td><td>92.14%</td><td>97.10% 98.23%</td></tr></table>

Macro-F1/AUC are pooled as in Table S12, so the ρ=0.90 rows difer slightly from main-text Tables 2 and 3. On iMM904 and iML1515, the stricter $\rho { = } 0 .$ 80 threshold retains a smaller and more label-diverse gene subset, on which macro-F1 is correspondingly lower; the qualitative pattern is consistent across all three GEMs. Higher ρ retains more genes and increases the aggregate score, and macro-F1 increases monotonically with ρ on every GEM.

## S3.7 Target-Level Bootstrap Uncertainty

As a complement to the fold-level Nadeau–Bengio analysis (Section S3.2), Table S15 reports a 95% confidence interval on HGPRGDel’s own macro-F1 and AUC obtained by resampling target metabolites with replacement (2,000 bootstrap resamples, seed 42) from the fold-averaged predictions. This characterizes uncertainty in HGPRGDel’s reported metrics under the target-based evaluation unit; it is not a paired comparison against other methods, since no per-target baseline predictions under the matched cross-validation protocol are available (Section S2.6).

Table S15 Target-level bootstrap 95% confidence intervals for HGPRGDel (target-dependent genes, 2,000 resamples over target metabolites, seed 42).
<table><tr><td>GEM</td><td>Targets</td><td>Macro-F1 (95% Cl)</td><td></td><td></td><td>AUC (95% CI)</td></tr><tr><td></td><td></td><td></td><td>[85.55, 92.35]</td><td>96.48%</td><td>[94.61, 98.00]</td></tr><tr><td>e_coli_core</td><td>41 105</td><td>89.04% 88.81%</td><td>[87.44, 90.05]</td><td>96.25%</td><td>[95.44, 96.94]</td></tr><tr><td>iMM904 iML1515</td><td>443</td><td>90.40%</td><td>[89.60, 91.20]</td><td>97.10%</td><td>[96.70, 97.50]</td></tr></table>

## S4 Additional Ablation

## S4.1 Ablation Variant Definitions

Each component-ablation variant (main-text Table 3; Table S17) alters exactly one architectural component of the full model. All variants retain the same gene and metabolite encoder architectures, the same prediction MLP (256 → 128 → 1), the same classifier input dimension $( d _ { g } + d _ { m } + D _ { \bf r x n } = 3 2 0 + 2 5 6 + 1 2 8 = 7 0 4 )$ , and the same cross-validation folds, seeds, and threshold protocol; only the component named below is changed. For every variant except No pretrained the encoders are the frozen ESM-2 and ChemBERTa backbones of the full model; No pretrained keeps the same architectures but replaces the frozen pretrained weights with random initialization, as detailed below.

• Mean Pooling. GPR-M stays enabled, but the SoftBooleanAggregator is replaced by a parameter-free aggregator that returns the arithmetic mean of the embeddings of all leaf genes of the reaction’s GPR rule (unique gene names, collected recursively). The parse tree is never evaluated, so AND/OR structure, the OR-branch attention module, and the learnable soft-AND temperature parameter ˆτ are all discarded. The output dimension remains $d _ { g } ,$ and the resulting reaction embeddings still feed Graph-M and Pred-M exactly as in the full model, so downstream modules and the classifier input are unchanged.

• No Graph-M. Hypergraph propagation is disabled ( ). The target metabolite representation degenerates to the raw Meta-M output, i.e. the frozen ChemBERTa embedding after its linear projection, with no propagation over reaction– incidence structure. GPR-M and Pred-M are fully retained. Because the graph encoder acts on the metabolite side only, gene representations are unafected both in the full model and in this variant.

• No GPR-M. GPR-M is disabled (use gpr=False) and no reaction embeddings are computed. Graph-M stays enabled, but its GPR-conditioned hyperedge channel is replaced by a zero block at every layer; in Pred-M, the reaction-context term becomes a zero vector passed through the association MLP, which returns a learned, pair-independent constant. GPR-derived information is therefore removed from both the topology and association pathways, while the classifier input dimension is unchanged.

• No pretrained. Both language-model encoders keep the ESM-2 and ChemBERTa architectures but are instantiated from their configurations with randomly initialized weights, and are trained end-to-end instead of being frozen (Supplementary Section S2.3). This variant shares the folds and seeds of the full model; the only diferences are the backbone weight initialization and the fact that both backbones are trainable rather than frozen.

No Graph-M is not the same as the NMA baseline of main-text Table 2: NMA disables both Graph-M and GPR-M and substitutes a fixed one-hop neighborhood-mean aggregation, whereas No Graph-M keeps GPR-M intact and applies no aggregation fallback.

## S4.2 Per-Fold Held-Out Results

Table S16 lists per-fold held-out test macro-F1 for all methods across three GEMs. Sample means and standard deviations over the five folds reproduce main-text Table 2.

## S4.3 Full Ablation Study (All Metrics)

Main-text Table 3 reports macro-F1 and AUC for the component ablation study. Table S17 extends this to all five metrics across all three GEMs.

## S4.4 AND-Branch Ablation of GPR-M

Table S18 compares three AND-node variants of the SoftBooleanAggregator on e coli core.

## S4.5 Graph-Encoding Comparison

Table S19 compares three Graph-M encoding variants on e coli core while holding GPR-M and Pred-M fixed.

## S5 Additional Case Studies

Case studies use a trained HGPRGDel model on e coli core restricted to target-dependent genes (n=66). They illustrate how GPR type shapes the continuous gene-retention ranking for three TCA-neighbor targets: succinate (succ c; main text), cytosolic α-ketoglutarate (akg c), and fumarate (fum c). The primary objects are ranking average precision (AP; positive class = curated reference label; $\mathrm { s c o r e } = 1 - \hat { y } _ { g , m } \big )$ and the ten lowest-retention genes; the main-text succinate figure additionally shows the ten highest-retention genes as a bidirectional contrast (panel B). Binary overlap after the global macro-F1 threshold, full-roster Jaccard, and a single-target weak-GC check of that one thresholded set [19, 8] are reported only as secondary operating-point descriptors (Section S3). The overlap precision and recall reported in this section are computed on the 66-gene target-dependent subset and are therefore not directly comparable to the full-roster recall and precision of main-text Table 4, which include pre-determined genes. As in Section S3.4 and Section S1.2, restricting to target-dependent genes removes the near-constant genes that are easiest to predict, so overlap metrics on this subset are systematically lower than full-roster ones; the two should not be read as competing estimates of the same quantity.

Table S16 Per-fold held-out test macro-F1 (%) on target-dependent gene–metabolite pairs for all methods. Each fold evaluates target metabolites excluded from training in that round (Section S1.4).
<table><tr><td>GEM</td><td>Method</td><td>Fold 1</td><td>Fold 2</td><td>Fold 3</td><td>Fold 4</td><td>Fold 5</td></tr><tr><td colspan="7"></td></tr><tr><td rowspan="5">e_coli_core</td><td>DNN</td><td>54.67</td><td>62.23</td><td>57.09</td><td>62.94</td><td>57.77</td></tr><tr><td>DeepGDel</td><td>65.27</td><td>73.43</td><td>68.22</td><td>74.28</td><td>69.60</td></tr><tr><td>NMA</td><td>67.03</td><td>74.19</td><td>68.71</td><td>75.67</td><td>68.95</td></tr><tr><td>GraphGDel</td><td>70.05</td><td>76.10</td><td>75.07</td><td>71.77</td><td>78.11</td></tr><tr><td>HGPRGDel</td><td>88.30</td><td>92.90</td><td>92.37</td><td>83.45</td><td>88.55</td></tr><tr><td colspan="8"></td></tr><tr><td rowspan="5">iMM904</td><td>DNN</td><td>63.48</td><td>67.01</td><td>64.54</td><td>67.23</td><td>64.89</td></tr><tr><td>DeepGDel</td><td>74.04</td><td>77.36</td><td>75.14</td><td>77.62</td><td>75.74</td></tr><tr><td>NMA</td><td>75.25</td><td>78.24</td><td>75.87</td><td>78.73</td><td>76.06</td></tr><tr><td>GraphGDel</td><td>79.71</td><td>82.45</td><td>81.81</td><td>80.61</td><td>83.12</td></tr><tr><td>HGPRGDel</td><td>88.22</td><td>87.08</td><td>88.58</td><td>88.24</td><td>90.95</td></tr><tr><td colspan="8"></td></tr><tr><td rowspan="5">iML1515</td><td>DNN</td><td>56.61</td><td>59.17</td><td>57.33</td><td>59.39</td><td>57.59</td></tr><tr><td>DeepGDel</td><td>61.03</td><td>63.96</td><td>61.92</td><td>64.31</td><td>62.48</td></tr><tr><td>NMA</td><td></td><td>64.86</td><td>62.69</td><td>65.38</td><td></td></tr><tr><td></td><td>62.27</td><td>72.71</td><td>72.22</td><td>71.31</td><td>62.90</td></tr><tr><td>GraphGDel HGPRGDel</td><td>70.65 89.66</td><td>90.91</td><td>89.69</td><td>91.34</td><td>73.25 90.46</td></tr></table>

Table S17 Full ablation study: all five metrics across three GEMs (mean ± s.d. over five held-out metabolite test folds on target-dependent pairs). Bold: best per GEM block.
<table><tr><td>GEM</td><td></td><td>Configuration Accuracy (%)</td><td> $M \mathrm { - P r e c . } \ ( \% )$ </td><td> $M \mathrm { - } \mathsf { R e c a l l \ ( \% ) }$ </td><td> $M a c r o \mathrm { - } \mathsf { F 1 } \left( \% \right)$ </td><td>AUC (%)</td></tr><tr><td rowspan="5">e_coli_core</td><td>Full</td><td> ${ \bf 8 9 . 3 4 \pm 3 . 7 6 }$ </td><td> ${ \bf 8 9 . 3 3 \pm 3 . 7 5 }$ </td><td> ${ \bf 8 9 . 2 7 \pm 3 . 8 2 }$ </td><td> $\mathbf { 8 9 . 1 1 \pm 3 . 8 1 }$ </td><td> ${ \bf 9 6 . 4 2 \pm 1 . 5 7 }$ </td></tr><tr><td>Mean Pooling</td><td> $7 9 . 2 4 \pm 3 . 2 9$ </td><td> $7 8 . 9 6 \pm 3 . 1 8$ </td><td> $7 9 . 5 1 \pm 3 . 0 1$ </td><td> $7 8 . 8 0 \pm 3 . 2 1$ </td><td> $8 8 . 1 0 \pm 3 . 2 4$ </td></tr><tr><td>No Graph-M</td><td> $7 8 . 3 1 \pm 3 . 2 6$ </td><td> $7 8 . 2 8 \pm 3 . 0 9$ </td><td> $7 7 . 2 8 \pm 3 . 0 3$ </td><td> $7 7 . 5 8 \pm 3 . 2 9$ </td><td> $8 6 . 5 5 \pm 3 . 3 2$ </td></tr><tr><td>No pretrained</td><td> $7 6 . 8 0 \pm 3 . 3 0$ </td><td> $7 6 . 5 0 \pm 3 . 2 0$ </td><td> $7 6 . 9 0 \pm 3 . 1 0$ </td><td> $7 6 . 5 0 \pm 3 . 3 0$ </td><td> $8 7 . 0 0 \pm 3 . 2 0 $ </td></tr><tr><td>No GPR-M</td><td> $7 6 . 6 8 \pm 3 . 2 8$ </td><td> $7 6 . 2 8 \pm 3 . 1 2$ </td><td> $7 6 . 4 3 \pm 3 . 0 5$ </td><td> $7 5 . 3 0 \pm 3 . 2 9$ </td><td> $8 4 . 6 0 \pm 3 . 3 4$ </td></tr><tr><td rowspan="5"></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Full</td><td> $\mathbf { 8 9 . 6 5 \pm 1 . 2 1 }$ </td><td> ${ \bf 8 8 . 6 4 \pm 1 . 2 7 }$ </td><td> ${ \bf 8 8 . 6 1 \pm 1 . 5 9 }$ </td><td> $\mathbf { 8 8 . 6 1 \pm 1 . 4 2 }$ </td><td> ${ \bf 9 6 . 2 2 \pm 1 . 1 2 }$ </td></tr><tr><td>Mean Pooling</td><td> $8 7 . 4 0 \pm 1 . 2 1$ </td><td> $8 6 . 0 0 \pm 1 . 1 2$ </td><td> $8 5 . 5 0 \pm 1 . 2 9$ </td><td> $8 5 . 3 0 \pm 1 . 1 9$ </td><td> $9 3 . 5 0 \pm 1 . 1 1$ </td></tr><tr><td>No Graph-M</td><td> $8 5 . 0 0 \pm 1 . 2 4$ </td><td> $8 3 . 8 0 \pm 1 . 2 1 $ </td><td> $8 3 . 5 0 \pm 1 . 3 0$ </td><td> $8 3 . 3 0 \pm 1 . 2 5$ </td><td> $9 2 . 1 0 \pm 1 . 1 4$ </td></tr><tr><td>No pretrained No GPR-M</td><td> $8 2 . 8 0 \pm 1 . 2 0 $ </td><td> $8 2 . 1 0 \pm 1 . 2 0$ </td><td> $8 1 . 9 0 \pm 1 . 3 0$ </td><td> $8 2 . 0 0 \pm 1 . 2 0 $ </td><td> $9 1 . 0 0 \pm 1 . 1 0$ </td></tr><tr><td rowspan="5">iML1515</td><td></td><td> $8 1 . 9 0 \pm 1 . 2 7$ </td><td> $8 1 . 5 0 \pm 1 . 2 3$ </td><td> $8 1 . 2 0 \pm 1 . 3 2 $ </td><td> $8 1 . 3 0 \pm 1 . 2 7$ </td><td> $8 9 . 5 0 \pm 1 . 1 5$ </td></tr><tr><td>Full</td><td> ${ \bf 9 0 . 4 3 \pm 0 . 7 1 }$ </td><td> ${ \bf 9 0 . 3 9 \pm 1 . 0 2 }$ </td><td> ${ \bf 9 0 . 4 3 \pm 0 . 9 5 }$ </td><td> ${ \bf 9 0 . 3 8 \pm 0 . 7 4 }$ </td><td> $\mathbf { 9 7 . 1 2 \pm 1 . 2 0 }$ </td></tr><tr><td>Mean Pooling</td><td> $8 8 . 0 0 \pm 0 . 8 2 $ </td><td> $8 7 . 2 0 \pm 1 . 0 5$ </td><td> $8 6 . 5 0 \pm 1 . 0 1$ </td><td> $8 6 . 4 0 \pm 0 . 8 2$ </td><td> $9 5 . 2 0 \pm 1 . 2 9$ </td></tr><tr><td>No Graph-M</td><td> $8 5 . 0 1 \pm 0 . 8 8$ </td><td> $8 4 . 7 7 \pm 1 . 0 8$ </td><td> $8 4 . 8 0 \pm 1 . 0 1 $ </td><td> $8 4 . 4 0 \pm 0 . 9 2$ </td><td> $9 1 . 5 0 \pm 1 . 3 6$ </td></tr><tr><td>No pretrained No GPR-M</td><td> $8 4 . 0 0 \pm 0 . 8 0$   $8 3 . 2 4 \pm 0 . 9 1$ </td><td> $8 3 . 4 0 \pm 1 . 0 0$   $8 2 . 6 0 \pm 1 . 1 3$ </td><td> $8 3 . 8 0 \pm 0 . 9 0$   $8 3 . 8 7 \pm 1 . 0 4$ </td><td> $8 3 . 3 5 \pm 0 . 8 0$   $8 2 . 4 0 \pm 0 . 9 6$ </td><td> $9 2 . 5 0 \pm 1 . 2 0$   $8 9 . 3 0 \pm 1 . 4 1 $ </td></tr></table>

## S5.1 Succinate Production

Succinate (succ c) is a dicarboxylic acid TCA intermediate with industrial relevance as a platform chemical and polymer precursor. Growth-coupled succinate production in e coli core requires blocking competing TCA exit routes while maintaining anaplerotic and

Table S18 AND-branch ablation of GPR-M on e coli core (mean ± s.d. over five held-out folds). “Mean-only” uses $\mathbf { z } _ { \mathrm { a n d } } { = } \mathbf { m } ;$ “Soft-minonly” uses $\mathbf { z } _ { \mathrm { a n d } } = \mathbf { s } ;$ “Combined” is the default. Bold: best.
<table><tr><td>AND-node configuration Macro-F1 (%)</td><td></td><td>AUC (%)</td></tr><tr><td>Combined (default)</td><td> $\mathbf { 8 9 . 1 1 \pm 3 . 8 1 }$ </td><td> ${ \bf 9 6 . 4 2 \pm 1 . 5 7 }$ </td></tr><tr><td>Mean-only</td><td> $6 8 . 6 4 \pm 4 . 0 8$ </td><td> $7 8 . 0 8 \pm 4 . 2 3$ </td></tr><tr><td>Soft-min-only</td><td></td><td></td></tr><tr><td></td><td> $6 9 . 3 8 \pm 2 . 2 2$ </td><td> $8 1 . 4 3 \pm 1 . 8 7$ </td></tr></table>

Table S19 Graph-encoding comparison on e coli core (macro-F1, mean ± s.d. over five folds). GPR-M and Pred-M are fixed. Hypergraph row matches main-text Table 3 Full.
<table><tr><td colspan="2"></td></tr><tr><td>Graph encoder</td><td>Macro-F1 (%)</td></tr><tr><td colspan="2"></td></tr><tr><td>Hypergraph (default) Bipartite MPNN</td><td> $\mathbf { 8 9 . 1 1 \pm 3 . 8 1 }$ </td></tr><tr><td></td><td> $7 9 . 4 4 \pm 2 . 7 7$ </td></tr><tr><td>Pairwise GCN</td><td> $7 5 . 9 9 \pm 2 . 9 1$ </td></tr></table>

Table S20 Ranking and threshold summary for three TCA-neighbor targets on e coli core (target-dependent genes, n=66). Top-10 hit: fraction of the ten lowest-retention calls labeled Deleted in the curated reference. Precision / Recall / F1: target-dependent-subset overlap after the global macro-F1 threshold. Jaccard: full-roster gene-set overlap versus the curated reference. All three targets have top-10 hit 10/10 (no false positives in the lowest-retention ten).
<table><tr><td>Target</td><td>AP</td><td>Top-10</td><td>Precision</td><td>Recall</td><td>F1</td><td></td><td>|Pred| Jaccard</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>0.71</td><td></td><td></td></tr><tr><td>succ_c</td><td>0.91</td><td>10/10</td><td>0.84</td><td>0.61 0.87</td><td></td><td>32 28</td><td>0.72 0.92</td></tr><tr><td>akg-c</td><td>0.96</td><td>10/10</td><td>0.96</td><td></td><td>0.92</td><td></td><td></td></tr><tr><td>fum_c</td><td>0.94</td><td>10/10</td><td>0.94</td><td>0.80</td><td>0.86</td><td>17</td><td>0.86</td></tr></table>

growth-supporting flux. Table S20 summarizes ranking and threshold metrics; Table S21 lists the ten lowest-retention calls (maintext case-study figure). We read the ordering as the deletion-preference signal, independent of whether any one thresholded set is flux-feasible.

Table S21 Top-10 lowest-retention HGPRGDel predicted deletions for succ c on e coli core (target-dependent genes). Retention score: model output $\hat { y } _ { g , m }$ (lower = stronger deletion preference). The curated reference label is 0 for deleted genes.
<table><tr><td>Gene</td><td>Retention score</td><td>Curated</td><td>Example reaction</td><td>GPR type</td></tr><tr><td></td><td></td><td>Deleted</td><td></td><td>OR</td></tr><tr><td>b1773</td><td>0.007</td><td></td><td>FBA</td><td></td></tr><tr><td>b1276</td><td>0.009</td><td>Deleted</td><td>ACONTa</td><td>OR</td></tr><tr><td>b0733</td><td>0.046</td><td>Deleted</td><td>CYTBD</td><td>AND</td></tr><tr><td>b0734</td><td>0.050</td><td>Deleted</td><td>CYTBD</td><td>AND</td></tr><tr><td>b2587</td><td>0.058</td><td>Deleted</td><td>AKGt2r</td><td>SINGLE</td></tr><tr><td>b3739</td><td>0.065</td><td>Deleted</td><td>ATPS4r</td><td>AND</td></tr><tr><td>b0727</td><td>0.111</td><td>Deleted</td><td>AKGDH</td><td>AND</td></tr><tr><td>b4151</td><td>0.113</td><td>Deleted</td><td>FRD7</td><td>AND</td></tr><tr><td>b1817</td><td>0.131</td><td>Deleted</td><td>FRUpts2</td><td>AND</td></tr><tr><td>b0809</td><td>0.137</td><td>Deleted</td><td>GLNabc</td><td>AND</td></tr></table>

Pathway interpretation. The highest-confidence deletions mix OR-type central-carbon isozyme calls (FBA, ACONTa) with AND type respiratory and TCA complex subunits (CYTBD, ATPS4r, AKGDH, FRD7) and a SINGLE transport call (AKGt2r), indicating that GPR-aware ranking does not collapse to a single reaction class. No false positives appear in the top-10 target-dependent genes.

## S5.2 α-Ketoglutarate Production

Biological background. Cytosolic α-ketoglutarate (akg c) is a TCA-cycle node and industrial platform precursor (amino-acid and biopolymer routes). Growth-coupled accumulation requires limiting downstream consumption of α-ketoglutarate while preserving anaplerosis and growth-supporting flux.

Ranking. Metrics are summarized in Table S20; Table S22 lists the ten lowest-retention calls. Under the Section S3 single-target FBA check, the default-threshold predicted deletion set is not weakly growth-coupled.

Table S22 Top-10 lowest-retention HGPRGDel predicted deletions for akg c on e coli core (target-dependent genes).
<table><tr><td>Gene</td><td>Retention score</td><td>Curated</td><td>Example reaction</td><td>GPR type</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>b3952</td><td>0.000</td><td>Deleted</td><td>PFL</td><td>AND</td></tr><tr><td>b1478</td><td>0.000</td><td>Deleted</td><td>ALCD2x</td><td>OR</td></tr><tr><td>b0727</td><td>0.001</td><td>Deleted</td><td>AKGDH</td><td>AND</td></tr><tr><td>b0811</td><td>0.002</td><td>Deleted</td><td>GLNabc</td><td>AND</td></tr><tr><td>b3739</td><td>0.004</td><td>Deleted</td><td>ATPS4r</td><td>AND</td></tr><tr><td>b0724</td><td>0.011</td><td>Deleted</td><td>SUCDi</td><td>AND</td></tr><tr><td>b0903</td><td>0.013</td><td>Deleted</td><td>PFL</td><td>AND</td></tr><tr><td>b2463</td><td>0.019</td><td>Deleted</td><td>ME2</td><td>SINGLE</td></tr><tr><td>b2925</td><td>0.024</td><td>Deleted</td><td>FBA</td><td>OR</td></tr><tr><td>b0729</td><td>0.026</td><td>Deleted</td><td>SUCOAS</td><td>AND</td></tr></table>

Pathway interpretation. The lowest-retention calls elevate pyruvate-formate lyase (PFL/b3952), a fermentation OR isozyme (ALCD2x/b1478), and the immediate downstream TCA consumer AKGDH/b0727, followed by energy and TCA AND blocks (ATPS4r, SUCDi, SUCOAS) with a SINGLE anaplerotic call (ME2). Relative to succinate, cytosolic α-ketoglutarate elevates PFL and fermentation OR calls ahead of several TCA consumers. In the succinate ranking, FBA/ACONTa and respiratory CYTBD occupy the earliest ranks. No false positives appear in the top-10 target-dependent genes.

Contrast with succinate and extracellular α-ketoglutarate. The shared core network therefore yields overlapping but reordered deletion priorities across the two cytosolic TCA targets. Extracellular akg e and cytosolic akg c yield the same top-10 target-dependent gene set in the same order; the akg c list is shown in Table S22 and is not duplicated. Under the Section S3 protocol, neither thresholded predicted set is weakly growth-coupled, whereas both curated reference sets are, illustrating that ranking can transfer across related targets while binary FBA feasibility remains set-specific.

## S5.3 Fumarate Production

Biological background. Fumarate (fum c) is a TCA-cycle intermediate adjacent to succinate and provides a third ranking case on the same target-dependent-gene roster.

Ranking. Metrics are summarized in Table S20; Table S23 lists the ten lowest-retention calls. The fumarate column of the main-text case-study heatmap (panel C) uses the same retention scores.

Pathway interpretation. The earliest ranks concentrate on respiratory and ATP-synthase AND complexes (CYTBD, multiple ATPS4r subunits) together with aconitase OR (ACONTa) and downstream TCA AND calls (AKGDH, PFL, THD2). Relative to succinate (where FBA leads) and cytosolic α-ketoglutarate (where PFL/ALCD2x lead), fumarate elevates oxidative-phosphorylation subunits, reinforcing that the shared core network yields target-dependent deletion orderings rather than a single fixed gene list. No false positives appear in the top-10 target-dependent genes.

<table><tr><td rowspan=1 colspan=1>Gene    Retention score   Curated   Example reaction   GPR typeb0733   0.007                    Deleted   CYTBD                      AND</td></tr><tr><td rowspan=1 colspan=1>b1276   0.008                   Deleted   ACONTa                    OR</td></tr><tr><td rowspan=1 colspan=1>b3739   0.089                    Deleted   ATPS4r                    AND</td></tr><tr><td rowspan=1 colspan=1>b3731   0.093                   Deleted   ATPS4r                    AND</td></tr><tr><td rowspan=1 colspan=1>b0734   0.097                    Deleted    CYTBD                      AND</td></tr><tr><td rowspan=1 colspan=1>b0727   0.104                   Deleted   AKGDH                     AND</td></tr><tr><td rowspan=1 colspan=1>b3734   0.115                    Deleted   ATPS4r                    AND</td></tr><tr><td rowspan=1 colspan=1>b3952   0.175                    Deleted   PFL                         AND</td></tr><tr><td rowspan=1 colspan=1>b1602   0.210                    Deleted   THD2                       AND</td></tr><tr><td rowspan=1 colspan=1>b0978   0.237                    Deleted   CYTBD                      AND</td></tr></table>

## S6 Complexity Analysis

## S6.1 GPR Parser: Scope and Algorithmic Complexity

The HGPRGDel GPRRuleParser accepts COBRA-style infix rules with and/or, parentheses, and whitespace-separated gene identifiers. Negation (not) and nonstandard dialects are not supported. Empty rules yield a zero reaction embedding; missing leaf genes are mapped to a zero vector.

Parsing a GPR rule of length L (character count) and tree depth D requires O(L) time and O(D) stack space. For all three evaluated GEMs, maximum parse-tree depth is 4 (Table S4) and mean length is well below 200 characters; parsing overhead is negligible relative to forward-pass computation.

Forward-pass complexity of GPR-M for a single reaction with K genes in its rule is $O ( K \cdot d _ { g } )$ for embedding aggregation plus $O ( K \cdot d _ { g } ^ { 2 } )$ for the MHA computation in OR nodes (gene embedding dimension $d _ { g } )$ . Because the OR query is a single pooled vector (Section S7.1), the attention score matrix is 1 × K rather than $K \times K \colon$ the query, key, value, and output projections contribute $O ( K \cdot d _ { g } ^ { 2 } )$ , while forming and applying the attention weights contributes only $O ( K \cdot d _ { g } )$ . Across all reactions in a GEM, total GPR-M forward-pass complexity is $O ( | R | \cdot \bar { K } \cdot d _ { g } + | R | \cdot \bar { K } \cdot d _ { g } ^ { 2 } )$ , where K<sup>¯</sup> is the mean number of genes per rule. For iML1515 (|R|=2712, K<sup>¯</sup> ≈ 2.0, d =320), this is fully dominated by the MHA projection term.

## S6.2 ESM-2 Embedding Coverage

All genes referenced in GPR rules across the three evaluated GEMs have corresponding ESM-2 embeddings (Table S24). The zerovector substitution mechanism (for genes without a sequence record) was therefore not invoked for any reaction in the experimenta datasets; every GPR tree leaf receives a non-zero gene embedding during both training and evaluation.

Table S24 ESM-2 embedding coverage for genes referenced in GPR rules. “Reactions with complete GPR coverage” counts reactions whose every gene has an ESM-2 embedding.
<table><tr><td>GEM</td><td>GPR-referenced genes</td><td>Genes with ESM-2</td><td>Gene coverage</td><td>Reactions (complete GPR)</td></tr><tr><td>e_coli_core</td><td>137</td><td>137</td><td>100%</td><td>69 / 69 (100%)</td></tr><tr><td>iMM904</td><td>905</td><td>905</td><td>100%</td><td>1,043 / 1,043 (100%)</td></tr><tr><td>iML1515</td><td>1,516</td><td>1,516</td><td>100%</td><td>2,266 /2,266 (100%)</td></tr></table>

## S6.3 Currency Metabolite Filtering

Following GraphGDel [12], we exclude high-flux cofactors from compound–reaction hyperedges when building the incidence matrix H, unless the metabolite is a supervised production target. Matching is case-insensitive on the BiGG/KEGG base identifier (compartment sufixes such as c are stripped); Table S25 lists the excluded cofactors and their per-GEM efect.

Table S25 Currency metabolites excluded from hypergraph incidence construction (unless a supervised target). The 20 filtered base metabolites are atp, adp, amp, nad, nadh, nadp, nadph, coa, h, h2o, pi, ppi, co2, nh4, o2, glutathione, na1, k, mg2, and ca2. Efect on each GEM: incidence kept / total.

<table><tr><td>GEM</td><td>Unique metabolites</td><td>Currency matched</td><td>Non-currency</td><td>Incidence kept / total</td></tr><tr><td>e_coli_core</td><td>72</td><td>20</td><td>52</td><td>175/360</td></tr><tr><td>iMM904</td><td>1226</td><td>88</td><td>1138</td><td>3,749 /6,517</td></tr><tr><td>iML1515</td><td>1877</td><td>41</td><td>1836</td><td>6,333 /10,575</td></tr></table>

## S7 Mathematical Details of Diferentiable GPR

This section provides the complete mathematical formulation of the SoftBooleanAggregator, including gradient expressions and numerical implementation. The goal is to make the diferentiable GPR construction self-contained without relying on the biologica motivation in the main text.

## S7.1 Soft Boolean Aggregation

Let $\left\{ g _ { 1 } , \ldots , g _ { K } \right\}$ be the genes in a GPR rule, with projected embeddings $\mathbf { u } _ { i } = W _ { \mathrm { p r o j } } \mathbf { e } _ { g _ { i } } \in \mathbb { R } ^ { d }$ (dimension $d = d _ { g } = 3 2 0 )$

## AND Node (Enzyme Complexes)

An AND node encodes a set of obligate subunits: all K genes must be present for the reaction to be catalytically active. The aggregation combines a mean branch and a soft-min branch.

Mean branch.

$$
\mathbf { m } = \frac { 1 } { K } \sum _ { i = 1 } ^ { K } \mathbf { u } _ { i }\tag{S3}
$$

The mean captures the average embedding of all subunits, summarizing overall complex capacity.

Soft-min branch. Applied independently to each embedding dimension $j \in \{ 1 , \ldots , d \}$

$$
s _ { j } = - \log \sum _ { i = 1 } ^ { K } \exp \biggl ( - \frac { ( u _ { i } ) _ { j } } { \tau } \biggr )\tag{S4}
$$

where $\tau = \mathrm { s o f t p l u s } ( \hat { \tau } ) + \epsilon > 0$ is the efective soft-AND temperature derived from the learnable scalar ˆτ (Section S7.3). This is a temperature-scaled log-sum-exp surrogate for min $\romannumeral 3$ over subunit j-coordinates; the aggregator consumes s directly, and $\tau s _ { j }$ itself is the soft minimum. Shifting by mi $\mathbf { \Psi } _ { 1 _ { i } } ( u _ { i } ) _ { j }$ (Section S7.3) shows that every shifted exponent is non-positive, so the resulting sum lies in $[ 1 , K ]$ and $s _ { j }$ − min ${ \bf \Gamma } ( u _ { i } ) _ { j } / \tau \in [ -$ − log $K , 0 ]$ . Hence $\tau s _ { j } $ min $\vdots ( u _ { i } ) _ { j }$ as $\tau  0$ (hard minimum, exact AND bottleneck). This global $1 / \tau$ factor is absorbed by the learned MLP $\phi _ { \wedge }$ that consumes s, leaving the AND bottleneck semantics unchanged.

The dimension-wise vector $\mathbf { s } = ( s _ { 1 } , \ldots , s _ { d } )$ encodes the bottleneck constraint: the dimension in which the limiting subunit has the smallest embedding value dominates.

AND output.

$$
\mathbf { z } _ { \mathrm { a n d } } = \phi _ { \wedge } ( [ \mathbf { m } \parallel \mathbf { s } ] )\tag{S5}
$$

where $[ \cdot \| \cdot ]$ denotes concatenation and $\phi _ { \Lambda } : \mathbb { R } ^ { 2 d }  \mathbb { R } ^ { d }$ is a two-layer MLP with ReLU activation (hidden width 256).

## OR Node (Isozyme Alternatives)

An OR node encodes a set of isozymes: any one of the K alternative genes is suficient for catalytic activity.

Multi-head attention (MHA) branch. The MHA mechanism computes a context-aware weighted combination of all isozyme embeddings:

$$
\mathbf { a } = \mathrm { M H A } ( \mathbf { q } , \mathbf { U } , \mathbf { U } ) \in \mathbb { R } ^ { d } , \qquad \mathbf { q } = \frac { 1 } { K } \sum _ { i = 1 } ^ { K } \mathbf { u } _ { i } , \qquad \mathbf { U } = [ \mathbf { u } _ { 1 } ^ { \top } ; \cdots ; \mathbf { u } _ { K } ^ { \top } ]\tag{S6}
$$

with $H { = } 4$ attention heads (main text). The query is a single pooled vector rather than the full set of isozyme embeddings, so this is single-query cross-attention over K keys and values. Attention weights are data-dependent, allowing the aggregator to up-weight the isozyme most relevant to the deletion context.

Log-sum-exp branch (soft-max). Applied dimension-wise:

$$
\ell _ { j } = \log \sum _ { i = 1 } ^ { K } \exp ( ( u _ { i } ) _ { j } )\tag{S7}
$$

The OR branch carries no temperature parameter. Shifting by $\operatorname* { m a x } _ { i } ( u _ { i } ) _ { j }$ gives $\ell _ { j } - \operatorname* { m a x } _ { i } ( u _ { i } ) _ { j } \in [ 0 _ { : }$ , log $K ] _ { \mathrm { { ; } } }$ , so $\ell _ { j }$ is a smooth upper bound on the strongest isozyme coordinate, tight to within log K. The OR semantics, whereby any suficient alternative enables the reaction, are encoded by focusing representation on the strongest available isozyme.

OR output.

$$
\mathbf { z } _ { \mathrm { o r } } = \phi _ { \vee } ( \left[ \mathbf { a } \right] | \ell ] )\tag{S8}
$$

where $\phi _ { \vee } : \mathbb { R } ^ { 2 d }  \mathbb { R } ^ { d }$ is a two-layer MLP with ReLU activation (hidden width 256).

Every node of the parse tree therefore maps $\mathbb { R } ^ { d } \to \mathbb { R } ^ { d }$ with $d = d _ { g } = 3 2 0$ , so the reaction embedding produced by GPR-M has dimension $d _ { g } ,$ not $D _ { \mathrm { r x n } } .$ The reduction to $D _ { \mathrm { r x n } } = 1 2 8$ is performed downstream: Pred-M pools $\mathbf { Z } _ { \mathrm { r x n } } ^ { ( r ) }$ over $R _ { g \cup m }$ and applies the association $\mathrm { M L P } \ \mathbb { R } ^ { d _ { g } }  \mathbb { R } ^ { D _ { \mathrm { r x n } } }$ , so that the classifier consumes $d _ { g } + d _ { m } + D _ { \mathrm { r x n } } = 7 0 4$ dimensions (Table S5).

## Single-Gene and Empty Rules

A single-gene rule (K=1) returns u directly (no aggregation needed). An empty GPR string (no associated gene) returns $\mathbf { 0 } \in \mathbb { R } ^ { d }$

## S7.2 Gradient Propagation

## Soft-Min Gradient (AND Node)

The gradient of $s _ { j }$ with respect to $( u _ { k } ) _ { j }$ is:

$$
\frac { \partial s _ { j } } { \partial ( u _ { k } ) _ { j } } = \frac { 1 } { \tau } \cdot \frac { \mathrm { e x p } \left( - \frac { ( u _ { k } ) _ { j } } { \tau } \right) } { \sum _ { i = 1 } ^ { K } \mathrm { e x p } \left( - \frac { ( u _ { i } ) _ { j } } { \tau } \right) } = \frac { 1 } { \tau } \mathrm { s o f t m a x } \Big ( - \frac { \mathbf { u } _ { ( \cdot ) j } } { \tau } \Big ) _ { k }\tag{S9}
$$

This is the softmax of the negated inputs: the gene with the smallest embedding value in dimension j receives the largest gradient weight. Consequently, during training, the loss signal flows predominantly through the subunit with the weakest representation, the bottleneck gene of the enzyme complex. This gradient routing encourages the model to accurately represent the limiting subunit, which determines deletion feasibility for AND-typed reactions.

$$
L o g - S u m  – E x p \ G r a d i e n t \ ( O R \ N o d e )
$$

The gradient of $\ell _ { j }$ with respect to $( u _ { k } ) _ { j }$ is:

$$
\frac { \partial \ell _ { j } } { \partial ( u _ { k } ) _ { j } } = \frac { \exp ( ( u _ { k } ) _ { j } ) } { \sum _ { i = 1 } ^ { K } \exp ( ( u _ { i } ) _ { j } ) } = \mathrm { s o f t m a x } \big ( \mathbf { u } _ { ( \cdot ) j } \big ) _ { k }\tag{S10}
$$

The gene with the largest embedding value receives the largest gradient weight. Training therefore concentrates representation quality on the most capable isozyme, consistent with OR semantics: the reaction is blocked only if all alternatives are deleted, making the strongest isozyme the critical blocking target.

## S7.3 Numerical Stability

Direct computation of $\exp ( - u / \tau )$ for small τ or large |u| can overflow or underflow in single-precision arithmetic. We apply the standard log-sum-exp identity:

$$
\log \sum _ { i = 1 } ^ { K } \exp ( x _ { i } ) = c + \log \sum _ { i = 1 } ^ { K } \exp ( x _ { i } - c ) , \quad c = \operatorname* { m a x } x _ { i }\tag{S11}
$$

For the soft-min branch, set $x _ { i } = - ( u _ { i } ) _ { j } / \tau ,$ so $c = - \operatorname* { m i n } _ { i } ( u _ { i } ) _ { j } / \tau$ :

$$
s _ { j } = \frac { \operatorname* { m i n } _ { i } ( u _ { i } ) _ { j } } { \tau } - \log \sum _ { i = 1 } ^ { K } \exp \left( \frac { \operatorname* { m i n } _ { i } ( u _ { i } ) _ { j } - ( u _ { i } ) _ { j } } { \tau } \right)\tag{S12}
$$

All exponent arguments are non-positive after shifting by the maximum, preventing overflow. The analogous shift applies to the log-sum-exp branch of OR nodes.

Temperature τ is parameterized as:

$$
\tau = \mathrm { s o f t p l u s } ( \hat { \tau } ) + \epsilon , \qquad \epsilon = 1 0 ^ { - 6 }\tag{S13}
$$

where ˆτ is the unconstrained learnable scalar and ${ \mathrm { s o f t p l u s } } ( x ) = \log ( 1 + e ^ { x } )$ . This guarantees $\tau > \epsilon > 0$ throughout training and prevents division by zero in the gradient computation. The initial value of ˆτ is set so that $\tau \approx 1$ at the start of training.

## S7.4 Empirical Fidelity of Soft Boolean Approximations

SoftBoolean fidelity is reported in the main-text mechanism figure (Panels A–B); OR-branch attention entropy appears in Fig. S4B. All numbers below come from the per-fold model states of the single protocol, under the graph convention of Section S2.6.1. For completeness, the per-GEM Spearman correlations (soft vs. hard leaf $L _ { 2 }$ scalars, i.e., the $L _ { 2 }$ norm of each leaf embedding after soft vs. hard aggregation) are: AND soft-min vs. hard-min $r _ { s } = 1 . 0 0 0$ , 0.987, 0.857 on e coli core (n=10), iMM904 (n=86), and iML1515 (n=221); OR log-sum-exp vs. hard-max $r _ { s } = 0 . 9 9 2$ , 0.998, 0.924 (n=32, 265, 743). The AND correlation is lower on iML1515 than on the two smaller GEMs, which we report as observed rather than attribute to a mechanism: the AND stratum there is the largest of the three (n=221 versus 10 and 86) and the soft-min gate compresses leaf scalars whose hard-min values span the widest range. OR-branch attention entropy $H _ { \mathrm { a t t } }$ on $i M L 1 5 . 1 5$ increases with isozyme group count K but remains below ln K for most reactions, indicating non-uniform isozyme weighting rather than degenerate uniform attention. Entropy below ln K establishes only that the weighting is non-uniform; it does not by itself demonstrate that the attention branch is functionally necessary. All evaluations use frozen leaf embeddings at the final trained checkpoint of each GEM.

## S8 Biological Interpretation

## S8.1 Reaction Representation Analysis

Reaction embeddings $\mathbf { Z } _ { \mathrm { r x n } } ^ { ( r ) }$ were extracted from trained HGPRGDel models using the final reaction representations produced by GPR-M. To avoid label leakage, embeddings were computed via fold-agnostic full-GEM forward passes with frozen ESM-2 and ChemBERTa encoders. The main-text mechanism figure (Fig. 4) provides six panels of quantitative mechanistic evidence: SoftBoolean AND fidelity (Panel $\mathrm { A } ) { \mathrm { ; } }$ OR fidelity (Panel $_ \mathrm { B } ) ;$ compact embedding diagnostics: linear probe AND/OR accuracy and nearest neighbor subsystem enrichment (Panel $\mathrm { C } ) { \mathrm { ; } }$ deletion F1 stratified by exclusive gene–GPR role (Panel D); GPR-rule representation separation via per-reaction silhouette (Panel E); and the association between the continuous deletion-priority score and reference-set membership, stratified by GPR role (Panel F). Panel D deletion F1 (deletion as the positive class; MIXED-rule genes excluded) is 88.2/87.4/88.9% (AND/OR/SINGLE) on e coli core, 96.2/86.8/92.5% on $i M M 9 0 4 ,$ and 81.1/88.8/79.6% on $i M L 1 5 . 1 5 ;$ stratum sizes are $n _ { \mathrm { g e n e s } } { = } 3 3 / 2 8 / 4 , 1 0 / 1 2 / 2 7 .$ , and $1 7 / 5 / 1 1$ , respectively. Figure S4 provides additional supporting evidence: Panel A shows a UMAP of iML1515 $( | R | { = } 2 , 7 1 2 )$ reaction embeddings colored by top-level GPR type, illustrating partial co-localization; Panel B shows OR-branch attention entropy $\left( H _ { \mathrm { a t t } } \right)$ vs. isozyme group size K, confirming non-uniform attention weighting. The linear probe underlying main-text Panel C is restricted to AND- and OR-type reactions (i.e., multi-gene rules); single-gene and empty reactions are excluded because their GPR type is trivially recoverable from reaction degree, not from embedding geometry. Embedding geometry was assessed using nearest-neighbor enrichment analysis (k=10) against 1,000 randomly generated baselines (Table S26). Two-dimensional visualizations were generated using UMAP $( n _ { \mathrm { n e i g h b o r s } } { = } 1 5 $ , min dist=0.1, seed 42).

Table S26 Reaction embedding organization tests. NN same-subsystem fraction compares observed neighbor purity to label-permutation baselines (k=10, 1,000 shufles of subsystem labels) in $\mathbf { Z } _ { \mathrm { r } \times \mathrm { n } }$ space; z-scores indicate standard deviations above the shufled mean. NMI uses KMeans with one cluster per annotated subsystem (or per GPR type); NMI is reported descriptively, with no random baseline computed, and is not used as standalone evidence of embedding quality.
<table><tr><td>GEM</td><td>|R|</td><td>NN same-subsystem</td><td>Random NN</td><td>z</td><td>NMI (subsystem)</td><td>NMI (GPR type)</td></tr><tr><td>e_coli_core</td><td>95</td><td>0.225</td><td> $0 . 1 2 3 \pm 0 . 0 1 3$ </td><td>7.7</td><td>0.404</td><td>0.471</td></tr><tr><td>iMM904</td><td>1577</td><td>0.196</td><td> $0 . 0 4 2 \pm 0 . 0 0 4$ </td><td>36.5</td><td>0.357</td><td>0.411</td></tr><tr><td>iML1515</td><td>2712</td><td>0.411</td><td> $0 . 0 7 6 \pm 0 . 0 0 3$ </td><td>104.8</td><td>0.287</td><td>0.504</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Silhouette scores (cosine) are −0.22 (e coli core), −0.31 (iMM904 ), and −0.25 (iML1515). We report these scores for completeness, but do not use them as evidence of embedding quality: metabolic subsystems share intermediates and are not expected to form compact, disjoint clusters in embedding space, and the silhouette score penalizes such overlap regardless of whether it is biologically meaningful. Nearest-neighbor enrichment and the linear probe are the primary quantitative evidence for embedding organization; they measure local neighborhood structure rather than global cluster compactness. Nearest-neighbor subsystem enrichment is consistently above random across all three GEMs (z=7.7, 36.5, and 104.8 for e coli core, iMM904, and $i M L 1 5 . 1 5 ,$ respectively), with the strongest enrichment on iML1515.

Figure S5 shows a qualitative UMAP of iML1515 reaction embeddings colored by four coarse metabolic categories. Global color mixing is expected because pathways share intermediates and is not used as quantitative evidence; organization is reported in the main-text mechanism figure (Panel C diagnostics).

![](images/fe024310a3324cd44afca3fee267ef655cd8affc6cde62c9af26f2d3883359e2.jpg)

![](images/89347e8cf5b3493a44c5cf2ce2ace2dc7eb744589e084d07776d382963a04be3.jpg)  
Figure S4 Supporting mechanism evidence. (A) UMAP of iML1515 reaction embeddings (|R|=2,712) colored by top-level GPR type (AND, OR, Single-gene, Empty). Partial co-localization indicates that the learned representation encodes GPR type without explicit supervision; quantitative assessment is reported in main-text Fig. 4C,D. (B) OR-branch attention entropy $H _ { \mathrm { a t t } }$ vs. isozyme group size K (iML1515); solid line = ln K (maximum entropy). Most OR reactions fall below ln K, indicating non-uniform isozyme weighting rather than degenerate uniform attention.

![](images/41cec27212dfe23ef6f1db4e4a62eb6a0f12e88d46a7f404b637138bec3b36a0.jpg)  
Figure S5 UMAP of GPR-M reaction embeddings on iML1515, colored by four coarse metabolic categories (transport, amino-acid metabolism, energy metabolism, cofactor metabolism). Reactions outside these categories or without a coarse mapping are shown in grey (n shown in figure). Global color mixing is expected because pathways share intermediates and is not used as quantitative evidence.