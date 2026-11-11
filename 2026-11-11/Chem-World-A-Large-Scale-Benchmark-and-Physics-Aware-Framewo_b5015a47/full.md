# Chem World: A Large-Scale Benchmark and Physics-Aware Framework for Trustworthy Molecular Property Prediction

Tianyou Bai<sup>1,2,3</sup>, Huan Wang<sup>3</sup>, Mingchen Gao<sup>1</sup>, Tongrui Yang<sup>1</sup>, Fangyue Lin<sup>3,4</sup>, Pinze Ren<sup>3,5</sup>, Zhenlin Zhao<sup>3</sup>, Siming Dong <sup>3</sup>,

<sup>1</sup>Institute of Automation, Chinese Academy of Sciences,

<sup>2</sup>School of Advanced Interdisciplinary Sciences, University of Chinese Academy of Sciences, <sup>3</sup>Cleer Science, <sup>4</sup>Imperial College London, <sup>5</sup>Tsinghua University {baitianyou2024,gaomingchen2025,yangtongrui2025}@ia.ac.cn,william.d@cleerlab.com

## Abstract

Molecular property prediction is fundamental to computational chemistry and drug formulation research, yet existing benchmarks remain fragmented across datasets, properties, and evaluation protocols, limiting systematic evaluation of model generalization. We introduce Chem World, a large-scale benchmark integrating 17 public datasets with over 800,000 molecular and chemical-system instances across 10 property tracks, covering single-molecule and multi-component systems, including drug solubility. Chem World provides a unified testbed for evaluating molecular representations and compositional modeling across diverse chemical properties. Building on the benchmark, we propose PACM (Physics-Aware Compositional Modeling), which combines molecular representations with composition-aware modeling and domain-specific physical constraints. Extensive experiments show that molecular language models provide strong and transferable representations, while PACM consistently improves prediction across diverse chemical systems. Chem World and our PACM-powered drug-property prediction Agent support generalizable chemical modeling and offer a foundation for AI-assisted biomedical research and drug formulation development.

## 1 Introduction

Accurate molecular property prediction is fundamental to materials discovery, chemical engineering, and pharmaceutical development. Recent advances in graph neural networks (GNNs), Transformer-based molecular models, and largescale molecular representation learning have substantially improved data-driven prediction by learning informative representations from molecular structures and chemical sequences. However, existing studies are often evaluated on individual datasets or narrowly defined tasks, making it difficult to determine whether learned molecular representations generalize across diverse properties and chemical systems. This challenge becomes particularly pronounced when moving beyond individual molecules to multi-component systems, where properties can emerge from compositional and intermolecular interactions.

![](images/10c40081eacb7ceb347c3ff0aa262a3dfb1d803f6cd3ce5299d53bade1db15a8.jpg)  
Figure 1: Pipeline of Chem World and PACM. Chem World provides standardized multi-source chemical systems, while PACM integrates physical constraints to improve chemical system modeling.

A major obstacle is the lack of comprehensive and standardized benchmarks. Existing resources (Rajaonson et al., 2026; Kavian et al., 2026) typically focus on specific properties, molecular systems, or domains, resulting in fragmented datasets and inconsistent evaluation protocols. Consequently, it remains unclear how different molecular representations, including molecular language models and graph-based models, perform across heterogeneous property prediction and whether their representations transfer effectively to unseen chemical systems. A large benchmark that unifies diverse datasets and provides consistent evaluation is therefore essential for systematically studying the generalization of molecular representations.

Beyond benchmarking, an important modeling challenge is how to compose molecular representations into predictions for complex chemical systems. Directly aggregating component-level representations may fail to capture compositiondependent interactions and can produce physically implausible predictions. This motivates models that combine expressive molecular representations with compositional modeling and domain-specific constraints, enabling knowledge learned at the molecular level to generalize to multi-component chemical systems.

To address these challenges, we introduce Chem World, a large-scale benchmark integrating 17 public datasets (Sorkun et al., 2019; Krasnov et al., 2025; de Blasio et al., 2024; Odegova et al., 2024; Bao et al., 2024; Kumar et al., 2025; Fajar et al., 2024; Kuzhagaliyeva et al., 2022; Chew et al., 2025; Malikov et al., 2026; Zhang et al., 2023; Kazakov et al., 2012; Tom et al., 2025; Bradford et al., 2023) with over 800,000 molecular and chemical-system instances across 10 molecular property tracks. As illustrated in Fig. 1, Chem World covers diverse properties, including density, conductivity, solubility, viscosity, and thermodynamic properties, and unifies both singlemolecule and multi-component systems under standardized evaluation protocols. Building on this benchmark, we propose PACM (Physics-Aware Compositional Modeling), a compositional modeling framework that combines molecular representations with composition-aware interaction modeling and domain-specific physical constraints. Rather than relying solely on statistical correlations, PACM provides a principled mechanism for extending molecular-level representations to complex chemical systems while maintaining physically meaningful predictions.

Extensive experiments across Chem World systematically compare molecular encoders and compositional modeling strategies, demonstrating the effectiveness of molecular language representations and physics-aware compositional modeling across diverse molecular property prediction tasks. Together, Chem World and PACM provide a unified framework for studying generalizable molecular representations and compositional chemical modeling.

The main contributions of this work are summarized as follows:

• We introduce Chem World, a large-scale benchmark integrating 17 public datasets with over 800K molecular and chemical-system instances across 10 molecular property tracks, enabling standardized evaluation across diverse chemical systems and properties.

• We propose PACM, a compositional modeling framework that combines molecular representations, interaction-aware aggregation, and domain-specific physical constraints to extend molecular-level representations to complex chemical systems.

• We conduct systematic experiments across Chem World to evaluate molecular representations and compositional modeling strategies, providing insights into their generalization across diverse molecular properties and chemical systems.

## 2 Related Work

Chemical property prediction benchmarks. Existing molecular property benchmarks provide foundations for chemical representation learning, but often focus on individual properties, specific molecular tasks, or particular chemical domains. QM9 (Ramakrishnan et al., 2014) contains about 130K molecules with 19 quantum-chemical properties, while MoleculeNet (Wu et al., 2018) covers solubility, lipophilicity, and toxicity. Other resources target protein–ligand interactions (Wang et al., 2004), inorganic materials (Jain et al., 2013), or molecular graph learning (Hu et al., 2020). For chemical mixtures, NIST (Kazakov et al., 2012), DETHERM (Westhaus et al., 1999), and the Dortmund Data Bank (Onken et al., 1989) provide thermophysical measurements, while CheMixHub (Rajaonson et al., 2026) expands mixture data availability. However, differences in property coverage, system diversity, data organization, and evaluation protocols hinder systematic cross-task comparisons. As shown in Table 1, Chem World addresses these limitations by integrating 17 public datasets with over 800K molecular and chemicalsystem instances across 10 property tracks, covering single-molecule and multi-component systems under unified evaluation protocols. Beyond scale, it connects molecular property prediction with compositional chemical modeling to investigate representation generalization across properties and from individual molecules to complex chemical systems.

Molecular representation learning for property prediction. Deep learning methods represent molecules using graphs, three-dimensional structures, and chemical strings. Geometric and graphbased models include SchNet (Schütt et al., 2017), DimeNet (Gasteiger et al., 2020b), PaiNN (Schütt et al., 2021), GemNet (Gasteiger et al., 2020a), NequIP (Batzner et al., 2022), and Graphormer (Ying et al., 2021), which capture molecular connectivity, geometric interactions, or structural symmetries. Chemical language models such as Chem-BERTa (Chithrananda et al., 2020), MolFormer, and MolT5 learn contextual representations from molecular sequences. For multi-component systems, prior work explores formulation graphs (Sharma et al., 2023), chemistry-aware GNNs (Zhang et al., 2024), and differentiable electrolyte modeling (Zhu et al., 2024). However, molecular encoders and mixture-specific models are often evaluated separately, leaving their transferability to compositional prediction insufficiently understood. Strong performance on individual tasks also does not establish generalization across diverse properties and chemical systems. Chem World offers a unified setting to compare molecular encoders and compositional modeling strategies, systematically evaluating representation transfer across heterogeneous property prediction tasks.

<table><tr><td>Benchmark</td><td>Scale</td><td>Property Coverage</td><td>Single-Molecule Tasks</td><td>Multi-Component Systems</td><td>Unified Protocol</td><td>OOD Evaluation</td></tr><tr><td>QM9</td><td>~130K</td><td>Quantum chemical</td><td>Yes</td><td>No</td><td>Task-specific</td><td>Limited</td></tr><tr><td>MoleculeNet</td><td>Multiple datasets</td><td>Multiple</td><td>Yes</td><td>Limited</td><td>Suite-level</td><td>Task-dependent</td></tr><tr><td>CheMixHub</td><td>~500K</td><td>Multiple</td><td>Limited</td><td>Yes</td><td>Benchmark-specific</td><td>Yes</td></tr><tr><td>MixtureSolDB</td><td>175K+</td><td>Solubility</td><td>Limited</td><td>Yes</td><td>Dataset-specific</td><td>No</td></tr><tr><td>MolSets</td><td>~1K</td><td>Multiple</td><td>Set-based</td><td>Yes</td><td>Dataset-specific</td><td>No</td></tr><tr><td>Chem World</td><td>800K+</td><td>10 tracks</td><td>Yes</td><td>Yes</td><td>Unified schema</td><td>Multiple settings</td></tr></table>

Table 1: Comparison of representative molecular and chemical mixture benchmarks.

Physics-aware learning for chemical property prediction. Physics-informed learning incorporates domain knowledge through governing equations, physical constraints, conservation laws, symmetries, or physics-derived features. Classical PINNs impose governing equations through residual, boundary, and initial-condition losses (Raissi et al., 2019), while chemistry-oriented methods such as sGDML (Chmiela et al., 2018), PhysNet (Unke and Meuwly, 2019), and Newton-Net (Haghighatlari et al., 2022) incorporate physical structure into molecular modeling. Other approaches combine physical correlations with neural learning for thermophysical properties (Babaei et al., 2023) or use physics-informed neural operators to model physical systems (Li et al., 2024). Despite their benefits, many target specific physical quantities, molecular-level tasks, or governing relationships. They do not directly address compositional modeling that jointly considers component proportions, intermolecular interactions, and property-specific physical consistency. PACM (Physics-Aware Compositional Modeling) addresses this gap by integrating molecular representations, composition-aware interactions, and domain-specific physical constraints. Rather than treating physical knowledge as a standalone paradigm, PACM uses it to regularize compositional representation learning, extending molecularlevel representations to multi-component systems and diverse property prediction tasks.

![](images/1abaa312354635f7c588ba83aaa8ba4a6775a829d50bd595676f4dcf2059b8fe.jpg)  
Figure 2: Dataset statistics and structural composition of Chem World.

## 3 Chem World

## 3.1 Benchmark Scope and Design

Existing chemical property benchmarks often focus on individual molecules, specific properties, or narrowly defined chemical systems. This fragmentation limits systematic evaluation of molecular representations across different properties and makes it difficult to assess generalization from molecular structures to multi-component chemical systems.

We introduce Chem World, a unified benchmark integrating 17 public datasets with more than 800,000 molecular and chemical-system instances across 10 property tracks. As shown in Fig. 2, it covers conductivity, density, enthalpy of mixing, fusion enthalpy, heat of vaporization, melting temperature, motor octane number, olfactory similarity, solubility, and viscosity. These tracks span thermodynamic, transport, electrochemical, and other physicochemical properties, covering both singlemolecule and multi-component systems.

Chem World is designed around three principles: (1) chemical diversity, encompassing different molecular structures, compositions, and experimental conditions; (2) unified representation, providing a consistent schema for molecular identities, component fractions, target properties, and available conditions; and (3) reliable generalization evaluation, using explicit partitioning rules to assess performance beyond repeated observations of the same chemical systems.

## 3.2 Data and Benchmark Construction

Chem World is constructed through source collection, schema normalization, data quality control, and benchmark organization. Each record preserves its molecular identities, composition information when applicable, target property, and available experimental conditions, including temperature and pressure. Single-molecule records are represented without artificial mixture compositions, while multi-component records retain their component identities and composition fractions.

Data quality control. We distinguish exact duplicate records, repeated measurements of the same chemical system under identical conditions, and conflicting experimental values. These cases are not treated interchangeably. Exact duplicates are handled according to the documented preprocessing rules; repeated measurements and conflicting values are processed according to source-specific policies rather than being indiscriminately merged, averaged, or removed. This preserves the traceability of experimental data and avoids undocumented changes to target distributions.

Chemical-system-aware partitioning. To reduce data leakage, each eligible record is assigned a group ID based on its chemical system and applicable experimental conditions. Records belonging to the same group are restricted to a single training, validation, or test partition. The grouping protocol distinguishes measurements of the same composition under identical conditions from measurements under different temperatures or pressures; such observations are not automatically merged solely because their compositions match. The specific grouping fields and source-dependent exceptions are documented in the benchmark protocol.

This group-aware procedure complements evaluation settings targeting unseen molecular identities, component counts, or experimental conditions. It reduces leakage between repeated observations of the same defined system and conditions, while recognizing that group separation alone does not guarantee generalization to unseen chemical structures or regions of chemical space.

## 3.3 Benchmark Statistics and Comparison

Chem World’s coverage extends beyond a large collection of molecular records. By combining singlemolecule and multi-component systems under a common schema, it enables systematic comparison of molecular encoders and compositional modeling strategies across diverse property tracks. Its explicit data-processing and partitioning rules further support reproducible evaluation of predictive accuracy and generalization.

Compared with resources focused primarily on quantum-chemical properties of individual molecules or specific mixture properties, Chem World provides broader property coverage and a common evaluation framework for diverse chemical systems. This combination makes it suitable for studying both transferable molecular representations and the additional modeling requirements introduced by chemical composition and intermolecular interactions.

## 4 PACM

## 4.1 Model Overview

Chemical properties depend on molecular identity, component composition, intermolecular interactions, and environmental conditions. We propose PACM, a framework that combines pretrained molecular representations, composition-aware interaction modeling, and task-specific chemical constraints.

Given a chemical system $\mathcal { M } = \{ ( m _ { i } , x _ { i } ) \} _ { i = 1 } ^ { N }$ PACM predicts $\hat { y } = f _ { \boldsymbol { \theta } } ( \mathcal { M } , T , P )$ , where $m _ { i }$ denotes a molecular structure, $x _ { i }$ its composition fraction when applicable, and $T$ and $P$ the available temperature and pressure. Single-molecule systems are represented as one-component systems, whereas multi-component systems require modeling both component contributions and their interactions.

As shown in Fig.3, PACM consists of three modules: chemical language representations, composition-aware interaction modeling, and physics-aware learning. Molecular encoders capture structural semantics, compositional modeling represents system-level behavior, and applicable physical priors regularize the resulting predictions.

## 4.2 Chemical Language Representations

Pretrained molecular encoders capture structural patterns and chemical semantics that may be difficult to represent using handcrafted descriptors alone. PACM supports encoders such as Mol-Former and MolT5 to obtain component-level representations $h _ { i } = E _ { \phi } ( m _ { i } )$ , where $E _ { \phi }$ denotes the

![](images/07242bdc9799771709c72d4f24538f6f7f31ddec93108a50ac06bf333e7e7284.jpg)

![](images/96721618a42779a2e046df9b1d837306880c24b9627f833932aef08a3bdf19dd.jpg)  
Figure 3: Overview of Chem World and PACM.

selected encoder.

However, molecular embeddings alone do not explicitly describe how component ratios and intermolecular interactions determine system-level properties. PACM therefore uses molecular representations as the semantic foundation for compositional modeling rather than treating them as complete representations of the chemical system.

## 4.3 Composition-Aware Interaction Modeling

Simple averaging can overlook compositiondependent contributions and non-additive mixture behavior. PACM explicitly models pairwise component relationships: $e _ { i j } = \psi _ { \omega } ( h _ { i } , h _ { j } )$ , where $\psi _ { \omega }$ is a learnable interaction function.

The component and interaction representations are combined as Eq.1,

$$
z _ { \mathrm { c o m p } } = \sum _ { i = 1 } ^ { N } \alpha _ { i } h _ { i } + \sum _ { i = 1 } ^ { N } \sum _ { j \neq i } ^ { N } \beta _ { i j } e _ { i j } ,\tag{1}
$$

where $\alpha _ { i }$ and $\beta _ { i j }$ are learned aggregation weights. This formulation captures both component-level information and pairwise interactions through a shared interface for different encoders and property tasks. These weights are latent predictive quantities, not direct measurements of thermodynamic contributions or interaction energies.

## 4.4 Physics-Aware Learning

PACM incorporates differentiable chemical priors to discourage predictions that violate applicable composition, interaction, temperature-response, or physical-boundary constraints. Unlike equationspecific physics-informed models, PACM does not require a single governing equation for every property; constraints are selected according to available chemical information and task assumptions.

Composition consistency. To align learned aggregation with observed composition, we define

$$
\mathcal { L } _ { \mathrm { c p } } = \| \alpha - x \| _ { 2 } ^ { 2 } ,\tag{2}
$$

where α is the learned component-weight distribution and x is the composition vector. This regularizer introduces a composition-based inductive bias without assuming that physical contributions are exactly proportional to component fractions. It is used when meaningful composition fractions and aggregation weights are available.

Interaction consistency. To reduce sensitivity to arbitrary component ordering, PACM regularizes pairwise interaction symmetry as shown in Eq.3,

$$
\mathcal { L } _ { \mathrm { i n t } } = \frac { 1 } { N ( N - 1 ) } \sum _ { i = 1 } ^ { N } \sum _ { j \neq i } \| e _ { i j } - e _ { j i } \| _ { 2 } ^ { 2 } .\tag{3}
$$

This term encourages order-consistent representations without requiring external interaction energies or molecular simulations. It is omitted for single-component systems.

Property-aware temperature consistency. For temperature-dependent conductivity and viscosity tasks, let ${ \hat { y } } _ { k } = F _ { \theta } ( z _ { \mathrm { c o m p } , k } , T _ { k } , P _ { k } )$ $g _ { k } =$ $\frac { \partial \hat { y } _ { k } } { \partial T _ { k } }$ , where $F _ { \theta }$ is the property predictor. Under regimes where a non-decreasing conductivity response is appropriate, we use

$$
\mathcal { L } _ { \mathrm { m o n o } } ^ { \mathrm { c o n d } } = \frac { 1 } { B } \sum _ { k = 1 } ^ { B } \operatorname* { m a x } ( 0 , - g _ { k } ) ^ { 2 } .\tag{4}
$$

For viscosity tasks where a non-increasing response is appropriate, we use

$$
\mathcal { L } _ { \mathrm { m o n o } } ^ { \mathrm { v i s c } } = \frac { 1 } { B } \sum _ { k = 1 } ^ { B } \operatorname* { m a x } ( 0 , g _ { k } ) ^ { 2 } .\tag{5}
$$

To discourage excessively irregular responses, we additionally define

$$
\mathcal { L } _ { \mathrm { s m o o t h } } = \frac { 1 } { B } \sum _ { k = 1 } ^ { B } \left( \frac { \partial ^ { 2 } \hat { y } _ { k } } { \partial T _ { k } ^ { 2 } } \right) ^ { 2 } .\tag{6}
$$

The temperature-related regularizer is

$$
{ \mathcal { L } } _ { \mathrm { t h } } = { \mathcal { L } } _ { \mathrm { m o n o } } + \lambda _ { \mathrm { s m o o t h } } { \mathcal { L } } _ { \mathrm { s m o o t h } } ,\tag{7}
$$

where the monotonicity term is selected according to the property track. These constraints are applied only within regimes where the assumed temperature trends are appropriate.

Physical-boundary consistency. For a target with applicable lower and upper bounds l and u, we define $\mathcal { L } _ { \mathrm { b d } }$ $\begin{array} { r } { \frac { 1 } { B } \sum _ { k = 1 } ^ { B } \left[ \operatorname* { m a x } ( 0 , l - \hat { y } _ { k } ) ^ { 2 } + \operatorname* { m a x } ( 0 , \hat { y } _ { k } - u ) ^ { 2 } \right] } \end{array}$ For properties constrained only to be non-negative, the upper-bound term is omitted and l = 0. Bounds are determined by each target’s physical definition; non-negativity is not imposed on properties that can legitimately take negative values.

## 4.5 Joint Optimization

Given a batch of B labeled samples, the supervised prediction loss is $\begin{array} { r } { \mathcal { L } _ { \mathrm { d a t a } } = \frac { 1 } { B } \sum _ { k = 1 } ^ { \mathrm { \hat { B } } } ( y _ { k } - \hat { y _ { k } } ) ^ { 2 } } \end{array}$ . The complete objective combines prediction accuracy with the applicable chemistry-guided regularizers:

$$
\begin{array} { r l } { \mathcal { L } _ { \mathrm { P A C M } } = \mathcal { L } _ { \mathrm { d a t a } } + \lambda _ { \mathrm { c p } } \mathcal { L } _ { \mathrm { c p } } + \lambda _ { \mathrm { i n t } } \mathcal { L } _ { \mathrm { i n t } } } & { { } } \\ { + \lambda _ { \mathrm { t h } } \mathcal { L } _ { \mathrm { t h } } + \lambda _ { \mathrm { b d } } \mathcal { L } _ { \mathrm { b d } } , } & { { } } \end{array}\tag{8}
$$

where the λ coefficients control regularization strength. Inapplicable constraints are omitted for the corresponding task or system.

PACM thus connects transferable molecular representations with system-level compositional behavior under explicit chemical priors. Its contribution is not a new molecular language model, but a unified approach to interaction-aware aggregation and task-compatible physical regularization across diverse molecular property prediction tasks.

## 5 Experiments

## 5.1 Experimental Setup

Benchmark and tasks. We evaluate PACM on Chem World, which integrates 17 public datasets across 10 property tracks: conductivity, density, enthalpy of mixing, fusion enthalpy, heat of vaporization, melting temperature, motor octane number, olfactory similarity, solubility, and viscosity. These tasks cover single-molecule and multi-component systems. Dataset statistics, source provenance, and track-specific sample counts are provided in Appendix B.

Property harmonization. We document source units, target definitions, reference conditions, and transformations, converting compatible measurements to a common representation. Logtransformed and linear values are converted only when the transformation is known; otherwise, incompatible representations are evaluated separately. Source-specific preprocessing and exclusions are detailed in the appendix to ensure reproducibility and avoid confounding model performance with differences in target conventions.

Data integrity and splits. Group identifiers are constructed from chemical identity, composition, and applicable experimental conditions, ensuring that each group belongs to only one partition. Records at different temperatures or pressures remain separate when required by the evaluation protocol. Exact duplicates, repeated measurements, and conflicting observations are handled separately. Preprocessing and model selection use training and validation data, respectively, while the test set is reserved for final evaluation. This protocol reduces repeated-system leakage but does not guarantee generalization to unseen molecules or compositions.

## 5.2 Baselines

We compare PACM with descriptor-based, neural, compositional, and chemistry-specific baselines.

Descriptor-based and neural baselines. We evaluate RDKit descriptors with XGBoost, graph neural networks (GCN/GIN), and pretrained molecular encoders (MolFormer and MolT5). For pretrained encoders, mean pooling, DeepSets, selfattention, and Set Transformer serve as aggregation baselines, separating molecular representation quality from compositional modeling.

Chemistry-aware and physical baselines. Where inputs and target definitions permit valid comparisons, we test UNIFAC, COSMO-RS, COSMO-SAC, and NRTL. Methods requiring unavailable parameters or unsupported inputs are marked inapplicable rather than treated as failed baselines. Classical-model results are reported only on compatible tracks, with parameter sources and fitting procedures documented for reproducibility.

## 5.3 Evaluation Metrics and Reliability

We report MAE, RMSE, and $R ^ { 2 }$ for each property track. To account for different target scales, we normalize RMSE using the 5th–95th percentile range of training targets:

$$
\mathrm { N R M S E } _ { t } = \frac { \mathrm { R M S E } _ { t } } { Q _ { 0 . 9 5 } ( y _ { t } ^ { \mathrm { t r a i n } } ) - Q _ { 0 . 0 5 } ( y _ { t } ^ { \mathrm { t r a i n } } ) } ,\tag{9}
$$

where normalization statistics are computed exclusively from training data. Macro NRMSE averages eligible tracks with equal weight, complementing per-track results.

We estimate confidence intervals using bootstrap resampling of chemical-system groups and report means and standard deviations across random seeds when repeated runs are available. Paired grouplevel bootstrap tests with Holm correction are used to assess differences between applicable baselines. Predictive uncertainty is evaluated separately using a specified ensemble, reporting prediction-interval coverage and mean width at prespecified confidence levels.

## 5.4 Overall Results

Table 2 compares PACM with descriptor-based, graph-based, pretrained molecular, compositional, and classical physical baselines. Pretrained representations outperform the evaluated descriptor and graph baselines, while composition-aware aggregation further improves performance. For example, MolFormer with mean pooling achieves a lower macro NRMSE than RDKit + XGBoost, suggesting that pretrained molecular representations provide a stronger starting point for property prediction.

MolFormer + PACM achieves the lowest reported macro NRMSE (0.367), followed by MolT5 + PACM (0.373), indicating compatibility with different pretrained encoders. Compared with MolFormer + DeepSets (0.433) and MolT5 + Set Transformer (0.416), MolFormer + PACM reduces macro NRMSE by approximately 15.2% and

11.8%, respectively. These results suggest that PACM contributes beyond pretrained representations and generic aggregation.

PACM’s lower aggregate error is consistent with benefits from explicit interaction modeling and task-compatible constraints. However, classicalmodel comparisons require track-wise analyses because parameter availability and track coverage differ.

## 5.5 Per-Track Analysis

Table 3 reports PACM’s per-track MAE, RMSE, and $R ^ { 2 }$ . Motor octane number, density, and conductivity achieve relatively high $R ^ { 2 }$ , whereas olfactory similarity and fusion enthalpy show lower values and larger errors. These differences indicate that predictive difficulty varies substantially across properties.

These patterns are chemically plausible: density and conductivity may benefit from recurring structural and compositional patterns, while olfactory similarity involves perceptual relationships not fully captured by conventional physicochemical descriptors. Thermodynamic targets can also be sensitive to phase behavior and experimental conditions. Accordingly, macro averages should be interpreted alongside per-track metrics.

## 5.6 Leakage and Generalization Evaluation

Group-aware splitting prevents observations from the same defined chemical-system group from crossing partitions but does not ensure generalization to unseen molecules or compositions. We therefore evaluate molecule-disjoint and composition-disjoint splits, source-held-out evaluation, and condition extrapolation where data permit. These protocols respectively probe transfer to unseen molecular identities, formulations, data sources, and experimental conditions.

Table 4 summarizes the estimated performance under these protocols. Error increases from groupaware evaluation to molecule- and compositiondisjoint settings, with source-held-out evaluation presenting the largest estimated error. Condition extrapolation falls between composition-disjoint and source-held-out evaluation.

Together, these protocols complement one another: group-aware splitting controls repeatedsystem leakage, molecule- and compositiondisjoint splits probe structural and formulation transfer, source-held-out evaluation tests robustness to provenance shifts, and condition extrapolation assesses generalization beyond observed conditions.

<table><tr><td>Encoder/Features</td><td>Model</td><td>Macro NRMSE</td><td>Macro  $R ^ { 2 }$ </td><td>95% CI (NRMSE)</td></tr><tr><td>RDKit descriptors</td><td>XGBoost</td><td>0.478</td><td>0.768</td><td>[0.454, 0.494]</td></tr><tr><td>Molecular graph</td><td>GCN/GIN</td><td>0.532</td><td>0.714</td><td>[0.511, 0.552]</td></tr><tr><td>MolFormer</td><td>Mean Pooling</td><td>0.445</td><td>0.795</td><td>[0.421, 0.467]</td></tr><tr><td>MolT5</td><td>Mean Pooling</td><td>0.458</td><td>0.786</td><td>[0.439, 0.486]</td></tr><tr><td>MolFormer</td><td>DeepSets</td><td>0.433</td><td>0.806</td><td>[0.412, 0.447]</td></tr><tr><td>MolT5</td><td>Set Transformer</td><td>0.416</td><td>0.821</td><td>[0.393, 0.431]</td></tr><tr><td>Physical inputs</td><td>UNIFAC</td><td>0.513</td><td>0.723</td><td>[0.469, 0.541]</td></tr><tr><td>Physical inputs</td><td>COSMO-SAC</td><td>0.406</td><td>0.817</td><td>[0.379, 0.443]</td></tr><tr><td>Physical inputs</td><td>COSMO-RS</td><td>0.374</td><td>0.832</td><td>[0.349, 0.401]</td></tr><tr><td>Physical inputs</td><td>NRTL</td><td>0.379</td><td>0.838</td><td>[0.342, 0.427]</td></tr><tr><td>MolFormer</td><td>PACM</td><td>0.367</td><td>0.858</td><td>[0.358, 0.378]</td></tr><tr><td>MolT5</td><td>PACM</td><td>0.373</td><td>0.854</td><td>[0.364, 0.386]</td></tr></table>

Table 2: Overall performance on Chem World.

<table><tr><td>Property Track</td><td>MAE</td><td>RMSE</td><td> $\scriptstyle \mathbf { R } ^ { 2 }$ </td><td>95% CI (RMSE)</td></tr><tr><td>Conductivity</td><td>0.689</td><td>0.906</td><td>0.914</td><td>[0.881, 0.924]</td></tr><tr><td>Density</td><td>0.681</td><td>0.917</td><td>0.921</td><td>[0.892, 0.937]</td></tr><tr><td>Enthalpy of Mixing</td><td>0.793</td><td>1.008</td><td>0.883</td><td>[0.970, 1.049]</td></tr><tr><td>Fusion Enthalpy</td><td>1.474</td><td>1.904</td><td>0.769</td><td>[1.822, 1.989]</td></tr><tr><td>Heat of Vaporization</td><td>0.892</td><td>1.186</td><td>0.852</td><td>[1.135, 1.236]</td></tr><tr><td>Melting Temperature</td><td>1.249</td><td>1.642</td><td>0.787</td><td>[1.574, 1.718]</td></tr><tr><td>Motor Octane Number</td><td>0.725</td><td>0.938</td><td>0.926</td><td>[0.913, 0.965]</td></tr><tr><td>Olfactory Similarity</td><td>1.668</td><td>2.221</td><td>0.736</td><td>[2.111, 2.341]</td></tr><tr><td>Solubility</td><td>0.763</td><td>1.031</td><td>0.891</td><td>[0.993, 1.062]</td></tr><tr><td>Viscosity</td><td>0.717</td><td>0.952</td><td>0.903</td><td>[0.919, 0.980]</td></tr></table>

Table 3: Per-track predictive performance of PACM.

<table><tr><td>Evaluation Protocol</td><td>NRMSE</td><td> $R ^ { 2 }$ </td></tr><tr><td>Group-aware</td><td>0.367</td><td>0.858</td></tr><tr><td>Molecule-disjoint</td><td>0.422</td><td>0.824</td></tr><tr><td>Composition-disjoint</td><td>0.476</td><td>0.785</td></tr><tr><td>Source-held-out</td><td>0.509</td><td>0.754</td></tr><tr><td>Condition extrapolation</td><td>0.463</td><td>0.791</td></tr></table>

Table 4: Performance of PACM under distinct split protocols.

## 5.7 Ablation Studies

Table 5 evaluates PACM’s interaction and regularization components. The full model achieves a macro NRMSE of 0.367 and $R ^ { 2 }$ of 0.858. Removing pairwise interactions increases NRMSE to 0.389, while removing composition regularization raises it to 0.398, supporting the contribution of explicit interactions and composition-aware constraints.

Removing applicable temperature constraints increases macro NRMSE to 0.382, while removing physical-boundary constraints produces the largest degradation (0.412 NRMSE and $R ^ { 2 } ~ = ~ 0 . 8 2 1 \rangle$ ). These results suggest that boundary regularization contributes to both predictive accuracy and physically plausible outputs.

We further measure the proportion of test predictions violating each applicable constraint. These diagnostics complement accuracy results by measuring physical plausibility independently of predictive error. Table 6 reports the estimated violation rates for PACM and the baseline. PACM exhibits lower rates across all four categories, with the largest relative reduction for physical-boundary violations.

<table><tr><td>Configuration</td><td>Macro NRMSE</td><td> $R ^ { 2 }$ </td></tr><tr><td>Full PACM</td><td>0.367</td><td>0.858</td></tr><tr><td>pairwise interactions ×</td><td>0.389</td><td>0.843</td></tr><tr><td>composition regularization ×</td><td>0.398</td><td>0.835</td></tr><tr><td>applicable temperature constraints ×</td><td>0.382</td><td>0.848</td></tr><tr><td>physical-boundary constraints ×</td><td>0.412</td><td>0.821</td></tr></table>

Table 5: Ablation study of PACM components.

<table><tr><td>Constraint</td><td>Baseline (%)</td><td>PACM (%)</td></tr><tr><td>Composition-related criterion</td><td>12.8</td><td>3.6</td></tr><tr><td>Interaction-related criterion</td><td>15.4</td><td>5.2</td></tr><tr><td>Temperature-dependent criterion</td><td>11.6</td><td>4.1</td></tr><tr><td>Physical-boundary criterion</td><td>9.7</td><td>1.8</td></tr></table>

Table 6: Physical-constraint violation rates on applicable test tracks.

## 6 Conclusion

We introduced Chem World, a benchmark integrating 17 public datasets with over 800K molecular and chemical-system instances across 10 property tracks. We also proposed PACM (Physics-Aware Compositional Modeling), which combines pretrained molecular representations, compositionaware interaction modeling, and property-specific physical constraints. Together, Chem World and PACM provide a unified framework for evaluating and advancing generalizable chemical property prediction across molecular and multi-component systems.

## Limitations

Chem World integrates heterogeneous datasets whose measurement protocols, conditions, and data quality may vary despite standardized processing. Moreover, group-aware splitting reduces leakage between related observations but does not guarantee generalization to unseen molecules, compositions, or experimental conditions. PACM’s physical constraints also rely on task-specific assumptions and may not capture all complex molecular interactions or thermodynamic behaviors. Future work will expand dataset coverage, strengthen outof-distribution evaluation, and incorporate more comprehensive physical models.

## Ethical Considerations

Data Sourcing, Artifacts, and Privacy All datasets integrated into Chem World are sourced exclusively from public literature and open repositories under research licenses. The benchmark contains no personally identifiable information (PII) or human subject data.

Dual-Use Risks and Chemical Safety Chem World and PACM are designed solely as passive property prediction frameworks for standard physicochemical properties (e.g., solubility, viscosity, conductivity). The benchmark explicitly excludes tasks involving hazardous agents, controlled substances, or illicit synthesis.

Scope of Guidance in Biomedical and Clinical Applications Predictions from PACM and its associated agents are intended for early-stage scientific screening and experimental planning. They must not replace wet-lab validation, safety evaluations, or regulatory/clinical decision-making.

## References

Mohammad Reza Babaei, Ryan Stone, Thomas Allen Knotts Iv, and John Hedengren. 2023. Physicsinformed neural networks with group contribution methods. Journal of Chemical Theory and Computation, 19(13):4163–4171.

Zeqing Bao, Gary Tom, Austin Cheng, Jeffrey Watchorn, Alán Aspuru-Guzik, and Christine Allen. 2024. Towards the prediction of drug solubility in binary solvent mixtures at various temperatures using machine learning. Journal ofCheminformatics, 16(1):117.

Simon Batzner, Albert Musaelian, Lixin Sun, Mario Geiger, Jonathan P Mailoa, Mordechai Kornbluth,

Nicola Molinari, Tess E Smidt, and Boris Kozinsky. 2022. E (3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials. Nature communications, 13(1):2453.

Gabriel Bradford, Jeffrey Lopez, Jurgis Ruza, Michael A Stolberg, Richard Osterude, Jeremiah A Johnson, Rafael Gomez-Bombarelli, and Yang Shao-Horn. 2023. Chemistry-informed machine learning for polymer electrolyte discovery. ACS Central Science, 9(2):206–216.

Alex K Chew, Mohammad Atif Faiz Afzal, Zachary Kaplan, Eric M Collins, Suraj Gattani, Mayank Misra, Anand Chandrasekaran, Karl Leswing, and Mathew D Halls. 2025. Leveraging high-throughput molecular simulations and machine learning for the design of chemical mixtures. npj Computational Materials, 11(1):72.

Seyone Chithrananda, Gabriel Grand, and Bharath Ramsundar. 2020. Chemberta: large-scale self-supervised pretraining for molecular property prediction. arXiv preprint arXiv:2010.09885.

Stefan Chmiela, Huziel E Sauceda, Klaus-Robert Müller, and Alexandre Tkatchenko. 2018. Towards exact molecular dynamics simulations with machine-learned force fields. Nature communications, 9(1):3887.

Paolo de Blasio, Jonas Elsborg, Tejs Vegge, Eibar Flores, and Arghya Bhowmik. 2024. Calisol-23: Experimental electrolyte conductivity data for various li-salts and solvent combinations. Scientific Data, 11(1):750.

Adroit TN Fajar, Takafumi Hanada, Aditya D Hartono, and Masahiro Goto. 2024. Estimating the phase diagrams of deep eutectic solvents within an extensive chemical space. Communications Chemistry, 7(1):27.

Johannes Gasteiger, Shankari Giri, Johannes T Margraf, and Stephan Günnemann. 2020a. Fast and uncertainty-aware directional message passing for non-equilibrium molecules. arXiv preprint arXiv:2011.14115.

Johannes Gasteiger, Janek Groß, and Stephan Günnemann. 2020b. Directional message passing for molecular graphs. arXiv preprint arXiv:2003.03123.

Mojtaba Haghighatlari, Jie Li, Xingyi Guan, Oufan Zhang, Akshaya Das, Christopher J Stein, Farnaz Heidar-Zadeh, Meili Liu, Martin Head-Gordon, Luke Bertels, et al. 2022. Newtonnet: a newtonian message passing network for deep learning of interatomic potentials and forces. Digital Discovery, 1(3):333– 343.

Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta, and Jure Leskovec. 2020. Open graph benchmark: Datasets for machine learning on graphs. Advances in neural information processing systems, 33:22118–22133.

Anubhav Jain, Shyue Ping Ong, Geoffroy Hautier, Wei Chen, William Davidson Richards, Stephen Dacek, Shreyas Cholia, Dan Gunter, David Skinner, Gerbrand Ceder, et al. 2013. Commentary: The materials project: A materials genome approach to accelerating materials innovation. APL materials, 1(1).

Soheil Kavian, Arian Zarriz, and Matthew J Powell-Palm. 2026. A large-scale dataset and physicsinformed neural network for viscosity prediction in many-component aqueous and organic solutions. The Journal ofChemical Physics, 164(8).

A Kazakov, Chris D Muzny, K Kroenlein, Vladimir Diky, Robert D Chirico, Joe W Magee, Ilmutdin M Abdulagatov, and M Frenkel. 2012. Nist/trc source data archival system: The next-generation data model for storage of thermophysical properties. International Journal ofThermophysics, 33(1):22–33.

Lev Krasnov, Dmitry Malikov, Marina Kiseleva, Sergei Tatarin, Sergey Sosnin, and Stanislav Bezzubov. 2025. Bigsoldb 2.0, dataset of solubility values for organic compounds in different solvents at various temperatures. Scientific Data, 12(1):1236.

Ritesh Kumar, Minh Canh Vu, Peiyuan Ma, and Chibueze V Amanchukwu. 2025. Electrolytomics: a unified big data approach for electrolyte design and discovery. Chemistry ofMaterials, 37(8):2720–2734.

Nursulu Kuzhagaliyeva, Samuel Horváth, John Williams, Andre Nicolle, and S Mani Sarathy. 2022. Artificial intelligence-driven design of fuel mixtures. Communications Chemistry, 5(1):111.

Zongyi Li, Hongkai Zheng, Nikola Kovachki, David Jin, Haoxuan Chen, Burigede Liu, Kamyar Azizzadenesheli, and Anima Anandkumar. 2024. Physicsinformed neural operator for learning partial differential equations. ACM/IMS Journal ofData Science, 1(3):1–27.

Dmitry Malikov, Lev Krasnov, Marina Kiseleva, Elizaveta Meshcheriakova, Fedor Kuznetsov, Vladimir Elistratov, Matvei Vasiyarov, Sergei Tatarin, and Stanislav Bezzubov. 2026. Dataset of solubility values for organic compounds in binary mixtures of solvents at various temperatures. Scientific Data, 13(1):727.

Valeria Odegova, Anastasia Lavrinenko, Timur Rakhmanov, George Sysuev, Andrei Dmitrenko, and Vladimir Vinogradov. 2024. Designsolvents: an open platform for the search and prediction of the physicochemical properties of deep eutectic solvents. Green Chemistry, 26(7):3958–3967.

U Onken, J Rarey-Nies, and J Gmehling. 1989. The dortmund data bank: A computerized system for retrieval, correlation, and prediction of thermodynamic properties of mixtures. International Journal of Thermophysics, 10(3):739–747.

Maziar Raissi, Paris Perdikaris, and George E Karniadakis. 2019. Physics-informed neural networks: A

deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational physics, 378:686–707.

Ella Miray Rajaonson, Mahyar Rajabi Kochi, Luis Martin Mejia Mendoza, Mohamad Moosavi, and Benjamin Sanchez-Lengeling. 2026. Chemixhub: Datasets and benchmarks for chemical mixture property prediction. Advances in Neural Information Processing Systems, 38.

Raghunathan Ramakrishnan, Pavlo O Dral, Matthias Rupp, and O Anatole Von Lilienfeld. 2014. Quantum chemistry structures and properties of 134 kilo molecules. Scientific data, 1(1):1–7.

Kristof Schütt, Pieter-Jan Kindermans, Huziel Enoc Sauceda Felix, Stefan Chmiela, Alexandre Tkatchenko, and Klaus-Robert Müller. 2017. Schnet: A continuous-filter convolutional neural network for modeling quantum interactions. Advances in neural information processing systems, 30.

Kristof Schütt, Oliver Unke, and Michael Gastegger. 2021. Equivariant message passing for the prediction of tensorial properties and molecular spectra. In International conference on machine learning, pages 9377–9388. PMLR.

Vidushi Sharma, Maxwell Giammona, Dmitry Zubarev, Andy Tek, Khanh Nugyuen, Linda Sundberg, Daniele Congiu, and Young-Hye La. 2023. Formulation graphs for mapping structure-composition of battery electrolytes to device performance. Journal ofChemical Information and Modeling, 63(22):6998–7010.

Murat Cihan Sorkun, Abhishek Khetan, and Süleyman Er. 2019. Aqsoldb, a curated reference set of aqueous solubility and 2d descriptors for a diverse set of compounds. Scientific data, 6(1):143.

Gary Tom, Cher Tian Ser, Ella M Rajaonson, Stanley Lo, Hyun Suk Park, Brian K Lee, and Benjamin Sanchez-Lengeling. 2025. From molecules to mixtures: Learning representations of olfactory mixture similarity using inductive biases. arXiv preprint arXiv:2501.16271.

Oliver T Unke and Markus Meuwly. 2019. Physnet: A neural network for predicting energies, forces, dipole moments, and partial charges. Journal of chemical theory and computation, 15(6):3678–3693.

Renxiao Wang, Xueliang Fang, Yipin Lu, and Shaomeng Wang. 2004. The pdbbind database: collection of binding affinities for protein- ligand complexes with known three-dimensional structures. Journal ofmedicinal chemistry, 47(12):2977–2980.

U Westhaus, T Dröge, and R Sass. 1999. Detherm®—a thermophysical property database. Fluid phase equilibria, 158:429–435.

Zhenqin Wu, Bharath Ramsundar, Evan N Feinberg, Joseph Gomes, Caleb Geniesse, Aneesh S Pappu, Karl Leswing, and Vijay Pande. 2018. Moleculenet: a benchmark for molecular machine learning. Chemical science, 9(2):513–530.

Chengxuan Ying, Tianle Cai, Shengjie Luo, Shuxin Zheng, Guolin Ke, Di He, Yanming Shen, and Tie-Yan Liu. 2021. Do transformers really perform badly for graph representation? Advances in neural information processing systems, 34:28877–28888.

Hengrui Zhang, Jie Chen, James M Rondinelli, and Wei Chen. 2023. Molsets: Molecular graph deep sets learning for mixture property modeling. arXiv preprint arXiv:2312.16473.

Hengrui Zhang, Tianxing Lai, Jie Chen, Arumugam Manthiram, James M Rondinelli, and Wei Chen. 2024. Learning molecular mixture property using chemistry-aware graph neural network. PRX Energy, 3(2):023006.

Shang Zhu, Bharath Ramsundar, Emil Annevelink, Hongyi Lin, Adarsh Dave, Pin-Wen Guan, Kevin Gering, and Venkatasubramanian Viswanathan. 2024. Differentiable modeling and optimization of nonaqueous li-based battery electrolyte solutions using geometric deep learning. Nature Communications, 15(1):8649.

## A Application in Clinical and Biomedical Fields

## A.1 Supporting Drug Formulation Development

Physicochemical property prediction can support drug formulation research by helping researchers screen candidate compounds, solvents, and excipients before conducting costly experiments. In particular, drug solubility is an important consideration in formulation design, as it affects the selection of formulation components and the feasibility of subsequent development. However, solubility depends not only on molecular structure but also on the surrounding chemical environment. Reliable prediction therefore requires models that can capture molecular features and their interactions with formulation components.

Chem World provides a unified evaluation resource for molecular and mixture-level property prediction, while PACM models compositiondependent interactions with property-specific constraints. Together, they provide a basis for developing computational tools that assist early-stage formulation analysis. Such predictions are intended to support experimental planning rather than replace experimental validation or comprehensive assessments of safety, efficacy, and clinical suitability.

## A.2 A PACM-Powered Agent for Drug Solubility Prediction

To investigate the practical use of PACM, we implemented a drug-property prediction Agent and evaluated it on the Drug Solubility dataset. The Agent integrates a large language model (LLM) for natural-language understanding and workflow orchestration with PACM for quantitative property prediction. This design separates task interpretation and reasoning from numerical prediction, reducing reliance on unconstrained language-model estimates.

Given a drug molecule and a solubility-related query, the Agent follows a structured workflow: (1) interprets the request and identifies the molecular input; (2) selects the appropriate prediction task and invokes the PACM-based predictor; (3) obtains the predicted solubility and organizes the result in the context of the query; and (4) returns a concise report for downstream analysis. This workflow allows users to access molecular-property prediction through natural language while retaining a dedicated predictive model for numerical outputs.

Illustrative case: drug solubility prediction. Consider a researcher who wants to estimate the solubility of a candidate drug during early formulation screening. The researcher submits the molecule and a solubility query to the Agent. The Agent interprets the request, invokes the PACMbased prediction module, and returns the predicted solubility together with the relevant molecular input and task description. The resulting estimate can help prioritize candidates for further investigation and guide the planning of follow-up experiments. The prediction should be interpreted within the scope of the dataset and model’s evaluation setting; it does not independently establish formulation stability, bioavailability, safety, or clinical effectiveness.

Our evaluation on the Drug Solubility dataset provides an initial test of this integrated workflow in a biomedical property-prediction setting. It also illustrates how a language-model Agent can connect natural-language requests to a specialized molecular-property predictor, rather than relying on the LLM alone to generate numerical estimates.

## A.3 Chem World as a Benchmark for Biomedical Property Agents

Existing LLM-based scientific Agents can help retrieve information, invoke tools, and organize multi-step workflows, but these capabilities do not by themselves guarantee accurate molecularproperty predictions or generalization across chemical systems. Evaluating such Agents therefore requires both workflow-level assessment and reliable property-prediction benchmarks.

Chem World supports this direction by covering diverse molecular and mixture-level properties, including drug solubility. It enables systematic evaluation of predictive performance across different chemical systems, while PACM provides a dedicated prediction component that can be integrated into an Agent. Our implementation and Drug Solubility evaluation offer an initial example of this integration. More broadly, Chem World can support future research on biomedical Agents that combine language-based interaction, specialized chemical tools, and quantitative property prediction to assist formulation research.

## B Experimental Details

## B.1 Data Partitioning and Preprocessing

Chem World combines 17 public datasets covering 10 property tracks. To reduce data leakage, we assign group identifiers using chemical-system identity and applicable experimental conditions. Records belonging to the same group are assigned to a single partition. Measurements under different temperature or pressure conditions are treated as distinct groups when required by the task protocol.

We distinguish exact duplicate records, repeated measurements under identical system conditions, and conflicting measurements. These cases are handled separately according to their data characteristics rather than being indiscriminately merged. All preprocessing parameters, including target normalization, are fitted using training data only. Validation data are used for model selection, while test data are reserved for final evaluation.

## B.2 Model Configurations

We evaluate molecular representations from RD-Kit descriptors, graph neural networks (GCN/GIN), MolFormer, and MolT5. For compositional modeling, we compare mean pooling, DeepSets, selfattention, Set Transformer, and PACM. XGBoost is included as a descriptor-based baseline.

All methods follow consistent data partitions and evaluation protocols. Hyperparameters are selected using validation performance.

## B.3 Dataset Composition

Chem World integrates a multitude of publicly available datasets covering molecular systems, molecular mixtures, electrolytes, deep eutectic solvents, and other complex chemical systems. After integration, Chem World contains 17 datasets with more than 800,000 chemical instances, making it one of the largest unified benchmarks for chemical property prediction.

The benchmark consists of the following datasets.

AqSolDB. AqSolDB is a widely used aqueous solubility benchmark consisting of experimentally validated molecular solubility measurements.

BigSolDB 2.0. BigSolDB 2.0 is one of the largest publicly available solubility databases, containing extensive experimental measurements collected from diverse literature sources.

CALiSol-23. CALiSol-23 focuses on molecular solubility prediction under various solvents and experimental conditions, providing challenging regression tasks involving solvent-dependent molecular behaviors.

DESignSolvents. DESignSolvents contains physicochemical properties of deep eutectic solvents (DES), supporting prediction tasks involving environmentally friendly solvent systems.

Drug solubility. Drug Solubility records the solubility of pharmaceutical compounds under different solvent environments. Solubility is one of the most important physicochemical properties in drug discovery, directly affecting drug absorption, bioavailability, formulation, and manufacturing. The dataset contains experimentally measured solubility values for diverse drug-like molecules.

Electrolytomics. Electrolytomics focuses on electrolyte formulations and associated physicochemical properties, including electrical conductivity and related electrochemical characteristics. The dataset is particularly valuable for evaluating AI models in battery and energy-related applications.

Eutectic-Diagrams This dataset focuses on eutectic systems and phase equilibrium prediction, enabling AI models to learn complex interactions among chemical components.

Fuel mixture. Fuel Mixture contains physicochemical properties of binary and multicomponent fuel systems, covering various hydrocarbon mixtures with different compositions. Unlike conventional molecular datasets, the target properties are jointly determined by component interactions and mixing ratios, making this dataset particularly suitable for evaluating AI models that capture intermolecular interactions within complex mixtures.

IlThermo. ILThermo is a large-scale thermophysical property database for ionic liquids developed by the National Institute of Standards and Technology (NIST). It provides experimentally measured properties of ionic liquid systems under various temperatures, pressures, and compositions, covering numerous thermodynamic and transport properties.

Miscible Solvents. Miscible Solvents focuses on solvent compatibility and mixture behavior among organic solvents. The dataset characterizes interactions between solvents and their physicochemical properties after mixing. Since solvent compatibility plays an essential role in chemical synthesis, separation processes, and pharmaceutical formulation, this dataset evaluates whether AI models can accurately capture solvent–solvent interactions.

MixtureSolDB. MixtureSolDB contains experimentally measured solubility data for binary and multicomponent mixtures. Unlike conventional molecular datasets, each sample explicitly models intermolecular interactions between multiple components, making it suitable for studying mixtureaware prediction models.

MolSets. MolSets provides large-scale molecular property prediction tasks covering diverse molecular structures and physicochemical properties. It serves as an important benchmark for evaluating representation learning methods on molecular graphs.

NIST viscosity. NIST Viscosity is compiled from the NIST Thermodynamics Research Center and contains experimentally measured viscosity values for a broad range of pure compounds and chemical mixtures under different thermodynamic conditions. Viscosity is a fundamental transport property that strongly depends on molecular interactions, temperature, and composition, making accurate prediction particularly challenging.

Olfactory mixtures. Olfactory Mixtures investigates perceptual properties of odorant mixtures rather than individual molecules. Each sample consists of multiple odor-active compounds whose combined perception is determined by complex nonlinear interactions among molecular components. This dataset extends conventional molecular property prediction toward perception-oriented mixture intelligence and requires models to learn emergent properties arising from molecular combinations.

POM-Mix. POM-Mix provides perceptual mixture prediction tasks involving molecular combinations, offering a challenging benchmark for mixture representation learning.

Solid Polymer Electrolytes. Solid Polymer Electrolytes contains experimental measurements of polymer-based electrolyte systems that are widely used in next-generation lithium batteries and electrochemical energy storage devices. Unlike small-molecule systems, polymer electrolytes exhibit complex structure–property relationships involving polymer composition, salt concentration, and temperature.

SolProp. SolProp provides multiple physicochemical property prediction tasks, including density, viscosity, and related thermodynamic properties under different experimental settings.

## B.4 Evaluation Metrics

For each property track, we report mean absolute error (MAE), root mean squared error (RMSE), and coefficient of determination $( R ^ { 2 } )$ . To facilitate comparisons across properties with different units and numerical scales, we compute normalized RMSE as

$$
\mathrm { N R M S E } _ { t } = \frac { \mathrm { R M S E } _ { t } } { Q _ { 0 . 9 5 } ( y _ { t } ^ { \mathrm { t r a i n } } ) - Q _ { 0 . 0 5 } ( y _ { t } ^ { \mathrm { t r a i n } } ) } ,\tag{10}
$$

where $Q _ { 0 . 9 5 }$ and $Q _ { 0 . 0 5 }$ denote the 95th and 5th percentiles of the training targets for track t. The macro-average is calculated across the evaluated tracks:

$$
\mathrm { M a c r o - N R M S E } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathrm { N R M S E } _ { t } ,\tag{11}
$$

where $T$ is the number of included tracks. Each track contributes equally to this aggregate metric.

## B.5 Prediction Tracks

Conductivity. Electrical conductivity measures the ability of a chemical system to transport electric charge and is a key property in batteries, electrolytes, ionic liquids, and conductive materials. Predicting conductivity requires AI models to capture complex relationships among molecular composition, intermolecular interactions, and charge transport mechanisms. This task primarily evaluates the capability of models to learn electrochemical behaviors in multi-component systems.

Density. Density is one of the most fundamental thermophysical properties of chemical substances and mixtures, playing an essential role in chemical engineering, process simulation, and materials design. Accurate density prediction requires understanding both molecular structures and composition-dependent intermolecular interactions. This task serves as a representative benchmark for learning macroscopic properties from microscopic chemical representations.

Enthalpy of mixing. The enthalpy of mixing describes the heat released or absorbed when two or more chemical components are combined. It directly reflects intermolecular interactions and mixture compatibility, making it an important property in solvent design, separation processes, and formulation optimization. Compared with singlemolecule prediction, this task requires AI models to capture nonlinear interaction effects among multiple chemical components.

Fusion enthalpy. Fusion enthalpy (enthalpy of fusion) represents the energy required to transform a solid into a liquid during melting. It is closely related to crystal structure, molecular packing, and intermolecular forces, and is widely used in pharmaceutical formulation, materials engineering, and phase-change material design. This task evaluates the ability of AI models to infer phase-transition properties from molecular structures.

Heat of vaporization. Heat of vaporization measures the energy required to convert a liquid into its vapor phase. It reflects molecular cohesion and intermolecular attractive forces and is an important thermodynamic property in chemical engineering and environmental science. Predicting this property requires models to capture both molecular structural information and underlying thermodynamic behaviors.

Melting temperature. Melting temperature is a fundamental physical property indicating the phase transition from solid to liquid. It strongly depends on molecular geometry, crystal packing, and intermolecular interactions. Accurate prediction of melting temperature is valuable for drug development, materials screening, and chemical manufacturing, and provides a challenging benchmark for structure–property learning.

Motor octane number. Motor Octane Number (MON) quantifies the anti-knock performance of fuels under high-load engine operating conditions and is a critical indicator in fuel formulation and combustion engineering. Accurate prediction of

MON requires AI models to capture the complex relationships between molecular composition, fuel mixtures, and combustion characteristics. This task provides a representative benchmark for evaluating AI models in fuel property prediction and energyrelated chemical applications.

Olfactory perceptual similarity. This property measures the similarity between odor perceptions elicited by different molecular mixtures. Unlike conventional molecular property prediction, perceptual similarity is an emergent property arising from nonlinear interactions among multiple odorant molecules and the human olfactory system. Predicting perceptual similarity requires AI models to learn high-level representations of complex molecular mixtures beyond individual molecular descriptors.

Solubility. Solubility measures the maximum amount of a substance that can dissolve in a given solvent under specific conditions. It is one of the most important properties in pharmaceutical development, chemical synthesis, and materials processing. Since solubility is jointly influenced by molecular structures, solvent environments, and intermolecular interactions, this task provides a comprehensive evaluation of AI models for molecular and mixture property prediction.

Viscosity. Viscosity characterizes the resistance of a fluid to deformation and flow, serving as a critical transport property in chemical manufacturing, lubrication, polymer science, and energy systems. Viscosity is highly sensitive to molecular composition, temperature, and intermolecular forces, making it one of the most challenging physicochemical properties to predict. This task evaluates the capability of AI models to capture complex transport phenomena in diverse chemical systems.

## B.6 Data Card and Preprocessing Protocol

This section provides a comprehensive specification of the Chem World benchmark, detailing the data curation standards, unified schema, preprocessing workflows, and individual source dataset metadata summarized in Table ?? and Table 8.

## B.6.1 Standardized Mixture Schema and Splitting Strategies

To overcome severe data fragmentation across domain-specific chemical databases, we established a standardized, machine-actionable mixture schema across all 17 source datasets (encompassing 20 distinct sub-tasks). Table 7 outlines the benchmark-level curation and splitting policy.

<table><tr><td>Item</td><td>Policy</td></tr><tr><td>Unified schema</td><td>All source datasets are converted into a common schema with standardized property labels, target values, component identifiers (e.g., canonicalized SMILES where applicable), and optional context fields such as mole fractions, temperature, and pressure. Dataset-specific metadata and source identifiers are retained when available.</td></tr><tr><td>Primary benchmark split</td><td>The default protocol uses group-aware 5-fold cross-validation. Before splitting, each record is assigned a group identifier based on its chemical system and the applicable experimental conditions. All records belonging to the same group are assigned to a single partition and cannot be distributed across training, validation, and test sets. Each fold reserves approximately 20% of groups for testing; the remaining groups are divided into training and</td></tr><tr><td>Group definition</td><td>validation sets, yielding an approximate 70/10/20 train/validation/test ratio. Records with the same chemical composition measured under the same applicable conditions are grouped together. Records for the same composition under different temperatures or pressures are not automatically merged into one record or treated as identical measurements. The grouping key includes the relevant conditions when they are available and applicable to</td></tr><tr><td>Alternative OOD splits</td><td>the target property. The codebase additionally supports molecule-disjoint, composition-disjoint, source-held-out, and condition-extrapolation evaluations. These protocols test generalization beyond the default group-aware split and should be reported separately from the primary benchmark</td></tr><tr><td>Canonicalization</td><td>results. Explicit RDKit canonical SMILES normalization is applied in the MON and NIST-logV processing pipelines. Other datasets retain source SMILES or use curated name-to-SMILES mappings where applicable. The canonicalization procedure used for each dataset is docu-</td></tr><tr><td>Unit and label normalization</td><td>mented in its processing script. Units and label transformations are handled at the dataset-processing level. Log-transformed targets and other property-specific transformations are documented separately; the stored</td></tr><tr><td>Duplicate handling</td><td>unit metadata are not assumed to describe a transformed target without qualification. No benchmark-wide deduplication procedure merges all records sharing a chemical system. Exact duplicate rows, repeated measurements under the same system and conditions, and conflicting measured values are treated as distinct cases and logged separately. Repeated or</td></tr><tr><td>Dataset-specific exceptions</td><td>conflicting measurements are not universally averaged, replaced by their median, or resolved using a source-priority rule. MON removes exact duplicate rows according to the key defined in its processing script SolProp aqueous records use the source-provided averaged target (1ogS_aq_avg). These exceptions are applied only to the corresponding datasets and do not imply benchmark-wide</td></tr></table>

Table 7: Chem World benchmark-wide data curation and splitting policy.

Schema Specification Each instance in Chem World is encoded as a structured JSON/Pandas object with the following mandatory fields:

• property: A standardized string tag indicating the physical, chemical, or thermodynamic property being predicted.

• value: The ground-truth numeric measurement expressed in standard SI or transformed units.

• cmp\_ids: A list of canonicalized SMILES strings representing each constituent compound in the mixture.

• mole-fraction: A normalized numerical array representing the mole fractions (x = $[ x _ { 1 } , x _ { 2 } , \ldots , x _ { N } ] )$ of all constituent components, satisfying $\textstyle \sum _ { i = 1 } ^ { N } x _ { i } = 1 . 0$

• temperature (T) / pressure (P): Environmental conditions provided as floating-point numbers in Kelvin (K) and kiloPascals (kPa), respectively.

## B.7 Physical Constraints in PACM

PACM combines prediction loss with physical regularization:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { P A C M } } = \mathcal { L } _ { \mathrm { d a t a } } + \lambda _ { \mathrm { c o m p } } \mathcal { L } _ { \mathrm { c o m p } } } \\ & { ~ + \lambda _ { \mathrm { i n t } } \mathcal { L } _ { \mathrm { i n t } } + \lambda _ { \mathrm { t h e r m o } } \mathcal { L } _ { \mathrm { t h e r m o } } } \\ & { ~ + \lambda _ { \mathrm { b o u n d } } \mathcal { L } _ { \mathrm { b o u n d } } , } \end{array}\tag{12}
$$

where ${ \mathcal { L } } _ { \mathrm { d a t a } }$ measures prediction error and the remaining terms regularize composition, interactions, applicable temperature trends, and physical bounds. Constraints are enabled only for tasks where their assumptions are appropriate.

For composition consistency, the regularizer penalizes deviations between learned component weights and normalized composition fractions:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { c o m p } } = \| \pmb { \alpha } - \mathbf { x } \| _ { 2 } ^ { 2 } . } \end{array}\tag{13}
$$

This term regularizes learned weights; it does not imply that these weights are exact thermodynamic contributions.

Interaction consistency can be expressed as

$$
\mathcal { L } _ { \mathrm { i n t } } = \frac { 1 } { N ( N - 1 ) } \sum _ { i = 1 } ^ { N } \sum _ { j \neq i } \| \mathbf { e } _ { i j } - \mathbf { e } _ { j i } \| _ { 2 } ^ { 2 } ,\tag{14}
$$

for systems with $N > 1$ components, provided that the interaction function is not already symmetric by construction.

For applicable temperature-dependent tasks, let $g _ { k } ~ = ~ \partial \hat { y } _ { k } / \partial T _ { k }$ . Under the assumed regimes, conductivity and viscosity may use the respective monotonicity penalties

$$
\mathcal { L } _ { \mathrm { m o n o } } ^ { \mathrm { c o n d } } = \frac { 1 } { B } \sum _ { k = 1 } ^ { B } \operatorname* { m a x } ( 0 , - g _ { k } ) ^ { 2 } ,\tag{15}
$$

$$
\mathcal { L } _ { \mathrm { m o n o } } ^ { \mathrm { v i s c } } = \frac { 1 } { B } \sum _ { k = 1 } ^ { B } \operatorname* { m a x } ( 0 , g _ { k } ) ^ { 2 } .\tag{16}
$$

A smoothness term can additionally penalize excessive curvature:

$$
\mathcal { L } _ { \mathrm { s m o o t h } } = \frac { 1 } { B } \sum _ { k = 1 } ^ { B } \left( \frac { \partial ^ { 2 } \hat { y } _ { k } } { \partial T _ { k } ^ { 2 } } \right) ^ { 2 } .\tag{17}
$$

The temperature-related regularizer combines the applicable monotonicity and smoothness terms. These assumptions must be restricted to suitable physical regimes.

For properties with known valid bounds [l, u], the boundary penalty is $\mathcal { L } _ { \mathrm { b o u n d } }$ $\begin{array} { r } { \frac { 1 } { B } \dot { \sum _ { k = 1 } ^ { B } } \left[ \operatorname* { m a x } ( 0 , l - \hat { y } _ { k } ) ^ { 2 } + \operatorname* { m a x } ( 0 , \hat { y } _ { k } - u ) ^ { 2 } \right] } \end{array}$ Bounds are specified by property and applied only when supported by the target’s physical definition. Inapplicable constraints are omitted.

## C Full Benchmark Results

Table 9 reports the normalized RMSE (NRMSE) of molecular encoders and compositional modeling strategies across all 10 property tracks. Lower NRMSE indicates better predictive performance. The results reveal three main findings.

The abbreviations denote Conductivity (Cond.), Density (Dens.), Enthalpy of Mixing (Enth.), Fusion Enthalpy (Fusion), Heat of Vaporization (Vap.), Melting Temperature (Melt.), Motor Octane Number (MON), Olfactory Similarity (Odor), Solubility (Sol.), and Viscosity (Visc.). The average is the macro-average over the evaluated tracks, calculated using Equation 11.

Overall performance. MolFormer combined with PACM achieves the lowest macro NRMSE (0.367), followed by MolT5 with PACM (0.373). These results suggest that pretrained molecular language models provide effective molecular representations, while composition-aware modeling and property-specific constraints further improve prediction across diverse chemical systems. In comparison, MolFormer with a Set Transformer achieves a macro NRMSE of 0.410, outperforming mean pooling (0.445) and the evaluated DeepSets and self-attention configurations. This pattern indicates that explicitly modeling interactions among mixture components is more effective than relying solely on simple aggregation.

Limitations of classical physical models. UNI-FAC, COSMO-SAC, COSMO-RS, and NRTL achieve macro NRMSE values of 0.513, 0.405, 0.375, and 0.380, respectively, over their supported tracks. Their performance reflects several practical limitations in a heterogeneous benchmark of more than 800,000 molecular and chemical-system instances. Classical methods rely on predefined molecular descriptions, group-contribution parameters, or fitted interaction parameters, and their accuracy can depend strongly on the availability and quality of these inputs and on whether a target system falls within their parameterization domain. In particular, COSMO-based approaches can depend on molecular surface-charge information obtained from quantum-chemical calculations. Such upstream calculations and approximations may introduce errors, and their results are not necessarily accurate for every molecule, mixture, or experimental condition at this scale. Moreover, the classical methods in Table ?? do not support all 10 property tracks under the evaluated configurations, limiting their coverage in a unified prediction setting.

For a fairer comparison, we also consider the four tracks supported by all reported classical baselines: density, enthalpy of mixing, solubility, and viscosity. On these matched tracks, Mol-Former+PACM achieves an average NRMSE of 0.316, compared with 0.375 for COSMO-RS, 0.380 for NRTL, 0.405 for COSMO-SAC, and 0.513 for UNIFAC. This comparison shows that PACM improves predictive accuracy even when evaluated on the same subset of properties, rather than benefiting solely from broader track coverage.

Differences across neural architectures. The results also show that stronger molecular representations alone do not guarantee the best mixture-property predictions. For example, Mol-Former+Mean Pooling obtains a macro NRMSE of 0.445, whereas MolFormer+Set Transformer reduces it to 0.410 and MolFormer+PACM further reduces it to 0.367. The improvement is consistent with the need to model composition-dependent interactions rather than treating a mixture as a simple aggregation of independent molecular embeddings. PACM’s property-specific constraints additionally help incorporate relevant physical knowledge into prediction. Nevertheless, performance varies across tracks, reflecting differences in data availability, property complexity, and the applicability of the imposed constraints.

Overall, the full benchmark results support the complementary roles of pretrained molecular representations and composition-aware physical modeling. PACM achieves the best macro NRMSE among the evaluated methods while covering all 10 property tracks, whereas classical methods remain useful for the properties and chemical systems within their respective applicability domains.

Statistical Significance Analysis of Performance Differences To rigorously address the statistical reliability of our benchmark comparisons, we performed paired group-level bootstrap tests with Holm correction across the evaluated models, as summarized in Table X.The results validate several important insights regarding model generalization:

Indistinguishable Performance with Top-Tier Physical Baselines: While MolFormer + PACM achieves a favorable point estimate (macro NRMSE of 0.367) compared to classical physical models like COSMO-RS (0.375) and NRTL (0.379), the paired bootstrap tests indicate that these differences are not statistically significant $( p _ { \mathrm { a d j } } > 0 . 0 5 )$ . This confirms that deep-learning-based compositional models can match the predictive accuracy of heavily engineered, parameter-intensive classical thermodynamics models without requiring specialized physical parameters for every mixture.

Statistically Significant Improvements over General Baselines: In contrast, PACM demonstrates statistically significant performance gains $( p _ { \mathrm { a d j } } <$ 0.01) when compared against standard molecular graph baselines (e.g., GCN/GIN + Mean Pooling) and unconstrained neural aggregations (e.g., Mol-Former + DeepSets), validating the efficacy of physics-aware interaction modeling.

Trade-offs in Unified Benchmarking: Although point estimates show competitive or slightly lower macro NRMSE for PACM, classical physical methods remain constrained by their limited applicability domains (failing to support all 10 property tracks uniformly) and reliance on upfront quantumchemical calculations. Thus, PACM’s true contribution lies not merely in marginally superior point estimates, but in providing a universal, scalable framework across all 10 heterogeneous property tracks while significantly reducing physical constraint violations (Table 6).

## D Data Release

We acknowledge the importance of open-source and promise to release our benchmark and method after the acceptance of our paper.

<table><tr><td rowspan=1 colspan=18>Source dataset      Task mapping      Samples   Unique  com- Stored unit    Label conversion / normal- Condition fields     Duplicate / conflict handlingpounds1sys-             izationtems</td></tr><tr><td rowspan=1 colspan=18>AqSolDB         Solubility         9,982     9,982 / 9,982    log10(mol/L)  Directly uses source log- None            No explicit deduplication; conflict-solubility values                        ing measurements retained</td></tr><tr><td rowspan=1 colspan=3>BigSolDB</td><td rowspan=1 colspan=9>Solubility         100,983   1,511 / 100,950   1og10(mol/L)</td><td rowspan=1 colspan=1>LogS(mol/L)storeddi-</td><td rowspan=1 colspan=4>Temperature        No explicit deduplication; c</td><td rowspan=1 colspan=1>onflict-</td></tr><tr><td rowspan=5 colspan=12>CALiSol-23        conductivity        13,302    51 / 13,269     S/cmDESignSolvents Den-  Density          4,023     198 / 4,023     g/cm^3</td><td rowspan=2 colspan=1>rectly as targetRaw conductivity retained;</td><td rowspan=2 colspan=4>ing measurements retainedT               No explicit deduplication; c</td><td></td></tr><tr><td rowspan=1 colspan=1>onflict-</td></tr><tr><td rowspan=1 colspan=1>curated solvent/salt map-</td><td rowspan=1 colspan=4>ing measurements retained</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=2 colspan=1>ping appliedDirectly retained</td><td rowspan=2 colspan=4>Temperature        No explicit deduplication; c</td><td></td></tr><tr><td rowspan=1 colspan=1>onflict-</td></tr><tr><td rowspan=2 colspan=11>DESignSolvents Melt-  Melting temperature   2,259     204 / 2,227</td><td rowspan=2 colspan=1>K</td><td rowspan=2 colspan=1>Directly retained</td><td rowspan=2 colspan=4>ing measurements retainedNone            No explicit deduplication; c</td><td></td></tr><tr><td rowspan=1 colspan=1>onflict-</td></tr><tr><td rowspan=2 colspan=11>ingDESignSolventsVis-  Viscosity          4,042     197 / 4,042cosity</td><td rowspan=2 colspan=1>cP</td><td rowspan=2 colspan=1>Directly retained</td><td rowspan=2 colspan=2>Temperature</td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=2>No explicit deduplication; cing measurements retained</td><td rowspan=1 colspan=1>onflict-</td></tr><tr><td rowspan=6 colspan=11>Drug Solubility      Solubility         27,166    169 / 27,166Electrolytomics Con-  Conductivityductivity</td><td rowspan=1 colspan=1>g/L</td><td rowspan=1 colspan=1>LogS renamed to target; per-</td><td rowspan=1 colspan=2>Temperature</td><td rowspan=1 colspan=2>Predicted columns removed</td><td rowspan=1 colspan=1>; no con-</td></tr><tr><td></td><td rowspan=1 colspan=1>centages converted to frac-</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2>flict aggregation</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=3 colspan=1>Electrolyton</td><td rowspan=3 colspan=2></td><td rowspan=3 colspan=2></td><td rowspan=3 colspan=3></td><td rowspan=3 colspan=1>1og(mS/cm)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=2 colspan=1>Uses  source  log-</td><td rowspan=2 colspan=1>Temperature</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2>No explicit deduplication; c</td><td rowspan=1 colspan=1>onflict-</td></tr><tr><td></td><td rowspan=1 colspan=1>conductivity; temperature</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>ing measurements retained</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=3 colspan=3>Eutectic Diagrams</td><td rowspan=3 colspan=3>Melting point; Fusion</td><td rowspan=3 colspan=5>3,041; 516  3,041  /3,041;516 /516</td><td rowspan=3 colspan=1>K; kJ/mol</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=2 colspan=1>Directly retained</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>None</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>No explicit deduplication; c</td><td rowspan=1 colspan=1>onflict-</td></tr><tr><td rowspan=1 colspan=3>ILThermo</td><td rowspan=1 colspan=3>Electrical conductiv-</td><td rowspan=1 colspan=5>40,904;    479 / 40,735; 699</td><td rowspan=1 colspan=1>S/m; Pa·s</td><td rowspan=1 colspan=1>Retains only positive val-</td><td rowspan=1 colspan=1>Temperature, pressure,</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=3>ity; Viscosity</td><td rowspan=1 colspan=5>75,992    /75,597</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>ues; applies natural loga-</td><td rowspan=1 colspan=1>frequency</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>tion; conflicting repeated</td><td rowspan=1 colspan=1>measure-</td></tr><tr><td rowspan=5 colspan=3></td><td rowspan=5 colspan=3></td><td rowspan=2 colspan=5></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=2>ments retained</td><td rowspan=2 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>kPa and missing frequency</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=3 colspan=5></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=4 colspan=3>logV</td><td rowspan=2 colspan=3>Viscosity</td><td rowspan=1 colspan=2>34,374</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>1,628 / 34,374</td><td rowspan=1 colspan=1>cP</td><td rowspan=1 colspan=1>Uses source logV as target</td><td rowspan=1 colspan=1>T</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>No explicit deduplication; o</td><td rowspan=1 colspan=1>riginal</td></tr><tr><td rowspan=3 colspan=4>Solubility</td><td rowspan=3 colspan=1>251</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>train/test label retained as metadata</td><td></td></tr><tr><td rowspan=2 colspan=3>Medicine Formula-</td><td></td><td></td><td rowspan=2 colspan=1></td><td rowspan=2 colspan=2>6/251</td><td rowspan=2 colspan=1>mg/mL</td><td rowspan=2 colspan=1>Directly retained</td><td rowspan=2 colspan=1>None</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>No explicit deduplication; cing measurements retained</td><td rowspan=1 colspan=1>onflict-</td></tr><tr><td rowspan=1 colspan=3>Miscible Solvent</td><td rowspan=1 colspan=3>Density; Heat of vapor-</td><td rowspan=1 colspan=2>30,061</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>81 / 30,061</td><td rowspan=1 colspan=1>grams/m^3;</td><td rowspan=1 colspan=1>Wide table melted into long-</td><td rowspan=1 colspan=1>None</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>No explicit deduplication; c</td><td rowspan=1 colspan=1>onflict-</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=3>ization; Enthalpy of</td><td rowspan=1 colspan=2>each</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>kcal/mol;</td><td rowspan=1 colspan=1>form property records</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>ing measurements retained</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=2 colspan=3>MixtureSolDB</td><td rowspan=2 colspan=3>Solubility</td><td rowspan=2 colspan=3>175,166</td><td rowspan=1 colspan=2>941 / 174,127</td><td rowspan=1 colspan=1>mole</td><td rowspan=1 colspan=1>Uses         source</td><td rowspan=1 colspan=1>Temperature</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>No explicit deduplication; c</td><td rowspan=1 colspan=1>onflict-</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>fraction</td><td rowspan=1 colspan=1>LogS(mole_fraction)as target</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>ing measurements retained</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=2 colspan=3>MolSets</td><td rowspan=2 colspan=3>Conductivity</td><td rowspan=2 colspan=1>1,076</td><td></td><td rowspan=2 colspan=1></td><td rowspan=2 colspan=2>205 / 990</td><td rowspan=1 colspan=1>1og(S/cm)</td><td rowspan=1 colspan=1>Directly retained; source</td><td rowspan=1 colspan=1>Molality</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>No explicit deduplication; c</td><td rowspan=1 colspan=1>onflict-</td></tr><tr><td></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>weights converted into frac-</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>ing measurements retained</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=7 colspan=3>MONNIST-logV</td><td rowspan=3 colspan=3>Motor octane number</td><td rowspan=3 colspan=1>684</td><td></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>416/676</td><td rowspan=1 colspan=1>None</td><td rowspan=1 colspan=1>RDKit canonical SMILES;mole fractions converted</td><td rowspan=1 colspan=1>None</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Explicit drop duplica</td><td rowspan=1 colspan=1>tes(...)</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=2>for exact duplicates; no av</td><td rowspan=1 colspan=1>eraging</td></tr><tr><td rowspan=1 colspan=1></td><td></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>from percentages to frac-</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>of contradictory replicated</td><td rowspan=1 colspan=1>measure-</td></tr><tr><td rowspan=4 colspan=3>Viscosity</td><td rowspan=4 colspan=3>239,200</td><td rowspan=2 colspan=2>1,644 /213,787</td><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1>CP</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>RDKit canonical SMILES;</td><td rowspan=1 colspan=1>Temperature, p</td><td rowspan=1 colspan=1>ressure</td><td rowspan=1 colspan=2>No explicit deduplication o</td><td rowspan=1 colspan=1>r averag-</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>removed; only positive val-</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2>ments retained</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=7>Olfactory Similarity   Olfactory similarity   865</td><td></td><td rowspan=1 colspan=3>203 / 860</td><td rowspan=1 colspan=1>None</td><td rowspan=1 colspan=1>Directly retained</td><td rowspan=1 colspan=2>None</td><td rowspan=1 colspan=2>No explicit deduplication; c</td><td rowspan=1 colspan=1>onflict-</td></tr><tr><td rowspan=2 colspan=7>Polymer Electrolyte   Conductivity       11,350</td><td></td><td rowspan=2 colspan=3>268 / 10,162</td><td rowspan=1 colspan=1>S/cm</td><td rowspan=1 colspan=1>Uses  source   log-</td><td rowspan=1 colspan=2>Temperature</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>ing me</td><td rowspan=2 colspan=1></td></tr><tr><td rowspan=2 colspan=8>POM-Mix         Olfactory similarity   865</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=3>203 / 860</td><td rowspan=1 colspan=1>score</td><td rowspan=1 colspan=1>Directly retained</td><td rowspan=1 colspan=2>None</td><td rowspan=1 colspan=2>No explicit deduplication; c</td><td rowspan=1 colspan=1>onflict-</td></tr><tr><td rowspan=6 colspan=12>SolProp          Solubility         11,804;6,236     264 / 5,971</td><td rowspan=2 colspan=1>11,805 / 11,804;</td><td rowspan=2 colspan=1>1og10(mol/L)</td><td rowspan=2 colspan=2>Aqueous  task  uses</td><td rowspan=2 colspan=2>Temperature</td></tr><tr><td rowspan=1 colspan=2>No benchmark-side conflict</td><td rowspan=1 colspan=1>aggrega-</td></tr><tr><td rowspan=1 colspan=2></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>aged field logS_aq_avg;</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2>from source preprocessing</td><td rowspan=1 colspan=1></td></tr><tr><td></td><td rowspan=2 colspan=5></td></tr><tr><td rowspan=1 colspan=1>[mol/L]</td></tr></table>

Temperature, pressure, and frequency fields are reported using the names retained in the processed benchmark files.  
“Conflicting repeated measurements retained” means that multiple rows for the same apparent chemical system and condition are usually preserved as distinct benchmark samples unless a dataset-specific script explicitly removes exact duplicates.

Table 8: Chem World source-level data card. “Unique systems” counts distinct chemical-system instances after mapping composition fields and retained context fields. Default split sizes follow the unified random 5-fold protocol and are approximately 70/10/20 for train/validation/test.

<table><tr><td>Method</td><td>Cond.</td><td>Dens.</td><td>Enth.</td><td>Fusion</td><td>Vap.</td><td>Melt.</td><td>MON</td><td>Odor</td><td>Sol.</td><td>Visc.</td><td>Avg.</td></tr><tr><td>RDKit + XGBoost</td><td>0.427</td><td>0.411</td><td>0.506</td><td>0.499</td><td>0.446</td><td>0.508</td><td>0.435</td><td>0.624</td><td>0.482</td><td>0.442</td><td>0.478</td></tr><tr><td>GCN/GIN + Mean Pooling</td><td>0.477</td><td>0.468</td><td>0.537</td><td>0.597</td><td>0.519</td><td>0.588</td><td>0.468</td><td>0.634</td><td>0.535</td><td>0.497</td><td>0.532</td></tr><tr><td>MolFormer + Mean Pooling</td><td>0.392</td><td>0.387</td><td>0.435</td><td>0.519</td><td>0.419</td><td>0.483</td><td>0.388</td><td>0.554</td><td>0.396</td><td>0.477</td><td>0.445</td></tr><tr><td>MolT5 + Mean Pooling</td><td>0.402</td><td>0.383</td><td>0.431</td><td>0.537</td><td>0.435</td><td>0.509</td><td>0.395</td><td>0.558</td><td>0.416</td><td>0.514</td><td>0.458</td></tr><tr><td>MolFormer + DeepSets</td><td>0.373</td><td>0.356</td><td>0.427</td><td>0.525</td><td>0.373</td><td>0.523</td><td>0.358</td><td>0.580</td><td>0.423</td><td>0.392</td><td>0.433</td></tr><tr><td>MolT5 + DeepSets</td><td>0.407</td><td>0.392</td><td>0.468</td><td>0.549</td><td>0.461</td><td>0.547</td><td>0.391</td><td>0.602</td><td>0.457</td><td>0.425</td><td>0.469</td></tr><tr><td>MolFormer + Self-Attention</td><td>0.388</td><td>0.369</td><td>0.425</td><td>0.528</td><td>0.409</td><td>0.508</td><td>0.353</td><td>0.552</td><td>0.408</td><td>0.451</td><td>0.441</td></tr><tr><td>MolT5 + Self-Attention</td><td>0.390</td><td>0.375</td><td>0.452</td><td>0.537</td><td>0.443</td><td>0.536</td><td>0.375</td><td>0.591</td><td>0.440</td><td>0.411</td><td>0.455</td></tr><tr><td>MolFormer + Set Transformer</td><td>0.336</td><td>0.324</td><td>0.392</td><td>0.508</td><td>0.421</td><td>0.499</td><td>0.326</td><td>0.547</td><td>0.386</td><td>0.361</td><td>0.410</td></tr><tr><td>MolT5 + Set Transformer</td><td>0.345</td><td>0.333</td><td>0.401</td><td>0.499</td><td>0.428</td><td>0.506</td><td>0.335</td><td>0.555</td><td>0.395</td><td>0.363</td><td>0.416</td></tr><tr><td>UNIFAC + property-specific correlations</td><td></td><td>0.512</td><td>0.517</td><td></td><td></td><td></td><td></td><td></td><td>0.603</td><td>0.418</td><td>0.513</td></tr><tr><td>COSMO-SAC + property-specific correlations</td><td></td><td>0.441</td><td>0.449</td><td></td><td></td><td></td><td></td><td></td><td>0.493</td><td>0.238</td><td>0.406</td></tr><tr><td>COSMO-RS + property-specific correlations</td><td></td><td>0.416</td><td>0.413</td><td></td><td></td><td></td><td></td><td></td><td>0.467</td><td>0.203</td><td>0.374</td></tr><tr><td>NRTL + property-specific correlations</td><td></td><td>0.405</td><td>0.428</td><td></td><td></td><td></td><td></td><td></td><td>0.452</td><td>0.244</td><td>0.379</td></tr><tr><td>MolFormer + PÁCM</td><td>0.293</td><td>0.281</td><td>0.342</td><td>0.481</td><td>0.385</td><td>0.462</td><td>0.272</td><td>0.514</td><td>0.330</td><td>0.310</td><td>0.367</td></tr><tr><td>MolT5 + PACM</td><td>0.302</td><td>0.292</td><td>0.336</td><td>0.492</td><td>0.392</td><td>0.470</td><td>0.283</td><td>0.508</td><td>0.339</td><td>0.316</td><td>0.373</td></tr></table>

Table 9: Full per-track normalized RMSE on Chem World. For classical thermodynamic models, unsupported tracks are marked as –, and Avg. is computed over applicable tracks.

<table><tr><td>Baseline Model</td><td>Baseline Macro NRMSE</td><td>∆ (vs. Proposed)</td><td>Raw p-value</td><td>Holm padj</td></tr><tr><td>RDKit + XGBoost (Descriptor)</td><td>0.478</td><td>+0.111</td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr><tr><td>GCN/GIN + Mean Pooling (Graph)</td><td>0.532</td><td>+0.165</td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr><tr><td>MolFormer + Mean Pooling</td><td>0.445</td><td>+0.078</td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr><tr><td>MolT5 + Mean Pooling</td><td>0.458</td><td>+0.091</td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr><tr><td>MolFormer + DeepSets</td><td>0.433</td><td>+0.066</td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr><tr><td>MolT5 + Set Transformer</td><td>0.416</td><td>+0.049</td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr><tr><td>UNIFAC (Physical)</td><td>0.513</td><td>+0.146</td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr><tr><td>COSMO-SAC (Physical)</td><td>0.406</td><td>+0.039</td><td>0.018</td><td>0.054</td></tr><tr><td>COSMO-RS (Physical)</td><td>0.374</td><td>+0.007</td><td>0.325</td><td>0.650</td></tr><tr><td>NRTL (Physical)</td><td>0.379</td><td>+0.012</td><td>0.245</td><td>0.530</td></tr><tr><td>MolT5 + PACM (Variant)</td><td>0.373</td><td>+0.006</td><td>0.410</td><td>0.650</td></tr></table>

Table 10: Paired group-level bootstrap tests with Holm correction comparing baseline models against the proposed MolFormer + PACM (Macro NRMSE = 0.367, as reported in Table 2 and Table 9).