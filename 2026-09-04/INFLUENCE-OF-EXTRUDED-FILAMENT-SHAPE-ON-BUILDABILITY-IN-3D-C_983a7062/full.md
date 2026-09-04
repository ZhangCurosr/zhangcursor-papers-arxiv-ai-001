# INFLUENCE OF EXTRUDED FILAMENT SHAPE ON BUILDABILITY IN 3D CONCRETE PRINTING: A GEOMETRY-INFORMED DEEP LEARNING–FEM APPROACH

A PREPRINT

Giacomo Rizzieri<sup>1,∗</sup>, Saif-Ur-Rehman<sup>2</sup>, Jörg F. Unger<sup>2</sup>, Annika Robens-Radermacher<sup>2</sup> <sup>1</sup>Department of Civil and Environmental Engineering, Politecnico di Milano Piazza Leonardo da Vinci, 32, 20133, Milano, Italy <sup>2</sup>Bundesanstalt für Materialforschung und -prüfung (BAM) Unter den Eichen 87, 12205, Berlin, Germany giacomo.rizzieri@polimi.it

September 4, 2026

## ABSTRACT

The geometric morphology of deposited filaments can significantly influence the structural performance and stability of 3D concrete-printed (3DCP) structures. However, most finite element (FEM)-based approaches for buildability assessment represent printed layers as simplified rectangles, potentially limiting predictive accuracy. This study proposes a geometry-informed modelling framework that integrates the deep-learning-based filament shape prediction tool ShapeGen3DCP with a layer-activation FEM approach to investigate the effect of realistic filament geometries on buildability. The framework generates geometry-aware numerical models directly from material and process parameters, eliminating the need for experimental filament characterization or computationally intensive fluid-flow simulations. Validation against experimental data and a parametric study of rectilinear walls demonstrate that extrusion parameters and the resulting filament geometry can significantly influence buildability predictions. Realistic filament representations are particularly important for free-flow deposition, whereas layer-pressing strategies are less sensitive to geometric simplifications. Among the investigated representations, an elliptical approximation provides an effective balance between geometric fidelity and modelling simplicity. When rectangular representations are preferred to enable regular computational meshes for faster simulations, defining their dimensions based on volume conservation improves prediction reliability compared with calibrating them using either the maximum filament width or the interlayer contact width. Overall, the proposed methodology demonstrates the importance of incorporating filament geometry into 3DCP simulations and provides practical guidance for selecting efficient and accurate geometric representations for buildability assessment.

Keywords 3D concrete printing (3DCP) · Additive manufacturing · Buildability assessment · Finite element modelling · Filament geometry · Machine learning

## 1 Introduction

3D Concrete Printing (3DCP) is an emerging additive manufacturing technology that involves the layer-by-layer extrusion of cementitious mortar to fabricate complex objects and structural components. It offers the potential to improve design flexibility, material efficiency, and construction automation by enabling geometrically optimized structures and reducing reliance on conventional formwork.

The geometric fidelity and surface quality of a printed element are strongly governed by the morphology of the deposited layers [1]. Beyond aesthetic considerations, layer geometry and interlayer morphology can significantly influence interlayer bond strength [2], the structural performance of the hardened material [3], and durability-related properties [4], for instance by modifying the ingress pathways available to deleterious agents within the hardened material [5]. By contrast, comparatively little is known about the influence of filament shape on buildability, defined as the maximum number of layers that can be successfully stacked with a given material and set of process conditions before structural collapse occurs. In particular, it remains unclear whether, and to what extent, layer geometry contributes to the mechanical response and overall stability of the printed object during deposition.

This lack of quantitative characterization is also reflected in the geometrical simplifications adopted by solid-based numerical models for simulating buildability in 3D concrete printing. Over the years, numerous approaches have been proposed, ranging from early FE-based formulations employing layer-by-layer activation to evaluate buckling and plastic collapse in rectilinear and cylindrical structures [6, 7], to more advanced modelling strategies extending the activation approach to element-by-element simulations for complex geometries [8], non-planar geometries [9], and large-scale structures [10]. These modelling frameworks have been further advanced through the incorporation of increasingly sophisticated constitutive formulations, accounting for phenomena such as elasto-plasticity with hardening [11, 12], finite-strain viscoelasticity [13], aging and damage evolution [14], coupled multi-physics interactions [15], and environmental effects on material behaviour [16].

However, despite significant modelling advancements, most solid-based FEM models still rely on the simplifying geometrical assumption that extruded filaments can be represented as idealized rectangular geometries, with dimensions typically derived from the nozzle size or from preliminary measurements of the deposited filament width when available. While the errors introduced by this hypothesis may remain negligible in cases involving extrusion through rectangular nozzles with stiff materials, it has been suggested that, for circular nozzles, this may represent a significant oversimplification, potentially leading to consistently reduced predictive accuracy [17, 18].

Accurately representing the actual filament geometry and morphology would therefore require prior knowledge of the deposited shape. Although this information can be obtained through preliminary extrusion tests, such an approach is often impractical and time-consuming, especially under complex printing conditions, and conflicts with the predictive objective of numerical modelling frameworks. A possible alternative, given that the relationships between process parameters, material properties, and resulting filament shape are non-trivial, is to rely on advanced fluid-based numerical models, which have seen significant development in recent years to simulate extrusion and layer deposition processes. In particular, continuum fluid models implemented within the Finite Volume Method (FVM) [19] and the Particle Finite Element Method (PFEM) [20, 21] have demonstrated high accuracy in predicting filament shapes for single and also few superimposed layers.

These high-fidelity models can resolve the filament scale in detail, but are generally computationally demanding, making them unsuitable for reproducing the entire printing process within a single computational framework. This limitation arises because many available solvers are developed for computational fluid dynamics (CFD) rather than solid mechanics, relying on constitutive models and numerical schemes that are not optimized to efficiently capture the evolving strength of fresh concrete and the structural response of printed elements during deposition.

For this reason, if the objective is to predict the entire 3D printing process and assess buildability, they are typically combined with complementary approaches (e.g., layer-activation FEM) or extended through advanced unified formulations. For example, [18] employed PFEM to simulate the extrusion of individual layers, which were subsequently superimposed and progressively activated to evaluate the buildability of a rectangular structure. Similarly, [22] introduced a unified elasto-viscoplastic fluid–solid constitutive model, but with the aim of extending the PFEM framework for simulating 3D printing across multiple process scales, from material extrusion to buildability assessment. In contrast, [23] proposed a Smoothed Particle Hydrodynamics (SPH)-based model to capture the extrusion process, combined with a finite element method (FEM) approach incorporating layer activation to evaluate buildability. Nonetheless, these developments remain relatively recent and still involve significant computational costs, together with a high level of expertise required for model implementation and application.

These limitations highlight the need for alternative approaches, beyond experimental measurements and fluid-dynamicsbased simulations, to enable rapid estimation of filament shape in 3DCP and potentially enhance the accuracy of standard FEM-based buildability models. For instance, by systematically analysing the influence of printing conditions and material characteristics, design charts have been developed to directly estimate filament width and height for free-flow deposition [24] and layer-pressing strategies [25] in 3DCP.

Data-driven methods have also started emerging to establish relationships between process parameters, material properties, and filament morphology. For example, regression-based machine learning models have been employed to predict filament width, height and contact width from printing parameters [26]. However, the applicability of the proposed model remains limited to layer-pressing printing conditions and the specific material composition investigated. Based on both material and process parameters, [27] developed a support vector machine learning algorithm trained to predict simplified cross-sectional representations of filament shapes (i.e., flat-topped polygonal profiles) for rectangularnozzle 3D concrete printing at corners. Recently, ShapeGen3DCP [28] was also proposed as a highly general deep learning framework for predicting layer shapes in 3D concrete printing with circular nozzles. The model takes as input both material rheological properties, described through the Bingham’s parameters, and key process parameters, and provides an instantaneous prediction of the complete cross-sectional profile of single and double layers, encoded using Fourier descriptors. Since the entire profile geometry is predicted, key geometric characteristics such as filament height, width, and interlayer contact can be directly extracted through simple post-processing operations.

The aim of this paper is to integrate data-driven filament shape prediction approaches into existing FEM-based buildability models for 3DCP, improving their predictive reliability by overcoming the limitations associated with simplified geometric representations. In the following, this integration is referred to as a geometrically informed modelling approach. Specifically, a workflow is proposed that couples the filament-shape prediction capabilities of ShapeGen3DCP with a layer-activation finite element (FEM) framework for buildability assessment. The framework is used to address three key questions: (i) how filament geometry affects buildability predictions; (ii) how this influence depends on printing strategy, process parameters, and material behaviour; and (iii) what level of geometric fidelity is required to obtain accurate predictions without excessive computational cost.

The article is organized as follows. Section 2 presents the overall workflow and details its three main components: (1) ShapeGen3DCP, a deep-learning-based tool for predicting filament geometries from material properties and printing parameters; (2) a mesh-generation procedure that combines the predicted filament shapes with the printing toolpath to obtain a geometrically accurate representation of the printed object; and (3) an elastic and elasto-plastic FEM framework for buildability assessment up to failure, followed by post-processing of the results. Section 3 presents the results. The workflow is first validated against experimental data on single-layer walls exhibiting elasto-plastic buckling failure, showing excellent agreement when realistic filament geometries are considered. A parametric study is then conducted on rectilinear walls, considering six printing strategies, five geometric discretizations, and three material models. The results quantify the influence of filament shape on buildability and reveal how this influence depends on both process conditions and material behavior. In particular, layer-pressing regimes are found to be less sensitive to geometric simplifications than free-flow deposition conditions. It is also recognized that elliptical cross-sections often provide a good balance between accuracy and computational efficiency, while a volume-conservation-based dimensioning approach is proposed to improve the accuracy of rectangular approximations. The framework is subsequently applied to cylindrical geometries, confirming the trends observed for walls. Finally, Section 4 summarizes the main findings and presents practical guidelines for incorporating filament geometry into buildability analyses, including a simplified design chart for single-layer 3D-printed walls.

## 2 Workflow description

This section presents the automated workflow for conducting geometrically informed buildability simulations in extrusion-based 3D printing. The framework is structured into three sequential and interdependent phases, schematically illustrated in Figure 1. Each phase is defined by specified inputs and outputs, enabling consistent data transfer across the pipeline.

1. The first phase addresses the generation of filament cross-sectional geometries for single- and double-layer deposition. These geometries represent the outcome of the coupled extrusion and deposition processes. In this work, the cross-sectional profile is predicted using ShapeGen3DCP [28], a recently developed deep learning framework capable of rapidly estimating filament shapes from process parameters (e.g., extrusion rate, nozzle speed) and material properties in the fluid state. This approach enables efficient and physically informed reconstruction of deposited filament geometries without resorting to computationally expensive multiphysic simulations.

![](images/ac4bf6e38210f0ea3971e46681b2c0865b7a0865e7d4db0d0279abbe7b20f344.jpg)

![](images/9890177cb67115ff0001cf55abf3d95928e2fc28aa941d5605da8245fa38d330.jpg)

![](images/0f4a625ed70d77ccd4f3845f014d8b4583917f9836fbdf6a6e6d5164dc49da04.jpg)  
Figure 1: Automated workflow integrating machine-learning based filament shape prediction (ShapeGen3DCP) with robust layer-by-layer mesh generation (gmsh) and buildability finite element simulations (in DOLFINx).

2. The predicted cross-sectional information is then passed to the second phase, where the goal is to construct a mesh representation of the overall printed object. At this stage, it is not necessary to strictly preserve the exact geometry predicted by ShapeGen3DCP. Instead, the focus shifts to creating simplified but mechanically meaningful approximations of the filament cross-section that can be used as building blocks for the layered computational mesh. For instance, the ShapeGen3DCP outputs can be used to define idealized shapes such as rectangular or elliptical profiles. This representative cross-section is then swept along the prescribed toolpath and replicated vertically to form a three-dimensional model of the printed object. Finally, the model is discretized into a tetrahedral finite element mesh using Gmsh, making it suitable for subsequent numerical analysis.

3. In the third phase, the generated computational domain is imported into a simulation environment for mechanical assessment. Buildability is evaluated through a finite element analysis framework in which layers are progressively activated to mimic the sequential nature of the printing process. A quasi-static elastoplastic formulation is employed to capture the accumulation of deformation and the onset of failure as the structure evolves during fabrication.

The pipeline is implemented in Python and managed using the rule-based workflow management system Snakemake [29], ensuring reproducible and automated execution of all processing steps. Each phase is executed in an isolated environment tailored to its dependencies – for instance, PyTorch for machine-learning-based filament cross-section prediction, Gmsh for geometry and mesh generation, and the DOLFINx [30] framework for finite element analysis. In this way, completed tasks are tracked automatically, and failed or outdated steps can be selectively recomputed without rerunning the entire pipeline. Still, the modular architecture allows individual workflow stages to be executed independently, facilitating debugging and the verification of intermediate results. The implementation is available on Zenodo [31].

## 2.1 ShapeGen3DCP: machine-learning predictions of filament shape outline

Accurate prediction of single- and multi-layer filament cross-sectional shapes has been extensively investigated in the literature using fluid-based finite volume [19] and finite element [21, 24] approaches. While these techniques yield high-fidelity results, they are computationally intensive – even for a few layers – and require substantial setup, limiting their use in integrated or real-time workflows.

To address these limitations, the present study leverages ShapeGen3DCP [28], a machine learning framework recently introduced in the literature for the prediction of filament cross-sectional shapes. The model takes as input both material properties in the fluid state – namely density $( \rho )$ , yield stress $\left( \tau _ { 0 } \right)$ , and plastic viscosity (µ) – and key process parameters, including nozzle diameter $( \phi _ { n } )$ , nozzle height $( h _ { n } ) .$ , printing velocity $( v _ { P } ) .$ , and flow velocity $( u _ { f } )$

ShapeGen3DCP was trained on a numerically generated dataset obtained using a validated Particle Finite Element Method (PFEM) for simulating additive manufacturing with cementitious materials [21]. The dataset includes filament geometries for approximately 180 different combinations of material and process parameters. A deep neural network maps these input parameters to a parametric representation of the filament cross-sections using Fourier descriptors, which inherently preserve geometric properties such as smoothness, closure, and symmetry (Figure 2). Importantly, validation against experimental data from multiple independent sources and the full-order PFEM model showed shape prediction errors generally within 1–10% for key geometric descriptors (e.g., maximum width, interlayer contact, height, cross-sectional area) with only a few cases reaching approximately 15%, demonstrating the accuracy and robustness of ShapeGen3DCP for advanced applications. For a detailed description of ShapeGen3DCP architecture, training procedure and validation the reader is referred to [22].

The model can predict both single-layer and two-layer extrusion configurations. In this study, the focus is on the latter, as two-layer cross-sections provide a detailed representation of interlayer interaction and merging behaviour. In addition, ShapeGen3DCP enables the post-processing of reconstructed filament shapes to extract reduced geometric descriptors, as illustrated in Figure 2.

For two-layer cross-sections, the relevant descriptors include the bottom layer’s maximum width $w _ { 1 }$ , the interlayer contact length $w _ { c } ,$ the bottom layer height $h _ { 1 }$ , the top layer height $h _ { 2 } .$ , the total height $h _ { t o t } = h _ { 1 } + h _ { t o t }$ , and the total cross-sectional area $A ,$ computed numerically using the shoelace formula applied to the reconstructed contour.

![](images/f3e15bef902a88729da328c65322efc87a1303ed8f63dafaa9eb68ad4e24e830.jpg)  
Figure 2: ShapeGen3DCP overview for the two-layer case: material properties in the fluid state and process parameters are provided as inputs, yielding a smooth, closed, symmetric cross-sectional profile. The drawing further highlights the key geometrical features that can be extracted from the resulting shape.

The interlayer contact length $w _ { c }$ and the bottom layer height $h _ { 1 }$ are determined by locating the interlayer contact point $( x _ { c } , y _ { c } )$ , defined as the point along the contour where the first derivative – approximated via finite differences – changes sign. The horizontal coordinate of this point gives half of the contact length, so that $w _ { c } = 2 x _ { c } ,$ while the vertical coordinate directly provides the bottom layer height, i.e. $h _ { 1 } = y _ { c }$

## 2.2 Filament shape post-processing and meshing strategy

Numerical models for 3DCP buildability assessment often rely on simplified assumptions regarding layer geometry. This simplification is typically driven by limited prior knowledge of the actual filament cross-section, as well as the need to reduce mesh complexity and computational cost. However, as demonstrated below, filament shape can significantly influence buildability predictions. It is therefore essential to adopt a representative cross-section that achieves an appropriate balance between geometrical fidelity and model complexity. Furthermore, even in the absence of experimental measurements, filament geometry can now be efficiently estimated using data-driven approaches, such as the method introduced in the previous section 2.1.

## 2.2.1 Representative layer shape

In this context, the present study adopts – as a reference – the two-layer cross-sectional geometry predicted by ShapeGen3DCP. The bottom layer is of particular interest, as it represents a filament that has not only been deposited but has also undergone deformation due to the localized pressure increase caused by material inflow during the extrusion of the subsequent layer. Starting from this configuration, different representative layer geometries are derived, corresponding to increasing levels of geometric fidelity, as illustrated in Figure 3.

• Full-width rectangle (rect. width). The cross-section is idealized as a rectangle of width equal to the bottom layer’s maximum width $( w = w _ { 1 } )$ ) and height equal to the bottom-layer height $( h = h _ { 1 } ) ;$ ; this simplification generally leads to an overestimation of the actual extruded material volume.

• Contact-width rectangle (rect. contact). The cross-section is idealized as a rectangle of width equal to the interlayer contact length $( w = w _ { c } )$ and height equal to the bottom-layer height $( h = h _ { 1 } ) ;$ ; this assumption generally underestimates the actual extruded material volume.

• Area-equivalent rectangle (rect. adapted). The cross-section is idealized as a rectangle with prescribed height $( h = h _ { 1 } )$ , while the width is adjusted to preserve the cross-sectional area of the point-derived profile (covered in the following), i.e., $w _ { \mathrm { a d } } = A _ { p o i n t s } / h _ { 1 }$ . This construction ensures exact conservation of the extruded material volume.

• Elliptical-capped rectangle (ellipse). The cross-section is modelled as a composite geometry consisting of a central rectangular core of width $w _ { c }$ and height $h _ { 1 }$ , flanked by two lateral regions defined by elliptical segments. The overall width of the cross-section is $w _ { 1 }$

• Point-derived profile (points). The cross-section is constructed directly from the ShapeGen3DCP point-wise profile prediction. The bottom layer is considered as it already encodes deformation effects induced by subsequent layer extrusion and deposition. To remove the influence of the rigid substrate, only the upper portion of the profile is retained (points with $y > 0 . 5 h _ { 1 } )$ . This partial profile is then mirrored about the y-axis to obtain a symmetric cross-section suitable for vertical stacking. The final geometry is a closed polyline with a user-defined number of points, bounded by $w _ { 1 }$ in width and $h _ { 1 }$ in height, with horizontal top and bottom segments of length $w _ { c }$ and an enclosed area of $A _ { p o i n t s }$

![](images/86098488ffdda74106122a26a6de4ab43ae92723ed428db060e90654f7e0a58a.jpg)  
Figure 3: Different representative layer geometries with increasing geometric fidelity: (a) rect. width, (b) rect. contact, (c) rect. adapted, (d) ellipse, and (e) points.

## 2.2.2 Mesh generation

The geometrical descriptions introduced in the previous section are converted into finite element meshes through an automated pipeline based on Gmsh. The adoption of Gmsh is motivated by its capability to construct geometries programmatically, handle complex curve definitions, and generate consistent three-dimensional unstructured meshes from parametrized inputs. The meshing procedure follows a sequential and modular workflow, which is consistent across all filament representations.

1. Definition ofthe cross-sectional outline. The starting point is the definition of a closed two-dimensional profile representing the filament cross-section. Depending on the chosen representation, this profile is either generated analytically (rectangular and elliptical cases) or directly derived from processed point data (point-based case). For the rectangular and elliptical geometries, a set of characteristic points is first defined from the prescribed geometric parameters (layer height, layer width, and contact width). These points are then connected to form the cross-section: by straight segments for the rectangular variants (rect. width, rect. contact, rect. adapted), and by a combination of straight segments and elliptical arcs along the lateral boundaries for the elliptical case (ellipse), allowing a smoother representation of the filament curvature.

In all cases, the resulting set of lines and curves defining the closed profile is assembled into a curve loop, which is subsequently used to define a planar surface representing the single-layer cross-section.

2. Generation ofa single three-dimensionalfilament. Once the two-dimensional surface is defined, it is transformed into the filament three-dimensional volume. Two alternative strategies are considered, depending on the target geometry.

For prismatic wall-like structures, the surface is extruded along the printing direction. The extrusion length corresponds to the filament length, and the discretization along this direction can be explicitly prescribed.

For axisymmetric configurations (e.g., cylindrical geometries), the same two-dimensional cross-section is instead revolved around a prescribed axis. The revolution is performed over a full angular span, generating a closed three-dimensional layer. Here, the same ShapeGen3DCP cross-section based on straight filaments is used directly, neglecting potential effects of the curvature on the filament geometry and volume.

![](images/df454496e6e29ff7d4a999fbe2cb48c8e5860083dc795e0a14c0c7721393bb65.jpg)  
(a) rect. width

![](images/1bd104eddf62e72e31f80aa72b38618db50c3daa752f78ba9cbf42e01750838e.jpg)  
(b) rect. adapted

![](images/fce2a611817c63e5b18a5fa57998397834ecc5e2e52a5c859e18f98d96af9cd3.jpg)  
(c) rect. contact

![](images/9fb4aa454b431fe0c3bff954888f9d8aa73309bf95a1d0f4ba1960470fe8a69f.jpg)  
(d) ellipse

![](images/a636a5d39febe5cd1f800b90be53acea0ca229332e9eadfbae765e5b52a8723f.jpg)  
(e) points  
Figure 4: Meshed computational domains generated in Gmsh by extruding reference layer geometries at varying accuracy levels and subsequently stacking the resulting layers vertically.

3. Layer-by-layer volume domain generation. The extrusion (or revolution) operations described above can be applied iteratively, with each step beginning from a surface profile that has been translated along the z-axis by the inter-layer distance. This process builds the final stacked geometry layer by layer, ensuring consistent vertical alignment and precise definition of the three-dimensional domain.

4. Mesh generation and layer-by-layer activation. After completion of the geometric construction, a threedimensional unstructured mesh is generated within Gmsh. In this work, only tetrahedral elements are used, providing robustness for arbitrary geometries without imposing constraints associated with structured discretizations. Figure 4 shows representative meshes corresponding to the different cross-sectional profiles defined in Section 2.2.1.

A key aspect of this approach is the preservation of geometrical interfaces between consecutive layers. As each layer is created as an independent volume, interlayer boundaries are naturally embedded in the mesh. This is particularly beneficial for finite element simulations, as it allows a direct and consistent implementation of layer-by-layer activation without additional geometric manipulation.

## 2.3 Simulation framework

The simulation framework adopts the general assumption, first proposed in [6], that freshly deposited 3D printable concrete can be modelled as a continuum solid. In this work, owing to the proposed integrated workflow, the effects of material flow during extrusion have already been accounted for in the computational mesh through the shape of the filaments. The simulation framework is implemented using the open-source finite element software DOLFINx [30] and uses the framework presented in [32] to integrate the user-defined constitutive model.

## 2.3.1 Governing equations and numerical formulation

The structural response during printing is governed by the quasi-static balance of linear momentum,

$$
\nabla \cdot \left[ \chi ( \mathbf { x } , t ) \pmb { \sigma } ( \mathbf { x } , t ) \right] + \chi ( \mathbf { x } , t ) \rho \pmb { g } = \mathbf { 0 } ,\tag{1}
$$

where $\sigma$ is the Cauchy stress, $\rho$ the constant material density and $\textbf {  { g } }$ the gravity vector. The full mesh of the target geometry is built a priori (Section 2.2.2), and the progressive layer-by-layer deposition is reproduced through a timedependent pseudo-density field $\chi ( \pmb { x } , t ) \in [ 0 , 1 ]$ defined over the entire domain, which scales both the stress contribution – and thereby the stiffness, since the consistent tangent is $\chi \partial \pmb { \sigma } / \partial \pmb { \varepsilon } -$ and the body force in eq. (1). For $\chi = 0$ , the material contributes neither stiffness nor weight; for $\chi = 1$ it responds as a fully active solid. In the sub-domain corresponding to the i-th deposited layer, $\chi$ rises continuously from 0 to 1 over the associated layer-activation interval.

Fresh, early-age concrete often exhibits a comparatively low internal friction angle (on the order of a few degrees) after deposition [33, 34], leading to negligible difference between pressure-sensitive (e.g. Mohr-Coulomb) and pressureindependent yield criteria in this regime [33]. Furthermore, fluid-based simulations of the deposition process standardly rely on pressure-independent viscoplastic (Bingham-type, von Mises) formulations [18, 21]. For these reasons and since the focus of the work is the layer shape influence, the deposited material is described by a $J _ { 2 }$ elasto-plastic constitutive law with non-linear isotropic (saturation-type) hardening, following the formulation introduced in [11]. With the additive decomposition $\varepsilon = \bar { \varepsilon ^ { e } } + \varepsilon ^ { p }$ of the strain into elastic and plastic parts, the model is derived from the Helmholtz free energy per unit volume, additively split into an elastic and a stored plastic (hardening) contribution,

$$
\psi ( \varepsilon ^ { e } , \alpha ) = \hat { \psi } ^ { e } ( \varepsilon ^ { e } ) + \hat { \psi } ^ { p } ( \alpha ) = { \scriptstyle { \frac { 1 } { 2 } } } \kappa \varepsilon _ { v } ^ { 2 } + \mu \varepsilon ^ { e ^ { \prime } } ; \varepsilon ^ { e ^ { \prime } } + ( \tau _ { \infty } - \tau _ { 0 } ) \left( - \frac { 1 } { \omega } + \alpha + \frac { 1 } { \omega } e ^ { - \omega \alpha } \right) ,\tag{2}
$$

where $\kappa$ and $\mu$ are the bulk and shear moduli, $\varepsilon _ { v } = \operatorname { t r } \varepsilon ^ { e }$ the volumetric elastic strain and $\varepsilon ^ { e \prime }$ its deviatoric part, α the accumulated plastic strain, $\tau _ { 0 }$ and $\tau _ { \infty }$ the initial and saturation yield limits, and ω the hardening factor controlling the transition between $\tau _ { 0 }$ and $\tau _ { \infty }$ . Standard Coleman–Noll arguments give the stress $\pmb { \sigma } = \partial \psi / \partial \pmb { \varepsilon } ^ { e ^ { - } } = \kappa \varepsilon _ { v } \mathbf { 1 } + 2 \mu \varepsilon ^ { e ^ { \bar { \prime } } }$ and the hardening stress conjugate to $\alpha , q = \partial \hat { \psi } ^ { p } / \partial \alpha = \left( \tau _ { \infty } - \tau _ { 0 } \right) \left( 1 - e ^ { - \omega \alpha } \right)$ . The yield function then reads,

$$
\begin{array} { r } { \phi ( \pmb { \sigma } , \alpha ) = \| \pmb { \sigma } ^ { \prime } \| - \sqrt { \frac { 2 } { 3 } } \left[ \tau _ { 0 } + \left( \tau _ { \infty } - \tau _ { 0 } \right) \left( 1 - e ^ { - \omega \alpha } \right) \right] , } \end{array}\tag{3}
$$

where $\pmb { \sigma } ^ { \prime }$ is the deviatoric Cauchy stress. The flow rule is associative, giving $\dot { \varepsilon } ^ { p } = \lambda n$ and $\dot { \alpha } = \sqrt { 2 / 3 } \lambda$ , with the unit deviatoric flow direction $\pmb { n } \overset { \cdot } { = } \pmb { \sigma } ^ { \prime } / \lVert \pmb { \sigma } ^ { \prime } \rVert$ and the plastic multiplier $\lambda \geq 0$ subjects to the Karush–Kuhn–Tucker conditions $\lambda \geq 0 , \phi \leq 0 , \lambda \phi = 0$

For the numerical solution, the quasi-static problem eq. (1) is transformed into its weak form and discretized in space using the finite element method with appropriate trial and test functions. Owing to the nonlinear constitutive response, the resulting system of equations is solved iteratively at each load step of the quasi-static analysis using a standard Newton–Raphson solver. At each quadrature point and each load step, the constitutive update is integrated implicitly (backward Euler) by a radial-return scheme: starting from the elastic trial stress, the yield criterion is checked and, if violated, a local Newton iteration on λ enforces consistency $( \phi = 0 )$ ; the stress, plastic strain and hardening variable are then updated and the consistent algorithmic tangent is returned to the global Newton–Raphson solver to preserve quadratic convergence.

Because the layer-by-layer build-up produces moderate-to-large rotations of the deposited material, the simulations are carried out within an updated Lagrangian framework with objective stress rates [35], Saif-Ur-Rehman et al. (2026, under review). The reference configuration is advanced to the current deformed configuration at the end of every converged load step, so that all subsequent quantities are referred to the latest mesh. Within each increment, the Cauchy stress is transported objectively through the Jaumann (co-rotational) rate: the skew-symmetric part of the displacement-gradient increment, $\Delta W \dot { = } \operatorname { s k w } ( \mathbf { \bar { \nabla } } \Delta \mathbf { { u } } )$ , defines a Hughes-Winget incremental rotation tensor, which is used to push the previous stress state forward before the constitutive update is evaluated at the midpoint configuration. This procedure prevents spurious stress accumulation under rigid-body rotations and keeps the material state consistent with the structural deformation.

Finally, the structural build-up and ageing of fresh concrete during printing are taken into account by letting each material parameter $P \in \{ E , \nu , \tau _ { 0 } , \tau _ { \infty } , \omega \}$ (with the elasticity parameters: Young’s modulus E, Poisson ratio ν) vary linearly with the simulation time,

$$
\begin{array} { r } { P ( t ) = P _ { 0 } + A _ { P } t , } \end{array}\tag{4}
$$

with initial value $P _ { 0 }$ at $t = 0$ and evolution rate $A _ { P } .$ . Both quantities are chosen from compression tests performed on samples of different ages, as discussed in Section 3.1, providing the gradual gain in stiffness and strength necessary to reproduce the experimentally observed buildability. The linear evolution is a common assumption in buildability simulations [36, 6, 8, 18, 13]. Owing to the relatively short time scale relevant for the investigated buildability analysis, in the order of minutes, the material remains in the dormant phase, such that a first-order linear approximation is sufficient [37].

## 2.3.2 Simulation set-up for buildability analysis

For the buildability simulations considered in the following, the layer cross-section is defined in the $x - y$ plane and extruded in the z-direction, as illustrated for the wall example in Figure 5. The base of the first layer is fixed in all spatial directions. Gravity $( g = 9 . 8 1 \mathrm { m / s ^ { 2 } } )$ is applied in the negative y-direction. The load contribution of each newly deposited layer is increased linearly over the activation time of one layer, $t _ { l } = l _ { l } / v _ { p }$ , where $l _ { l }$ denotes the layer length and $v _ { p }$ the printing velocity. This linear ramp accounts for the fact that, in reality, material is deposited continuously over the extrusion length, while in the simulation, all elements of a layer are activated simultaneously. Thus, the increase in self-weight during deposition is approximated within the layer activation interval. The load step assumes a dual role in the present formulation. It constitutes the physical increment over which the age-dependent stiffness and the deposition of self-weight are advanced. Through the load imposed within that interval it further prescribes the strain and rotation increments upon which the accuracy of the rate-independent return mapping and of the objective stress update depends.

![](images/04f96b69f5578c1918c3d40052fce64369ec2117ffc8fb25dd80a1e0360d5801.jpg)  
Figure 5: Geometry and boundary conditions for buildability simulation at the example of printing a wall.

To trigger the first buckling mode, a linear geometric imperfection is introduced over the height of the structure in the out-of-plane direction (x). The imperfection amplitude is defined per layer as a multiple α of the layer width w<sub>i</sub> (w<sub>1</sub>, $w _ { c } ,$ , or $w _ { a d } ,$ depending on the adopted filament representation defined in Sect. 2.2).

Failure of the printed structures is identified using a displacement-based criterion. The simulation is terminated when any active node reaches an out-of-plane displacement exceeding a prescribed threshold. In the following, a threshold of 0.1 times the layer width is used for the wall simulations based on a comparison with the analytical solution in the elastic case. For cylinder simulations, a threshold of one layer width is used as in [11]. Other stopping criteria based on eigenvalue analysis are discussed in Saif-Ur-Rehman et al. (2026, under review).The corresponding number of activated layers at failure is then computed from the ratio of the failure time $t _ { \mathrm { f a i l } }$ to the layer activation time:

$$
n _ { \mathrm { a c t } } = { \frac { t _ { \mathrm { f a i l } } } { t _ { l } } } .\tag{5}
$$

Finally, it is worth mentioning that different filament representations result in different total structural volumes, which in turn affect the applied load and, consequently, the simulation results. Only the points and rect. adapted representations are volume-preserving, whereas the others aim to preserve the geometrical characteristics predicted from ShapeGen3DCP. The material density can be adjusted so that all structures have the same total weight. The effect of such a density adjustment on the buildability is discussed in the validation example in Sect. 3.1.

## 3 Results

This section first evaluates the proposed workflow by examining its predictions against published experimental observations from the literature, assessing its ability to predict buildability. The framework is then employed to investigate the influence of filament shape and its geometrical discretization on the buildability of rectilinear walls under different material behaviours, ranging from purely elastic to various elasto-plastic regimes. Finally, buildability is assessed for cylindrical structures to identify the combined effects of filament shape and global structural geometry.

## 3.1 Validation of geometry-informed approach

As an initial assessment of the methodology proposed in Section 2, the simulation results are examined against printing experiments reported by Tripathi et al. [12]. This study was selected because it provides all the data required for the present analysis, including the printing parameters and the fluid and solid material properties. Moreover, it considers multiple material compositions and reports extensive buildability experiments on simple wall geometries, providing a suitable basis for assessing the predictions of the proposed framework for two distinct materials.

![](images/c9130aab81eb49fa0e2285aa45bf3acee137690c996ba974a0830fb4dbec102a.jpg)  
(a) Material LM 30

![](images/76d6816f4d64361b41cd66a504a52d3a4284a2eb2d1e7030ffa9e3865124786a.jpg)  
(b) Material LSM 30  
Figure 6: Stress-strain curves from a uniaxial cube simulation under compression compared with the experimentally measured stress-strain curves from [12].

First, the parameters of the elastoplastic material model with nonlinear hardening are chosen using the stress-strain curves obtained from compression tests on samples of different ages reported by Tripathi et al. For simplicity, these compression tests are reproduced numerically as uniaxial, displacement-controlled compression simulations. The material parameters are selected to match the reported stress-strain curves for the two materials LM 30 and LSM 30 from Tripathi et al. for two ages, assuming the common linear time evaluation in between (see eq. (4)). Figure 6 compares the experimental and simulated stress-strain responses and shows close agreement. The material parameters used in the subsequent analyses are summarized in Table 1.

Table 1: Material parameters and evolution rates for the elastoplastic material law with nonlinear hardening (Young’s modulus $E ,$ Poisson ratio $\nu ,$ initial yield stress $\tau _ { 0 } .$ , final yield stress $\tau _ { \infty }$ , hardening factor ω and their corresponding evolution rates $A _ { p } )$
<table><tr><td>material</td><td> $E$   $\mathrm { ( P a ) }$ </td><td> $A _ { E }$  (Pa/s)</td><td> $\nu$  (-)</td><td> $A _ { \nu }$  (1/s)</td><td> $\tau _ { 0 }$  (Pa)</td><td> $A _ { \tau _ { 0 } }$  (Pa/s)</td><td> $\tau _ { \infty }$  (Pa)</td><td> $A _ { \tau _ { \infty } }$  (Pa/s)</td><td> $\omega$  (-)</td><td> $A _ { \omega }$  (1/s)</td></tr><tr><td>LM 30 LSM 30</td><td>989999.4 2509998.4</td><td>22.11 650.</td><td>0.3 0.3</td><td>0. 0.</td><td>1210. 3490.</td><td>0.25 0.583</td><td>7000. 21000.</td><td>3.1 9.</td><td>10. 10.</td><td>0. 0.</td></tr></table>

Second, the complete simulation workflow is executed. The filament cross-sectional geometry is predicted using ShapeGen3DCP for a nozzle diameter of 20 mm, a nozzle height of 15 mm, a printing velocity of $v _ { p } = 1 5 \mathrm { m m / s } .$ , a flow velocity of $u _ { f } = 1 7 . 9 2 \mathrm { m m } / \mathrm { s } ,$ and a density of $2 1 1 2 . 5 \mathrm { k g } / \mathrm { m } ^ { 3 }$ , as reported by Tripathi et al. The shear yield stress is derived from the initial yield stress by dividing it by $\sqrt { 3 }$ according to von Mises assumption, and the viscosity is assumed to be 15 Pa·s. For LM 30, the resulting cross-section yields a layer width of 30.3 mm, an interlayer contact length of 22.13 mm, and a layer height of 11.7 mm. For LSM 30, the corresponding values are 27.63 mm, 19.22 mm, and 12.21 mm, respectively. These dimensions are consistent with the approximate generic layer size of 25 mm × 15 mm reported by Tripathi et al., although no material-specific distinction is provided there.

Subsequently, meshes of the 200 mm long wall specimens are generated using approximately four linear tetrahedral finite elements over one layer height for each of the filament representations introduced in Section 2.2. For each approach, the printing simulation described in Section 2.3 is carried out with fixed displacements at the base, an imperfection magnitude of $\alpha = 0 . 0 0 0 2 5$ , and a load step of 0.25 s, which was determined through a convergence study.

As discussed in Sect. 2.3, the different filament representations result in different total volumes. Only the points and rect. adapted representations are volume-preserving, whereas the remaining approaches preserve specific geometric characteristics predicted by ShapeGen3DCP. To assess the influence of this difference, two simulation set-ups are considered. In the first set-up, the material density is adjusted such that all structures have the same total weight and therefore represent the same amount of deposited material. The points representation is used as the reference case, and the density is scaled by the factor $A _ { \mathrm { p o i n t s } } / A _ { i }$ , with i ∈ {rect. width, rect. contact, rect. adapted, ellipse, points}. In the second set-up, no density adjustment is applied. Consequently, the different filament representations lead to different total structural masses, reflecting the volume differences inherent to the respective geometric approximations.

## 3.1.1 Validation with experimental data

Table 2 summarizes the number of activated layers at failure, as defined by Eq. (5), for both set-ups (with and without density adjustment) and compares these values with the experimentally observed numbers of printable layers reported by Tripathi et al. Overall, good agreement is obtained between simulation and experiment, which supports the validity of the proposed framework.

Table 2: Comparison of activated layers at failure for the two materials and varying filament representations. The simulations are run with (w/) and without (w/o) density adjustment. For validation, the number of layers at failure reported in [12] is given.
<table><tr><td colspan="2"></td><td rowspan="2">rect. width</td><td rowspan="2">rect. adapted</td><td rowspan="2">rect. contact</td><td rowspan="2">ellipse</td><td rowspan="2">points</td><td rowspan="2">exp. from [12]</td></tr><tr><td>material</td><td>density adjustment</td></tr><tr><td>LM 30</td><td>w/</td><td>13.69</td><td>11.92</td><td>9.90</td><td>11.60</td><td>11.65</td><td>8 ± 1</td></tr><tr><td>LM 30</td><td>w/o</td><td>13.04</td><td>11.92</td><td>10.71</td><td>11.40</td><td>11.67</td><td>8± 1</td></tr><tr><td>LSM 30</td><td>w/</td><td>22.08</td><td>19.27</td><td>15.50</td><td>18.56</td><td>18.60</td><td>16 ± 2</td></tr><tr><td>LSM 30</td><td>w/o</td><td>20.10</td><td>19.27</td><td>18.23</td><td>17.90</td><td>18.62</td><td>16 ± 2</td></tr></table>

The influence of the filament representation is more pronounced for the stiffer material LSM 30, for which a larger number of layers can be printed before failure. Nevertheless, the ellipse and points representations show very good agreement with each other, whereas the rect. width representation overestimates, and the rect. contact representation underestimates the buildability. The rect. adapted representation predicts failure at an intermediate level. This observation is independent of the density adjustment. Figures 7 and 8 illustrate the simulated failure states for the different filament representations, showing the observed trends of over- and underestimation. The contour plots show the von Mises stress distribution of the deformed structure at the point of failure computed with density adjustment.

![](images/b656326bd964382c43e62afaa0bb4de7d8beeac283715dd4520a55d612c33de5.jpg)

![](images/069584851ed5f3f7b18f12409aabd8a947f33235f17ef81499a1ebe6a2732bc6.jpg)

![](images/57092dad50b01343e61bd0d894299de6b7573b553083c158732956e7267fab99.jpg)

![](images/b38b21075f29eecdfd5a603496f2f8c56f20965853351085da78220af3101177.jpg)

![](images/201c079e9ac554438c89dab3010b75d07048aeaed5835aff4fc8646fa739dd82.jpg)  
Figure 7: Material LM 30 (with density adjustment): Stress contour plot over deformed structure at point of failure using different filament representations (rect. width, rect. adapted, rect. contact, ellipse, points). The deformation is scaled by a factor of 4.

![](images/b282ab3c1d62745899a5a1890d2c442f09423f7464646edb53bc9abd0fc44b31.jpg)  
Figure 8: Material LSM 30 (with density adjustment): Stress contour plot over deformed structure at point of failure using different filament representations (rect. width, rect. adapted, rect. contact, ellipse, points). The deformation is scaled by a factor of 4.

## 3.1.2 Influence of density adjustment

As expected, the density adjustment does not affect the reference (points) and the volume-preserving rect. adapted representations. Furthermore, its influence is negligible for the softer material LM 30, which fails at a comparatively low number of layers. For the stiffer material LSM 30, differences of less than three layers are observed for the non-volume preserving representations. Without density adjustment, the rect. width representation predicts lower buildability, as the increased cross-sectional area results in a higher structural weight and, consequently, larger self-weight loading. Conversely, the rect. contact representation predicts higher buildability because of its lower total mass. Since the volume of the ellipse representation is very close to that of the points representation, the influence of the density adjustment is negligible in this case. As a result, the overestimation associated with rect. width and the underestimation associated with rect. contact are partially compensated when no density adjustment is applied. To isolate the effect of the geometric representation, the density-adjusted set-up is adopted for all subsequent investigations.

Overall, good agreement is obtained between simulation and experiment. While this comparison does not constitute a complete validation of the framework, it demonstrates that the model reproduces the experimentally observed values and highlights the importance of the assumed filament cross-section for reliable prediction of buildability. This is sufficient for the subsequent investigation of the relative influence of different filament representations on buildability.

## 3.2 Numerical investigation on the influence of filament shape on buildability: rectilinear walls

This section investigates the influence of process parameters and layer geometry on the buildability of rectilinear 3D-printed walls under different material behaviours. Specifically, we consider both changes in the physical filament shape resulting from different printing conditions (Sec. 3.2.1) and those arising in the numerical model from different filament representations (Sec. 2.2).

We first examine the influence of filament shape and its discretization on the elastic buckling of rectilinear walls, assuming a purely linear-elastic material model (Sec. 3.2.2). This allows us to isolate the effects of process parameters and filament shape on buildability from those arising from material nonlinearities. Within the same linear-elastic framework, we then investigate the combined effects of filament-shape discretization and varying Young’s modulus on buildability predictions (Sec. 3.2.3) using a fixed printing setup. The numerical results are also compared with an analytical buckling approach, providing further validation of the numerical framework.

Finally, we investigate the buildability of rectilinear walls using a more realistic elasto-plastic material model (Sec. 3.2.4). Specifically, two representative materials with distinct mechanical properties are considered: one characterised by a relatively low yield stress and pronounced plasticity, and the other by a higher yield stress and a more extended elastic regime. Together, these materials span the transition from buildability failure dominated by plastic yielding to failure governed primarily by elastic buckling.

## 3.2.1 Process parameters and filament geometry prediction

First, six representative printing conditions are defined to span a realistic range of operating regimes. The nozzle diameter is kept constant at $\phi _ { n } = 2 0 \mathrm { m m }$ , while two nozzle heights are considered to capture distinct deposition behaviours. Namely, a smaller nozzle height $h _ { n } = 1 0$ mm is employed to achieve a more layer-pressing printing condition, while a larger one, $h _ { n } = 2 0 \mathrm { m m }$ , is representative of a more free-flow deposition regime. In addition to the nozzle geometric parameters, the printing regime is also affected by the speed ratio $v ^ { * } = v _ { p } / u _ { f }$ . For each nozzle height, three cases are generated by varying the print and inflow velocities to obtain $v ^ { * } = 0 . 5 , \ 1 , 2 .$ This results in six distinct combinations, identified as ID1–ID6, which systematically explore the coupled effects of nozzle geometry and kinematics; the corresponding parameters are summarized in Table 3.

Material properties in the fluid state are kept identical across all cases to isolate the effect of process parameters. Specifically, the density is fixed at $\rho = 2 1 1 2 . 5 \mathrm { k g } / \mathrm { m } ^ { 3 }$ , the plastic viscosity at $\mu = 1 5 \mathrm { P a \cdot s } ,$ and the yield stress is set to a representative value of 1300 Pa, chosen as an intermediate value between the two considered in the validation (Section 3.1).

Table 3: Process parameters of the six different print scenarios ID1-ID6.
<table><tr><td>ID</td><td> $\phi _ { n } \ : ( \mathrm { m m } )$ </td><td> $h _ { n } \ ( \mathrm { m m } )$ </td><td> $v _ { p } \ ( \mathrm { m m / s } )$  (mm/s)  $u _ { f }$ </td></tr><tr><td>ID1</td><td>20</td><td>10</td><td>20 40</td></tr><tr><td>ID2</td><td>20</td><td>10</td><td>20 20</td></tr><tr><td>ID3</td><td>20</td><td>10 40</td><td>20</td></tr><tr><td>ID4</td><td>20</td><td>20 20</td><td>40</td></tr><tr><td>ID5</td><td>20</td><td>20</td><td>20 20</td></tr><tr><td>ID6</td><td>20</td><td>20 40</td><td>20</td></tr></table>

Subsequently, filament geometries for all print scenarios are predicted using ShapeGen3DCP. As illustrated in detail in [28], ShapeGen3DCP provides fast and reliable predictions, with typical deviations of 1–10% from experimental measurements. These deviations are comparable to the intrinsic uncertainties of the printing process and can therefore be considered acceptable for estimating the actual filament shape. For each condition, the cross-sectional profiles of two superimposed layers are computed and reported in Figure 9.

The generated cross-sections are also post-processed to extract the key geometric features reported in Table 4, namely the base layer width $( w _ { 1 } )$ , the base layer height $( h _ { 1 } )$ , and the interlayer contact length $( w _ { c } )$ . From these quantities, the filament aspect ratio of the base layer $( w _ { 1 } / h _ { 1 } )$ is computed, providing a direct measure of the layer spreading and the filament roundness ratio $w _ { 1 } / w _ { c }$ is defined to characterize the curvature of the sides of the layer’s profile.

Table 4: Post-processed geometrical features, roundness ratio and two-layers area.
<table><tr><td>ID</td><td> $h _ { n }$  (mm)</td><td> $v ^ { \ast } \left( - \right)$ </td><td>(mm)  $w _ { 1 }$ </td><td> $w _ { c } \left( m m \right)$ </td><td> $h _ { 1 } \ ( m m )$ </td><td> $w _ { 1 } / h _ { 1 } \left( - \right)$ </td><td> $w _ { 1 } / w _ { c } \left( - \right)$ </td><td> $A _ { S G } ( m m ^ { 2 } )$ </td></tr><tr><td>ID1</td><td>10</td><td>0.5</td><td>65.2</td><td>58.5</td><td>13.6</td><td>4.8</td><td>1.1</td><td>1337.2</td></tr><tr><td>ID2</td><td>10</td><td>1</td><td>32.1</td><td>27.4</td><td>8.4</td><td>3.8</td><td>1.2</td><td>567.8</td></tr><tr><td>ID3</td><td>10</td><td>2</td><td>19.8</td><td>10.97</td><td>9.8</td><td>2.0</td><td>1.8</td><td>316.0</td></tr><tr><td>ID4</td><td>20</td><td>0.5</td><td>36.3</td><td>26.2</td><td>16.3</td><td>2.2</td><td>1.4</td><td>1145.8</td></tr><tr><td>ID5</td><td>20</td><td>1</td><td>23.1</td><td>11.3</td><td>15.4</td><td>1.5</td><td>2.0</td><td>605.3</td></tr><tr><td>ID6</td><td>20</td><td>2</td><td>18.8</td><td>7.0</td><td>13</td><td>1.44</td><td>2.7</td><td>407</td></tr></table>

From the filament cross-sectional results, different printing regimes can be identified. Cases ID1 and ID2 exhibit flatter and wider filaments, associated with high aspect ratios $( w _ { 1 } / h _ { 1 } \gg 1 )$ , which indicate pronounced lateral spreading and a roundness ratio close to unity $( w _ { 1 } / w _ { c } \simeq 1 )$ , i.e., nearly rectangular profiles with limited curvature. These features reflect the outcome of a marked layer-pressing printing regime, associated with moderately high speed ratios and a small nozzle height.

In contrast, cases ID5 and ID6 produce rounder and taller filaments, with aspect ratios closer to unity $( w _ { 1 } / h _ { 1 } \simeq 1 )$ reflecting reduced spreading and increased filament height. These cases also exhibit larger values of the roundness ratio $( w _ { 1 } / w _ { c } \gg 1 )$ , which indicate more pronounced curvature. This behaviour is indicative of a free-flow deposition regime, in which the print speed is generally higher than the inflow speed, and the nozzle height is comparable to or greater than the nozzle diameter.

![](images/9c00dfbeb65a30e2749fc8013d4f576503a00e941f63ab4713e1fa45b8e920e5.jpg)

![](images/06a9023ca7cfd466b417d1eaa4749c0414320c8e55bb31a6512e23c88de459b6.jpg)

![](images/8ebdaa30c1355683dad17c0fbc8b1ed2399257a49b6725dd5dbf4f9be0f13147.jpg)

![](images/578dea00bb4461b3d468608cc7dc15a637cb38a140b6fb6418857ed733ad10c2.jpg)

![](images/c8d4e229584cc87198d345039cbccfaaacb13523c38b8009bebf220acfcf7103.jpg)

![](images/3be5505939b9bb1195ff768845ede62c3f8a3cd8344c8e5be6e70c922aa6d5db.jpg)  
Figure 9: Two-layers cross-sectional geometries predicted for the six different print scenarios ID1-ID6.

Cases ID3 and ID4 lie between the layer-pressing and free-flow deposition regimes, due to their intermediate values of aspect ratio and roundness ratio. In fact, for ID3, the effect of high print speed is balanced by the reduced nozzle height, whereas for ID4, the opposite holds.

Table 4 also reports the predicted cross-sectional areas. The influence of the speed ratio on filament geometry is clearly observed: within each group of three cases (i.e., at fixed nozzle height), the cross-sectional area decreases as the speed ratio increases. This trend reflects the inverse relationship between the deposited material volume per unit length and the speed ratio, in agreement with mass conservation.

## 3.2.2 Influence of filament shape on elastic buckling

Buckling simulations are performed for the six printing scenarios defined in Table 3 and characterized in Table 4, considering 200 mm long wall structures with a fixed base (see Section 2.3) and varying filament representations (rect. width, rect. contact, rect. adapted, ellipse, point as defined in Section 2.2).

To isolate pure elastic buckling effects, an elastic material model is considered, using for simplicity the elastic parameters obtained from the calibration of material LM 30 (see Table 1). The Young’s modulus is rounded to 990000 Pa, with a rate of increase over time of 22.11 Pa/s, and a Poisson’s ratio of 0.3.

As in the validation study, an imperfection magnitude of α = 0.00025 and a load step of 0.25 s are applied. To ensure a fair comparison between the different geometrical representations, the density adjustment setup studied in Sec. 3.1 is used throughout the following analyses to preserve the total mass and, consequently, the dead load.

Figure 10 reports the number of activated layers at failure, as defined in eq. (5), for all printing scenarios. The results are shown as a function of the speed ratio, separately for nozzle heights of 10 mm (ID1–ID3, left) and 20 mm (ID4–ID6, right). Consistent with the validation study, the point-based and elliptical filament representations lead to nearly identical buildability predictions. In contrast, rect. contact systematically underestimates the number of buildable layers, whereas rect. width systematically overestimates it. The rect. adapted representation also overestimates the buildability, but to a lesser extent than rect. width. Note, a similar number of activated layers for different printing scenarios (e.g., for ID1 and ID2) leads to different absolute heights at failure since the layer heights differ.

![](images/7bf464ddb5e8074b27983c5e4462ca9c721be7fe3fa221151b76bb78aa65ac72.jpg)  
(a) Nozzle height $h _ { n } = 1 0$ mm

![](images/a75d40b4f7285bc8d2a05f14d8c16460ed92035e45107a0958379b848fefe1dc.jpg)  
(b) Nozzle height $h _ { n } = 2 0$ mm

Figure 10: Elastic study: Number of activated layers at failure predicted for the different filament representations and printing scenarios listed in Table 3.  
![](images/80cfa3daf591b1715fb7532b4d206f2ef4fdd36a9a63507764a7fd1a340c0781.jpg)  
Figure 11: Elastic study: Absolute difference in the number of activated layers at failure with respect to the points representation for the six printing scenarios listed in Table 3 shown as a function of the roundness ratio.

A clear influence of the process parameters can also be observed. For cases with pronounced layer pressing (Figure 10a, speed ratios 0.5 and 1.0, corresponding to ID1 and ID2), the effect of the filament representation is less pronounced than for the free-flow cases (Figure 10b, speed ratios 1.0 and 2.0, corresponding to ID5 and ID6), which exhibit more rounded filament geometries. To quantify this trend, Figure 11 shows the absolute difference in the number of activated layers at failure with respect to the points representation for all six printing scenarios listed in Table 4. For each representation, this difference is computed as the number of activated layers at failure of the respective representation minus that obtained with points. With increasing roundness ratio, the deviation from points increases. For rect. contact, the difference decreases from approximately −1 to about −3 layers, indicating an increasing underestimation of buildability. For rect. adapted, the difference increases from about 1 to 5 layers, while for rect. width it increases from about 3 to approximately 9 layers, indicating an increasing overestimation. In contrast, the difference between ellipse and points remains below one layer for all cases considered, showing that ellipse captures the influence of the filament shape very accurately, independently of the roundness ratio.

## 3.2.3 Combined influence of filament shape discretization and Young’s modulus on elastic buckling

A second study is conducted to specifically investigate the combined influence of Young’s modulus and filament-shape representation on the buildability of rectilinear walls. For this purpose, printing scenario ID3 (Φ = 2 and $h _ { n } = 1 0$ mm) is selected, as it produces a pronounced, rounded filament geometry and, consequently, exhibits a marked sensitivity to the adopted filament representation, as observed in the previous comparison. The material is still assumed to behave linearly elastically, while the Young’s modulus is varied from 61.875 kPa to 1980 kPa, thereby covering the broad range of values reported in the literature. The rate of increase of Young’s modulus over time is kept unchanged, as are all other simulation settings, consistently with the elastic simulations described in Section 3.2.2.

Following the bifurcation analysis from Suiker [36], the analytical expression for the critical wall height is used to compute a reference value for comparison:

$$
l _ { c r i t } = 1 . 9 8 6 3 5 \left( \frac { D } { \rho g w _ { i } } \right) ^ { 1 / 3 } ,\tag{6}
$$

with $\begin{array} { r } { D = \frac { E w _ { i } ^ { 3 } } { 1 2 ( 1 - \nu ) ^ { 2 } } . ~ w _ { i } } \end{array}$ denotes the wall width associated with the selected rectangle-based filament representation $( w _ { 1 } , w _ { c } , \mathrm { o r } w _ { a d } )$ . For the analytical solution, a constant Young’s modulus is assumed, i.e., no evolution of the stiffness is considered. The gravitational acceleration is set to $g = \mathrm { { \bar { 9 } . 8 1 m / s ^ { 2 } } }$ , and the density ρ used in the corresponding structural simulation is adopted for the calculation of the analytical critical height. The corresponding number of layers is computed by dividing $l _ { c r i t }$ by the layer height $h _ { 1 }$

![](images/7154a6ed98ac328af1547cae9b54ea7606c8d9e23e413bae8af9daee9cac5151.jpg)  
Figure 12: Elastic ID3 case: Number of activated layers at failure for varying Young’s moduli. The dashed lines indicate the analytical prediction obtained from eq. (6), expressed as $l _ { c r i t } / h _ { 1 }$ , for the rectangle-based filament representations. For the ellipse case, only the simulation result is shown.

Figure 12 showcases the simulated number of activated layers at failure for the different Young’s moduli considered, together with the predictions obtained from the analytical formula. First, it can be observed that, for all rectangle-based filament representations, the simulation results are in good agreement with the analytical solution over the entire range of Young’s moduli investigated. The small deviations between the numerical and analytical results can be attributed to differences in the underlying assumptions, including the stopping criterion and the reference configuration adopted Overall, these results further support the validity of the numerical framework in reliably capturing the governing elastic buckling behaviour.

As expected, the maximum number of buildable layers decreases with decreasing Young’s modulus, as a reduction in material stiffness lowers the structural resistance to elastic instability and buckling under self-weight. Consistent with the previous study, rect. contact leads to lower buildability predictions, whereas rect. width and rect. adapted yield higher values. The ellipse representation, which is only evaluated numerically since the analytical expression in Eq. (6) is derived exclusively for rectangular cross-sections, again lies between the rectangle-based approaches.

Table 5: Elastic ID3 case: Percentage changes in the number of activated layers for varying Young’s moduli [kPa] and filament representations with respect to ellipse.
<table><tr><td>E in kPa</td><td>61.875</td><td>123.75</td><td>247.5</td><td>495</td><td>990</td><td>1980</td></tr><tr><td>rect. width</td><td>46.94</td><td>45.65</td><td>44.35</td><td>43.40</td><td>42.38</td><td>42.35</td></tr><tr><td>rect. adapted</td><td>25.85</td><td>24.46</td><td>23.91</td><td>22.92</td><td>22.16</td><td>21.95</td></tr><tr><td>rect. contact</td><td>-14.97</td><td>-15.76</td><td>-16.96</td><td>-17.36</td><td>-18.01</td><td>-18.18</td></tr></table>

To quantify the influence of the filament representation, Table 5 reports the percentage change in the number of activated layers at failure with respect to the corresponding value obtained using the ellipse representation. The relative deviations are nearly independent of Young’s modulus. For smaller Young’s moduli, the lower number of buildable layers results in a higher sensitivity of the percentage values. As the Young’s modulus increases, the relative deviations converge and approach nearly constant values. This observation suggests that the influence of the filament representation can be generalized across elastic materials. For the wall simulations considered here, the rect. contact representation underestimates the buildability by up to 18 %, whereas rect. adapted and rect. width overestimate it by up to 26 % and 47 %, respectively, relative to the ellipse representation.

## 3.2.4 Influence of filament shape on elasto-plastic buckling

The influence of filament shape and its representation on buildability is further investigated in the case of elasto-plastic material behaviour, specifically using the two material models LM 30 and LSM 30 introduced in Section 3.1. The six printing scenarios defined in Table 3 are thus again reproduced, with all five filament representations considered for each scenario. The simulations are performed on 200 mm-long wall structures, using the same boundary conditions, numerical settings, and assumptions adopted in the elastic studies. These include fixed displacements at the base, the same imperfection magnitude, identical temporal and spatial discretizations, and density adjustment to preserve the total mass, and hence the dead load, irrespective of the adopted layer-shape representation.

![](images/76cf21debd530d2f6fce8aa1db2b392c701820e787bf65ccd2a3abd5ea616d3a.jpg)  
(a) Nozzle height $h _ { n } = 1 0$ mm

![](images/46374c0ea5beac4ab14cb816c65f67f474d7cb40780d88a8e6a7538f1f031d48.jpg)  
(b) Nozzle height $h _ { n } = 2 0$ mm  
Figure 13: Plastic study LM 30: Comparison of the number of layers at failure using different filament representations for the different printing cases.

Figures 13 and 14 show the number of activated layers at failure for LM 30 and LSM 30 for each layer shape, respectively. Again, the results are shown as a function of the speed ratio for all printing scenarios, with a nozzle height of 10 mm on the left and 20 mm on the right. Compared to the purely elastic case, the absolute number of activated layers at failure is reduced for both elasto-plastic materials, reflecting the earlier onset of collapse due to plasticity. Due to its higher yield stress and stiffer elastic response, LSM 30 consistently allows more layers to be built before elasto-plastic buckling than LM 30. For example, in print scenario ID2, the wall printed with LSM 30 fails within the 30rd layer, whereas using LM 30 it already fails within the 18th layer. In the pure elastic case, 36 layers were reached with similar elastic properties to LM 30.

![](images/570be993b2dc163fbc078136be0799f6eb9779dd322de826c66a4d7d356c0c0e.jpg)  
(a) Nozzle height $h _ { n } = 1 0$ mm

![](images/d992b489db969387ff6f3026b16ab2eae9039544ba11eb7d2e9ced1f6a12afae.jpg)  
(b) Nozzle height $h _ { n } = 2 0$ mm  
Figure 14: Plastic study LSM 30: Comparison of the number of layers at failure using different filament representations for the different printing cases.

The sensitivity to the filament representation follows the same trend as in the elastic case, being more pronounced for free-flow scenarios. The absolute values in the number of layers depend on the material properties and differ from the elastic study. For that, Figure 15 reports the absolute difference in the number of activated layers at failure with respect to the points representation as a function of the roundness ratio for both materials. As in the elastic case, the influence of the filament representation increases for more rounded filaments, corresponding to reduced layer pressing. For LM 30, the absolute deviations remain relatively small, with rect. width overestimates the buildability by less than four layers. For LSM 30, the absolute deviations are larger and reach magnitudes comparable to the elastic study, with overestimations of up to approximately nine layers.

Overall, these results demonstrate that, when plastic yielding occurs, plasticity reduces the absolute buildability due to elasto-plastic buckling mechanisms but does not alter the fundamental influence of the filament representation. In particular, the ellipse approximation fits the most realistic points representation very well, whereas rectangle-based representations can lead to significant over- or underestimation of buildability, especially for more rounded filament geometries.

![](images/59b1f8c4cf2e6f1ad756f28c08bf5157b4e22775b1643b75711a34b297e10131.jpg)  
(a) LM 30

![](images/fc516b9f5c712e8129c540eaa02927e2745df964609015d6e58008e77d5d08d2.jpg)  
(b) LSM 30  
Figure 15: Plastic study: Absolute difference in the number of activated layers at failure with respect to the points representation for the six printing scenarios listed in Table 3, shown as a function of the roundness ratio for both materials.

## 3.2.5 Comparative influence of filament shape and discretization on buildability

To enable a direct comparison across all investigated material models (purely elastic as well as elasto-plastic LM 30 and LSM 30), Figure 16 presents the percentage change in the number of activated layers at failure with respect to the corresponding value obtained with the points representation, which is taken as the most realistic reference, over the roundness ratio. Although the absolute number of activated layers differs substantially between the elastic and elasto-plastic cases, as discussed in the preceding subsections, the relative changes with respect to the points representation are remarkably similar for all materials.

For all material models, consistent trends are observed with respect to the filament representation and the roundness ratio. The influence of the filament shape increases for more rounded filament geometries, corresponding to reduced layer pressing and increased free-flow. Rect. width and rect. adapted approximations of the layer shape tend to overestimate the buildability, with relative increases of up to approximately 80%. In contrast, the rect. contact approach underestimates the number of activated layers by up to about 30%. These relative deviations are largely independent of whether elastic or elasto-plastic material behavior is considered. Furthermore, the elastic study with varying Young’s modulus for case ID3 (Table 5) shows that the relative over- and underestimation introduced by the different filament representations remains largely unchanged over a wide range of Young’s moduli. This observation complements the results obtained for the different material models and suggests that the influence of the filament representation is primarily governed by geometric rather than material effects. Note that these results are based on the densityadjusted set-up. Without density adjustment, the observed over- and underestimation of the non-volume-preserving representations, especially rect. width and rect. contact, can be reduced (see Sect. 3.1).

In addition to filament shape effects, Figure 17 highlights the influence of geometric slenderness described by the layer width $w _ { 1 }$ (the wall length is the same in all cases) on the absolute buildability. The wall height at failure is computed as the number of activated layers at failure multiplied by the layer height to account for the varying heights. The possible wall height increases with increasing layer width, i.e., for broader filament profiles. This trend is consistent across all material models and filament representations. For clarity, only the points representation is shown in this comparison, as it provides the most realistic description of the filament geometry. The results confirm that less slender profiles allow higher total heights before elastic and elasto-plastic buckling. In general, plastic yielding reduces buildability compared to purely elastic material behavior.

![](images/f7c724a74aaaa45d3540d5ce249a5360085e4b2b80b71e5b65590d78e60ed3da.jpg)  
Figure 16: Change of number of activated layers with respect to the points shape in percentage for the different cases: elastic (solid), plastic LM 30 (dashed), and plastic LSM (dash-dotted) as a function of roundness ratio.

![](images/a62727c648a187efe9d32009ee46d582704c7be888d6a81cc15c18039a515e4f.jpg)  
Figure 17: Wall height at failure (number of activated layers × layer height) for the points representation and different materials: elastic (solid), plastic LM 30 (dashed), and plastic LSM (dash-dotted) as a function of the layer width w<sub>1</sub>.

## 3.3 Numerical investigation on the influence of filament shape on buildability: cylinders

Extending the investigation from rectilinear wall prints, the proposed framework is applied to study the influence of filament representations on the buildability of cylindrical structures with radii of 50 and 150 mm.

The objective is to assess whether the trends identified for rectilinear walls in Sect. 3.2 also apply to cylindrical geometries. The analysis is intended as a qualitative assessment without validation with experimental data. For that purpose, the printing scenario ID3 is selected, resulting in rounded filament profiles. The buildability simulations are performed using the softer elasto-plastic material LM 30. A larger imperfection amplitude of $\alpha = 0 . 0 0 5$ is prescribed to promote buckling in the x-direction. Under these conditions, a symmetric failure mode with respect to the x-axis develops, allowing the simulations to be carried out on one half of the cylinder using symmetric boundary conditions. The load step is set to $\Delta t = 2$ s leading to feasible computational costs, and a mesh size comparable to that used in the wall simulations is selected. As in the previous investigations, the density-adjusted set-up is adopted.

Table 6 reports the resulting number of activated layers at the time of failure, defined as the point at which the outof-plane deformation reaches one layer width (see Sect. 2.3). Since a close agreement between the points and ellipse representations was observed for rectilinear walls in Section 3.2, the ellipse representation is adopted here for simplicity as the most realistic reference. The same trends identified in the previous investigations are also observed for cylindrical geometries. The rect. width shape leads to a pronounced overestimation of the buildability, which is reduced when using the rect. adapted approach. In contrast, rect. contact results in an underestimation with respect to the ellipse case.

Table 6: Comparison of activated layers at failure for cylinders with two different radii and varying filament representations.
<table><tr><td>radius [mm]</td><td></td><td></td><td>rect. width rect. adapted rect. contact</td><td>ellipse</td></tr><tr><td>50</td><td>29.25</td><td>27.00</td><td>21.75</td><td>26.50</td></tr><tr><td>150</td><td>41.58</td><td>33.17</td><td>19.17</td><td>30.83</td></tr></table>

Figure 18 and Figure 19 show the deformed configurations of the cylinders at the point of failure. Due to the different radii, two distinct failure mechanisms are observed, which influence the relative over- and underestimation rates with respect to the ellipse representation. The more slender cylinder with $r = 5 0$ mm fails by global buckling, similar to the rectilinear wall case. In contrast, for the larger cylinder with r = 150 mm, failure is characterized by pronounced bulging in the upper region, which exceeds the prescribed out-of-plane deformation limit. In both cases, the failure mode itself is not altered by the choice of filament representation, confirming that this is governed by the global geometry. Nevertheless, for the larger cylinder radius, the magnitude of over- and underestimation relative to the ellipse representation increases significantly. While for $r = 5 0$ mm the deviation ranges between approximately −18% and +10%, it increases to about −38% and +35% for r = 150 mm.

![](images/e80f964b0e37dae73561eda228c6e46312eb037cca1f3b289f42ba334925293d.jpg)

![](images/1b72cfadc429d709300a85583c84d59a5f8d85c48b8c9a5515ffa163649035d7.jpg)

![](images/7a3472b646e1d0f568f964d897717a22f5af2f6bb3a3f0f211ca3761af9f2b30.jpg)

![](images/00611fe4f88428c609eddb3eb6993bde730b4930294f1aef0b8c9965c570e7a4.jpg)  
Figure 18: Cylinder radius 50 mm: Stress contour plots over deformed structure at point of failure using different filament representations (rect. width, rect. adapted, rect. contact, ellipse). The deformation is scaled by a factor of 2.

![](images/fca120cba285e7d70310ea8c4ae4348a5c4c14fb8b755b6066dd505c78e9d326.jpg)  
Figure 19: Cylinder radius 150 mm: Stress contour plots over deformed structure at point of failure using different filament representations (rect. width, rect. adapted, rect. contact, ellipse). The deformation is scaled by a factor of 2.

## 4 Discussion: results summary and simulation guidelines

This section discusses the main findings of the study and derives general indications and guidelines that can support both designers and numerical modellers. In particular, the results provide insight into the influence of filament cross-sectional geometry on the buildability of thin-walled 3D-printed structures and on the sensitivity of buildability predictions to different geometric discretization choices adopted in numerical simulations.

• The results demonstrate that the filament cross-sectional shape strongly influences structural buildability. Layer pressing produces wider and flatter filaments, leading to less slender geometries with higher bending stiffness and increased resistance to buckling. Conversely, free-flow deposition generates more rounded filaments with reduced width and interlayer contact area, decreasing structural stiffness and promoting earlier instability.

• Similarly, in numerical simulations, the adopted cross-sectional representation can significantly affect buildability predictions. The influence is particularly relevant for free-flow deposition, where capturing the rounded filament geometry is essential, whereas layer-pressing strategies are less sensitive due to their flatter crosssections, which are better represented by simple rectangular approximations.

• A useful parameter for assessing the approximation error in rectilinear walls is the roundness ratio $( w _ { 1 } / w _ { c } )$ defined as the ratio between the maximum filament width w and the interlayer contact length $w _ { c }$ . This parameter provides a compact measure of the deviation of the filament cross-section from an ideal rectangular shape and was found to correlate strongly with the magnitude of the approximation error.

Based on these observations, a quantitative design chart is developed to estimate the error associated with buildability predictions obtained using different filament cross-sectional representations strategies for the case of perfectly elastic rectilinear walls. Figure 20 presents the proposed design chart, which relates the expected prediction error to the filament roundness ratio. The chart was derived from the results reported in Figure 17 by fitting linear regression trends to the data obtained for each cross-sectional approximation.

Several observations can be drawn from Figure 20. First, the linear regressions provide a good representation of the numerical data, suggesting that the relationship between discretization error and roundness ratio can be reasonably approximated by a linear trend within the investigated range. Second, the figure highlights systematic biases associated with the different geometric approximations. The rect. width and rect. adapted filament representations tend to overestimate buildability, predicting higher number of layers than those obtained with the reference filament geometry. In contrast, the rect. contact approximation systematically underestimates buildability. Finally, the ellipse approximation shows the best overall agreement with the reference results (points). Moreover, based on the previous parametric study (Table 5), the proposed chart is expected to be largely independent of the elastic modulus, as it represents a relative error measure. When plasticity effects are introduced, the analyses in Sect. 3.2 observed a similar trend, although the magnitude of the error is affected: overestimating approaches become less conservative, while the rect. contact approximation, which underestimates buildability, shows increased deviations. These effects of plasticity in the buildability error are indicated in the design chart by arrows.

![](images/d9c8eac7bb4ecefe952fa46cae8eeca63eb9dfb6be7bc408d9cd41454266dece.jpg)  
Figure 20: Simplified design chart indicating buildability percentage errors associated to different filament cross-sections discretization choices at varying of the roundness ratio for rectilinear walls.

Overall, the proposed design chart provides a practical framework for selecting filament representation strategies and quantifying the uncertainty associated with geometric simplifications in buildability simulations of thin-walled 3D-printed structures. Although it was derived for rectilinear walls, the present results demonstrate that the chart can also provide some meaningful guidance for curvilinear objects. Across the range of roundness ratios investigated, the discrepancies observed for both cylindrical configurations remained in fact within the uncertainty bounds predicted by the design chart. Specifically, the cylindrical case studies reveal that a curvilinear toolpath can enhance the geometric stiffness of the structure and reduce the sensitivity of buildability predictions to the adopted filament representation. As a result, although further research is needed to develop correction factors for highly curved structures, the design chart can in principle be expected to provide conservative error estimates when applied to curvilinear geometries.

The cylindrical-wall examples, which combine relatively large structural dimensions with a high number of buildable layers before collapse, also raise the important topic of computational cost. In all studies presented in this work, the element size was kept constant across the different filament representations to ensure a fair comparison of the resulting buildability predictions. The adopted mesh resolution was selected to adequately capture the geometry of the ellipse and points representations, which require a finer discretization due to their curved boundaries. Consequently, the performed simulations do not provide a representative basis for comparing computational efficiency. Due to their simpler and more regular geometry, rectangle-based models can generally be discretized with coarser meshes while maintaining accurate structural predictions. Therefore, they can significantly reduce computational time and memory requirements in large-scale buildability simulations.

Table 7: Decision map of layer shape approximation. In case of e.g. cylinders radii matters, too.
<table><tr><td>Requirement</td><td>Roundness ratio  $w _ { 1 } / w _ { c } \approx 1$ </td><td>Roundness ratio  $w _ { 1 } / w _ { c } > 1$ </td></tr><tr><td>High accuracy</td><td>ellipse</td><td>ellipse</td></tr><tr><td>Fast computation</td><td>any rectangle</td><td>rect. contact (conservative)</td></tr><tr><td>Risk of overestimation acceptable rect. adapted or rect. width (with caution)</td><td></td><td>rect. adapted (with caution)</td></tr></table>

In conclusion, we summarize the main findings in Table 7, and provide practical guidelines for selecting an appropriate cross-sectional representation:

1. A point-wise representation of the filament cross-section is generally unnecessary for accurate buildability predictions. Among the investigated approaches, the ellipse approximation consistently provided results close to the reference geometry while requiring a simpler geometric description, representing the best compromise between accuracy and modelling effort.

2. Rectangular approximations remain valuable when computational efficiency or conservative estimates are prioritized. The rect. contact discretization consistently underestimates buildability and can therefore be used as a conservative lower-bound approach, whereas the rect. width formulation tends to overestimate buildability and should be applied with caution in safety-critical assessments.

3. The rect. adapted discretization merits separate discussion. Although it also tends to overestimate buildability, its error remains considerably smaller than that of rec. width. Moreover, despite its simplicity, its geometry is designed to preserve the volume of the reference (most realistic) representation, unlike the other discretizations. Consequently, the total weight is preserved without the need for artificial density adjustment. This is particularly advantageous for dynamic and contact simulations, where density adjustment may be undesirable, as it can alter inertia, natural frequencies, and contact-related responses.

## 5 Conclusion

This work presented a geometrically informed workflow for buildability assessment in 3D concrete printing, integrating the deep-learning-based filament shape prediction framework ShapeGen3DCP with a layer-activation finite element model. By enabling the automatic generation of geometry-aware numerical models directly from material and process parameters, the proposed methodology eliminates the need for preliminary filament characterization or computationally expensive fluid-dynamics simulations. Application of the framework demonstrated that filament geometry can substantially influence buildability predictions and that incorporating realistic, process-informed crosssectional representations significantly improves the predictive capability of standard FEM-based approaches.

The validation and parametric studies demonstrated that the influence of filament geometry on buildability predictions depends strongly on both the adopted printing strategy and the governing failure mechanism of the printed structure. In particular, rounder filaments associated with free-flow deposition were found to be substantially more sensitive to geometric simplifications than the flatter filaments produced by layer-pressing strategies. Likewise, rectilinear walls exhibited greater sensitivity than geometrically stiffer configurations, such as cylindrical structures. In contrast, the influence of filament geometry was found to be only weakly dependent on the constitutive material behaviour, with limited sensitivity to variations in elastic stiffness or to the adoption of elastic versus elasto-plastic material models.

Among the investigated filament cross-sectional representations, elliptical geometries consistently provided the best compromise between prediction accuracy and modelling simplicity. The use of a more complex realistic polyline representation resulted in only marginal improvements in predictive accuracy, while introducing substantially greater geometrical complexity. On the other side, when rectangular representations are preferred due to their regular mesh topology and computational efficiency, enforcing volume conservation through an appropriate dimensioning strategy can significantly enhance the reliability of the resulting simulations.

These findings motivate the development of a new class of geometrically informed buildability models, in which extrusion process parameters and the resulting filament geometry are explicitly integrated into structural simulations. In this work, this paradigm was realized by coupling ShapeGen3DCP with a layer-activation FEM framework, demonstrating that machine-learning-based geometry prediction can be effectively integrated into existing numerical workflows without excessive increasing computational complexity. More broadly, the proposed framework illustrates how data-driven geometry prediction can bridge the gap between process modelling and structural analysis, enabling more realistic yet computationally efficient simulations of the printing process.

From a practical perspective, the proposed workflow and the accompanying design guidelines provide a straightforward methodology for selecting an appropriate geometric representation according to the required balance between accuracy and computational cost. The presented design chart allows users to estimate the uncertainty introduced by different filament cross-sectional representations based on the filament roundness ratio. The proposed framework therefore represents a practical step towards more reliable and predictive numerical tools for the design, optimization, and structural assessment of 3D concrete printing processes.

## Acknowledgements

The authors acknowledge the Gastwissenschaftlerprogramm of Bundesanstalt für Materialforschung und -prüfung (BAM), which supported a two-month research stay of G. Rizzieri at BAM during which the work presented in this paper was initiated. Additionally, the first author acknowledges the project "Material and Process Modelling for Lunar 3D Printing", CUP D47G25000060006, pursuant to Notice No. 47 of 20/02/2025 (‘Decree for the recruitment of international post-doctoral researchers’), PNRR, Mission 4, Component 2, Investment 1.2, financed by the European Union - NextGeneration EU.

## Data availability

The workflow implementation, including all scripts and results, is available as a dataset publication in [31].

## References

[1] R. J. Wolfs, T. A. Salet, N. Roussel, Filament geometry control in extrusion-based additive manufacturing of concrete: The good, the bad and the ugly, Cem. Concr. Res. 150 (2021). doi:10.1016/j.cemconres.2021. 106615.

[2] T. Marchment, J. Sanjayan, M. Xia, Method of enhancing interlayer bond strength in construction scale 3D printing with mortar by effective bond area amplification, Materials & Design 169 (2019) 107684. doi:10.1016/ j.matdes.2019.107684.

[3] V. N. Nerella, S. Hempel, V. Mechtcherine, Effects of layer-interface properties on mechanical performance of concrete elements produced by extrusion-based 3D-printing, Construction and Building Materials 205 (2019) 586–601. doi:10.1016/j.conbuildmat.2019.01.235.

[4] M. Nodehi, F. Aguayo, S. E. Nodehi, A. Gholampour, T. Ozbakkaloglu, O. Gencel, Durability properties of 3D printed concrete (3DPC), Automation in Construction 142 (2022) 104479. doi:10.1016/j.autcon.2022. 104479.

[5] P. Bran-Anleu, T. Wangler, V. N. Nerella, V. Mechtcherine, P. Trtik, R. J. Flatt, Using micro-XRF to characterize chloride ingress through cold joints in 3D printed concrete, Mater Struct 56 (3) (2023) 51. doi:10.1617/ s11527-023-02132-w.

[6] R. Wolfs, F. Bos, T. Salet, Early age mechanical behaviour of 3D printed concrete: Numerical modelling and experimental testing, Cement and Concrete Research 106 (2018) 103–116. doi:10.1016/j.cemconres.2018. 02.001.

[7] R. J. M. Wolfs, A. S. J. Suiker, Structural failure during extrusion-based 3D printing processes, Int. J. Adv. Manuf. Technol. 104 (1) (2019) 565–584. doi:10.1007/s00170-019-03844-6.

[8] T. Ooms, G. Vantyghem, R. Van Coile, W. De Corte, A parametric modelling strategy for the numerical simulation of 3D concrete printing with complex geometries, Addit. Manuf. 38 (2021). doi:10.1016/j.addma.2020. 101743.

[9] V. Nguyen-Van, B. Panda, G. Zhang, H. Nguyen-Xuan, P. Tran, Digital design computing and modelling for 3-D concrete printing, Autom. Constr. 123 (2021) 103529. doi:10.1016/j.autcon.2020.103529.

[10] L. Jendele, J. Rymes, J. Cervenka, M. Herzfeldt, Time dependent modelling of concrete for the simulation of 3D printing, in: Proceedings of the Fib Symposium 2025, Antibes, France, 2025.

[11] Saif-Ur-Rehman, A. Robens-Radermacher, J. Unger, R. Wolfs, Assessing Structural Failure in Extrusion-Based 3D Concrete Printing Using a Plasticity Model with Non-Linear Hardening, 1st Edition, CRC Press, London, 2026, pp. 775–785. doi:10.1201/9781003660026-91.

[12] A. Tripathi, S. A. Nair, N. Neithalath, A comprehensive analysis of buildability of 3D-printed concrete and the use of bi-linear stress-strain criterion-based failure curves towards their prediction, Cement and Concrete Composites 128 (2022) 104424. doi:10.1016/j.cemconcomp.2022.104424.

[13] B. Nedjar, Incremental viscoelasticity at finite strains for the modelling of 3D concrete printing, Comput Mech 69 (1) (2022) 233–243. doi:10.1007/s00466-021-02091-5.

[14] Q. Chen, G. B. Barbat, M. Cervera, Finite element buildability analysis of 3D printed concrete including failure by elastic buckling and plastic flow, Engineering Structures 340 (2025) 120675. doi:10.1016/j.engstruct. 2025.120675.

[15] M. Pierre, S. Ghabezloo, P. Dangla, R. Mesnil, M. Vandamme, J.-F. Caron, Multiphysics modelling of 3D concrete printing: From material model to process simulation and optimisation, Additive Manufacturing 109 (2025) 104847. doi:10.1016/j.addma.2025.104847.

[16] A. Gribonval, M. Pierre, N. Ducoulombier, K. Sab, R. Mesnil, J. Bleyer, Multi-physics modelling of 3Dprinted concrete evolution in environmental conditions, Cement and Concrete Research 196 (2025) 107918. doi:10.1016/j.cemconres.2025.107918.

[17] X. Liu, B. Sun, The influence of interface on the structural stability in 3D concrete printing processes, Addit. Manuf. 48 (2021) 102456. doi:10.1016/j.addma.2021.102456.

[18] J. Reinold, K. Daadouch, G. Meschke, Numerical simulation of three dimensional concrete printing based on a unified fluid and solid mechanics formulation, Front. Struct. Civ. Eng. 18 (4) (2024) 491–515. doi: 10.1007/s11709-024-1082-2.

[19] R. Comminal, W. R. Leal da Silva, T. J. Andersen, H. Stang, J. Spangenberg, Modelling of 3D concrete printing based on computational fluid dynamics, Cem. Concr. Res. 138 (2020). doi:10.1016/j.cemconres.2020. 106256.

[20] J. Reinold, V. N. Nerella, V. Mechtcherine, G. Meschke, Extrusion process simulation and layer shape prediction during 3D-concrete-printing using the Particle Finite Element Method, Autom. Constr. 136 (2022). doi:10. 1016/j.autcon.2022.104173.

[21] G. Rizzieri, L. Ferrara, M. Cremonesi, Numerical simulation of the extrusion and layer deposition processes in 3D concrete printing with the Particle Finite Element Method, Comput Mech 73 (2) (2024) 277–295. doi: 10.1007/s00466-023-02367-y.

[22] G. Rizzieri, D. Bos, R. Wolfs, L. Ferrara, M. Cremonesi, A unified fluid-solid elasto-viscoplastic finite element model for the simulation of 3D concrete printing across process scales, Computer Methods in Applied Mechanics and Engineering 457 (2026) 118929. doi:10.1016/j.cma.2026.118929.

[23] D. An, Z. Zhu, M. Rahman, Y. Zhang, R. C. Yang, Process-informed smooth particle hydrodynamics-finite element (SPH-FE) simulation of 3D concrete printing: From flow behaviour to structural failure, Journal of Building Engineering 120 (2026) 115447. doi:10.1016/j.jobe.2026.115447.

[24] G. Rizzieri, S. Meni, M. Cremonesi, L. Ferrara, A Particle Finite Element Method for investigating the influence of material and process parameters in 3D Concrete Printing, Comput. Struct. 316 (2025) 107883. doi:10.1016/ j.compstruc.2025.107883.

[25] P. Carneau, R. Mesnil, O. Baverel, N. Roussel, Layer pressing in concrete extrusion-based 3D-printing: Experiments and analysis, Cem. Concr. Res. 155 (May 2022). doi:10.1016/j.cemconres.2022.106741.

[26] A. Alhussain, J. P. Duarte, N. C. Brown, Developing a data-driven filament shape prediction model for 3D concrete printing, Front. Built Environ. 10 (Feb. 2024). doi:10.3389/fbuil.2024.1363370.

[27] W. Lao, M. Li, T. N. Wong, M. J. Tan, T. Tjahjowidodo, Improving surface finish quality in extrusion-based 3D concrete printing using machine learning-based extrudate geometry control, Virtual Phys. Prototyp. 15 (2) (2020) 178–193. doi:10.1080/17452759.2020.1713580.

[28] G. Rizzieri, F. Lanteri, L. Ferrara, M. Cremonesi, ShapeGen3DCP: A deep learning framework for layer shape prediction in 3D concrete printing, Comput. Struct. 323 (2026) 108142. doi:10.1016/j.compstruc.2026. 108142.

[29] F. Mölder, K. P. Jablonski, B. Letcher, M. B. Hall, C. H. Tomkins-Tinch, V. Sochat, J. Forster, S. Lee, S. O. Twardziok, A. Kanitz, A. Wilm, M. Holtgrewe, S. Rahmann, S. Nahnsen, J. Köster, Sustainable data analysis with Snakemake, F1000Res 10 (2021) 33. doi:10.12688/f1000research.29032.1.

[30] I. A. Baratta, J. P. Dean, J. S. Dokken, M. Habera, J. S. Hale, C. N. Richardson, M. E. Rognes, M. W. Scroggs, N. Sime, G. N. Wells, DOLFINx: The next generation FEniCS problem solving environment (Dec. 2023). doi:10.5281/ZENODO.10447666.

[31] G. Rizzieri, Saif-Ur-Rehman, J. F. Unger, A. Robens-Radermacher, Dataset for publication: Influence of Extruded Filament Geometry on Buildability in 3D Concrete Printing: A Geometry-Informed Deep Learning–FEM Approach, [Dataset]. Zenodo (2026). doi:10.5281/zenodo.22195476.

[32] S. M. Rosenbusch, P. Diercks, V. Kindrachuk, J. F. Unger, Integrating custom constitutive models into FEniCSx: A versatile approach and case studies, Advances in Engineering Software 206 (2025) 103922. doi:10.1016/j. advengsoft.2025.103922.

[33] A. Suiker, R. Wolfs, S. Lucas, T. Salet, Elastic buckling and plastic collapse during 3D concrete printing, Cement and Concrete Research 135 (2020) 106016. doi:10.1016/j.cemconres.2020.106016.

[34] R. Wolfs, F. Bos, T. Salet, Triaxial compression testing on early age concrete for numerical analysis of 3D concrete printing, Cement and Concrete Composites 104 (2019) 103344. doi:10.1016/j.cemconcomp.2019.103344.

[35] T. J. R. Hughes, J. Winget, Finite rotation effects in numerical integration of rate constitutive equations arising in large-deformation analysis, Numerical Meth Engineering 15 (12) (1980) 1862–1867. doi:10.1002/nme. 1620151210.

[36] A. S. J. Suiker, Mechanical performance of wall structures in 3D printing processes: Theory, design tools and experiments, International Journal of Mechanical Sciences 137 (2018) 145–170. doi:10.1016/j.ijmecsci. 2018.01.010.

[37] N. Roussel, G. Ovarlez, S. Garrault, C. Brumaud, The origins of thixotropy of fresh cement pastes, Cement and Concrete Research 42 (1) (2012) 148–157. doi:10.1016/j.cemconres.2011.09.004.