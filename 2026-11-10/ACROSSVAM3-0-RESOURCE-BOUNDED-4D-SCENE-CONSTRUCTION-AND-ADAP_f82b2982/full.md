# ACROSSVAM3.0: RESOURCE-BOUNDED 4D SCENE CONSTRUCTION AND ADAPTIVE SURFACE ASSOCIATION

Nan Wu<sup>∗</sup> Hong Yee Hung

Across Physcis (Beijing) Technology Co., Ltd

## ABSTRACT

Combining video generation, dynamic reconstruction and monocular SLAM requires consistent coordinates, explicit handling of missing outputs and predictable memory use. We present ACROSSVAM3.0, an offline system that converts text, images or monocular RGB video into a versioned point representation. Frozen components run sequentially on one GPU; map corrections update cameras, points and trajectories together. The coordination layer adds no neural parameters. Three fresh input-mode runs completed with a maximum sampled task-process-tree GPU footprint of 17.14 GB. In a frozen 16-scene video suite, every scene yielded a 4D reconstruction and 15 yielded a non-empty SLAM map. Strict cross-window matching reduced local surface error but lost long-range track coverage on TAP-Vid-DAVIS. We therefore developed multi-hypothesis adaptive surface chaining (MHAC), which retains several candidate matches and an explicit missing state. After selection on disjoint development data, MHAC improved Average Jaccard over a position-only baseline by 0.1394 on a frozen ten-video holdout (95% bootstrap interval [0.1074, 0.1689]). Across 265 RoboTAP videos, the mean paired gain was 0.0610 ([0.0510, 0.0715]), with 215 wins, one tie and 49 losses. These results support resource-bounded scene construction and an external tracking gain under the tested protocol. They do not establish metric scale, semantic identity or high-fidelity novel-view rendering.

## 1 INTRODUCTION

Video generators, monocular reconstruction models, and learned SLAM systems now provide strong but different views of a dynamic scene. Large video diffusion transformers synthesize plausible motion from text or an image (WanTeam et al., 2025); feed-forward reconstructors infer timevarying geometry and trajectories from monocular RGB (Luo et al., 2026); and learned dense SLAM estimates camera motion, maps, loop closures, and recovery state (Murai et al., 2025). Combining these capabilities could provide a common route from an instruction, a reference image, or an observed video to a dynamic three-dimensional scene. In practice, the combination is fragile. Each component uses its own coordinate frame, scale, preprocessing, confidence convention, and memory profile, while generated frames must not be confused with physical observations.

The difficulty lies in coordinating outputs whose assumptions differ. A local reconstruction may have an arbitrary scale; a SLAM update may change the reference frame; a rejected match may leave a trajectory without support. Generated images also require different provenance from recorded observations. We therefore ask whether a common scene representation can preserve these distinctions, propagate coordinate changes consistently and remain within a single-GPU budget.

We study this problem under a strict operational contract: one process-intensive component at a time, one 40-GB-class GPU per task, RGB-only inference, short offline scenes, and relative rather than metric scale. Figure 1 summarizes the resulting system. Text and image inputs are first converted to RGB video; observed monocular videos bypass generation and additionally enter RGB-only SLAM. A windowed 4D reconstructor produces dense local point trajectories. Robust Sim(3) alignment maps accepted windows into a versioned scene frame, while rejected windows remain explicitly local.

![](images/e2a98a167dc54a9814444c7462a27f2b712691f4f31b76b4c2cfb1a0d7b0b205.jpg)  
Figure 1: System architecture and the separate calibrated rendering extension. (a) Text and image inputs pass through video generation, while observed RGB enters reconstruction and SLAM directly. Frozen components execute sequentially on one GPU. Immutable local 4D windows are aligned through accepted Sim(3) transforms, and pose-version updates keep cameras, points and trajectories consistent. Generated RGB never constrains the physical SLAM map. (b) The highresolution extension instead requires synchronized multiview RGB and supplied calibration. It fits each capture independently and renders observed times from new camera views; it is separate from the monocular evaluation.

Motion evidence, equal-budget trajectory retention, and cross-window surface association operate on the exported representation instead of modifying pretrained networks.

Cross-window association exposes a second question: should temporal consistency be enforced by increasingly strict matches? Our predeclared TAP-Vid-DAVIS study gives a negative answer for the initial method. Aggregating geometry and colour over three overlap frames did not improve Average Jaccard (AJ) over a simpler one-frame rule and reduced the number of valid chains. This failure motivates MHAC (multi-hypothesis adaptive surface chaining), which preserves a bounded candidate lattice and trades geometric precision against coverage before query-specific decoding. We select one configuration on disjoint development data, pass a frozen ten-video holdout gate, and evaluate it once on all 265 RoboTAP videos under a hash-bound protocol.

## The system provides:

1. an executable three-input system that preserves provenance while producing explicit timevarying point geometry, camera trajectories, and RGB-only SLAM artifacts under a measured single-GPU contract;

2. a versioned coordination layer in which local 4D arrays remain immutable and map corrections atomically update cameras, points, trajectories, and coordinate metadata;

3. an evidence-based study of motion ranking, support-aware multi-frame surface correspondence, and fixed-budget trajectory retention on frozen real-video scenes; and

4. a multi-hypothesis association procedure motivated by a failed strict-chain experiment and evaluated on a disjoint holdout and 265 external videos without post-test reselection.

The evaluated profile uses point splats at relative scale and processes short scenes offline. Generated frames never enter the physical SLAM map. A separate calibrated multiview extension is shown in Section 5.1; it changes the input requirements and is not part of the monocular evaluation.

## 2 RELATED WORK

Video generation and 4D reconstruction. Diffusion and flow-based video models have scaled generation through latent representations and transformer backbones (Ho et al., 2022; Rombach et al., 2022; Peebles & Xie, 2023; Lipman et al., 2023; WanTeam et al., 2025). Dynamic scene reconstruction instead seeks geometry and motion that remain coherent across time. Neural fields and Gaussian representations support dynamic novel-view synthesis (Wu et al., 2024; Yang et al., 2024; Lin et al., 2024), while MoSca and Shape of Motion use motion scaffolds or low-dimensional motion bases to consolidate monocular cues (Lei et al., 2025; Wang et al., 2025). 4RC provides feed-forward dense 4D reconstruction and conditional temporal querying (Luo et al., 2026). ACROSSVAM3.0 uses these advances as frozen components; its contribution lies in resource-bounded coordination, failure-preserving evaluation, and surface association rather than a new generative or reconstruction backbone.

Learned geometry and SLAM in dynamic scenes. Classical and learned SLAM systems estimate camera trajectories and persistent maps, with DROID-SLAM demonstrating recurrent dense bundle adjustment and ORB-SLAM3 providing a mature feature-based baseline (Teed & Deng, 2021; Campos et al., 2021). DUSt3R and MASt3R recast matching and reconstruction in learned pointmap space (Wang et al., 2024; Leroy et al., 2024); MASt3R-SLAM builds tracking, local fusion, graph construction, loop closure, and global optimization on these priors (Murai et al., 2025). Dynamicscene systems remove uncertain regions or jointly represent static and moving content (Zheng et al., 2025; Fan et al., 2026). We retain the full SLAM state machine for observed RGB but keep dynamic reconstruction outputs separate from map constraints. This avoids treating a generated or moving surface as evidence for the physical camera trajectory.

Long-range point and surface association. Tracking-any-point benchmarks measure both localization and occlusion across long videos (Doersch et al., 2022). TAPIR, PIPs, CoTracker, and BootsTAP improve temporal refinement, joint tracking, or training data (Doersch et al., 2023; Harley et al., 2022; Karaev et al., 2024; Doersch et al., 2024); RoboTAP demonstrates the relevance of persistent point tracks to robot imitation (Vecerik et al., 2024). Our cross-window problem differs from training a 2D tracker: each node is a dense 3D trajectory inside one reconstruction window, and edges must reconcile separate local coordinate systems. We use TAP-Vid metrics to quantify the resulting projected chains while keeping the graph query-independent until decoding.

Resource-bounded composition. Model compression and efficient adaptation reduce training or deployment cost, but peak memory in a composed system is also controlled by scheduling, activation lifetime, preprocessing, and persistent state. ACROSSVAM3.0 treats the resource envelope as an experimental variable. It records task-process-tree memory separately from unrelated device use, launches large components sequentially, validates every output before adoption, and reports the exact resolution, sequence length, and sampling interval to which each memory statement applies.

## 3 METHOD

## 3.1 PROBLEM FORMULATION AND SCOPE

The input is one of a text prompt $x ^ { \mathrm { t e x t } }$ , an RGB reference image $x ^ { \mathrm { i m g } }$ , or a monocular RGB sequence $X = \{ I _ { t } \} _ { t = 0 } ^ { T - 1 }$ . The system returns a SCENE4D artifact

$$
\boldsymbol { S } = \left\{ \mathcal { W } _ { k } , { \mathbf A } _ { k } , v _ { k } , \mathcal { C } , \mathcal { M } \right\} _ { k = 1 } ^ { K } ,\tag{1}
$$

where $\mathcal { W } _ { k }$ is an immutable local 4D window, $\mathbf { A } _ { k } \in \mathrm { S i m } ( 3 )$ maps that window to its declared scene frame, v<sub>k</sub> is a pose version, $\mathcal { C }$ contains per-frame cameras, and M is optional SLAM state for observed video. Text and image inputs first produce a generated RGB sequence ${ \hat { X } } = G ( x )$ ; they do not produce M. All scales are relative.

The operational contract requires a single physical GPU per task and a sampled task-process-tree peak below 40,000,000,000 bytes. Large neural components run in separate subprocesses and never remain resident concurrently. This contract is a measured property of a fixed profile, not a proof for arbitrary lengths or resolutions.

## 3.2 HETEROGENEOUS INPUT ROUTING

For text and image modes, a frozen 5B Wan2.2 TI2V model produces 17 RGB frames at $8 3 2 \times 4 8 0$ using 12 sampling steps (WanTeam et al., 2025). The text encoder executes on CPU and model weights are offloaded between stages. Image mode records the reference image hash and uses it only as the generator condition. Video mode decodes at most 60 frames and records source timestamps when available. Every frame manifest declares whether pixels are observed or generated. Only observed video enters SLAM.

The pipeline writes a durable stage record before starting each subprocess. A stage is accepted only when it exits successfully, resource logs are present, array shapes and finite values pass validation, and output hashes can be reloaded. Existing output directories are never overwritten. This design keeps implementation failures, preflight rejections, and completed artifacts distinguishable.

## 3.3 WINDOWED EXPLICIT 4D REPRESENTATION

The frozen 4RC backbone processes windows of $L = 1 6$ frames with an overlap of four (Luo et al., 2026). For valid time indices t and dense source pixels i, a local window stores points $\mathbf { p } _ { k , t , i } \in \mathbb { R } ^ { 3 }$ trajectory positions $\tau _ { k , t , i } \in \mathbb { R } ^ { 3 }$ , camera poses $\mathbf { T } _ { w c , t } ^ { ( k ) }$ , intrinsics ${ \bf K } _ { t } ^ { ( k ) }$ , colours, visibility states, and raw confidence scores. The identifier

$$
q _ { k , i } = ( k \ll 3 2 ) + i\tag{2}
$$

is stable inside a window but does not imply semantic identity across windows. Padded inputs needed by the backbone remain in the raw record; only valid frames enter the canonical artifact.

Local arrays are immutable after export. A scene-space point is materialized as

$$
\mathbf { p } _ { k , t , i } ^ { w } = s _ { k } \mathbf { R } _ { k } \mathbf { p } _ { k , t , i } + \mathbf { t } _ { k } ,\tag{3}
$$

where $\mathbf { A } _ { k } = \left( s _ { k } , \mathbf { R } _ { k } , \mathbf { t } _ { k } \right)$ . Camera rotations receive ${ \bf R } _ { k }$ while camera centres receive the full similarity transform. This distinction prevents the scale component from corrupting orthonormal rotations.

## 3.4 SLAM ANCHORING AND ATOMIC MAP UPDATES

Observed RGB runs a pinned MASt3R-SLAM instance before 4D reconstruction (Murai et al., 2025). The SLAM branch exports timestamped camera poses, keyframes, pointmaps, graph edges, loop/relocalization events, and restartable backend state. Retrieval proposals are not reported as loop closures unless the backend accepts their constraints.

Each 4D window is aligned either to accepted SLAM keyframe geometry or to an already anchored overlapping window. Candidate static correspondences are split deterministically: 80% fit a robust Sim(3), and 20% measure held-out reprojection. Alignment is rejected if geometry is degenerate, consensus falls below 35%, spatial coverage falls below 25%, fewer than 20 held-out pairs have positive depth, or held-out median reprojection exceeds 3 pixels. Rejection preserves the window in a named local frame rather than inserting an identity transform and pretending global alignment.

When the SLAM graph changes, common keyframes induce a new window transform $\mathbf { A } _ { k } ^ { \prime }$ . The update is committed only if every affected window passes centre-residual and rotation checks. Cameras, points, trajectories, and metadata then advance from $v _ { k }$ to $v _ { k } + 1$ together. The source arrays and their hashes remain unchanged, so every global state can be audited against the same local reconstruction.

## 3.5 GEOMETRY–MOTION EVIDENCE AND FIXED-BUDGET RETENTION

For trajectory i, we compute a world-coordinate residual score $r _ { i } ^ { 3 d }$ after compensating for camera and accepted window transforms, and an image-motion residual $r _ { i } ^ { 2 d }$ after a global image transform.

Because their scales are incomparable and confidences are uncalibrated, we fuse ranks rather than multiply scores:

$$
\begin{array} { r } { d _ { i } = \frac { 1 } { 2 } \operatorname { r a n k } ( r _ { i } ^ { 3 d } ) + \frac { 1 } { 2 } \operatorname { r a n k } ( r _ { i } ^ { 2 d } ) . } \end{array}\tag{4}
$$

The result is a category-agnostic motion diagnostic, not a semantic segmentation probability.

To retain at most K trajectories, the budget sampler assigns half of the quota to spatial coverage and the remainder to high speed, turning, and visibility-transition events in proportions 0.20, 0.15, and 0.15. Every comparison uses identical K and deterministic seeds. Sampling occurs after reconstruction, so it reduces stored and transmitted trajectory state but does not reduce the backbone’s already incurred memory peak.

## 3.6 FROM STRICT CHAINS TO MULTI-HYPOTHESIS ADAPTIVE ASSOCIATION

For adjacent windows a and b, the initial P4 rule selected a single mutual nearest neighbour using three shared frames. This reduced false matches locally but could terminate an otherwise useful chain. MHAC instead constructs a query-independent directed lattice. For candidate edge $e = ( i , j )$ , the feature vector is

$$
\mathbf { f } _ { e } = \big [ \bar { r } _ { e } , r _ { e } ^ { \operatorname * { m a x } } , \sigma ( r _ { e } ) , u _ { e } , c _ { e } , \rho _ { e } ^ { a \to b } , \rho _ { e } ^ { b \to a } , m _ { e } \big ] ,\tag{5}
$$

containing mean, maximum, and dispersion of reprojection error over four overlap frames; visible support fraction; normalized colour distance; forward and reverse neighbour rank; and the nearest-tosecond-nearest margin. A fixed weighted score

$$
C _ { e } = \mathbf { w } ^ { \top } ( \mathbf { f } _ { e } \oslash \mathbf { s } )\tag{6}
$$

ranks at most three outgoing candidates. The selected C02 configuration uses the eight frozen weights $( 0 . 3 0 , 0 . 1 8 , 0 . 0 8 , - 0 . 1 \bar { 2 } , 0 . \bar { 1 } 6 , 0 . 0 6 , 0 . 0 6 , - 0 . 0 4 )$ , ordinary limits of 2-pixel mean reprojection, 4- pixel maximum reprojection, and 0.15 normalized colour distance, and no rescue edge.

The lattice is sealed before point queries or ground truth are loaded. Each benchmark query then performs bounded dynamic programming with a missing state. Query identities never interact, preventing one labelled track from informing another. Candidate selection used 30 calibration and 10 selection-validation videos from TAP-Vid RGB-Stacking; ten distinct development-holdout videos and all 265 RoboTAP videos were reserved under the protocol described in Section 4.

## 4 EXPERIMENTAL DESIGN

## 4.1 FROZEN COMPONENTS AND EXECUTION PROFILE

The main neural components contain 13,594,090,959 instantiated parameters: 4,999,787,712 in the video diffusion transformer, 5,680,910,336 in the text encoder, 704,688,668 in the VAE, 1,520,065,387 in 4RC, and 688,638,856 in MASt3R. The coordination layer adds no neural parameters and performs no end-to-end training. All weights, source revisions, and file hashes are pinned. The low-memory profile uses 17 generated frames or at most 60 observed frames, 16-frame reconstruction windows, four-frame overlap, and point-splat visualization.

GPU use is sampled every 50 ms. We report both the maximum memory attributable to the task process tree and total device use, and we flag any foreign compute process. This interval can miss shorter transients; all resource statements therefore retain the sampling qualifier. Runs launch only when the allowed device is otherwise idle.

## 4.2 DATASETS AND SPLIT DISCIPLINE

The frozen real-video suite contains 16 short scenes drawn from TUM RGB-D, Bonn dynamic RGB-D, and DyCheck (Sturm et al., 2012; Gao et al., 2022). Ground-truth depth, pose, or motion references are isolated from inference and used only by the evaluator. Each scene contributes one independent unit; frames, pixels, tracks, and overlap folds are nested observations.

Long-range projected tracking uses TAP-Vid metrics at $2 5 6 \times 2 5 6$ (Doersch et al., 2022). P4 used all 30 TAP-Vid-DAVIS videos once under a protocol frozen before payload access. P5 uses the 50 RGB-Stacking videos in a fixed 30/10/10 calibration, selection-validation, and development-holdout partition drawn before data access. TAP-Vid-DAVIS is prohibited from further tuning. The final external set is all 265 RoboTAP videos (Vecerik et al., 2024). Its payload was opened only after the selected method, code, analysis, holdout decision, and protocol hashes were frozen; every video remained in the denominator.

Table 1: Evidence status in the completed frozen study. “Development” indicates data used for design or deterministic selection and cannot support an external-generalization claim.
<table><tr><td>Evidence block</td><td>Independent units</td><td>Status</td><td>Resource peak</td><td>Claim boundary</td></tr><tr><td>Three-input acceptance</td><td>9 runs + 3 fresh final runs</td><td>Complete</td><td>17.14 GB</td><td>Executability of fixed short- scene profile; generation quality separate</td></tr><tr><td>Frozen real-video suite</td><td>16 scenes</td><td>Complete</td><td>16.27 GB</td><td>Point 4D reconstruction, RGB- only SLAM runtime, scene- level diagnostics</td></tr><tr><td>P4 TAP-Vid-DAVIS</td><td>30 videos</td><td>Complete; primary hypothesis failed</td><td>26.54 GB</td><td>External falsification of strict three-frame chain</td></tr><tr><td>P5 RGB-Stacking cali- bration</td><td>30 videos</td><td>Complete; develop- ment</td><td>16.27 GB</td><td>Candidate design only</td></tr><tr><td>P5 selection validation</td><td>10 videos</td><td>Complete; develop- ment</td><td>16.27 GB</td><td>Deterministic candidate selec- tion only</td></tr><tr><td>P5 development holdout</td><td>10 videos</td><td>Complete; gate passed</td><td>16.27 GB</td><td>One-shot authorization of final external test</td></tr><tr><td>P5 RoboTAP</td><td>265 videos</td><td>Complete; external gate passed</td><td>16.27 GB</td><td>Bounded external robot-video tracking gain vs. B0</td></tr></table>

## 4.3 METRICS AND STATISTICAL UNITS

SLAM is summarized by absolute trajectory error (ATE), one-step translation RPE, and one-step rotation RPE when timestamp association meets the frozen threshold. Motion ranking uses ROC AUC and balanced accuracy against a geometric depth-change reference, not semantic labels. Surface association reports held-out reprojection error on correspondences excluded from fitting. Budget sampling reports event recall and spatial coverage at a common K = 8,192.

Point tracking reports AJ, average position accuracy across 1, 2, 4, 8, and 16 pixels, and occlusion accuracy. Macro averages weight videos equally. Confidence intervals use 10,000 scene- or videolevel percentile bootstrap resamples. Because the D25 analysis plan was written after its frozen test outputs were generated, those intervals are descriptive and no p-values are reported. P4 and P5 use their preregistered comparison directions and seeds.

## 5 RESULTS

## 5.1 CALIBRATED MULTIVIEW CAPTURES SUPPORT HIGH-RESOLUTION DYNAMIC VIEWS

The separate extension uses synchronized Panoptic Studio RGB recordings and supplied camera calibration (Joo et al., 2016). It fits a dynamic Gaussian representation independently to each capture, then combines Gaussian appearance with geometry-checked textures from the fitting views. It does not use the monocular SLAM or 4RC pipeline. Figure 2 shows uniformly spaced frames from three completed orbit videos.

The captures contain 360 observed frames each, spanning 12.012 s at 30000/1001 fps. Their fitting/validation camera counts are 27/3, 24/3 and 26/3, respectively. The depicted times and subjects participated in scene fitting; the virtual camera positions are newly rendered views. Validation views were inspected during development. Thus, the images are neither unseen-scene inference nor an independent rendering benchmark. Residual footwear deformation and hand softness remain visible; the demonstration does not establish that perceptual quality is fully resolved.

![](images/d48ac85c65474926662221d803004dba44298361c71fe46757bae14d18bcb44e.jpg)  
Figure 2: Calibrated multiview reconstruction, separate from the monocular experiments. Rows show three separately fitted captures. Columns are frames 60, 180 and 300 of each 360-frame orbit video, at 2.002, 6.006 and 10.010 s; time and camera azimuth advance together. Panels use the same central crop of the native 1920 × 1080 frames, without additional colour adjustment or retouching. Training-view silhouette support reduces some detached splats, but shoe deformation and hand softness remain. Source recordings and calibration: CMU Panoptic Studio.

## 5.2 ALL THREE INPUTS COMPLETED WITHIN THE MEASURED SINGLE-GPU ENVELOPE

The final acceptance run executed text, image, and observed-video inputs sequentially from fresh run directories. Text and image each generated 17 frames and produced two SCENE4D submaps; the 60-frame observed video produced five submaps, a non-empty SLAM map, seven keyframes, and eight accepted graph edges. All exported scenes reloaded with array and hash validation. The maximum sampled task-process-tree footprint was 17,135,828,992 bytes (17.14 GB in decimal units), below both the 36-GB operating target and 40-GB hard contract. Across the earlier nine-run memory acceptance, every mode completed three times without generation or scene caches.

This result establishes execution of the fixed profile rather than quality at arbitrary duration. The generated examples use 17 frames and 12 steps; a separate 121-frame action-quality gate did not pass. The video branch uses at most 60 frames and relative scale. No experiment ran two large components concurrently.

Table 2 reports the measured execution profile for each route.

Table 2: Measured execution of the three input routes. One fresh final acceptance run per route, without generation or scene caches. Time is end-to-end wall time in seconds; GPU memory is the sampled task-process-tree peak in decimal GB. Pass denotes execution and exported-scene validation, not perceptual quality.
<table><tr><td>Input</td><td>Frames</td><td>Submaps</td><td>Time (s)</td><td>GPU (GB)</td><td>Execution</td></tr><tr><td>Text</td><td>17</td><td>2</td><td>262.32</td><td>17.12</td><td>Pass</td></tr><tr><td>Image</td><td>17</td><td>2</td><td>259.59</td><td>17.14</td><td>Pass</td></tr><tr><td>Observed RGB</td><td>60</td><td>5</td><td>155.03</td><td>16.27</td><td>Pass</td></tr></table>

## 5.3 FROZEN REAL VIDEOS EXPOSE BOTH CAPABILITY AND FAILURE

All 16 frozen real-video scenes produced structurally valid 4RC outputs (Figure 3). Fifteen produced non-empty SLAM maps. The remaining scene, dycheck wheel, exited without a usable map and was retained as a failure. Of 12 scenes preregistered for pose evaluation, 11 met timestamp-association requirements. Their median ATE RMSE was 0.0170 m with a descriptive scene-bootstrap 95% interval of [0.0066, 0.0428] m; median translation RPE was 0.0109 m [0.0024, 0.0211], and median mean rotation RPE was 0.718<sup>◦</sup> [0.292<sup>◦</sup>, 1.074<sup>◦</sup>]. The nominal metric units follow the evaluation references and do not convert the complete output to a guaranteed metric-scale reconstruction.

The maximum task-process-tree GPU footprint across these 16 reconstructions was 16,265,510,912 bytes. The single SLAM failure and one unscorable pose case prevent a claim of universal robustness. They also demonstrate why success-rate accounting must remain separate from numeric metrics computed only on valid trajectories.

## 5.4 COORDINATION MODULES IMPROVE TWO NARROW DIAGNOSTICS

The equal-rank motion fusion achieved a median ROC AUC of 0.684 [0.596, 0.769] across 12 scenes and median balanced accuracy of 0.593 [0.559, 0.696] at the frozen 0.65 threshold. One scene had AUC 0.491, an explicit counterexample. Since the frozen test retained only the fused score, this result establishes absolute discrimination against the geometric reference but not superiority over either component score on test data.

Three-frame geometry–colour correspondence reduced scene-level median held-out reprojection error by 0.272 pixels [0.174, 0.381], or 27.1% [19.9%, 30.4%], relative to the one-frame rule. Fifteen of 16 scenes improved. This endpoint is restricted to jointly visible surfaces in adjacent overlapping windows; it does not test object identity across long occlusions.

At the same K = 8,192, motion-diversity sampling increased event recall by a median 0.0993 [0.0967, 0.1035] over uniform sampling and improved all 16 scenes. Median spatial-coverage ratio was 1.215 [1.197, 1.259]. Because this sampler runs after reconstruction, the result supports storageefficient trajectory retention but not lower neural-inference memory.

![](images/8312fe1353b7122478fe5fd856db384fa7a521fc415efca72114895b3ca864b0.jpg)

![](images/698dd2863d513a6bc81732dde4d94908a0cbb24db1ec29c03a5ee1c6172a7fa7.jpg)

![](images/c98d518a5c4b918b65c9c637ee182e6d29c5b0c0268fcf127020c65d67f45e99.jpg)

d  
![](images/28984d67900f55d55ef1bc2e9efb26604e7c628663aa44b123532d4ae697b159.jpg)  
Figure 3: Frozen real-video evaluation establishes a bounded resource and coordination result. (a) Motion-ranking ROC AUC across 12 reference-equipped scenes; the dashed line is chance. (b) Paired reduction in held-out overlapping-surface error for the three-frame geometry–colour rule relative to a one-frame rule. (c) Paired event-recall gain for motion-diversity sampling relative to uniform sampling at $K \ = \ 8 { , } 1 9 2$ . (d) Sampled task-process-tree GPU peak for the 16 4D reconstructions relative to the 40-GB contract. Points are independent scenes; coloured horizontal marks and bands show scene medians and descriptive 10,000-resample percentile intervals where applicable.

## 5.5 EXTERNAL TRACKING FALSIFIES THE STRICT-CHAIN HYPOTHESIS

P4 completed all 30 TAP-Vid-DAVIS videos and 1,999 frames without excluding low-performing cases (Figure 4). In the preregistered first-query mode, the strict three-frame rule obtained AJ 13.21%, compared with 13.26% for the appearance-aware one-frame rule and 14.63% for the positiononly baseline. The paired P4–B1 difference was −0.050 percentage points with a video-bootstrap 95% interval of [−0.391, +0.275] percentage points. The P4–B0 difference was −1.423 points $[ - 2 . 2 8 0 , - 0 . 6 2 6 ]$

The query assignment rate was common across variants, but visible-chain coverage fell from 28.53% for B0 to 22.11% for B1 and 21.47% for P4. This diagnostic is consistent with premature pruning: the strict rule can improve local overlap precision yet lose complete projected tracks. It does not by itself prove that coverage is the only causal mechanism. The maximum task-process-tree GPU footprint was 26,543,652,864 bytes, and two contamination-triggered attempts were preserved separately before clean reruns.

![](images/72a2b418ed5c57e4831c6e779425d527f6943dab45d6f2f37e549c5e7a265089.jpg)

![](images/39604ebe9fd40410108f49c3025e8d228c35199652b29235dc592cf323f45b10.jpg)

![](images/8bbe593b31e17d5f7ab4550559c277a2de9e7b569145121890027f11cf08933d.jpg)  
Figure 4: The preregistered P4 external experiment rejects stricter matching as a sufficient route to long-range tracking. (a) First-query AJ across all 30 TAP-Vid-DAVIS videos. (b) Paired video-level differences for the prespecified comparisons. (c) Query assignment and visible-chain coverage, showing that equal initial assignment does not prevent later chain loss. Intervals resample videos 10,000 times.

## 5.6 FROZEN HOLDOUT AND ROBOTAP TESTS VALIDATE ADAPTIVE ASSOCIATION

The P4 result fixed the design objective for P5: retain a small set of plausible surface edges instead of tightening a single binary rule. Four candidates advanced from the 30-video calibration split and were evaluated without label access during reconstruction or graph construction on ten selectionvalidation videos. All four passed the prespecified coverage, occlusion, and candidate-count guardrails (Figure 5).

The deterministic rule selected C02. Its first-query AJ was 0.4993, with a paired improvement over B0 of 0.1219 and a 95% video-bootstrap interval of [0.0856, 0.1701]; all ten videos improved. Visible-chain coverage was 0.6233, 0.1349 above B0, and occlusion accuracy was 0.7710. C02 and C03 differed by only 0.00035 AJ, so the evidence supports reproducible selection rather than statistical superiority of C02 over every alternative.

The selection result was not used as confirmatory evidence. On the frozen ten-video development holdout, C02 improved AJ over B0 by 0.1394 with a video-bootstrap 95% interval of [0.1074, 0.1689] and improved all ten videos (Figure 6). Macro visible-chain coverage increased by 0.2020. The mean effect, win-count, coverage, and resource checks all passed, authorizing the hash-bound external protocol without candidate reselection.

All 265 RoboTAP videos then completed and remained in the aggregate. Mean first-query AJ increased from 0.1994 for B0 to 0.2604 for MHAC. The paired improvement was 0.0610 (95% interval [0.0510, 0.0715]), with 215 wins, one tie, and 49 losses. Macro visible-chain coverage increased by 0.0685, and the maximum sampled task-process-tree footprint was 16,265,510,912 bytes (15.15 GiB). All four preregistered external checks passed. The 49 losses show material heterogeneity, so this result supports external robot-video point-tracking generalization against B0 under this protocol rather than universal superiority.

## 5.7 THE RELEASE IS REPRODUCIBLE WITHIN ITS FROZEN PROFILE

The source release includes pinned third-party revisions, weight hashes, environment locks, a lightweight wheel, three command-line entry points, reference artifacts, and integrity manifests. A clean launcher installation, artifact reload, and all three entry modes passed; the historical final acceptance recorded 113 passing tests. Subsequent packaging checks recorded 119 passing tests without rerunning the full GPU pipeline. Full CUDA dependency installation from an empty machine image remains outside the completed evidence because the release uses prebuilt pinned environments. Licences remain those of the underlying components, including non-commercial restrictions where applicable.

![](images/349d3501b5cf99a41074078e7f5aae1bdfd505614fa010b9bd2295e45e2defe8.jpg)

![](images/31e5ce410e0e91206c35078f4b1a64c8aba143c84d6a92acb41f10800c5aeb5d.jpg)  
Selection-validation split only (n=10 videos); all candidates and videos retained.

Figure 5: P5 candidate selection on ten development videos. (a) Paired first-query AJ improvement over the position-only B0 baseline; error bars are 10,000-resample video-bootstrap 95% intervals. (b) visible-chain coverage improvement over B0. C02 was selected by the frozen lexicographic rule. These are development-selection results, not holdout or external-test evidence.  
![](images/c69754959486131daef927eb2273e66dd88f558796352d95613b9c267ef744ca.jpg)

![](images/b7b526d292e805506024814f5182104ec019409c6e7e85024947d930e4c6bf81.jpg)

![](images/bf80cbbacc4e730de863c5ef3f2d2f8a7c385d7b0fc5d43856ff9eb99379bfd7.jpg)  
All videos retained; intervals use 10,000 video-level bootstrap resamples. Blue marks positive paired effects.

Figure 6: Frozen confirmatory tests support a bounded external gain for MHAC. (a) Every development-holdout video improves over B0. (b) Paired RoboTAP AJ for all 265 videos; the dashed identity line separates wins and losses. (c) Sorted paired effects expose both the positive mean and the retained negative cases; the orange line and band show the mean and its 10,000-resample video-bootstrap 95% interval. No video was removed after evaluation.

## 6 DISCUSSION

The results distinguish local correspondence accuracy from complete-track coverage. Three-frame matching lowered overlap reprojection error in the frozen scene suite, yet performed worse than the position-only baseline on TAP-Vid-DAVIS. The endpoints measure different properties: overlap error conditions on supported matches, whereas AJ also penalizes missing tracks. Retaining several candidates is useful when a locally plausible match should not determine the entire chain.

The RoboTAP gain is consistent with this interpretation. Visible-chain coverage increased by 6.85 percentage points alongside the AJ improvement. However, MHAC changes candidate retention, missing-state decisions and occlusion behaviour together, so the experiment does not isolate coverage as the sole cause. The 49 videos favouring B0 also show that the selected rule is not uniformly better. A future mechanism study would need fresh evaluation data and separate interventions on these decisions.

The systems result concerns coordinate consistency and resource use. Immutable local arrays preserve the reconstruction evidence, while versioned transforms keep map corrections consistent across cameras and trajectories. Sequential execution permits the frozen components to share one

GPU. This arrangement is useful for offline inspection and reuse, but it does not remove uncertainty inherited from the components.

The evaluated system has several limits. Text and image runs establish execution for 17 frames; the separate 121-frame action-quality gate failed. Point splats do not establish high-fidelity novel views, and projected tracking does not establish semantic identity after occlusion. The external comparison is restricted to B0 on the frozen robot-video protocol, without a comprehensive comparison to specialized trackers. Resource measurements are sampled every 50 ms and may miss shorter peaks. The calibrated Gaussian extension provides a separate view-synthesis demonstration from dense multiview inputs, with unresolved footwear artefacts; it does not close these monocular evidence gaps.

## 7 CONCLUSION

ACROSSVAM3.0 produces versioned dynamic point scenes from three input routes and adds RGBonly SLAM for observed video. The tested short-scene profile ran within the measured single-GPU budget. The association experiments show why local geometric precision alone is insufficient: strict matching lost long-range coverage, while a bounded candidate lattice improved mean AJ over B0 on the frozen RoboTAP evaluation. The retained failures define the present scope of the result and the cases that require further study.

## AI USE STATEMENT

AI tools were used solely to review experimental code and polish the language of the manuscript. The research ideas, experimental design and initial manuscript draft were developed independently by the authors. The authors take full responsibility for the accuracy and integrity of the final work.

## REFERENCES

Carlos Campos, Richard Elvira, Juan J. Gomez Rodriguez, Jose M. M. Montiel, and Juan D. Tardos. ORB-SLAM3: An accurate open-source library for visual, visual–inertial, and multimap SLAM. IEEE Transactions on Robotics, 37(6):1874–1890, 2021.

Carl Doersch, Ankush Gupta, Larisa Markeeva, Adria Recasens, Lucas Smaira, Yusuf Aytar, Joao Carreira, Andrew Zisserman, and Yi Yang. Tap-vid: A benchmark for tracking any point in a video. In Advances in Neural Information Processing Systems, volume 35, 2022. URL https://proceedings.neurips.cc/paper\_files/paper/ 2022/hash/58168e8a92994655d6da3939e7cc0918-Abstract.html.

Carl Doersch, Yi Yang, Mel Vecerik, Dilara Gokay, Ankush Gupta, Yusuf Aytar, Joao Carreira, and Andrew Zisserman. Tapir: Tracking any point with per-frame initialization and temporal refinement. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 10061–10072, 2023. URL https://openaccess.thecvf.com/content/ ICCV2023/html/Doersch\_TAPIR\_Tracking\_Any\_Point\_with\_Per-Frame\_ Initialization\_and\_Temporal\_Refinement\_ICCV\_2023\_paper.html.

Carl Doersch, Pauline Luc, Yi Yang, Dilara Gokay, Skanda Koppula, Ankush Gupta, Joseph Heyward, Ignacio Rocco, Ross Goroshin, Joao Carreira, and Andrew Zisserman. Bootstap: Bootstrapped training for tracking-any-point. In Proceedings of the Asian Conference on Computer Vision, pp. 3257–3274, 2024. URL https://openaccess.thecvf.com/ content/ACCV2024/html/Doersch\_BootsTAP\_Bootstrapped\_Training\_ for\_Tracking-Any-Point\_ACCV\_2024\_paper.html.

Zhicheng Fan, Zitong Wu, Zhaoxing Fan, Xiao Zhang, Biao Hou, and Bo Ren. De4D-SLAM: Gradient-isolated static-dynamic decoupling for monocular SLAM in dynamic environments. In Proceedings ofthe 43rd International Conference on Machine Learning, volume 306 of Proceedings ofMachine Learning Research, pp. 28883–28903, 2026.

Hang Gao, Ruilong Li, Shubham Tulsiani, Bryan Russell, and Angjoo Kanazawa. Monocular dynamic view synthesis: A reality check. In Advances in Neural Information Processing Systems, volume 35, 2022.

Adam W. Harley, Zhaoyuan Fang, and Katerina Fragkiadaki. Particle video revisited: Tracking through occlusions using point trajectories. In European Conference on Computer Vision, pp. 59–75. Springer, 2022. URL https://arxiv.org/abs/2204.04153.

Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J. Fleet. Video diffusion models. In Advances in Neural Information Processing Systems, volume 35, 2022. URL https://proceedings.neurips.cc/paper/2022/hash/ 39235c56aef13fb05a6adc95eb9d8d66-Abstract-Conference.html.

Hanbyul Joo, Tomas Simon, Xulong Li, Hao Liu, Lei Tan, Lin Gui, Sean Banerjee, Timothy Godisart, Bart Nabbe, Iain Matthews, Takeo Kanade, Shohei Nobuhara, and Yaser Sheikh. Panoptic studio: A massively multiview system for social interaction capture. arXiv preprint arXiv:1612.03153, 2016. URL https://arxiv.org/abs/1612.03153.

Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Cotracker: It is better to track together. In European Conference on Computer Vision, 2024. URL https://arxiv.org/abs/2307.07635.

Jiahui Lei, Yijia Weng, Adam W. Harley, Leonidas Guibas, and Kostas Daniilidis. MoSca: Dynamic gaussian fusion from casual videos via 4d motion scaffolds. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6165–6177, 2025.

Vincent Leroy, Yohann Cabon, and Jerome Revaud. Grounding image matching in 3d with MASt3R. arXiv preprint arXiv:2406.09756, 2024.

Youtian Lin, Zuozhuo Dai, Siyu Zhu, and Yao Yao. Gaussian-flow: 4d reconstruction with dynamic 3d gaussian particle. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21136–21145, 2024.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=PqvMRDCJT9t.

Yihang Luo, Shangchen Zhou, Yushi Lan, Xingang Pan, and Chen Change Loy. 4rc: 4d reconstruction via conditional querying anytime and anywhere. In Proceedings of the 43rd International Conference on Machine Learning, 2026.

Riku Murai, Eric Dexheimer, and Andrew J. Davison. MASt3R-SLAM: Real-time dense SLAM with 3D reconstruction priors. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4195–4205, 2023. URL https://openaccess.thecvf.com/content/ICCV2023/html/Peebles\_ Scalable\_Diffusion\_Models\_with\_Transformers\_ICCV\_2023\_paper.html.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10684–10695, 2022. URL https://openaccess.thecvf.com/content/CVPR2022/html/Rombach\_ High-Resolution\_Image\_Synthesis\_With\_Latent\_Diffusion\_Models\_ CVPR\_2022\_paper.html.

Jurgen Sturm, Nikolas Engelhard, Felix Endres, Wolfram Burgard, and Daniel Cremers. A benchmark¨ for the evaluation of RGB-D SLAM systems. In IEEE/RSJ International Conference on Intelligent Robots and Systems, pp. 573–580, 2012.

Zachary Teed and Jia Deng. DROID-SLAM: Deep visual SLAM for monocular, stereo, and RGB-D cameras. In Advances in Neural Information Processing Systems, volume 34, 2021.

Mel Vecerik, Carl Doersch, Yi Yang, Todor Davchev, Yusuf Aytar, Guangyao Zhou, Raia Hadsell, Lourdes Agapito, and Jon Scholz. RoboTAP: Tracking arbitrary points for few-shot visual imitation. In IEEE International Conference on Robotics and Automation, 2024.

Qianqian Wang, Vickie Ye, Hang Gao, Weijia Zeng, Jake Austin, Zhengqi Li, and Angjoo Kanazawa. Shape of motion: 4d reconstruction from a single video. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 9660–9672, 2025.

Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. DUSt3R: Geometric 3d vision made easy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

WanTeam, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. URL https://arxiv.org/abs/2503.20314.

Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 20310–20320, 2024.

Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 20331–20341, 2024.

Jianhao Zheng, Zihan Zhu, Valentin Bieri, Marc Pollefeys, Songyou Peng, and Iro Armeni. WildGS-SLAM: Monocular gaussian splatting SLAM in dynamic environments. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11461–11471, 2025.

## A ARTIFACT AND CLAIM MAP

Table 3 maps each principal statement to its authoritative artifact. Development records are kept separate from frozen-test and external records.

Table 3: Authoritative evidence artifacts for the reported results.
<table><tr><td>Claim</td><td>Artifact</td><td>Evidence class</td></tr><tr><td>Three-input final acceptance</td><td>reports/phase3/d30_final_acceptance. json</td><td>Final engineering accep- tance</td></tr><tr><td>Frozen 16-scene metrics</td><td>reports/phase3/d25_statistics.json and case CSV</td><td>Frozen test; descriptive in- ference</td></tr><tr><td>P4 external negative result</td><td>reports/phase4/p4_tapvid_aggregate. json</td><td>Preregistered external test</td></tr><tr><td>P5 candidate selection</td><td>reports/phase5/p5_mhac_selection_ validation.json</td><td>Development selection</td></tr><tr><td>P5 frozen holdout</td><td>reports/phase5/p5_mhac_development_ holdout.json</td><td>One-shot development holdout</td></tr><tr><td>P5 RoboTAP external result</td><td>reports/phase5/p5_robotap_external_ aggregate.json</td><td>Frozen external test</td></tr><tr><td>P5 completion audit</td><td>reports/phase5/p5_final_completion_ audit.json</td><td>Protocol and execution au- dit</td></tr><tr><td>Model count and provenance</td><td>models_manifest.json</td><td>Hash-verified implementa- tion record</td></tr></table>

## B SCENE4D FIELD SEMANTICS

Each local submap stores timestamps, frame identifiers, $\mathbf { T } _ { w c } ,$ , intrinsics, image size, dense points, point confidence, RGB, dense trajectories, trajectory identifiers, colour, trajectory confidence, and ternary visibility. Confidence values are raw model scores and are not interpreted as calibrated probabilities. Visibility denotes projection/depth support, not ground-truth occlusion. A staticprobability value of −1 denotes unknown. These conventions are validated on load and preserved in the scene manifest.

## C FAILURE AND MISSINGNESS POLICY

A task can end as completed, failed, blocked by a frozen prerequisite, or contaminated by an unrelated GPU process. Failures remain in task counts. A contaminated attempt may be rerun only under a rule frozen before its outcome is known, and both attempts remain recorded. Numeric summaries do not impute missing trajectories or empty maps. Predeclared denominator units are scenes or videos, never frames or pixels.

## D CONFIRMATORY EVALUATION AUDIT

The final P5 result was admitted only after the frozen completion audit verified the following conditions:

1. the ten-video holdout passed its four-part gate before external payload access;

2. the final protocol bound the selected method, holdout report, input inventory, analysis seed, and hashes;

3. all 265 admitted RoboTAP units completed and remained in the aggregate;

4. inference and labels were isolated, task processes used one GPU each, and the sampled peak remained below 40 GB; and

5. the reported bootstrap interval reproduces under the frozen seed and no candidate was reselected after external evaluation.

All 18 machine-readable completion checks passed. The audit records retries, raw-output pruning, proxy state, resource logs, and per-video report hashes; it does not convert the external result into evidence for claims outside the specified protocol.

## E EXACT WORKING-PROFILE CONFIGURATION

The current released profile uses one allowed physical GPU, no concurrent large-model process, 832 × 480 generation, 17 frames, 12 sampling steps, CPU text encoding, model offload, 60 maximum observed frames, 10 frames per second for un-timestamped inputs, 16-frame reconstruction windows, four-frame overlap, 512-pixel reconstruction size, and point-splat rendering. The hard task limit is 40,000,000,000 bytes and the operating target is 36,000,000,000 bytes.