# Reflection with Action-Induced Visual Differences for Desktop GUI Agents

Yijie Ma, Chaoyue Niu, Fan Wu, Guihai Chen Shanghai Jiao Tong University

## Abstract

The Planner-Operator-Reflector (POR) framework is widely used in GUI agents to maintain objective alignment in complex tasks through modular collaboration. However, desktop GUIs introduce a key challenge: large, dense interfaces often exhibit subtle or scattered state changes, placing most of the burden on the reflector, which must compare pre- and post-action screens, while the planner and operator reason over a single state. Existing reflectors collapse change detection and outcome verification into one step, leaving evidence implicit and yielding weakly grounded decisions. To address this limitation, we propose Evidence-First Reflection (EFR), a two-stage reflector that explicitly decouples action-induced visual differences extraction from outcome verification. EFR identifies the action location and candidate changed regions with Set-of-Marks annotations, describes and filters action-relevant changes, and makes the final judgment from the cleaned evidence. This evidence-reasoning decoupled design makes reflection better grounded in screen transitions, while reducing both visual search complexity and reasoning burden. Experiments on OSWorld-Verified and WindowsAgentArena demonstrate that EFR improves reflector accuracy by 7.11%, yielding average end-to-end task success gains of 5.94% and 4.95% on the two benchmarks, respectively.

## 1 Introduction

Graphical User Interface (GUI) agents [1, 8, 20, 22, 25, 26, 32, 37, 38] are designed to execute user instructions by interacting with digital interfaces in a human-like manner, typically leveraging visionlanguage models (VLMs). By unifying the observation space into screenshots and the action space into human-like operations, GUI agents can handle diverse tasks across both mobile and desktop environments [6, 32]. This broader task scope introduces higher complexity, motivating existing studies to adopt the mainstream Planner-Operator-Reflector (POR) framework [1, 2, 13, 28, 36], which decomposes agent control into collaborative modules to maintain task alignment and reduce the burden on any single module. The planner formulates a high-level plan, the operator executes the corresponding low-level action, and the reflector assesses whether the action has successfully achieved the intended goal. Serving as a critical safeguard in this workflow, the reflector helps ensure action accuracy and prevents error accumulation over steps. It typically performs this assessment in a single inference pass by observing the pre- and post-action screenshots and comparing the observation with the intended action to generate a judgment [2, 15, 24, 28, 36].

However, reflection in desktop environments is generally more challenging than in mobile environments, as desktop screens are often several to even dozens of times larger. Such expansive interfaces tend to exhibit more complex layouts and a richer set of elements, making it difficult to comprehensively detect and interpret GUI changes [1, 15, 21, 37]. This challenge disproportionately affects the reflector compared to the planner and operator: while the latter mainly need to reason about the current interface state, the reflector must jointly analyze both pre- and post-action screenshots to determine whether an action has succeeded. Figure 1 shows two representative examples: one where the action outcome is extremely subtle, and the other where the action induces multiple, spatially distributed changes. In both cases, the reflector struggles to capture and organize the complete set of interface changes, leading it to judge the action outcome based on only a limited and potentially incorrect subset of changes. In our experiment with GUI-Owl-32B trajectories using the conventional POR framework on OSWorld-Verified [30, 31], 61 out of 223 failed trajectories (27.35%) are caused by incorrect reflector judgments. Among these reflection failures, 43 cases (70.49%) stem from page difference identification errors, while 18 cases (29.51%) result from success judgment errors given correctly identified page differences, as detailed in Appendix Table 4. This indicates that the conventional reflector is not sufficiently reliable in complex desktop environments.

![](images/e7cac4fdc345fbbeb64dc19471e23bc4ee04ed65b63c52f4488a946c86ee865a.jpg)  
(a) Only a cursor change.

![](images/9f644f60655e30c5a1a2f44eb1fe334602d8f2989f9cc237e4fc93bc541bef9a.jpg)  
(b) Many page regions change simultaneously.

Figure 1: Two representative examples of interface changes in desktop environments. Fig. 1a: the action outcome is indicated by a subtle cursor movement after clicking the position of the letter “a”, making the change easy to overlook. Fig. 1b: switching the year triggers extensive updates across the interface, making it challenging to organize all relevant evidence to assess the action’s effect.  
![](images/a3dc053badf326ed55aa417df4fcaf5d025ee10168cd1b0808bcc8b72805a6f9.jpg)  
Figure 2: The conventional reflector performs one-step reasoning to judge action success directly, whereas EFR first describes page changes from SoM annotations as explicit evidence and then verifies the action based on the extracted evidence.

By deconstructing the logic of conventional reflectors, we find that they always couple two operations with different roles: first establishing what changed as observable evidence, and then interpreting whether the evidence supports a successful action. When these operations are coupled into one inference step, the evidence used for the final judgment remains implicit and cannot be checked before the VLM reaches a conclusion. To address this limitation, we revisit the internal design of the reflector itself and propose Evidence-First Reflection (EFR), a two-stage reflector that explicitly decouples visual evidence extraction from action outcome verification. It mitigates potential bias by decoupling the objective observation of page changes from subjective assumptions regarding intended action outcomes, while reducing the visual burden by repurposing Set-of-Marks (SoM), originally used to label candidate page elements, to annotate page changes before and after the action. Figure 2 shows the core difference between conventional reflector and EFR. Specifically, EFR first leverages SoM to annotate action-induced change regions and the grounded action location, prompting the VLM to objectively describe and filter action-relevant page changes as evidence. Then, it judges whether the action succeeded based on the cleaned evidence and the intended action. By separating “what changed” from “whether the action succeeded”, EFR reduces both the visual burden and the reasoning burden of reflection. In summary, the key contributions include:

• We observe that the inherent complexity of desktop environments substantially complicates change identification, thereby compromising the reliability of conventional reflectors that infer action success directly in a single reasoning step.

• We reconstruct the reflector as a two-stage sequential process that explicitly separates page difference identification from action success judgment. The first stage describes and organizes actionrelevant visual changes from SoM annotations, and the second stage verifies action success based on the extracted evidence, making the dependency between evidence construction and outcome judgment explicit. This design grounds the final judgment in a checkable evidence chain and reduces interference between visual search and outcome reasoning.

• We implement the POR framework following Mobile-Agent-v3 [36] and evaluate EFR with both GUI-specialized and general backbones on OSWorld-Verified [30, 31] and WindowsAgentArena [4] benchmarks. Averaged over three runs, EFR improves end-to-end task success rates across all model-benchmark settings, with average gains of 5.94% and 4.95%, respectively. Further analysis demonstrates that the performance gains are strongly correlated with improvements in reflector accuracy. Ablation studies validate the necessity of each module. Notably, the performance improvement comes with moderate per-task latency overheads, averaging 15.8% and 13.3%, respectively.

## 2 Related Work

POR Framework. The key of POR framework is to explicitly decouple planning, execution, and feedback verification [18, 23, 24]. This paradigm shares a lineage with the reasoning-acting loop exemplified by ReAct [35]. Many studies adopt this paradigm and further refined it to better handle complex tasks. For instance, Mobile-Agent-v3 [36] introduces Planner-Operator-Reflector-Notetaker framework to handle multi-step navigation and error correction. Mobile-Agent-E [28] and MobileUse [13] enhance hierarchical planning and introduce an on-demand, reflection-driven workflow. Agent-S2 [2] uses dynamic collaboration to plan various modules. While these studies largely retain the POR framework as-is and enhance it with additional components or strategies, our work revisits its internal modular design by reconstructing the reflector module.

Reflector. Reflector is typically used to convert failures into reusable memory or to support hierarchical closed-loop correction [7, 18, 23]. In GUI agents, the use of reflectors has been further extended within the overall agent pipeline: MobileUse [13] introduces hierarchical reflection to identify and correct errors across different granularities, InfiGUIAgent [16] and D-Artemis [19] perform reflection at different stages of the action execution process, and BacktrackAgent [29] leverages reflection for trajectory rollback. Their improvements are largely about deploying the reflector in different roles or stages of the agent pipeline. As a result, most approaches still implicitly assume that the reflector can operate as a single-step module. In contrast, our work focuses on its internal reasoning logic and improves it by explicitly decoupling evidence extraction from success judgment.

SoM in GUI Agent. SoM annotates interface content by overlaying explicit markers, such as letters, numbers, or bounding boxes, on screenshots [33]. In prior GUI agents, it was used to identify candidate elements before taking actions [6, 9, 12, 14, 17, 34]. For instance, AppAgent [38] overlays semi-transparent numeric markers onto candidate elements parsed from XML on mobile screenshots, enabling the model to execute operations using numeric indices rather than raw coordinates. UFO [37] simultaneously provides the agent with a raw screenshot and an annotated screenshot, encoding UI controls with numbers and colors. In addition to these specific methods, environments like VisualWebArena [11] and OSWorld [31] also employ standardized element-level SoM observation representations to facilitate element recognition. In contrast, we extend SoM beyond target-element localization and apply it to post-action page-change identification within the reflection module.

## 3 Problem Formulation

We first describe the workflow of the conventional POR framework, which employs a closed-loop iterative process to complete tasks. Within a single execution round, the Planner first formulates a comprehensive plan and decomposes it into discrete sub-goals based on the current state and overall task goal. Subsequently, the Operator receives the active sub-goal and generates a grounded action for execution. Finally, the Reflector verifies whether the executed action achieves the intended goal by comparing pre- and post-action screenshots. Formally, at each step t, the operations of the modules are defined as follows:

Planner. Given the task goal G, the Planner leverages the current screenshot $S _ { t }$ together with the previous state, comprising the action history $H _ { < t }$ and the reflection from the preceding step $R _ { t - 1 }$

![](images/dc57013c2e788fb1bcb969d439ca99f67e5349dc06138b26d310b1d53322f760.jpg)  
Figure 3: Overview of EFR in POR framework. Given the pre-action screenshot $S _ { p r e } ,$ post-action screenshot $S _ { p o s t } ,$ , grounded action description $A _ { g r o u n d } ,$ and intended action description $A _ { i n t e n d } ,$ EFR first annotates candidate visual differences and the action location with SoM, then filters actionrelevant evidence, and finally judges action success based on the cleaned evidence.

to derive the next sub-goal $P _ { t }$

$$
P _ { t } = f _ { \mathrm { p l a n n e r } } ( G , S _ { t } , H _ { < t } , R _ { t - 1 } ) .\tag{1}
$$

Operator. The Operator then grounds the sub-goal $P _ { t }$ in the current visual observation $S _ { t }$ and outputs the executable action $A _ { t }$ :

$$
A _ { t } = f _ { \mathrm { o p e r a t o r } } ( P _ { t } , S _ { t } ) .\tag{2}
$$

Reflector. Finally, after execution, the Reflector compares the pre-action screenshot $S _ { t } ^ { p r e }$ with the post-action screenshot $S _ { t } ^ { p o s t }$ to assess the outcome of the action $A _ { t }$ and produce the reflection $R _ { t } \mathrm { : }$

$$
R _ { t } = f _ { \mathrm { r e f l e c t o r } } ( P _ { t } , A _ { t } , S _ { t } ^ { p r e } , S _ { t } ^ { p o s t } ) .\tag{3}
$$

Despite the modular decomposition of the POR framework, the reflection phase remains a monolithic challenge. To produce a reliable assessment, the Reflector must simultaneously address two distinct objectives: (1) identifying visual transformations between pre- and post-action screenshots, (2) determining whether these changes satisfy the intended goal. Conventional methodologies typically process these steps in a single pass by directly feeding the action description and both screenshots into the VLM. However, this design is brittle in desktop environments where screenshots are high-resolution and information-dense. Specifically, subtle changes are often overlooked, while intricate changes can lead to fragmented analysis and hasty conclusions. To address these limitations, it is essential to equip the reflector with the ability to comprehensively perceive the page changes, and to assess action correctness strictly on the basis of the observed state transition.

## 4 Method

In this section, we introduce EFR, a two-stage evidence-based reflection framework designed to extract salient features under complex visual transformations and formulate judgments grounded in explicit evidence. Figure 3 illustrates the overall framework, with detailed descriptions of each component provided in the subsequent subsections. Within the POR loop, EFR replaces the conventional reflector: after the planner generates a plan and the operator executes the corresponding action, EFR evaluates whether the action produced correct result and feeds the reflection result back into the next planning step. Rather than directly inferring action succeeded from raw pre- and postaction screenshots, EFR explicitly decouples visual change recognition from outcome verification. This approach is built upon a twofold strategy: first, we preprocess the screenshots to reduce the perceptual burden associated with detecting subtle or distributed interface changes; second, we implement a two-stage reasoning pipeline to mitigate the cognitive complexity of success evaluation. Specifically, EFR utilizes SoM to annotate both the grounded action location and visual differences between the screenshots, thereby constructing a foundation of explicit visual evidence. Then it systematically describes key changes and filters out environmental noise. By synthesizing this refined evidence with the intended goal, EFR determines the correctness of the executed action. Appendix B presents an example, illustrating the comparison between the conventional reflector and EFR.

![](images/462ec994d91eba8ead421473adf0ccba659bba7d001d3960cf9489ce9a221d5f.jpg)  
(a) Pre-action marked screenshot

![](images/6cf5311a11135fdb9842ffeef2393c37ca73e9887c50f5198a0aaedff92d04c9.jpg)  
(b) Post-action marked screenshot  
Figure 4: A concrete example of SoM annotation. Fig. 4a shows the pre-action marked screenshot, where the executed action location and difference regions are annotated. Fig. 4b shows the postaction marked screenshot, where difference regions are annotated.

## 4.1 Page Changes Annotated With SoM

Identifying page changes is a specialized perceptual task for which general-purpose VLMs often underperform, whereas task-specific computer vision methods are more effective and efficient. Instead of relying on the VLM to detect changes unaided, we first apply the SoM technique to explicitly annotate visual differences. Existing SoM methods were typically designed to assist target-element localization during action execution. We repurpose this idea for the reflection stage to support pagechange analysis. This step explicitly converts dispersed differences between two screenshots into consolidated, structured evidence that the VLM can directly reference: changed regions are marked with red bounding boxes and indexed for region-level comparison. In addition, we annotate the action coordinates with blue circles to provide spatial context regarding where the action was executed. These explicit visual markers allow the VLM to directly locate salient changes on the page and reason over them as structured visual evidence.

Given the pre-action screenshot $S _ { p r e }$ , the post-action screenshot $S _ { p o s t } .$ , and the grounded action $A _ { g r o u n d }$ which contains action type and coordinates, we first identify candidate changes through pixel-wise comparison. Nearby changed regions are then merged into coherent visual-change areas, which are finally mapped to rectangular bounding boxes. These boxes and their corresponding indices are then overlaid onto both screenshots. In addition, the action coordinates are annotated on the pre-action screenshot. This procedure yields two intermediate representations: $S _ { n r e } ^ { s o m }$ , which depicts the pre-action screenshot marked with change areas and action coordinates, and $S _ { p o s t } ^ { s o m }$ , which only displays change areas on the post-action screenshot:

$$
( S _ { p r e } ^ { s o m } , S _ { p o s t } ^ { s o m } ) = f _ { \mathrm { s o m } } ( S _ { p r e } , S _ { p o s t } , A _ { g r o u n d } ) .\tag{4}
$$

By incorporating SoM, we explicitly indicate the locations of visual changes. An illustrative example is presented in Figure 4. Further details regarding the identification and annotation process are provided in the Appendix C. This approach reduces the perceptual load, thereby enabling the VLM to perform localized comparisons within the annotated regions.

## 4.2 Action-Induced Page Difference Analysis and Filtering

After obtaining the annotated screenshots, the next phase involves an intermediate step aimed at extracting explicit visual evidence. An illustrative example is presented in Figure 5. During this stage, the VLM is provided with the marked pre- and post-action screenshots $( \bar { S } _ { p r e } ^ { s o m } , S _ { p o s t } ^ { s o m } )$ and the grounded action $A _ { g r o u n d }$ . Importantly, $A _ { g r o u n d }$ contains only the action type and its coordinates, rather than the original intended action $A _ { i n t e n d } .$ . For example, the intended action may be “click the button in the upper-left corner,” whereas the grounded action is represented as “click at coordinate $( x , y ) . ^ { \mathsf { \prime } }$ This design encourages the reflector to extract evidence from the actual execution result rather than from the expected intention, reducing intention bias and improving the objectivity of the intermediate visual evidence.

Given these inputs, the VLM generates explicit visual evidence in the form of region-level change descriptions. For each annotated region, it compares the corresponding visual contents in the marked pre- and post-action screenshots and independently describes the observed change. Since all detected changes are annotated, these regions may also contain incidental system noise, such as clock refreshes. Therefore, during this region-wise traversal, the VLM further distinguishes whether each described change reflects system noise or a meaningful state transition. Finally, it outputs the set of change descriptions C, categorized into system noise $\mathcal { N }$ and action relevant changes $\bar { \mathcal { E } } \mathrm { : }$

![](images/651444e419691820ad105c0eca707bc26ed0961d13b6a30c0af726ffca6edeff.jpg)  
Figure 5: From total changes to clean evidence. The reflector first compares the annotated preaction and post-action screenshots and outputs change descriptions together with relevant/noise tags. In this example, the bottom right Area 5 is filtered out as environment noise. The remaining relevant regions are then redrawn to form the clean evidence pair.

$$
( C , N , \mathcal { E } ) = f _ { \mathrm { d e s c } } ( S _ { p r e } ^ { s o m } , S _ { p o s t } ^ { s o m } , A _ { g r o u n d } ) .\tag{5}
$$

We retain only the descriptions and markers associated with meaningful regional changes while filtering out system noise. This procedure requires re-labeling the original screenshots, $S _ { p r e }$ and $S _ { p o s t }$ , to derive $S _ { p r e } ^ { c l e a n }$ and $S _ { p o s t } ^ { c l e a n }$ , which isolate non-noise changes for use in the subsequent success judgment phase. We also filter out noise-related descriptions from $C$ and obtain $C ^ { c l e a n }$

$$
( S _ { p r e } ^ { c l e a n } , S _ { p o s t } ^ { c l e a n } , C ^ { c l e a n } ) = f _ { \mathrm { c l e a n } } ( S _ { p r e } , S _ { p o s t } , \mathcal { E } , C ) .\tag{6}
$$

## 4.3 Action Outcome Verification Based on Evidence

Upon obtaining the filtered evidence, the final stage involves determining action success. At this step, the screenshots $S _ { p r e } ^ { c l e a n }$ and $S _ { p o s t } ^ { c l e a n }$ , the filtered change description $\mathsf { \bar { C } } ^ { c l e a n }$ , and the original intended action $A _ { i n t e n d }$ are jointly provided to the reflector. Based on this filtered visual evidence, the VLM evaluates whether the action has achieved the intended objective and outputs the reflection $R ,$ which consists of a success label, key evidence, and a status summary for downstream modules:

$$
R = f _ { \mathrm { j u d g e } } ( S _ { p r e } ^ { c l e a n } , S _ { p o s t } ^ { c l e a n } , C ^ { c l e a n } , A _ { i n t e n d } ) .\tag{7}
$$

## 5 Evaluation

## 5.1 Evaluation Setup

Benchmarks. We use two benchmarks across diverse environments for computer-based agents: OSWorld-Verified [30, 31] and WindowsAgentArena [4]. OSWorld-Verified comprises 369 real-world Linux system tasks. In accordance with official guidelines and recent evaluations [21, 27, 31, 36], we exclude 8 Google Drive-related tasks that require manual configuration, resulting in a 361-task subset. This evaluation spans 10 application domains, including Chrome, GIMP, LibreOffice, Thunderbird, VLC, and VSCode. WindowsAgentArena consists of 154 Windows system tasks encompassing various applications; we include the complete task set for our analysis. Appendix D shows the task type.

Baselines and Models. We adopt the conventional POR framework as our baseline, adhering to the implementation established in Mobile-Agent-v3 [36]. Our method replaces the standard reflector module with the proposed EFR variant. To ensure a comprehensive evaluation across diverse architectures, we select both GUI-specialized and general-purpose backbones from open-source and proprietary families. Specifically, we benchmark both POR and POR-EFR using four models: GUI-Owl-32B [36], Qwen3-vl-32B-instruct [3], Kimi-K2.5 [10], and Seed-1.8 [5].

Table 1: Mean success rates (%) over three runs on the OSWorld-Verified benchmark. Rows correspond to task types. For each model backbone, the table reports the mean POR result, the mean EFR result, and their absolute difference ∆ in percentage points. The best mean accuracy in each row is bolded; ties are bolded together.
<table><tr><td>Task Type</td><td colspan="3">GUI-Owl-32B [36]</td><td colspan="3">Qwen3-VL-32B [3]</td><td colspan="3">Kimi-K2.5 [10]</td><td colspan="3">Seed-1.8 [5]</td></tr><tr><td></td><td>POR</td><td>EFR</td><td>∆</td><td>POR</td><td>EFR</td><td>∆</td><td>POR</td><td>EFR</td><td>∆</td><td>POR</td><td>EFR</td><td>∆</td></tr><tr><td>OS</td><td>61.11</td><td>62.50</td><td>+1.39</td><td>56.94</td><td>55.56</td><td>-1.38</td><td>59.72</td><td>68.06</td><td>+8.34</td><td>59.72</td><td>70.83</td><td>+11.11</td></tr><tr><td>Office</td><td>30.16</td><td>33.62</td><td>+3.46</td><td>23.02</td><td>26.72</td><td>+3.70</td><td>50.97</td><td>61.20</td><td>+10.23</td><td>47.23</td><td>51.25</td><td>+4.02</td></tr><tr><td>Daily</td><td>50.24</td><td>58.02</td><td>+7.78</td><td>42.03</td><td>48.67</td><td>+6.64</td><td>57.88</td><td>64.50</td><td>+6.62</td><td>49.00</td><td>58.36</td><td>+9.36</td></tr><tr><td>Professional</td><td>64.63</td><td>80.95</td><td>+16.32</td><td>66.67</td><td>73.47</td><td>+6.80</td><td>80.95</td><td>78.91</td><td>-2.04</td><td>62.59</td><td>63.27</td><td>+0.68</td></tr><tr><td>Workflow</td><td>16.92</td><td>24.60</td><td>+7.68</td><td>15.48</td><td>21.51</td><td>+6.03</td><td>32.50</td><td>41.66</td><td>+9.16</td><td>25.32</td><td>25.78</td><td>+0.46</td></tr><tr><td>Overall</td><td>37.82</td><td>44.91</td><td>+7.09</td><td>33.36</td><td>38.38</td><td>+5.02</td><td>52.36</td><td>59.74</td><td>+7.38</td><td>44.88</td><td>49.16</td><td>+4.28</td></tr></table>

Table 2: Mean success rates (%) over three runs on the WindowsAgentArena benchmark. For each model backbone, ∆ denotes the mean EFR result minus the mean POR result in percentage points. The best mean accuracy in each row is bolded; ties are bolded together.
<table><tr><td rowspan="2">Task Type</td><td colspan="3">GUI-Owl-32B [36]</td><td colspan="3">Qwen3-VL-32B [3]</td><td colspan="3">Kimi-K2.5 [10]</td><td colspan="3">Seed-1.8 [5]</td></tr><tr><td>POR</td><td>EFR</td><td>∆</td><td>POR</td><td>EFR</td><td>∆</td><td>POR</td><td>EFR</td><td>∆</td><td>POR</td><td>EFR</td><td>∆</td></tr><tr><td>Office</td><td>5.43</td><td>8.66</td><td>+3.23</td><td>6.98</td><td>6.98</td><td>+0.00</td><td>16.28</td><td>18.60</td><td>+2.32</td><td>36.43</td><td>37.02</td><td>+0.59</td></tr><tr><td>Web Browsing</td><td>21.33</td><td>32.76</td><td>+11.43</td><td>30.00</td><td>43.22</td><td>+13.22</td><td>44.11</td><td>56.44</td><td>+12.33</td><td>47.44</td><td>49.78</td><td>+2.34</td></tr><tr><td>Windows System</td><td>48.61</td><td>44.44</td><td>-4.17</td><td>47.22</td><td>51.39</td><td>+4.17</td><td>73.61</td><td>79.17</td><td>+5.56</td><td>83.14</td><td>80.56</td><td>-2.58</td></tr><tr><td>Coding</td><td>58.33</td><td>75.00</td><td>+16.67</td><td>66.67</td><td>65.28</td><td>-1.39</td><td>73.47</td><td>81.81</td><td>+8.34</td><td>58.33</td><td>67.92</td><td>+9.59</td></tr><tr><td>Media &amp; Video</td><td>39.63</td><td>34.92</td><td>-4.71</td><td>34.92</td><td>48.52</td><td>+13.60</td><td>59.35</td><td>61.30</td><td>+1.95</td><td>24.79</td><td>26.40</td><td>+1.61</td></tr><tr><td>Windows Utilities</td><td>16.67</td><td>25.00</td><td>+8.33</td><td>16.67</td><td>30.56</td><td>+13.89</td><td>41.67</td><td>41.67</td><td>+0.00</td><td>42.53</td><td>58.33</td><td>+15.80</td></tr><tr><td>Overall</td><td>29.04</td><td>34.24</td><td>+5.20</td><td>31.60</td><td>37.55</td><td>+5.95</td><td>47.40</td><td>52.88</td><td>+5.48</td><td>48.16</td><td>51.32</td><td>+3.16</td></tr></table>

Configurations. All experiments are conducted within the Docker environments provided by the respective benchmarks. The agent operates on raw, screenshot-only observations without further preprocessing or structured UI metadata; all actions are executed via the pyautogui library, for details see Appendix F. We set the maximum number of steps to 50 for GUI-specialized and proprietary models, while open-source general-purpose models are restricted to a maximum of 15 steps. We repeat each main POR and POR-EFR experiment three times and report the mean success rate across 3 runs in Tables 1 and 2.

## 5.2 Main Results and Analysis

Comparison with Baseline. On OSWorld-Verified, Table 1 shows that replacing the standard reflector in POR with EFR consistently improves the mean overall success rate across all four backbones, with an average gain of 5.94%. The strongest EFR setting, POR-EFR with Kimi-K2.5, achieves the best mean overall score of 59.74%, improving over its matched POR counterpart by 7.38%. On WindowsAgentArena, Table 2 similarly shows consistent mean improvements across all four backbones, yielding an average gain of 4.95%. POR-EFR with Kimi-K2.5 achieves the best mean overall score of 52.88%, improving over its matched POR counterpart by 5.48%. Across the three runs, the standard deviations of the overall success rates range from 0.17% to 1.13%. Specifically, the POR/EFR standard deviations for GUI-Owl-32B, Qwen3-VL-32B, Kimi-K2.5, and Seed-1.8 are 0.26/0.17, 0.76/0.90, 0.25/0.17, and 0.42/0.56 on OSWorld-Verified, and 0.41/0.63, 0.68/1.13, 0.18/0.25, and 0.54/0.18 on WindowsAgentArena, respectively. These deviations are small relative to the observed overall gains, indicating that the improvements are stable across runs. The corresponding per-app results for the originally reported run are shown in Figure 6.

![](images/832c9deff7827564d5e4a7fc33fad7cb5d8727321181bb1e4421aa8b5f16c4fa.jpg)  
Figure 6: Per-app success-rate comparison between POR and POR-EFR on OSWorld-Verified and WindowsAgentArena. Each subplot corresponds to one backbone model.

Table 3: Ablation results of GUI-Owl-32B on OSWorld-Verified.
<table><tr><td>Setting</td><td>Overall (%)</td><td>Drop (%)</td></tr><tr><td>POR baseline (w/o both)</td><td>37.86</td><td>-6.97</td></tr><tr><td>Only Two-stage Judging (w/o SoM)</td><td>42.11</td><td>-2.72</td></tr><tr><td>Only SoM (w/o Two-stage Judging)</td><td>38.13</td><td>-6.70</td></tr><tr><td>Diff. Identification w/ Intended Action</td><td>43.45</td><td>-1.38</td></tr><tr><td>Outcome Verification w/o Screenshots</td><td>34.25</td><td>-10.58</td></tr><tr><td>Full method (SoM + Two-stage Judging)</td><td>44.83</td><td>一</td></tr></table>

Analysis over Different Task Categories. At the category level, the three-run means show that EFR provides the clearest gains on several task categories rather than uniformly across all of them. On OSWorld-Verified, Kimi-K2.5 improves by 10.23% and 9.16% on “Office” and “Workflow” tasks, reaching mean success rates of 61.20% and 41.66%, respectively. On WindowsAgentArena, Kimi-K2.5 improves by 12.33% on “Web Browsing”, reaching 56.44%, while Seed-1.8 improves by 15.80% on “Windows Utilities”, reaching 58.33%. These tasks often involve dense interfaces and frequent state transitions. EFR is particularly beneficial in such settings because it makes page changes more explicit and helps the agent deal with complex situations. We also observe a few category-level decreases compared with the conventional reflector. These cases are scattered across task categories and model backbones, suggesting that they do not indicate a systematic degradation on a specific task type. We provide representative failure examples in the Appendix H and summarize their causes in Appendix H.4.

Analysis over Different Model Backbones. EFR brings consistent mean overall gains across all backbones. On OSWorld-Verified, the improvements are 7.09%, 5.02%, 7.38%, and 4.28% for GUI-Owl-32B, Qwen3-VL-32B, Kimi-K2.5, and Seed-1.8, respectively. On WindowsAgentArena, the corresponding gains are 5.20%, 5.95%, 5.48%, and 3.16%. The gains are not determined solely by the capability of the backbone, nor do stronger models make EFR unnecessary. For example, Kimi-K2.5 obtains a larger improvement than GUI-Owl-32B on OSWorld-Verified and a comparable improvement on WindowsAgentArena, indicating that even strong models still benefit from explicit page-difference evidence and serial reflection.

## 5.3 Impact of Reflector Judgment on Task Success

To analyze the relationship between reflector accuracy and overall task success, we manually annotated the correctness of the reflectors judgments for each step within the GUI-Owl-32B trajectories on the OSWorld-Verified benchmark. The result is provided in Appendix Figure 8. Our EFR framework improves the reflectors judgment accuracy by 7.11%. A Pearson correlation coefficient of 0.86 indicates a strong positive correlation between reflector accuracy and end-to-end task success across various task categories. These findings suggest that a more reliable reflector enables the agent to evaluate execution outcomes more precisely, thereby enhancing overall performance.

![](images/442dbe21ff74b6d851bf0fcc2a1cc4a4d0f001fc7f5c0393dd5d59e4c0d7d9c3.jpg)

![](images/35cdb55f5875b68954197f0fd9e0836bb75b8314960af9c58bd4080b4e26f814.jpg)

![](images/2952d3523c45f1bb956652abad83ebcf5739b3073819024f05965c9008cebb00.jpg)

![](images/5e77e81e5bd18aafdf7552eef7066647f64a3d9bda4061b2e223778480243f2a.jpg)

![](images/271f0d40de27e0d8c2e2b0afa3d6e7a9a7542f8986731ce7c038077ac2e3c35a.jpg)

![](images/3b6bf7d2032ed4d12eabdb1bcba34286bfe416f2517eab2844e2d87ac6b7d0d3.jpg)  
Figure 7: Average per-task cost comparison between POR and POR-EFR on OSWorld-Verified and WindowsAgentArena. The first row reports OSWorld-Verified and the second row reports WindowsAgentArena. From left to right, each row shows latency, token usage, and executed steps.

## 5.4 Ablation Study

We conduct an ablation study on the OSWorld-Verified benchmark by using GUI-Owl-32B model.

Effect of SoM. Removing the SoM module requires the reflector to directly analyze the page differences and subsequently evaluate action success based on global difference descriptions. This configuration results in a 2.72% decrease in the success rate, demonstrating that SoM facilitates the difference analysis phase by providing localized focus.

Effect of Page Difference Identification. Ablating the page difference identification module, whereby the reflector evaluates success directly after receiving the marked screenshots results in a significant 6.70% decrease in the success rate. This drop underscores the critical role of the page difference identification process within our framework. The explicit separation of difference analysis and success judgment is essential for realizing the full benefits of the marked visual evidence.

Replace Grounded with Intended Action in Page Difference Identification Stage. We further replace the grounded action with the original intended action during the page difference identification stage. This variant leads to a 1.38% drop in accuracy. Since page difference identification aims to describe what actually changes after execution, conditioning this process on the intended action may cause the VLM to focus on expected outcomes rather than the observed screen transition.

Action Outcome Verification without Screenshots. We further evaluate a variant in which the action outcome verification stage relies only on the change descriptions, without access to screenshots. This setting results in a 10.58% drop in accuracy. This degradation suggests that textual change descriptions alone are insufficient for reliable outcome verification in complex GUI environments. When the interface is dense and contains multiple changes, the verifier may perceive only scattered changes, rather than forming a holistic understanding of the page structure.

## 5.5 Cost Analysis

Figure 7 compares the average per-task cost of POR and POR-EFR on two benchmarks. Averaged over the four backbone models, POR-EFR increases per-task latency by 15.8% and token usage by 14.9% on OSWorld-Verified, and by 13.3% and 17.9% on WindowsAgentArena, respectively. This overhead mainly stems from the additional VLM call introduced for page-difference analysis, which increases the cost of each reflection step. Although EFR reduces redundant execution by lowering the average number of executed steps by 11.5% on OSWorld-Verified and 3.7% on WindowsAgentArena, this reduction only partially offsets the increased per-step reflection cost. Detailed per-step latency and token costs are provided in Appendix G.

## 6 Conclusion

We revisit the role of the reflector in GUI agents and contend that its complexity and functionality are frequently underestimated in desktop environments. We thus introduce EFR, a two-stage reflection framework that decouples explicit visual evidence extraction from subsequent success judgment. Empirical evaluations demonstrate its superior performance. More broadly, our findings highlight that building GUI agents requires more than directly applying existing methods. While POR and SoM provide methodological foundations, their effectiveness depends on how well they are adapted to the perceptual, spatial, and interactive properties of GUI tasks.

## References

[1] Saaket Agashe, Jiuzhou Han, Shuyu Gan, Jiachen Yang, Ang Li, and Xin Eric Wang. Agent S: An open agentic framework that uses computers like a human. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum? id=lIVRgt4nLv.

[2] Saaket Agashe, Kyle Wong, Vincent Tu, Jiachen Yang, Ang Li, and Xin Eric Wang. Agent S2: A compositional generalist-specialist framework for computer use agents. In Conference on Language Modeling, 2025. URL https://openreview.net/forum?id=zg5is4GJ3R.

[3] Shuai Bai, Yuxuan Cai, Ruizhe Chen, et al. Qwen3-VL Technical Report. arXiv preprint arXiv:2511.21631, 2025. doi: 10.48550/arXiv.2511.21631. URL https://arxiv.org/abs/ 2511.21631.

[4] Rogerio Bonatti, Dan Zhao, Francesco Bonacci, Dillon Dupont, Sara Abdali, Yinheng Li, Yadong Lu, Justin Wagle, Kazuhito Koishida, Arthur Bucker, Lawrence Keunho Jang, and Zheng Hui. Windows agent arena: Evaluating multi-modal OS agents at scale. In Proceedings of the 42nd International Conference on Machine Learning, pages 4874–4910. PMLR, 2025. URL https://proceedings.mlr.press/v267/bonatti25a.html.

[5] ByteDance Seed. Seed1.8 model card: Towards generalized real-world agency, 2025. URL https://lf3-static.bytednsdoc.com/obj/eden-cn/lapzild-tss/ ljhwZthlaukjlkulzlp/research/Seed-1.8-Modelcard.pdf.

[6] Kanzhi Cheng, Qiushi Sun, Yougang Chu, Fangzhi Xu, Yantao Li, Jianbing Zhang, and Zhiyong Wu. SeeClick: Harnessing GUI grounding for advanced visual GUI agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9313–9332, Bangkok, Thailand, 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.505. URL https://aclanthology.org/2024. acl-long.505/.

[7] Yu Du, Fangyun Wei, and Hongyang Zhang. Anytool: Self-reflective, hierarchical agents for large-scale API calls. In Proceedings of the 41st International Conference on Machine Learning, pages 11812–11829. PMLR, 2024. URL https://proceedings.mlr.press/ v235/du24h.html.

[8] Difei Gao, Lei Ji, Zechen Bai, Mingyu Ouyang, Peiran Li, Dongxing Mao, Qinchen Wu, Weichen Zhang, Peiyi Wang, Xiangwu Guo, Hengxu Wang, Luowei Zhou, and Mike Zheng Shou. AssistGUI: Task-oriented desktop graphical user interface automation. arXiv preprint arXiv:2312.13108, 2023. doi: 10.48550/arXiv.2312.13108. URL https://arxiv.org/abs/ 2312.13108.

[9] Boyu Gou, Ruohan Wang, Boyuan Zheng, Yanan Xie, Cheng Chang, Yiheng Shu, Huan Sun, and Yu Su. Navigating the digital world as humans do: Universal visual grounding for GUI agents. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=kxnoqaisCT.

[10] Kimi. Kimi k2.5: Visual agentic intelligence, 2025. URL https://www.kimi.com/blog/ kimi-k2-5.

[11] Jing Yu Koh, Robert Lo, Lawrence Jang, Vikram Duvvur, Ming Lim, Po-Yu Huang, Graham Neubig, Shuyan Zhou, Russ Salakhutdinov, and Daniel Fried. VisualWebArena: Evaluating multimodal agents on realistic visual web tasks. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 881–905, Bangkok, Thailand, 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024. acl-long.50. URL https://aclanthology.org/2024.acl-long.50/.

[12] Hongxin Li, Jingfan Chen, Jingran Su, Yuntao Chen, Li Qing, and Zhaoxiang Zhang. AutoGUI: Scaling GUI grounding with automatic functionality annotations from LLMs. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 10323–10358, Vienna, Austria, 2025. Association for Computational Linguistics. doi: 10.18653/v1/2025.acl-long.510. URL https://aclanthology.org/2025. acl-long.510/.

[13] Ning Li, Xiangmou Qu, Jiamu Zhou, Jun Wang, Muning Wen, Kounianhua Du, Xingyu Lou, Qiuying Peng, Jun Wang, and Weinan Zhang. Mobileuse: A GUI agent with hierarchical reflection for autonomous mobile operation. arXiv preprint arXiv:2507.16853, 2025. doi: 10.48550/arXiv.2507.16853. URL https://arxiv.org/abs/2507.16853.

[14] Tao Li, Gang Li, Jingjie Zheng, Purple Wang, and Yang Li. MUG: Interactive multimodal grounding on user interfaces. In Findings of the Association for Computational Linguistics: EACL 2024, pages 231–251, St. Julian’s, Malta, 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-eacl.17. URL https://aclanthology.org/2024. findings-eacl.17/.

[15] Haowei Liu, Xi Zhang, Haiyang Xu, Yuyang Wanyan, Junyang Wang, Ming Yan, Ji Zhang, Chunfeng Yuan, Changsheng Xu, Weiming Hu, and Fei Huang. PC-Agent: A hierarchical multi-agent collaboration framework for complex task automation on PC. arXiv preprint arXiv:2502.14282, 2025. doi: 10.48550/arXiv.2502.14282. URL https://arxiv.org/abs/ 2502.14282.

[16] Yuhang Liu, Pengxiang Li, Zishu Wei, Congkai Xie, Xueyu Hu, Xinchen Xu, Shengyu Zhang, Xiaotian Han, Hongxia Yang, and Fei Wu. InfiGUIAgent: A multimodal generalist GUI agent with native reasoning and reflection. arXiv preprint arXiv:2501.04575, 2025. doi: 10.48550/ arXiv.2501.04575. URL https://arxiv.org/abs/2501.04575.

[17] Yadong Lu, Jianwei Yang, Yelong Shen, and Ahmed Awadallah. OmniParser for pure vision based GUI agent. arXiv preprint arXiv:2408.00203, 2024. doi: 10.48550/arXiv.2408.00203. URL https://arxiv.org/abs/2408.00203.

[18] Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. Self-Refine: Iterative refinement with self-feedback. In Advances in Neural Information Processing Systems, 2023. URL https://proceedings.neurips.cc/paper\_files/ paper/2023/hash/91edff07232fb1b55a505a9e9f6c0ff3-Abstract-Conference. html.

[19] Hongze Mi, Yibo Feng, Wenjie Lu, Yuqi Wang, Jinyuan Li, Song Cao, He Cui, Tengfei Tian, Xuelin Zhang, Haotian Luo, Di Sun, Jun Fang, Hua Chai, Naiqiang Tan, and Gang Pan. D-Artemis: A deliberative cognitive framework for mobile GUI multi-agents. arXiv preprint arXiv:2509.21799, 2025. doi: 10.48550/arXiv.2509.21799. URL https://arxiv.org/abs/ 2509.21799.

[20] Runliang Niu, Jindong Li, Shiqi Wang, Yali Fu, Xiyu Hu, Xueyuan Leng, He Kong, Yi Chang, and Qi Wang. ScreenAgent: A vision language model-driven computer control agent. In Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, IJCAI-24, pages 6433–6441. International Joint Conferences on Artificial Intelligence Organization, 2024. doi: 10.24963/ijcai.2024/711. URL https://doi.org/10.24963/ijcai.2024/711.

[21] Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, Wanjun Zhong, Kuanye Li, Jiale Yang, Yu Miao, Woyu Lin, Longxiang Liu, Xu Jiang, Qianli Ma, Jingyu Li, Xiaojun Xiao, Kai Cai, Chuang Li, Yaowei Zheng, Chaolin Jin, Chen Li, Xiao Zhou, Minchao Wang, Haoli Chen, Zhaojian Li, Haihua Yang, Haifeng Liu, Feng Lin, Tao Peng, Xin Liu, and Guang Shi. UI-TARS: Pioneering automated GUI interaction with native agents. arXiv preprint arXiv:2501.12326, 2025. doi: 10.48550/arXiv.2501.12326. URL https://arxiv.org/abs/2501.12326.

[22] Christopher Rawles, Sarah Clinckemaillie, Yifan Chang, Jonathan Waltz, Gabrielle Lau, Marybeth Fair, Alice Li, William E. Bishop, Wei Li, Folawiyo Campbell-Ajala, Daniel Kenji Toyama, Robert James Berry, Divya Tyamagundlu, Timothy P. Lillicrap, and Oriana Riva. AndroidWorld: A dynamic benchmarking environment for autonomous agents. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview. net/forum?id=il5yUQsrjC.

[23] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. In Advances in Neural Information Processing Systems, 2023. URL https://proceedings.neurips.cc/paper\_files/ paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference. html.

[24] Junyang Wang, Haiyang Xu, Haitao Jia, Xi Zhang, Ming Yan, Weizhou Shen, Ji Zhang, Fei Huang, and Jitao Sang. Mobile-Agent-v2: Mobile device operation assistant with effective navigation via multi-agent collaboration. In Advances in Neural Information Processing Systems, 2024. doi: 10.52202/079017-0088. URL https://proceedings.neurips.cc/paper\_files/paper/2024/hash/ 0520537ba799d375b8ff5523295c337a-Abstract-Conference.html.

[25] Junyang Wang, Haiyang Xu, Jiabo Ye, Ming Yan, Weizhou Shen, Ji Zhang, Fei Huang, and Jitao Sang. Mobile-Agent: Autonomous multi-modal mobile device agent with visual perception. In ICLR 2024 Workshop on Large Language Model (LLM) Agents, 2024. URL https://openreview.net/forum?id=jE6pDYCnVF.

[26] Shuai Wang, Weiwen Liu, Jingxuan Chen, Yuqi Zhou, Weinan Gan, Xingshan Zeng, Yuhan Che, Shuai Yu, Xinlong Hao, Kun Shao, Bin Wang, Chuhan Wu, Yasheng Wang, Ruiming Tang, and Jianye Hao. GUI agents with foundation models: A comprehensive survey. arXiv preprint arXiv:2411.04890, 2024. doi: 10.48550/arXiv.2411.04890. URL https://arxiv. org/abs/2411.04890.

[27] Xinyuan Wang, Bowen Wang, Dunjie Lu, Junlin Yang, Tianbao Xie, et al. OpenCUA: Open foundations for computer-use agents. arXiv preprint arXiv:2508.09123, 2025. doi: 10.48550/ arXiv.2508.09123. URL https://arxiv.org/abs/2508.09123.

[28] Zhenhailong Wang, Haiyang Xu, Junyang Wang, Xi Zhang, Ming Yan, Ji Zhang, Fei Huang, and Heng Ji. Mobile-Agent-E: Self-evolving mobile assistant for complex tasks. arXiv preprint arXiv:2501.11733, 2025. doi: 10.48550/arXiv.2501.11733. URL https://arxiv.org/abs/ 2501.11733.

[29] Qinzhuo Wu, Pengzhi Gao, Wei Liu, and Jian Luan. BacktrackAgent: Enhancing GUI agent with error detection and backtracking mechanism. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 4250–4272, Suzhou, China, 2025. Association for Computational Linguistics. doi: 10.18653/v1/2025.emnlp-main.212. URL https://aclanthology.org/2025.emnlp-main.212/.

[30] Tianbao Xie and Wei Li. Introducing OSWorld-Verified. xlang.ai, July 2025. URL https: //xlang.ai/blog/osworld-verified.

[31] Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, Yitao Liu, Yiheng Xu, Shuyan Zhou, Silvio Savarese, Caiming Xiong, Victor Zhong, and Tao Yu. OSWorld: Benchmarking multimodal agents for open-ended tasks in real computer environments. In Advances in Neural Information Processing Systems, 2024. doi: 10.52202/079017-1650. URL https://proceedings.neurips.cc/paper\_files/ paper/2024/hash/5d413e48f84dc61244b6be550f1cd8f5-Abstract-Datasets\_and\_ Benchmarks\_Track.html.

[32] Yiheng Xu, Zekun Wang, Junli Wang, Dunjie Lu, Tianbao Xie, Amrita Saha, Doyen Sahoo, Tao Yu, and Caiming Xiong. Aguvis: Unified pure vision agents for autonomous GUI interaction. In Proceedings of the 42nd International Conference on Machine Learning, pages 69772– 69805. PMLR, 2025. URL https://proceedings.mlr.press/v267/xu25ae.html.

[33] Jianwei Yang, Hao Zhang, Feng Li, Xueyan Zou, Chunyuan Li, and Jianfeng Gao. Setof-mark prompting unleashes extraordinary visual grounding in GPT-4V. arXiv preprint arXiv:2310.11441, 2023. doi: 10.48550/arXiv.2310.11441. URL https://arxiv.org/abs/ 2310.11441.

[34] Yuhao Yang, Yue Wang, Dongxu Li, Ziyang Luo, Bei Chen, Chao Huang, and Junnan Li. Aria-UI: Visual grounding for GUI instructions. In Findings of the Association for Computational Linguistics: ACL 2025, pages 22418–22433, Vienna, Austria, 2025. Association for Computational Linguistics. doi: 10.18653/v1/2025.findings-acl.1152. URL https://aclanthology.org/2025.findings-acl.1152/.

[35] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models. In International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=WE\_ vluYUL-X.

[36] Jiabo Ye, Xi Zhang, Haiyang Xu, Haowei Liu, Junyang Wang, Zhaoqing Zhu, Ziwei Zheng, Feiyu Gao, Junjie Cao, Zhengxi Lu, Jitong Liao, Qi Zheng, Fei Huang, Jingren Zhou, and Ming Yan. Mobile-Agent-v3: Fundamental agents for GUI automation. arXiv preprint arXiv:2508.15144, 2025. doi: 10.48550/arXiv.2508.15144. URL https://arxiv.org/abs/ 2508.15144.

[37] Chaoyun Zhang, Liqun Li, Shilin He, Xu Zhang, Bo Qiao, Si Qin, Minghua Ma, Yu Kang, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang, and Qi Zhang. UFO: A UI-focused agent for windows OS interaction. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 597–622, Albuquerque, New Mexico, 2025. Association for Computational Linguistics. doi: 10.18653/v1/2025.naacl-long.26. URL https://aclanthology.org/2025.naacl-long.26/.

[38] Chi Zhang, Zhao Yang, Jiaxuan Liu, Yanda Li, Yucheng Han, Xin Chen, Zebiao Huang, Bin Fu, and Gang Yu. AppAgent: Multimodal agents as smartphone users. In Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems, pages 1–20, New York, NY, USA, 2025. Association for Computing Machinery. doi: 10.1145/3706598.3713600. URL https://doi.org/10.1145/3706598.3713600.

A Reflector Accuracy 16   
B Step-Level Reflection Example 17   
B.1 Conventional Reflector 17   
B.2 EFR 18   
C Set-of-Marks Annotation Details 18   
C.1 Visual Change Area Annotation 18   
C.2 Action Location Annotation 19   
D Benchmark Task Grouping Details 19   
E Complete EFR Prompt 20   
F Action Space 22   
G Per-Step Cost Analysis 23   
H Case Study 23   
H.1 Failure to Success 24   
H.2 Failure to Failure 27   
H.3 Success to Failure 29   
H.4 EFR Failure Case Analysis 31

## A Reflector Accuracy

We provide two manually annotated analyses based on GUI-Owl-32B trajectories on the OSWorld-Verified benchmark. First, Figure 8 compares reflector judgment accuracy with end-to-end task success across five categories. Overall, replacing the conventional reflector with EFR increases average reflector accuracy from 80.39% to 87.50%, while average task success rises from 44.91% to 51.94%. The trend is particularly clear on Professional tasks, where reflector accuracy improves by 9.88% and task success increases by 16.32%. Across all method-category points in the figure, reflector accuracy and task success have a Pearson correlation of 0.86, suggesting that more reliable intermediate verification is strongly associated with higher final task completion.

Second, Table 4 reports how often failed trajectories can be attributed to reflector errors under the conventional POR framework and EFR. EFR reduces reflector-attributed failures from 61 of 223 failed trajectories (27.35%) to 27 of 199 (13.57%), a decrease of 13.78%. More specifically, pagedifference identification errors decrease from 43 of 223 failures (19.28%) to 14 of 199 (7.04%), a 12.25% reduction, while success-judgment errors decrease from 18 of 223 (8.07%) to 13 of 199 (6.53%), a 1.54% reduction. Thus, EFR reduces both error types, with the larger improvement arising from more reliable page-difference identification.

![](images/3048b1b56bc79656067b76634250fd08116cab4ff20cec53c2f83ead1260b17f.jpg)  
Figure 8: Task success rate vs. reflector success rate in GUI-Owl-32B trajectories on the OSWorld-Verified benchmark.

Table 4: Manual failure attribution for GUI-Owl-32B under the conventional POR framework and EFR on OSWorld-Verified, using the same model and mutually exclusive error categories. “Diff. Error” denotes page-difference identification errors, “Judge. Error” denotes success-judgment errors given correctly identified page differences, and “Total” denotes all failed trajectories. For the reflector-error rate, ∆ denotes EFR minus POR in percentage points. The lower rate in each row is bolded; ties are bolded together.
<table><tr><td rowspan="2">Application</td><td colspan="2">Reflector</td><td colspan="2">Diff. Error</td><td colspan="2">Judge. Error</td><td colspan="2">Total</td><td colspan="3">Rate (%)</td></tr><tr><td>POR</td><td>EFR</td><td>POR</td><td>EFR</td><td>POR</td><td>EFR</td><td>POR</td><td>EFR</td><td>POR</td><td>EFR</td><td>Δ</td></tr><tr><td>chrome</td><td>3</td><td>0</td><td>2</td><td>0</td><td>1</td><td>0</td><td>26</td><td>19</td><td>11.54</td><td>0.00</td><td>-11.54</td></tr><tr><td>gimp</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>11</td><td>5</td><td>0.00</td><td>0.00</td><td>+0.00</td></tr><tr><td>libreoffice_calc</td><td>23</td><td>11</td><td>16</td><td>7</td><td>7</td><td>4</td><td>37</td><td>41</td><td>62.16</td><td>26.83</td><td>-35.33</td></tr><tr><td>libreoffice_impress</td><td>12</td><td>2</td><td>8</td><td>0</td><td>4</td><td>2</td><td>31</td><td>30</td><td>38.71</td><td>6.67</td><td>-32.04</td></tr><tr><td>libreoffice_writer</td><td>4</td><td>1</td><td>4</td><td>1</td><td>0</td><td>0</td><td>14</td><td>7</td><td>28.57</td><td>14.29</td><td>-14.28</td></tr><tr><td>multi_apps</td><td>17</td><td>11</td><td>12</td><td>6</td><td>5</td><td>5</td><td>77</td><td>70</td><td>22.08</td><td>15.71</td><td>-6.37</td></tr><tr><td>OS</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>9</td><td>9</td><td>0.00</td><td>0.00</td><td>+0.00</td></tr><tr><td>thunderbird</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>4</td><td>6</td><td>0.00</td><td>0.00</td><td>+0.00</td></tr><tr><td>vlc</td><td>1</td><td>1</td><td>1</td><td>0</td><td>0</td><td>1</td><td>9</td><td>8</td><td>11.11</td><td>12.50</td><td>+1.39</td></tr><tr><td>vs_code</td><td>1</td><td>1</td><td>0</td><td>0</td><td>1</td><td>1</td><td>5</td><td>4</td><td>20.00</td><td>25.00</td><td>+5.00</td></tr><tr><td>All</td><td>61</td><td>27</td><td>43</td><td>14</td><td>18</td><td>13</td><td>223</td><td>199</td><td>27.35</td><td>13.57</td><td>-13.78</td></tr></table>

![](images/2c02f82653a8ca67c04e7e05058cfd51ec9e9352b2da300bd97b90b12f34c196.jpg)  
(a) Raw pre-action screenshot

![](images/01bc603199505bafdbb2d2f0a492add7dc8f54ae67509cf422ebc3b5e0d94db7.jpg)  
(b) Raw post-action screenshot

![](images/7b1ab13158889d43b1f03abc2bef999b574b0e27e22f82093bacd8779853cfb3.jpg)  
(c) SoM-marked pre-action screenshot

![](images/3e332b370cc2d32f2e5745f45ffe2cfa644fd8c380e94eed6716ff261126e027.jpg)  
(d) SoM-marked post-action screenshot

![](images/1344fa982a026b86a1ff84401dce54a58cb4ae3ad284b99070bbbfe99d4464e5.jpg)  
(e) Clean evidence before action

![](images/69302706d2c7d87470bf52520c357acba732c0caed1afd7e89177d9a9d473bba.jpg)  
(f) Clean evidence after action  
Figure 9: A one-step reflection example for clicking the “By date” button in Chrome History. The clean evidence keeps Areas 1–4 and removes Area 5, which only reflects the taskbar clock update.

## B Step-Level Reflection Example

We provide a concrete one-step example to illustrate the difference between the conventional reflector and EFR. The intended action is Click the “By date” button in Chrome History to sort the history view by date, while the actually executed grounded action is Click at coordinate (519,232). This is a simple example in which both reflectors make the correct judgment. The raw screenshots and EFR intermediate screenshots are shown in Figure 9. The conventional reflector directly reasons over the raw pre-action and post-action screenshots, whereas EFR generates explicit intermediate evidence before making the final reflection judgment.

## B.1 Conventional Reflector

## • Step 1: One-step action verification.

VLM input: the intended action: Click the “By date” button in Chrome History to sort the history view by date. The raw pre-action and post-action screenshots: Figure 9a and Figure 9b.

VLM output: The observed screen changes from the grouped history view to the date-based history view, indicate that the action succeeded.

## B.2 EFR

## • Step 1: SoM annotation.

SoM module input: The raw pre-action and post-action screenshots: Figure 9a and Figure 9b.   
The grounded action: Click at coordinate (519,232).

SoM module output: The marked pre-action screenshot: Figure ${ 9 \mathrm { c } } ,$ page change regions and action position are marked. The marked post-action screenshot: Figure 9d, only page change regions are marked.

## • Step 2: Page-difference identification.

VLM input: the SoM-marked pair in Figures 9c and 9d, plus the grounded action Click at coordinate (519,232).

VLM output: a structured evidence description: Area 1 shows that the mouse cursor moves away; Area 2 shows that the URL changes from chrome://history/grouped to chrome://history; Area 3 shows that the selected history mode changes from “By group” to “By date”; Area 4 shows that the center message changes from “No results” to “Your browsing history appears here”; Area 5 is identified as environment noise because it only reflects the taskbar clock update. The clean evidence output is Figure 9e and Figure 9f, which retain Areas 1–4 and remove Area 5.

## • Step 3: Evidence-based action verification.

VLM input: the intended action Click the “By date” button in Chrome History to sort the history view by date, the structured evidence from Step 2, and the clean evidence screenshots in Figures 9e and 9f.

Actual output: The action succeeds because Areas 2–4 show the page changing from the grouped history view to the date-based history view, which matches the intended action.

## C Set-of-Marks Annotation Details

## C.1 Visual Change Area Annotation

Algorithm 1 summarizes the visual difference annotation procedure used in EFR. The algorithm takes a pair of pre- and post-action screenshots and outputs two consistently annotated images together with the merged difference boxes. At this stage, the goal is only to identify where visual changes occur; the action-location annotation step is described in the next subsection. All visualchange markers use red bounding boxes, with each box assigned a unique index so that the reflector can refer to the same changed region across the pre- and post-action screenshots.

## Algorithm 1: Visual Difference Region Annotation

Input: Pre-action screenshot $S _ { p r e }$ , post-action screenshot $S _ { p o s t }$   
Output: Annotated screenshots $\hat { S } _ { p r e } , \hat { S } _ { p o s t }$ and merged box set $\boldsymbol { B }$   
1: function SHOULDMERGE(b<sub>i</sub>, $b _ { j } , M )$   
2: Compute the union box u of $b _ { i }$ and $b _ { j }$   
3: Compute the x/y gaps between $b _ { i }$ and $b _ { j }$   
4: Compute the difference density of u on binary mask M   
5: if $\cdot _ { b _ { i } }$ and $b _ { j }$ are tightly adjacent then   
6: return density(u) τ<sub>dens</sub>   
7: else   
8: return False   
9: end if   
10: end function   
11: Convert $S _ { p r e }$ and $S _ { p o s t }$ to grayscale images   
12: D pixel-wise absolute difference between the two grayscale images   
13: M binary mask obtained by thresholding D   
14: Extract external contours from M and convert each contour to a bounding box   
15: Remove boxes with a size of at most 2 pixels

16: Remove boxes fully contained in the top status-bar area   
17: Expand each remaining box by 5 pixels on all sides   
18: the resulting box set   
19: repeat   
20: changed False   
21: for all pairs $( b _ { i } , b _ { j } )$ in do   
22: if SHOULDMERGE $( b _ { i } , b _ { j } , M )$ then   
23: Replace $( b _ { i } , b _ { j } )$ with their union in   
24: changed True   
25: break   
26: end if   
27: end for   
28: until changed = False   
29: Sort from top to bottom and from left to right   
30: Draw the same red bounding boxes and indices in on $S _ { p r e }$ and $S _ { p o s t }$   
31: return annotated images $\hat { S } _ { p r e } , \hat { S } _ { p o s t }$ and

The output of Algorithm 1 constitutes the candidate visual evidence used by the reflector. However, these boxes alone do not indicate where the current action was actually executed. To connect the detected visual changes with the concrete operation site, we further annotate the action location on the pre-action screenshot in the next subsection.

## C.2 Action Location Annotation

Compared with the difference-region identification in Algorithm 1, action-location annotation is a lighter-weight post-processing step. Based on the geometric parameters associated with the actual action $A _ { g r o u n d }$ executed by the operator, it directly visualizes where the action takes place on the annotated pre-action screenshot $\hat { S } _ { p r e } ,$ thereby providing the reflector with a spatial anchor aligned with the observed visual changes. We denote this process as

$$
( \tilde { S } _ { p r e } , L ) = f _ { \mathrm { a c t } } ( \hat { S } _ { p r e } , A _ { g r o u n d } ) .\tag{8}
$$

where $\tilde { S } _ { p r \epsilon }$ denotes the pre-action screenshot annotated with the action marker, and L denotes the geometric location information of the action.

In our implementation, actions are divided into three categories in this stage, and the detailed action types are provided in Appendix F. For actions that can be reduced to a single screen point, we directly read the execution coordinates $( x , y )$ and map them onto the pixel grid. We then draw a blue hollow circle centered at that location on $S _ { p r e }$ to highlight the actual operation site. For the drag action, we further parse the drag start point $( x _ { s } , y _ { s } )$ and end point $( x _ { e } , y _ { e } )$ from the executable code. The drag trajectory is visualized with yellow markers: the start point is shown as a yellow hollow circle, the end point is shown as a yellow filled circle, and the two points are connected by a yellow arrow indicating the movement direction. For actions that do not require explicit screen coordinates, such as keyboard shortcuts or waiting actions, no action-location annotation is added. Accordingly, the geometric representations of point-based actions, drag actions, and coordinate-free actions are written as

$$
L _ { \mathrm { p o i n t } } = \bigl ( ( x , y ) \bigr ) , \qquad L _ { \mathrm { d r a g } } = \bigl ( ( x _ { s } , y _ { s } ) , ( x _ { e } , y _ { e } ) \bigr ) , \qquad L _ { \mathrm { n o n e } } = \emptyset .\tag{9}
$$

This action annotation step does not involve any additional visual inference; it merely projects the parameters of the action already executed by the operator back into the screenshot space. Its purpose is to explicitly indicate where the actual operation took place, thereby assisting the reflector in its observation. Finally, the action-location marker and the candidate difference boxes together form $S _ { p r e } ^ { s o m }$ in the main text, while the post-action screenshot retains only the difference boxes to form $S _ { p o s t } ^ { \sqrt { s } \omega \overline { { m } } }$

## D Benchmark Task Grouping Details

We summarize the correspondence between the app-level categories and the domain-level categories used in Tables 1 and 2. The left table reports OSWorld-Verified and the right table reports WindowsAgentArena. The category mappings follow the official benchmark definitions [4, 31].

Table 5: Correspondence between the original app-level partition and the merged task categories used in the main results.  
(a) OSWorld-Verified category mapping.
<table><tr><td>Domain</td><td>Included Apps</td><td># Tasks</td><td>%</td></tr><tr><td>OS</td><td>OS tasks</td><td>24</td><td>6.6%</td></tr><tr><td>Office</td><td>LibreOffice Calc, Impress, Writer</td><td>117</td><td>32.4%</td></tr><tr><td>Daily</td><td>Chrome, Thunderbird, VLC</td><td>78</td><td>21.6%</td></tr><tr><td>Professional</td><td>VS Code, GIMP</td><td>49</td><td>13.6%</td></tr><tr><td>Workflow</td><td>Tasks involving multiple apps</td><td>93</td><td>25.8%</td></tr><tr><td>Total</td><td></td><td>361</td><td>100.0%</td></tr></table>

(b) WindowsAgentArena category mapping.
<table><tr><td>Domain</td><td>Included Apps</td><td># Tasks</td><td>%</td></tr><tr><td>Office</td><td>LibreOffice Calc, LibreOffice Writer</td><td>43</td><td>27.9%</td></tr><tr><td>Web Browsing</td><td>Chrome, Microsoft Edge</td><td>30</td><td>19.5%</td></tr><tr><td>Windows System</td><td>File Explorer, Settings</td><td>24</td><td>15.6%</td></tr><tr><td>Coding</td><td>VS Code</td><td>24</td><td>15.6%</td></tr><tr><td>Media &amp; Video</td><td>VLC</td><td>21</td><td>13.6%</td></tr><tr><td>Windows</td><td>Clock, Paint, Notepad, Windows Calculator</td><td>12</td><td>7.8%</td></tr><tr><td>Utilities Total</td><td></td><td>154</td><td>100.0%</td></tr></table>

## E Complete EFR Prompt

We report the prompt of the two-stage EFR reflector. To make the runtime assembly explicit, we replace concrete values with bracketed placeholders.

## Stage 1: Page Difference Identification:

<table><tr><td>Prompt in stage 1 (System prompt):</td></tr><tr><td>You are a GUI Visual Change Analyzer. Your task is to identify and describe page changes that are directly related to a specific action, based on the provided action description and screenshots. INPUT DATA: 1. Action Description: Text explaining the user&#x27;s intended operation. 2. Before-Action Screenshot (first attached image): Contains Action Markers (e.g., blue circles, yellow arrows) indicating the operation location. It also contains the same Red Bounding Boxes with unique IDs as the After Image, highlighting the regions where visual differences were detected between the two screenshots. 3. After-Action Screenshot (second attached image): Contains Red Bounding Boxes with unique IDs highlighting all detected visual differences. The area outside the Red Bounding Boxes has not undergone any changes.</td></tr><tr><td>4. Bounding Boxes List: Contains the list of the coordinates of the Red Bounding Boxes, mapped to Red Box IDs in both images. OUTPUT STRUCTURE: Step 1: Filtering Analysis [List every Box ID and judge it. Format: ID - Reason - Verdict] * Area 1: [It is just the system clock changing time. Not relevant to the action.] - [Noise] * Area 2: [It is a dropdown menu appearing near the click location, which is caused by the click action.] - [Relevant] * Area 3: [It is the system clock changing time and a menu appearing near the click location,</td></tr><tr><td>caused by the click action.] - [Relevant] Step 2: Final Description [Based ONLY on the “Relevant&quot; areas from Step 1, generate the final description.] 1. Detailed Changes: [Describe specific changes inside relevant red boxes by logical groups.] - Areas [List of Relevant IDs]: [E.g., “- Areas [2, 5]: A dropdown menu appeared showing options &#x27;Save&#x27; and &#x27;Exit&#x27;.&quot;] 2. Description: [Summarize the visual changes caused by the action in a concise manner,</td></tr></table>

<table><tr><td>You are a GUI Task Verifier. Your role is to determine if the previous action was successful. INPUT DATA: 1. Before-Action Screenshot (first attached image): This image shows the GUI state BE-</td></tr><tr><td>FORE the action was executed. Red Bounding Boxes with IDs have been artificially overlaid on this image to mark WHERE changes will happen.</td></tr><tr><td>2. After-Action Screenshot (second attached image): This image shows the GUI state AF- TER the action was executed. The same Red Bounding Boxes with IDs are overlaid to highlight the regions where changes occurred.</td></tr><tr><td>3. Visual Change Description (provided in text): This is the ABSOLUTE GROUND TRUTH text report describing what actually changed AFTER the action. You MUST pri- oritize and trust the “Visual Change Description”.</td></tr></table>

Stage 2: Action Outcome Verification:

## Prompt in stage 2 (System prompt):

## Prompt in stage 2 (User prompt):

## Latest Action

## Visual Changes After Action

[]

— Carefully examine the information provided above to determine whether the last action produced the expected behavior. If the action was successful, update the progress status accordingly. If the action failed, identify the failure mode and provide reasoning on the potential reason causing this failure.

Pro Tip: In rare cases, the UI might not visibly change even if a click action is performed correctly for example, when clicking on a color before drawing. In such situations, you can assume the action was successful and proceed for example, by drawing a line. When the user instruction involves adjusting some values (e.g., brightness, contrast, steps), be sure to check if the values meet expectations after the operation. If the adjusted value is very close to the target, treat the action as successful and note the remaining gap in Progress Status.

Provide your output in the following format containing four parts:

Analyse (Describe the criteria for judgment)

Final Visual State:[Describe the state AFTER the action by comparing the before screenshot (first image) and after screenshot (second image), and cross-referencing with the “Visual Changes After Action” text. Remember to absolutely trust the text if there’s a conflict. Then describe if the final visual state meets the expected status or makes reasonable progress towards it]

## Outcome

Choose from the following options. Give your response as “A”, “B” or “C”:

A: Successful or Partially Successful. The result of the last action meets the expectation, OR the action was successfully executed and made incremental progress towards the goal (even if the final objective is not completely fulfilled yet).

B: Failed. The reason for the failure is a grounding error.

C: Failed. The reason for the failure is an error in the action itself.

Error Description

If the action failed (Outcome B or C), provide a detailed description of the error and the potential reason causing this failure. If the action succeeded or partially succeeded (Outcome A), put “None” here.   
Progress Status   
Update the progress status.

## F Action Space

To remain consistent with the underlying execution interface and to avoid introducing additional confounding factors into the reflector analysis due to differences in low-level action API design, we adopt the action space definition of Mobile-Agent-v3 [36]. Specifically, the operator first outputs a structured action. If the action contains an element description, a grounding model is first invoked to resolve the corresponding element into screen coordinates, after which the grounded result is compiled into executable desktop-operation code. Table 6 summarizes the action types used in this work, their semantic meanings, and the corresponding executable code templates. For readability, we omit the shared import pyautogui prefix in the table and retain only the core execution statements.

Table 6: Action space directly adopted from Mobile-Agent-v3 [36]. The code column shows the executable template after grounding. Shared import prefixes are omitted for brevity.
<table><tr><td rowspan="2">Action Type</td><td colspan="2">Action Details</td></tr><tr><td>Description</td><td>Executable Code Template</td></tr><tr><td>click</td><td>Click at a grounded screen location.</td><td>pyautogui.click(x=x, y=y)</td></tr><tr><td>double_click</td><td>Double-click at a grounded screen loca-</td><td>pyautogui.doubleClick(x=x, y=y)</td></tr><tr><td>right_click</td><td>tion. Right-click at a grounded screen location.</td><td>pyautogui.rightClick(x=x, y=y)</td></tr><tr><td>type</td><td>Focus an input region and type text, optionally clear- ing the field or pressing</td><td>pyautogui.doubleClick(x=x, y=y); hotkey(&quot;ctrl&quot;, &quot;a&quot;); press(&quot;delete&quot;); pyautogui.typewrite(text, interval=1.0);</td></tr><tr><td>hotkey</td><td>Enter. Execute a keyboard short- cut combination.</td><td>pyautogui.press(&quot;enter&quot;) pyautogui.hotkey(key1, ..., keyn)</td></tr><tr><td>scroll</td><td>Move to a grounded loca- pyautogui.moveTo(x, y); tion and scroll by a signed</td><td>pyautogui.scroll(value)</td></tr><tr><td>wait</td><td>amount. Pause execution for a spec- time. sleep(t)</td><td></td></tr><tr><td>done</td><td>ified duration. Mark the current task as DONE</td><td></td></tr><tr><td>drag</td><td>completed. Drag from one grounded locationto another</td><td>pyautogui.moveTo(x1, y1); pyautogui.dragTo(x2, y2, duration=1.0);</td></tr></table>

In terms of grounding, click, double\_click, right\_click, type, and scroll each require resolving only a single screen point; drag requires separately resolving both the start point and the end point; whereas hotkey, wait, and done do not depend on visual grounding. This also explains why the action-location annotation in the previous subsection only needs to distinguish between three cases.

## G Per-Step Cost Analysis

Figure 10 reports the average latency and token usage per executed step. POR-EFR increases latency and token usage per step by 30.8% and 29.8% on OSWorld-Verified, and by 17.7% and 22.4% on WindowsAgentArena, respectively. These results show that EFR introduces additional cost at each reflection step, while the step reduction shown in Figure 7 helps keep the final per-task overhead moderate.

Latency per Step (OSWorld-Verified)  
![](images/1193a3f8d1fb62cc9c843f51290ede95597628fcc7f497659391a7d955d00550.jpg)  
Latency per Step (WindowsAgentArena)

Token per Step (OSWorld-Verified)  
![](images/ce69b582531840f5ead2f3a5f574058d47d2f9f4b5224e28cd3fa006c5f9f8dc.jpg)

![](images/5172606c4869c4c619406eb78be2ffd20cb28f080cb65fef7054f2c45eba83fd.jpg)

Token per Step (WindowsAgentArena)  
![](images/d125ebc16db3b5f962778aae564afac2dbe10a01b478075a04b60f717ae0cc60.jpg)  
Figure 10: Average per-step cost comparison between POR and POR-EFR on OSWorld-Verified and WindowsAgentArena. The first row reports OSWorld-Verified and the second row reports WindowsAgentArena. From left to right, each row shows latency and token usage averaged over executed steps.

## H Case Study

Before examining individual trajectories, we first summarize how EFR changes task-level outcomes relative to the conventional reflector, as illustrated in Figure 11. On OSWorld-Verified, 84.05% of samples remain failure failure, 12.68% are converted from failure success, and 3.27% shift from success failure. On WindowsAgentArena, the corresponding proportions are 85.51%, 10.19%, and 4.30%, respectively. These transition statistics show that EFR improves task success mainly by recovering previously failed samples while introducing only a small fraction of negative flips. At the same time, the large failure failure proportion indicates that many GUI-agent tasks remain unsolved even after applying our method, highlighting that robust GUI agents still require substantial further progress. We illustrate each of these three types with representative examples.

![](images/1b3c865e5e1d3093e70d3f812e427ce98942bdc58b7de7da05e21772e8dcc36b.jpg)  
Figure 11: Outcome transition distributions on OSWorld-Verified and WindowsAgentArena. Each pie chart shows the proportion of trajectories that remain failed, are corrected from failure to success, or regress from success to failure after applying our method.

## H.1 Failure to Success

In this section, we compare two paired trajectories from OSWorld-Verified. Each pair contains a successful trajectory with EFR and a failed trajectory with the original one-step reflector. Case 1 illustrates the “small-change” regime, where the decisive evidence is only a tiny cursor or formatting change. Case 2 illustrates the “distributed-change” regime, where the target cell looks correct but the action also damages unrelated regions. In both cases, the operator can produce similar low-level actions; the key difference is whether the reflector can first organize the relevant visual evidence before deciding success.

## H.1.1 Case 1: Change the 2 in “H2O” to a subscript in LibreOffice Writer

This case corresponds to libreoffice\_writer/0b17a146-2934-46c7-8727-73ff6b6483e8. The task is to convert the “2” in “H2O” into a subscript. The success and failure trajectories differ mainly in how the reflector treats very subtle page changes. This example shows how EFR improves the reflector in the small-change regime: by explicitly grounding judgment in observed evidence, it converts a nearly invisible cursor change into a usable signal, preventing repeated false failure judgments.

![](images/3a26c43eff86ba82d26a99ccea41318ecb04673441cb59cc5886e0b3342449a9.jpg)  
Top: Failed trajectory with the conventional reflector.

![](images/2f47315919ffe4a6fb8c8e131a67527081ff1abc0a23a873db17c310a42eeb27.jpg)  
Bottom: Successful trajectory with EFR.  
Figure 12: Case 1: EFR succeeds by recognizing subtle cursor and formatting evidence, while the conventional reflector repeatedly misses the small change and keeps retrying the same click.

## Failed trajectory with the conventional reflector.

1. Action: Click the character “2”. Reflector: No visible page change is reported, so the click is judged as failed. However, a small cursor occurs next to character “2”.

2. Action: Click the character “2” again. Reflector: The reflector still reports no visible page change and again judges the action as failed. However, a small cursor occurs next to character “2”.

3. Action: Click the character “2” for a third time. Reflector: The reflector still reports no visible page change and again judges the action as failed. However, a small cursor occurs next to character “2”.

4. Action: Click the character “2” for a fourth time. Reflector: The reflector again reports no visible change, the trajectory remains stuck in repeated retries, and the task ultimately fails.

## Successful trajectory with EFR.

1. Action: Click the character “2”. Reflector: A text cursor appears next to “2”, indicating that the click has landed on the correct position.

2. Action: Drag over the character “2”. Reflector: The background color of “2” changes, showing that the character has been successfully selected.

## 3. Action: Press Ctrl+Shift+B.

Reflector: The “2” is rendered as a subscript, so the reflector marks the task as successful.

## H.1.2 Case 2: Add Wednesday 12:00 course slot in Calc schedule

This case corresponds to multi\_apps/3a93cae4-ad3e-403e-8c12-65303b271818. The task is to enter the course content into the Wednesday 12:00 slot in a Calc spreadsheet. Here the critical difficulty is not a missing change, but a misleading partial success: the target cell is updated, yet unrelated cells are also modified. This example shows how EFR improves the reflector in the largechange regime: instead of checking only whether the intended region contains the expected changes, it first organizes the full set of differences and can therefore reject actions with harmful side effects.

2: Click  
![](images/0102eac58d8b6d545961696b8b5baa1d97568ede794a1bac5446a1d2a3efcf22.jpg)

5: Fail  
![](images/b9182572350ed60ca3bdd0dd1b112e0842fab1d889b88411e6f8c9c6facc93f1.jpg)

![](images/ef4c94c3af18b2dce96688d45cb5e2403ba109e71e36d539aef87f8775bc5e8c.jpg)

Top: Failed trajectory with the conventional reflector.  
![](images/8824d2e061acc9afe8293f6eab39e36da0fb8316f2c2030768676210118dcef2.jpg)

![](images/4a645097714a48f6e54eb56dc3e698d9bb2e8c61bbd12e633b2a284d848b35e3.jpg)  
Bottom: Successful trajectory with EFR.

Figure 13: Case 2: EFR rejects a target-cell update that also damages other cells, triggers recovery, and succeeds on retry; the conventional reflector accepts the same corrupted intermediate state as success.

## Failed trajectory with the conventional reflector.

1. Action: Click the target xlsx file.

Reflector: The page switches to the spreadsheet, confirming that the file is open.

2. Action: Click the Wednesday 12:00 cell.

Reflector: The changed border indicates that the target cell has been selected.

3. Action: Select the cell content, delete the original text, and type the new course content. Reflector: The reflector focuses only on the target cell, observes that the expected content appears there, and therefore judges the action as successful, while overlooking unintended changes in other cells. However, the contents of other cells are deleted as well.

4. Action: Press Enter to confirm the edit. 4. Action: Press Enter to confirm the edit

Reflector: The target cell exits edit mode, so the trajectory appears successful locally, even though the spreadsheet has already been corrupted globally.

## Successful trajectory with EFR.

1. Action: Click the target xlsx file.

Reflector: The page switches to the spreadsheet, confirming that the file has been opened successfully.

2. Action: Click the Wednesday 12:00 cell. Reflector: The cell border changes, indicating that the target cell has been selected.

3. Action: Select the cell content, delete the original text, and type the new course content. Reflector: Although the target cell is filled with the desired content, the reflector also observes large-scale page changes and missing text in unrelated regions, so this step is judged as failed because the action has harmful side effects.

Reflector: The spreadsheet returns to the earlier state, confirming that the incorrect edit has been rolled back successfully.

5. Action: Re-enter the course content in the target cell. Reflector: This time only the intended cell changes, so the reflector judges the action as successful and the task is completed.

## H.2 Failure to Failure

In this section, we examine a trajectory pair where both the conventional reflector and EFR fail from OSWorld-Verified. This type of case is important because it clarifies the scope of reflector improvement: EFR can make reflection more evidence-aware, but the reflector is still only one module in a larger GUI-agent system. The essential reason for failure is the errors in the planner and operator. Even if the reflector produces a correct judgment, the subsequent execution trajectory does not necessarily change as a result.

## H.2.1 Case 3: Move the table on Page 3 to the bottom of the slide

This case corresponds to libreoffice\_impress/ac1b39ff-ee4d-4483-abce-c117e98942f0. The task is to move the table on Page 3 to the bottom of the slide. Both trajectories eventually fail, but they reveal different limitations. The conventional reflector mostly accepts each apparent movement as successful, while EFR detects one failed drag action. However, after detecting the failure, the overall agent does not roll back effectively, and later actions still suffer from grounding errors.

![](images/c5558963ea02ea5c5616d8279a227aa2c55ec4fd75586c88a67b0385f68f4f50.jpg)  
Top: Failed trajectory with the conventional reflector.

![](images/754cda6a604c0e94fba44d262172cb0aea04543085441c0dfdfe74abb5bb65d7.jpg)  
Bottom: Failed trajectory with EFR.  
Figure 14: Case 3: Both reflectors fail to complete the slide-editing task. EFR identifies an unsuccessful drag action, but the agent still does not roll back or re-plan correctly, and grounding errors continue to affect later steps.

## Failed trajectory with the conventional reflector.

1. Action: Click Page 3 in the left slide thumbnail panel. Reflector: The main canvas switches to Page 3, indicating that the target slide has been opened successfully.

2. Action: Click the table to select it. Reflector: Green control points appear around the table, showing that the object has been selected.

Reflector: The object appears to move down, so the reflector judges the drag as successful. However, the table is still above the text, and the actual manipulation is likely applied to the wrong object or an incorrect selection region.

4. Action: Click the table again to adjust its position. Reflector: Green control points appear again, showing that the object has been selected.

Reflector: The object moves further toward the bottom area, action successful.

## Failed trajectory with EFR.

1. Action: Click Page 3 in the left slide thumbnail panel. Reflector: The main canvas switches to Page 3, indicating that the target slide has been opened successfully.

2. Action: Click the table to select it. Reflector: Green control points appear around the table, showing that the object has been selected.

3. Action: Drag the selected object downward.

Reflector: The page does not visibly change, action failed.

4. Action: Try again to drag the selected object downward. Reflector: The text appears to move down instead of the table, so the reflector judges the drag as failed. However, the agent does not roll back. Instead, it continues with the next steps, which still suffer from grounding errors.

5. Action: Click the table again to adjust the position. Reflector: The table is selected and green control points appear, action successful.

6. Action: Continue dragging the table below the text. Reflector: The selected object moves further toward the bottom region, action successful.

7. Action: Click an empty area of the slide. Reflector: The table selection is canceled successfully.

Reflector: The file is saved successfully, but the saved state still contains the unresolved layout error.

## H.3 Success to Failure

In this section, we examine a negative transition trajectory introduced by EFR from OSWorld-Verified. This example shows that the current reflector can still make an incorrect final judgment even when it observes the key visual evidence, revealing that success judgment remains difficult even when visual evidence is sufficient.

## H.3.1 Case 4: Place the photo on the desktop and rename it to export.jpg

This case corresponds to gimp/77b8ab4d-994f-43ac-8930-8ca087d7c4b4. The task is to export the opened image to the desktop and rename it as export.jpg. The conventional reflector completes the export correctly, while EFR fails because it accepts export.jpg.jpg as a successful filename. Notably, the original trajectory succeeds because its operator takes the correct action, rather than because the conventional reflector is intrinsically better than EFR. This example highlights a remaining limitation of the reflector: EFR can improve evidence extraction, but the final success judgment still depends on correctly interpreting the semantics of the observed evidence.

![](images/e1d5e207105bb564b858eb7550229a600b965dd4bb90ad38dcfce12ef692afca.jpg)  
Top: Successful trajectory with the conventional reflector.

![](images/c1a44f7e152d6bf6bc301fae4fb808a1c3500195d29518105acb061e217d81c1.jpg)  
Bottom: Failed trajectory with EFR.  
Figure 15: Case 4: The conventional reflector exports the image as export.jpg, while EFR observes the duplicated suffix export.jpg.jpg but incorrectly judges it as harmless.

## Successful trajectory with the conventional reflector.

1. Action: Click File. Reflector: A drop-down menu appears, indicating that the click succeeds.

2. Action: Click Export As. Reflector: The export configuration dialog appears, confirming that the export operation has been opened successfully.

3. Action: Triple-click the filename input box. Reflector: The current filename is selected, so the input field is ready to be replaced.

4. Action: Type export.jpg. Reflector: The filename changes to export.jpg, indicating that the rename step succeeds.

5. Action: Click Export. Reflector: A new export configuration dialog appears, confirming that the first export click succeeds.

6. Action: Click Export again. Reflector: The export configuration dialog disappears, indicating that the export has been confirmed and the task is completed successfully.

## Failed trajectory with EFR.

1. Action: Click File. Reflector: A drop-down menu appears, indicating that the click succeeds.

2. Action: Click Export As. Reflector: The export configuration dialog appears, confirming that the export operation has been opened successfully.

3. Action: Type export.jpg in the filename input box. Reflector: The filename text is selected, so the reflector judges that the filename field has been activated successfully.

4. Action: Type export.jpg. Reflector: The filename becomes export.jpg.jpg. EFR correctly observes the duplicated suffix, but incorrectly judges that the extra suffix is caused by system behavior and does not affect the task.

5. Action: Click Export. Reflector: A new export configuration dialog appears, confirming that the first export click succeeds.

6. Action: Click Export again.

Reflector: The export configuration dialog disappears, so the reflector judges the export as complete, but the final filename is still incorrect.

## H.4 EFR Failure Case Analysis

The two EFR failure cases above reflect two different sources of remaining errors. In Case 3, the main issue is fine-grained grounding. Moving an object in a slide requires accurately selecting the intended table and dragging it to a precise location. EFR detects that one drag action does not produce the intended effect, and later observes that the text rather than the table appears to move. However, this reflection signal is not effectively converted into a recovery strategy by the subsequent planner and operator. The agent continues to issue similar manipulation actions, so the trajectory still accumulates unresolved layout errors. This case shows that even when the reflector correctly identifies a local execution failure, the full agent may still fail if later modules cannot revise the plan or repair the state.

In Case 4, the failure comes from both an input error and a semantic misunderstanding of the observed change. The agent first fails to properly replace the existing filename, causing the typed string to produce export.jpg.jpg. EFR observes this duplicated suffix, but incorrectly interprets it as a harmless system behavior rather than a violation of the target filename requirement. As a result, the agent accepts the intermediate state and proceeds to export the file with an incorrect name. This case shows that explicit visual evidence alone is not always sufficient: the final judgment still requires correctly mapping observed changes to task semantics, especially when small textual differences determine task success.

Overall, these cases indicate that the scattered negative results are not caused by a systematic weakness of EFR on a particular task category. Instead, they arise from limitations of the broader agent pipeline: precise grounding and manipulation remain difficult, recovery from harmful actions is still weak, and the verifier can still misinterpret the semantics of correctly observed evidence.