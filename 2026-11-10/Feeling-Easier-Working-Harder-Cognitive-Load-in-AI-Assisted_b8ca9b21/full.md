# Feeling Easier, Working Harder: Cognitive Load in AI-Assisted Code Comprehension

Maryangela White   
Computer Science Department   
Colorado State University   
Fort Collins, United States   
maryangela.white@colostate.edu Maximilian Rose   
Computer Science Department Colorado State University Fort Collins, United States   
maximilian.rose@colostate.edu

Amy McDonnell Psychology Department Colorado State University Fort Collins, United States amy.mcdonnell@colostate.edu

Francisco Ortega   
Computer Science Department   
Colorado State University   
Fort Collins, United States   
f.ortega@colostate.edu

Abstract—Background: Software developers spend significant time reading and understanding code, a task that is increasingly being supplemented by generative AI. While these tools aim to reduce the intrinsic difficulty of reading code, interacting with them introduces a dual-task environment that may impose extraneous cognitive load that is challenging to capture.

Aims: This study investigates how access to generative AI affects developers’ real-time cognitive workload (RQ1), objective code comprehension performance (RQ2), and self-perceived code comprehension strategies (RQ3).

Method: We conducted a 3x2 within-subjects study with 20 novice computer science students who reviewed Java code snippets of varying lengths with or without AI assistance, followed by an AI-unassisted comprehension quiz. Objective cognitive workload was continuously captured using a standardized tactile Detection-Response Task (tDRT) during both reading and quiz phases, and then compared to subjective post-quiz NASA-TLX and SEQ scores.

Results: AI assistance significantly increased objective cognitive workload, raising tDRT reaction times by 11.2%, yet participants perceived workload remained unchanged. Code comprehension scores were high under both conditions, with no evidence that AI assistance improved comprehension at the difficulty levels evaluated; scores clustering near the maximum limited power to detect benefits on harder tasks. Qualitatively, participants reported trusting the AI conditionally—verifying its output against their own reading for more complex snippets, but directly trusting it for simpler code—and, while many felt their reading process was unchanged, a notable subset shifted toward skimming or reading the AI’s explanations in place of the code itself.

Conclusions: These findings reveal that developers do not accurately perceive the cognitive resources they expend when managing AI tools. This study provides evidence of added cognitive effort associated with using generative AI, pointing towards a need to account for this dual-task penalty in software engineering tools and education.

Index Terms—Cognitive Workload, Detection Response Tasks, Generative AI, Code Comprehension, Wearable Sensor

## I. INTRODUCTION

For decades, software developers have spent up to 70% of their time reading and understanding code rather than

Matthew Buckman   
Computer Science Department   
Colorado State University   
Fort Collins, United States   
matthew.buckman@colostate.edu   
Bianca Trinkenreich   
Computer Science Department   
Colorado State University   
Fort Collins, United States   
bianca.trinkenreich@colostate.edu

writing it [1, 2]. Generative AI (GenAI) assistants promise to ease this burden by translating convoluted logic into clear, natural-language explanations [3]. However, emerging evidence suggests that human-AI collaboration splits the developer’s attention, trading the effort of writing code for the hidden mental strain of prompt-formulation, context-switching, and output verification [4, 5]. Because this mental effort is largely hidden behind developers’ generally positive impressions of GenAI tools, they frequently underestimate the cognitive effort they actually expend [6, 7]. This means traditional cognitive workload measurements like the NASA Task Load Index [8] fail to capture this discrepancy, as post-task surveys only capture what a developer remembers after the task is over [9].

To uncover this hidden mental workload, researchers have commonly relied on continuous physiological tracking (e.g. EEG, fNIR, and eye-tracking) [10]. While these methods yield rich data, they require expensive equipment, fragile calibrations, and noise-sensitive signals, making them impractical for everyday software engineering environments. We address this instrumentation gap by introducing the Tactile Detection-Response Task (tDRT) to SE research. The tDRT is a low-cost, ISO-standardized behavioral instrument [11] that quantifies cognitive workload via fluctuations in reaction times to simple physical stimuli.

By leveraging the real-time tracking capabilities of the tDRT alongside traditional performance and subjective metrics, we can better examine the hidden costs of AI-assisted program comprehension. To guide this investigation, we present three core research questions:

RQ1. How does access to GenAI assistance affect developers real-time cognitive workload during code comprehension?

RQ2. How does access to GenAI assistance affect observable code comprehension performance across tasks of varying complexity?

RQ3. How do developers perceive the impact of GenAI assistance on their overall code comprehension process and strategies?

To explore these questions, we conducted a $3 \times 2$ withinsubjects experiment with 20 novice computer science students. Participants evaluated Java code snippets across three sizes under both AI-assisted and unassisted conditions. Cognitive workload was captured continuously using the tDRT and supplemented by post-task NASA-TLX and SEQ metrics. Comprehension performance was assessed via multiple-choice questions probing maintenance, summarization, mental execution, and structural understanding [12], question types that mirror realistic code review demands in collaborative software projects [13, 14]. Finally, to capture changes in strategy, we conducted a qualitative analysis of participant experiences, focusing on how utilizing the AI altered their reading methods and calibration of trust.

Our results show that AI assistance significantly increased objective cognitive workload (RQ1), raising tDRT reaction times by 11.2% over the unassisted condition $( \chi ^ { 2 } = 1 7 . 3 7 , p <$ .001), even though subjective questionnaire ratings remained unchanged. Notably, this cognitive load increase persisted even after the AI was no longer in use; during quiz phases (where AI was not accessible), participants in the AI-condition still exhibited reaction times 6.93% higher than the control group $( p = 0 . 0 0 7 )$ . Ultimately, this discrepancy demonstrates that computer science students from our study did not accurately perceive the cognitive resources expended when using GenAI for code comprehension. The subjective ease of interacting with GenAI may mask a measurable, sustained objective cost. For code comprehension performance (RQ2), scores were high under both conditions, and we found no evidence that AI assistance improved comprehension at the difficulty levels evaluated. Qualitatively (RQ3), developers reported adapting their strategies based on task difficulty, where they verified AI output against their own reading for complex snippets, but directly accepted explanations for simpler code. Furthermore, while many participants felt their habits remained unchanged between conditions, a notable subset shifted towards skimming the code during the AI-assisted condition, or relying entirely on the AI’s natural language explanation.

This paper makes novel contributions to SE research by introducing the tDRT methodology within an SE context and providing empirical evidence of the hidden cognitive costs that accompany generative AI usage in code comprehension. The remainder of this paper is structured as follows. Section II reviews existing research on cognitive workload and generative AI usage in software engineering. Sec. III details our data collection and analysis, followed by results in Sec. IV, and discussion in Sec. V. Finally Sec. VI concludes the paper and provides suggestions for possible directions of future work.

## II. RELATED WORK

## A. LLM Use in Software Engineering

The integration of Large Language Models (LLMs) into software development has radically altered the cognitive landscape of code comprehension. As code synthesis tools become widespread, a developer’s primary task frequently shifts from active code generation to passive code reading, critical evaluation, and verification [15, 16]. Neural imaging research confirm that the cognitive profiles of reading and writing code are neurologically distinct [17], meaning that developers adopting a “reviewer” role alters their underlying mental demands. Rather than outright reducing mental effort, outsourcing code generation to an AI assistant trades the cognitive load of syntax synthesis for the load of structural analysis.

This transition introduces a compounding behavioral cycle: because analyzing unfamiliar, AI-generated code remains mentally taxing, developers frequently escalate their reliance by offloading the comprehension process itself. Rather than analyzing the code independently, developers employ scaf folding tools such as CodeMap or GILT to automate the interpretation of program flow and logic structures [3, 18]. In extreme cases, developers outsource comprehension completely, prompting models to make changes within systems they do not fully understand [3, 15]. Additionally, while these automated tools aim to streamline comprehension, they can inadvertently increase cognitive load. Engaging with AI-generated code explanations forces the developer into a dual-task workflow where they must simultaneously manage the primary task and calibrate their trust in the AI’s accuracy [5]. This cognitive overhead can incentivize superficial code skimming, causing developers to overlook latent bugs [19], while fostering automation bias that risks long-term skill degradation [20]. Because these behavioral states are dynamic and can vary across a task, evaluating them requires continuous, real-time tracking of a developer’s workload.

## B. Measurement of Cognitive Load

The NASA Task Load Index has been the standard for measuring perceived cognitive load for nearly 40 years [8], however its subjectivity has drawbacks [9, 21]. For objectively measuring real time cognitive load, physiological modalities such as eye tracking and electroencephalograms (EEGs) can be processed to gauge mental effort. While these devices enable live tracking during a task, they are often cost-prohibitive, require highly technical calibration, and are sensitive to environmental noise [10]. To avoid these issues, behavioral modalities such as the Detection Response Task (DRT) offer an alternative, well-validated method for measuring real-time cognitive workload by requiring participants to respond to periodic stimuli while performing a primary task [22]. These devices have been widely used throughout automotive research to assess distractions while driving. Their application has also begun to expand into other human factor domains, such as software engineering and interface evaluation [22, 23]. Because the DRT provides a robust, low-overhead method for tracking fluctuations in cognitive workload, we utilize this metric as a core behavioral measure in our study.

## C. Dynamics of Code Comprehension

Program comprehension is driven by a dynamic interaction between the developer’s internal cognitive strategies and external code characteristics. Literature establishes two primary paradigms of code reading: bottom-up processing, where syntax is mentally grouped into higher-level abstractions, and top-down processing, where existing domain knowledge and semantic “beacons” are utilized [24]. Rather than relying on a single mode, developers typically alternate between these strategies based on the demands of the task and available mental resources [25].

Code scale can heavily influence these cognitive shifts: small code snippets facilitate linear, bottom-up comprehension, while larger code pieces encourage a shift towards top-down abstraction [26]. Consequently, the impact of AI-assistance during comprehension may vary between code sizes. For smaller snippets requiring statement-level processing, AI explanations may reduce the line-by-line interpretive load; for larger snippets where effective navigation and structural chunking become vital [25, 26], AI may instead assist by surfacing high-level structure. The nature and magnitude of AI’s benefit likely differs across these scales. To capture these potential interaction effects, this study evaluates comprehension across three code sizes.

Because developers alternate between these strategies, evaluating the impact of AI assistance requires a multi-dimensional assessment framework. As such, the code samples and comprehension questions utilized in this study were developed according to the code comprehension framework by Wyrich et al. [12]. According to this framework, code comprehension measurement can be divided into four parts: maintenance, program summarization, mental execution, and structural mapping. Evaluating the impact of AI-assistance across these dimensions ensures that our performance metrics capture the multi-layered nature of program comprehension.

## III. RESEARCH METHODOLOGY

Our study investigated the impact of generative AI-assistance on cognitive workload, objective performance, and user perception across varying code sizes. We manipulated two independent variables: code snippet size (small, medium, large) and AI availability (with/without), utilizing a within-subjects design to control for individual differences in programming ability, reading speed, and prior Java familiarity [1].

Participants completed six Java comprehension tasks. Within each task, tDRT reaction time was recorded as a proxy for real-time cognitive workload, followed by post-task evaluations using the NASA-TLX and SEQ scales. After completing each task, participants were given a one-minute rest period before the next task, to help mitigate cumulative fatigue. This study was conducted with approval from the Institutional Review Board of Colorado State University.

## A. Participants and Recruitment

The final analysis sample consisted of $N = 2 0$ software engineering students recruited from two Java-based courses at Colorado State University. To justify our sample size, we evaluated statistical strength using a dual approach based on both a standardized two-condition difference [27] and a simulation-based sensitivity analysis [28]. Following common practice in previous DRT studies, which analyze data with linear mixed-effects models but utilize two-condition difference calculations for sample planning, our design $( N = 2 0 , \alpha = . 0 5 ,$ 80% power) corresponds to a minimum detectable standardized effect of approximately $d \ = \ . 6 6$ , falling between Cohen’s conventional medium (.50) and large (.80) benchmarks [29].

However, because this approximation does not capture the repeated-measures and random-effects structure of our data, we also conducted a SIMR simulation-based sensitivity analysis [28] using the mixed-effects structure of our primary model (500 simulations run). For our primary comparison of interest (AI vs no-AI condition), our design achieves 80% power at $\alpha = . 0 5$ for an AI-condition coefficient of 0.083 on $\log _ { 2 }$ reaction time, corresponding to a detectable change of approximately 5.9% in RT $( 2 ^ { 0 . 0 8 3 } - 1 )$ . (Note: Due to computational constraints, this sensitivity analysis focuses specifically on the main effect of the condition rather than the subsequent interaction models).

The sample had a gender distribution of 60% men, 30% women, and 10% non-binary. Full demographic information is presented in Table I

TABLE I: Participant Demographic and Experience Summary $( N = 2 0 )$
<table><tr><td>Attribute</td><td>Category</td><td>Percentage (%)</td></tr><tr><td>Age</td><td>18-20</td><td>45%</td></tr><tr><td></td><td>21-23</td><td>30%</td></tr><tr><td></td><td>24-26</td><td>10%</td></tr><tr><td></td><td>27-29</td><td>10%</td></tr><tr><td></td><td>30+</td><td>5%</td></tr><tr><td>Java Experience</td><td>&lt; 1 year</td><td>5%</td></tr><tr><td></td><td>1–2 years</td><td>45%</td></tr><tr><td></td><td>3–4 years</td><td>35%</td></tr><tr><td></td><td>5+ years</td><td>15%</td></tr><tr><td>LLM Usage</td><td>Never</td><td>30%</td></tr><tr><td></td><td>Rarely</td><td>20%</td></tr><tr><td></td><td>Occasionally</td><td>5%</td></tr><tr><td></td><td>Frequently</td><td>40%</td></tr><tr><td></td><td>Very Frequently</td><td>5%</td></tr></table>

## B. Tactile Detection Response Task (tDRT)

The Detection-Response Task (DRT) is a standardized human factors tool for measuring fluctuations in cognitive workload during a primary task [11]. The participant is asked to respond to a stimulus presented every 3-5 seconds by pressing a button attached to their finger. The DRT’s ability to measure cognitive workload fluctuations comes from the responses to the stimuli slowing or not occurring when the primary task requires greater cognitive resources [23].

While the DRT supports tactile, visual, and auditory configurations, a tactile stimulus modality was selected to prevent sensory interference with the primary visual code review task [22]. The stimulus was administered via a commercial tDRT system manufactured by Red Scientific [30], utilizing a small vibrating motor attached to the participant’s shoulder or arm and a finger-mounted response button (see Figure 1).

During the study, tDRT data was recorded across all six tasks and partitioned into reading and quiz segments. To control for confounding physical artifacts, the experimenter manually flagged periods when participants were actively typing, as responding via the finger-mounted DRT button can physically conflict with keyboard use. During pre-processing, these timestamps were cross-checked against screen recordings, and DRT responses falling within those intervals were excluded (similar researcher-recorded temporal markers have been used in prior DRT studies to define task intervals [31]). Because the goal was simply to identify observable typing periods rather than infer subtle cognitive states, millisecond-level annotation was unnecessary. Removing these typing windows, which accounted for 2.6% of total observations, was a conservative methodological choice. Because typing occurred exclusively during the AI-assisted condition, retaining those confounded trials would have artificially inflated the estimated AI effect (increasing the coefficient from 0.1536 [11.2%] to 0.1828 [13.5%]). Furthermore, during the quiz-phase (where no typing occurred in either condition) RT remained significantly higher following AI-assisted tasks. Ultimately, this filtering ensured that the dataset isolated cognitive workload rather than mechanical interference from querying the LLM.

Raw tDRT data underwent standard pre-processing and filtering. In accordance with ISO 17488 guidelines, any response recorded under 100 ms was classified as an accidental trigger and excluded, while signals receiving no response within 2, 500 ms were categorized as misses. [22]. Only valid responses within this window were included in reaction time calculations, with all misses and invalid triggers omitted. Although the DRT yields both reaction time and hit rate metrics [22], overall hit rates remained consistently high (above 95%). This is a common phenomenon in stationary, nondriving DRT applications where participants easily maintain baseline compliance. Because this lack of variance in hit rate restricts statistical modeling, reaction time was selected as the dependent variable for all subsequent DRT analyses. This choice is supported by prior evidence showing that RT remains sensitive even when accuracy metrics show a ceiling effect [32]. Nevertheless, we conducted a descriptive check: hit rates averaged 93% in the AI-assisted condition compared to 97% in the non-AI condition. Although minor, this slight increase in misses under AI assistance aligns directionally with the reaction time findings, offering converging evidence of an increased cognitive load.

## C. Generative AI Assistance

During the AI-assisted condition, participants were provided access to the publicly available web interface of chat-based Gemini for the code reading phase. Gemini was selected because of the time-constrained nature of the tasks, which required a model with consistently fast response latency [33]. To prevent carryover effects between trials, each task used a new chat session opened in an incognito browser window, ensuring that the model had no access to prior conversation history from earlier trials or participants.

![](images/0d9bec1ddb01f7bcf3c1cb19a6f8a6389c36d2010f1a24c5ea41b87c135ea016.jpg)  
Fig. 1: tDRT Set Up

Participants were free to decide how often to consult the AI system during the code review phase. Interaction with the system was free-form; participants could ask questions, paste the provided code snippets, or request explanations, with no constraints on prompt wording or format. No training in prompt engineering was provided to approximate how computer science students typically interact with generative AI tools to understand code. While differences in prompting strategies may influence the type or quality of assistance received, such variability reflects realistic usage patterns and therefore was considered part of the natural interaction with the system. Participants were instructed to avoid inputting any personal or sensitive information. All interactions with the AI system, including prompts and model responses, were captured via screen recording software to assist in typing interval crossreferencing.

## D. Code Snippets

The tasks required participants to review Java code snippets of varying sizes: two small snippets (approx. 30 lines each), two medium-sized snippets (approx. 45 lines each), and two larger code sections (approx. 70 lines each). Manipulating snippet lengths allowed for analysis of code comprehension strategies under different scaling constraints.

To ensure high internal validity, all code snippets were developed specifically for this study. This allowed for close control over cyclomatic complexity and domain neutrality, ensuring that differences in cognitive workload could be safely attributed to experimental conditions, rather than incidental features of the code [13]. Following previous research [14], snippets isolated specific semantic structures and avoided unfamiliar Java constructs or external library dependencies. Furthermore, to minimize confounding variables such as domain-specific jargon [34], we followed the Wyrich et al. framework [12], and utilized core language constructs covered within the participants’ university curriculum.

## E. Code Comprehension Questions

Code comprehension was assessed via four multiple-choice questions per code task. The questions covered maintenance, summarization, mental execution, and structure [12] and were designed to require the level of understanding needed to modify or extend the code. This emulates a realistic developer experience where one must comprehend existing logic to contribute to a collaborative project.

To ensure participants completed their primary analysis during the initial code reading phase, a restrictive time window (2, 2.5, or 3 minutes) was applied to this comprehension phase; the code snippet remained visible during the quiz, but the time constraint discouraged re-reading it in lieu of the prior analysis. Specifically within the AI-assisted tasks, this constraint prevented participants from bypassing the initial code review and simply deferring code reading to the quiz phase when the AI tool was no longer accessible. Complete code snippets and comprehension questions are available in the replication package [35].

## F. Experimental Design

The study employed a 3 × 2 within-subject design, manipulating code size and AI assistance. For each of the three code size categories, two unique code snippets were prepared, one to be completed with AI assistance, and one without. This resulted in six total code review tasks per participant.

To mitigate potential learning and fatigue effects across the study, the order of the three code size categories was randomized for each participant. Within each size block, the two assigned tasks were completed consecutively. This blocked design was chosen because different code sizes require different reading approaches [26, 36]; constantly shifting between snip pet lengths could cause frequent context switching that would artificially inflate cognitive load measurements[37]. While assignment of AI assistance to a code snippet within a size block was counterbalanced across participants, task sequence order within a specific size category was randomized. This within-block randomization prevented order or priming effects without introducing excessive experimental combinations, while also ensuring that each code snippet received an equal allocation of AI-assisted and unassisted trials.

## G. Questionnaires

Immediately following each code task, participants completed two questionnaires to measure subjective cognitive load: the NASA-TLX [8] and the SEQ [38]. During this questionnaire window the tDRT device was deactivated. Upon finishing all six tasks, participants completed an exit survey, including demographics (summarized in Table I), closed questions about how frequently they use LLM tools for coding and if they have used LLMs for code comprehension before, and open questions about the experiment on the extent to which they trusted the output provided by the LLM during the code comprehension tasks, and whether and how the AI-assistance had changed their approach to reading code.

## H. Data Analysis

1) RQ1 Data Analysis: To evaluate the impact of generative AI assistance on cognitive workload, tDRT and self-reported outcomes data was analyzed using R (version 4.6.0) and the dplyr (v1.2.1), lmerTest (v3.2.1), emmeans (v2.0.3), and pbkrtest (v0.5.5) packages.

We used Linear Mixed-Effects (LME) models because they account for the correlation between repeated measurements within participants and accommodate missing data without excluding participants with incomplete observations [39]. Prior to model fitting, the distribution of reaction time data was checked via visual inspection of a QQ-plot. Confirming a characteristic right-skew, reaction time values were log - transformed to better meet the assumptions of normality, following established empirical standards [40].

The LME framework splits variables into fixed and random effects. Fixed effects represent the experimental factors manipulated or measured to assess their systematic impact on our dependent variables (reaction time or subjective workload). Importantly, our core fixed effects and the specific interactions we tested – the main effect of AI-assistance (assisted vs. no-assistance), its interaction with Task Phase (Reading vs. Quiz), its interaction with Code Size (Small, Medium, and Large), and its interaction with Java Experience – were determined during study design, prior to any data collection or analysis. Java experience was specified as a fixed effect because recruitment was restricted to students with an established minimum proficiency in Java, and we hypothesized that AI impact would vary by baseline skill. Controlled experimental factors that occurred across all participants, such as Code Size and Task Phase, were likewise designated as fixed effects.

Random effects, conversely, account for unmanipulated sources of variation inherent in the sample population. Participant ID and Code Snippet ID were designated as random intercepts to control for individual differences between participants and difficulty variances across specific coding tasks. To determine final model configurations and avoid overparameterization, we utilized Akaike Information Criteria (AIC) model comparison. AIC balances goodness-of-fit against model complexity by penalizing additional parameters, guarding against overfitting (where a model captures sample noise rather than true underlying patterns) [41]. Because maximal LME specifications risk convergence failures, we started from a baseline specification (the primary effect of interest and random participant intercept) before sequentially evaluating additional variables, retaining them only if they lowered the AIC score. For instance, to evaluate whether broader student demographics (Gender, GPA, and LLM Usage Frequency) introduced meaningful baseline variance, we tested them in the model; however they explained negligible variance and increased AIC scores, threatening convergence without improving fit. Consequently, they were omitted. In contrast, Participant ID, Snippet ID, Code Size and Task Phase were retained because their inclusion consistently optimized model performance. This systematic procedure prevented arbitrary feature selection while accurately accounting for the repeated-measures dependencies of the data prior to conducting likelihood ratio tests.

With the final model structures established, hypothesis testing was executed via Likelihood Ratio Tests (LRT) [39]. An LRT evaluates a specific factor by comparing a ”full” model containing the term of interest (such as AI assistance) against an identical null model lacking only that term. Because each test must isolate a single factor or interaction, testing multiple effects requires separate model comparisons; bundling multiple new variables into a single test would obscure which specific variable is driving any observed change in model fit. If the LRT shows that the full model fits the data significantly better than the null model, it means the factor in question has a statistically significant impact.

To isolate specific facets of how AI-assistance impacted cognitive workload across our tDRT and questionnaire (NASA-TLX & SEQ) measurements, our analysis was partitioned into six LME models and corresponding LRTs (summarized in Table IV):

• Model 1 (Global Main Effect): Evaluates the overall impact of AI-assistance on tDRT reaction times across the entire study.

• Model 2 (Task Phase Interaction): Tests whether the cognitive impact of AI-assistance varied significantly between code reading phase and quiz phase.

• Model 3 (Code Size Interaction): Tests if the impact of AI assistance changed based on the size of the code snippet.

• Model 4 (Java Experience Interaction): Examines whether a participant’s programming experience affects how AIassistance impacted their reaction time.

• Model 5 & 6 (Subjective Workload): Evaluates the impact of AI-assistance on perceived workload and task difficulty, using unweighted NASA-TLX sub-scale averages and single-item SEQ responses respectively.

2) RQ2 Data Analysis: Code comprehension was assessed via the accuracy score (0–4) on the four multiple-choice questions per task, analyzed in R using the same lme4/lmerTest toolchain as RQ1. We first computed descriptive accuracy means by AI condition and code size (Table V). Our planned analysis modeled the 0–4 accuracy score using a linear mixedeffects model with AI assistance as a fixed effect and byparticipant and by-snippet random intercepts, testing the effect of AI assistance via a likelihood ratio test as in RQ1. Because a four-point sum is a bounded, discrete outcome that does not fully satisfy the assumptions of a linear model, we additionally fit an ordinal (cumulative-link) mixed model with the same fixed- and random-effects structure as a sensitivity analysis, using the ordinal package [42].

3) RQ3 Data Analysis: We analyzed participants’ responses to the two open-ended questions on trust and reading-behavior changes using inductive thematic analysis. Responses were first segmented into discrete meaning units to isolate distinct ideas. Two researchers collaboratively conducted open-coding on these segments, organizing them into an affinity diagram using Miro [43]. Affinity diagramming is a common synthesis technique in UX research that supports the collaborative grouping of related observations into higher-level themes [44]. Through a negotiated agreement strategy [45], a third researcher joined discussions to iteratively refine groupings into coherent themes, which were then reviewed by a fourth researcher with expertise in qualitative analysis and human factors in software engineering. No predefined codebook was used; instead, themes were developed inductively from the data.

## I. Study Procedures

Upon arriving at the study location, participants reviewed an informed consent form and provided verbal consent, with the reminder that they could withdraw at any time. The experimenter then briefed them on the overall procedure, the code tasks, and the use of the tDRT. The participant was then fitted with the tDRT equipment, placing the response button on the thumb of their non-dominant hand and securing the vibrating motor secured to their shoulder or arm (Figure 1). To ensure familiarity with the tDRT stimulus and response mechanism, participants completed a brief practice session with a sample code snippet and a single multiple-choice question, before continuing to the primary experimental tasks.

![](images/8b91675823cf2d9952c6306b86fc385bcd6decbe9379def18beaa0431305c613.jpg)  
Fig. 2: Study Procedure Diagram

![](images/9b9baf374265585b77f02702bd7c01b76cc55bec55a57c41101034393e85ea7b.jpg)  
Fig. 3: Study Set Up

Tasks were completed at a workstation set-up with a laptop and adjacent secondary monitor. Code snippets and quiz questions were displayed on the laptop, while the connected monitor was reserved for displaying the Gemini browser interface during the AI-assisted code reading (Figure 3). The presentation order of the three code size categories and the task order within each size category were randomized, as detailed in Figure 2.

Each of the six tasks followed a three-phase sequence:

1) Code Reading Phase: Participants analyzed the code snippet with free access to the secondary LLM window (AI-assisted condition), or with the screen minimized (control condition). To ensure sufficient engagement with the code, a ”Next” button only appeared after a sizedependent minimum threshold (2.5, 3, or 3.5 minutes). Participants automatically advanced to the next phase if the maximum time limit was reached (4, 5, or 6 minutes).

TABLE II: Study Measurements
<table><tr><td>Measurement</td><td>Variable Type</td><td>Source</td></tr><tr><td>tDRT reaction times</td><td>Dependent</td><td>tDRT Device [11]</td></tr><tr><td>Quiz scores</td><td>Dependent</td><td>Comprehension Quiz [35]</td></tr><tr><td>TLX and SEQ scores</td><td>Dependent</td><td>Post Quiz Survey [8, 38]</td></tr><tr><td>Background questions</td><td>Independent</td><td>Post Study Survey [35]</td></tr></table>

TABLE III: Experimental Protocol for Participant #1
<table><tr><td>Task Order</td><td>Code Size &amp; ID</td><td>AI Assist.</td></tr><tr><td>1</td><td>Small1</td><td>No</td></tr><tr><td>2</td><td>Small2</td><td>Yes</td></tr><tr><td>3</td><td>Medium2</td><td>Yes</td></tr><tr><td>4</td><td>Medium1</td><td>No</td></tr><tr><td>5</td><td>Large2</td><td>Yes</td></tr><tr><td>6</td><td>Largel</td><td>No</td></tr></table>

2) Comprehension Phase: Participants completed the four multiple-choice questions on the laptop screen. Participants could manually advance after answering all questions, or were automatically advanced if a maximum time limit was reached (2, 2.5, or 3 minutes).

3) Subjective Evaluation Phase: The tDRT device was deactivated, and participants completed a NASA-TLX questionnaire and SEQ to report perceived workload and task difficulty.

Following the questionnaire, participants rested for one minute before proceeding to the next task.

## IV. ANALYSES AND RESULTS

Twenty-two participants completed the study, but data from two were excluded from analysis: one due to technical issues that disrupted the study structure, and one who did not interact with the AI system during the AI-assisted condition, rendering their AI-condition data invalid. The resulting twenty participants’ demographic information is presented in Table I.

We now address the research questions introduced in Section I.

A. How does access to GenAI assistance affect developers’ real time cognitive workload during code comprehension? (RQ1)

RQ1 is answered by assessing developer workload through objective tDRT response times and subjective NASA-TLX and SEQ metrics. As detailed in the methodology(Section III-B), raw tDRT data underwent pre-processing to exclude typing intervals and invalid reaction times, yielding a final dataset of 8, 048 tDRT responses for statistical modeling. To isolate the specific facets of this data we executed six LME models, with four analyzing the tDRT metrics and two evaluating the questionnaires, as summarized in Table IV.

TABLE IV: tDRT & Questionnaire LRT Results
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>DV</td><td rowspan=1 colspan=1>Tested Effect</td><td rowspan=1 colspan=1> $\chi ^ { 2 }$ </td><td rowspan=1 colspan=1>DF</td><td rowspan=1 colspan=1>p-value</td></tr><tr><td rowspan=1 colspan=1>Model 1</td><td rowspan=1 colspan=1>RT</td><td rowspan=1 colspan=1>AI Assistance</td><td rowspan=1 colspan=1>17.37</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1> $< . 0 0 1 ^ { * * }$ </td></tr><tr><td rowspan=1 colspan=1>Model 2</td><td rowspan=1 colspan=1>RT</td><td rowspan=1 colspan=1>AI×Task Phase</td><td rowspan=1 colspan=1>7.10</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0.008**</td></tr><tr><td rowspan=1 colspan=1>Model 3</td><td rowspan=1 colspan=1>RT</td><td rowspan=1 colspan=1>AI×Code Size</td><td rowspan=1 colspan=1>8.22</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>0.016*</td></tr><tr><td rowspan=1 colspan=1>Model 4</td><td rowspan=1 colspan=1>RT</td><td rowspan=1 colspan=1>AI×Java Exp</td><td rowspan=1 colspan=1>0.89</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>0.828</td></tr><tr><td rowspan=1 colspan=1>Model 5</td><td rowspan=1 colspan=1>TLX Score</td><td rowspan=1 colspan=1>AI Assistance</td><td rowspan=1 colspan=1>1.80</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0.180</td></tr><tr><td rowspan=1 colspan=1>Model 6</td><td rowspan=1 colspan=1>SEQ Score</td><td rowspan=1 colspan=1>AI Assistance</td><td rowspan=1 colspan=1>1.96</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0.161</td></tr></table>

Note: <sup>∗</sup>Statistically significant at the $p < . 0 5$ level; <sup>∗∗</sup>significant at the $p < . 0 1$ level.

![](images/ec99b8e2f66cde2ea34ef08204ae90a529b707e3ba250ab9a2ebce0924a8c543.jpg)  
Fig. 4: Main Effect of AI Assistance on Cognitive Workload

Model 1 isolated the global main effect of AI assistance on tDRT reaction times, specifying AI Assistance, Task Phase, and Code Size as fixed factors. A likelihood ratio test confirmed that the inclusion of the AI Assistance factor significantly improved overall model fit $( \chi ^ { 2 } = 1 7 . 3 7 , p < . 0 0 1$ , see Table IV). This indicates that the presence of AI assistance was a significant predictor of reaction time, which indicates an increase in cognitive workload. Back-transforming the model coefficients from the $\log _ { 2 }$ scale revealed that AI assistance increased average reaction times by 11.2%. This effect is visualized in Figure 4, which displays the estimated marginal means of tDRT response times across conditions.

To assess whether the impact of AI assistance varied by experiment stage (reading or quiz), we evaluated the interaction between AI Assistance and Task Phase (Model 2, Table IV). The LRT confirmed a significant interaction $( \chi ^ { 2 } ~ = ~ 7 . 1 0 ,$ $p = 0 . 0 0 8 )$ , indicating that the influence of AI on cognitive workload was not uniform across task stages.

As illustrated in Figure 5, overall reaction times increased for both groups when moving to the quiz phase. The control condition exhibited a steeper relative increase across this transition, which was due to starting from a lower baseline during the reading phase. Specifically, post-hoc pairwise comparison showed that during code reading, AI-assistance elevated reaction times by 12.7% $( p < . 0 0 1 )$ . Once participants moved to the quiz phase, this gap narrowed to a smaller, but still statistically distinct 6.9% $( p ~ = ~ 0 . 0 0 7 )$ increase compared to the control condition. These results demonstrate a significant carryover effect, where AI usage during the reading phase resulted in a higher cognitive workload during the subsequent quiz phase, despite AI no longer being in-use.

![](images/3f04074ef3e1299e4d5bc0930b1da32ea6108c5b07429709f3e697d2ffad2601.jpg)  
Fig. 5: Interaction between AI Assistance and Task Phase on Cognitive Workload

Model 3 evaluated the interaction between AI-assistance and Code Size to determine if the impact of AI on cognitive workload changed based on code snippet size. The LRT confirmed a significant interaction effect $( p = 0 . 0 1 6 )$ , though AI assistance elevated reaction times across all three size categories: Small $( 8 . 9 \% , p < . 0 0 1 )$ , Medium $( 1 5 . 2 \% , p < . 0 0 1 )$ and Large (9.5%, $p < . 0 0 1 )$

Post-hoc interaction contrasts indicated that the magnitude of this reaction time increase was significantly more pronounced in medium-sized tasks than in either small $( \mathrm { z } = 2 . 4 7 , \mathrm { p } = 0 . 0 3 6 )$ or large tasks $( \mathrm { z } = 2 . 4 4 , \mathrm { p } = 0 . 0 3 9 )$ . The contrasts also confirmed that the AI-influenced reaction time increases in the small and large tasks were statistically equivalent to each other $( z = 0 . 2 5 ,$ $\mathrm { p } = 0 . 9 6 8 )$ . Interestingly, Figure 6 shows that this variance is caused by a baseline dip within the control condition during medium-sized tasks, rather than a spike in workload caused by AI usage. Ultimately, AI-assistance overrode the significant relief in cognitive load observed in the control condition during medium-sized tasks, maintaining a persistently higher workload across all code lengths.

Finally, Model 4 evaluated the interaction between AI Assistance and Java Experience to assess whether the impact of AI on cognitive workload varied with user expertise (Table IV). No significant interaction was found between these variables $( p \ : = \ : 0 . 8 2 8 )$ , demonstrating that the AI-induced cognitive workload increase remained consistent regardless of participants’ level of Java familiarity.

![](images/efaa5b61c66084189bdc74033f4ba119b27ca33b680dcfce401071da4e75aece.jpg)  
Fig. 6: Interaction between AI Assistance and Code Size on Cognitive Workload

Following the tDRT analysis, Models 5 and 6 evaluated the questionnaire data to determine if the presence of AI-assistance altered participants’ perception of their workload and task difficulty (Table IV). The LRTs showed that AI-assistance did not significantly impact scores for either the NASA-TLX $( p = 0 . 1 8 0 )$ or the SEQ $( p = 0 . 1 6 1 )$ . These results indicate that while AI assistance significantly increased objectively measured cognitive workload, it did not lead to a statistically significant change in perceived workload or self-reported task difficulty.

RQ1. AI assistance significantly increased objective cognitive workload, as indicated by tDRT reaction times 11.2% higher in the AI-assisted condition. This effect persisted beyond active tool use: even after AI was removed, reaction times in the AI-assisted condition remained elevated during the quiz phase. Perceived workload, as measured by NASA-TLX and SEQ, was unchanged between conditions. Rather than reducing cognitive load, AI assistance introduced a dual-task penalty that participants did not detect through self-report alone.

B. How does access to GenAI assistance affect observable code comprehension performance across tasks of varying complexity? (RQ2)

To evaluate whether access to generative AI improved code comprehension, we analyzed accuracy on the four-question quiz (scored 0–4) using our planned mixed-effects model, with AI assistance as a fixed effect and by-participant and by-snippet random intercepts.

Mean accuracy was high and nearly identical across condi tions (AI: M= 3.22, SD= 0.99; No-AI: M= 3.23, SD= 1.08; Table V). Scores clustered near the top of the scale: a perfect score across all four facets was achieved on roughly half of all tasks. Because the four question types award partial credit across distinct comprehension facets (maintenance, summarization, mental execution, structural mapping) [12], participants reliably answered most individual facets, compressing the aggregate score toward its maximum and producing a ceiling effect.

TABLE V: Code Comprehension Performance by Condition
<table><tr><td>Measure</td><td>AI Assisted</td><td>No AI</td></tr><tr><td>Mean accuracy, 0–4 (SD)</td><td>3.22 (0.99)</td><td>3.23 (1.08)</td></tr><tr><td>Complete comprehension (all 4 facets)*</td><td>50.0%</td><td>58.3%</td></tr></table>

<sup>∗</sup>Descriptive rate of tasks receiving a perfect score; reported to characterize the ceiling effect.

The planned model found no significant effect of AI assistance on comprehension accuracy $( \chi ^ { 2 } ( 1 ) < 0 . 0 1 , p = . 9 8 2 )$ The ordinal sensitivity analysis agreed, likewise showing no significant effect of AI $( \beta = - 0 . 1 6 , z = 0 . 4 0 , p = . 6 9 2 )$ confirming the null was not an artifact of fitting a linear model to the bounded, ceiling-affected scale.

We therefore found no evidence that AI assistance improved code comprehension at the difficulty levels evaluated in this study. We read this as an absence of evidence for a benefit, not as evidence that AI confers none.

RQ2. AI assistance did not improve code comprehension. Quiz scores were high and nearly identical between conditions (AI: M= 3.22; No-AI: M= 3.23), and both the planned mixed-effects model (p = .982) and an ordinal sensitivity analysis (p = .692) found no significant effect of AI. Because scores clustered near the maximum, the study had limited power to detect benefits on harder tasks: we find no evidence that AI improved comprehension at the difficulty levels tested, rather than evidence that it provides none.

C. How do developers perceive the impact of GenAI assistance on their overall code comprehension process and strategies? (RQ3)

To understand the human factors underlying our objective metrics, we analyzed the open-text post-task responses regard ing both participants’ trust in the LLM and its impact on their code reading behavior. Our thematic analysis of these responses revealed that the trust and reading behaviors were intersectional, the level of trust impacted how they utilized the AI, and the output received during usage impacted their trust. As such, we utilized an affinity diagram to analyze response snippets together. From this combined dataset, we identified three behavioral themes:

Resistance (Low/No AI-Use, High Human Agency) Participants exhibiting resistance reported using the GenAI only to confirm an understanding they had already formed independently. Within this strategy, trust was restricted, with the AI utilized only for validating higher-level structures or independently verified conclusions: “I trust it enough to get me the bones of a piece of useful code, but not the full thing.” For these individuals, code reading mechanics remained largely unchanged: “I still read the code top down”

Scaffolding (Moderate/High AI-Use, Balanced Human Agency) The most common strategy was scaffolding, where participants used the GenAI as an initial orientation tool to provide a structural overview that accompanied their reading of the code. Within this theme trust was conditional, varying according to how complex a participant perceived the task to be. Where the code felt straightforward, participants accepted the AI explanation at face value: “I generally trusted it because the code was relatively simple.”

Conversely, for responses addressing more complex logic, participants maintained agency by verifying the AI against their own developing mental model: “I fully trusted it only if what I though[t] was the output matched the LLM’s interpretation and I was also semi-confident in my solo interpretation.” This scaffolding approach reshaped participants’ reading mechanics, allowing them to bypass low-level syntax tracking in favor of a macro-level evaluation of the code: “It moved me from reading the code in smaller blocks to reading it in larger blocks by method.”

Displacement (High AI-Use, High Over-Reliance) The third identified strategy was displacement, where participants allowed the GenAI output to actively substitute for direct code reading. Within this theme, trust shifted towards over-reliance, causing participants to minimize or entirely bypass linear code analysis. This resulted in participants lightly skimming the text or offloading the cognitive task of comprehension to the AI: “I then almost exclusively read the LLM output instead of the code.” For some the code was reduced to a secondary reference: “[I ended up] only glancing at the code when I had a small question or curiosity about the way it was implemented.” This substitution did not always pay off; one participant reported it cost them comprehension, as they “took longer to read the excessive and incomprehensive writing by the ai and missed a lot of vital information that just reading the code would have imparted.”

RQ3. Coding participants’ open-text responses surfaced three recurring strategies, distinguished by how much they used the AI and how far they relied on it, with trust consistently conditional and scaled to perceived task complexity. Some set the tool aside and read independently (Resistance); most used it as a macro-level reading guide while checking it against the code (Scaffolding); and roughly a quarter let its explanations displace direct code reading entirely (Displacement).

## V. DISCUSSION, LIMITATIONS AND IMPLICATIONS

A comprehension aid that adds load. Prior work has framed GenAI assistance primarily as a tool for reducing the cognitive burden of code comprehension, offloading explanation and summarization to the model [3]. Our results challenge that framing. AI assistance did not reduce workload; it increased it, reliably and across all code sizes and experience levels. The reason, we argue, is that AI introduces a dual-task structure that prior framing overlooked. The comprehension task does not disappear when AI is present; rather, it is joined by a second task of managing the AI interaction. This mental overhead is explicitly reflected in our qualitative findings, where participants across the Resistance and Scaffolding themes reported consciously verifying the AI output against the code. This active verification behavior embodies the precise mechanism of our proposed dual-task load, where the developer must simultaneously process the code and compare the AI’s output against their own mental model. Whether the net effect is ultimately beneficial depends on whether the assistance gained outweighs the overhead incurred, a trade-off future work with more complex tasks should prioritize.

Feeling easier does not mean working less. Despite an 11.2% elevation in tDRT reaction times, neither NASA-TLX nor SEQ scores differed between conditions. This dissociation between objective and perceived workload is not new [9, 21], but GenAI creates a specific condition that prior work did not anticipate. Unlike a complex query language or an unfamiliar IDE, which feel demanding, GenAI assistants are designed to reduce apparent friction. Formulating prompts, evaluating text-heavy model replies [46], and switching attention between code and AI interface [47] all consume attentional resources beneath awareness, producing a metacognitive blind spot that self-report instruments like the NASA-TLX are structurally unable to detect [9, 21].

The cost that stays when the tool leaves. Our results show that once the AI was removed, participants in the AIassisted condition still exhibited reaction times 6.93% higher than controls $( p = 0 . 0 0 7 )$ under identical conditions. If the cost of using AI were simply tool-operation overhead, it would have disappeared when the tool was removed. Instead, this sustained elevation indicates that AI assistance likely short-circuited the active inference and execution tracing necessary to build a stable mental model. When the tool was removed during testing, participants had a weaker cognitive foundation, making the quiz phase significantly more effortful. This behavior appears to extend traditional automation complacency frameworks [48], suggesting that the primary risk of generative tools may not be merely over-reliance, but a displacement of the effortful processing required for independent knowledge construction – a hypothesis with implications for long-term encoding and recall that future longitudinal research should directly test.

The overhead is in the tool, not the task. The interaction analyses revealed that developer experience did not moderate the workload increase. Participants with more Java familiarity were more efficient readers in the unassisted condition, yet this efficiency did not reduce the attentional cost of AI interaction, as the AI Assistance × Java Experience interaction was not significant (see Table IV, Model 4, $p = 0 . 8 2 8 )$ . Although the AI Assistance × Code Size interaction was significant (see

Table IV, Model 3, $p = 0 . 0 1 6 )$ , post-hoc contrasts revealed that the AI-induced elevation was statistically equivalent at small and large sizes (8.9% and 9.5% respectively; $z = 0 . 2 5 , p =$ 0.968), and the apparent medium-size difference was driven by a control-condition dip rather than a spike in AI-induced workload. If AI were reducing intrinsic load, its overhead should shrink where code is simpler and expand where code is harder, yet the AI-induced cost remained largely constant across sizes. Future research should investigate whether this pattern holds under conditions of greater complexity and wider experience ranges than those studied here.

## A. Threats to Validity and Limitations

1) External Validity.: Several aspects may limit the general izability of our findings to the broader software engineering community. Most notable, our sample was recruited entirely from a university student population. The cognitive workload profiles of students, who are frequently still consolidating syntax and architectural structures, may not generalize to professional software engineers with established mental models. Furthermore, because these student participants held varying levels of GenAI familiarity and were still developing their interaction strategies, our results likely capture the cognitive load of relatively novice AI adoption, rather than mature, expert workflows. Moreover, the AI agent was accessed via a web interface on a secondary monitor to maintain experimental control. In professional settings, generative tools are often integrated directly into the IDE. The contextual switching overhead imposed by an external browser interface may differ from the workload associated with native integrations [3, 16].

2) Internal Validity.: A primary threat to our internal validity involves potential mechanical interference during the AIassisted reading phase, where the physical act of typing prompts could disrupt responses with the finger-mounted tDRT button. To isolate cognitive workload from these physical artifacts, all trials involving active typing were removed from our analysis. Furthermore, the highest workload measurements occurred during the subsequent quiz phases where no typing took place, confirming that mechanical interference from tool interaction does not account for the sustained workload impact.

3) Conclusion validity.: The main threat to our conclusion validity is that elevated cognitive workload is not inherently detrimental; it can sometimes reflect higher engagement that leads to better information retention. While we believe the elevated cognitive load observed in our study was due to dualtask demands, because of the ceiling effect in our performance data, we are limited in our ability to determine if the increased cognitive load would have yielded performance benefits in more complex programming tasks. This is an avenue for future work, which should utilize a greater variety of task types and difficulty to help map if this additional load benefits or harms comprehension.

## B. Implications for Practice

While our findings are grounded in a student sample working with short code snippets, they offer preliminary insights relevant to practitioners and educators integrating generative AI into software development workflows.

First, practitioners must remain cautious when tools make programming tasks simply “feel” easier. The discrepancy between our subjective perceptions and objective measurements (the 11.2% workload increase) indicates that developers may operate under a false impression of reduced effort. This aligns with the efficiency illusions observed by [7], where users predicted AI-assisted tasks to be significantly faster despite completion times remaining equivalent to unassisted work. While their study suggested that reduced subjective mental effort drives this temporal miscalibration, our findings demonstrate that even this impression of reduced effort can mask objective cognitive strain. Consequently, stakeholders evaluating AI tool efficiency should incorporate objective cognitive load and performance metrics rather than relying solely on subjective feedback, which can inadvertently mask mental fatigue.

Second, the sustained workload elevation observed when the tool was removed during the quiz phase suggests that relying on AI during initial comprehension carries a lingering cost when independent execution is required. This aligns with recent work on GenAI dependency and cognitive offloading, which indicates that AI assistance during learning can induce a false sense of competence during use, but compromise subsequent unassisted performance [49, 50]. Our workload metrics show that when this assistance is abruptly withdrawn, transitioning back to unassisted problem-solving demands greater mental effort than working independently from the start.

## C. Implications for Researchers

Our study introduces key methodological and theoretical contributions for future human-centric software engineering and human-AI interaction research.

First, our findings suggest that relying solely on subjective questionnaires is insufficient for capturing the full cognitive impact of AI tools. The contrast between our equivalent questionnaire ratings across conditions and the significant 11.2% objective workload increase provides a clear example of the workload dissociation paradigm [9]. Because users routinely adapt to increased dual-task demands without conscious awareness of their mental resource expenditure [51], self-reported instruments alone fail to capture a tool’s true cognitive impact. Future empirical software engineering studies should adopt multimodal assessment frameworks that combine subjective scales with real-time objective metrics to expose these hidden fluctuations in cognitive load. Second, to our knowledge, this study represents the first successful application of the Detection Response Task (DRT) within a software engineering context. Prior software comprehension research has relied on eye-tracking or EEG, which require complex calibration and are sensitive to noise. Our results show that the tDRT serves as an accessible and sensitive metric for capturing real-time fluctuations in developer mental load.

## VI. CONCLUSION

This study presents an empirical evaluation of how GenAI impacts software developers during code comprehension, investigating its effects on real-time cognitive workload, observable performance, and comprehension strategies.

We conducted a controlled within-subjects experiment $( N = 2 0 )$ utilizing a tactile Detection-Response Task (tDRT). Our findings challenge the assumption that GenAI inherently reduces developer effort. Instead, AI assistance significantly increased cognitive workload, raising tDRT reaction times by 11.2%, an elevation that was sustained into the unassisted testing phase. Despite this additional cognitive load, participants’ subjective difficulty ratings remained similar between conditions. Our qualitative analysis contextualizes this dissociation through three behavioral themes: Resistance, Scaffolding, and Displacement, which illustrate how distinct interaction styles and verification behaviors drive the mental overhead of human-AI collaboration.

While a performance ceiling effect limited our ability to differentiate final accuracy outcomes, our findings clearly expose the hidden mental workload accompanying GenAI use. As software engineering increasingly integrates human-AI collaboration, research must look beyond immediate task completion to design paradigms that support active mental model formation, and to investigate whether the sustained cognitive costs observed here carry implications for long-term technical expertise.

## REFERENCES

[1] X. Xia, L. Bao, D. Lo, Z. Xing, A. E. Hassan, and S. Li, “Measuring program comprehension: A large-scale field study with professionals,” IEEE Transactions on Software Engineering, vol. 44, no. 10, pp. 951–976, 2017.

[2] R. Minelli, A. Mocci, and M. Lanza, “I know what you did last summeran investigation of how developers spend their time,” in 2015 IEEE 23rd international conference on program comprehension. IEEE, 2015, pp. 25–35.

[3] D. Nam, A. Macvean, V. Hellendoorn, B. Vasilescu, and B. Myers, “Using an LLM to Help With Code Understanding,” in Proceedings of the IEEE/ACM 46th International Conference on Software Engineering, ser. ICSE ’24. New York, NY, USA: Association for Computing Machinery, Apr. 2024, pp. 1–13. [Online]. Available: https://dl.acm.org/doi/10.1145/3597503.3639187

[4] H. Mozannar, G. Bansal, A. Fourney, and E. Horvitz, “Reading Between the Lines: Modeling User Behavior and Costs in AI-Assisted Programming,” in Proceedings of the CHI Conference on Human Factors in Computing Systems. Honolulu HI USA: ACM, May 2024, pp. 1–16. [Online]. Available: https://dl.acm.org/doi/10.1145/3613904.3641936

[5] N. Tang, M. Chen, Z. Ning, A. Bansal, Y. Huang, C. McMillan, and T. J.-J. Li, “Developer behaviors in validating and repairing llm-generated code using ide and eye tracking,” in 2024 IEEE Symposium on Visual Languages and Human-Centric Computing (VL/HCC). IEEE, 2024, pp. 40–46.

[6] P. Vaithilingam, T. Zhang, and E. L. Glassman, “Expectation vs. experience: Evaluating the usability of code generation tools powered by large language models,” in Chi conference on humanfactors in computing systems extended abstracts, 2022, pp. 1–7.

[7] S. Yu, M. Cheng, A. Jabbar, I. Sucholutsky, K. M. Collins, D. Jurafsky, and R. D. Hawkins, “The efficiency-gain illusion: People underestimate the rate of ai use and overestimate its benefits on simple tasks,” arXiv preprint arXiv:2605.22687, 2026.

[8] S. G. Hart, “Nasa-task load index (nasa-tlx); 20 years later,” Proceedings of the Human Factors and Ergonomics Society Annual Meeting, vol. 50, no. 9, pp. 904–908, 2006. [Online]. Available: https://doi.org/10.1177/154193120605000909

[9] Y.-Y. Yeh and C. D. Wickens, “Dissociation of performance and subjective measures of workload,” Human factors, vol. 30, no. 1, pp. 111–120, 1988.

[10] U. Obaidellah, M. Al Haek, and P. C.-H. Cheng, “A survey on the usage of eye-tracking in computer programming,” ACM Computing Surveys (CSUR), vol. 51, no. 1, pp. 1–58, 2018.

[11] ISO 17488:2016. [Online]. Available: https://www.iso.org/standard/598 87.html

[12] M. Wyrich, J. Bogner, and S. Wagner, “40 years of designing code comprehension experiments: A systematic mapping study,” ACM Comput. Surv., vol. 56, no. 4, Nov. 2023. [Online]. Available: https://doi.org/10.1145/3626522

[13] L. Jain and S. Singh, “Designing the code snippets for experiments on code comprehension of different software constructs,” International Journal of Computer Sciences and Engineering, vol. 7, pp. 310–318, 03 2019.

[14] S. Ajami, Y. Woodbridge, and D. G. Feitelson, “Syntax, predicates, idioms - what really affects code complexity?” in 2017 IEEE/ACM 25th International Conference on Program Comprehension (ICPC), 2017, pp. 66–76.

[15] Y. Qiao, M. I. H. Shihab, and C. Hundhausen, “A systematic literature review of the use of GenAI assistants for code comprehension: Implications for computing education research and practice,” ACM Transactions on Computing Education, vol. 26, no. 2, pp. 1–33, 2026.

[16] E. A. Haque, C. Brown, T. D. LaToza, and B. Johnson, “Towards Decoding Developer Cognition in the Age of AI Assistants,” Jan. 2025, arXiv:2501.02684 [cs]. [Online]. Available: http://arxiv.org/abs/2501.026 84

[17] R. Krueger, Y. Huang, X. Liu, T. Santander, W. Weimer, and K. Leach, “Neurological divide: An fmri study of prose and code writing,” in Proceedings of the acm/ieee 42nd international conference on software engineering, 2020, pp. 678–690.

[18] J. Gao, Y. Xue, X. Xie, S. Thant, E. Lee, and B. Xu, “Understanding codebase like a professional! Human-AI collaboration for code comprehension,” in Proceedings of the 34th IEEE/ACM International Conference on Program Comprehension (ICPC), 2026, to appear. Available: https://doi.org/10.48550/arXiv.2504.04553.

[19] N. Perry, M. Srivastava, D. Kumar, and D. Boneh, “Do Users Write More Insecure Code with AI Assistants?” in Proceedings of the 2023 ACM SIGSAC Conference on Computer and Communications Security, ser. CCS ’23. New York, NY, USA: Association for Computing Machinery, Nov. 2023, pp. 2785–2799. [Online]. Available: https://dl.acm.org/doi/10.1145/3576915.3623157

[20] M. Kazemitabaar, O. Huang, S. Suh, A. Z. Henley, and T. Grossman, “Exploring the Design Space of Cognitive Engagement Techniques with AI-Generated Code for Enhanced Learning,” in Proceedings of the 30th International Conference on Intelligent User Interfaces, ser. IUI ’25. New York, NY, USA: Association for Computing Machinery, 2025, pp. 695–714. [Online]. Available: https://doi.org/10.1145/3708359.3712104

[21] T. Fritz, A. Begel, S. C. Muller, S. Yigit-Elliott, and M. Z¨ uger, “Using¨ psycho-physiological measures to assess task difficulty in software development,” in Proceedings of the 36th international conference on software engineering, 2014, pp. 402–413.

[22] K. Stojmenova and J. Sodnik, “Detection-response task—uses and limitations,” Sensors, vol. 18, no. 2, p. 594, 2018.

[23] R. J. Innes, N. J. Evans, Z. L. Howard, A. Eidels, and S. D. Brown, “A broader application of the detection response task to cognitive tasks and online environments,” Human factors, vol. 63, no. 5, pp. 896–909, 2021.

[24] M.-A. Storey, “Theories, methods and tools in program comprehension: past, present and future,” in 13th International Workshop on Program Comprehension (IWPC’05). IEEE, 2005, pp. 181–191.

[25] A. Von Mayrhauser and A. M. Vans, “Program comprehension during software maintenance and evolution,” Computer, vol. 28, no. 8, pp. 44–55, 1995.

[26] ——, “Dynamic code cognition behaviors for large scale code,” in Proceedings 1994 IEEE 3rd Workshop on Program Comprehension-WPC’94. IEEE, 1994, pp. 74–81.

[27] D. Trommler, T. Morgenstern, I. Karl, F. Naujoks, J. F. Krems, and A. Keinath, “Impact of the difficulty of the box task on its sensitivity when combined with a detection response task to assess secondary task demand while driving,” Transportation Research Part F: Traffic Psychology and Behaviour, vol. 114, pp. 888–905, 2025.

[28] P. Green and C. J. MacLeod, “Simr: An r package for power analysis of generalized linear mixed models by simulation,” Methods in Ecology

and Evolution, vol. 7, no. 4, pp. 493–498, 2016.

[29] J. Cohen, “A power primer.” Psychological bulletin, vol. 112, no. 1, p. 155, 1992.

[30] J. Cooper. Detection response task (DRT) | ISO 17488. Product page, accessed June 2026. [Online]. Available: https://redscientific.com/devic es/drt

[31] J. M. Cooper, C. L. Wheatley, M. M. McCarty, C. J. Motzkus, C. L. Lopes, G. G. Erickson, B. R. W. Baucom, W. J. Horrey, and D. L. Strayer, “Age-Related Differences in the Cognitive, Visual, and Temporal Demands of In-Vehicle Information Systems,” Frontiers in Psychology, vol. Volume 11 - 2020, 2020. [Online]. Available: https://www.frontiersi n.org/journals/psychology/articles/10.3389/fpsyg.2020.01154

[32] T. A. Ranney, G. S. Baldwin, L. A. Smith, E. N. Mazzae, and R. S. Pierce, “Detection response task (DRT) evaluation for driver distraction measurement application,” National Highway Traffic Safety Administration, Vehicle Research and Test Center, Tech. Rep. DOT HS 812 077, 11 2014, final Report. [Online]. Available: https://www.nhtsa.gov/node/7096

[33] K. Shahzad and S. Iqbal, “Comparative Analysis of ChatGPT, DeepSeek, and Gemini for Automated Code Generation,” in 2025 18th International Conference on Engineering of Modern Electric Systems (EMES), May 2025, pp. 1–4, iSSN: 2836-9866. [Online]. Available: https://ieeexplore.ieee.org/document/11045587/

[34] J. Siegmund, C. Kastner, S. Apel, C. Parnin, A. Bethmann, T. Leich,¨ G. Saake, and A. Brechmann, “Understanding understanding source code with functional magnetic resonance imaging,” in Proceedings of the 36th International Conference on Software Engineering, ser. ICSE 2014. New York, NY, USA: Association for Computing Machinery, 2014, p. 378–389. [Online]. Available: https://doi.org/10.1145/2568225.2568252

[35] Anonymous, “Replication package,” Mar. 2026, available: https://doi.or g/10.5281/zenodo.22967942.

[36] T. Busjahn, R. Bednarik, A. Begel, M. Crosby, J. H. Paterson, C. Schulte, B. Sharif, and S. Tamm, “Eye Movements in Code Reading: Relaxing the Linear Order,” in 2015 IEEE 23rd International Conference on Program Comprehension, May 2015, pp. 255–265, iSSN: 1092-8138. [Online]. Available: https://ieeexplore.ieee.org/document/7181454

[37] S. Monsell, “Task switching,” Trends in cognitive sciences, vol. 7, no. 3, pp. 134–140, 2003.

[38] J. Sauro and J. R. Lewis, “Chapter 8 - Standardized Usability Questionnaires,” in Quantifying the User Experience. Boston: Morgan Kaufmann, Jan. 2012, pp. 185–240. [Online]. Available: https: //www.sciencedirect.com/science/article/pii/B9780123849687000084

[39] D. Bates, M. Machler, B. Bolker, and S. Walker, “Fitting linear mixed-¨ effects models using lme4,” Journal of statistical software, vol. 67, pp. 1–48, 2015.

[40] H. M. Hammouri, R. T. Sabo, R. Alsaadawi, and K. A. Kheirallah, “Handling skewed data: A comparison of two popular methods,” Applied Sciences, vol. 10, no. 18, p. 6247, 2020.

[41] A. Chakrabarti and J. K. Ghosh, “AIC, BIC and recent advances in model selection,” Philosophy of statistics, pp. 583–605, 2011.

[42] R. H. B. Christensen, ordinal—Regression Models for Ordinal Data, 2025, r package version 2025.12-29. [Online]. Available: https://CRAN.R-project.org/package=ordinal

[43] Miro, Miro, 2024, collaborative whiteboarding platform. [Online]. Available: https://miro.com

[44] K. Holtzblatt, J. B. Wendell, and S. Wood, Rapid contextual design: a how-to guide to key techniques for user-centered design. Elsevier, 2004.

[45] D. R. Garrison, M. Cleveland-Innes, M. Koole, and J. Kappelman, “Revisiting methodological issues in transcript analysis: Negotiated coding and reliability,” The internet and higher education, vol. 9, no. 1, pp. 1–8, 2006.

[46] C. R. Catalan, L. M. Dizon, P. N. Monderin, and E. Kuang, ““I’m Not Reading All of That”: Understanding software engineers’ level of cognitive engagement with agentic coding assistants,” arXiv preprint arXiv:2603.14225, 2026.

[47] P. Trakoolgerntong, T. Xiao, M. Kondo, C. Ragkhitwetsagul, M. Choetkiertikul, P. Sangaroonsilp, and Y. Kamei, “Ailinkpreviewer: Enhancing code reviews with llm-powered link previews,” in 2025 32nd Asia-Pacific Software Engineering Conference (APSEC). IEEE, 2025, pp. 1021–1024.

[48] R. Parasuraman and D. H. Manzey, “Complacency and bias in human use of automation: An attentional integration,” Human factors, vol. 52, no. 3, pp. 381–410, 2010.

[49] J. M. Lodge and L. Loble, “Artificial intelligence, cognitive offloading

and implications for education,” 2026, University of Technology Sydney, doi:10.71741/4pyxmbnjaq.31302475.

[50] OECD, “OECD Digital Education Outlook 2026: Exploring Effective Uses of Generative AI in Education,” OECD Digital Education Outlook, vol. 2026, Jan. 2026. [Online]. Available: https://www.oecd.org/en/publ ications/oecd-digital-education-outlook-2026 062a7394-en.html

[51] M. A. Vidulich, “The cognitive psychology of subjective mental workload,” in Advances in psychology. Elsevier, 1988, vol. 52, pp. 219–229.