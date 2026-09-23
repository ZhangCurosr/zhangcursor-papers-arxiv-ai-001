# Does AI Save Time on Product Design? A Randomized Controlled Experiment of AI Prompt-to-Design Workflows

REMY STEWART, Figma, USA

OLABODE ANISE, Figma, USA

ANDREW HOGAN, Figma, USA

AUGUSTUS GRIFFIN, Figma, USA

AI tools for digital product design now ofer prompt-to-design capabilities, allowing designers and their non-designer colleagues to create prototypes through conversational workflows with large language models (LLMs). While these tools promise time savings, experimental evidence in product design remains limited compared with evidence from software engineering. We conducted a randomized controlled trial with 50 product designers and 50 product managers to evaluate prospective time savings from leveraging Figma Make in design work. Participants attempted three standardized design tasks with or without access to Figma Make. Among participants who completed the study tasks, access to Figma Make was associated with approximately 20% shorter completion times, with larger gains among product managers. Our findings suggest that prompt-to-design tools may enable product managers to further contribute to design work, while the benefits for professional designers may be task dependent

CCS Concepts: • Human-centered computing → Empirical studies in HCI; User studies; Systems and tools for interaction design.

Additional Key Words and Phrases: Design Software, AI Time Savings, AI Productivity, AI-Assisted Design, Randomized Controlled Trial

## ACM Reference Format:

Remy Stewart, Olabode Anise, Andrew Hogan, and Augustus Grifin. 2026. Does AI Save Time on Product Design? A Randomized Controlled Experiment of AI Prompt-to-Design Workflows. 1, 1 (September 2026), 23 pages. https://doi.org/10.1145/nnnnnnn.nnnnnnn

## 1 Introduction

A core value proposition behind why individuals should adopt AI-powered tools is to automate routine tasks and save time. The promise of increased productivity has helped drive both the development of AI tools and their adoption in the workplace. In a 2026 survey, 79% of corporate organizations reported adopting generative AI to some degree [43]. Prior research identifies employee productivity as a partial mediator between AI capabilities and organizational performance and links workforce productivity to organizational readiness and innovative practices [21, 55].

Digital product design is one of the many professions rapidly adopting AI tools, with 91% of designers reporting at least weekly usage in another 2026 survey [13]. HCI research examining why designers integrate AI into their workflows consistently identifies perceived productivity gains, defined by metrics such as tasks completed, workflows automated, and overall time saved [6, 29, 34]. Concurrently, other HCI research has identified how AI tools can jeopardize productivity, as well as how productivity alone does not engage with AI’s impacts on other dimensions such as design values, innovation, and collaboration [22, 26, 50, 53, 62]. Productivity gains can often come with tradeofs on overall design quality via dimensions such as originality and legibility [54]. Challenges with AI prompting and the stochasticity of outputs can also make tasks harder to complete [47].

Although designers often cite time savings as a reason for adopting AI tools, there has been limited work to quantify the time savings from AI adoption on everyday design tasks. Experimental research pertaining to time-saving measurement has instead focused on evaluating AI coding tools such as GitHub Copilot, Claude Code, and Cursor on software engineers’ task completion times [8, 32, 38, 40]. Randomized controlled trials (RCTs) have been a primary method within this body of research due to their particular strength towards mitigating confounders. This study asks a question previously examined in software engineering by employing the same RCT design: Does using AI save design professionals time in their everyday work?

AI tools for design are evolving rapidly alongside advances in AI research and engineering. With these advances, two notable trends have emerged: the rise of conversational or "vibe-based" prompt-to-design tools and lower barriers to participation in design work for non-designers. Prompt-to-design tools are one of the latest developments in integrating AI into design software. Earlier integrations focused on narrower tasks, such as auto-completion or image editing, or relied on external chatbots like ChatGPT. By contrast, tools such as Lovable, Figma Make, and v0 can generate fully deployable designs and their supporting code through iterative conversations with users, extending AI’s role across design and software development. These prompt-to-design tools may reduce the expertise required to contribute to design work by shifting fluency with concepts such as design systems and interaction prototyping from the individual to the supporting AI tool. This has enabled other job roles to participate further in design projects alongside product designers, such as with product managers (PMs), UX researchers, and software engineers [23, 27].

This study examines two potential benefits of prompt-to-design tools: reducing the time required for everyday design tasks and enabling non-designers to contribute to design work. We investigate these hypothesized benefits through an experiment testing Figma Make, a prompt-to-design tool integrated into the widely used Figma design platform [13]. Our research questions are as follows:

RQ1: Does using AI tooling such as Figma Make on common design work save time in minutes compared to not using AI tools?

RQ2: Are there diferences on the impact of time spent on design work with or without AI tools for designers compared to non-designers?

RQ3: How does using AI tools in design work additionally afect design quality, ease of work, and platform usability?

We engaged with these three research questions by conducting a moderated between-subjects randomized controlled trial in which participants attempted a standardized series of edits on a social media UI design. Our results demonstrated a 20% improvement in cumulative time savings among participants who completed the study tasks, as well as a 16% gain in rated design task ease and a 15% improvement in perceived Figma usability. The most sizable gains were driven by product managers over product designers, with product designers not reporting a statistically significant aggregate time savings gain on their own. Our findings indicate that product manager time savings occurred on less complex design tasks, while time savings for product designers only emerged on the most complex tested task. We discuss these findings in relation to ongoing advances in AI design tools.

Manuscript submitted to ACM

## 2 Related Work

## 2.1 Measuring AI Tooling’s Impact on Productivity

Experimental research of AI’s impact on productivity has been led by measuring time savings for programming by software engineers, as well as additional industries such as customer service and consulting [11, 12, 36, 38, 56]. The magnitude and direction of AI’s impact have been mixed. While studies such as Peng et al. found reductions of up to 55% in time to task completion with 2023’s AI tools, other work such as Becker et al. found productivity regressions by 19% when considering AI in 2025 [3, 40]. Measured efect sizes have often varied by level of experience in a particula profession. Bryjolfsson et al. found that the aggregate 15% increase in issues resolved per hour was predominantly driven by gains among newer and lower-skilled employees, while more experienced workers saw minimal gains. Similarly, Cui et al. reported a 25% increase in completed programming tasks, with larger gains among less experienced developers [4, 10].

Additional dimensions such as the study setting, employed AI tool, and experiment design have been found to substantially impact variability in measured productivity outcomes within randomized controlled trials [31]. The heterogeneity of these measurements reflects the nuances around when and how AI can enable productivity gains. A subset of studies on AI’s impact on productivity within coding further complicates the narrative of whether AI provides a net positive productivity impact. He et al.’s work on programming supported by Cursor’s AI agent found short-term task completion time improvements, but a reduction in overall code quality that slows down long-term velocity due to accumulated tech debt [17]. Afroz et al. flagged similar tradeofs between faster task completion enabled by AI that is then ofset by increased demands to verify and test AI output [1].

Similar dynamics play out in the world of AI tool adoption for design and productivity, even if scholarship has not predominantly focused on experimental approaches. Randomized experiments testing AI’s impact on design work have concentrated on alternative measures such as creativity [24], divergent thinking [58], ability to problem reframe [45], or when users decide to use or not use AI given trade ofs on latency and error rates [42]. Some scholars contest whether net productivity gains actually occur when considering how AI introduces new complexities and management requirement [39, 46], while others note that Gen AI’s impact on productivity and eficiency is most holistically understood through multiple alternative dimensions including craftsmanship and user empowerment [48, 49].

HCI scholarship building from interviews, surveys, and other methodologies provides diverse insights regarding both how designers seek out AI tools for productivity gains and the challenges they encounter when using AI that reduces work eficiency. AI tools are highlighted in early design project stages for enabling fast brainstorming and iterations, assisting designers to overcome the "blank canvas problem" and mock up many initial ideas quickly [5, 37, 52, 59, 60]. Designers testify to the advantage of delegating routine, time intensive design work to AI tools, allowing them to create valuable designs more eficiently [29, 50]. However, incorrect or poor quality output, insuficient human-centered context to create optimal designs, and cognitive overload caused by managing AI tools can reduce AI’s productivity benefits [7, 18, 51, 61]. The above heterogeneous efects of AI tooling on design productivity demonstrate how multiple concurrent dynamics influence whether AI saves designers time.

## 2.2 The Rise of Prompt-to-Design AI Tools

The innovations of AI within design have shifted from Gen AI creative content tooling such as Midjourney and Stable Difusion, to LLM-based chatbots via ChatGPT, to the latest iterations being AI agents capable of advanced cross-platform work and vibe coding interfaces through conversational workflows. Vibe coding is a particular LLM-user interaction paradigm within software engineering where users delegate code creation and edits predominantly to the AI tool, leaning instead into project innovation and creativity via AI-human co-collaboration [14, 41, 44].

Our work investigates time savings from prompt-to-design tools branching from vibe coding that designers and their collaborators are now rapidly adopting. Hwang and Kang present the concept of "vibe design" to refer to many of these AI products as a designated framework that incorporates both design and user testing agentic support [19], but our conceptualization in this research is broader. These AI tools rely on prompt-based conversations between humans and LLMs in chat panel interfaces where chains of prompts lead to iterative and exploratory design creations. These tools create both UI and UX components of digital designs, as well as supporting code to enable deployment of created designs into production environments. The product design software market is now flushed with AI applications from this ecosystem such as Lovable, Replit, v0, Google Stitch, Figma Make, Framer, Bolt, and Claude Design.

Research on prompt-to-design AI tools so far indicates the benefits these products provide for quickly exploring and iterating on designs. Li et al.’s study participants noted the collapsing of design stages into single conversational workflows through adopting these tools as a particular advantage that reduced cognitive burden and freed up bandwidth for creativity [27]. Kobiella et al.’s participants testify to these tools’ eficiency with quickly advancing initial ideas to working prototypes, additionally handling code implementation that designers may not have technical backgrounds on [23].

An additional trend within the rise of prompt-to-design products is how they contribute to blurring the lines of which job roles are participating in design work. Many of these products intentionally aim for non-designers as key expansion audiences. AI design tools speed up creating code for new design system artifacts for engineers to refine and integrate into established code bases [20]. Engineers, data scientists, and product managers at Microsoft have designed their own productivity assistants as early adopters of recent Gen AI tooling enabled at the company [33]. Non professional designers report the ability to move from passive observers to active collaborators with both designers and AI through adopting end-to-end AI tooling [30]. We are therefore interested in measuring whether AI enabled time savings occur in design work beyond just product designers to reflect the expanding scope of who contributes to design empowered by AI design products.

This study focuses on measuring time savings from Figma Make as a popular AI design tool. 68% of professional product designers surveyed in 2026 use Figma AI including Figma Make [13]. Figma Make created interactive prototypes and the code underlying generated designs with the notable advantage of producing outputs that source from designer’s preexisting design systems - the diverse components of design interfaces consistently referenced throughout an organization - which design organizations tend to already house within Figma libraries. Other Figma Gen AI enabled features such as Figma widgets have been the tools of focus in prior HCI work [15], as well as understanding how designers integrate external AI design tools with their preexisting Figma based workflows [9, 20]. This study builds from these previous investigations by studying Figma Make as one of the latest AI prompt-to-design features launched directly for the Figma platform. Figma Make is additionally targeted to beyond designers as key audiences, with product managers as a main secondary user base [57]. In sum, we aim to quantify productivity impacts for both product designers and product managers using Figma Make as defined by time to task completion through a laboratory setting randomized controlled experiment.

## 3 Methodology

## 3.1 Experiment Design

We conducted a between-subjects randomized experiment to evaluate whether incorporating Figma Make into a design workflow reduced the time required to complete UI modifications when compared to manual editing in Figma Design. First, participants completed an eligibility questionnaire that additionally collected demographic information Eligible participants were then randomly assigned to the treatment or control condition. Participants in both conditions attempted the same three design tasks using identical starting designs and reference images of the intended outputs After the design tasks, participants completed an exit survey assessing perceived Figma usability, self-reported design quality, and prior AI tool usage.

3.1.1 Recruitment. We collaborated with the external research firm MeasuringU to recruit participants and moderate the study. MeasuringU recruited 50 product designers and 50 product managers from their preexisting research panels. Within each role, we randomly assigned participants to the treatment or control condition, resulting in 25 participants from each role in each condition. The sample size was identified by referencing prior AI time savings experiments samples [38, 40], as well as forecasting minimum detectable efect size scenarios and modeling required samples for adequate statistical power. Eligibility was determined from prospective participants’ self-reported responses to the recruitment screener. Participants qualified for the study if they had at least two years of experience as a product designer or product manger, viewed or edited a Figma file at least once per month, currently worked in a role involving at least one digital product, and had never been employed by Figma. All participants signed consent forms and received a monetary honorarium.

3.1.2 Experiment Flow. To test the study’s central research questions, we designed an experiment focused on modifying and extending an existing UI design. Participants in both the treatment and control conditions attempted the same three tasks, using the same starting designs and reference images of the intended outputs. We opted to use a social media application as the shared scenario for each of the tasks because we expected participants, irrespective of industry background, to be familiar with its basic features. We sought guidance from multiple professional product designers regarding which tasks would arise when iterating on a social media design prototype. We used this input to select categories, which became the distinct tasks of the experiment scenario:

(1) Visual styling: UI changes to the base design, such as adjusting color palettes, text, or images. Task 1 asked participants to convert the starting light-mode version of the design into a dark-mode variant.

(2) Interface modification: Adding content to an existing interface element. Task 2 asked participants to add a "Help and Support" option to a dropdown settings menu

(3) Interaction prototyping: Creating interactions and animations within the existing design. Task 3 asked participants to create a comment flyout that appeared when the social media post’s ’Comments’ icon was clicked.

All participants attempted the three tasks in the same fixed order: dark-mode conversion, menu modification, and comment flyout creation. Figure 1 illustrates the workflow for each task. Each task’s starting and final design is provided in Appendix A. For each task, participants received an initial design in an editable Figma frame whose components participants could directly edit. We provided the target design as a static image rather than an editable Figma frame so that the internal components of the finished design were not readily available to participants. Participants edited their designs in a designated workspace between the starting frame and the reference image. Edits from earlier tasks did not carry over to later tasks.

![](images/ac601e1cbee0ebbe7447b2eaaff075e3d3c6aaa2de926fe9aed80a2fa2a527ae.jpg)  
Fig. 1. Experiment task flow for designing and time to completion measurement

Control participants were instructed to use the starting design frame and directly edit the design in the working space to match the reference image. Treatment participants were instead instructed to include the starting frame as an attachment alongside their initial prompt to Figma Make. They could send as many prompts to Figma Make as desired to edit the design. Treatment participants were additionally allowed to return the design modified with Figma Make to the experiment file at any point to conduct manual edits. All participants were asked to place their updated designs in the task workspace when they believed they had successfully matched the reference image of the desired changes. This allowed moderators to compare the submitted designs with the reference images using task-specific completion criteria supplied by the study team.

Participants in both conditions attempted the same tasks and were evaluated against the same completion criteria. The sole diference was that treatment participants had access to Figma Make and no other Figma AI features. This was done in order to isolate Figma Make’s efect on task completion time. Participants in the treatment group were given five minutes to review a primer on Figma Make sourced from Figma Help Center articles. Participants were permitted to use external resources to complete each study task, including oficial Figma documentation. We conducted internal pilot trials of the study with product designers, product managers, designer advocates, and researchers, and revised the experiment multiple times based on their feedback.

3.1.3 Moderation & Data Collection. All MeasuringU moderators were trained on how to administer the experiment by the internal study team, including following a standardized script and moderation documentation. Moderators walked participants through the instructions of each design task, reviewed each task’s final design to verify it matched the reference photo, and assisted participants with any troubleshooting errors when working through the scenarios.

Time to completion for each task was recorded through a time-tracking Figma widget built for the study. Moderators started the timer at the participant’s instruction and stopped the timer after verifying, according to the moderation Manuscript submitted to ACM

guidelines, that the participant’s design matched the reference image. The widget additionally sent timestamped logs to an external database, which enabled the study team to cross-check the widget’s displayed completion times with separate telemetry.

After each task, participants rated how easy it was to complete the task on a seven-point scale from 1 ("very hard") to 7 ("very easy"). After completing all tasks, participants completed an exit survey collecting data on the perceived usability of Figma Make using the System Usability Scale (SUS) [25], self-reported design quality, and prior experience with AI tools. The questions asked in the exit survey can be found in Appendix B.

## 3.2 Data Analysis

Data was collected through a combination of the experiment’s time-tracking widgets and supporting telemetry, task completion questions answered after each task, the eligibility screener, and the exit survey. We analyzed the collected data as follows.

We employed generalized estimating equations (GEE) models to measure diferences in completion rate and time to completion for each of the individual tasks as well as the cumulative time to complete all three scenarios. GEE models account for the correlations in repeated observations within participants rather than treating a given participant’s performance on each task as independent measurements. By clustering per participant and adjusting standard errors to account for task outcome interdependence, we estimated both individual task performance as well as aggregate time to completion across the full trial. GEE models particularly excel at estimating the population-level average efects of time to task completion compared to alternative generalized linear model and mixed-efects models [35].

All GEE models included covariates for participant demographics, design work background, and AI experience to account for these characteristics when estimating the treatment efect. For selected categorical covariates, we combined categories with few participants to reduce the number of coeficients estimated. Completion times were log-transformed to address their right-skewed distribution. We assessed model assumptions regarding participant independence and residual normality. All models used cluster-robust sandwich standard errors [28] and are reported with 95% confidence intervals. Average marginal efects (AMEs) and standard errors estimated by the models are reported in minutes and converted to relative percent diferences.

For categorical results regarding ease and usability scores as well as quality ratings, we employed Chi-squared testing as well as Mann–Whitney U tests as an alternative specification to test result robustness. Chi-squared results were additionally supplemented with adjusted standardized residuals to isolate the response categories contributing most to the observed diferences along with the overall Chi-squared score [16].

## 4 Results

## 4.1 Descriptive Statistics

A full review of descriptive statistics for the collected variables can be found in Appendix C. The sample was evenly split by gender. Most participants were aged 25–54 and had at least 10 years of professional experience. Participants primarily worked full time at organizations with 1–1,000 employees. The most common industries were information technology and banking and financial services. Most participants also worked with web-based applications and digital experiences. Product designers in the sample tended to have more professional experience, are older, and were more likely to be self-employed than product managers who participated in the study.

Manuscript submitted to ACM

Table 1. Completion rates, time to completion, task ease scores, and System Usability Scale (SUS) scores. Values reported as means & standard deviation.
<table><tr><td></td><td>Full Sample</td><td>Designers</td><td>PMs</td></tr><tr><td>Completion Rate</td><td></td><td></td><td></td></tr><tr><td>Task 1 (Dark Mode)</td><td>93%</td><td>98%</td><td>88%</td></tr><tr><td>Task 2 (Settings)</td><td>91%</td><td>96%</td><td>86%</td></tr><tr><td>Task 3 (Comment Modal)</td><td>73%</td><td>90%</td><td>56%</td></tr><tr><td colspan="4">Time to Completion (minutes)</td></tr><tr><td>Task 1 (Dark Mode)</td><td>11:23 (8:05)</td><td>8:27 (5:11)</td><td>14:25 (9:25)</td></tr><tr><td>Task 2 (Settings)</td><td>9:58 (6:16)</td><td>8:15 (5:54)</td><td>11:46 (6:11)</td></tr><tr><td>Task 3 (Comment Modal)</td><td>25:40 (9:13)</td><td>24:29 (9:14)</td><td>26:54 (9:08)</td></tr><tr><td>Tasks 1-3 (Cumulative)</td><td>46:33 (15:14)</td><td>41:13 (14:32)</td><td>52:12 (13:59)</td></tr><tr><td colspan="4">Task Ease Scores</td></tr><tr><td>Task 1 (Dark Mode)</td><td>4.9 (1.7)</td><td>5.7 (1.4)</td><td>4.1 (1.5)</td></tr><tr><td>Task 2 (Settings)</td><td>4.8 (1.7)</td><td>5.3 (1.7)</td><td>4.3 (1.7)</td></tr><tr><td>Task 3 (Comment Modal)</td><td>3.4 (1.8)</td><td>4.2 (1.7)</td><td>2.7 (1.5)</td></tr><tr><td>SUS Score</td><td>60.9 (2.2)</td><td>70.1 (2.8)</td><td>51.2 (2.7)</td></tr></table>

Use of AI, Figma Design, and Figma AI features difers greatly between product designers and product managers. While the majority of both groups used AI tools daily, designers typically used AI for design work daily or a few times a week, whereas product managers generally used it once a week or less. The most popular AI tools across both job roles are ChatGPT, Google Gemini, and Claude. The strong majority of designers use Figma multiple times a day and have used Figma AI before, while product managers varied in how frequently they used Figma and most had not used Figma AI before.

## 4.2 RQ1 - Time Savings from Figma Make

The average cumulative time to completion across all tasks for all participants was 46 minutes. Completion rates and completion times varied substantially across tasks. Task 2 was the fastest task at an average of 10 minutes for all participants, while Task 3 where participants created a comment flyout interaction took the longest at an average of 26 minutes. Completion rates exceeded 90% for Tasks 1 and 2, but fell to 73% for Task 3. Treatment group participants submitted on average 2.9 Figma Make prompts for Task 1, 2.6 prompts for Task 2, and 5.2 prompts for Task 3. The complete distributions of individual task times for all participants can be found in Figure 2, and Table 1 summarizes completion rates and mean completion times.

Table 2. Full completion rate Treatment vs. Control GEE deltas across individual and average tasks. Values are percentage diferences and standard errors.
<table><tr><td></td><td>Full Sample</td><td>Product Designers</td><td>Product Managers</td></tr><tr><td>Task 1 (Dark Mode)</td><td>0.8 (4.9)</td><td>−7.0 (4.9)</td><td>8.6 (8.9)</td></tr><tr><td>Task 2 (Settings)</td><td>0.8 (5.1)</td><td>−10.9 (5.8)</td><td>12.5 (8.9)</td></tr><tr><td>Task 3 (Comment Modal)</td><td>19.9** (6.8)</td><td>1.6 (7.9)</td><td> $3 8 . 1 ^ { ^ { \star \star \star } } \left( 1 1 . 4 \right)$ </td></tr><tr><td>Tasks 1–3 (average)</td><td>7.2 (4.1)</td><td>−5.4 (4.5)</td><td> $1 9 . 7 ^ { ^ { \star \star } } \left( 7 . 4 \right)$ </td></tr></table>

\* � < 0.05, \*\* � < 0.01, \*\*\* � < 0.001.

![](images/6e2e15682bf306eb49fa4ec5e1371cd011969b54a20244a9a55262ed0308ceca.jpg)

![](images/0f4c37a3966efe69b035b5a78d8ce5d0aa12eb6c86cc7801329afbe105a624ec.jpg)

![](images/90eb4cf767e5bd907f70cefd966f6439fe0b9d4cf2659feb7ec3256a8e0ccbc3.jpg)  
Fig. 2. Distributions of study participant’s time on each task

We first examined whether access to Figma Make afected participants’ likelihood of completing the tasks. In the analysis of completion probability, we did not find statistically significant completion diferences for Tasks 1 or 2, or for the average across all three tasks (Table 2). We did however find that the estimated probability of completing Task 3 was 19.9 percentage points higher (SE=6.8%, p=0.004) in the Figma Make condition than in the control condition

Table 3 outlines treatment vs control deltas in minutes for participants who successfully completed each task. Cumulatively for the entire sample, having access to Figma Make versus no access to Make saves participants approximately 15 minutes of time, equivalent to a 27% reduction in completion time (AME=15:08, SE=3.18, p<0.0001). Comparing the average of time saved across all tasks equally given the skewed role of Task 3 on cumulative time resulted in a 20% reduction in time to completion (AME=10:53, SE=3:25, p=0.0014). Within the individual tasks, Tasks 2 and 3 both reported time savings improvements at 27% (AME=2:52, SE=0:56, p=0.002) and 32% (AME=11:21, SE=2:35, p<0.0001) relative time to task completion reductions respectively. Task 1 did not report a statistically significant efect in task time, but it is directionally reported at a 4.5% relative time reduction.

We also examined associations between participant characteristics, task completion, and completion time. We have three key terms in our models - the independent variable of cumulative time to completion across the tasks, the primary dependent variable of treatment vs control group assignment, and the secondary dependent variable comparing product managers to product designers. All other controlled variable coeficients, standard errors, and significance test results can be referenced in Appendix D.

Most demographic and AI experience variables report no statistical significance across both the completion rates and time to completion models. The primary exception to this trend is regarding users who never use AI tools in their work,

Manuscript submitted to ACM

which is only 2% of the sample. Compared with participants who used AI tools for work a few times a week, these participants had an estimated completion rate 30.3 percentage points lower and an estimated cumulative completion time 35 minutes longer (Appendix D). Other lower signal coeficient results are that self employed and members of 10,000+ sized organizations had lower completion rates, that participants in communications, marketing, and media industries had higher completion rates, and that participants in information technology completed study tasks faster.

Table 4 presents estimates from models with and without a treatment-by-role interaction. Within the baseline non-interacted GEE model, we observe a 20% time reduction impact (AME=10:30, SE=3:18, p=0.0014) when removing the influence of job role into its own term at a 10 minutes and 30 seconds reduction. This estimate was similar in magnitude to the averaged time to completion in the fully saturated GEE model outlined in Table 3. Said replication across both approaches corroborates an approximately 20 percentage magnitude as the efect size for Figma Make’s impact on time to task completion for the collective population of product designers and product manager within this study.

## 4.3 RQ2 - Designers versus Non-Designers Time Savings

Completion rates across tasks sizably vary between product designers compared to product managers, with completion per task consistently above 90% for designers, but between 86-88% for product managers on Tasks 1 and 2 and dropping to 56% on Task 3. The average cumulative time to completion across all tasks for product designers is 41 minutes, while product managers reported an average cumulative time of 52 minutes. The only significant diferences in completion rates across the tasks comes from product managers and is primarily driven by the results of Task 3. Being in the treatment group raised PM completion of Task 3 by 38% (SE=11.4%, p=0.0008), producing a 20% average completion rate lift for this group (SE=7.4%, p=0.007).

The estimated diferences in completion time per task and cumulatively were largest among product managers. Being in the treatment group saved cumulatively 22 minutes and 46 seconds for PMs within the study, equating to a 35% reduction in full experiment time to completion $( \mathrm { S E } { = } 5 { : } 1 7 , \mathrm { p } { < } 0 . 0 0 0 1 )$ . Said gains were predominantly driven by Task 3 $( \mathrm { A M E } { = } 1 5 { : } 0 0 , \mathrm { S E } { = } 4 { : } 3 3 , \mathrm { p } { = } 0 . 0 0 1 )$ and also supported by Task $2 \left( \mathrm { A M E { = } 4 { : } 3 4 , S E { = } 1 { : } 3 5 , p { = } 0 . 0 0 4 } \right)$ . For product designers, improvements were only observed for Task 3 with a 7 minute and 41 second time reduction equivalent to a 26% improvement (SE=2:47, p=0.006). These gains drove the overall time reduction across the tested workflow for product designers by a marginally significant 17% (AME=7:30, SE=3:49, p=0.049).

In the model without interaction terms (Table 4), we observed a statistically significant 17 minute and 17 second coeficient for product managers compared to product designers as the reference group (SE=3:37, p<0.0001). This

Table 3. Time-to-completion Treatment vs. Control GEE minute deltas across individual and cumulative tasks. Values are population averages and standard errors.
<table><tr><td></td><td>Full Sample</td><td>Product Designers</td><td>Product Managers</td></tr><tr><td>Task 1 (Dark Mode)</td><td>0:56 (1:01)</td><td>-1:21 (1:06)</td><td> $3 { : } 1 3 \left( 1 { : } 4 3 \right)$ </td></tr><tr><td>Task 2 (Settings)</td><td> $2 { : } 5 2 ^ { ^ { \star \star } } \left( 0 { : } 5 6 \right)$ </td><td> $1 { : } 1 0 \ ( 1 { : } 0 0 )$ </td><td> $4 { : } 3 4 ^ { ^ { \star \star } } \left( 1 { : } 3 5 \right)$ </td></tr><tr><td>Task 3 (Comment Modal)</td><td>11:21*** (2:35)</td><td> ${ 7 } { : 4 1 } ^ { { \star \star } } \left( 2 { : 4 7 } \right)$ </td><td> ${ 1 5 } { : 0 0 } ^ { { \star } { \star } { \star } } \left( 4 { : 3 3 } \right)$ </td></tr><tr><td>Tasks 1-3 (cumulative)</td><td> $1 5 { : } 0 8 ^ { \ ^ { \star \star \star } } \ ( 3 { : } 1 1 )$ </td><td> $7 { : 3 0 } ^ { \star } ~ ( 3 { : 4 9 } )$ </td><td> $2 2 { : } 4 6 ^ { ^ { \star \star \star } } \left( 5 { : } 1 7 \right)$ </td></tr></table>

\* � < 0.05, \*\* � < 0.01, \*\*\* � < 0.001.

Manuscript submitted to ACM

Does AI Save Time on Product Design? A Randomized Controlled Experiment of AI Prompt-to-Design Workflows 11

Table 4. GEE key contrasts on cumulative completion time. Values are population averages and standard errors in minutes.
<table><tr><td></td><td>Baseline</td><td>Interaction</td></tr><tr><td>Treatment</td><td> $- 1 0 { : } 3 0 ^ { ^ { \star \star } } \left( 3 { : } 1 8 \right)$ </td><td>-3:03 (3:45)</td></tr><tr><td>Job Role (Reference: Product Designer)</td><td></td><td></td></tr><tr><td>Product Manager</td><td> $1 7 { : } 1 7 ^ { \star \star \star } \left( 3 { : } 3 7 \right)$ </td><td> $2 6 { : 3 0 } ^ { { \star \star \star } } \left( 5 { : 4 9 } \right)$ </td></tr><tr><td>Treatment × Job Role</td><td></td><td></td></tr><tr><td>Treatment × Product Designer</td><td></td><td>-3:03 (3:45)</td></tr><tr><td>Treatment × Product Manager</td><td></td><td> $- 2 1 { : 5 8 } ^ { { \star \star \star } } \left( 5 { : 4 7 } \right)$ </td></tr></table>

\* � < 0.05, \*\* � < 0.01, \*\*\* � < 0.001.

translates to relatively 39% more minutes in cumulative task completion in the study solely from being a product manager compared to product designers.

Interacting treatment assignment with job roles enables further exploration for how Make access contributed to time to completion at diferent magnitudes for product designers and product managers. The impact for product managers on being assigned to the treatment group in task time to completion is statistically significant at a 21 minute and 58 second reduction, equating to a 33% relative time reduction when compared to product managers in the control group (SE=5:47, p=0.0001). This implies that having Make access almost makes up for product manager’s comparative time to completion disadvantage in tested design work compared to product managers without Make access.

Reductions in time to completion is also observed for treatment group assigned product designers compared to control group product managers, but this term is not statistically significant. This observationally implies an 8% reduction in time to completion for treatment group product designers, but the lack of statistical significance denotes that we cannot conclusively say that this result is not due to random chance alone.

## 4.4 RQ3 - Design Quality, Task Ease, and System Usability

To address RQ3, we examined task-ease ratings collected after each task and exit-survey responses on perceived usability and design quality. Ease scores were collected on a 1 to 7 Likert scale where 1 rated the task as “very dificult” and 7 rated that task as “very easy”. Task ease scores closely matched the observed trends in time to completion. Across both conditions, mean task-ease ratings were similar for Task 1 (M=4.91) and Task 2 (M=4.83), while Task 3 received lower ratings (M=3.44). Product managers reported lower mean ratings than product designers on all three tasks, with diferences of 1.50, 0.98, and 1.36 points, respectively.

The mean System Usability Scale (SUS) score was approximately 61 on a 0–100 scale. This score fell between the means associated with “OK” (50.9) and “Good” (71.4) in Bangor et al.’s adjective-rating SUS study [2]. As with task-ease ratings, mean SUS scores were higher among product designers (70.1) than product managers (51.2)

In the exit survey, participants were asked questions regarding their perceived quality of their design work on all tasks, as well as whether Figma Make successfully assisted treatment group participants. Participants primarily rated their designs as somewhat matching the instructions for all experiment tasks, and they believed that the quality of their designs was either about the same or somewhat worse than what they would create outside of the experiment Among treatment participants, 78% (39/50) somewhat or strongly agreed that Figma Make would increase their ability to contribute to the design process. When asked whether they trusted Figma Make to interpret their intentions correctly, 42% (21/50) somewhat agreed, while 32% (16/50) neither agreed nor disagreed.

Table 5. Ease & System Usability Scale (SUS) scores hypothesis test results. Values are average score deltas and standard errors.
<table><tr><td></td><td></td><td>Full Sample Product Designers Product Managers</td><td></td></tr><tr><td>Task Ease</td><td></td><td></td><td></td></tr><tr><td>Task 1 (Dark Mode)</td><td>0.18 (0.66)</td><td>-0.65 (0.78)</td><td> $0 . 9 9 ^ { ^ { \star } } \left( 0 . 8 4 \right)$ </td></tr><tr><td>Task 2 (Settings)</td><td> $1 . 1 8 ^ { ^ { \star \star \star } } \left( 0 . 6 5 \right)$ </td><td>0.57 (0.94)</td><td> $1 . 7 7 ^ { ^ { \star \star \star } } \left( 0 . 8 1 \right)$ </td></tr><tr><td>Task 3 (Comment Modal) 0.92** (0.69)</td><td></td><td>0.31 (0.98)</td><td>1.50*** (0.79)</td></tr><tr><td>Tasks  $1 + 2 + 3$ </td><td> $0 . 7 6 ^ { ^ { \star \star } } \left( 0 . 5 7 \right)$ </td><td>0.07 (0.77)</td><td> $1 . 4 2 ^ { ^ { \star \star \star } } \left( 0 . 6 1 \right)$ </td></tr><tr><td>SUS Score</td><td></td><td></td><td></td></tr><tr><td>SUS Score</td><td> $1 0 . 0 \ ^ { ^ { \star } } \left( 8 . 4 0 \right)$ </td><td>2.94 (11.4)</td><td> $1 6 . 6 \AA ^ { \star \star } \left( 1 0 . 0 \right)$ </td></tr></table>

Note. \* � < 0.05, \*\* � < 0.01, \*\*\* � < 0.001.

Table 5 reports between-condition diferences in task-ease and System Usability Scale (SUS) scores. Task ease reported statistically significant improvements on two out of the three of the tasks, leading to a 0.8 point average improvemen on the 1-7 task ease scale (SE=0.29, p=0.009). SUS scores improved by a significant 10 points for treatment group participants on the score’s 0-100 scale (SE=4.23, p=0.02). These gains are equivalent to a 16% relative improvement in ease and 15% improvement in system usability enabled by Make access on the experiment tasks across all participants.

Gains in ease and system usability ratings for participants are driven predominantly by product managers. All three tested tasks demonstrated significant improvements in ease for PMs with Make access than without, leading to an average relative percent improvement of 37% (delta=+1.30 rated score, SE=0.32, p=0.0002). System usability via the SUS score increased by a relative 32% for product managers (delta=+14.8 rated score, SE=5.18, p=0.006), which places the expected score for treatment group PMs at approximately the SUS scale average score. Neither task ease nor system usability reported any statistically significant changes for product designers.

Finally, we compared responses to the two design-related exit-survey questions between the treatment and control conditions using chi-squared tests (Table 6). For both questions of how closely participants believed their designs matched the scenario instructions $( \chi ^ { 2 } ( 3 ) = 8 . 0 5 , \mathrm { p } \mathrm { = } 0 . 0 4 5 , \mathrm { V } \mathrm { = } 0 . 2 8 )$ , as well as how they would rate the quality of their experiment designs versus what they could have made outside of the study $( \chi ^ { 2 } ( 4 ) = 1 5 . 4 6 , \mathrm { p } { = } 0 . 0 0 4 , \mathrm { V } { = } 0 . 3 9 )$ , there is a statistically significant diference for treatment group participants compared to control. For the question of whether their design matched the scenario instructions, the control group’s greater likelihood of “slightly matched” (22% vs 4%)

Table 6. Quality ratings chi-squared test results.
<table><tr><td></td><td>Full Sample</td><td>Product Designers</td><td>Product Managers</td></tr><tr><td>“How closely do you believe your designs in this study matched the instructions</td><td> $8 . 1 ^ { ^ { \star } }$  &quot;Slightly matched&quot; for</td><td>1.2</td><td> $7 . 7 ^ { ^ { \star } }$  &quot;Slightly matched&quot; for</td></tr><tr><td>that were provided in each scenario?&quot;</td><td>control,“Somewhat matched” for treatment</td><td></td><td>control</td></tr><tr><td>“How would you rate the quality of your created designs during this study com-</td><td>15.5** “Much better&quot; for treat-</td><td>4.0</td><td> $1 6 . 0 \AA ^ { \star \star }$ </td></tr><tr><td>pared to what you could have made in Figma outside of this study?&quot;</td><td>ment, “Much worse&quot; for control</td><td></td><td>“Much better&quot; for treat- ment, “Much worse” for control</td></tr></table>

Note. \* � < 0.05, \*\* � < 0.01, \*\*\* � < 0.001.

compared to the treatment group’s greater tendency towards “somewhat matched” (66% vs 46%) drives the chi-squared results. For the question of the quality of their designs within the study compared to what participants could have made outside of the experiment, a higher proportion of “much better” answers for treatment participants (20% vs 2%) and a higher proportion of “much worse” for control participants (14% vs 2%) is what primarily contributes to the chi-square term.

These findings are once again statistically significant for product managers, but they are not significant for product designers when considering job role breakouts. These results remain the same with post hoc correction tests. When considering an ordinal Mann-Whitney U test specification, we find the question of whether participants’ designs matches the scenario instructions to no longer be significant in aggregate, but remaining significant for product managers specifically. In contrast, the result trends remain the same with ordinal testing regarding the quality of designs comparing within versus outside the experiment.

## 5 Discussion

## 5.1 Synthesis of Results

Estimated completion times for successfully completed tasks were approximately 20% lower in the Figma Make condition than in the control condition when averaging across the three tasks. The estimates account for diferences in demographic characteristics, professional experience, and AI tool familiarity. Participants in the Figma Make condition also reported higher task ease and perceived usability scores. These findings suggest that access to Figma Make may support increased productivity and ease of use in structured, reference-based design tasks. Whether these benefits extend to broader design work remains uncertain.

For product managers, access to Figma Make was associated with shorter completion times and higher task ease and perceived usability scores. Product managers in the treatment condition also rated their designs more favorably relative to what they thought they could produce outside the study. These findings suggest productivity benefits beyond time savings, encompassing perceived ease of use and self-assessed design quality. Moreover, prompt-to-design tools may help product managers translate their design intentions and ideas into interface changes. The gains observed among product managers, who generally used Figma less frequently in our sample, also are consistent with prior findings that AI assistance can benefit users with less experience or lower baseline performance [4, 11, 36].

Comparatively, the results for product designers were less definitive. We did not find a statistically significant reduction in cumulative completion time among designers with access to Figma Make. However, designers in the treatment condition who completed Task 3, which required the creation of a comment flyout and was therefore the most complex and time intensive task, did so more quickly. This result supports the notion that prompt-to-design tools may ofer benefits and improvements for more involved, time-intensive tasks within established design workflows, even when there isn’t an overall time-savings.

The broader implication of these findings is therefore that prompt-to-design tools may reduce the time and efort required for specific UI design tasks, supporting designers’ established workflows while lowering barriers to participation for non-designers.

## 5.2 Limitations

There are five central limitations to this study. First, our time-to-completion analysis was limited to successfully completed tasks. Because Figma Make also afected the likelihood that a participant would complete a given task, particularly for product managers on Task 3, the subset of participants who completed each task may difer between the treatment and control conditions. Therefore, the reported time reduction should be interpreted alongside the completion results.

Second, the experiment used three standardized tasks that participants attempted in a fixed order, each with a specified target design. While this structure allowed consistent measurement across each condition, it does not capture the ambiguity, need to encompass stakeholder feedback, and changing requirements that are common in professional design projects. Therefore, the observed diferences in completion rates and completion times with Figma Make may not extend to diferent types of design tasks or tasks conducted in a diferent order.

Third, task completion was verified by moderators by comparing participants’ submitted designs with the reference images. Although moderators followed the same guidelines, human variability in moderation may still impact the final results.

Fourth, participants encountered stochastic rendering failures and occasionally needed additional prompts before Figma Make produced an editable design. These failures resulted in increased completion times among treatment participants. Because this behavior existed within the version of Figma Make available during the study, it was included in all measurements. As a result, our reported diferences in time to completion between treated and non-treated participants may underestimate the true efect of prompt-to-design tools.

Finally, the experiment population consisted of 100 participants who viewed or edited Figma files at least monthly and had at least two years of experience as either a product designer or a product manager. These findings therefore do not necessarily generalize to newer Figma users, other design roles, or users of other prompt-to-design systems. Furthermore, product managers were the only non-design role included; thus, it is unclear if these results would extend to a broader non-designer population.

## 5.3 Future Directions

While this study provides an initial causal time savings measurement provided by AI design tools within an RCT framework, there are several diferent areas that researchers can explore in future work. Future experiments should consider including larger and more varied sets of design tasks as well as evaluating open-ended assignments without a predetermined output sequence. Longitudinal field studies could examine whether initial gains persist as users develop and refine prompting strategies and whether time saved during initial production is ofset by stakeholder review or later iterations.

Future studies should also incorporate expert assessments of design quality with predefined rubrics. These rubrics should include dimensions such as visual quality, accessibility, and design-system or brand alignment. This would provide evidence on whether creators’ assessments of design quality are supported by independent expert evaluations and whether increased productivity comes at the expense of design quality. Researchers could also extend this work to other prompt-to-design tools, product versions, and professional populations. Testing diferent tools and product versions would help establish the boundaries of the observed efects, while testing diferent professional populations would provide insight into how those efects vary across design-adjacent roles.

## 6 Acknowledgments

## Acknowledgments

We would like to extend our gratitude to Caitlin Wang, John Doherty, Minami Rojas, Clancy Slack, Julia Kirkpatrick, Shane Johnston, Prasant Loukendi, Mallory Dean, Lauren Byrne, Wayne Lin, Taryn Cowart, Jackie Chui, Hauke Sandhaus, Tammy Tassembaum, Jiayan Yu, Rodrigo Davies, Anna Astrom, Alia Fite, Madeline Staford, Emma Webster, Alex Praeger, and the team at MeasuringU for helping to make this study possible.

## References

[1] Sadia Afroz, Zixuan Feng, Tyler Menezes, Katie Kimura, Bianca Trinkenreich, Igor Steinmacher, and Anita Sarma. 2026. The Fast and Spurious: Developer Productivity with GenAI. arXiv:2510.24265 [cs.SE] https://arxiv.org/abs/2510.24265

[2] Aaron Bangor, Philip Kortum, and James Miller. 2009. Determining What Individual SUS Scores Mean: Adding an Adjective Rating Scale. Journal of Usability Studies 4, 3 (May 2009), 114–123.

[3] Joel Becker, Nate Rush, Elizabeth Barnes, and David Rein. 2025. Measuring the Impact of Early-2025 AI on Experienced Open-Source Developer Productivity. arXiv:2507.09089 [cs.AI] https://arxiv.org/abs/2507.09089

[4] Erik Brynjolfsson, Danielle Li, and Lindsey Raymond. 2025. Generative AI at Work. The Quarterly Journal ofEconomics (2025), 889–942. doi:10.1093 qje/qjae044

[5] Elena Cavallin and Simone Spagnol. 2026. When Designers Sweat: Behavioral Traces of GenAI Co-Creation. In Proceedings of the 2026 CHI Conference on Human Factors in Computing Systems (Barcelona, Spain) (CHI ’26). Association for Computing Machinery, New York, NY, USA. https://hdl.handle.net/11578/377009

[6] Inha Cha, Catherine Wieczorek, and Richmond Y. Wong. 2026. The Values of Value in AI Adoption: Rethinking Eficiency in UX Designers’ Workplaces. In Proceedings ofthe 2026 CHI Conference on Human Factors in Computing Systems (CHI ’26). Association for Computing Machinery, New York, NY, USA, Article 697, 17 pages. doi:10.1145/3772318.3790429

[7] Inha Cha and Richmond Y. Wong. 2025. Understanding Socio-technical Factors Configuring AI Non-Use in UX Work Practices. In Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems (CHI ’25). Association for Computing Machinery, New York, NY, USA, Article 1110, 17 pages. doi:10.1145/3706598.3713140

[8] Valerie Chen, Ameet Talwalkar, Robert Brennan, and Graham Neubig. 2026. Code with Me or for Me? How Increasing AI Automation Transforms Developer Workflows. In Proceedings of the 2026 CHI Conference on Human Factors in Computing Systems (Barcelona, Spain) (CHI ’26). Association for Computing Machinery, New York, NY, USA, 19 pages. doi:10.1145/3772318.3790850

[9] Xiang ’Anthony Chen, Tifany Knearem, and Yang Li. 2025. The GenUI Study: Exploring the Design of Generative UI Tools to Support UX Practitioners and Beyond. In Proceedings of the 2025 ACM Designing Interactive Systems Conference (DIS ’25). Association for Computing Machinery, New York, NY, USA, 1179–1196. doi:10.1145/3715336.3735780

[10] Zheyuan Cui, Mert Demirer, Sonia Jafe, Leon Musolf, Sida Peng, and Tobias Salz. 2025. The Efects of Generative AI on High-Skilled Work: Evidence from Three Field Experiments with Software Developers. SSRN Electronic Journal (20 Aug. 2025). doi:10.2139/ssrn.4945566

[11] Fabrizio Dell’Acqua and et al. 2026. Navigating the Jagged Technological Frontier: Field Experimental Evidence of the Efects of AI on Knowledge Worker Productivity and Quality. Organization Science 0, 0 (2026), 1–21. doi:10.1287/orsc.2025.21838

[12] Mert Demirer, Leon Musolf, and Liyuan Yang. 2026. Writing Code vs. Shipping Code: Productivity Efects Across Generations ofAI Coding Tools Working Paper 35275. National Bureau of Economic Research. doi:10.3386/w35275

[13] Designer Fund and Foundation Capital. 2026. AI in Design Report 2026. Technical Report. Designer Fund and Foundation Capital. https: //stateofaidesign.com/ Accessed: 2026-06-22

[14] Ahmed Fawzy, Amjed Tahir, and Kelly Blincoe. 2026. Vibe Coding in Practice: Motivations, Challenges, and a Future Outlook – A Grey Literature Review. In 2026 IEEE/ACM 48th International Conference on Software Engineering: Software Engineering in Practice (Rio de Janeiro, Brazil) (ICSE-SEIP ’26). Association for Computing Machinery, New York, NY, USA, 12 pages. doi:10.1145/3786583.3786866

[15] K. J. Kevin Feng, Q. Vera Liao, Ziang Xiao, Jennifer Wortman Vaughan, Amy X. Zhang, and David W. McDonald. 2025. Canvil: Designerly Adaptation for LLM-Powered User Experiences. In Proceedings ofthe 2025 CHI Conference on Human Factors in Computing Systems. Association for Computing Machinery, New York, NY, USA, Article 932, 22 pages. doi:10.1145/3706598.3713139

[16] Priscilla E. Greenwood and Mikhail S. Nikulin. 1996. A Guide to Chi-Squared Testing. John Wiley & Sons, New York.

[17] Hao He, Courtney Miller, Shyam Agarwal, Christian Kästner, and Bogdan Vasilescu. 2026. Speed at the Cost of Quality: How Cursor AI Increases Short-Term Velocity and Long-Term Complexity in Open-Source Projects. arXiv:2511.04427 [cs.SE] doi:10.1145/3793302.3793349

[18] Matthew K. Hong, Shabnam Hakimi, Yan-Ying Chen, Heishiro Toyoda, Charlene Wu, and Matt Klenk. 2023. Generative AI for Product Design: Getting the Right Design and the Design Right. arXiv:2306.01217 [cs.HC] https://arxiv.org/abs/2306.01217

[19] Jisun Hwang and Juyoung Kang. 2026. Vibe Design: Human-in-the-loop AI Agents for UI Design with Large Language Models. Proceedings of the 59th Hawaii International Conference on System Sciences (2026), 4453–4462. doi:10.24251/HICSS.2026.530

Manuscript submitted to ACM

[20] Diah A. Irawati, Elif Bolukbasi, and Andreas Riener. 2025. Advancing Generative AI Collaboration in Design-to-Code Workflows: Insights from Two Empirical Studies. In Proceedings ofthe 24th International Conference on Mobile and Ubiquitous Multimedia (MUM ’25). Association for Computing Machinery, New York, NY, USA, 33–46. doi:10.1145/3771882.3771913

[21] Belayneh Yitayew Kassa and Eyob Ketema Worku. 2025. The impact of artificial intelligence on organizational performance: The mediating role of employee productivity. Journal ofOpen Innovation: Technology, Market, and Complexity 11, 1 (2025), 1–17. doi:10.1016/j.joitmc.2025.100474

[22] Abidullah Khan, Atefeh Shokrizadeh, and Jinghui Cheng. 2025. Beyond Automation: How Designers Perceive AI as a Creative Partner in the Divergent Thinking Stages of UI/UX Design. In Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems (CHI ’25) (2025), 1–12. doi:10.1145/3706598.3713500

[23] Charlotte Kobiella, Daniela Breidenstein, and Albrecht Schmidt. 2026. From Throw-Away to Takeaway: How GenAI and Vibe Coding Accelerate Prototyping Across Technical Skill Levels. In Proceedings of the 2026 CHI Conference on Human Factors in Computing Systems (CHI ’26) (2026), 1–23 doi:10.1145/3772318.3790757

[24] Harsh Kumar, Jonathan Vincentius, Ewan Jordan, and Ashton Anderson. 2025. Human Creativity in the Age of LLMs: Randomized Experiments on Divergent and Convergent Thinking. In Proceedings ofthe 2025 CHI Conference on Human Factors in Computing Systems (CHI ’25). Association for Computing Machinery, New York, NY, USA, Article 23, 18 pages. doi:10.1145/3706598.3714198

[25] James R. Lewis. 2018. The System Usability Scale: Past, Present, and Future. International Journal of Human–Computer Interaction 34, 7 (2018), 577–590. doi:10.1080/10447318.2018.1455307

[26] Jie Li, Hancheng Cao, Laura Lin, Youyang Hou, Ruihao Zhu, and Abdallah El Ali. 2024. User Experience Design Professionals’ Perceptions ofGenerativ Artificial Intelligence. Proceedings ofthe CHI Conference on Human Factors in Computing Systems (CHI ’24), (2024), 1–18. doi:10.1145/3613904.3642114

[27] Jie Li, Youyang Hou, Laura Lin, Ruihao Zhu, Hancheng Cao, and Abdallah El Ali. 2026. Vibe Coding in Product Teams: Reconfiguring AI-Assisted Workflows, Prototyping, and Collaboration. In Proceedings ofthe 5th Annual Symposium on Human-Computer Interaction for Work (CHIWORK ’26). Association for Computing Machinery, New York, NY, USA, Article 1, 16 pages. doi:10.1145/3808045.3808062

[28] Kung-Yee Liang and Scott L. Zeger. 1986. Longitudinal data analysis using generalized linear models. Biometrika 73, 1 (1986), 13–22. doi:10.1093/ biomet/73.1.13

[29] Yi Luo. 2025. Designing With AI: A Systematic Literature Review on the Use, Development, and Perception of AI-Enabled UX Design Tools. Advances in Human-Computer Interaction 2025, 1 (2025), 3869207. doi:10.1155/ahci/386920

[30] Lin Ma, Jing Chen, Xinggang Hou, Yidan Qiao, Xinwei Gao, Yuan Feng, and Dengkai Chen. 2026. Rethinking Design Roles: A Study on Designer and User Perceptions During Human-AI Co-Creation. Universal Access in the Information Society 25 (2026), 56. doi:10.1007/s10209-026-01326-7

[31] Sebastian Maier, Moritz Gunzenhäuser, Jonas Schweisthal, Manuel Schneider, and Stefan Feuerriegel. 2026. A meta-analysis of the efect of generative AI on productivity and learning in programming. arXiv:2605.04779 [cs.SE] https://arxiv.org/abs/2605.04779

[32] Amr Mohamed, Maram Assi, and Mariam Guizani. 2026. The Impact of LLM-Assistants on Software Developer Productivity: A Systematic Review and Mapping Study. ACM Trans. Softw. Eng. Methodol. (April 2026). doi:10.1145/3809494

[33] Suchismita Naik, Austin L. Toombs, Amanda Snellinger, Scott Saponas, and Amanda K Hall. 2025. Designing with Multi-Agent Generative AI: Insights from Industry Early Adopters. In Proceedings of the 2025 ACM Designing Interactive Systems Conference (DIS ’25). Association for Computing Machinery, New York, NY, USA, 1961–1972. doi:10.1145/3715336.3735823

[34] Syeda Masooma Naqvi, Ruichen He, and Harmanpreet Kaur. 2025. Catalyst for Creativity or a Hollow Trend?: A Cross-Level Perspective on The Role of Generative AI in Design. In Proceedings ofthe 2025 CHI Conference on Human Factors in Computing Systems (CHI ’25). Association for Computing Machinery, New York, NY, USA, Article 194, 16 pages. doi:10.1145/3706598.3713233

[35] John M. Neuhaus, John D. Kalbfleisch, and Walter W. Hauck. 1991. A comparison of cluster-specific and population-averaged approaches for analyzing correlated binary data. International Statistical Review 59, 1 (1991), 25–35. doi:10.2307/1403572

[36] Shakked Noy and Whitney Zhang. 2023. Experimental Evidence on the Productivity Efects of Generative Artificial Intelligence. Science 381, 6654 (2023), 187–192. doi:10.1126/science.adh2586

[37] Srishti Palani and Gonzalo Ramos. 2024. Evolving Roles and Workflows of Creative Practitioners in the Age of Generative AI. In Proceedings ofthe 16th Conference on Creativity & Cognition (Chicago, IL, USA). Association for Computing Machinery, New York, NY, USA, 170–184. doi:10.1145/ 3635636.3656190

[38] Elise Paradis, Kate Grey, Quinn Madison, Daye Nam, Andrew Macvean, Vahid Meimand, Nan Zhang, Ben Ferrari-Church, and Satish Chandra. 2025. How Much Does AI Impact Development Speed? An Enterprise-Based Randomized Controlled Trial. 2025 IEEE/ACM 47th International Conference on Software Engineering: Software Engineering in Practice (ICSE-SEIP) (2025), 618–629. doi:10.1109/ICSE-SEIP66354.2025.00060

[39] Paul C Parsons. 2026. Myths and Ironies of AI-Assisted Design. In Designing Interactive Systems Conference (DIS ’26) (DIS ’26). Association for Computing Machinery, New York, NY, USA, 13 pages. doi:10.1145/3800645.3813086

[40] Sida Peng, Eirini Kalliamvakou, Peter Cihon, and Mert Demirer. 2023. The Impact of AI on Developer Productivity: Evidence from GitHub Copilot arXiv (2023), 1–19. doi:arXiv:2302.06590

[41] Veronica Pimenova, Sarah Fakhoury, Christian Bird, Margaret-Anne Storey, and Madeline Endres. 2025. Good Vibrations? A Qualitative Study of Co-Creation, Communication, Flow, and Trust in Vibe Coding. arXiv:2509.12491 [cs.SE] https://arxiv.org/abs/2509.12491

[42] Han Qiao, Jo Vermeulen, George Fitzmaurice, and Justin Matejka. 2025. To Use or Not to Use: Impatience and Overreliance When Using Generative AI Productivity Support Tools. In Proceedings ofthe 2025 CHI Conference on Human Factors in Computing Systems (CHI ’25). Association for Computing Machinery, New York, NY, USA, Article 1122, 18 pages. doi:10.1145/3706598.3714103

[43] Sha Sajadieh, Loredana Fattorini, Raymond Perrault, Yolanda Gil, Vanessa Parli, Lapo Santarlasci, Juan Pava, Nestor Maslej, Russ Altman, Erik Brynjolfsson, Carla Brodley, Jack Clark, Virginia Dignum, Vipin Kumar, James Landay, Terah Lyons, James Manyika, Juan Carlos Niebles, Yoav Shoham, Elham Tabassi, Russell Wald, Toby Walsh, and Dan Weld. 2026. The AI Index 2026 Annual Report. Annual Report. AI Index Steering Committee, Institute for Human-Centered AI, Stanford University, Stanford, CA.

[44] Advait Sarkar and Ian Drosos. 2025. Vibe coding: programming through conversation with artificial intelligence. In Proceedings ofthe 36th Annual Conference ofthe Psychology ofProgramming Interest Group (PPIG 2025). 32 pages.

[45] Joongi Shin, Anna Polyanskaya, Andrés Lucero, and Antti Oulasvirta. 2025. No Evidence for LLMs Being Useful in Problem Reframing. In Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems (CHI ’25). Association for Computing Machinery, New York, NY, USA, Article 243, 25 pages. doi:10.1145/3706598.3713273

[46] Auste Simkute, Lev Tankelevitch, Viktor Kewenig, Ava Elizabeth Scott, Abigail Sellen, and Sean Rintel. 2025. Ironies of Generative AI: Understanding and Mitigating Productivity Loss in Human-AI Interaction. International Journal ofHuman–Computer Interaction 41, 5 (2025), 2898–2919. doi:10. 1080/10447318.2024.2405782

[47] Hari Subramonyam, Divy Thakkar, Andrew Ku, Juergen Dieber, and Anoop K. Sinha. 2025. Prototyping with Prompts: Emerging Approaches and Challenges in Generative AI Design for Collaborative Software Teams. In Proceedings ofthe 2025 CHI Conference on Human Factors in Computing Systems. Association for Computing Machinery, New York, NY, USA, Article 882, 22 pages. doi:10.1145/3706598.3713166

[48] Na Sun and Donald Kalar. 2025. Gemini at Work: Knowledge Workers’ Perceptions and Assessment of Productivity Gains. In Proceedings ofthe 2025 ACM Designing Interactive Systems Conference (Funchal, Portugal) (DIS ’25). Association for Computing Machinery, New York, NY, USA, 3681–3695. doi:10.1145/3715336.3735679

[49] Ian P. Swift and Debaleena Chattopadhyay. 2026. A Difractive Analysis of GenAI through Perspectives on Productivity. In Proceedings ofthe 5th Annual Symposium on Human-Computer Interaction for Work (CHIWORK ’26). Association for Computing Machinery, New York, NY, USA, Article 20, 12 pages. doi:10.1145/3808045.3808074

[50] Macy Takafoli, Sijia Li, and Ville Mäkelä. 2024. Generative AI in User Experience Design and Research: How Do UX Practitioners, Teams, and Companies Use GenAI in Industry? Proceedings of the 2024 ACM Designing Interactive Systems Conference (DIS ’24) (2024), 1579–1593. doi:10.1145/3643834.3660720

[51] Lev Tankelevitch, Viktor Kewenig, Auste Simkute, Ava Elizabeth Scott, Advait Sarkar, Abigail Sellen, and Sean Rintel. 2024. The Metacognitive Demands and Opportunities of Generative AI. In Proceedings ofthe 2024 CHI Conference on Human Factors in Computing Systems (Honolulu, HI, USA) (CHI ’24). Association for Computing Machinery, New York, NY, USA, Article 680, 24 pages. doi:10.1145/3613904.3642902

[52] Jakob Tholander and Martin Jonsson. 2023. Design Ideation with AI - Sketching, Thinking and Talking with Generative Machine Learning Models. In Proceedings ofthe 2023 ACM Designing Interactive Systems Conference (Pittsburgh, PA, USA) (DIS ’23). Association for Computing Machinery, New York, NY, USA, 1930–1940. doi:10.1145/3563657.3596014

[53] Severi Uusitalo, Antti Salovaara, Tero Jokela, and Marja Salmimaa. 2024. ”Clay to Play With”: Generative AI Tools in UX and Industrial Design Practice. Proceedings ofthe 2024 ACM Designing Interactive Systems Conference (DIS ’24) (2024), 1566–1578. doi:10.1145/3643834.3661624

[54] Samangi Wadinambiarachchi, Ryan M. Kelly, Saumya Pareek, Qiushi Zhou, and Eduardo Velloso. 2024. The Efects of Generative AI on Design Fixation and Divergent Thinking. In Proceedings of the 2024 CHI Conference on Human Factors in Computing Systems (Honolulu, HI, USA) (CHI ’24). Association for Computing Machinery, New York, NY, USA, Article 380, 18 pages. doi:10.1145/3613904.3642919

[55] Sean Watts and Tanya Munir. 2025. Bridging the gap: exploring innovation enablers, challenges and AI adoption for enhanced workforce productivity. International Journal of Productivity and Performance Management 75, 3 (10 2025), 1030–1046. arXiv:https://www.emerald.com/ijppm/articlepdf/75/3/1030/10330225/ijppm-01-2025-0001en.pdf doi:10.1108/IJPPM-01-2025-0001

[56] Thomas Weber, Maximilian Brandmaier, Albrecht Schmidt, and Sven Mayer. 2024. Significant Productivity Gains through Programming with Large Language Models. Proc. ACM Hum.-Comput Interact. 8 (2024), 1–29. doi:10.1145/3661145

[57] Emma Webster. 2026. 3 Ways Product Teams Are Building Conviction Faster with Figma Make. Figma. https://www.figma.com/blog/3-ways-product teams-are-building-conviction-faster-with-figma-make/ Figma Blog

[58] Xiaotong (Tone) Xu, Arina Konnova, Bianca Gao, Cindy Peng, Dave Vo, and Steven P. Dow. 2025. Productive vs. Reflective: How Diferent Ways of Integrating AI into Design Workflows Afect Cognition and Motivation. In Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems (CHI ’25). Association for Computing Machinery, New York, NY, USA, Article 24, 15 pages. doi:10.1145/3706598.3713649

[59] Huiran Yi, Lu Xian, Yile Zhang, Jingyan Zeng, and Zifan Zhang. 2026. Designing with AI at Work: Designers’ Expertise and Pragmatic Decision-Making in Workplace AI Transformation. In Proceedings ofthe 2026 ACM Conference on Fairness, Accountability, and Transparency (Montreal, QC, Canada) (FAccT ’26). Association for Computing Machinery, New York, NY, USA, 8322–8343. doi:10.1145/3805689.3812239

[60] Tianyu Zhou, Ying Liu, Maneesh Kumar, and Sibao Wang. 2026. Generative AI-enabled chatbots for user-centred design: a state-of-the-art review. Journal ofEngineering Design 0, 0 (2026), 1–28. doi:10.1080/09544828.2026.263348

[61] Zijian Zhu, Tao Yu, Yijing Wang, and Junping Xu. 2025. Revolutionizing Imaging Design Content With AIGC: User-Centered Challenges, Opportunities, and Workflow Evolution. IEEE Access 13 (2025), 87600 – 87620. doi:10.1109/ACCESS.2025.3567750

[62] Naomi Chikumbi Zulu, Maria Åkesson, and Michel Thomsen. 2026. What Does Generative AI Mean for My Professional Competence? Insights into the Perspectives of Digital Creative Professionals. Interacting with Computers, Article iwaf056 (March 2026). doi:10.1093/iwc/iwaf056 Advance online publication.

A Scenario Designs  
A.1 Task 1 - "Create an alternative version of the social media post in dark mode."  
![](images/8380ee4c6d3271a868ebb2eecf5e23185e4860fdf4d8754155d3c3e69581bf7d.jpg)

![](images/ceea9ea1ae3729de97538354a2e226eb04af673268471f52483c49a6bd1723cd.jpg)

A.2 Task 2 - "Add a “Help & Support” option to the setings drop down menu."  
![](images/527b39fed9c27061a3f7ad8fb0416bb0c0cdeef25c80d4d0b903a25d1749a487.jpg)  
Manuscript submitted to ACM

![](images/70a21765247e244d6ab5723703a12bfb5d417420e50bad2c7d302557b707704b.jpg)

## A.3 Task 3 - "Create a comment flyout generated from the botom of the homepage after clicking ’Comment’”.

![](images/ec644760047d5f8b0fbf0cd5052a5573b2686bcb593b7f58a3a0d8852899009f.jpg)

B Exit survey questions

## B.1 System Usability Questions

Q1. I think that I would like to use Figma frequently.

Q2. I found Figma unnecessarily complex.

Q3. I thought Figma was easy to use.

Q4. I think that I would need the support of a technical person to be able to use Figma.

Q5. I found the various functions in Figma were well integrated.

Q6. I thought there was too much inconsistency in Figma.

Q7. I would imagine that most people would learn to use Figma very quickly.

Q8. I found Figma very cumbersome to use.

Q9. I felt very confident using Figma.

Q10. I needed to learn a lot of things before I could get going with Figma.

## B.2 Design Quality Questions

Q11. How closely do you believe your designs in this study matched the instructions provided in each scenario?

Q12. How would you rate the quality of your created designs during this study compared to what you could have made in Figma outside of this study?

Q13. Have you ever used Figma Prototyping before this study?

## B.3 Treatment-Only Questions

Q14. I trusted the AI tools available in this study to interpret my intentions correctly.

Manuscript submitted to ACM

Q15. I believe Figma AI would increase my ability to contribute to the design process.

## B.4 AI Use Questions

Q16. How often do you use AI tools for work?

Q17. How often do you use AI tools for design?

## C Participant Atribute Descriptive Statistics

Table 7. Participant demographics and background characteristics distributions.
<table><tr><td>Variable</td><td>Full Sample</td><td>Designers</td><td>PMs</td></tr><tr><td>Age</td><td></td><td></td><td></td></tr><tr><td>18-24</td><td>2%</td><td>4%</td><td>0%</td></tr><tr><td>25-34</td><td>29%</td><td>24%</td><td>35%</td></tr><tr><td>35-44</td><td>32%</td><td>35%</td><td>29%</td></tr><tr><td>45-54</td><td>29%</td><td>35%</td><td>22%</td></tr><tr><td>55-64</td><td>6%</td><td>0%</td><td>12%</td></tr><tr><td>65+</td><td>2%</td><td>2%</td><td>2%</td></tr><tr><td>Gender</td><td></td><td></td><td></td></tr><tr><td>Male</td><td>50%</td><td>55%</td><td>45%</td></tr><tr><td>Female</td><td>50%</td><td>45%</td><td>55%</td></tr><tr><td>Other</td><td>一</td><td>一</td><td>-</td></tr><tr><td>Education</td><td></td><td></td><td></td></tr><tr><td>Undergraduate degree</td><td>64%</td><td>73%</td><td>55%</td></tr><tr><td>Graduate degree</td><td>36%</td><td>27%</td><td>45%</td></tr><tr><td>Employment Status</td><td></td><td></td><td></td></tr><tr><td>Employed full time</td><td>78%</td><td>69%</td><td>88%</td></tr><tr><td>Freelance / contractor</td><td>16%</td><td>20%</td><td>12%</td></tr><tr><td>Self-employed</td><td>6%</td><td>11%</td><td>一</td></tr><tr><td>Years of Experience</td><td></td><td></td><td></td></tr><tr><td>2–5 years</td><td>11%</td><td>8%</td><td>14%</td></tr><tr><td>5-10 years</td><td>29%</td><td>21%</td><td>37%</td></tr><tr><td>10+ years</td><td>60%</td><td>71%</td><td>49%</td></tr><tr><td>Industry Sector</td><td></td><td></td><td></td></tr><tr><td>Information technology</td><td>46%</td><td>35%</td><td>57%</td></tr><tr><td>Banking or financial services</td><td>20%</td><td>18%</td><td>22%</td></tr><tr><td>Media, entertainment, arts</td><td>9%</td><td>14%</td><td>4%</td></tr><tr><td>Communications or marketing</td><td>8%</td><td>16%</td><td>一</td></tr></table>

Continued on next page

Table 7. Participant demographics and background characteristics. Continued.
<table><tr><td>Variable</td><td>Full Sample</td><td>Designers</td><td>PMs</td></tr><tr><td>Retail</td><td>5%</td><td>4%</td><td>6%</td></tr><tr><td>Other</td><td>12%</td><td>14%</td><td>10%</td></tr><tr><td>Size of Organization</td><td></td><td></td><td></td></tr><tr><td>1-1,000 employees</td><td>55%</td><td>62%</td><td>47%</td></tr><tr><td>1,000–10,000 employees</td><td>21%</td><td>14%</td><td>25%</td></tr><tr><td>10,000+ employees</td><td>24%</td><td>24%</td><td>28%</td></tr><tr><td>Frequency of Figma Design Use</td><td></td><td></td><td></td></tr><tr><td>Multiple times a day</td><td>55%</td><td>73%</td><td>37%</td></tr><tr><td>Once a day</td><td>14%</td><td>8%</td><td>20%</td></tr><tr><td>Once a week</td><td>18%</td><td>14%</td><td>22%</td></tr><tr><td>Once a month or less</td><td>13%</td><td>6%</td><td>20%</td></tr><tr><td>Have Ever Used Figma AI</td><td></td><td></td><td></td></tr><tr><td>Yes</td><td>49%</td><td>61%</td><td>37%</td></tr><tr><td>No</td><td>51%</td><td>39%</td><td>63%</td></tr><tr><td>Frequency of AI Tool Use for Work</td><td></td><td></td><td></td></tr><tr><td>Daily</td><td>78%</td><td>69%</td><td>88%</td></tr><tr><td>A few times a week</td><td>15%</td><td>20%</td><td>10%</td></tr><tr><td>Once a week</td><td>5%</td><td>8%</td><td>2%</td></tr><tr><td>Once a month</td><td>0%</td><td>一</td><td>一</td></tr><tr><td>Never</td><td>2%</td><td>4%</td><td>一</td></tr><tr><td>Frequency of AI Tool Use for Design</td><td></td><td></td><td></td></tr><tr><td>Daily</td><td>15%</td><td>26%</td><td>4%</td></tr><tr><td>A few times a week</td><td>35%</td><td>45%</td><td>25%</td></tr><tr><td>Once a week</td><td>20%</td><td>12%</td><td>29%</td></tr><tr><td>Once a month</td><td>13%</td><td>10%</td><td>16%</td></tr><tr><td>Never</td><td>17%</td><td>8%</td><td>27%</td></tr><tr><td>Previously Used AI Tools</td><td></td><td></td><td></td></tr><tr><td>Figma Make</td><td>34%</td><td>43%</td><td>25%</td></tr><tr><td>ChatGPT</td><td>94%</td><td>92%</td><td>96%</td></tr><tr><td>Lovable</td><td>34%</td><td>26%</td><td>43%</td></tr><tr><td>Google Gemini</td><td>79%</td><td>77%</td><td>82%</td></tr><tr><td>Bolt</td><td>11%</td><td>12%</td><td>10%</td></tr><tr><td>GitHub Copilot</td><td>26%</td><td>18%</td><td>35%</td></tr><tr><td>Claude</td><td>72%</td><td>65%</td><td>80%</td></tr></table>

Continued on next page

Table 7. Participant demographics and background characteristics. Continued.
<table><tr><td>Variable</td><td>Full Sample Designers</td><td></td><td>PMs</td></tr><tr><td>Replit</td><td>14%</td><td>10%</td><td>18%</td></tr><tr><td>Cursor</td><td>24%</td><td>20%</td><td>29%</td></tr></table>

## D Regression model covariate terms

Table 8. GEE covariate coeficients for completion and cumulative time
<table><tr><td></td><td>Completion (pp) Cumulative time</td><td></td></tr><tr><td colspan="3">Age (Reference: 18–34)</td></tr><tr><td>35-44</td><td>−1.4 (4.6)</td><td>-0:57 (4:22)</td></tr><tr><td>45-54</td><td>−3.5 (6.2)</td><td>6:00 (6:02)</td></tr><tr><td>55+</td><td>-12.3 (11.1)</td><td>13:27 (10:01)</td></tr><tr><td colspan="3">Gender (Reference: Male)</td></tr><tr><td>Female</td><td>3.5 (4.3)</td><td>0:08 (3:01)</td></tr><tr><td colspan="3">Education (Reference: Undergraduate)</td></tr><tr><td>Graduate degree</td><td>−6.1 (3.9)</td><td>-1:36 (3:29)</td></tr><tr><td colspan="3">Employment Status (Reference: Freelance/Contractor)</td></tr><tr><td>Full Time</td><td>3.1 (4.7)</td><td>-2:31 (4:33)</td></tr><tr><td>Self Employed</td><td> $- 2 0 . 8 ^ { ^ { \star \star } } \left( 7 . 0 \right)$ </td><td>6:24 (8:31)</td></tr><tr><td colspan="3">Years of Experience (Reference: 2–5 years)</td></tr><tr><td>5–10 years</td><td> $- 1 3 . 8 \mathrm { \Lambda } ^ { \star } \left( 5 . 7 \right)$ </td><td>3:42 (5:06)</td></tr><tr><td>10+ years</td><td>–9.5 (6.9)</td><td>8:22 (6:19)</td></tr><tr><td colspan="3">Industry (Reference: Banking &amp; financial services)</td></tr><tr><td>Information technology</td><td>11.3 (5.7)</td><td>-8:36* (4:20)</td></tr><tr><td>Communications &amp; Marketing</td><td>13.8* (6.4)</td><td>-6:30 (4:46)</td></tr><tr><td>Media &amp; Entertainment</td><td>13.5* (6.6)</td><td>-0:26 (5:41)</td></tr><tr><td>Retail</td><td>15.3 (7.9)</td><td>-12:39* (5:35)</td></tr><tr><td>Other</td><td>-1.6 (8.4)</td><td>-1:09 (4:37)</td></tr><tr><td colspan="3">Organization Size (Reference: 1–1,000 employees)</td></tr><tr><td>1,000-10,000 employees</td><td>-9.3 (5.0)</td><td>2:13 (3:26)</td></tr><tr><td>10,000+ employees</td><td>-12.6** (4.8)</td><td>5:21 (4:21)</td></tr><tr><td colspan="3">Figma Design Use Frequency (Reference: Few times a year or less)</td></tr><tr><td>Multiple times a day</td><td>4.9 (6.7)</td><td>3:18 (5:09)</td></tr><tr><td>Once a day</td><td>3.9 (8.3)</td><td>1:15 (5:30)</td></tr></table>

Table 8 – continued from previous page
<table><tr><td></td><td>Completion (pp) Cumulative time</td><td></td></tr><tr><td>Once a week</td><td>−4.8 (8.2)</td><td>1:36 (5:37)</td></tr><tr><td>Used Figma AI (Reference: Have not used)</td><td></td><td></td></tr><tr><td>Have used</td><td>5.7 (4.7)</td><td> $- 6 { : 4 3 } ^ { \star } \left( 3 { : 1 3 } \right)$ </td></tr><tr><td>AI Tools For Work Frequency (Reference: Few times a week)</td><td></td><td></td></tr><tr><td>Daily</td><td>4.4 (3.9)</td><td>-3:49 (3:57)</td></tr><tr><td>Once a week</td><td>-3.4 (12.4)</td><td>0:06 (7:57)</td></tr><tr><td>Never</td><td> $- 3 0 . 3 ^ { ^ { \ast \star } } \left( 1 0 . 9 \right)$ </td><td> $3 5 { : 0 0 } ^ { \star \star } \left( 1 3 { : 0 8 } \right)$ </td></tr><tr><td>AI Tools for Design Frequency (Reference: Few times a week)</td><td></td><td></td></tr><tr><td>Daily</td><td>−8.4 (7.1)</td><td>-4:14 (4:28)</td></tr><tr><td>Once a week</td><td>16.8* (7.0)</td><td>-0:37 (4:18)</td></tr><tr><td>Once a month</td><td>−1.9 (7.9)</td><td>5:58 (7:45)</td></tr><tr><td>Never</td><td>0.3 (7.7)</td><td>-4:11 (4:27)</td></tr></table>

\* � < 0.05, \*\* � < 0.01, \*\*\* � < 0.001.  
Received 20 February 2007; revised 12 March 2009; accepted 5 June 2009