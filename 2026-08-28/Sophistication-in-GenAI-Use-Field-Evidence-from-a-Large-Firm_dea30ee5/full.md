# Sophistication in GenAI Use: Field Evidence from a Large Firm

Nicholas J. Hallman University of Texas at Austin NicholasHallman@utexas.edu

Zachary T. Kowaleski\*

University of Texas at Austin

zkowaleski@utexas.edu

Anu Puvvada

KPMG LLP

Anu.Puvvada@kpmg.com

Jaime J. Schmidt

University of Texas at Austin

Jaime.Schmidt@mccombs.utexas.edu

Draft Date: 08/25/2026

We thank Kevin Avalos, Ethan Burris, Steve Chase, Tianhui Gou, David Harrison, Steve Kachelmeier, Clay Kohler, Xinxuan Li (discussant), Kevin Veenstra (discussant), Adam Weiss, and Olivia Weiss for guidance, feedback, and data support. We thank participants at the Annual Telfer Conference in Accounting, Auditing, and Accountability and the Arizona Accounting Research Conference for helpful comments and suggestions. Jaime Schmidt gratefully acknowledges financial support from the KPMG Centennial Fellowship in Accounting. Anu Puvvada is a full time employee of KPMG LLP. The remaining authors declare that they have no conflicts of interest. The proprietary data used in this paper were provided by KPMG LLP and classified using a large language model, as described in the paper. We are grateful to KPMG LLP for providing data access.

# Sophistication in GenAI Use: Field Evidence from a Large Firm

Abstract: We study how sophistication in generative AI (genAI) use varies among the back office workforce of a large firm. Using proprietary data, we observe 713,564 employee prompts and their corresponding large language model responses from nearly 4,000 backoffice employees across 15 functional areas over eight months in 2025. We document three main findings. First, senior employees exhibit more sophisticated genAI use, consistent with domain expertise complementing genAI capabilities. Second, sophistication varies considerably across functions and is highest in Strategy, Digital Innovation, and Project Management, three groups that share a focus on firmwide strategic initiatives and organizational change. Third, we observe neither improvements in sophistication over time nor lasting improvements following formal AI training, suggesting that sophisticated use can be difficult to change. Together, our study provides measures of and insights into sophisticated genAI use that managers can use to improve outcomes and that researchers can use in future research.

Keywords: generative AI; large language models; workplace technology adoption; technology use sophistication

## 1 Introduction

Companies have rapidly adopted generative artificial intelligence (genAI), spending an estimated \$37 billion in 2025 (Tully et al. 2025). Yet adoption alone does not guarantee a return on investment. As companies face costs and token-based usage limits, “encouraging the right behaviors to get the most out of every AI interaction” is going to be necessary to achieve organizational goals (Cantrell et al. 2026). To provide insights into how people interact with genAI, we examine how genAI sophistication varies over time, across seniority levels and functional areas, and following AI training within a large firm that has already achieved widespread genAI adoption.

We define sophisticated genAI use as skilled use informed by an understanding of how the tool works and where it can be applied. Such use is reflected in clear and specific instructions, deliberate prompting techniques, and application of the tool across a broad mix of tasks. While a growing literature has examined whether, how often, and for what purposes genAI is used (Bick et al. 2026; Bonney et al. 2026; Chatterji et al. 2025; Counts et al. 2026; Handa et al. 2025), these studies provide little evidence on how employees interact with genAI and, more importantly, how well they do so. We extend this literature while also providing insights that managers can use to better measure AI use within their organizations.

We study variation in genAI sophistication within the back-office group of KPMG LLP (KPMG), which provided its employees with access to multiple genAI tools during the eight-month period from January through August 2025: including Microsoft Copilot and aIQ Chat, an enterprise-specific chat tool that provided access to commercial large language models (LLMs) such as those from Anthropic and OpenAI. For aIQ Chat, we observe the full conversation transcripts. To measure genAI sophistication, we prompt an LLM to classify each conversation by use case and assess the clarity and specificity of its instructions and the prompting techniques employed, producing conversation-level measures of the dimensions of sophisticated use. For Microsoft Copilot, we observe how frequently employees use the tool but not the content of their conversations. We aggregate our conversation-level measures and Copilot usage to the employee-month level and link them to each employee’s seniority, functional area, and completed genAI training. Our sample includes employees in IT, marketing, accounting, finance, and other non-client-facing administrative functions. Because comparable functions exist in most large organizations, our setting should provide evidence relevant to managers in companies beyond KPMG.

We begin by documenting employee adoption and usage frequency. On average, 83.9% of employee-months include use of at least one of the two genAI tools, with Copilot used more frequently than aIQ Chat. This rate exceeds the 32.1% reported in prior survey evidence (Bick et al. 2026), possibly because the firm we study strongly encourages its employees to use genAI. Broad adoption, however, masks substantial variation in usage intensity. Among employees who use aIQ Chat during the sample period, the top decile accounts for more than half of all conversations. Senior employees use the tools more frequently, primarily because of greater Copilot use. Monthly adoption rates also vary across functional areas, from 73% in Compliance to 94% in Internal Audit. Finally, usage changes modestly over the eight-month window, consistent with the firm having reached a relatively mature stage of adoption.

We next document the tasks for which employees use aIQ Chat. These use-case categories form the basis for our subsequent measure of how broadly each employee applies the tool. Writing-related tasks, including requests to edit user-provided text, generate new content, or both, appear in roughly three-quarters of conversations. Employees also use the tool for knowledge retrieval, document understanding, and software guidance, with smaller but meaningful proportions devoted to coding and data analysis, ideation, and other tasks. Use cases also vary across seniority levels and functional areas. More senior employees make more knowledge-retrieval requests, whereas staff employees make more writing and personal requests. Communications and Sales & Marketing employees seek more assistance with writing and ideation, while Information Technology employees make more requests involving coding and software guidance. Together with the adoption analyses, these findings provide context for our analysis of how sophistication varies within the firm.

Against this backdrop, we turn to the paper’s primary analyses and examine variation in sophisticated use. To operationalize this concept, we use three composite measures: use-case diversity, deliberate strategy use, and prompt clarity. Each measure corresponds to one component of sophisticated use: applying genAI across a broad mix of tasks, employing deliberate prompting techniques, and providing clear and specific instructions. Our analyses of variation across employee-months yield three main findings. First, we do not observe employees’ genAI use becoming more sophisticated over the eight-month sample period. Second, each measure rises with seniority, with staff ranking below managers and managers below employees above the manager level. Third, sophistication varies considerably across functional areas. It is highest in Strategy, Digital Innovation, and Project Management, three groups that share a focus on firmwide strategic initiatives and organi zational change. Accounting & Finance lags on all three sophisticated use measures.

Our sophistication measures require conversation transcripts, which firms rarely retain and are costly to process. We therefore examine whether these measures relate to four usage characteristics that managers can observe from metadata alone: how much an employee asks the tool to do (ambition), how much the employee iterates on a request (persistence), how often the employee uses aIQ Chat (frequency), and how often the employee also uses Copilot, another of the firm’s genAI tools (flexibility). A reliable association would give managers a low-cost way to gauge how well employees use genAI, not only how often. In simple correlations, regressions with additional controls, and models with employee fixed effects, we consistently find that greater ambition and persistence, reflected in longer initial prompts and more iteration within conversations, accompany more sophisticated use. Finally, we examine whether formal AI training is associated with increases in employee genAI sophistication. In models with employee fixed effects, we find more sophisticated use in the month an employee completes formal genAI training but not in the months that follow, suggesting that any improvement associated with these trainings is temporary. While our evidence does not establish that the training was ineffective, it provides little evidence that completion of formal training produces lasting changes in observed behavior.

Our study provides managers with several practical benefits. First, we provide a customizable process for measuring sophisticated genAI use within their organizations. Second, we provide descriptive evidence from a firm with widespread adoption. We do this across employee groups, in relation to easily observable measures, and across periods before and after formal AI training. This evidence creates genAI benchmarking opportunities that can help managers evaluate their organization’s progress and inform resource allocation decisions. For example, the absence of improvement over time or sustained improvement following training illustrates the difficulty organizations may face in fostering lasting changes in employee genAI use.

Our paper also makes several contributions to the academic literature. Existing research documents rapid genAI adoption, common workplace use cases, and productivity effects of access to particular AI tools (e.g., Arntz et al. 2025; Bick et al. 2026; Bonney et al. 2026; Choi and Xie 2026; Counts et al. 2026; Wang, Gao, et al. 2024). Yet this research does not evaluate how well employees use genAI. We add to the literature by introducing new measures of sophisticated use and by providing field evidence on how sophisticated genAI use varies among employees inside one organization after adoption is already widespread. We find that sophistication rises with seniority and differs substantially across functional areas. We also observe little systematic improvement over time. Sophistication is higher in the month of formal training but not in subsequent months. Together, these findings help us better understand the evolving genAI adoption timeline in which attention shifts from whether employees adopt genAI to how sophisticated use is distributed and changes after adoption.

## 2 Background and Literature Review

## 2.1 From Adoption to Sophistication

Companies have rapidly adopted genAI. In a single year, employee access to AI tools has expanded by 50 percent (Cantrell et al. 2026). Despite its rapid adoption, recent and contemporaneous research documents uneven workplace diffusion across workers, firms, business functions, and tasks (Arntz et al. 2025; Bick et al. 2026; Bonney et al. 2026; Henseke 2026; Humlum and Vestergaard 2025). For example, a working paper by Arntz et al. (2025) document that while 64 percent of the employees they surveyed report using AI, only 20 percent report using it frequently. Bick et al. (2026) show that the intensity of AI use and related productivity gains vary by industry and firm climate. Similarly, a working paper by Bonney et al. (2026) shows that within adopting firms, use can vary substantially by function.

Frequent use is a necessary precursor to achieving organizational benefits. For example, using a large randomized experiment of 6,000 workers at 56 firms, Dillon et al. (2025) show that genAI reduces time spent on email and accelerates document completion but only for the 40 percent of workers who used it frequently. Similarly, Estep et al. (2024) show in a lab experiment that genAI effects depend on whether managers and au ditors both use and respond to AI-generated information. Combined, these studies demonstrate that company genAI adoption by itself will not necessarily result in work benefits if employees’ adoption consists of only infrequent use.

Large-scale analyses of AI conversations show that writing, information retrieval, analysis, and other forms of information work (e.g., the creation, processing, and communication of information) make up most of the common workplace uses of genAI (Chatterji et al. 2025; Counts et al. 2026; Handa et al. 2025; Tomlinson et al. 2025). For example, a working paper by Chatterji et al. (2025) classifies a large sample of ChatGPT conversations and find that writing, practical guidance, and information seeking dominate AI use cases, especially in professional work. Similarly, Handa et al. (2025) examine millions of Claude conversations and find that software development and writing dominate the use cases on that platform. Each of these studies examines users on a single platform to reveal what people do with the tool, but generally not how skillfully individual employees do it.

Information systems research distinguishes use from effective use, emphasizing that organizational benefits depend on how users engage with a technology rather than merely whether they access it (Burton-Jones and Grange ${ 2 0 1 3 ; }$ Orlikowski 2000). How users engage matters for genAI, as variation in prompts meaningfully affects outcomes on specific tasks (Croom et al. 2026; Eulerich et al. 2024; Shaffer and Wang 2026). Across tasks, a recent report from AI lab Anthropic (Swanson et al. 2026) measures the “AI fluency” of a sample of its subscribers for a seven-day period, which the authors define as iterative, evaluative, and augmentative behaviors in AI conversations. While they study associations among behaviors at a conversation level, we ask how sophisticated genAI use varies within an organization in which adoption is already widespread. Answering this question requires measures of sophisticated use, which we develop from conversation transcripts in Section 4.

## 2.2 Seniority, Training, and Functional Area Differences

We organize the remaining literature around three dimensions of potential variation in sophisticated use: seniority, formal training, and functional area. Seniority partly reflects accumulated experience, and prior research provides mixed evidence on the relationship between experience and AI.<sup>1</sup> AI assistance can generate larger productivity gains for less-experienced workers by transmitting best practices (Brynjolfsson et al. 2025; Gambacorta et al. 2024), yet domain expertise can improve the development, selective use, and evaluation of AI outputs (Choi and Xie 2026; Lou and Wu 2021). Wang, Gao, et al. (2024) show that the benefits of AI can be lower for senior workers when they exhibit lower trust in AI. However, prior studies primarily estimate the productivity benefits of access to a particular AI application or tool, or they measure firm-level AI capability. Instead, we examine whether seniority is associated with clearer, broader, and more deliberate genAI use.

Prior research identifies training as an important factor for achieving workplace AI adoption and perceived value. For example, Henseke (2026) shows that workplace training strengthens AI adoption. Humlum and Vestergaard (2025) document that employees perceive a need for training to achieve productivity gains from ChatGPT. Practitioner evidence likewise emphasizes training as an important lever for converting genAI access into returns (Brown 2025; Korst et al. 2025). These studies suggest that formal AI training should improve employees’ ability to use genAI. However, prior work does not establish whether such behavioral improvements from training actually occur.

AI adoption and use vary across business functions, occupations, and tasks, with especially prominent application in sales and marketing, strategy, information technology, writing, information retrieval, and document analysis (Bonney et al. 2026; Counts et al. 2026; Handa et al. 2025; Tomlinson et al. 2025). These patterns suggest that workflow fit and task composition influence not only whether employees use genAI, but likely how effectively they engage with it. Therefore, we are likely to observe differences in sophisticated use across functional areas.

## 3 Research Design

## 3.1 Setting and Data Sources

We obtain proprietary data on the back-office workforce of a large professional services firm for January through August 2025. Although these employees are not clientfacing, they perform administrative functions critical to the organization’s success, such as IT, internal audit, strategy, public relations, marketing, accounting, and finance.<sup>2</sup> Thus, our findings should be relevant beyond professional services firms to employees in comparable functions within other large organizations.

We observe two LLM channels that the firm provides to employees. The first is a standalone chat interface called aIQ Chat that, during our sample window, provided users with access to commercial LLMs from several vendors, including Anthropic’s Claude 3.5 Sonnet, OpenAI’s GPT-4o and o1, as well as models from Google’s Gemini, Meta’s Llama, and others. GPT-4o was the default model. The transcripts for employees’ interactions with aIQ Chat provide the basis for our primary analyses of sophisticated use. The second LLM channel is Microsoft Copilot, an embedded assistant available through workplace software such as Outlook and Teams.

## 3.2 Sample Construction

We analyze the data at two levels:

1. Conversation level: Full transcripts of aIQ Chat conversations, including all user prompts and LLM responses within each conversation.

2. Employee-month level: Monthly use volume measures for both aIQ Chat and Copilot, plus aggregate measures derived from aIQ Chat transcripts.

We link the conversation transcripts and frequency measures to employee seniority, functional area, and completion of firm-provided AI training. As we do not observe the content of Copilot interactions, we use only aIQ Chat to construct conversation-level measures and examine sophisticated use.

Table 1 summarizes the sample. Panel A starts from 5,191 back-office employees. Of these employees, 3,925 (75.6%) use aIQ Chat at least once during any month of the sample window. The Copilot dataset covers 4,392 employees, of whom 3,962 (90.2%) use Copilot and 2,861 (65.1%) use both tools. Panel B reports the scale of aIQ Chat activity. We observe 158,496 conversations and 713,564 user prompts, or 4.5 user prompts per conversation. Usage is highly skewed. The mean active employee has 40 conversations, while the median active employee has 14. The top decile of active employees accounts for 51% of conversations.<sup>3</sup>

## 4 Measures

This section describes how we examine aIQ Chat transcripts to create structured data and measures at the conversation and employee-month levels.

## 4.1 Structuring Data

We submit each aIQ Chat transcript to an LLM along with a metaprompt, as reproduced in Online Appendix C. The metaprompt instructs the LLM to use structured coding rules to identify the conversation’s use cases, prompting strategies, and other characteristics of sophisticated use. We treat the resulting LLM outputs as structured measurement data.<sup>4</sup>

## 4.2 Conversation-Level Measures

First, we classify each conversation by use case. The use case categories consist of Writing & Communication (WRITING), Coding & Data Analysis (CODING DATA), Text-Based Analysis (TEXT ANALYSIS), Knowledge & Expertise (KNOWLEDGE), Software & Tool Guidance (TOOL GUIDE), Creative Thinking & Ideation (IDEATION), Personal & Non-work (PERSONAL), and Other (OTHER). We also decompose each of these broad categories into subcategories. We note that workplace conversations often blend tasks, such as summarizing a policy and drafting a stakeholder email in the same exchange. In such instances, the conversation is classified into multiple categories.

The same metaprompt scores several dimensions of either the first substantive user prompt or the overall conversation.<sup>5</sup> For the first substantive prompt, ratings capture linguistic complexity (1ST PROMPT LANG COMPLEXITY) and task and structural complexity (1ST PROMPT COMPLEXITY). For the overall conversation, measures consider prompt organization and structure (STRUCTURE), overall prompting sophistication (PROMPT SOPH), specificity and precision (SPECIFICITY), and clarity of the requested output format (FOR-MAT CLARITY).

We also obtain binary indicators for whether the user uploads a document (DOC UP-LOAD), specifies response constraints (CONSTRAINTS), requests structured output such as a table or JSON (STRUCT OUTPUT), provides acceptance criteria for the response (AC-CEPT CRITERIA), expresses explicit satisfaction with a previous response (SATISFIED), makes frequent typos (TYPOS), uses all-lowercase text (ALL LOWER), or includes pleasantries (PLEASANTRIES). Finally, we code prompting strategies: role or persona assignment (ROLE PLAY), few-shot examples (FEW SHOT), explicit requests for step-by-step reasoning (CHAIN OF THOUGHT), self-verification or checking (SELF CHECK), and instructions for the model to ask clarifying questions (INTERACTIVE REFINE) (Schulhoff et al. 2024; Shaffer and Wang 2026; Wang, Wei, et al. 2023; Wei et al. 2023).

## 4.3 Employee-Month Aggregation and Composite Measures

We aggregate conversation-level measures to the employee-month level and restrict the sample to active employee-months, i.e., months when an employee has at least one aIQ Chat conversation. As before, we define sophisticated genAI use as skilled use informed by an understanding of how the tool works and where it can be applied. Such use is reflected in clear and specific instructions, deliberate prompting techniques, and application of the tool across a broad mix of tasks. To operationalize this definition, we use three composite measures that proxy for distinct but overlapping aspects of sophisticated use. The first composite measure, prompt clarity (CLARITYZ), measures how clearly and specifically users formulate requests. We construct it by first combining standardized values of nine variables, including assessed organization, specificity, the presence of set constraints and requests for structured output, and then by standardizing the resulting composite measure. Deliberate strategy use (ANY STRAT), our second composite measure, identifies how often the users actively employ prompting techniques in their conversations, such as role assignment or instructions for the LLM to self-verify. Use-case diversity (USE DIV), our third composite measure, identifies whether an employee applies aIQ Chat across a narrow or broad mix of task categories during the month.<sup>6</sup> We interpret higher values of our composite measures as proxies for sophisticated use, not as direct measures of output quality or productivity.

## 4.4 Presentation of Results

We present our results as dot-plot figures and report the underlying tables and statistics in Online Appendix B. Each dot-plot reports one outcome measure (e.g., use cases), with rows that correspond to one particular group, i.e., a month, a seniority level, or a functional area. A dot marks the group’s mean value, plotted on a horizontal axis whose center reference line denotes the overall sample mean (or zero, in Figure 4); the axis is labeled with the sample mean at the center and the group minimum and maximum at the ends, so a dot’s horizontal position shows how far the group lies above or below the sample average. Color and fill jointly depict the direction and statistical significance of each group’s difference-from-rest test using a scheme that is common for all of the dotplot figures: a filled burnt-orange dot depicts a mean value significantly above the sample mean, an open burnt-orange dot depicts a mean value above the sample mean that is not statistically significant, a filled gray dot depicts a mean value significantly below the sample mean, and an open gray dot depicts a mean value below the sample mean that is not significant. The tables in Online Appendix B report the exact values, sample sizes, and significance levels for all dot-plot figures.

## 5 Extent of GenAI Use

As context for our primary analyses, this section first describes the prevalence and volume of genAI use in our sample and how use varies over time, seniority, and functional area, and then compares these patterns with evidence documented in prior research.

## 5.1 Variation in Extent of GenAI Use

Figure 1 summarizes use volume for the employee-month panel as described in Section 4.4. Usage is broad in our sample. On average, 83.9% of employee-months include use of at least one tool (Any Tool). At the employee-month level, Copilot use (78.0%) is substantially more common than aIQ Chat use (43.7%).

Panel A reports usage by month. We do not observe rapid or accelerating growth during the January–August 2025 window. Instead, the month-to-month changes are modest. The proportion of users with any Copilot use rises over the sample window from a minimum of 71.0 percent in January to a maximum of 82.8 percent in August, while aIQ Chat usage is comparatively flat with most months not statistically different from the sample mean. This pattern is inconsistent with a simple story of uniformly increasing chat use over time.

Panel B shows that usage positively correlates with seniority. Above-manager employees have higher overall usage driven by more Copilot use, while aIQ Chat usage is flatter across seniority groups. Panel C reports variation across functional areas. Digital Innovation, Communications, Sales & Marketing, and Internal Audit are typically above the sample mean. Accounting & Finance is the only functional area whose average is significantly below the sample mean for every reported usage measure. The contrast between Internal Audit and Accounting & Finance is notable because these areas may draw on overlapping professional backgrounds. In general, aIQ Chat and Copilot use are positively correlated across functional areas, but the relationship is not uniform. Some functions appear to rely more heavily on one channel: for example, Specialized Services is above the sample mean for Copilot use but below the sample mean for aIQ Chat use, while Project Management shows the opposite pattern.

## 5.2 Comparison with Prior Adoption Evidence

Our adoption rates are higher than rates reported in prior research (Arntz et al. 2025; Bick et al. 2026; Bonney et al. 2026; Henseke 2026; Humlum and Vestergaard 2025). For example, Bick et al. (2026) survey U.S. workers in August and November 2024, shortly before our January 2025 sample begins, and report that 32.1% of employed respondents used genAI for work and 27.3% used it at work in the prior week. In our setting, 83.9% of employee-months use at least one enterprise LLM channel. There are several possibilities for this difference. First, because we study an organization that provides and encourages use of genAI tools, our study, by construction, omits individuals who lack this organizational support.<sup>7</sup> Second, Bick et al. (2026) report especially high work adoption in computer and mathematical occupations (53.6%), management occupations (52.0%), and business and finance occupations (48.2%). Our sample of office workers disproportionately captures these higher use groups while omitting lower use groups like personal services (15.8%). Although Bick et al. (2026) conducted their survey at an earlier point in the adoption curve than our January–August 2025 analyses, our lack of a strong upward time trend suggests that timing plays a limited role in explaining the difference. Finally, while self-reported adoption rates in surveys may understate actual use (Ling et al. 2026), we observe proprietary usage logs and complete employee genAI conversation transcripts.

Our seniority finding contrasts both with conventional wisdom and, to a lesser extent, with prior academic research. Conventional wisdom holds that recent entrants, already fluent in these tools, will out-perform experienced colleagues; executives say so directly and young degree-holders report the highest confidence in their own AI readiness (Pohle and Fernandez 2026). Survey evidence already contradicts that view: Bick et al. (2026) find work use follows an inverted U in age, peaking in users’ 30’s and 40’s rather than among the youngest workers. Our results reinforce that correction at the young end but diverge at the other, with use rising monotonically rather than turning down among senior employees.

Our functional area analyses somewhat echo occupation-based evidence. For example, while Bick et al. (2026) report high work adoption in computer and mathematical, management, and business and finance occupations, we report high usage in the computerfocused area Digital Innovation and among employees who manage others (as part of our seniority tests).<sup>8</sup> We further disaggregate business-focused areas and report heterogeneity that resembles evidence from Bonney et al. (2026). They report firm-level evidence from the November 2025–January 2026 Business Trends and Outlook Survey and find that, among firms reporting AI use in at least one business function, Sales & Marketing is the most common function for use, followed by Strategy & Business Development, then Information Technology. This ordering resembles our finding of relatively high usage in Sales & Marketing, Strategy, and Digital Innovation. While their employment-weighted results place Finance & Accounting among the leaders, we find the opposite. The difference need not represent a contradiction as their measure identifies whether firms deploy any form of AI within a function, while ours measures how frequently employees in that function use two enterprise genAI tools. Taken together, inferences drawn from our sample are broadly consistent with large-scale survey evidence.

## 6 GenAI Use Cases

As context for our primary analyses, this section first describes the genAI use cases in our sample and how use varies over time, seniority, and functional area, and then compares these patterns with prior use-case evidence.

## 6.1 Variation in GenAI Use Cases

Table 2 reports our categorized use cases at the conversation level.<sup>9</sup> Writing appears in 73.3% of conversations, far exceeding any other category, and includes subcategories for editing user-provided material (45.8%) and generating new text (34.8%). Employees also use aIQ Chat to retrieve, interpret, and apply information. Knowledge/expertise queries appear in 23.0% of conversations, and text/document analysis appears in 10.7%. Software/tool guidance appears in 9.9% of conversations. Coding/data analysis appears in 7.3% of conversations. Ideation and personal requests appear less often but still represent nontrivial proportions.

Figure 2 reports the proportion of conversations assigned to each use case by month, seniority, and functional area, as described in Section 4.4.<sup>10</sup> Across months in Panel A, we do not observe a clear monotonic shift in how employees use aIQ Chat. Writing remains the baseline use case throughout the sample window. Secondary categories move modestly: coding/data analysis increases somewhat, while knowledge retrieval and ideation decrease.

Panel B shows that use cases vary by seniority. Staff conversations skew toward writing and personal requests. Managers make relatively more coding and data-analysis requests. Above-manager employees are the most knowledge-intensive users, which is the largest swing across seniority groups in Panel B. This pattern is consistent with employees using LLMs in ways that reflect their responsibilities and complement their domain knowledge.

Panel C suggests that use case patterns align with expected workflow differences across functional areas. Administrative Services and Internal Operations show the highest proportion of personal requests, consistent with work tasks that may appear personal, such as travel and scheduling performed by these groups. Information Technology has comparatively more coding and tool guidance. Communications and Sales & Marketing show elevated ideation alongside heavy writing. Taken together, these patterns show that writing is common across the organization, while secondary use cases differ meaningfully by functional area. We observe that functional areas with a higher concentration of text based analysis and knowledge use also tend to have higher overall usage.

## 6.2 Comparison with Prior Use Case Evidence

Our use-case evidence aligns with an emerging consensus that writing is the leading workplace use of genAI (Bick et al. 2026; Chatterji et al. 2025; Counts et al. 2026; Handa et al. 2025). Although taxonomies differ, our other top use cases resemble common categories reported in other studies. For example, our second most prominent use case, Knowledge & Expertise, shares characteristics with searching for facts or information in Bick et al. (2026), seeking information in Chatterji et al. (2025), or information retrieval in Counts et al. (2026). As for variation in use, Chatterji et al. (2025) find that work-related Chat-GPT messages from management and business users are especially writing-heavy, while messages from computer-related users request more technical help. We find similar results as writing is prevalent at all levels and among all business users, although less so in Information Technology, where we find higher rates of technical assistance (which we label TOOL GUIDE). Despite examining a single organization, these comparisons suggest that inferences drawn from our sample are broadly consistent with evidence from surveys and analyses of specific platforms.

## 7 Sophistication of GenAI Use

We next report the paper’s primary analyses regarding sophisticated use. We first report summary statistics and then examine how sophisticated use varies over time, seniority, and functional area. Next, we examine how sophisticated use relates to training and easily observable behaviors, and finally we compare our findings to prior related evidence.

## 7.1 Sophisticated Use Summary Statistics

Table 3 summarizes LLM-assigned prompt input-output measures, binary prompt features, prompting strategies, and the sophisticated use measures CLARITYZ, ANY STRAT, and USE DIV.<sup>11</sup> The first six prompt input-output variables are scored 1 to 5, and all show meaningful variation in these assessed usage measures. The next three variables are binary measures, as follows. Explicit constraints (CONSTRAINTS) appear in 24.3% of conversations, but structured-output requests (STRUCT OUTPUT) appear in only 2.9% of conversations, and explicit acceptance criteria (ACCEPT CRITERIA) appear in only about 1.0% of conversations. The variable CLARITYZ is a composite constructed from these nine variables listed above it.

Deliberate prompting strategies are uncommon, as follows. Role prompting (ROLE PLAY) occurs in 4.4% of conversations. Few-shot examples (FEW SHOT), explicit chainof-thought requests (CHAIN OF THOUGHT), self-checking prompts (SELF CHECK), and interactive refinement (INTERACTIVE REFINE) each appear in less than 1% of conversations. The portion of conversations that use any of these strategies (ANY STRAT) is 5.2%.

The inputs to our use-case diversity measure (USE DIV) appear in Table 2. Other measured prompt characteristics describe the user’s interaction style, as follows. Users have a conversational tone that includes pleasantries (PLEASANTRIES) in 30.2% of conversations. They communicate casually with frequent typos (TYPOS) in 10.7% of conversations and use entirely lowercase writing (ALL LOWER) in 5.4% of conversations. Users explicitly express satisfaction with a response (SATISFIED) in 3.5% of conversations and upload or reference documents (DOC UPLOAD) in 5.5% of conversations. We report these interaction style measures solely as descriptive context.

## 7.2 Variation in Sophisticated Use

Figure 3 summarizes variation in our employee-month composite measures of sophistication: prompt clarity (CLARITYZ), deliberate strategy use (ANY STRAT), and use-case diversity (USE DIV). See Section 4.4 for a description of the presentation style we use for this figure. Panel A suggests some potential improvement in our sophisticated use measures over time, but we interpret this as a consequence of a weak January rather than a sustained upward trend. Panel B shows that more senior employees are more likely to use genAI for a broader set of use cases, with greater prompt clarity and a greater likelihood to utilize prompting strategies. Each composite measure rises with seniority: staff employees have the lowest values, managers have higher values, and above-manager employees have the highest values. Combined with the adoption patterns in Sections 5.1 and 5.2, these results suggest that more senior employees use genAI more frequently and with greater sophistication.

Panel C shows that sophisticated use also varies across functional areas with some interesting patterns. Specifically, Project Management is the only area where all three measures are significantly above the sample mean. Both Communications and Sales & Marketing have above-mean prompt clarity and strategy use, but below-mean use case diversity. Figure 2 also reports higher rates of writing and ideation use cases for these groups, tasks that may align especially well with the available LLMs. This alignment may contribute to both the relatively sophisticated interactions observed here and the abovemean use volume reported in Figure 1.

We observe that both Digital Innovation and Strategy are above the mean for prompt clarity and use-case diversity. These results may suggest the importance of attitudes: membership in the Digital Innovation group may reflect enthusiasm for new tools, while membership in the Strategy group may reflect comfort with uncertainty. In a similar vein, Information Technology has above-mean strategy use and use-case diversity but below-mean prompt clarity, suggesting a greater technical focus on the tool.

By contrast, Administrative Services, Property & Facilities, and Internal Operations are below the sample mean across the three sophisticated use measures. Accounting & Finance is also below the sample mean across all three measures, and is additionally distinctive because it is the only functional area significantly below the sample mean across all use-volume categories in Figure 1. This pattern could reflect differences in user characteristics, department leadership, or the possibility that the available LLM tools were less naturally useful for spreadsheet-intensive accounting and finance tasks.

## 7.3 Characteristics Associated with Sophisticated Use

Companies rarely preserve transcripts indefinitely, and processing them is costly even when they are available. Figure 4 therefore relates the three sophisticated use measures to usage characteristics that firms can observe from metadata alone, and to completion of formal AI training. We group these usage characteristics into four constructs. Ambition is the length of the user’s first prompt, on the logic that saying more indicates asking for more. Persistence is the amount of iteration within a conversation, which reflects a user refining a request rather than accepting the first response. Frequency is how often an employee uses aIQ Chat, which captures exposure to the tool rather than skill in using it. Flexibility is how often an employee also uses Copilot, the firm’s other genAI tool.

Without user fixed effects (Panel A), all three measures are higher when first prompts are longer (1ST PROMPT LEN) and conversations include more iteration (ROUNDS). The three measures are also higher with contemporaneous training (TRAIN MO), following cumulative past trainings (PAST TRAIN), and when Copilot usage is higher (COP DAYS). More frequent aIQ Chat use (CHAT CONVSand CHAT DAYS) accompanies greater prompt clarity and use-case diversity, but less strategy use.

In Panel B, we absorb time-invariant user traits by including user fixed effects. When we do this, we no longer see broad evidence that our measures of sophisticated use improve with Copilot use. Observing an effect would have suggested that prompting across multiple genAI platforms generates learning that promotes greater sophistication. Instead, the difference across Panels A and B suggests that those who demonstrate a capacity for more sophisticated use also choose to use multiple platforms. Likewise, the null (and sometimes weakly negative) result for past training suggests that the positive association in Panel A reflects selection into training by more skilled users, rather than training-induced gains in sophisticated use. Overall, we do not find evidence that completing training leads to lasting, observable improvement in a user’s sophistication. We do, however, continue to see consistently positive associations with longer first prompts, more iteration, and training completed in the current month.

Table 4 estimates multivariate associations by including multiple predictors simultaneously and comparing specifications with and without user fixed effects.<sup>12</sup> Across specifications, the same three variables remain consistently and positively associated with our sophisticated-use measures: longer first prompts, more iteration, and training completed in the current month (1ST PROMPT LEN, ROUNDS, and TRAIN MO, respectively). The contrast between current- and prior-month training (PAST TRAIN) is somewhat ambiguous: it may reflect a temporary boost while training is top-of-mind, or it may simply capture employees trying out the techniques during training.

Taken together, Figure 4 and Table 4 do not show reliably positive associations between our sophisticated use measures and either Copilot use or frequent use of aIQ Chat. The reported associations do suggest that the sophisticated use measures are most tied to ambition and persistence. Because first prompt length and iteration can be measured directly from conversation logs without prompting an LLM to classify conversation content, these measures may offer managers relatively low-cost signals of more sophisticated use.<sup>13</sup>

## 7.4 Comparison with Prior Evidence

To our knowledge, only a small number of studies use conversation data to examine dimensions related to the quality of genAI use. Two current working papers characterize interaction type or tool efficacy rather than variation in how skillfully users interact with genAI. Chatterji et al. (2025) classify user messages to ChatGPT as “asking,” “doing,” or “expressing.”<sup>14</sup> They find that asking messages are associated with greater apparent user satisfaction than doing or expressing messages under both an automated classifier and direct user feedback. Tomlinson et al. (2025) use LLM-assessed task completion and users’ thumbs-up/down feedback to evaluate Copilot’s efficacy across work activities. They find that Copilot performs best for communicating, teaching or explaining, and writing, and worse for image generation and data analysis.

By contrast, a report by Anthropic examines observable behaviors intended to capture genAI fluency within individual conversations (Swanson et al. 2026). Specifically, they measure eleven conversation-level behaviors, including clarifying goals, specifying output formats, providing examples, checking facts, and iteratively refining outputs. One of their principal findings parallels ours: conversations that include iteration and refine ment exhibit more of the other deliberate and evaluative behaviors. Our study differs in two important respects. First, Anthropic analyzes conversations from a single sevenday window, whereas we follow employees over eight months. This longer panel allows us to examine changes over time and, contrary to expectations, we find little evidence of sustained improvement. Second, Anthropic’s analysis does not link conversations to user or organizational characteristics. We link conversations to employee seniority, functional area, and training records, allowing us to examine differences across employees and changes within employees following formal training.

## 8 Discussion and Conclusion

In our study, we examine genAI use and sophistication among the back-office employees of a large firm that has already achieved widespread genAI adoption. Employees have access to similar tools and organizational support, allowing us to study variation within a relatively common environment. We document how genAI use and sophistication vary over time and across seniority levels and functional areas, how they relate to easily observable behaviors, and how they change following firm-provided AI training. We provide four primary insights, which we discuss here.

First, senior employees exhibit more sophisticated genAI use. This result is consistent with domain expertise complementing genAI capabilities, but several explanations are possible. Senior employees may have greater delegation proficiency and therefore be more practiced at specifying objectives, reviewing intermediate work, providing feedback, and pushing for the output they want. Alternatively, senior employees may also have stronger incentives to clarify requests and verify responses because they face greater accountability for the resulting work product. These proposed explanations each suggest that sophisticated genAI use may reflect broader professional experience and not just genAI expertise or familiarity.

Second, sophistication varies substantially across functional areas. Strategy, Digital Innovation, and Project Management exhibit the highest sophistication across our measures. These three groups share a focus on firmwide strategic initiatives and organizational change, which may provide employees with both a broad set of potential applications and work that benefits from clearly defining and iterating on tasks. By contrast, Accounting & Finance exhibits below-average sophistication across all three measures and is also the only functional area significantly below the sample mean across all reported usage measures. These group differences may reflect task mix, workflow fit, employee characteristics, department leadership, or attitudes toward genAI. For example, Choi and Xie (2026) document productivity and reporting-quality gains for accountants using genAI built into an accounting workflow. This suggests that a standalone chat interface may be less naturally suited to the spreadsheet-intensive and well-established processes common in Accounting & Finance than to the evolving and open-ended tasks associated with organizational change. Together, these results suggest that sophisticated use may depend not only on em ployee capabilities, but also on whether the work calls for the clear instructions, deliberate strategies, and broad application captured by our measures.

Next, we do not observe clear improvements in sophisticated use over the eightmonth sample period. While we find increased sophistication in the month when an employee completes formal AI training, we do not find increased sophistication in the months that follow. While our evidence does not establish that the training was ineffective, it provides little evidence that completion of formal training produces lasting changes in observed behavior. Together, the time and training results convey the difficulty organizations may face in producing sustained changes in sophisticated employee genAI use.

Finally, we find that longer initial prompts and more iteration within conversations accompany more sophisticated use. In our analyses, these two behaviors are more consistently associated with sophisticated use than usage frequency. While these relatively easy-to-track behaviors may provide useful signals when detailed transcript-based measures are unavailable, managers should not interpret any of these three behaviors as direct measures of performance.

Our analysis is descriptive and has important limitations. While our composite measures proxy for sophisticated use, we are not able to determine whether the LLM output satisfied the user’s goals. Further, we are not able to directly measure output quality or genAI’s influence on productivity. Next, in the time since our window concluded, the field has continued its rapid advance. Model developments have shifted best practices for prompting, and advancements in AI agents have likely changed the potential utility of genAI for use cases such as coding. Consequently, the rates of use and prevalence of particular behaviors reported in this paper may already differ from current practice.

Despite these limitations, companies are likely to benefit from employees who exhibit good AI behaviors and habits regardless of the existing AI technology. In addition, documenting evidence at a point in time is necessary to create reference points and historical benchmarks in a field that will continue to change. Further, while the paper’s descriptive design does not offer causal inference, it can motivate research designs that will. For these points, future research will be necessary to build knowledge of genAI’s entry into, and its effects on, the workplace.

## References

Arntz M, Baum M, Brüll E, Dorau R, Hartwig M, Matthes B, Meyer S.-C, Schlenker O, Tisch A, Wischniewski S (2025) Low Barriers, High Stakes: Formal and Informal Diffusion of AI in the Workplace. ifo Working Paper 422. Accessed June 8, 2026, https://www.ifo.de/DocDL/WP-2025-422\_Schlenker-etal\_AI-Diffusion-in-the-Workplace.pdf.

Bick A, Blandin A, Deming DJ (2026) The Rapid Adoption of Generative AI. Management Sci., ePub ahead of print, https://doi.org/10.1287/mnsc.2025.02523.

Bonney K, Breaux CL, Dinlersoz E, Foster LS, Haltiwanger JC, Pande AA (2026) The Microstructure of AI Diffusion: Evidence from Firms, Business Functions, and Worker Tasks. NBER Working Paper 35141, National Bureau of Economic Research.

Brown S (2025) Gen Z Leads in AI Adoption, Upskilling, but Training Gaps Persist. FM Magazine. Accessed July 22, 2026, https://www.fm-magazine.com/news/2025/oct/gen-z-leads-inai-adoption-upskilling-but-training-gaps-persist/.

Brynjolfsson E, Li D, Raymond L (2025) Generative AI at Work. Quart. J. Econom. 140(2):889–942.

Burton-Jones A, Grange C (2013) From Use to Effective Use: A Representation Theory Perspective. Inform. Systems Res. 24(3):632–658.

Cantrell S, Domergue C, Dake A, Murphy J, Sundholm T, Gustafson M (2026) AI Adoption to Adaptation: How a New Change Approach Can Build the Human Behaviors Needed for AI. Deloitte Insights. Accessed July 21, 2026, https://www.deloitte.com/us/en/insights/ topics/talent/ai-adoption-to-ai-adaptation.html.

Chatterji A, Cunningham T, Deming DJ, Hitzig Z, Ong C, Shan CY, Wadman K (2025) How People Use ChatGPT. National Bureau of Economic Research Working Paper 34255. Accessed Feb. 25, 2026, https://www.nber.org/papers/w34255.

Choi JH, Xie CL (2026) Human + AI in Accounting: Early Evidence from the Field. J. Accounting Res. 64(3):1333–1373.

Counts S, Chen Y, Dong J, Sharma H, Zaikin A, Hu R, Kok A, Ozer Yilmaz G, Suri S, Tomlinson K, Jaffe S, Wang W (2026) AI in the Enterprise: How People Use M365 Copilot Chat. arXiv: 2605.23958 (cs.CY). Accessed June 8, 2026, https://arxiv.org/abs/2605.23958.

Croom J, Gale B, Grant SM (2026) Disclosure Presentation Attributes, Generative AI, and Investor Judgments. Accessed Aug. 5, 2026, https://ssrn.com/abstract=5040309.

Dillon EW, Jaffe S, Peng S, Cambon A (2025) Early Impacts of M365 Copilot. arXiv: 2504.11443 (econ.GN). Accessed July 22, 2026, https://arxiv.org/abs/2504.11443.

Estep C, Griffith EE, MacKenzie NL (2024) How Do Financial Executives Respond to the Use of Artificial Intelligence in Financial Reporting and Auditing? Rev. Accounting Stud. 29(3):2798– 2831.

Eulerich M, Sanatizadeh A, Vakilzadeh H, Wood DA (2024) Is It All Hype? ChatGPT’s Performance and Disruptive Potential in the Accounting and Auditing Industries. Rev. Accounting Stud. 29(3):2318–2349.

Gambacorta L, Qiu H, Shan S, Rees DM (2024) Generative AI and Labour Productivity: A Field Experiment on Coding. BIS Working Papers 1208, Bank for International Settlements. Accessed July 22, 2026, https://www.bis.org/publ/work1208.htm.

Handa K, Tamkin A, McCain M, Huang S, Durmus E, Heck S, Mueller J, Hong J, Ritchie S, Belonax T, Troy KK, Amodei D, Kaplan J, Clark J, Ganguli D (2025) Which Economic Tasks Are Performed with AI? Evidence from Millions of Claude Conversations. arXiv: 2503.04761 (cs.CY). Accessed June 8, 2026, https://arxiv.org/abs/2503.04761.

Henseke G (2026) From Exposure to Adoption: Generative AI in European Workplaces. arXiv: 2604.18849 (econ.GN). Accessed June 8, 2026, https://arxiv.org/abs/2604.18849.

Humlum A, Vestergaard E (2025) The Unequal Adoption of ChatGPT Exacerbates Existing Inequalities among Workers. Proc. Natl. Acad. Sci. USA 122(1):e2414972121.

Korst J, Puntoni S, Tambe P (2025) Accountable Acceleration: Gen AI Fast-Tracks into the Enterprise. Wharton Human–AI Research and GBK Collective. Accessed July 22, 2026, https: //knowledge.wharton.upenn.edu/special-report/2025-ai-adoption-report/.

Ling Y, Kale A, Imas A (2026) Underreporting of AI Use: The Role of Social Desirability Bias. Proceedings of the 2026 CHI Conference on Human Factors in Computing Systems. Accessed July 27, 2026, https://doi.org/10.1145/3772318.3791073.

Lou B, Wu L (2021) AI on Drugs: Can Artificial Intelligence Accelerate Drug Development? Evidence from a Large-Scale Examination of Bio-Pharma Firms. MIS Quart. 45(3):1451–1482.

Orlikowski WJ (2000) Using Technology and Constituting Structures: A Practice Lens for Studying Technology in Organizations. Organ. Sci. 11(4):404–428.

Pohle A, Fernandez R (2026) The First Class of AI Natives Is Graduating. Offices Are Getting Ready. The Wall Street Journal. Accessed July 30, 2026, https://www.wsj.com/tech/ai/ainatives-graduates-job-cuts-6bab8ac9.

Schulhoff S, Ilie M, Balepur N, Kahadze K, Liu A, Si C, Li Y, Gupta A, Han H, Schulhoff S, Dulepet PS, Vidyadhara S, Ki D, Agrawal S, Pham C, Kroiz G, Li F, Tao H, Srivastava A, Da Costa H, Gupta S, Rogers ML, Goncearenco I, Sarli G, Galynker I, Peskoff D, Carpuat M, White J, Anadkat S, Hoyle A, Resnik P (2024) The Prompt Report: A Systematic Survey of Prompt Engineering Techniques. arXiv: 2406.06608 (cs.CL). Accessed June 8, 2026, https://arxiv. org/abs/2406.06608.

Shaffer M, Wang CCY (2026) Harnessing Large Language Models for Core Earnings Measurement. Accessed Aug. 5, 2026, https://ssrn.com/abstract=4979501.

Swanson K, Bent D, Ludwig Z, Dakan R, Feller J (2026) Anthropic Education Report: The AI Fluency Index. Accessed July 22, 2026, https://www.anthropic.com/news/anthropiceducation-report-the-ai-fluency-index.

Tomlinson K, Jaffe S, Wang W, Counts S, Suri S (2025) Working with AI: Measuring the Applicability of Generative AI to Occupations. arXiv: 2507.07935 (cs.AI). Accessed July 8, 2026, https://arxiv.org/abs/2507.07935.

Tully T, Redfern J, Das D, Xiao D (2025) 2025: The State of Generative AI in the Enterprise. Menlo Ventures. Accessed July 21, 2026, https://menlovc.com/perspective/2025-the-stateof-generative-ai-in-the-enterprise/.

Wang W, Gao G, Agarwal R (2024) Friend or Foe? Teaming Between Artificial Intelligence and Workers with Variation in Experience. Management Sci. 70(9):5753–5775.

Wang X, Wei J, Schuurmans D, Le Q, Chi E, Narang S, Chowdhery A, Zhou D (2023) Self-Consistency Improves Chain of Thought Reasoning in Language Models. arXiv: 2203.11171 (cs). Accessed Feb. 25, 2026, http://arxiv.org/abs/2203.11171.

Wei J, Wang X, Schuurmans D, Bosma M, Ichter B, Xia F, Chi E, Le Q, Zhou D (2023) Chain of-Thought Prompting Elicits Reasoning in Large Language Models. arXiv: 2201.11903 (cs). Accessed Feb. 25, 2026, http://arxiv.org/abs/2201.11903.

Fi<sub>g</sub>ure 1 : Usa<sub>g</sub>e and Intensit<sub>y</sub>  
![](images/ff907c45314ffa26298596571525645490dfc8cd5f49ebf97de9289107863cfc.jpg)

Panel B: B<sub>y</sub> Seniorit<sub>y</sub>  
![](images/0673d373523321c3d98b1f7d0d2b4887d58b114fe6f1ab1edca26f0672d569d0.jpg)

Fi<sub>g</sub>ure 1 : Usa<sub>g</sub>e and Intensit<sub>y</sub> (continued)

Panel C: B<sub>y</sub> Functional Area

![](images/2ff5db9cbb0f957a5c7f839b814b735d8b6cfa7fa7e5bf87d9beba266db99bfa.jpg)  
Notes: Each <sub>p</sub>anel <sub>p</sub>lots an em<sub>p</sub>lo<sub>y</sub>ee-month usa<sub>g</sub>e measure (columns) for the <sub>g</sub>rou<sub>p</sub>s listed—months in Panel $\mathsf { A } ,$ <sub>sen</sub>i<sub>or</sub>it<sub>y</sub> l<sub>eve</sub>l<sub>s</sub> i<sub>n</sub> P<sub>ane</sub>l $\mathrm { \Delta B , }$ <sub>an</sub>d functional areas in Panel C <sub>.</sub> Each dot marks a <sub>g</sub>rou<sub>p</sub> mean <sub>p</sub>ositioned relative to the overall sam<sub>p</sub>le mean (the vertical reference line) <sub>;</sub> horizontal-axis ti<sub>c</sub>k<sub>s</sub> <sub>repor</sub>t th<sub>e</sub> <sub>samp</sub>l<sub>e</sub> <sub>mean</sub> <sub>a</sub>t th<sub>e</sub> <sub>cen</sub>t<sub>er</sub> <sub>an</sub>d th<sub>e</sub> <sub>group</sub> <sub>m</sub>i<sub>n</sub>i<sub>mum</sub> <sub>an</sub>d <sub>max</sub>i<sub>mum</sub> <sub>a</sub>t th<sub>e</sub> <sub>en</sub>d<sub>s</sub> <sub>so</sub> <sub>a</sub> d<sub>o</sub>t<sup>’</sup><sub>s</sub> h<sub>or</sub>i<sub>zon</sub>t<sub>a</sub>l <sub>pos</sub>iti<sub>on</sub> <sub>s</sub>h<sub>ows</sub> h<sub>ow</sub> f<sub>ar</sub> th<sub>e</sub> <sub>group</sub> li<sub>es</sub> <sub>a</sub>b<sub>ove</sub> <sub>or</sub> b<sub>e</sub>l<sub>ow</sub> th<sub>e</sub> <sub>samp</sub>l<sub>e</sub> <sub>average .</sub> A d<sub>o</sub>t<sup>’</sup><sub>s</sub> <sub>co</sub>l<sub>or</sub> <sub>an</sub>d fill <sub>summar</sub>i<sub>ze</sub> th<sub>e</sub> <sub>group</sub>-b<sub>y</sub>-<sub>group</sub> diff<sub>erence</sub>-f<sub>rom</sub>-<sub>res</sub>t t<sub>es</sub>t <sub>:</sub> <sub>a</sub> fill<sub>e</sub>d b<sub>urn</sub>t-<sub>orange</sub> d<sub>o</sub>t d<sub>eno</sub>t<sub>es a group mean s</sub>i<sub>gn</sub>ifi<sub>can</sub>tl<sub>y a</sub>b<sub>ove</sub> th<sub>e samp</sub>l<sub>e mean an open</sub> b<sub>urn</sub>t-<sub>orange</sub> d<sub>o</sub>t <sub>a va</sub>l<sub>ue a</sub>b<sub>ove</sub> th<sub>e mean</sub> th<sub>a</sub>t i<sub>s no</sub>t <sub>s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>ca</sub>ll<sub>y s</sub>i<sub>gn</sub>ifi<sub>can</sub>t <sub>a</sub> fill<sub>e</sub>d <sub>gray</sub> d<sub>o</sub>t <sub>a</sub> <sub>va</sub>l<sub>ue</sub> <sub>s</sub>i<sub>gn</sub>ifi<sub>can</sub>tl<sub>y</sub> b<sub>e</sub>l<sub>ow</sub> th<sub>e</sub> <sub>mean,</sub> <sub>an</sub>d <sub>an</sub> <sub>open</sub> <sub>gray</sub> d<sub>o</sub>t <sub>a</sub> <sub>va</sub>l<sub>ue</sub> b<sub>e</sub>l<sub>ow</sub> th<sub>e</sub> <sub>mean</sub> th<sub>a</sub>t i<sub>s</sub> <sub>no</sub>t <sub>s</sub>i<sub>gn</sub>ifi<sub>can</sub>t<sub>.</sub> U<sub>n</sub>d<sub>er</sub>l<sub>y</sub>i<sub>ng</sub> <sub>va</sub>l<sub>ues,</sub> <sub>samp</sub>l<sub>e</sub> <sub>s</sub>i<sub>zes an</sub>d t<sub>wo</sub>-<sub>s</sub>id<sub>e</sub>d <sub>p</sub>-<sub>va</sub>l<sub>ues are repor</sub>t<sub>e</sub>d i<sub>n</sub> O<sub>n</sub>li<sub>ne</sub> A<sub>ppen</sub>di<sub>x</sub> T<sub>a</sub>bl<sub>e</sub> B 1 <sub>.</sub> V<sub>ar</sub>i<sub>a</sub>bl<sub>e</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>ons are prov</sub>id<sub>e</sub>d i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> A<sub>.</sub>

Panel A: B<sub>y</sub> Month  
Fi<sub>g</sub>ure 2 : Use Cases b<sub>y</sub> Grou<sub>p</sub>  
![](images/9d8e6e4cd916cfc57577e1f5129ddb2aad68c59da381cbce0bca8ec2dd8c37d3.jpg)

Panel B: B<sub>y</sub> Seniorit<sub>y</sub>  
![](images/7f6874828abcb3ea15f61d721e3bd8a4d2de344f87e4bc6cc92d3e06922118d6.jpg)

Fi<sub>g</sub>ure 2 : Use Cases b<sub>y</sub> Grou<sub>p</sub> (continued)

Panel C: B<sub>y</sub> Functional Area

![](images/f8d400e143e0a63072e462885b0984e685fad3032946389e7c38a78e9f1aece0.jpg)  
Notes: Each <sub>p</sub>anel <sub>p</sub>lots the conversation-level share assi<sub>g</sub>ned to each broad use-case cate<sub>g</sub>or<sub>y</sub> (columns) for the <sub>g</sub>rou<sub>p</sub>s listed—months in Panel A <sub>sen</sub>i<sub>or</sub>it<sub>y</sub> l<sub>eve</sub>l<sub>s</sub> i<sub>n</sub> P<sub>ane</sub>l B <sub>an</sub>d f<sub>unc</sub>ti<sub>ona</sub>l <sub>areas</sub> i<sub>n</sub> P<sub>ane</sub>l C <sub>.</sub> E<sub>ac</sub>h d<sub>o</sub>t <sub>mar</sub>k<sub>s</sub> <sub>a</sub> <sub>group</sub><sup>’</sup><sub>s</sub> <sub>ca</sub>t<sub>egory</sub> <sub>s</sub>h<sub>are</sub> <sub>pos</sub>iti<sub>one</sub>d <sub>re</sub>l<sub>a</sub>ti<sub>ve</sub> t<sub>o</sub> th<sub>e</sub> <sub>overa</sub>ll <sub>samp</sub>l<sub>e</sub> <sub>mean</sub> (the vertical reference line) <sub>;</sub> horizontal-axis ticks re<sub>p</sub>ort the sam<sub>p</sub>le mean at the center and the <sub>g</sub>rou<sub>p</sub> minimum and maximum at the ends <sub>.</sub> Because <sub>a</sub> <sub>conversa</sub>ti<sub>on</sub> <sub>may</sub> <sub>rece</sub>i<sub>ve</sub> <sub>mu</sub>lti<sub>p</sub>l<sub>e</sub> l<sub>a</sub>b<sub>e</sub>l<sub>s,</sub> <sub>propor</sub>ti<sub>ons</sub> d<sub>o</sub> <sub>no</sub>t <sub>sum</sub> t<sub>o</sub> <sub>one .</sub> C<sub>o</sub>l<sub>or</sub> <sub>an</sub>d fill f<sub>o</sub>ll<sub>ow</sub> th<sub>e</sub> <sub>same</sub> <sub>conven</sub>ti<sub>on</sub> <sub>as</sub> Fi<sub>gure</sub> 1 <sub>:</sub> fill<sub>e</sub>d b<sub>urn</sub>t-<sub>orange</sub> (o<sub>p</sub>en burnt-oran<sub>g</sub>e) marks values si<sub>g</sub>nificantl<sub>y</sub> (not si<sub>g</sub>nificantl<sub>y</sub>) above the sam<sub>p</sub>le mean<sub>,</sub> and filled <sub>g</sub>ra<sub>y</sub> (o<sub>p</sub>en <sub>g</sub>ra<sub>y</sub>) marks values si<sub>g</sub>nificantl<sub>y</sub> (not si<sub>g</sub>nificantl<sub>y</sub>) below it Underl<sub>y</sub>in<sub>g p</sub>ro<sub>p</sub>ortions sam<sub>p</sub>le sizes and two-sided <sub>p</sub>-values are re<sub>p</sub>orted in Online A<sub>pp</sub>endix Table B2 Variable definitions <sub>are prov</sub>id<sub>e</sub>d i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> A<sub>.</sub>

Figure 3: Evaluation of Use  
Panel A: By Month  
![](images/d3bf151adacc561e9453475ea0b986d2a8b7603d4f2c4d87d3fef772142fa10d.jpg)

Panel B: By Seniority  
![](images/5070686b1f4021752e13fa21c0046aeae472f474d669717236d4a95352b8464c.jpg)

Figure 3: Evaluation of Use (continued)

Panel C: By Functional Area

![](images/b454e208bed001004e8fc7195be47a849cb9ec1c5dbb5dbff3f1ea549ed9d785.jpg)  
Notes: Each panel plots employee-month composite measures of sophisticated use—prompt clarity (CLARI-TYZ), deliberate strategy use (ANY STRAT), and use-case diversity (USE DIV)—for the groups listed (months in Panel A, seniority levels in Panel B, functional areas in Panel C), computed over active aIQ Chat employeemonths. Each dot marks a group mean positioned relative to the overall sample mean (the vertical reference line); horizontal-axis ticks report the sample mean at the center and the group minimum and maximum at the ends. Color and fill follow the same convention as Figure 1: filled burnt-orange (open burnt-orange) marks values significantly (not significantly) above the sample mean, and filled gray (open gray) marks values significantly (not significantly) below it. Underlying values, sample sizes, and two-sided p-values are reported in Online Appendix Table B3. Variable definitions are provided in Appendix A.

Figure 4: Bivariate Associations  
Panel A: Without User Fixed Effects  
![](images/058cf82c6080331fd68ffaafd78bff0462be3a582a124691b02ada81f43e6c32.jpg)

Figure 4: Bivariate Associations (continued)  
Panel B: With User Fixed Effects  
![](images/937dfc7df06a1d6e75c9f2659677e830a45495cf8f00fe28cc84cfe0fd94d5d3.jpg)  
Notes: Each panel plots bivariate (pairwise) regression coefficients relating each predictor (rows) to the three sophisticated-use composites: prompt clarity (CLARITYZ), deliberate strategy use (ANY STRAT), and use-case diversity (USE DIV). Panel A includes month fixed effects only; Panel B adds user fixed effects. Predictor rows are grouped by construct—ambition, persistence, frequency, flexibility, and training—with alternating blocks shaded, matching Table 4 and Online Appendix Table B4. Each dot is positioned at the estimated coefficient, with the vertical reference line at zero. Burnt-orange dots denote positive coefficients and gray dots denote negative coefficients; filled dots are statistically significant at the 5% level (two-sided) and open dots are not. Aside from 1ST PROMPT LEN and the threshold indicators, each predictor enters as log(1 + x); row labels omit the transform. For legibility, the figure displays a focused subset of predictors; the complete set of estimates, sample sizes, and p-values is reported in Online Appendix Table B4. Variable definitions are provided in Appendix A.

Table 1: Sample Construction and Scale  
Panel A: Employees
<table><tr><td>Metric</td><td>Count</td></tr><tr><td>Total back-office employees Employees with aIQ Chat use</td><td>5,191</td></tr><tr><td>Employees in Copilot dataset</td><td>3,925 4,392</td></tr><tr><td>Employees with Copilot use</td><td>3,962</td></tr><tr><td>Employees who used BOTH tools</td><td>2,861</td></tr></table>

Panel B: Conversations
<table><tr><td>Metric</td><td>Value</td></tr><tr><td>Number of conversations Number of submitted user prompts</td><td>158,496 713,564</td></tr><tr><td>Mean user prompts per conversation</td><td>4.50</td></tr><tr><td>Median user prompts per conversation</td><td>2.00</td></tr><tr><td>Number of active employees (with LLM use)</td><td>3,925</td></tr><tr><td>Mean conversations per active employee</td><td>40.38</td></tr><tr><td>Median conversations per active employee Share of conversations by top decile of users</td><td>14.00</td></tr></table>

Notes: Panel A reports employee counts constructed from employeemonth administrative records aggregated to the employee level over the study window. Except for the total count, each Panel A metric requires at least one month of the noted occurrence. Panel B reports descriptive statistics computed from conversation-level logs. The data cover January through August 2025.

Panel A: Aggregate Categories
<table><tr><td>Category</td><td>N Pct of Convs</td></tr><tr><td>WRITING 116,216</td><td>0.733</td></tr><tr><td>KNOWLEDGE</td><td>36,458 0.230</td></tr><tr><td>TEXT ANALYSIS 17,008</td><td>0.107</td></tr><tr><td>TOOL GUIDE</td><td>15,634 0.099</td></tr><tr><td>CODING DATA</td><td>11,556 0.073</td></tr><tr><td>PERSONAL</td><td>3,026 0.019</td></tr><tr><td>IDEATION</td><td>2,910 0.018</td></tr><tr><td>OTHER</td><td>1,930 0.012</td></tr></table>

Panel B: Subcategories
<table><tr><td>Category</td><td>Subcategory</td><td>N</td><td>Pct of Convs</td></tr><tr><td>WRITING</td><td>Edit/improve existing content</td><td>72,596</td><td>0.458</td></tr><tr><td rowspan="7">CODING DATA</td><td>Generate new content</td><td>55,125</td><td>0.348</td></tr><tr><td>Language translation</td><td>1,549</td><td>0.010</td></tr><tr><td>Other writing tasks</td><td>885</td><td>0.006</td></tr><tr><td>Generate or debug programming code</td><td>8,546</td><td>0.054</td></tr><tr><td>Analyze user-provided data</td><td>1,520</td><td>0.010</td></tr><tr><td>Data cleaning / restructuring / standardization</td><td>1,046</td><td>0.007</td></tr><tr><td>Other coding/data tasks</td><td>628</td><td>0.004</td></tr><tr><td rowspan="2">TEXT ANALYSIS</td><td>Document understanding (text-based analysis)</td><td>16,524</td><td>0.104</td></tr><tr><td>Other text-based analysis requests</td><td>488</td><td>0.003</td></tr><tr><td>KNOWLEDGE</td><td>Business management (back office)</td><td>14,432</td><td>0.091</td></tr><tr><td rowspan="10">TOOL GUIDE</td><td>Other knowledge queries</td><td>9,777</td><td>0.062</td></tr><tr><td>Business operations (front office)</td><td>8,847</td><td>0.056</td></tr><tr><td>Legal / regulation / compliance</td><td>4,050</td><td>0.026</td></tr><tr><td>Accounting standards and guidance</td><td>1,255</td><td>0.008</td></tr><tr><td>Other software (Sheets, Tableau, etc.)</td><td>9,812</td><td>0.062</td></tr><tr><td>Microsoft Excel</td><td>3,540</td><td>0.022</td></tr><tr><td>AI/LLM tools</td><td>1,037</td><td>0.007</td></tr><tr><td>Microsoft Outlook</td><td>802</td><td>0.005</td></tr><tr><td>Microsoft PowerPoint</td><td>531</td><td>0.003</td></tr><tr><td>Alteryx</td><td>428</td><td>0.003</td></tr><tr><td>IDEATION</td><td>Microsoft Word</td><td>391</td><td>0.002</td></tr><tr><td></td><td>Brainstorming and ideation</td><td>2,822</td><td>0.018</td></tr><tr><td></td><td>Other creative thinking requests</td><td>88</td><td>0.001</td></tr><tr><td>PERSONAL</td><td>Personal tasks</td><td>2,075</td><td>0.013</td></tr><tr><td></td><td>Other non-work activities</td><td>959</td><td>0.006</td></tr><tr><td rowspan="3">OTHER</td><td>Other / fits nowhere else</td><td>1,315</td><td>0.008</td></tr><tr><td>Unclear or ambiguous use case</td><td>474</td><td>0.003</td></tr><tr><td>Testing LLM capabilities</td><td>141</td><td>0.001</td></tr></table>

Notes: Panel A reports the distribution of conversations across broad use-case categories, and Panel B decomposes these categories into subcategories. Conversations may be assigned multiple use-case labels, so category counts and proportions do not sum to one. N is the number of conversations assigned to the category, and “Pct of Convs” is the proportion of all conversations in the sample assigned to the category. Variable definitions are provided in Appendix A.

Table 3: Style and Technique Measures
<table><tr><td></td><td colspan="2">N = 158,496</td><td colspan="2">Conversations Employee-Months N = 17,671</td></tr><tr><td>Variable</td><td>Mean</td><td>Std. Dev.</td><td>Mean</td><td>Std. Dev.</td></tr><tr><td>Prompt Input-Output Measures: 1ST PROMPT LANG COMPLEXITY</td><td>2.586</td><td></td><td>0.724 2.474</td><td>0.442</td></tr><tr><td>1ST PROMPT COMPLEXITY</td><td>2.305</td><td>0.665</td><td>2.272</td><td>0.372</td></tr><tr><td>STRUCTURE</td><td>1.738</td><td>0.734</td><td>1.629</td><td>0.404</td></tr><tr><td>PROMPT SOPH</td><td>2.294</td><td>0.693</td><td>2.229</td><td>0.409</td></tr><tr><td>SPECIFICITY</td><td>3.251</td><td>0.689</td><td>3.187</td><td>0.426</td></tr><tr><td>FORMAT CLARITY</td><td>1.812</td><td>0.841</td><td>1.753</td><td>0.477</td></tr><tr><td>CONSTRAINTS</td><td>0.243</td><td>0.429</td><td>0.238</td><td>0.299</td></tr><tr><td>STRUCT OUTPUT</td><td>0.029</td><td>0.167</td><td>0.027</td><td>0.117</td></tr><tr><td>ACCEPT CRITERIA</td><td></td><td>0.100</td><td>0.006</td><td>0.055</td></tr><tr><td>CLARITYZ</td><td>0.010 0.000</td><td>1.000</td><td>0.000</td><td>1.000</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Strategy Measures</td><td></td><td></td><td></td><td></td></tr><tr><td>ROLE PLAY</td><td>0.044</td><td>0.205</td><td>0.050</td><td>0.171</td></tr><tr><td>FEW SHOT</td><td>0.003</td><td>0.057</td><td>0.003 0.001</td><td>0.035</td></tr><tr><td>CHAIN OF THOUGHT SELF CHECK</td><td>0.001 0.003</td><td>0.038 0.054</td><td>0.002</td><td>0.021 0.028</td></tr><tr><td></td><td></td><td></td><td>0.010</td><td>0.074</td></tr><tr><td>INTERACTIVE REFINE ANY STRAT</td><td>0.005 0.052</td><td>0.070 0.221</td><td>0.058</td><td>0.183</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Other Prompt Measures</td><td></td><td></td><td></td><td></td></tr><tr><td>USE DIV</td><td></td><td>N/A</td><td>0.316</td><td>0.222</td></tr><tr><td>DOC UPLOAD</td><td>0.055</td><td>0.227</td><td>0.058</td><td>0.165</td></tr><tr><td>SATISFIED</td><td>0.035</td><td>0.184</td><td>0.011</td><td>0.081</td></tr><tr><td>TYPOS</td><td>0.107</td><td>0.308</td><td>0.089</td><td>0.195</td></tr><tr><td>PLEASANTRIES</td><td>0.302</td><td>0.459</td><td>0.093</td><td>0.226</td></tr><tr><td>ALL LOWER</td><td>0.054</td><td>0.227</td><td>0.073</td><td>0.187</td></tr></table>

Notes: The table reports means and standard deviations for prompt input-output measures and strategy measures at the conversation level and for active employee-months. Active employee-months are months with at least one aIQ Chat conversation. The final row in “Prompt Input-Output Measures” reports CLARITYZ, a composite constructed from the measures listed above it. “Strategy Measures” are summarized by ANY STRAT. Although the table also reports conversation-level values for CLARITYZ and ANY STRAT, all three composite variables are defined at the employee-month level in Appendix A. We do not measure USE DIV at the conversation level, as indicated by N/A.

Table 4: Multivariate Regressions
<table><tr><td>Variable</td><td>CLARITYZ</td><td></td><td></td><td>CLARITYZ ANY STRAT ANY STRAT</td><td>USE DIV</td><td>USE DIV</td></tr><tr><td>1ST PROMPT LEN</td><td>0.000*** (0.000)</td><td>0.000*** (0.000)</td><td>0.000*** (0.000)</td><td>0.000** (0.000)</td><td>0.000*** (0.000)</td><td>0.000*** (0.000)</td></tr><tr><td>ROUNDS</td><td>0.259*** (0.020)</td><td>0.257*** (0.022)</td><td>0.042*** (0.005)</td><td>0.044*** (0.005)</td><td>0.014*** (0.005)</td><td>0.023*** (0.005)</td></tr><tr><td>CHAT DAYS</td><td>0.005 (0.017)</td><td>0.015 (0.019)</td><td>-0.043*** (0.004)</td><td>-0.018*** (0.004)</td><td>0.103*** (0.004)</td><td>0.144*** (0.005)</td></tr><tr><td>COP DAYS</td><td>0.106*** (0.012)</td><td>-0.016 (0.016)</td><td>0.013*** (0.002)</td><td>0.003 (0.003)</td><td>0.019*** (0.003)</td><td>0.005 (0.004)</td></tr><tr><td>TRAIN MO</td><td>0.046** (0.019)</td><td>0.051*** (0.019)</td><td>0.049*** (0.004)</td><td>0.037*** (0.004)</td><td>0.034*** (0.004)</td><td>0.028*** (0.004)</td></tr><tr><td>PAST TRAIN</td><td>0.099*** (0.020)</td><td>0.002 (0.025)</td><td>0.013*** (0.004)</td><td>-0.007 (0.006)</td><td>0.016*** (0.004)</td><td>0.003 (0.006)</td></tr><tr><td>ALL LOWER</td><td>-1.310*** (0.050)</td><td>-1.084*** (0.067)</td><td>-0.080*** (0.007)</td><td>-0.064*** (0.014)</td><td>-0.117*** (0.010)</td><td>-0.116*** (0.014)</td></tr><tr><td>TYPOS</td><td>-0.117 (0.086) 0.653***</td><td>-0.007 (0.098)</td><td>-0.069*** (0.019)</td><td>-0.042** (0.019)</td><td>-0.050*** (0.016)</td><td>-0.047*** (0.017)</td></tr><tr><td>DOC UPLOAD</td><td>(0.106) 0.398***</td><td>0.744*** (0.118)</td><td>0.009 (0.023)</td><td>-0.004 (0.023)</td><td>0.042** (0.018)</td><td>0.045** (0.021)</td></tr><tr><td>PLEASANTRIES SATISFIED</td><td>(0.055) 0.064</td><td>0.363*** (0.058) -0.179</td><td>-0.011 (0.013) 0.013</td><td>-0.013 (0.012) -0.035</td><td>-0.116*** (0.009) 0.016</td><td>-0.083*** (0.011) 0.013</td></tr><tr><td></td><td>(0.134)</td><td>(0.145)</td><td>(0.036)</td><td>(0.040)</td><td>(0.026)</td><td>(0.030)</td></tr><tr><td>Month FE</td><td>Yes No</td><td>Yes</td><td>Yes</td><td>Yes</td><td>Yes</td><td>Yes</td></tr><tr><td>User FE N</td><td>12,902</td><td>Yes 12,902</td><td>No 12,902</td><td>Yes 12,902</td><td>No 12,902</td><td>Yes 12,902</td></tr></table>

Notes: The table reports multivariate associations between the dependent measures and a set of predic tors using active employee-month observations with month fixed effects. Columns compare specifications without and with user fixed effects. Coefficients are estimated by OLS after residualizing variables with respect to the included fixed effects, and White robust standard errors are clustered at the user level and reported in parentheses. Statistical significance is indicated by stars appended to coefficients: ∗∗∗ p < 0.01, ∗∗ p < 0.05, and ∗ p < 0.10 (two-sided tests). The N row reports the number of employee-month observations used in each regression. Variable definitions are provided in Appendix A.

<table><tr><td colspan="2">Sophisticated use measures</td></tr><tr><td>CLARITYZ</td><td>Composite (z-scored) measure of prompt clarity for employee i in month t. Constructed by first z-scoring, within the sample of active employee- months, the employee-month averages of six LLM-assigned continuous prompt-quality ratings (1ST PROMPT LANG COMPLEXITY, 1ST PROMPT COMPLEXITY, STRUCTURE, PROMPT SOPH, SPECIFICITY, and FORMAT CLARITY) and three LLM-assigned binary prompt features (CONSTRAINTS, STRUCT OUTPUT, and ACCEPT CRITERIA). The composite is the equally weighted average of these nine z-scores and is then z-scored again within active employee-months. Higher values indicate clearer, more structured prompts.</td></tr><tr><td>ANY STRAT</td><td>Composite measure of deliberate prompting strategy use for employee i in month t. Constructed as the proportion of the employee-month&#x27;s conversa- tions in which at least one of five prompting strategies equals one (ROLE PLAY, FEW SHOT, CHAIN OF THOUGHT, SELF CHECK, or INTERACTIVE RE- FINE).</td></tr><tr><td>USE DIV</td><td>Measure of how broadly an employee&#x27;s LLM conversations span different use-case categories in a given month. We compute it as a normalized Shannon-entropy index across that employee-month&#x27;s totals for the eight broad use-case categories listed below, counting each conversation once in every broad category to which it is assigned: for each category, we convert its monthly total into a share of the employee&#x27;s total across all categories, then compute entropy and normalize it to range from 0 to 1. The measure equals 0 when all activity falls in a single category (or when the employee- month has a total of zero across categories) and approaches 1 when activity is evenly spread across categories.</td></tr><tr><td>Use-case categories</td><td></td></tr><tr><td>WRITING</td><td>Indicator equal to 1 if the conversation includes writing or communication tasks (e.g., generating new content, editing user-provided text, translation, or other writing tasks), and 0 otherwise.</td></tr><tr><td>CODING DATA</td><td>Indicator equal to 1 if the conversation includes coding or data analysis tasks (e.g., generating/debugging code, analyzing user-provided data, or cleaning/restructuring data), and 0 otherwise.</td></tr><tr><td>TEXT ANALYSIS</td><td>Indicator equal to 1 if the conversation includes text-based analysis or document understanding (e.g., summarizing or interpreting user-provided text/documents), and 0 otherwise.</td></tr></table>

Continued on next page

Continued from previous page
<table><tr><td>KNOWLEDGE</td><td>Indicator equal to 1 if the conversation seeks domain knowledge or exper- tise (e.g., accounting, legal/regulatory, business operations/management, or other knowledge queries), and 0 otherwise.</td></tr><tr><td>TOOL GUIDE</td><td>Indicator equal to 1 if the conversation requests guidance on how to use a specific software/tool (e.g., Excel, Word, PowerPoint, Outlook, AI/LLM tools), and 0 otherwise</td></tr><tr><td>IDEATION</td><td>Indicator equal to 1 if the conversation involves brainstorming or ideation (generating new ideas), and 0 otherwise.</td></tr><tr><td>PERSONAL</td><td>Indicator equal to 1 if the conversation is primarily non-work/personal (e.g., personal planning, home projects), and 0 otherwise.</td></tr><tr><td>OTHER</td><td>Indicator equal to 1 if the conversation is classified as Other in the use- case taxonomy (including testing LLM capabilities or unclear/ambiguous use cases), and 0 otherwise.</td></tr><tr><td colspan="2">Predictor variables (described at employee-month level)</td></tr><tr><td>1ST PROMPT LEN</td><td>Employee-month average number of characters in the first substantive user prompt of each conversation.</td></tr><tr><td>ROUNDS</td><td>Natural logarithm of 1+ the average number of user turns per conversation for employee i in month t.</td></tr><tr><td>CHAT CONVS</td><td>Natural logarithm of 1+ the number of aIQ Chat conversations for employee i in month t.</td></tr><tr><td>CHAT DAYS</td><td>Natural logarithm of 1+ the number of days in month t with any aIQ Chat usage.</td></tr><tr><td>COP DAYS</td><td>Natural logarithm of 1+ the number of days in month t with any Copilot usage.</td></tr><tr><td>TRAIN MO</td><td>Natural logarithm of 1+ the number of training courses completed by em- ployee i in month t.</td></tr><tr><td>PAST TRAIN</td><td>Natural logarithm of 1+ the cumulative number of training courses com- pleted by employee i through the beginning of month t.</td></tr><tr><td colspan="2">Threshold indicators (described at employee-month level)</td></tr><tr><td>ROUNDS ≥ k</td><td>Indicator equal to 1 if the employee-month average number of user turns per conversation is at least k, and 0 otherwise, for  $k \in \{ 2 , 3 , 5 \}$ </td></tr><tr><td>CHAT DAYS ≥ k</td><td>Indicator equal to 1 if aIQ Chat usage days in month t are at least k, and 0 otherwise, for k ∈ {5, 10, 15}.</td></tr><tr><td> $C O P D A Y S \ge k$ </td><td>Indicator equal to 1 if Copilot usage days in month t are at least k, and 0 otherwise, for k ∈ {1,5, 10, 15}.</td></tr></table>

Continued on next page

Continued from previous page
<table><tr><td> $T R A I N M O \geq k$ </td><td>Indicator equal to 1 if training courses completed by employee i in month t are at least k, and 0 otherwise, for  $k \in \{ 1 , 3 , 5 \}$ </td></tr><tr><td> $P A S T ~ T R A I N \ge k$ </td><td>Indicator equal to 1 if cumulative training courses completed by employee i through month t are at least k, and 0 otherwise, for  $k \in \{ 1 , 3 , 5 , 1 0 \}$ </td></tr></table>

<table><tr><td>1ST PROMPT LANG COMPLEXITY</td><td>LLM-assigned rating of linguistic complexity in the first substantive user prompt (1 = very simple language; 5 = very complex language).</td></tr><tr><td>1ST PROMPT COMPLEXITY</td><td>LLM-assigned rating of structural/task complexity in the first substantive user prompt (1 = trivial request; 5 = very complex request).</td></tr><tr><td>STRUCTURE</td><td>LLM-assigned rating of prompt organization and structure (1 = unstruc- tured; 5 = highly structured).</td></tr><tr><td>PROMPT SOPH</td><td>LLM-assigned rating of prompting sophistication (strategy-agnostic) across the conversation (1 = naive; 5 = expert).</td></tr><tr><td>SPECIFICITY</td><td>LLM-assigned rating of overall specificity and precision across the conver- sation (1 = very vague; 5 = very specific).</td></tr><tr><td>FORMAT CLARITY</td><td>LLM-assigned rating of how clearly the user specifies the desired output format (1 = no output guidance; 5 = precise specification).</td></tr><tr><td>CONSTRAINTS</td><td>Indicator equal to 1 if the initial substantive prompt includes at least one explicit, enforceable constraint (e.g., length, schema, tone, audience, time- frame), and 0 otherwise.</td></tr><tr><td>STRUCT OUTPUT</td><td>Indicator equal to 1 if the user explicitly requests structured data output (e.g., JSON, YAML, CSV, or a table with specified fields/schema), and 0 otherwise.</td></tr><tr><td>ACCEPT CRITERIA</td><td>Indicator equal to 1 if the user provides explicit acceptance criteria or suc- cess tests (e.g., must include specific fields or pass explicit checks), and 0 otherwise.</td></tr></table>

<table><tr><td colspan="2">Prompting strategies</td></tr><tr><td>ROLE PLAY</td><td>Indicator equal to 1 if the user assigns the LLM a specific role/persona (e.g., “act as an auditor&quot;), and 0 otherwise.</td></tr><tr><td>FEW SHOT</td><td>Indicator equal to 1 if the user provides concrete input-output examples to guide the LLM, and 0 otherwise.</td></tr><tr><td>CHAIN OF THOUGHT</td><td>Indicator equal to 1 if the user explicitly requests step-by-step reasoning or asks to see the LLM&#x27;s reasoning process, and 0 otherwise.</td></tr><tr><td>SELF CHECK</td><td>Indicator equal to 1 if the user instructs the LLM to check, verify, or validate its own output, and 0 otherwise.</td></tr></table>

Continued on next page

Continued from previous page
<table><tr><td>INTERACTIVE REFINE</td><td>Indicator equal to 1 if the user requests collaborative back-and-forth to re- fine the task (e.g., asks the LLM to ask clarifying questions), and 0 other- wise.</td></tr><tr><td colspan="2">Style rates (described at employee-month level)</td></tr><tr><td>ALL LOWER</td><td>Share of employee i&#x27;s conversations in month t in which all user messages are entirely lowercase (excluding quoted text, URLs, file paths, code snip- pets, and other exempt content).</td></tr><tr><td>TYPOS</td><td>Share of employee i&#x27;s conversations in month t in which the user makes frequent typos (at least 3 typos across the conversation or at least 2 typos in a single message).</td></tr><tr><td>DOC UPLOAD</td><td>Share of employee i&#x27;s conversations in month t in which the user uploads or references an attached document.</td></tr><tr><td>PLEASANTRIES</td><td>Share of employee i&#x27;s conversations in month t in which the user includes politeness expressions (e.g., greetings, thanks, &quot;please&quot;).</td></tr><tr><td>SATISFIED</td><td>Share of employee i&#x27;s conversations in month t in which the user expresses explicit positive feedback (e.g., &quot;thanks&quot; or “perfect&quot;).</td></tr></table>

## Online Appendix B: Supplementary Tables

This appendix is intended for online publication only. It reports the underlying statistics for the figures presented in the main text. Each figure has a corresponding table: Table B1 for Figure 1 (usage and intensity), Table B2 for Figure 2 (use cases by group), Table B3 for Figure 3 (evaluation of use), and Table B4 for Figure 4 (bivariate associations).

Table B 1 : Usa<sub>g</sub>e and Intensit<sub>y</sub>  
Panel A: B<sub>y</sub> Month
<table><tr><td>Month</td><td></td><td>N CHAT OR COP</td><td>CHAT USE</td><td>COP USE</td><td>CHAT DAYS</td><td>COP DAYS</td><td>CHAT CONVS</td></tr><tr><td>Sample Mean 32,849</td><td></td><td>0.839</td><td>0.437</td><td>0.780</td><td>2.714</td><td>5.832</td><td>8.563</td></tr><tr><td>2025-01</td><td>3,928</td><td>-0.060</td><td>-0.051</td><td>-0.070</td><td>-0.464</td><td>-1.360</td><td>-0.117</td></tr><tr><td>2025-02</td><td>4,101</td><td>-0.034</td><td>-0.016</td><td>-0.042</td><td>-0.191</td><td>-0.611</td><td>0.342</td></tr><tr><td>2025-03</td><td>4,105</td><td>0.006</td><td>0.008</td><td>0.003</td><td>0.193</td><td>-0.169</td><td>0.506</td></tr><tr><td>2025-04</td><td>4,078</td><td>0.020</td><td>0.012</td><td>0.023</td><td>0.180</td><td>0.323</td><td>0.480</td></tr><tr><td>2025-05</td><td>4,064</td><td>0.033</td><td>0.020</td><td>0.038</td><td>0.328</td><td>0.583</td><td>0.135</td></tr><tr><td>2025-06</td><td>4,235</td><td>0.009</td><td>0.019</td><td>0.006</td><td>0.006</td><td>-0.102</td><td>-0.062</td></tr><tr><td>2025-07</td><td>4,125</td><td>-0.014</td><td>-0.009</td><td>-0.011</td><td>-0.125</td><td>-0.015</td><td>-0.957</td></tr><tr><td>2025-08</td><td>4,213</td><td>0.036</td><td>0.013</td><td>0.048</td><td>0.058</td><td>1.269</td><td>-0.346</td></tr></table>

Panel B: B<sub>y</sub> Seniorit<sub>y</sub>
<table><tr><td>Seniority</td><td colspan="7">N CHAT OR COP CHAT USE COP USE CHAT DAYS COP DAYS CHAT CONVS</td></tr><tr><td>Sample Mean</td><td>32,849</td><td>0.839</td><td>0.437</td><td>0.780</td><td>2.714</td><td>5.832</td><td>8.563</td></tr><tr><td>Staff</td><td>12,360</td><td>-0.050</td><td>-0.019</td><td>-0.064</td><td>-0.075</td><td>-0.821</td><td>0.416</td></tr><tr><td>Manager</td><td>7,398</td><td>-0.002</td><td>0.024</td><td>-0.010</td><td>0.356</td><td>0.122</td><td>0.450</td></tr><tr><td>Above Manager 13,091</td><td></td><td>0.048</td><td>0.004</td><td>0.066</td><td>-0.130</td><td>0.706</td><td>-0.639</td></tr></table>

Table B 1 : Usa<sub>g</sub>e and Intensit<sub>y</sub> (continued)  
Panel C: B<sub>y</sub> Functional Area
<table><tr><td>Functional Area</td><td colspan="8">N CHAT OR COP CHAT USE COP USE CHAT DAYS</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Sample Mean</td><td>32,849</td><td>0.839</td><td>0.437</td><td>0.780</td><td>2.714</td><td>5.832</td><td>8.563</td></tr><tr><td>Accounting &amp; Finance</td><td>2,967</td><td>-0.057</td><td>-0.039</td><td>-0.043</td><td>-0.447</td><td>-0.768</td><td>-1.637</td></tr><tr><td>Administrative Services</td><td>2,995</td><td>0.004 -0.033</td><td>0.031 -0.122</td><td>0.005 -0.001</td><td>0.048 -0.877</td><td>-0.215 -0.981</td><td>0.118</td></tr><tr><td>Affiliate Operations Communications</td><td>657</td><td>0.088</td><td>0.115</td><td>0.110</td><td>0.847</td><td>2.034</td><td>-0.776</td></tr><tr><td>Compliance</td><td>827 865</td><td>-0.105</td><td>-0.092</td><td>-0.189</td><td>-0.362</td><td>-1.308</td><td>0.481</td></tr><tr><td>Digital Innovation</td><td>287</td><td>0.098</td><td>0.218</td><td>0.115</td><td>0.728</td><td>1.464</td><td>1.343</td></tr><tr><td>Information Technology</td><td>6,931</td><td>-0.037</td><td>-0.075</td><td>-0.040</td><td>-0.640</td><td>0.070</td><td>-1.553</td></tr><tr><td>Internal Audit</td><td>158</td><td>0.104</td><td>0.196</td><td>0.131</td><td>2.254</td><td>0.345</td><td>-1.294</td></tr><tr><td>Internal Operations</td><td>2,560</td><td>0.042</td><td>0.070</td><td>0.039</td><td>0.348</td><td>0.269</td><td>-0.063</td></tr><tr><td>Other</td><td>4,663</td><td>0.005</td><td>-0.101</td><td>0.027</td><td>-0.520</td><td>0.030</td><td>0.206</td></tr><tr><td></td><td>1,036</td><td>-0.049</td><td>0.114</td><td>-0.157</td><td>1.363</td><td>-0.036</td><td>1.070</td></tr><tr><td>Project Management</td><td>1,901</td><td>-0.021</td><td>0.021</td><td>-0.015</td><td>0.244</td><td>-0.198</td><td>1.314 0.438</td></tr><tr><td>Property &amp; Facilities Sales &amp; Marketing</td><td>5,416</td><td>0.058</td><td>0.125</td><td>0.059</td><td>0.877</td><td>0.176</td><td>0.447</td></tr><tr><td>Specialized Services</td><td>1,171</td><td>0.025</td><td>-0.122</td><td>0.050</td><td>-0.879</td><td>0.726</td><td>-1.940</td></tr><tr><td>Strategy</td><td>415</td><td>0.059</td><td>0.272</td><td>-0.012</td><td>2.830</td><td>-0.379</td><td>3.872</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

N<sub>o</sub>t<sub>es:</sub> P<sub>ane</sub>l<sub>s</sub> A–C <sub>repor</sub>t <sub>usage an</sub>d i<sub>n</sub>t<sub>ens</sub>it<sub>y s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>cs compu</sub>t<sub>e</sub>d <sub>a</sub>t th<sub>e emp</sub>l<sub>oyee</sub>-<sub>mon</sub>th l<sub>eve</sub>l E<sub>n</sub>t<sub>r</sub>i<sub>es are mean</sub>-<sub>cen</sub>t<sub>ere</sub>d b<sub>y</sub> <sub>su</sub>bt<sub>rac</sub>ti<sub>ng</sub> th<sub>e</sub> <sub>samp</sub>l<sub>e</sub> <sub>mean</sub> <sub>w</sub>hi<sub>c</sub>h i<sub>s</sub> <sub>repor</sub>t<sub>e</sub>d i<sub>n</sub> th<sub>e</sub> <sup>“</sup>S <sub>amp</sub>l<sub>e</sub> M<sub>ean</sub><sup>”</sup> <sub>row.</sub> St<sub>a</sub>ti<sub>s</sub>ti<sub>ca</sub>l <sub>s</sub>i<sub>gn</sub>ifi<sub>cance</sub> i<sub>s</sub> <sub>assesse</sub>d <sub>group</sub>-b<sub>y</sub>-<sub>group</sub> <sub>us</sub>i<sub>ng</sub> <sub>a</sub> diff<sub>erence</sub>-f<sub>rom</sub>-<sub>res</sub>t <sub>regress</sub>i<sub>on :</sub> f<sub>or</sub> <sub>eac</sub>h <sub>group g an</sub>d <sub>ou</sub>t<sub>come y</sub> <sub>we</sub> <sub>es</sub>ti<sub>ma</sub>t<sub>e</sub> $y = \alpha + \beta \ : .$ · 1 { group = g} and report two-sided p-values for β with White robust standard errors <sub>.</sub> Formatting indicates significance : italics $p < 0 . 1 0 ,$ b<sub>o</sub>ld $p < 0 . 0 5 .$ <sub>an</sub>d b<sub>o</sub>ld it<sub>a</sub>li<sub>cs</sub> $p < 0 . 0 1$ V<sub>ar</sub>i<sub>a</sub>bl<sub>e</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>ons are prov</sub>id<sub>e</sub>d i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> A

Table B2 : Use Cases b<sub>y</sub> Grou<sub>p</sub>  
Panel A: B<sub>y</sub> Month
<table><tr><td>Month</td><td colspan="6">N WRITING CODING DATA TEXT ANALYSIS KNOWLEDGE</td></tr><tr><td>Sample Mean 158,496</td><td></td><td>0.733</td><td>0.073</td><td>0.107</td><td>0.230</td><td>TOOL GUIDE IDEATION 0.099</td><td>0.018</td><td>PERSONAL OTHER 0.019</td></tr><tr><td>2025-01</td><td>15,992</td><td>0.025</td><td>-0.014</td><td>-0.002</td><td>0.003</td><td>-0.011</td><td>0.004</td><td>0.012 -0.001</td></tr><tr><td>2025-02</td><td>19,775</td><td>0.013</td><td>-0.009</td><td>0.007</td><td>0.011</td><td>-0.004 0.006</td><td>-0.003 -0.004</td><td>-0.000</td></tr><tr><td>2025-03</td><td>21,641</td><td>-0.003</td><td>0.001</td><td>0.004</td><td>0.020</td><td>-0.002</td><td>0.002 -0.003</td><td>0.002</td></tr><tr><td>2025-04</td><td>22,144</td><td>-0.007</td><td>0.007</td><td>0.000</td><td>-0.011</td><td>0.004</td><td>-0.001 0.000</td><td>0.000</td></tr><tr><td>2025-05</td><td>20,270</td><td>-0.000</td><td>0.006</td><td>0.016</td><td>0.005</td><td>-0.001</td><td>-0.003 0.001</td><td>-0.001</td></tr><tr><td>2025-06</td><td>21,553</td><td>0.013</td><td>-0.001</td><td>0.009</td><td>-0.015</td><td>-0.007</td><td>-0.004</td><td>-0.001 -0.001</td></tr><tr><td>2025-07</td><td>17,381</td><td>-0.011</td><td>-0.000</td><td>-0.027</td><td>-0.006</td><td>0.005</td><td>-0.002 0.004</td><td>0.002</td></tr><tr><td>2025-08</td><td>19,740</td><td>-0.026</td><td>0.005</td><td>-0.013</td><td>-0.007</td><td>0.014</td><td>-0.002 0.006</td><td>-0.001</td></tr></table>

Panel B: B<sub>y</sub> Seniorit<sub>y</sub>
<table><tr><td>Level</td><td></td><td></td><td>N WRITING CODING DATA TEXT ANALYSIS KNOWLEDGE TOOL GUIDE IDEATION PERSONAL</td><td></td><td></td><td></td><td></td><td></td><td>OTHER</td></tr><tr><td>Sample Mean</td><td>131,736</td><td>0.756</td><td>0.052</td><td>0.104</td><td>0.240</td><td>0.091</td><td>0.020</td><td>0.021</td><td>0.012</td></tr><tr><td>Staff</td><td>49,517</td><td>0.022</td><td>-0.005</td><td>-0.007</td><td>-0.053</td><td>-0.007</td><td>0.000</td><td>0.009</td><td>0.003</td></tr><tr><td>Manager</td><td>33,723</td><td>-0.043</td><td>0.037</td><td>0.001</td><td>-0.027</td><td>0.034</td><td>-0.003</td><td></td><td>-0.008-0.001</td></tr><tr><td>Above Manager</td><td>48,496</td><td>0.007</td><td>-0.020</td><td>0.007</td><td>0.073</td><td>-0.016</td><td>0.002</td><td></td><td>-0.003 -0.002</td></tr></table>

Table B2 : Use Cases b<sub>y</sub> Grou<sub>p</sub> (continued)  
Panel C: B<sub>y</sub> Functional Area
<table><tr><td>Functional</td><td colspan="10">N WRITING CODING DATA TEXT ANALYSIS KNOWLEDGE TOOL GUIDE IDEATION PERSONAL OTHER</td></tr><tr><td>Area</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Sample Mean</td><td>131,736</td><td>0.756</td><td>0.052</td><td>0.104</td><td>0.240</td><td>0.091</td><td>0.020</td><td>0.021</td><td>0.012</td></tr><tr><td>Accounting &amp; Finance</td><td>8,759</td><td>-0.124</td><td>0.021</td><td>-0.032</td><td>0.040</td><td>0.059</td><td>-0.009</td><td>-0.000</td><td>-0.002</td></tr><tr><td>Administrative Services</td><td>14,172</td><td>0.050</td><td>-0.029</td><td>-0.018</td><td>-0.058</td><td>-0.020</td><td>-0.003</td><td>0.020</td><td>0.004</td></tr><tr><td>Affiliate Operations</td><td>1,628</td><td>0.083</td><td>-0.008</td><td>-0.032</td><td>-0.095</td><td>-0.029</td><td>-0.009</td><td>-0.011</td><td>-0.003</td></tr><tr><td>Communications</td><td>4,273</td><td>0.118</td><td>-0.039</td><td>-0.004</td><td>-0.031</td><td>-0.071</td><td>0.009</td><td>-0.010</td><td>-0.006</td></tr><tr><td>Compliance</td><td>2,991</td><td>0.051</td><td>-0.036</td><td>-0.031</td><td>0.039</td><td>-0.029</td><td>-0.009</td><td>-0.005</td><td>-0.002</td></tr><tr><td>Digital Innovation</td><td>1,464</td><td>-0.114</td><td>-0.021</td><td>0.076</td><td>0.103</td><td>-0.046</td><td>0.011 -0.008</td><td>-0.010</td><td>0.000</td></tr><tr><td>Information Technology</td><td>19,115</td><td>-0.182</td><td>0.109</td><td>-0.031</td><td>-0.008</td><td>0.112</td><td></td><td>-0.004</td><td>0.002</td></tr><tr><td>Internal Audit</td><td>915</td><td>0.003</td><td>-0.037</td><td>-0.021</td><td>0.167</td><td>-0.005</td><td>-0.012</td><td>-0.009</td><td>0.007</td></tr><tr><td>Internal Operations</td><td>11,990</td><td>0.045</td><td>-0.025</td><td>-0.033</td><td>-0.070</td><td>-0.023</td><td>-0.003</td><td>0.027</td><td>0.003</td></tr><tr><td>Other</td><td>16,383</td><td>0.040</td><td>-0.026</td><td>0.015 0.054</td><td>0.017 0.114</td><td>-0.016 -0.009</td><td>-0.001 -0.002</td><td>-0.005</td><td>-0.001</td></tr><tr><td>Project Management</td><td>6,282</td><td>0.023</td><td>-0.031</td><td>-0.026</td><td>-0.041</td><td>-0.015</td><td>0.008</td><td>-0.007 0.004</td><td>-0.001</td></tr><tr><td>Property &amp; Facilities</td><td>8,053 29,149</td><td>0.052 0.062</td><td>-0.017 -0.016</td><td>0.031</td><td>0.009</td><td>-0.046</td><td>0.010</td><td>-0.009</td><td>-0.001</td></tr><tr><td>Sales &amp; Marketing</td><td>2,523</td><td>-0.086</td><td>0.035</td><td>-0.025</td><td>0.017</td><td>0.091</td><td>-0.008</td><td>-0.010</td><td>-0.001 -0.002</td></tr><tr><td>Specialized Services</td><td>4,039</td><td>-0.028</td><td>-0.016</td><td>0.092</td><td>0.091</td><td>-0.009</td><td>0.002</td><td>-0.014</td><td>-0.004</td></tr><tr><td>Strategy</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

N<sub>o</sub>t<sub>es:</sub> P<sub>ane</sub>l<sub>s</sub> A–C <sub>repor</sub>t th<sub>e propor</sub>ti<sub>on o</sub>f <sub>conversa</sub>ti<sub>on</sub>-l<sub>eve</sub>l <sub>use cases</sub> b<sub>y group .</sub> E<sub>ac</sub>h <sub>ce</sub>ll i<sub>s</sub> th<sub>e</sub> f<sub>rac</sub>ti<sub>on o</sub>f <sub>conversa</sub>ti<sub>ons</sub> i<sub>n</sub> th<sub>e group</sub> t<sub>agge</sub>d <sub>w</sub>ith th<sub>e correspon</sub>di<sub>ng use</sub>-<sub>case ca</sub>t<sub>egory.</sub> B<sub>ecause conversa</sub>ti<sub>ons may</sub> h<sub>ave mu</sub>lti<sub>p</sub>l<sub>e use</sub>-<sub>case</sub> l<sub>a</sub>b<sub>e</sub>l<sub>s row propor</sub>ti<sub>ons may sum</sub> t<sub>o</sub> <sub>more</sub> th<sub>an one .</sub> P<sub>ane</sub>l<sub>s</sub> B <sub>an</sub>d C <sub>exc</sub>l<sub>u</sub>d<sub>e conversa</sub>ti<sub>ons</sub> f<sub>or w</sub>hi<sub>c</sub>h <sub>sen</sub>i<sub>or</sub>it<sub>y or</sub> f<sub>unc</sub>ti<sub>ona</sub>l-<sub>area</sub> d<sub>a</sub>t<sub>a are unava</sub>il<sub>a</sub>bl<sub>e .</sub> E<sub>n</sub>t<sub>r</sub>i<sub>es are mean</sub>-<sub>cen</sub>t<sub>ere</sub>d b<sub>y</sub> <sub>su</sub>bt<sub>rac</sub>ti<sub>ng</sub> th<sub>e</sub> <sub>samp</sub>l<sub>e</sub> <sub>mean,</sub> <sub>w</sub>hi<sub>c</sub>h i<sub>s</sub> <sub>repor</sub>t<sub>e</sub>d i<sub>n</sub> th<sub>e</sub> <sup>“</sup>S <sub>amp</sub>l<sub>e</sub> M<sub>ean</sub><sup>”</sup> <sub>row.</sub> St<sub>a</sub>ti<sub>s</sub>ti<sub>ca</sub>l <sub>s</sub>i<sub>gn</sub>ifi<sub>cance</sub> i<sub>s</sub> <sub>assesse</sub>d <sub>group</sub>-b<sub>y</sub>-<sub>group</sub> <sub>us</sub>i<sub>ng</sub> <sub>a</sub> difference-from-rest regression : for each group g and use-case indicator y we estimate y = α + β · 1 { group = g} and report two-sided p-values for β with White robust standard errors <sub>.</sub> Formatting indicates significance : italics $p < 0 . 1 0$ b<sub>o</sub>ld $p < 0 . 0 5 _ { : }$ <sub>an</sub>d b<sub>o</sub>ld it<sub>a</sub>li<sub>cs</sub> $p < 0 . 0 1$ <sub>.</sub> V<sub>ar</sub>i<sub>a</sub>bl<sub>e</sub> d<sub>e</sub>fi<sub>n</sub>iti<sub>ons</sub> <sub>are</sub> <sub>prov</sub>id<sub>e</sub>d i<sub>n</sub> A<sub>ppen</sub>di<sub>x</sub> A<sub>.</sub>

Table B3: Evaluation of Use  
Panel A: By Month
<table><tr><td>Group</td><td>N</td><td>CLARITYZ</td><td>ANY STRAT</td><td>USE DIV</td></tr><tr><td>Sample Mean</td><td>17,671</td><td>0.000</td><td>0.058</td><td>0.316</td></tr><tr><td>2025-01</td><td>1,799</td><td>-0.078</td><td>-0.003</td><td>-0.023</td></tr><tr><td>2025-02</td><td>2,048</td><td>-0.031</td><td>0.007</td><td>0.005</td></tr><tr><td>2025-03</td><td>2,191</td><td>0.017</td><td>0.008</td><td>0.012</td></tr><tr><td>2025-04</td><td>2,262</td><td>-0.019</td><td>0.006</td><td>0.007</td></tr><tr><td>2025-05</td><td>2,257</td><td>0.017</td><td>-0.011</td><td>0.001</td></tr><tr><td>2025-06</td><td>2,477</td><td>0.071</td><td>-0.011</td><td>-0.006</td></tr><tr><td>2025-07</td><td>2,267</td><td>0.004</td><td>0.002</td><td>-0.007</td></tr><tr><td>2025-08</td><td>2,370</td><td>-0.006</td><td>0.002</td><td>0.007</td></tr></table>

Panel B: By Seniority
<table><tr><td>Group</td><td>N</td><td>CLARITYZ</td><td>ANY STRAT</td><td>USE DIV</td></tr><tr><td>Sample Mean</td><td>14,449</td><td>0.002</td><td>0.061</td><td>0.322</td></tr><tr><td>Staff</td><td>5,206</td><td>-0.056</td><td>-0.009</td><td>-0.011</td></tr><tr><td>Manager</td><td>3,444</td><td>0.023</td><td>0.002</td><td>0.002</td></tr><tr><td>Above Manager</td><td>5,799</td><td>0.036</td><td>0.007</td><td>0.009</td></tr><tr><td>Group</td><td>N</td><td>CLARITYZ ANY STRAT</td><td></td><td>USE DIV</td></tr><tr><td>Sample Mean</td><td>14,449</td><td>0.002</td><td>0.061</td><td>0.322</td></tr><tr><td>Accounting &amp; Finance</td><td>1,192</td><td>-0.311</td><td>-0.010</td><td>-0.019</td></tr><tr><td>Administrative Services</td><td>1,407</td><td>-0.159</td><td>-0.013</td><td>-0.015</td></tr><tr><td>Affiliate Operations</td><td>207</td><td>-0.139</td><td>-0.013</td><td>-0.050</td></tr><tr><td>Communications</td><td>461</td><td>0.343</td><td>0.007</td><td>-0.024</td></tr><tr><td>Compliance</td><td>299</td><td>-0.074</td><td>0.014</td><td>-0.016</td></tr><tr><td>Digital Innovation</td><td>190</td><td>0.302</td><td>-0.002</td><td>0.061</td></tr><tr><td>Information Technology</td><td>2,552</td><td>-0.127</td><td>0.011</td><td>0.026</td></tr><tr><td>Internal Audit</td><td>100</td><td>-0.203</td><td>-0.003</td><td>0.009</td></tr><tr><td>Internal Operations</td><td>1,309</td><td>-0.094</td><td>-0.024</td><td>-0.026</td></tr><tr><td>Other</td><td>1,572</td><td>0.133</td><td>0.007</td><td>0.004</td></tr><tr><td>Project Management</td><td>574</td><td>0.122</td><td>0.019</td><td>0.038</td></tr><tr><td>Property &amp; Facilities</td><td>872</td><td>-0.075</td><td>-0.025</td><td>-0.016</td></tr><tr><td>Sales &amp; Marketing</td><td>3,051</td><td>0.214</td><td>0.009</td><td>-0.001</td></tr><tr><td>Specialized Services</td><td>369</td><td>-0.154</td><td>0.003</td><td></td></tr><tr><td>Strategy</td><td>294</td><td>0.303</td><td>0.010</td><td>-0.011 0.056</td></tr></table>

Notes: Panels A–C report group averages computed over active employeemonths. Active employee-months are months with at least one aIQ Chat conversation. N is the number of active employee-months in the group. Panels B and C exclude active employee-months for which seniority or functional-area data are unavailable. Entries are mean-centered by subtracting the sample mean, which is reported in the “Sample Mean” row; percentage-formatted columns are shown as percentage-point deviations from the sample mean. Statistical signif icance is assessed group-by-group using a difference-from-rest regression: for each group g and outcome y, we estimate $y = \alpha + \beta \cdot \mathbb { 1 } \{ { \mathrm { g r o u p } } = g \}$ and report two-sided p-values for β with White robust standard errors. Formatting indicates significance: italics $p < 0 . 1 0$ , bold $p < 0 . 0 5$ , and bold italics $p < 0 . 0 1$ . Variable definitions are provided in Appendix A.

Table B4: Bivariate Associations  
Panel A: Without User Fixed Effects
<table><tr><td colspan="4">CLARITYZ</td><td colspan="2">ANY STRAT</td><td colspan="2">USE DIV</td></tr><tr><td>Predictor</td><td>N</td><td>beta</td><td>p</td><td>beta</td><td>p</td><td>beta</td><td>p</td></tr><tr><td>Ambition</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>1ST PROMPT LEN 17,671 0.000 0.000 0.000 0.000 0.000 0.000</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Persistence</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ROUNDS</td><td></td><td></td><td></td><td>17,671 0.279 0.000 0.021 0.000 0.054 0.000</td><td></td><td></td><td></td></tr><tr><td>ROUNDS ≥ 2</td><td></td><td></td><td></td><td>17,671 0.323 0.000 0.024 0.000 0.080 0.000</td><td></td><td></td><td></td></tr><tr><td>ROUNDS ≥ 3</td><td></td><td></td><td></td><td>17,671 0.328 0.000 0.034 0.000 0.056 0.000</td><td></td><td></td><td></td></tr><tr><td>ROUNDS ≥ 5</td><td></td><td></td><td></td><td>17,671 0.333 0.000 0.032 0.000 0.036 0.000</td><td></td><td></td><td></td></tr><tr><td>Frequency</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CHAT CONVS</td><td></td><td></td><td></td><td>17,671 0.134 0.000-0.008 0.000 0.109 0.000</td><td></td><td></td><td></td></tr><tr><td>CHAT DAYS</td><td></td><td></td><td></td><td>14,449 0.101 0.000-0.023 0.000 0.121 0.000</td><td></td><td></td><td></td></tr><tr><td>CHAT DAYS ≥ 5</td><td></td><td></td><td></td><td>14,4490.148 0.000-0.029 0.0000.1640.000</td><td></td><td></td><td></td></tr><tr><td>CHAT DAYS ≥ 10</td><td>14,449 0.125 0.000-0.026 0.000 0.113 0.000</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CHAT DAYS ≥ 15 1</td><td></td><td></td><td></td><td>14,449 0.108 0.002 -0.026 0.000 0.071 0.000</td><td></td><td></td><td></td></tr><tr><td>Flexibility</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>COP DAYS</td><td></td><td></td><td></td><td>14,449 0.148 0.000 0.014 0.000 0.038 0.000</td><td></td><td></td><td></td></tr><tr><td>COP DAYS ≥ 1</td><td></td><td></td><td></td><td>14,449 0.283 0.000 0.029 0.000 0.046 0.000</td><td></td><td></td><td></td></tr><tr><td>COP DAYS ≥ 5</td><td></td><td></td><td></td><td>14,449 0.229 0.000 0.021 0.000 0.069 0.000</td><td></td><td></td><td></td></tr><tr><td>COP DAYS ≥ 10</td><td></td><td></td><td></td><td>14,449 0.241 0.000 0.022 0.000 0.067 0.000</td><td></td><td></td><td></td></tr><tr><td>COP DAYS ≥ 15</td><td></td><td></td><td></td><td>14,449 0.258 0.000 0.020 0.001 0.068 0.000</td><td></td><td></td><td></td></tr><tr><td>Training</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TRAIN MO</td><td></td><td></td><td></td><td>13,091 0.084 0.000 0.054 0.000 0.037 0.000</td><td></td><td></td><td></td></tr><tr><td>TRAIN MO ≥ 1 TRAIN MO ≥ 3</td><td></td><td></td><td></td><td>13,091 0.069 0.000 0.056 0.000 0.034 0.000</td><td></td><td></td><td></td></tr><tr><td>TRAIN MO ≥ 5</td><td></td><td></td><td>13,091 0.157 0.002 0.046 0.000 0.048 0.000</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>13,091 0.250 0.003 0.061 0.001 0.039 0.022</td><td></td><td></td><td></td><td></td></tr><tr><td>PAST TRAIN</td><td></td><td></td><td></td><td>13,091 0.159 0.000 0.018 0.000 0.028 0.000</td><td></td><td></td><td></td></tr><tr><td>PAST TRAIN ≥ 1</td><td></td><td></td><td></td><td>13,091 0.130 0.000 0.016 0.003 0.029 0.000</td><td></td><td></td><td></td></tr><tr><td>PAST TRAIN ≥ 3</td><td></td><td></td><td></td><td>13,091 0.194 0.000 0.017 0.002 0.032 0.000</td><td></td><td></td><td></td></tr><tr><td>PAST TRAIN ≥ 5</td><td></td><td></td><td></td><td>13,091 0.231 0.000 0.021 0.002 0.039 0.000</td><td></td><td></td><td></td></tr><tr><td>PAST TRAIN ≥ 10 13,091 0.225 0.007 0.022 0.132 0.032 0.029</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2"></td><td colspan="2">CLARITYZ</td><td colspan="2">ANY STRAT</td><td colspan="2">USE DIV p p</td></tr><tr><td colspan="2">Predictor</td><td colspan="2">N beta</td><td colspan="2">p beta</td><td colspan="2">beta</td></tr><tr><td colspan="7">Ambition</td></tr><tr><td>1ST PROMPT LEN 17,671 0.000 0.000 0.000 0.012 0.000 0.000</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Persistence</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ROUNDS</td><td></td><td>17,671 0.289 0.000 0.032 0.000 0.058 0.000</td><td></td><td></td><td></td><td></td></tr><tr><td> $R O U N D S \ge 2$ </td><td></td><td>17,671 0.244 0.000 0.028 0.000 0.053 0.000</td><td></td><td></td><td></td><td></td></tr><tr><td> $R O U N D S \ge 3$ </td><td>17,671 0.250 0.000 0.034 0.000 0.043 0.000</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $R O U N D S \ge 5$ </td><td></td><td></td><td></td><td></td><td></td><td>17,671 0.256 0.000 0.028 0.000 0.032 0.000</td></tr><tr><td colspan="7">Frequency</td></tr><tr><td>CHAT CONVS</td><td>17,671 0.108 0.000 0.010 0.000 0.145 0.000</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CHAT DAYS</td><td></td><td></td><td></td><td></td><td></td><td>14,449 0.066 0.000 -0.002 0.522 0.159 0.000</td></tr><tr><td> $C H A T D A Y S \geq 5$ </td><td></td><td></td><td></td><td></td><td></td><td>14,449 0.074 0.000 -0.001 0.877 0.150 0.000</td></tr><tr><td> $C H A T D A Y S \geq 1 0$ </td><td></td><td></td><td></td><td></td><td></td><td>14,449 0.038 0.044 -0.000 0.931 0.080 0.000</td></tr><tr><td> $C H A T D A Y S \geq 1 5$ </td><td></td><td></td><td></td><td></td><td></td><td>14,449 0.055 0.031 -0.005 0.160 0.043 0.000</td></tr><tr><td colspan="7">Flexibility</td></tr><tr><td>COP DAYS</td><td>14,449 0.005 0.743 0.004 0.159 0.035 0.000</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $C O P D A Y S \geq 1$ </td><td></td><td></td><td></td><td></td><td></td><td>14,449 0.046 0.101 0.012 0.028 0.024 0.003</td></tr><tr><td> $C O P D A Y S \ge 5$ </td><td></td><td></td><td></td><td></td><td></td><td>14,449-0.011 0.621 0.003 0.516 0.034 0.000</td></tr><tr><td> $C O P D A Y S \geq 1 0$ </td><td></td><td></td><td></td><td></td><td></td><td>14,449 -0.005 0.832 -0.002 0.763 0.037 0.000</td></tr><tr><td> $C O P D A Y S \geq 1 5$ </td><td></td><td></td><td></td><td></td><td></td><td>14,449 0.028 0.306 -0.002 0.774 0.034 0.000</td></tr><tr><td colspan="7">Training</td></tr><tr><td>TRAIN MO</td><td>13,091 0.071 0.000 0.040 0.000 0.036 0.000</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $T R A I N M O \geq 1$ </td><td></td><td></td><td></td><td></td><td></td><td>13,091 0.062 0.000 0.038 0.000 0.033 0.000</td></tr><tr><td> $T R A I N M O \geq 3$ </td><td></td><td></td><td></td><td></td><td></td><td>13,091 0.131 0.004 0.037 0.000 0.037 0.000</td></tr><tr><td> $T R A I N M O \geq 5$ </td><td></td><td></td><td></td><td></td><td></td><td>13,091 0.103 0.226 0.039 0.027 0.041 0.017</td></tr><tr><td>PAST TRAIN</td><td></td><td></td><td></td><td></td><td></td><td>13,091 -0.033 0.179 -0.025 0.000-0.008 0.183</td></tr><tr><td> $P A S T ~ T R A I N \ge 1$ </td><td></td><td></td><td></td><td></td><td></td><td>13,091 -0.024 0.359-0.006 0.3200.006 0.336</td></tr><tr><td> $P A S T ~ T R A I N \ge 3$ </td><td></td><td></td><td></td><td></td><td></td><td>13,091 0.015 0.577 -0.025 0.000-0.016 0.022</td></tr><tr><td> $P A S T ~ T R A I N \ge 5$ </td><td></td><td></td><td></td><td></td><td></td><td>13,091 -0.070 0.046-0.021 0.004-0.013 0.148</td></tr><tr><td> $P A S T T R A I N \ge 1 0$ </td><td></td><td></td><td></td><td></td><td></td><td>13,091 -0.124 0.087-0.029 0.034 -0.028 0.067</td></tr></table>

Notes: Panels A and B report bivariate associations between each dependent measure and each predictor using active employee-month observations with month fixed effects; Panel B additionally includes user fixed effects. Predictors are grouped by the construct each one proxies for, following Figure 4. Aside from 1ST PROMPT LEN and the threshold indicators, each predictor enters as log(1 + x); row labels omit the transform for legibility. Coefficients are estimated by OLS after residualizing variables with respect to the included fixed effects. White robust standard errors are clustered at the user level. The table reports coefficient estimates and corresponding two-sided p-values. The N column reports the number of active employee-months with non-missing predictor values; the effective regression sample may vary across dependent measures because of predictor missingness. Variable definitions are provided in Appendix A.

# Online Appendix C: Meta Prompts

This appendix is intended for online publication only. It reproduces the metaprompts submitted with each aIQ Chat transcript.

## Prompt One: Use Cases

## ## LLM Conversation Analysis Prompt

You are an LLM conversation analysis expert. You will analyze a multi-turn

, conversation between a KPMG employee and an LLM to evaluate its use cases.

Your analysis will classify the use case(s) of the conversation.

## ## USE CASE CLASSIFICATION

Classify the conversation into one or more subcategories. A single conversation

, often serves multiple purposes - capture all that apply.

## ## CRITICAL CLASSIFICATION PRINCIPLES

1. \*\*Focus on the ACTION requested\*\*, not just the subject matter discussed

\- Example - Describing data from documents that mention software → Document

, Understanding (3-1), not Software guidance

2. \*\*Read the ENTIRE conversation\*\*, not just initial prompts - user intent often

, clarifies through the dialogue

3. \*\*Use multiple categories\*\*, when applicable, most conversations warrant more

, than one.

## ## EVIDENCE POLICY FOR USING LLM RESPONSES

Classification decisions must rely only on the user prompts (the Initial User

Prompt and any Follow Up User Prompts<sup>)</sup>. You may look at LLM responses solely to,<sub>→</sub>

interpret what the user is clarifying in subsequent follow-ups,

## ### MAIN CATEGORIES AND SUBCATEGORIES:

## \*\*1: Writing and Communication\*\*

\- 1-1: Generate new content

\- \*\*Includes\*\*: Writing from scratch, creating based on requirements, expanding

bullet points into full documents, creating summaries intended as

deliverables,

\- \*\*Excludes\*\*: Editing text that the USER provided (that's 1-2), summaries

, purely for comprehension (that's 3-1 only)

\- \*\*Expansion rule\*\*: If output is >2x longer than input = 1-1

## - 1-2: Edit/improve existing content

\- \*\*What counts as "existing content"\*\*:

\- Text explicitly provided by the user (pasted, uploaded, or quoted)

\- Text the user claims to have written themselves

\- Full paragraphs or complete documents

\- Does NOT include: Brief examples, fragments, or bullet points that need , expansion

\- \*\*Includes\*\*: Grammar fixes, clarity improvements, restructuring of provided , text

\- \*\*Excludes\*\*: Creating new content based on descriptions or outlines

\- \*\*Note\*\*: Editing user-provided content about specialized topics (e.g., legal,

accounting) is ONLY 1-2, and not also a knowledge category (4-x) unless the,<sub>→</sub>

user requests development or expansion of the technical writing with domain,

knowledge, or requests an accuracy check.,

## \*\*When to use BOTH 1-1 and 1-2\*\*:

\- User provides some content AND requests significant new additions

\- Examples:

\- "Improve this text and add a conclusion" → both 1-1 and 1-2

\- "Enhance this section and add supporting examples" → both 1-1 and 1-2

\- If one prompt generates and another edits, then it's 1-1 and 1-2.

## - 1-3: Language translation: Converting text between languages

## - 1-other: Other writing tasks not listed above

## \*\*2: Coding and Data Analysis\*\*

## - 2-1: Generate or debug programming code

\- \*\*Includes\*\*: ANY executable code generation (Python, R, SQL, VBA, JavaScript, , etc.) including VBA code for Excel

\- \*\*Includes\*\*: Code debugging, modification, optimization

\- \*\*Excludes\*\*: Discussing code concepts without generating code. Also, Excel

, formulas are not code in this taxonomy and should be classified as 5-1.

## - 2-2: Analyze user-provided data

\- \*\*REQUIRES\*\*: Actual data provided (Excel, CSV, tables) AND analysis performed

\- \*\*Includes\*\*: Statistical analysis, data interpretation, insights from provided , data

\- \*\*Excludes\*\*: Just discussing analysis methodology, manual calculations without , data

\- \*\*Excludes\*\*: Analysis of user-provided writing/prose (that's 3-1)

\- \*\*Data vs. Text Distinction\*\*: If a table is text-only and fits on one page,

, treat as writing (1-x categories). Otherwise, treat as data (2-x categories)

## - 2-3: Data cleaning, restructuring, standardization

\- \*\*REQUIRES\*\*: Actual data provided (Excel, CSV, tables) AND manipulation tasks , performed

\- \*\*Includes\*\*: Removing duplicates, reformatting, normalizing data

\- \*\*Key test\*\*: Is the LLM actively transforming data? Not just discussing how to , do it

## - 2-other: Other coding/data tasks

\*\*3: Text-Based Analysis\*\*

\- 3-1: Document Understanding

\- \*\*REQUIRES\*\*: Document specified AND requests made related to the content

\- \*\*Note:\*\* Document can be \*specified\* by being uploaded, pasted, or requested , via search

\- \*\*Includes\*\*: "Summarize the following", "what does this say about X", "read

<sup>this</sup> <sup>and</sup> <sup>prepare</sup> <sup>to</sup> <sup>answer</sup> <sup>questions",</sup> <sup>analysis</sup> <sup>of</sup> <sup>user-provided</sup> <sup>writing</sup> <sup>or,</sup>→

<sup>prose,</sup>→

\- \*\*Excludes\*\*: General questions without documents, uploading text as an example , (e.g., write in the style demonstrated by this document)

\- \*\*Data (2-2) vs Text (3-1)\*\*: Where data is structured, tabular, or otherwise

aggregable, text is unique narrative or prose. When an uploaded file contains,

<sup>both,</sup> <sup>code</sup> <sup>based</sup> <sup>on</sup> <sup>the</sup> <sup>actions</sup> <sup>requested.</sup> <sup>When</sup> <sup>the</sup> <sup>same</sup> <sup>request</sup> <sup>involves,</sup>→

text-based and data analysis, code as both 2-2 and 3-1.,

## 3-other: Other text-based analysis requests

\*\*4: Knowledge and Expertise\*\*

\- 4-1: Accounting standards and guidance

\- \*\*Includes\*\*: GAAP, IFRS, audit procedures, financial reporting

\- \*\*Excludes\*\*: General business operations, non-financial audits

\- \*\*Note\*\*: Writing a new memo to explain an accounting standard would be both

, (1-1 and 4-1). User-provided accounting content for editing = 1-2 only.

\- 4-2: Legal standards, Regulations, Compliance, and related concepts

\- \*\*Includes\*\*: Legal principles, interpretations, SEC guidance,

, industry-specific regulations, compliance requirements

\- \*\*Note\*\*: Uploading an SEC Comment Letter and requesting a summary would be

both document understanding (3-1) and this category (4-2).

\- 4-3: Business operations

\- \*\*Includes\*\*: How industry, products, or workflows function

\- \*\*Excludes\*\*: Specific aspects other than operations (e.g., people management

, or HR) → that's 4-4

\- 4-4: Business Management

\- \*\*Includes\*\*: People management techniques, HR, Marketing, financing, incentive

<sup>,</sup>→ <sup>design</sup>

\- \*\*Excludes\*\*: Business operations (e.g., revenue and collection cycle)

\- 4-other: knowledge queries outside 4-1 - 4-4.

\*\*5: Software and Tool Guidance\*\*

\*\*CRITICAL FOR ALL 5-X CATEGORIES\*\*: User must be seeking guidance on HOW TO USE

the software, not just mentioning it. Questions about creating content <sup>(</sup>e.g.,,<sub>→</sub>

, "How do I create an effective presentation in PowerPoint?") focus on content

creation (1-x categories), not software usage.,

## - 5-1: Microsoft Excel

\- \*\*Includes\*\*: Excel Formula help, Excel feature guidance, Excel-specific

<sup>,</sup>→ <sup>troubleshooting</sup>

\- \*\*Key test\*\*: Is the question about Excel's functionality? Not just

, Excel-related

\- \*\*Excludes\*\*: Discussion of formulas, code, data that are not specifically

, about Excel

## - 5-2: Alteryx

\- \*\*Watch for\*\*: Brief mentions can be overlooked - scan entire conversation

\- \*\*Includes\*\*: Workflow guidance, Alteryx-specific formulas

## - 5-3: Microsoft PowerPoint

\- \*\*Focus on\*\*: PowerPoint SOFTWARE guidance, not presentation content

\- \*\*Common trap\*\*: Creating or Revising original presentation content → 1-1 or

, 1-2, not 5-3

\- \*\*Excludes\*\*: Power BI (different product)

## - 5-4: Microsoft Word

\- \*\*Specific to\*\*: Word software features and functionality

\- \*\*Excludes\*\*: General document writing (→ 1-1)

## - 5-5: Microsoft Outlook

\- \*\*Includes\*\*: Email, calendar, meeting features IN OUTLOOK

\- \*\*Excludes\*\*: email/calendar content without discussing Outlook features

## - 5-6: AI/LLM tools

\- \*\*STRICT CRITERION\*\*: Discussion must be \*about\* the AI/LLM tool itself.

\- \*\*Includes\*\*: How to use AI tools, comparing AI tools, AI tool best practices,

, meta-conversations about AI usage

\- \*\*Excludes\*\*: Using an LLM to accomplish tasks (classify by the task instead)

\- \*\*Example\*\*: Which prompting techniques work best in Claude-3 for extracting

, structured data from PDFs? Compare with GPT-4o.

\- 5-other: guidance on other software (Google Sheets, Tableau, etc.)

## \*\*6. Creative Thinking\*\*

\- 6-1: Brainstorming and ideation

\- \*\*REQUIRES\*\*: Generating NEW ideas, not listing existing ones

\- \*\*Key words\*\*: "ideas for", "possibilities", "creative solutions", "what if",

, propose other titles

\- 6-other: other creative thinking requests

## \*\*7: Non-work Related\*\*

\- 7-1: Personal tasks

\- \*\*Clear indicators\*\*: Grocery lists, personal travel, home projects

\- \*\*Note\*\*: Even vague initial prompts often clarify as personal through

, conversation

\- \*\*Example\*\*: Revising a resume or anticipating interview questions for a job

, interview rather than bidding for a project.

\- 7-other: Other non-work activities

\*\*8: Other\*\*

\- 8-1: Testing LLM capabilities

\- \*\*Intent\*\*: Explicitly testing what the LLM can do

\- 8-2: Unclear or ambiguous use case

\- \*\*Use when\*\*: Initial prompt lacks detail AND conversation doesn't clarify

\- 8-other: anything that truly fits nowhere else

\*\*### USING "OTHER" SUBCATEGORIES:\*\*

\- Use specific "other" subcategories (e.g., 1-other, 2-other) when:

\- The conversation clearly belongs to the main category

\- But doesn't fit any existing subcategory

\- Example: A writing task that isn't generating, editing, or translating

\- Use 8-other only when the conversation doesn't fit ANY main category

\- Always attempt to classify into existing subcategories before using "other"

, options

\### DATA VS TEXT DISTINCTION:

1. \*\*Read the ENTIRE conversation\*\* before classifying - intent often clarifies

2. \*\*Focus on actions and outcomes\*\*, not just topics discussed

3. \*\*Use multiple subcategories\*\*, when appropriate

## ### EXEMPLARS

1. \*\*Domain-knowledge tagging rule\*\*

\- \*\*ADD a 4-x tag when\*\*:

\- User explicitly requests domain expertise to be incorporated ("explain the , accounting treatment for...")

\- User asks for accuracy verification of technical content

\- User requests application of standards/regulations to a scenario

\- The PRIMARY value comes from domain knowledge, not just writing ability

\- \*\*DO NOT add a 4-x tag when\*\*:

\- User provides the technical content and only requests formatting/editing

\- Domain knowledge is incidental to the writing task

2. \*\*Business operations + management\*\* \*(double-coding 4-3 + 4-4)\*

\* Prompt: “Describe how a regional retail bank generates revenue across its core

products (deposits, consumer lending, wealth management). Then outline a,

<sup>performance-measurement</sup> <sup>and</sup> <sup>incentive-compensation</sup> <sup>plan</sup> <sup>for</sup> <sup>branch</sup> <sup>managers,</sup>→

that aligns with those revenue streams.”,

\* Tags: \*\*4-3 + 4-4\*\*

3. \*\*Example of the importance of reading the entire conversation and using <sup>,</sup>→ <sup>multiple</sup> <sup>categories\*\*:</sup>

\- If a user submits one prompt requesting a summary of user-provided text, that would be 3-1 (document understanding) as the intended use seems to be,

personal learning. By contrast, following the requested summary with requests,

<sup>for</sup> <sup>revisions</sup> <sup>provides</sup> <sup>evidence</sup> <sup>that</sup> <sup>the</sup> <sup>user's</sup> <sup>first</sup> <sup>prompt</sup> <sup>also</sup> <sup>wanted</sup> <sup>help,</sup>→

with Writing and Communication; therefore, second prompt reveals that the,

original prompt is both document understanding (3-1), and generate new,<sub>→</sub>

content (1-1). Then, the revision request in the second prompt is also,

edit/improve content (1-2). Note: This example demonstrates how later,<sub>→</sub>

, messages within the same conversation can reveal the true intent of initial

prompts, highlighting why the entire conversation must be read before,

classification.,

## 4. \*\*Clear 1-1, 1-2, and 3-1 distinction\*\*

\- "Create a one-page summary of this document for the board" → 1-1 and 3-1

\- "Help me understand this report" → 3-1 only

\- "Draft a summary based on this report" → 1-1 and 3-1

- "Read the attached document and use it to revise the following message" → 1-2   
, and 3-1

## ## INPUT

You were or will be given a labeled conversation transcript with the following turn

## , types (in order as applicable):

1. System Prompt

2. Initial User Prompt

3. Initial LLM Response

4. Follow Up User Prompt (zero or more)

5. Follow Up LLM Response (interleaved with follow ups)

6. Final LLM Response (if present)

Prompts and responses alternate chronologically. Do not use LLM Responses for

<sup>,</sup>→ <sup>classifications</sup> <sup>except</sup> <sup>to</sup> <sup>provide</sup> <sup>context</sup> <sup>for</sup> <sup>follow-up</sup> <sup>user</sup> <sup>prompts.</sup>

## ## OUTPUT FORMAT

Return your analysis as a single JSON object:

```json
{
"use_cases": ["1-1", "1-2", "4-4", "8-other"]
}
```

\- Replace the array values with every sub-category that applies to the conversation , you just analyzed.

CRITICAL FORMATTING INSTRUCTIONS:

\- Return ONLY the raw JSON object

\- Do NOT wrap the JSON in markdown code blocks

\- Do NOT include any markdown in your response

\- DO NOT DO NOT DO NOT include \n before entries in the JSON

\- Do NOT include any text before or after the JSON

\- Start your response with { and end with }

\- Ensure the JSON is valid and properly formatted

IMPORTANT: Return ONLY the JSON object. No other text, explanation, or formatting.

## Prompt Two: Strategies

## ## LLM Conversation Analysis Prompt

You are an LLM conversation analysis expert. You will analyze a multi-turn , conversation between a KPMG employee and an LLM to evaluate its prompting , strategies.

## ### PROMPTING STRATEGIES:

Identify which of these prompting strategies the user employed, capture all that <sup>,</sup>→ <sup>apply:</sup>

1. \*\*Role-playing\*\* (CODE: "role\_play"): User explicitly assigns the LLM a specific <sup>,</sup>→ <sup>identity,</sup> <sup>persona,</sup> <sup>or</sup> <sup>professional</sup> <sup>role</sup>

\- \*\*Unique marker:\*\* Direct identity assignment using "You are", "Act as",

, "Pretend to be", "Take the role of"

\- \*\*Key distinction:\*\* The LLM must be told to BE someone/something, not just

, write in a style

2. \*\*Few-shot\*\* (CODE: "few\_shot"): User provides concrete examples of desired , input-output pairs

\- \*\*Unique marker:\*\* Actual examples showing "if input is X, output should be Y"

\- \*\*Key distinction:\*\* Must show complete examples, not just templates or formats

3. \*\*Chain-of-thought\*\* (CODE: "chain\_thought"): User requests to see the reasoning , process or structured thinking

\- \*\*Unique marker:\*\* Requests for thinking/reasoning visibility: "show your work",

<sup>"explain</sup> <sup>your</sup> <sup>reasoning",</sup> <sup>"think</sup> <sup>step-by-step",</sup> <sup>"walk</sup> <sup>through</sup> <sup>your</sup> <sup>logic",,</sup>→

, "break this down", "analyze step-by-step", "walk me through", "explain your , analysis"

\- \*\*Key distinction:\*\* Focus is on exposing the cognitive process or structured , analysis, not just organizing information

4. \*\*Self-verification\*\* (CODE: "self\_verify"): User instructs LLM to check or

, validate its own output

- \*\*Unique marker:\*\* Commands to self-review: "double-check your answer", "verify   
, this is correct", "review for errors"

- \*\*Key distinction:\*\* The LLM must be told to evaluate its own work, not just be   
, careful

5. \*\*Interactive refinement\*\* (CODE: "interactive\_refine"): User requests

, collaborative back-and-forth to refine the task

\- \*\*Unique marker:\*\* "help me think through this", "ask me questions", "let's

<sup>figure</sup> <sup>this</sup> <sup>out</sup> <sup>together",</sup> <sup>"guide</sup> <sup>me</sup> <sup>through",</sup> <sup>"interview</sup> <sup>me</sup> <sup>about",</sup> <sup>"help</sup> <sup>me,</sup>→   
clarify my thoughts",

\- \*\*Key distinction:\*\* User seeks a collaborative dialogue rather than a response

## ### CLASSIFICATION RULES:

- Analyze the ENTIRE conversation, not just initial prompts

- A prompt can have MULTIPLE strategies, capture all that apply

- Only code strategies that are EXPLICITLY requested

- If none apply, mark as "none"

- Each strategy must have clear, unambiguous linguistic markers

## ### INTENSITY SCORING:

For each identified strategy, assign intensity:

- \*\*Light\*\* (1): Brief or minimal use

- \*\*Moderate\*\* (2): Clear but not elaborate use

```typescript
- **Heavy** (3): Extensive or sophisticated use
```

Example: "role\_play\_2" = moderate role-playing

## ## INPUT

You were or will be given a labeled conversation transcript with the following turn

,<sub>→</sub> types (in order as applicable):

1. System Prompt

2. Initial User Prompt

3. Initial LLM Response

4. Follow Up User Prompt (zero or more)

5. Follow Up LLM Response (interleaved with follow ups)

6. Final LLM Response (if present)

Prompts and responses alternate chronologically. Do not use LLM Responses for

<sup>,</sup>→ <sup>classifications</sup> <sup>except</sup> <sup>to</sup> <sup>provide</sup> <sup>context</sup> <sup>for</sup> <sup>follow-up</sup> <sup>user</sup> <sup>prompts.</sup>

## ## OUTPUT FORMAT

Return your complete analysis as a single JSON object:

{   
"strategies": ["role\_play\_1", "chain\_thought\_2", "interactive\_refine\_3", "none"]   
}

## CRITICAL FORMATTING INSTRUCTIONS:

\- Return ONLY the raw JSON object

\- Do NOT wrap the JSON in markdown code blocks

\- Do NOT include any markdown in your response

\- DO NOT DO NOT DO NOT include \n before entries in the JSON

\- Do NOT include any text before or after the JSON

\- Start your response with { and end with }

\- Ensure the JSON is valid and properly formatted

IMPORTANT: Return ONLY the JSON object. No other text, explanation, or formatting.

## Prompt Three: Other Factors

## ## LLM Conversation Analysis Prompt

## ## Instructions

You are an LLM conversation analysis expert. You will analyze a multi-turn

, conversation between a user and an LLM to evaluate various traits.

## ### CRITICAL CLASSIFICATION PRINCIPLES

1. \*\*Look at the ENTIRE conversation\*\*, not just initial prompts — user intent

, often clarifies through the dialogue

2. \*\*Effective clarification through dialogue should not be penalized\*\* — rate the

overall journey, not just initial clarity

3. \*\*Focus on the user's instructional language\*\* — Analyze how users frame

<sup>requests,</sup> <sup>not</sup> <sup>the</sup> <sup>content</sup> <sup>they</sup> <sup>submit</sup> <sup>for</sup> <sup>work.</sup> <sup>Exclude</sup> <sup>from</sup> <sup>analysis:</sup> <sup>quoted,</sup>→

text, documents being revised, or any material the user wants,

rewritten/translated/fixed. An exception to this principle applies to item 2,<sub>→</sub>

below.,

## ### EVIDENCE POLICY FOR USING LLM RESPONSES

Base ratings only on the user prompts (the Initial User Prompt and any Follow Up

User Prompts<sup>)</sup>. You may look at LLM responses solely to interpret what the user,<sub>→</sub>

is clarifying in subsequent follow-ups; do not use LLM response content as,

evidence for any rating or indicator.,

## ### QUALITATIVE RATINGS (1–5 Scale)

1. \*\*Initial Language Complexity\*\*: Rate linguistic complexity of the FIRST

substantive user prompt (based on wording and syntax, not task structure or,

domain expertise),

\- 1 = Very simple (common vocabulary, very short sentences, minimal modifiers)

\- 2 = Simple (basic vocabulary, mostly single-clause sentences, limited

, qualifiers)

- 1 = Unstructured text block

, spec; clear variable slots)

- 1 = No output guidance

2. \*\*Initial Complexity\*\*: Rate sophistication of the FIRST substantive user prompt   
, (based on structural complexity, not domain expertise)   
- 1 = Trivial (simple question like "What is X?")   
- 2 = Simple (straightforward task like "Explain Y" or "List Z")   
- 3 = Moderate (multi-part request like "Compare X and Y" or "How does A affect   
, B?")   
- 4 = Complex (sophisticated analysis like "Evaluate X considering Y constraints")   
- 5 = Very complex (multi-layered like "Analyze X considering Y constraints while   
, optimizing for Z")   
- Note: Skip standard greetings; rate first substantive request   
- \*\*Exception to critical classification principle #3\*\*: If the work product's   
inherent complexity directly impacts task difficulty (e.g., debugging quantum,   
, computing code vs. fixing a typo), note this context but still rate based on   
, how clearly the user articulates the task

- 3 = Moderate (some technical terms or qualifiers, multi-clause sentences,   
, standard punctuation)   
- 4 = Complex (dense phrasing, advanced vocabulary/jargon, nested clauses,   
,<sub>→</sub> frequent hedging<sup>)</sup>   
- 5 = Very complex (highly technical or jargon-heavy, convoluted syntax with   
, multiple embedded clauses)   
- Note: Skip standard greetings; assess the language form only—not the complexity   
, of the task requested   
- Per Principle #3: Exclude work product from analysis (see classification   
, principles)

3. \*\*Prompt Structure Quality\*\*: Organizational signals independent of prompting

- 2 = Minimal structure (single paragraph, few separators)

- 3 = Some structure (bullets or numbered steps; minor inconsistencies)

- 4 = Well structured (clear sections/headings, ordered steps, placeholders for   
, variables)

- 5 = Highly structured (sections such as context → task → constraints → output

\- Note: Do not evaluate strategy choices (no CoT/few-shot judgment here).

\- Per Principle #3: Exclude work product from analysis (see classification

4. \*\*Output Format Clarity\*\*: Clarity of requested output format/specification

- 2 = Vague guidance (e.g., "summarize")

- 3 = Basic format hints (e.g., "bullet list", "markdown")

\- 4 = Clear format spec (sections, headings, approximate length, audience/tone)

\- 5 = Precise spec (schema/template provided; acceptance criteria for correctness)

5. \*\*User Prompting Sophistication\*\*: Strategy-agnostic assessment of prompting

\- 1 = Naive (single vague request)

\- 2 = Basic (simple request with one clarification)

\- 3 = Intermediate (multi-part request, some constraints, uses structure)

\- 4 = Advanced (clear scoping, robust constraints, well-organized)

\- 5 = Expert (templates/variables, acceptance criteria, anticipates failure modes)

\- Note: Do not score specific strategies (few-shot/CoT) here.

\- Per Principle #3: Exclude work product from analysis (see classification

,<sub>→</sub> principles<sup>)</sup>

6. \*\*Specificity Score\*\*: Rate clarity and precision of user prompts across the

entire conversation

\- 1 = Very vague (unclear objectives, ambiguous requests)

\- 2 = Somewhat vague (general direction but lacking details)

\- 3 = Moderately specific (clear intent, some details missing)

\- 4 = Mostly specific (well-defined, minor ambiguities)

\- 5 = Very specific (precise, unambiguous, complete context)

\- Note: If user starts vague but clarifies well through dialogue, rate the overall

, journey positively rather than negatively rating the initial vagueness

## ### BINARY INDICATORS

7. \*\*Document Uploaded\*\*: Did the user upload any files/attachments?

\- Look for: File references, "uploaded", "the attached" or other language

, referring to an attachment

\- Exclude: reference to attachment is in quoted portion of text, rather than in

, the actual request (as described by classification principle #3).

8. \*\*Constraints Present\*\*: Did the initial substantive prompt include explicit

, constraints (length, schema, tone, audience, timeframe)?

\- true = includes at least one actionable constraint, i.e., a requirement that is

, measurable or structurally binding.

\- false = the prompt offers no constraints or has only generic style cues or vague

format hints such as “Be concise,” “shorten this,” “provide a simple x,” “use,

<sup>bullet</sup> <sup>points,”</sup> <sup>“brief</sup> <sup>overview,”</sup> <sup>“high</sup> <sup>level</sup> <sup>only,”</sup> <sup>“keep</sup> <sup>it</sup> <sup>short,”</sup> <sup>“make</sup> <sup>it,</sup>→

readable,” etc.,

9. \*\*Structured Output Requested\*\*: Did the user explicitly request structured data

## <sup>,</sup>→ <sup>output?</sup>

\- true = User explicitly asked for JSON, YAML, CSV, or Markdown tables with

, specific fields/schema

\- false = No explicit request for structured data output

\- Exclude: General code requests, unstructured lists, vague formatting requests,

, or keep response under x number of words.

10. \*\*Acceptance Criteria Provided\*\*: Did the user define success tests (e.g., must

, include X fields, match Y style, pass Z checks)?

11. \*\*User Satisfied\*\*: Did the user express explicit positive feedback?

\- Look for: "thanks", "perfect", "exactly what I needed", "great", "helpful"

\- Exclude: Neutral acknowledgments like "okay", "I see".

\- Key test: Positive feedback should be expected in the first sentence of a

, "Follow Up User Prompt", not later in that prompt, and not in the "Initial

User Prompt",

12. \*\*Frequent Typos\*\*: Does the user make frequent typing errors?

\- true = 3+ typos across the conversation OR 2+ typos in a single message

\- false = Fewer typos than the threshold

\- Count: Obvious misspellings, transposed letters, missing/extra characters

\- Exclude: Intentional abbreviations (ur, thx), domain-specific terms, proper

, nouns, grammar issues

13. \*\*All Lowercase\*\*: Does the user write entirely in lowercase?

\- true = 100% of user messages are entirely lowercase (excluding exemptions below)

\- false = User uses any capitalization in their messages

\- Exclude from analysis: URLs, file paths, email addresses, code snippets,

<sup>single-word</sup> <sup>responses,</sup> <sup>provided</sup> <sup>content</sup> <sup>as</sup> <sup>described</sup> <sup>by</sup> <sup>classification,</sup>→

principle #3.,

14. \*\*Pleasantries Used\*\*: Does the user include polite social expressions?

\- true = User includes at least one pleasantry in their own words

\- false = No pleasantries used

\- Look for: Greetings ("hello", "hi", "good morning"), closings ("thanks", "thank

you", "appreciate it"), politeness markers ("please", "could you", "would you,

mind"), social niceties ("hope you're well", "how are you"),<sub>→</sub>

\- Exclude: Pleasantries within provided content as described by classification

, principle #3.

## ### ANALYSIS GUIDELINES

\- Analyze the ENTIRE conversation (typically 2–20 turns)

\- Consider the overall journey, not just initial state

\- Be consistent in ratings across conversations

## ### FALLBACK BEHAVIOR

, inventing data.

## ## Input

You were or will be given a labeled conversation transcript with the following turn

, types (in order as applicable):

1. \*\*System Prompt\*\*

2. \*\*Initial User Prompt\*\*

3. \*\*Initial LLM Response\*\*

4. \*\*Follow Up User Prompt\*\* (zero or more)

5. \*\*Follow Up LLM Response\*\* (interleaved with follow ups)

6. \*\*Final LLM Response\*\* (if present)

Prompts and responses alternate chronologically. Base ratings only on the user , prompts; use LLM responses only for context of follow-ups.

```markdown
## CRITICAL FORMATTING INSTRUCTIONS
- Return ONLY the raw JSON object
- Do NOT wrap the JSON in markdown code blocks
- Do NOT include any markdown in your response
- DO NOT include \n before entries in the JSON
- Do NOT include any text before or after the JSON
- Start your response with { and end with }
- Ensure the JSON is valid and properly formatted
```

## ## REQUIRED JSON SCHEMA

Provide the output as a single JSON object with exactly these keys and value types (listed in the same order as the rubric above):

## { {

"initial\_language\_complexity": <integer 1-5>,   
"initial\_complexity": <integer 1-5>,   
"prompt\_structure\_quality": <integer 1-5>,   
"output\_format\_clarity": <integer 1-5>,   
"user\_prompting\_sophistication": <integer 1-5>,   
"specificity\_score": <integer 1-5>,   
"document\_uploaded": <boolean>,   
"constraints\_present": <boolean>,   
"structured\_output\_requested": <boolean>,   
"acceptance\_criteria\_provided": <boolean>,   
"user\_satisfied": <boolean>,   
"frequent\_typos": <boolean>,   
"all\_lowercase": <boolean>,   
"pleasantries\_used": <boolean>   
}