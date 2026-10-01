# Community-Driven API and AI Writer Design for Openly Scaling Community Notes

Brad Miller Community Notes, SpaceXAI

Keith Coleman Community Notes, SpaceXAI

Jay Baxter Community Notes, SpaceXAI

Sophie Hilgard Community Notes, SpaceXAI

Jiansong Chao Community Notes, SpaceXAI

Daniel Ortiz Community Notes, SpaceXAI

## Abstract

Community Notes is a crowd-sourced approach for adding context to posts on X. Contributors propose and rate notes, forming the inputs to an open-source, open-data algorithm that determines which notes show broadly to users. Since September 2025, Community Notes’ AI Note Writer API<sup>1</sup> has provided an open, public interface for using AI to propose notes, while adhering to the founding prin ciple that users, not the platform or an AI, control which notes show on X. Explicit note requests and user posts on X determine the AI API post feeds, ensuring that AI note writing responds to demand from X users. We present the design, operation and impact of the AI API, including analysis of the interaction between AI and human generated notes across topics. Unless otherwise stated, measurements and system description reflect June 2-29, 2026.

The Community Writer is the largest AI API client and contributes the bulk of AI API output, generating 52% of notes selected as Helpful and shown broadly on X. The writer is guided by com munity input during both training and operation to prioritize, draft, evaluate and delete proposed notes. Beyond scale, the writer also ofers speed, submitting the first proposed, non-deleted note on 60% of posts when compared to other writers. AI note writing is additive on top of human note writers, extending coverage of Community Notes on X. Among posts that have Helpful notes, 42% have only AI notes, indicating human raters did not feel motivated to propose an alternative. In contrast, 30% have only human notes, reflecting contribution beyond the scope of AI writing. The Community Writer is open-source software released under the Apache 2.0 license.

## Keywords

Community Notes, fact-checking, large language models, content moderation, misinformation, crowdsourcing

## 1 Introduction

Community Notes began in January 2021 [4] as a pilot program developing a crowd-sourced approach to addressing misleading information. Operating without centralized control, Community Notes relies on contributors to propose and rate notes that add context to social media posts. An open-source, open-data algorithm processes contributor ratings to identify consensus among users who normally disagree [19, 26]. The algorithm assigns notes either Currently Rated Helpful (CRH) or Currently Rated Not Helpful (CRNH) status, or Needs More Ratings (NMR) if no consensus is reached. Notes that are CRH show broadly on X, including to non-contributors, alongside the original post.

Prior work has found Community Notes to be both accurate [1] and efective at reducing sharing of misleading posts [3, 17], lead ing to transformational efects on social media. Meta, TikTok and YouTube have incorporated Community Notes style approaches [13, 18, 27], with Meta ending third-party fact-checking in the United States and introducing Community Notes [14]. Proposing notes requires efort, which impacts note supply and creates latency when using Community Notes to address misleading information [3].

On September 2, 2025, Community Notes admitted the first AI writer via the AI API [20]. The AI API allows the public to operate AI writers that consume feeds of candidate posts and propose notes, yielding an open, decentralized approach to increase the supply of timely notes. The content of note feeds is emergent from user engagement on X, including the explicit Request a Community Note feature [24] and user posts on X (e.g. @grok is this true?). Since launch, AI writers have helped increase daily CRH note volume by 86.0%.<sup>2</sup> Importantly, the API changes how notes can be written, not how they are shown: as with human authored notes, contributors decide via ratings which notes show broadly on X [11].

The Community Writer (CW) is the largest client of the AI API, generating 41% of proposed notes on X.<sup>3</sup> The CW combines taskspecific, fine-tuned and commercially available models to generate proposed notes and filter candidates for submission. The CW integrates community input throughout the design to yield higher quality notes that are more likely to be found Helpful by users. SpaceXAI has released the CW as open-source software under the Apache 2.0 license.<sup>4</sup>

This work provides the following contributions:

• Openness Community Notes AI API, including admissions process, quota allocation and feed hierarchy, allowing the public to operate AI Community Note writers on X.

• Scale Eficient mechanisms to prioritize, draft, evaluate, and delete proposed notes, economizing contributor time to yield 52% of CRH notes while consuming 22% of ratings.

• Speed AI API client writing first non-deleted note on 60% of posts with a proposed note from any other writer, allowing increased visibility yielding 55% of CRH note views.

The remainder of the paper is organized as follows: Section 2 presents prior work, Section 3 presents the AI API, and Sections 4 and 5 present the Community Writer. Section 6 evaluates the API and writer, with discussion and conclusion in Sections 7 and 8.

## 2 Background and Related Work

The majority of related work has focused on evaluating the accuracy and efects of Community Notes. Allen et al. examine the accuracy of Community Notes on X in relation to COVID-19 and vaccines, finding that 97% of examined notes were entirely accurate [1]. Slaughter et al. examine the efects of Community Notes on difusion of misleading information on X, including a 46% drop in reposts after a Community Note is attached [17]. Similarly, Chuai et al. found that Community Notes reduce the spread of misleading posts by 62% on average, and increase the odds that users delete misleading posts by 103% [3]. Drolsbach et al. demonstrate that the contextualization of Community Notes, specifically tailoring note content to the misleading post, increases user trust in fact-checks relative to more generic misinformation flags [7].

Other works have examined the relationship between communitybased and professionally generated, third-party fact-checks. Borenstein et al. analyze the content of Community Notes, finding that <post, note> pairs addressing previously documented misleading information are twice as likely to cite fact-checking sources compared to other sources [2]. Similarly, Zhao and Naaman compare community-based and professional fact-checks, and find that com munity approaches ofer a speed advantage and tend to build on professional fact-checks for common misleading information topics [28].

Several works have examined AI approaches to fact-checks, but evaluate outside of production environments. Singh et al. develop an LLM-based approach to fact-checks, but evaluation relies on ofline comparison with reference fact-checks and two human evaluators rather than actual user engagement [16]. Zhou et al. present a fact-checking pipeline, although the evaluation relies on expert evaluation and user study rather than production deployment [29]. De et al. develop Supernotes, an approach for aggregating proposed Community Notes to synthesize a single superior note, and evaluate in an ofline environment with recruited study participants [6].

In contrast to prior work which has focused on generating individual fact-checks, we present the operation of an open API that drives added context across an entire platform.<sup>5</sup> We also present the design of the largest API client, including open-source infrastructure and modeling components that support operation at scale. Li and Bakker also present and evaluate an API client, although the evaluation is scoped to comparison with human contributors and does not include other AI writers. The evaluation finds that the client operates with higher latency relative to human note writers but that note quality outperforms human averages when controlling for diferences in rating exposure [10]. In a passive analysis, Mantzarlis finds that AI note writers maintain higher pooled average and median CRH rates compared to human writers [12]. Our evaluation introduces additional measurements related to latency, quality, and quantity across topics, including diferentiating top human contributors and comparison against other API clients.

## 3 Community Notes AI API Overview

The Community Notes AI API exposes five functions, allowing clients to request candidate posts, evaluate potential notes, submit proposed notes, delete notes, and obtain real-time note status and ratings. While the purpose of the API is to increase the timely supply of Community Notes,<sup>6</sup> the API design must also consider the needs of API clients and the contributors that rate AI proposed notes. To support contributors, the API includes both an earn-in process by which clients gain the ability to publish notes for contributor review and a quota system that throttles note creation in proportion to note status outcomes. To support clients, the API structures candidate posts into feeds based on X user note requests, such that the smaller feeds contain posts with the highest demand where notes are most likely to achieve CRH status.

The AI API exists as a set of endpoints within the broader X API, which includes functionality to search, retrieve, and publish posts and associated metadata [22]. Creating an AI note writer requires signing up for the X API and AI Note Writer API, as well as accepting the X Developer Terms [20]. AI note writers must be associated with an X account that is not already a Community Notes contributor and has a verified email and phone number.

Newly created API clients must earn the ability to publish notes that collect ratings. AI note writers earn in by fulfilling three note quality checks based on their 50 most recently submitted notes. The URL validity check requires that at least 95% of notes had URLs that resolved to a 200 HTTP status code after any redirects. The API also requires that at least 98% of notes are not classified as harassment or abuse by an open-source classifier trained on human generated Community Notes and rating tags [21].

The final check applies the ClaimOpinion model, which supports the note evaluation endpoint [23]. The ClaimOpinion model predicts whether a note will likely be viewed as Helpful based on whether it addresses the claims in a post and does so without imparting opinion. Training for the ClaimOpinion model is open source [21], fine-tuning DistilRoBERTa-base [8] to predict Community Notes rating tags and note status outcomes. Completing earn-in requires that at least 30% of submitted notes score above the bottom 20% of CRH note scores, and no more than 30% of submitted notes receive scores below the bottom 5% of CRH note scores.

After completing earn-in, AI writers are subject to a quota system that throttles submissions based on note status outcomes. In general, AI writers are rewarded with higher quota in exchange for higher hit rate, defined as (CRH-CRNH) / (CRH+CRNH+NMR), where CRH, CRNH and NMR denote quantities of notes with corresponding status over a recency window. To avoid penalizing writers for submitting notes on lower visibility posts, the 14-day hit rate calculation excludes notes that have fewer than 10 ratings and have not achieved CRH or CRNH status. Appendix A presents the quota algorithm, which includes edge cases addressing sudden drops in quality, new writers, and sudden increases in note writing. Submission quota is non-linear: a 10% hit rate allows publishing 50 notes / day, but a 20% hit rate allows the maximum of 500 notes / day. The quota system forces AI writers operating at scale to outperform human writers, who average a 10.4% CRH rate and 9.0% CRNH rate on notes that have at least 10 ratings or CRH or CRNH status.

Given that note submission is limited, the API supports clients by structuring candidate posts into diferent feeds according to the amount of user signal that the post is misleading. Table 1 presents four feed sizes: Small, Large, XL, and XXL. The API assigns posts to feeds based on explicit use of the Request a Community Note feature, implicit signal inferred from other user engagement (e.g. @grok is this true?), and requester helpfulness scores derived from the outcomes of previous requests [24]. Posts first appear in the XXL feed and work toward smaller feeds as signal accumulates, with the Small feed yielding the highest concentration of posts that will receive CRH notes but also the highest latency.

<table><tr><td>Feed</td><td>Explicit Signal</td><td>Implicit Signal</td><td>Daily Posts</td><td>Min Hit Rate</td><td>CRNH Rate Max</td><td>90d Net CRH</td></tr><tr><td>Small</td><td>Multiple</td><td>None</td><td>428</td><td></td><td></td><td></td></tr><tr><td>Large</td><td>Multiple</td><td>None</td><td>2,308</td><td>5%</td><td>10%</td><td></td></tr><tr><td>XL</td><td>Single</td><td>None</td><td>6,489</td><td>5%</td><td>10%</td><td></td></tr><tr><td>XXL</td><td>None</td><td>Single</td><td>24,389</td><td>5%</td><td>10%</td><td>100</td></tr></table>

Table 1: Explicit and implicit columns indicate post inclusion criteria. Min Hit Rate, CRNH Rate Max, and 90d Net CRH qualify clients for feed access.

Clients gain access to larger feeds with lower density of misleading posts based on submitted note status outcomes. Access beyond the Small feed requires at least 100 notes written, a hit rate of at least 5% and a CRNH rate of at most 10%. Accessing the XXL feed also requires a net CRH, defined as CRH-CRNH, of at least 100 in the last 90 days. Feed qualification requirements ensure that client access remains in proportion with benefit to X users.

The combination of feed and quota design yields a system that provides as many CRH notes to X users as possible, while economizing the contributor ratings that make Community Notes possible. As the amount of user demand signal decreases with progressive feeds, the quota system forces clients to exercise discernment in when to submit a note. Clients that accurately predict when notes will achieve CRH status receive increased quota, which can be applied to process larger and increasingly challenging feeds.

Since launch in September 2025, engagement with the AI Note Writer API has grown. The clients include academic researchers [10], independent developers [5, 15], and the CW, which operates with multiple accounts to support experimental features and models while maintaining a reliable production system.<sup>7</sup> Figure 1 presents engagement of API clients. Of the 65 client accounts that began earn-in, 32 ultimately earned writing ability and submitted a note for contributor ratings, and 19 of those accounts remain active. Among the accounts that did not earn in and begin publishing notes, 13 did not submit the 50 notes required to complete earn-in, 5 failed ClaimOpinion quality checks, and 15 completed earn-in but never published notes for contributor review.

## 4 Community Writer Overview

The CW combines task-specific classifiers based on numeric and categorical features with fine-tuned and commercially available models to prioritize, draft, evaluate, and delete proposed notes. At each stage, CW components leverage community inputs during both training and inference to yield results that maximize CRH notes available to X users.

![](images/4bd4a48a2b4ea89aa73aa7071f92296cadc128d49e3a1106fe2108e7ca28d189.jpg)  
Figure 1: Roughly half of created clients earn-in and publish a note. After publishing, most clients remain active.

By design, the CW runs separate from SpaceXAI internal infrastructure and faces the same operational constraints as any other AI note writer. In particular, the CW does not access any note writing inputs or interface other than the publicly available AI API. Similarly, the CW is subject to the same earn-in and submission quota requirements that apply to all AI note writers. After submission, any proposed notes are subject to the same open-source, open-data scoring algorithm to determine helpfulness status that applies to all other AI and human note writers.

## 4.1 Design Goals

The CW design reflects several goals:

Community Driven. The CW should be both reflective of and responsive to Community Notes contributors. In particular, note production and deletion should be guided by contributor engagement and tuned to maximize CRH notes shown on X. Similarly, the CW should prioritize posts based on X user engagement.

Scalable with Low Latency. The CW should be able to scale note production to match the availability of feed posts and contributor ratings. Correspondingly, the CW should minimize both the queueing latency between retrieving and processing candidate posts and the processing latency required to generate notes.

Economize Ratings. Among NMR notes, 54% have fewer than 10 ratings, which is the efective minimum required to reach CRH status.<sup>8</sup> Recognizing that contributor attention is a finite resource, the CW should seek to maximize the display of CRH notes relative to the contributor ratings required.

## 4.2 Design Overview

The CW combines modeling and queueing components with AI API calls to manage note creation, revision, and deletion. Figure 2 presents a full diagram of pipeline components. We present an overview below and detail individual components in Section 5.

The CW polls the AI API for candidate posts and assigns the posts to last-in, first-out (LIFO) work queues. Work queue assignments reflect a combination of feed size and the Notable Post Model (NPM), which identifies the posts where proposed notes are most likely to earn CRH status. The NPM is trained on Community Notes data, including features based on content and user engagement available on the X API. LIFO work queues are serviced in a fixed order, with priority given to the smallest feeds and highest NPM scores. The queue structure ensures that the CW responds well in the event of any backlog, maintaining the lowest queueing latency for posts with the most user demand and CRH note potential.

![](images/2139f0191b0ec582462a283164366d46c49915b9dcb96642ba42bb5bc9196f67.jpg)  
Figure 2: The Community Writer. Writing begins with fetching and prioritizing posts, followed by drafting and evaluating notes before submission. Feedback loops monitor contributor ratings to delete or revise proposed notes.

Writers service queues, researching posts and generating candidate notes which rejectors evaluate and filter for submission. A writer may decline to propose a note if the writer concludes that the post does not require added context or suficient sourcing is unavailable. The CW samples from both Grok 4.20 and CN Grok, a custom Grok model with specialized post-training to improve research, judgment, and presentation based on Community Notes data. After sampling up to three candidate notes from each writing model in parallel, the CW selects the single best note from each writer using the ClaimOpinion model exposed on the AI API. The Helpfulness Rejector, Recent Context Rejector, and Screenshot Rejector evaluate note quality, necessity, and sourcing to detect notes that are unlikely to achieve CRH status. If both Grok 4.20 and CN Grok generate a note that passes all rejectors, then the CW will submit the CN Grok note and may submit the Grok 4.20 note.<sup>9</sup>

Beyond the linear flow outlined above, several feedback loops support note coverage and quality. The Retry Queue allows a second writing attempt on posts that were initially processed within 45 minutes of post creation. After a 45 minute delay, during which X users may increase engagement and leave suggestions in note requests, the NPM will re-score the post, and writers may attempt to generate proposed notes. Likewise, the Revision Queues repeat the note writing process after a 1 hour and 3 hour delay for posts where the CW previously published a note that has not yet achieved CRH status. The Revision Rejector reviews revisions and passes candidates that are meaningfully diferent from the original. Lastly, the Deletion Model monitors note ratings in real-time and calls the AI API to delete proposed notes that are unlikely to achieve CRH status.<sup>10</sup> The rejectors and Deletion Model both function to economize ratings, as the CW both avoids and deletes underperforming notes.<sup>11</sup>

## 5 Pipeline Component Details

This section presents the NPM, writers, rejectors, and Deletion Model in more detail. We present performance of the NPM and Deletion Model in isolation, and evaluate the entire CW in Section 6.

## 5.1 Notable Post Model

The NPM predicts whether a post will receive a CRH note based on a combination of post content and metadata. The NPM sources training posts from the API feeds to eliminate any distribution skew between training and production data and labels posts according to whether there was a CRH note, regardless of writing source. Since achieving CRH status efectively requires 10 ratings, the NPM identifies some posts as unlikely candidates for a CRH note given the visibility of the post. Pruning low visibility posts avoids collecting any ratings on posts that are fundamentally unlikely to yield a CRH note, consistent with the goal of economizing contributor ratings.

The NPM combines several families of features. Post engagement features reflect user actions on X, including reposts, replies, likes, quotes, bookmarks, and views. Author features capture number of followers and follows, posting history, and Community Notes history, including number of prior notes, ratings received, and status outcomes. Author features also include the verification status and whether the author is a known parody account. Post context features include age, language, time of day the post was created and enqueued, presence of media content, and a text embedding of the post using all-mpnet-base-v2 [9]. Note that all features are either obtained through the publicly accessible AI API or constructed based on the CW’s history of note submissions and status outcomes.

The NPM features are time sensitive and often interact. To avoid skew, the CW records all features each time any post appears in any API feed, allowing subsequent training on temporally accurate feature data. To capture feature interactions, feature pre-processing generates feature variants with engagement counts normalized by both post age and impression count. After preprocessing, the model uses a multilayer perceptron architecture with a single hidden layer to capture additional interactions among the features. The full feature extraction and training code of the NPM is available in the open-source release.

The NPM is most accurate on the largest feed sizes, where concentration of posts likely to receive a CRH note is lowest. Figure 3 shows how well the NPM decides which posts are worth trying to write a note on, with higher AUCs indicating better predictions of which posts will receive a CRH note. Since the CW integrates the NPM, Figure 3 excludes notes generated by the CW and evaluates the NPM based on notes from other writers. Note that in this framing recall measures the fraction of posts without CRH notes that can be dropped from the CW, while false positives capture regretted pruning of posts that ultimately do receive a CRH note.

![](images/48019c51e7ad3481e3f6435ff34eb3d91ea042e6c0107d09c09cc5a45534743c.jpg)  
Figure 3: Since larger feeds require note requests from fewer users, the distribution shifts towards posts which are more easily identified as unlikely to receive a CRH note.

## 5.2 Writers and Rejectors

The CW contains two distinct writing models to improve proposed note coverage. Both models run with reasoning enabled and tooling to access web content, including search, URL, and image retrieval. Both models also have full access to public content on X, including ability to search X and fetch posts, user profiles, images, and video. The full prompts for both writing models are available in the opensource release.

CN Grok Writer uses a relatively brief prompt because the model behavior has been optimized through post-training specifi cally to research and write Community Notes. The training process uses inputs about what people previously found helpful so that the model produces proposed notes people will find helpful. The prompt includes the post to analyze, as well as brief guidance on the qualities of a Community Note, an option to decline if the post is not misleading, and guidance to always match the post language. The CN Grok prompt also includes an aggregated list of the top suggested sources supplied in note requests from X users. Suggested sources are links to X posts that the user feels provide context on why the post is misleading.

Grok 4.20 Writer uses a prompt that is comparatively detailed. The prompt provides guidance for the model to consider whether the post is misleading, evaluate whether any inaccuracies are substantive such that readers will likely value a Community Note, and potentially draft a proposed note to correct any inaccuracies. The prompt also includes a set of criteria to refine proposed notes, promoting notes that are convincing, concise, objective, and well sourced.

The four rejectors included in the CW rely on a mix ofcustomized models and in-context learning to filter notes that are unlikely to achieve CRH status. The CW collects 5 samples each from the Helpfulness Rejector and Recent Context Rejector, while sampling for the Screenshot Rejector varies conditioned on outcomes. The full prompts for all four rejectors are available in the open-source release.

Helpfulness Rejector is a customized Grok model with reasoning and the same tool use abilities as the writing models. The model has been customized during post-training on Community Notes data to predict whether a note is likely to be CRH in the context of a particular post. The CW customizes the rejection threshold based on the post feed size and NPM prediction such that the rejector is more permissive as user demand for a note increases.

Recent Context Rejector filters notes based on similarity to recent note status outcomes to avoid repeated mistakes. The rejector consists of a prompt that includes the post, proposed note, the 100 most recent CRH notes, and the 100 most recent notes that were either CRNH or deleted based on contributor ratings. Since Community Notes releases note status data with a 48 hour delay, the recent notes are sourced from prior note submissions made by the CW. The prompt guides the model to only reject notes that match specific, identifiable patterns in prior CRNH or deleted notes.

Screenshot Rejector scrutinizes sourcing for proposed notes on posts with media. The rejector receives as input a high resolution screenshot of the post, including any media, as well as high resolution, full-page screenshots of any source links. If the post includes video, then the input includes multiple screenshots captured at intervals throughout the video. The prompt includes clear guidance on the extent to which media in the post must match media in the source link screenshots. For example, if a note claims that the post includes media that was taken out of context, then the sourcing must contain media that is an exact match to the post to establish the original context for the media. The CW initially queries the Screenshot Rejector once. If the initial query fails, then the CW issues three additional queries, all of which must pass for the proposed note to pass the Screenshot Rejector.

Revision Rejector economizes contributor ratings by moderating whether multiple notes can be proposed for a single post. The Revision Rejector prompt includes the post, original note, and revised note, and guides the model to systematically identify and evaluate deltas between the notes. Additions should introduce new claims, improve sourcing, or strengthen the argument, and removals should make the note more concise, focused, or correct. The rejector prompt requests a continuous score, which is then scaled by a sequence matching similarity metric applied to the proposed notes with URLs removed.

## 5.3 Deletion Model

The Deletion Model identifies notes that are unlikely to obtain CRH status and proactively deletes the notes. Deleted notes no longer show to contributors on X, helping to economize ratings, but do still count against note submission quota and the daily writing limit calculation.

The Deletion Model depends on the AI API, which exposes functionality to retrieve real-time note status and ratings for prior submissions. Access to real-time ratings requires fulfilling the same requirement of at least 100 net CRH notes in the last 90 days that applies to the XXL feed. Real-time rating information includes counts of Helpful, Somewhat Helpful, and Not Helpful ratings as well as rating tags, which capture detailed note qualities (e.g. insuficient sourcing, misses key points, etc.). The API aggregates helpfulness and tag counts into three bins corresponding to positive, neutral, and negative values of the raterfactor, a continuous value learned from past contributor ratings that represents contributor viewpoint.

<table><tr><td></td><td>Posts</td><td colspan="2">Age (min.)</td><td colspan="2">Feed Coverage</td><td colspan="2">NPM Pass</td><td colspan="2">NPM Coverage</td><td>Posts</td><td colspan="2">Draft Note</td><td colspan="2">Submitted</td><td colspan="2">CRH</td></tr><tr><td>Feed</td><td>Feed Size</td><td>p10</td><td>p50</td><td>X Views</td><td>H-CRH</td><td>n</td><td>%</td><td>X Views</td><td>H-CRH</td><td>Un-Noted</td><td>n</td><td>%</td><td>n</td><td>%</td><td>n</td><td>%</td></tr><tr><td>Small</td><td>11,995</td><td>125</td><td>604</td><td>0.9%</td><td>24.3%</td><td>11,995</td><td>100.0%</td><td>0.9%</td><td>24.3%</td><td>8,613</td><td>6,938</td><td>80.6%</td><td>1,969</td><td>28.4%</td><td>206</td><td>10.5%</td></tr><tr><td>Large</td><td>64,627</td><td>78</td><td>597</td><td>2.4%</td><td>52.5%</td><td>55,840</td><td>86.4%</td><td>2.4%</td><td>51.4%</td><td>52,091</td><td>41,922</td><td>80.5%</td><td>12,496</td><td>29.8%</td><td>988</td><td>7.9%</td></tr><tr><td>XL</td><td>181,702</td><td>52</td><td>529</td><td>5.0%</td><td>70.3%</td><td>98,858</td><td>54.4%</td><td>4.6%</td><td>64.9%</td><td>96,869</td><td>69,857</td><td>72.1%</td><td>14,124</td><td>20.2%</td><td>1,361</td><td>9.6%</td></tr><tr><td>XXL</td><td>682,889</td><td>9</td><td>185</td><td>8.1%</td><td>73.4%</td><td>116,649</td><td>17.1%</td><td>5.8%</td><td>55.8%</td><td>116,599</td><td>76,787</td><td>65.9%</td><td>8,981</td><td>11.7%</td><td>946</td><td>10.5%</td></tr><tr><td>Retry</td><td>182,143</td><td>一</td><td>一</td><td>一</td><td>1</td><td>14,253</td><td>7.8%</td><td>一</td><td>1</td><td>14,253</td><td>8,353</td><td>58.6%</td><td>1,117</td><td>13.4%</td><td>58</td><td>5.2%</td></tr><tr><td>Revise</td><td>37,388</td><td></td><td></td><td>1</td><td></td><td>36,632</td><td>98.0%</td><td></td><td>1</td><td>0</td><td>35,993</td><td>98.3%</td><td>1,363</td><td>3.8%</td><td>78</td><td>5.7%</td></tr></table>

Table 2: CW outcomes per post across AI API feeds. The Notable Post Model (NPM) filters more posts on larger feeds, while maintaining coverage of posts with human CRH (H-CRH) notes. Un-Noted refers to posts with no CW proposed note. The NPM, writers and rejectors in combination allow similar CRH rates per post with one or more submitted notes across feeds.

![](images/4a8d5812a3c5803cd98a65bf988661e49f2498736a5c3cee30475005810cb373.jpg)  
Figure 4: Deletion Model performance improves as ratings accumulate. The CW applies the Deletion Model every 60 seconds as soon as 3 ratings are available.

The Deletion Model predicts note status using logistic regression applied to the rating count aggregates. The feature preparation includes both discretization and polynomial crosses, which capture interactions between features. Since the CW applies the Deletion Model in real-time as ratings arrive, the training process avoids using the full set of ratings on historical notes. Rather, the training data preparation extracts features up to 10 times at intervals spaced through the progression of the first 30 ratings, allowing production data to remain in-distribution as ratings accrue.

Figure 4 presents the Deletion Model performance on evaluation data, including separate curves reflecting how many ratings were used in the prediction. Deletion Model predictive accuracy improves as more ratings become available. In production, the CW polls ratings every 60 seconds and applies the model as ratings arrive. The CW includes an elevated deletion threshold, allowing deletion based on 3 or 4 ratings when the model is highly confident, as well as a standard threshold applied when 5 or more ratings are available. The full feature extraction and training code of the Deletion Model is available in the open-source release.

## 6 Evaluation

This section evaluates AI writing over a 28 day period from June 2, 2026 through June 29, 2026.<sup>12</sup> We begin by profiling the API feeds and progression of each feed through the CW, including feed size, NPM efects, and submission outcomes. Then, we measure the eficacy of proposed notes from the CW, including comparison with other AI and human writers. We find the CW has been impactfu and eficient, accounting for 41% of proposed notes, 52% of CRH notes and 55% of CRH note views while consuming only 22% of contributor ratings. We also find that AI and human writers have been broadly complementary: among posts with a CRH note, 42% have no human proposed note and 30% have no AI proposed note.

Recall that the API includes four feed sizes, each characterized by distinct inclusion criteria. Table 2 presents the size, latency, and post coverage for each feed size. Decreased inclusion signal requirements for larger feeds yield exponential growth, with post volume increasing approximately 3-5x across each feed size. Correspondingly, posts also appear in larger feeds faster: the median post age decreases from 604 to 185 minutes from the Small to XXL feed, with 10% of posts appearing in the XXL feed in 9 minutes or less.

Larger feeds also yield increased coverage of both posts with human written CRH notes and X platform content as a whole. Since human contributors are able to submit notes on any post, we present the feed coverage of posts with human CRH (H-CRH) notes as a measure of how well the feed has covered posts that may receive a Community Note. The Small feed contains a relatively dense concentration of misleading posts, covering 24.3% of H-CRH notes while averaging 428 posts daily. By comparison, the XXL feed is \~57x larger, and increases coverage of H-CRH notes \~3x to 73.4%. Note that all feeds skew towards content which has high visibility on X, with the XXL feed covering posts generating 8.1% of X post views daily.

The CW accommodates diferences in feed characteristics by customizing thresholds to each feed size. To maintain a high CRH rate, the CW uses higher thresholds for the ClaimOpinion evaluator and Helpfulness Rejector when operating on larger feed sizes, which reflect lower levels of user demand for proposed notes. Rejection thresholds for the XL and XXL feeds are also segmented by the NPM, with higher thresholds applied in cases where the NPM was less confident the post may receive a CRH note.

Note Writer Contribution and Consumption

![](images/a0e8ccdbb12b33b26a19dbda81efaa8e4c45d49e8558cd582c53106a847a1b5b.jpg)  
Figure 5: The CW maintains above-average CRH rate and impressions per CRH note, and does so while consuming below-average amounts of contributor ratings.

Due to diferences in both post characteristics and pipeline configuration, the rate of post progression through the CW varies for each feed. Table 2 presents the progression of each feed size. The influence of the NPM varies with feed size: the NPM is disabled for the Small feed, but passes only 17.1% of posts in the XXL feed. Simultaneously, the NPM efectively retains the majority of posts that receive a human CRH note, with H-CRH coverage on the XXL feed decreasing from 73.4% to 55.8% as a result of NPM filtering. After NPM filtering, the frequency of writers generating a draft note decreases with larger feed sizes, reflecting the decreased concentration of misleading posts in larger feeds. Rejectors ultimately mediate submission, yielding CRH rates between 7.9% and 10.5% for each API feed size.

The remainder of the evaluation deals with comparison between the CW and other AI and human writers. The evaluation diferentiates performance of top writers as designated by X Community Notes, which requires achieving a lifetime hit rate of at least 4% and net CRH of at least 10. Top writers enjoy several special abili ties, including priority in rating notifications sent to X Community Notes contributors and visibility into note requests [25]. Achieving top writer status is relatively dificult: only 3.5% of human writers active during the evaluation period meet the top writer criteria.

The CW demonstrates improvements to proposed note CRH rates, views of CRH notes on X, and contributor ratings consumed. Figure 5 details the distribution of proposed and CRH notes, contributor ratings, and CRH note views across contributor segments. Notice that the CW accounts for 41% of proposed notes but 52% of CRH notes, demonstrating an above average rate of achieving CRH status. CRH notes from the CW also account for 55% of CRH note views on X, reflecting reduced note creation latency relative to other sources. Simultaneously, the CW consumes only 22% of ratings, economizing use of contributor time.

The NPM and rejectors allow the CW to scale by improving the CRH yield on submitted notes. Figure 6 presents the density of posts with human CRH notes in each feed and the distribution of feed adoption by diferent writers. AI writers other than the CW submit 64% of notes from the Small feed and 31% of notes from the Large feed, which have a higher density of posts with human CRH notes than the XL or XXL feeds. The NPM and rejection mechanisms accommodate the class imbalance of the XL and XXL feeds, allowing the CW to process larger feeds without exhausting note writing quota. Figure 6 provides comparisons to human writers as a reference; humans are free to write on any post at any time.

![](images/98a38e09540d3ab2efe6d9cf5803e8fd2a4ec094e2eaa84d336baab7841e4670.jpg)

![](images/c5882f6648f612e1f605e42f6274ed683d79a089a59f2f4b1c568412206477fd.jpg)  
Figure 6: Decreased density in larger feeds complicates scaling, which requires progressively precise prediction of whether a draft note will achieve CRH status.

![](images/401199e2036e347d40d3b4f4a32d4f336c2bfa3489f051377ac8dd95025d8f5a.jpg)  
Figure 7: Stable notes include any note that is CRH, CRNH, deleted or meets a rating minimum. Conditioned on stability, the CW outperforms on both CRH and low quality outcomes.

Despite processing feeds with a lower density of misleading posts, CW proposed note quality compares well with other AI writers and humans. The Per Note portions of Figure 7 present the rates of CRH and low quality outcomes over all proposed notes.<sup>13</sup> We define low quality outcomes as any note which is either CRNH or deleted by the author. The Deletion Model results in higher deletion rates and lower CRNH rates since many notes are deleted before reaching CRNH status. For comparison, Figure 7 simulates applying the Deletion Model to all writers using historical data.

Variations in the visibility of posts complicate comparison of note status outcomes, which efectively require 5 and 10 ratings to reach CRNH and CRH status respectively. Figure 7 presents outcomes for stable notes, defined as any note that is CRH, CRNH, deleted by the author or has at least 5 ratings (low quality notes) or 10 ratings (CRH notes). Measurement over stable notes increases the rates of CRH and low quality outcomes across all writers. The CW maintains the highest CRH rate<sup>14</sup> despite having more deleted notes, many of which are below rating thresholds.

![](images/2dc8aaee72e6f3ce00e6c0f8714208a0bea8d6b119af34c46e687069f188586b.jpg)  
Figure 8: Compared to other writers, the CW submits the first proposed, non-deleted note on 60% of posts.

Since note status depends on contributor willingness to supply ratings, Figure 7 also presents the rate of status outcomes per rating. Measuring outcomes per rating captures the experience of contributors, reflecting the likelihood that the efort to make a rating yields a CRH note. The CW ofers a 29% likelihood of ratings leading to a CRH note, compared to 17% for notes written by top writers.

Operating on larger feeds improves latency, as decreased user signal requirements allow posts to enter feeds sooner. Figure 8 presents the amount of lead time by which CW proposed notes precede proposed notes from other sources, including the efect of CW deletions. By design, the plot can only include posts where there is a proposed note from both the CW and another writer. Among non-deleted notes, the CW publishes the first proposed note on 60% of posts and the first AI proposed note on 95% of posts, leading to CRH status faster and increasing views of CRH notes.

AI writers provide broad impact across topics in Community Notes, most often complementing the work of human writers. Figure 9 presents the distribution of CRH note sources across 10 topics, with post topics identified and assigned by Grok 4.20. Most AI CRH notes occur on posts where there is no proposed human note, indicating that human contributors saw the AI note, found it helpful, and decided not to propose an alternative. Simultaneously, human writing remains a critical contribution, fundamentally extending the coverage of Community Notes: 41% of posts with CRH notes have no AI written CRH note, and 30% have no AI proposed note entirely, inclusive of deleted CW notes.

## 7 Discussion and Limitations

Community Guidance and Authority AI writers add note production beyond humans, while remaining subject to human authority. The community guides the CW in particular through API feeds, model training, and model inputs, focusing notes according to community taste and demand. AI writers have been efective, supporting 86.0% growth in CRH notes shown broadly on X.<sup>15</sup> Human CRH notes continue, often thriving on topics that are rapidly evolving or require nuanced awareness. All notes remain subject to human raters, whose ratings demonstrate the utility ofAI generated notes and the ongoing value of human note writing.

Contributor Experience Scaling AI note writing requires ratings from people who volunteer their time and attention to shape Community Notes. Recognizing that rater attention is a critical resource, the Notable Post Model and Deletion Model yield direct savings by focusing ratings on notes most likely to become CRH. Beyond immediate gains, rating a note that becomes CRH and displays broadly on X is satisfying to contributors, demonstrating their time and attention were productive and well used, potentially yielding improved retention and engagement among Community Notes volunteers.

![](images/ae0fd3189b8edfd99dbd058af60d12a678ec9c5d4d11fa57a263f272968d5404.jpg)  
Figure 9: AI writers extend coverage, with most AI CRH notes occurring on posts without a human proposed note.

Limitations While this work presents and details AI note writing at scale, comparison between writing techniques is limited. We do not control for diferences in the operational cost or post feed sizes of AI writers and compare the Community Writer to all other AI writers in aggregate. Segmenting AI writers is dificult, as AI writers may change over time, operate multiple accounts, and combine multiple writing behaviors within a single account. We are pursuing future work which will create a direct, principled forum for comparing AI writing techniques to advance open research while producing the best notes for contributors and people on X.

## 8 Conclusion

This work presents both the Community Notes AI API and Community Writer, which combine to ofer an efective, open approach to increase the coverage of Community Notes on X. We demonstrate that AI note writing complements notes from human contributors, ofering improvements to speed, rating eficiency, and the yield of Currently Rated Helpful notes shown broadly on X. While AI ofers strengths, human contributors remain the core and ultimate authority of Community Notes, continuing to govern the display of both AI and human proposed notes.

## References

[1] Matthew R. Allen, Nimit Desai, Aiden Namazi, Eric Leas, Mark Dredze, Davey M. Smith, and John W. Ayers. 2024. Characteristics of X (Formerly Twitter) Com munity Notes Addressing COVID-19 Vaccine Misinformation. JAMA 331, 19 (May 2024), 1670–1672. doi:10.1001/jama.2024.4800

[2] Nadav Borenstein, Greta Warren, Desmond Elliott, and Isabelle Augenstein. 2025. Can community notes replace professional fact-checkers? In Proceedings ofthe 63rd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers). 535–552.

[3] Yuwei Chuai, Moritz Pilarski, Thomas Renault, David Restrepo-Amariles, Aurore Troussel-Clément, Gabriele Lenzini, and Nicolas Pröllochs. 2024. Community based fact-checking reduces the spread of misleading posts on social media. arXiv:2409.08781 [cs.SI] https://arxiv.org/abs/2409.08781

[4] Keith Coleman. 2021. Introducing Birdwatch, a community-based approach to misinformation. https://blog.x.com/en\_us/topics/product/2021/introducingbirdwatch-a-community-based-approach-to-misinformation. Accessed Septem ber 2026.

[5] ConSense 2025. AI Against Autocracy. https://consenseai.org/. Accessed September 2026.

[6] Soham De, Michiel A. Bakker, Jay Baxter, and Martin Saveski. 2025. Supernotes: Driving consensus in crowd-sourced fact-checking. In Proceedings of the ACM on Web Conference 2025. 3751–3761.

[7] Chiara Patricia Drolsbach, Kirill Solovev, and Nicolas Pröllochs. 2024. Community notes increase trust in fact-checking on social media. PNAS Nexus 3, 7 (2024), pgae217. doi:10.1093/pnasnexus/pgae217

[8] Hugging Face. 2019. distilbert/distilroberta-base. https://huggingface.co/ distilbert/distilroberta-base. Accessed September 2026.

[9] Hugging Face. 2021. sentence-transformers/all-mpnet-base-v2. https:// huggingface.co/sentence-transformers/all-mpnet-base-v2. Accessed September 2026.

[10] Haiwen Li and Michiel A. Bakker. 2026. AI Fact-Checking in the Wild: A Field Evaluation of LLM-Written Community Notes on X. arXiv:2604.02592

[11] Haiwen Li, Soham De, Manon Revel, Andreas Haupt, Brad Miller, Keith Coleman, Jay Baxter, Martin Saveski, and Michiel Bakker. 2025. Scaling Human Judgment in Community Notes with LLMs. Journal ofOnline Trust and Safety 3, 1 (Sept. 2025). doi:10.54501/jots.v3i1.255

[12] Alexios Mantzarlis. 2026. 8 AI bots now write 50% of X’s Community Notes. https: //indicator.media/p/8-ai-bots-now-write-50-of-x-s-community-notes. Accessed September 2026.

[13] Meta. 2025. Community Notes. https://transparency.meta.com/features/ community-notes. Accessed September 2026.

[14] Meta. 2025. Testing Begins for Community Notes on Facebook, Instagram and Threads. https://about.fb.com/news/2025/03/testing-begins-community-notesfacebook-instagram-threads/. Accessed September 2026.

[15] Nathan Young. 2025. World’s First AI Community Note. https://nathanpmyoung. substack.com/p/worlds-first-ai-community-note. Accessed September 2026.

[16] Sahajpreet Singh, Kokil Jaidka, and Min-Yen Kan. 2026. GitSearch: Enhancing community notes generation with gap-informed targeted search. arXiv:2602.08945

[17] Isaac Slaughter, Axel Peytavin, Johan Ugander, and Martin Saveski. 2025. Com munity notes reduce engagement with and difusion of false information online. Proceedings of the National Academy of Sciences 122, 38 (2025), e2503413122. doi:10.1073/pnas.2503413122

[18] TikTok. 2025. Footnotes. https://newsroom.tiktok.com/footnotes. Accessed September 2026.

[19] Stefan Wojcik, Sophie Hilgard, Nick Judd, Delia Mocanu, Stephen Ragain, M.B. Fallin Hunzaker, Keith Coleman, and Jay Baxter. 2022. Birdwatch: Crowd wisdom and bridging algorithms can inform understanding and reduce the spread of misinformation. arXiv:2210.15723 [cs.SI]

[20] X Community Notes. 2025. AI Note Writer API: Overview. https:// communitynotes.x.com/guide/en/api/overview. Accessed September 2026.

[21] X Community Notes. 2025. ClaimOpinion Model Training. https://github.com/ twitter/communitynotes/blob/main/evaluator/evaluator\_training.ipynb. Ac cessed September 2026

[22] X Community Notes. 2025. Community Notes API for X API v2. https://docs.x. com/x-api/community-notes/introduction. Accessed September 2026.

[23] X Community Notes. 2025. Evaluate a Community Note. https://docs.x.com/xapi/community-notes/evaluate-a-community-note. Accessed September 2026.

[25] X Community Notes. 2025. Top Contributors. https://communitynotes.x.com/ guide/en/contributing/top-contributors. Accessed September 2026.

[26] X Community Notes. 2026. Note ranking algorithm. https://communitynotes.x. com/guide/en/under-the-hood/ranking-notes. Accessed September 2026.

[27] YouTube. 2024. Testing new ways to ofer viewers more context and information on videos. https://blog.youtube/news-and-events/new-ways-to-ofer-viewersmore-context/. Accessed September 2026.

[28] Andy Zhao and Mor Naaman. 2023. Insights from a comparative study on the variety, velocity, veracity, and viability of crowdsourced and professional fact-checking services. Journal ofOnline Trust and Safety 2, 1 (2023).

[29] Xinyi Zhou, Ashish Sharma, Amy X. Zhang, and Tim Althof. 2026. Customized large language models can outperform Community Notes in correcting misin formation. arXiv:2403.11169

## A API Quota Algorithm

Algorithm 1 Progressive writing quota   
$N H _ { 5 } , N H _ { 1 0 }$ ← Num. CRNH in the last 5, 10 CRH/CRNH notes   
��<sub>20</sub>, ��<sub>100</sub> ← (CRH-CRNH)/Total over last 20, 100 notes   
��<sub>14�</sub> ← (CRH-CRNH)/Total over last 14 days among notes   
with at least 10 ratings, CRH or CRNH status   
�� ← Average daily notes written in last 30 days   
� ← Total notes ever submitted   
� � ← Daily writing limit   
1: if $N H _ { 1 0 } \geq 8$ then ⊲ Quality regression killswitch   
2: �� ← 2   
3: else if $N H _ { 5 } \ge 3$ then   
4: �� ← 5   
5: else if � < 20 then ⊲ New writer   
6: �� ← 10   
7: else ⊲ Quota ramp-up   
8: �� ← max(��<sub>14�</sub>, ��<sub>100</sub>)   
9: if �� < 0.05 then   
10: �� ← 300 · max(�� , ��)   
11: else if �� < 0.10 then   
12: �� ← 15 + 700 (�� − 0.05)   
13: else if �� < 0.15 then   
14: �� ← 50 + 3000 (�� − 0.10)   
15: else if �� < 0.20 then   
16: �� ← 200 + 6000 (�� − 0.15)   
17: else   
18: �� ← 500   
19: end if   
20: �� ← max 5, ⌊min(5 ��, ��)⌋ ⊲ Limit rate of change   
21: end if

## B Ratings per Status Outcome

![](images/14c297cdae7666e5624ac55da60b73ed7b4ecf72d9e3576f296abaae740feaeb.jpg)  
Figure 10: 54% of NMR notes receive fewer than 10 ratings.

While the AI API ofers a scalable approach to increasing the supply of notes, contributor ratings remain the sole mechanism that allows notes to show broadly on X. In efect, notes generally require at least 10 ratings to achieve CRH status and at least 5 ratings to achieve CRNH status. During the evaluation period from June 2-29, 2026, 54% of NMR notes receive fewer than 10 ratings, which efectively prevents the notes from reaching CRH status. Low rating counts reflect the free and voluntary nature of contributing to Community Notes: contributors are free to propose notes on low visibility posts and free to decline to rate notes that they don’t find helpful. Figure 10 presents the distribution of rating counts for CRH, CRNH, and NMR notes, excluding any notes that were deleted, along with curves detailing the number of ratings when a note first achieved CRH or CRNH status.