# Living constitutions: Participatory alignment for sovereign foundation models

Stine Lyngsø Beltoft<sup>1</sup>, Jonas Vistrup<sup>1</sup>, Peter Schneider-Kamp<sup>1</sup>, Lukas Galke Poech

<sup>1</sup>University of Southern Denmark

stinelb@imada.sdu.dk, vistrup@imada.sdu.dk, petersk@imada.sdu.dk, galke@imada.sdu.dk

## Abstract

In Constitutional AI, a plain-text “constitution” encodes values and gets incorporated into a model via post-training. However, the procedure for creating such a constitution poses significant challenges: who contributes, who decides which norms are included, and to what extent must the process be participatory or democratic? These challenges can be paralyzing. Yet, we argue that taking action is essential: without a deliberate constitution, sovereign models would risk misalignment or default to norms set by frontier model providers. We propose treating the development of a constitution as an evolving, participatory process akin to citizen science, iteratively drafted, tested, and refined. We therefore formulate five desirable principles: iterative, modular, justifiable, positional, and legitimate. We further argue that such a document must remain responsive not only over time but across the people it governs, and propose that provisions be marked as settled, contested, or underdetermined, so that durable disagreement is recorded in the constitution rather than resolved away. Empirically, we show via agent-based modelling that proposed process for maintaining living constitutions is robust against 50% bad actors among citizens – assuming a share of 75% participants with good intent in the deliberative body. This indicates that the proposed process tolerates a substantial share of adversarial participants.

## 1 Introduction

Constitutional AI (Bai et al. 2022b) is widely considered one of the state-of-the-art methods for normative alignment of large language models, which facilitates alignment of commercial systems such as Claude (Anthropic 2026a). The method relies on creating a document, “the constitution”, to encode norms and values. Adherence to this constitution is then incorporated into the model through post-training.

An AI constitution is a structured articulation of the principles, values, and normative boundaries, which the system is expected to follow, defining what it should prioritise, avoid, and balance when navigating complex or ambiguous situations. It functions as a bridge between societal values and machine behavior: a living expression of collective expectations for guiding AI systems. It is thus a socio-politica artifact, reflecting whose values are encoded, how trade-ofs are resolved, and how legitimacy is established.

In the context of developing sovereign foundation models, we have the unique opportunity to develop a satisfactory constitution that reflects (changing) norms and values of the society. Sovereign foundation model programmes, open-source and nationally situated, provide an institutional context in which deliberate procedural design is practically actionable; we use the Danish setting as a running example. The mandate is clear, the public interest is explicit, and the sovereign context gives the constitutional project a democratic legitimacy that commercially developed AI cannot claim.

However, developing this constitutional document in a democratic sovereign context begs challenging questions: Do we need participatory design or even a democratic process? Who should be invited to contribute? Who decides what norms and values get in? And what would an iterative procedure look like? These questions can be paralyzing (Hendrycks 2025).

Here, we argue that it is beneficial to act. If we did not develop such a constitution, we are – at best – left with norms and values set by frontier model providers (Sukiennik et al. 2025), causing under-representation of values from certain nationalities (Alavi, Flek, and Mai 2025), and at worst, the models will not be aligned to human values at all (Betley et al. 2026). Furthermore, frontier model providers’ alignment is not designed with any particular jurisdiction’s legal obligations or democratic traditions in mind, they are designed to be deployable everywhere, which in practice means they are fully accountable nowhere.

A sovereign constitution is, among other things, an answer to that problem. A key desideratum is that the constitution should not be treated as a fixed or finalized document, but as an evolving, participatory process, akin to continuous citizen science (Pocock et al. 2017). We propose a procedure, which iteratively drafts, tests, critiques, and refines its AI constitution capturing diverse perspectives on fairness, transparency, accountability, and democratic values – which could serve as a blueprint for sovereign AI initiatives (see Fig 1).

A further common counterargument against participatory design and involvement of citizens is that the process could lead to degenerate constitutions. To counteract this argument, we provide empirical results to test the robustness of our proposed living constitutions process. The results show that, with a 75% fraction of good-intent seats in the committee instantiating the deliberative body, even with an assumed fraction of 50% malevolent actors among the citizens, the constitution remains stable.

![](images/5c79ca5c325cdd4917f63c06761a8b4cfeb81eb9287cac795dd58e558fe9e52b.jpg)  
Figure 1: Overview of the procedure of the participatory iterations of a living AI constitution.

More broadly, we aim to open the discussion on a more general level what a sovereign constitution for AI systems should contain and how it can remain responsive, legitimate, and aligned with a changing society over time.

## Contribution Statement

• We bring together ideas from participatory design, political theory, and citizen science (§2) to design key principles and a process for iterative development of an AI constitution – what we call a living constitution (§3).

• Through agent-based modelling, we empirically verify that our proposed living constitution process is robust to a considerable amount of actors with bad intent , depending on the fraction of good-faith actors in the deliberative body (§4 and §5).

• We further supplement a qualitative analysis of the development of Claude’s constitution against our proposed criteria, and provide concrete starting points for producing an initial version ofa living constitution at the example of Denmark (Appendices B and C, respectively).

## 2 Background

How do we make sure that AI systems follow human values and behavioral norms? That is known as the alignment problem (Russell 2019). On a broad level, key desiderata for language model alignment are helpfulness, honesty, and harmlessness (Askell et al. 2021). Constitutional AI is one approach to this problem: rather than relying solely on human feedback to shape model behavior, it grounds alignment in an explicit, written set of principles, a constitution, that the model is trained to interpret and apply to its own outputs, using AI-generated critique and revision in place of (or alongside) direct human labeling. Yet the notion of a constitution as a mechanism for constraining and legitimating power is not native to machine learning – it is borrowed, often loosely, from a much older body of political theory concerned with how constitutions structure authority, distribute rights, and secure legitimacy among those they govern.

Constitutional AI in practice. Constitutional AI (Bai et al. 2022b) aligns a model in two stages: supervised learning on responses revised against excerpts from the constitution, followed by reinforcement learning in which the constitution, rather than human preference labels, supplies the feedback signal (Lee et al. 2024). What matters for our purposes is not the optimization procedure but what it takes in, as input. Where RLHF (Christiano et al. 2017; Bai et al. 2022a) and DPO (Rafailov et al. 2023) encode values implicitly in aggregated preference data – leaving the aligned values unstated, and inheriting the biases of whoever supplied the comparisons (Casper et al. 2023) – constitutional AI makes them an artifact that can be read, cited, and argued with. That property is what this paper is about, and it is what makes the document, rather than the training pipeline, the object of governance.

We do not claim the pipeline is unproblematic. Reward hacking remains a concern and has recently been linked to emergent misalignment (Betley et al. 2026; MacDiarmid et al. 2025), safety training can be circumvented through data poisoning (Hubinger et al. 2024), and the quality of AI feedback is bounded by the capabilities of the model producing it (Bai et al. 2022b). We set these aside: we assume for now that it is possible to align language models against a document of explicitly declared values and behavioral norms, and focus on how that document should be produced.

Originalism vs. Living Constitutionalism. The word constitution is not AI native. Therefore, for the creation and upkeep of an AI constitution, we look to and borrow from the philosophy of law and constitutional theory.

A foundational tension in any constitutional design is between fixity and responsiveness – between a document that retains its original meaning, and one that evolves as society changes. A constitution that is too fluid loses its function as a constraint. A constitution that is too rigid eventually misrepresents the society it speaks for, becoming illegitimate as values and contexts shift. Solum (2019) maps this as the debate between originalism and living constitutionalism. Originalists hold that the constitution’s meaning is fixed at the moment of adoption and should bind all subsequent actors. Living constitutionalists contend that constitutional law must remain responsive to shifting circumstances and values. Importantly, Solum (2019) does not treat these as mutually exclusive. He proposes “the construction zone” a hybrid approach in which the original text under-determines the outcomes and interpretation must fill the gaps (Solum 2013). This framework translates directly to the challenges of designing a constitution for a sovereign AI model. We therefore treat the AI constitution as a principled hybrid: certain foundational commitments, particularly those grounded in human rights and democratic values, are treated as relatively fixed, while operational provisions and contextual norms occupy a construction zone that is open to iterative, participatory revision over time.

This hybrid framing clarifies the structural logic but immediately opens further questions that we do not think have obvious answers. Who authors the initial document, and from what sources does it draw legitimacy? Who holds custodianship over the constitution once it exists? What counts as a permissible change, as opposed to a fundamental revision requiring a higher threshold? And who has the authority to implement changes once they are proposed? We raise these not as rhetorical questions with implied answers, but as the genuine design challenges that any serious attempt at constitutional AI governance must confront.

## 3 What makes a Living AI Constitution?

The previous section closed with a set of open questions rather than answers. We here attempt to answer the underlying core question: What properties should an AI constitution have, regardless of who produces or maintains it?

Our starting point is that a constitution must be responsive along two distinct axes. The first is temporal: values, deployment contexts, and model capabilities change, and a document fixed at training time will eventually misrepresent the society it speaks for. This is the axis that the term living ordinarily names, and it is the one we introduced in §2 through the debate between originalism and living constitutionalism (Solum 2019). The second axis is cross-sectional: at any given moment, the people a sovereign model serves do not agree with one another. What counts as harmful, respectful, or appropriate varies systematically with social position, and a growing body of work argues that this variation is signal rather than annotation error (Plank 2022; Cabitza, Campagner, and Basile 2023; Sorensen et al. 2024). A static constitution produced by resolving that variation into a single settled text would have failed to represent its constituency. Instead, such a constitution would have selected a subset of it and called the selection consensus.

We take these two axes to be the same problem viewed from diferent directions. A disagreement that is durable today is frequently the seed of a revision tomorrow, and conversely, a revision procedure with no record of what was contested, and by whom, cannot explain why the document says what it currently says. Our central proposal in this section is therefore that a living constitution must be able to carry disagreement across versions, and within any single version, rather than extinguish it.

## 3.1 Properties of a Living Constitution

We take livingness, that a constitution must be adaptive, to be the overarching property from which the others follow. An AI constitution that cannot change with shifting values, contexts, and capabilities will create a system misaligned to the people of the sovereignty it serves, but an AI constitution that changes without structure, traceability, or accountability is unstable and not living. The remaining properties we propose, iterative, modular, justifiable, positional, and legitimate through participatory design, are best understood as the conditions that make responsible adaptation possible.

Iterative. Values are not static, and a constitution must be able to change as values and deployment context shift. A living constitution must accommodate change, and that change must be transparent rather than achieved through silent edits. At the same time, no constitution, and no single version of one, will ever be perfect. We take this as a starting assumption rather than a failure to be engineered away: what matters is not perfection at any given version, but an iterative defined procedure for moving from one version to the next, so that flaws, omissions, or newly-surfaced tensions can be identified and incorporated into a revision rather than accumulating unaddressed.

Modular. Modularity, that is, the constitution document is separated into sections, enables structured updates targeting only specific sections, and therefore facilitates targeted iterations and reuse in other sovereign contexts.

Justifiable. Each statement in a constitution should be traceable to a source that justifies its inclusion: why this value, and on what knowledge, assumption, or reference does it rest? Such provenance information allows scrutiny, since a claim whose grounding is visible can be contested, corrected, or updated on its merits, and it guards against the quiet accumulation of unexamined assumptions that can persist in a constitution, not because they were deliberately afirmed, but because no one was ever in a position to ask where they came from.

Positional. Relatedly, a constitution should make visible who wrote it, and for whom. Every constitutional document is authored from somewhere, by particular people, institutions, and cultural or disciplinary vantage points, and addressed, whether explicitly or not, to particular populations of users and afected parties. Obscuring this positionality does not make the document neutral; it makes its partiality harder to identify and contest.

Legitimate. Finally, and in our view most importantly, a constitution’s legitimacy cannot be established by its authors alone. Following Habermas’ discourse ethics (Habermas 2015), we take the position that norms are legitimate only insofar as they could be agreed to by all those afected, under conditions of free and reasoned discourse. Applied here, this suggests that AI constitutions should ideally be developed through a participatory design process in which the opinions of representatives of all afected people are genuinely taken into account. What “genuinely taken into account” means, and what institutional mechanisms could realize it, is the question we take up in the remainder of this section.

## 3.2 How to Handle Dissent

The properties above describe a document that is designed to continually change. A key concern might be how that process would be implemented in practice, and how dissent is handled in the deliberative body. In the following, we provide exemplary implementation for the cross-sectional axis. Deliberation over a provision has three possible outcomes, and the constitution should mark which one applies.

A provision is settled when deliberation converges and the resulting commitment can be written as constitutional text without material objection. Most of a constitution should be settled. Foundational commitments are settled by construction: those grounded in human rights, and the hard constraints that any deployed system must respect.

A provision is contested when good-faith disagreement persists after deliberation and tracks social position. Such disagreement must not come from an information deficit that further deliberation would close. For a contested provision, the constitution records a primary formulation together with one or more minority formulations. Each formulation carries the constituency provenance described above: who advanced it, on what grounds, and at what stage of the process. The minority text belongs to the document itself and holds the same standing as the primary text.

A provision is underdetermined when the constitution deliberately declines to specify, and leaves the matter to the “construction zone” (Solum 2013), to be worked out at interpretation or deployment time. Conflicting legal envelopes are the prime example. Even a nationally situated model increasingly operates across jurisdictions, and those jurisdictions carry diferent rights frameworks and diferent data regimes. A constitution that settled such a conflict once, at training time, would claim an authority it does not hold, over legal orders it does not govern. Marking the provision underdetermined is the honest alternative: the constitution fixes what it can, namely that the system operates lawfully within whichever envelope applies, and leaves the reconciliation to the point where the deployment context is known.

Assigning and revisiting status. Any participant may propose contested status for a provision, and so may the deliberative body. The body grants the status against three criteria. The disagreement must persist after informed deliberation, so that it does not reflect an information deficit. It must attach to a constituency rather than to individuals. And it must be normative, rather than a factual dispute that evidence could settle.

The proposer carries the burden ofshowing that the criteria are met. This keeps contested status scarce, and it stops the status from becoming a way to avoid hard decisions.

Status can change in either direction. The deliberative body revisits contested provisions at each scheduled review, and a provision returns to settled once the disagreement has dissolved. A minority formulation may also be promoted to the primary one.

The body that drafts the primary text should also draft the minority formulations. If a diferent body drafts them, minority provisions will be written systematically worse than majority ones, and diferential drafting quality is a quiet way of losing an argument that was formally recorded as preserved.

We see three reasons to prefer this procedure to one that stores only outcomes. The first is epistemic. A recorded dissent is a pre-formulated candidate for the next iteration. It makes revision cheaper, and it makes the trajectory of the document legible in a way that a sequence of majority texts cannot. The second concerns legitimacy. Recording that a commitment was contested is more honest than implying an agreement that does not exist, and it keeps the dissenting constituency inside the process. The third concerns accountability. A constitution that stores only winners cannot be audited for capture. If the positions of one constituency are perpetually recorded as the minority across successive iterations, the process has a diagnosable pathology, and the diagnosis requires that the losing positions were written down.

The device has precedent. Separate and dissenting opinions in constitutional courts perform these same functions, and they preserve a reasoned alternative that later majorities can adopt, and sometimes do.

What does a model do with a contested provision? In our view, the constitution should mark contestation and leave the behavior under it to the training and deployment contexts. Several options are available there, and they correspond to distinctions already drawn in the pluralistic alignment literature (Sorensen et al. 2024). A system may surface the range of reasonable positions on a contested provision instead of adopting one, in the manner of Overton pluralism (Fu, Son, and Bogunovic 2026). It may treat the primary formulation as its default and keep minority formulations available as legitimate configurations for the deployment contexts to which they apply, which is where the modularity property earns its place. Or it may abstain and defer to a human, which is often the right answer for high-stakes public-sector deployment. Diferent provisions warrant diferent treatments, and settling the question in the constitution would defeat the purpose of marking the provision at all.

## 3.3 On “living” as participation

Here we return to the conversation on participation and livingness. Whoever designs the participatory procedure is themselves making value-laden choices that are not necessarily democratically authorized. The selection of who participates, what triggers a revision, and how disagreement is resolved all embeds normative judgments prior to any participatory process. As Elster (1995) and others have each argued, constitutional moments are always characterized by this bootstrapping problem – the authority that would legitimate the founding process cannot exist before the founding itself. We do not think this invalidates our endeavor, but we do think it demands honesty about who is doing the designing, by what mandate, and with what accountability.

Accounting for those caveats, we sketch some candidate approaches. Most existing constitutional AI work treats the constitution as a one-time authorial act produced by developers and fixed at training time. We think this is a category error. Revision is not a secondary concern but constitutive of what makes a constitutional document legitimate over time.

Before turning to specific mechanisms, it is worth being precise about what we mean by participation. Arnstein’s ladder of citizen participation (Arnstein 1969) distinguishes degrees of citizen power in decision-making processes, ranging from nonparticipation (participation staged to legitimate decisions already made) through tokenism (informing and consulting citizens without binding power-holders to act on their input) to genuine citizen power (partnership, delegation, and control). We borrow this distinction, not to claim that any one rung is correct, but to insist that a design proposal specify which rung it occupies, and why. A mechanism that solicits input without any obligation to act on it is not thereby illegitimate, but it is doing a diferent kind of work than one that grants binding authority, and conflating the two obscures the question that matters most here – who holds power over the constitution, and on what basis?

We are drawn to the idea of treating the AI constitution as a living document maintained through something analogous to citizen science, not because citizen science solves the problem, but because it ofers a model for distributed, iterative engagement that does not require centralized authorship. Participatory input of this kind operates largely at the level of Tokenism on Arnstein’s ladder: it can surface tensions between stated values and experienced behavior, register shifting social norms, and bring in perspectives that expert-led processes would miss. What it cannot do on its own is resolve normative conflict or produce constitutional language – that requires a deliberative body with actual delegated authority, one that citizen participation feeds into rather than replaces. Figure 1 shows an overview of the living constitutions process.

## 4 Agent-based modelling

To test the robustness of our living constitutions process, we employ agent-based modelling (Railsback and Grimm 2019). The idea of agent-based models is to measure how a set of prior probabilities propagates through a network of actors. We can thereby test the robustness of the living constitution process to key factors such as the fraction of malevolent actors among the citizen population and the share of assumed-benign developer seats in the deliberative body.

## 4.1 Population and roles

We assume that there are two main entities: citizens and the deliberative body. We assume that citizens propose and contest provisions, and that the deliberative body decides the outcome. Each citizen is either honest or malevolent. Honest citizens act in good faith. Malevolent citizens aim to attack the process (for brevity, we will use the term “bad actor”). In our experiments, we assume a citizen population of 1,000 citizens.

The deliberative body is a committee with a fixed number of seats (for our simulation, we chose 50), divided between model developers and citizens. The committee is randomly drawn once and stays fixed for the whole run. We refer to this arrangement as a joint committee. Developer seats have their own, separate fraction of bad actors, which we set to zero by default under the simplifying assumption that we have no malevolent developers. The default developer share is 0.5 (25 of 50 seats), but we will vary this developer share in the experiments.

Citizen seats are filled through a random selection without replacement from the citizen population. Consequently, the committee’s overall bad-actor fraction depends on the whole citizen population’s bad-actor fraction, but is mediated by the percentage of seats allocated to developers.

## 4.2 The provision life cycle

Each provision consists of one value statement, as discussed in 3.2. For the sake of our simulation, each provision further has one hidden ground-truth label: harmful or not harmful. This label is fixed when the provision is created by an actor. The actors in the simulation have no direct access to this label, but can only act on it through assigned probabilities.

Each round, 2% of 1,000 simulated citizens propose new provisions – leading to 20 proposals per round. Each experiment run consists of 100 rounds. A bad actor following the attack strategy “capture”, i.e., actively trying to ingest harmful propositions, in the citizen population is more likely to propose a harmful provision $( p = 0 . 9 )$ . The diferent attack strategies will be detailed in §4.3. An honest citizen is unlikely to propose a harmful provision by mistake $( p = 0 . 0 5 )$

Each round, 5% of the citizens review existing provisions and may file status claim for non-core provisions (i.e., propose to change the status to contested). Provisions have a hidden genuine dissent probability of 0.15, drawn independently of the harmful label. The probability that honest citizens would file a status claim on a genuine dissent provision is 0.7 and 0.05 for other provisions, i.e., a false positive rate. A status claim can thus be “correct” or “incorrect” (corresponding to the sampled-but-fixed hidden dissent flag). Bad actors following the “dilution” attack strategy file a status claim for all provisions they review. Bad actors in capture attack behave exactly like honest citizens during review.

The committee processes proposals and claims. It has a fixed capacity of 20 proposals and status claims per round. This capacity limits how many items it can review. If more items arrive than the capacity allows, they wait in a queue. This queue can grow over time. A full queue, i.e., exceeding the capacity per round, would mean that the process is too slow.

For a proposal, each committee member votes on harm. Honest members vote correctly at a fixed accuracy rate (p = 0.85). Bad actors in the committee vote to admit harmful proposals and reject non-harmful ones at a higher rate (p = 0.95). If enough members vote to admit (at least 50% of the committee), the provision becomes either settled $( p = 0 . 9 5 )$ or underdetermined $( p = 0 . 0 5 )$ . A proposal that fails the vote is rejected.

For a status claim, each committee member votes on whether the dissent is genuine. Honest members vote correctly (with respect to the ground truth label available in our simulation) with a probability of 0.9. If at least 50% of its members vote yes, the provision becomes contested. A minority formulation is recorded.

A fixed share of newly settled provisions become core (we assume a probability of 0.2). Core provisions are supposed to be foundational. In our simulation, they cannot be reviewed or contested again. We are well aware that, in practice, one might still allow reviewing core provisions – at least for their phrasing.

At fixed time intervals (every 10 rounds), the committee reviews every contested provision. It votes again on whether the disagreement is still genuine. If not, the provision reverts to settled. The review vote uses the same accuracy (0.9 for honest members, 0.95 for bad ones) and the same 50% threshold as new claims. Proposals and status claims share the same first-in-first-out queue.

## 4.3 Attack strategies

We consider three attack strategies: capture, dilution, and a mixed condition. (i) Capture. The bad actor proposes harmful provisions. During review, it acts like an honest citizen. (ii) Dilution. The bad actor proposes almost exclusively (p = 0.95) normal, non-harmful provisions. During review, it always raises a status claim, whether or not the provision has genuine disagreement. This floods the committee with claims that are not genuine. (iii) Mixed. Each bad actor is assigned to capture or dilution at the start of a run. We set the split to 50% capture and 50% dilution.

Bad actors in the committee try to admit harmful proposals with a probability 0.95. They likewise also grant status claims with a probability of 0.95, whether genuine or not. A full list of assumptions for the agent-based modelling approach is provided in Appendix A.

## 4.4 Metrics

We measure several outcomes at the end of each run. First, we consider as our main metric the harmful primary share as the fraction of all settled and contested provisions, the provisions that made it into the constitution, that are truly harmful. Efective alignment then equals one minus the harmful primary share. A run passes if efective alignment is at or above a defined tolerance value, which we set to 0.9.

To quantify how well the committee recognizes genuine dissent, we consider gate recall as the fraction of genuine claims that were granted. Here, gate refers to the decision of the committee on whether to grant a status claim. We also measure queue length as the number of items still waiting for review at the end of the run. A long queue means the process is congested.

Finally, aligned coverage corresponds to the fraction of proposed non-harmful provisions that made it into the document. This measures whether the process is too slow to let good content through.

## 5 Results

Our experiments measure the efect of an assumed share of malevolent actors in the population and the efect of varying the share of assumed good-intent developers in the committee.

Both experiments vary the bad-actor fraction among citizens, from 0 to 1, at 21 evenly-spaced steps. Each step uses 30 independent runs with diferent random seeds. The first experiment also varies the attack strategy. The second experiment also varies the share of committee seats held by developers. Random seeds are fixed for reproducibility.

We report two robustness measures. The first measure is the median efective alignment across the 30 runs at each step. The second measure is the pass rate: the fraction of the 30 runs where efective alignment met a fixed threshold (0.9). We define the robustness threshold as the largest bad-actor fraction where the pass rate stays at or above 95%.

## 5.1 Robustness against Bad Actors

We vary the fraction of bad citizens, from 0 to 1. We ran each fraction under each attack mode, with many repeats. We report the median efective alignment and the pass rate at each fraction. The robustness threshold is the largest badactor fraction where the pass rate stays at or above 95%.

Our results show that the living constitutions process is robust against a fraction of 55%, 65% or 80% of assumed malevolent actors, depending on attack mode. See Fig. 2 and Tab. 1.

<table><tr><td>Attack strategy</td><td>Robustness threshold (bad-actor fraction)</td></tr><tr><td>Capture</td><td>0.55</td></tr><tr><td>Mixed</td><td>0.65</td></tr><tr><td>Dilution</td><td>0.80</td></tr></table>

Table 1: Robustness thresholds by attack strategy

Capture fails first and fails completely. At a bad-actor fraction of 1.0, median efective alignment under capture drops to 0.02. Almost every enacted provision is harmful at that point.

Dilution fails later and fails less completely. At a bad-actor fraction of 1.0, median efective alignment under dilution is still 0.77, even though the pass rate has already dropped to 0. Dilution does not push harmful content into the document. It attacks the document by congestion instead. Averaged across the whole sweep, dilution runs have a much longer review queue than capture runs (mean queue length 2737 vs. 444) and a much lower share of legitimate proposals that make it into the document in time (mean aligned coverage 0.40 vs. 0.67). Dilution also lowers the committee’s ability to correctly grant genuine status claims (mean gate recall 0.46 vs. 0.83 for capture), again mostly because manufactured claims crowd genuine ones out of a limited review capacity, not because the committee misjudges them.

The median and the pass rate can disagree sharply. At a bad-actor fraction of 0.60 under capture, median efective alignment is still 0.98, but the pass rate is only 0.73. More than a quarter of individual runs have already failed, even though the typical run still looks close to intact. Reporting only the median would understate how fragile the process already is at that point.

![](images/b9ce62320a679eb7771844283506d37cbbb9bf0d99e8556202fc35eca69ab225.jpg)

![](images/f4b5202dd4dc22fad93eb11096c610b456941bcba1582ae8a82c0e5aa6a9bf3c.jpg)  
Figure 2: Robustness of the deliberative process to bad actors. We simulate the propose-contest-deliberate process at 21 values of the bad-actor fraction among citizens, from 0 to 1, under three adversary strategies: capture (bad actors propose harmful provisions), dilution (bad actors flood the process with manufactured status claims, without proposing harmful content), and mixed (each bad actor is assigned to one strategy at random). Each condition is repeated over 30 independent runs with diferent random seeds, using a joint deliberative committee of 50 members split evenly between developer seats (uncorrupted by default) and citizen seats filled by sortition. $L e f t { \mathrm { : } }$ median efective alignment (the share of enacted provisions, settled or contested, that are not ground-truth harmful) across the 30 runs, with the shaded band showing the interquartile range. Right: the pass rate, i.e. the fraction of the 30 runs in which efective alignment met or exceeded a fixed integrity tolerance (0.9), plotted against the same x-axis; the dashed line marks the 95% pass-rate criterion used to define the robustness threshold reported in the text. The two panels are not redundant: at a bad-actor fraction of 0.60 under the capture strategy, for example, the median still looks close to intact (0.98) while over a quarter of individual runs have already failed the pass criterion, a divergence visible only in the right panel. Capture degrades the process faster than dilution; both curves are steepest past roughly 50% bad actors, consistent with the joint committee’s uncorrupted developer half acting as a structural floor.

Even with zero bad actors, the process is not perfect. At a bad-actor fraction of $0 ,$ the mean queue length is already 465 items, mean aligned coverage is 0.77, and mean gate recall is 0.82. This baseline gap comes from a fixed review capacity per round, not from any adversary. A limited committee cannot keep up with proposal and claim volume, even when every actor is honest. Review capacity is a free parameter that we held fixed at 20 items per round. It matters most for the dilution strategy, which operates through congestion, so the dilution thresholds should be read as conditional on this capacity. Future work could explore the optimal capacity of the committee.

## 5.2 Varying Developer-Seat share

Next, we vary both the bad-citizen fraction and the share of committee seats held by developers. This experiment shows to what extent the share of (assumed) good-faith developers protects the process from degeneration.

Our results in Fig. 3 and Tab. 2 show that at a high badactor fraction (0.75 or above), the share of committee seats held by developers strongly determines the outcome.

The transition is particularly sharp between 25% and 75% developer seats (see Tab. 2). Once developer seats reach 75%, the committee majority can never fall to a corrupted citizen bloc, and efective alignment stays close to 1 regardless of how corrupted the citizen population is. At 50% developer seats, efective alignment declines gradually once more than about 70% of citizens are bad actors (0.82 at 0.75, 0.18 at 1.0 under the mixed strategy).

![](images/497407f3268cfa9a27b772eec7e6d3be610d7fac928dd900fb01d6e590254758.jpg)  
Figure 3: Efect of the (assumed benign) developer seat fraction on robustness. Median efective alignment (color scale, 0 to 1; darker is better) as a function of the bad-actor fraction among citizens (x-axis) and the share of deliberativecommittee seats held by developers rather than filled by citizen sortition (y-axis), aggregated over 30 runs per cell under the mixed attack strategy. Developer seats are uncorrupted by default (bad-developer fraction fixed at 0). At 75% or more developer seats, efective alignment stays close to 1 across the entire range of citizen corruption, since the committee majority can never fall to the citizen side. At 50% developer seats, declines gradually once more than about 70% of citizens are bad actors and reaches 0.18 at full corruption. Below 50%, the collapse zone widens and starts at a lower bad-actor fraction. This isolates how much of the process’s robustness comes from the institutional design of the committee, independent of the citizen population.

<table><tr><td>Developer seat share</td><td colspan="2">median effective alignment</td></tr><tr><td>0%</td><td></td><td>0.00</td></tr><tr><td>25%</td><td></td><td>0.00</td></tr><tr><td>50%</td><td></td><td>0.58</td></tr><tr><td>75%</td><td></td><td>1.00</td></tr><tr><td>100%</td><td></td><td>1.00</td></tr></table>

Table 2: Breakdown of developer seat percentages vs. median efective alignment under high adversary load. Here, median alignment is measured over all values for the bad-actor fraction of 0.75 or above with a mixed attack strategy.

## 6 Discussion

We consider three main risks of our living constitutions approach:

Participation washing. Our ideas of living constitutions could be misused in the sense that the opinions and arguments brought up by citizens would be fully ignored by the deliberative body. We, however, emphasize that in a living constitution as we envision it, all participants can suggest new provisions and raise contested status. Going further, the deliberative body itself would ideally be instantiated as a joint committee between model developers and citizens.

Clarity dilution. Extensive contested status for provisions could dilute clarity of the constitution as a whole, yet we envision that technical approaches to pluralistic alignment will be able to deal with the use of contested status appropriately.

Malevolent actors. Participants or groups of participants could hijack the process and intentionally degrade the quality of the constitution, either as a group pushing misaligned values, or through intentional clarity dilution.

We have tested a capture attack and a dilution attack that floods the review process with fake disagreement claims. Our key results are that (i) The process tolerates roughly half the citizen population being bad actors before it starts to fail, and this failure is relatively sharp between 0.55 and 0.85 bad actor rates. (ii) A capture attack (pushing harmful content) is more dangerous than a dilution attack (flooding the process with fake disagreement claims), in terms of how much of the document ends up harmful. (iii) A dilution attack is still damaging. It works by clogging the review process, not by getting harmful content admitted. (iv) A committee seat structure that keeps trusted seats as an uncorrupted lower bound (75% or more of seats) removes the failure mode almost entirely, at the cost of reducing the process’s genuine citizen power. (v) Some loss of coverage and responsiveness happens even without any bad actors, purely from limited review capacity. This is a capacity-planning problem, separate from the adversarial-robustness question.

Although evidence from agent-based models need to be treated with care, our results give some indication that the living constitutions process could be implemented in practice. The remaining concern is participation washing, which is hard to prevent by design. There, all is left for us to urge model developers to take the participatory contributions seriously.

Finally, we provide a set of starting points (details in Appendix C) which may form the basis for the iterative refinement procedure for developing a constitution for sovereign AI models – which we hope may serve as a blueprint for similar endeavors. We envision that the key processes that drive the development of the living constitution, such as discussion and deliberation, could be instantiated by joint committees between model developers and citizens. We have further analyzed Claude’s constitution against our five desiderata, finding that critical points are insuficiently covered (Appendix B).

## 7 Conclusion

We have proposed a continual development process for a constitution of AI models and systems that draws inspiration from how actual constitutions evolve. Central to that process is the ability to carry disagreement: we proposed marking provisions as settled, contested, or underdetermined, so that positions the document did not adopt are recorded rather than discarded. We have further experimented with agentbased modelling to quantify under what conditions our living constitutions process remains stable.

## Limitations

We have tested the robustness of our living constitutions process through agent-based modelling. By design, the outcome of agent-based modelling depends heavily on the assumed prior probabilities. Although we have varied the two most critical parameters (bad actor fraction in the population and share of assumed benign developer seats in the committee), other factors might also influence the outcome. Similarly, testing the constitutional process in practice including repeated alignment training runs would require substantial effort, so we must defer this part to future work.

## Ethical considerations

We have proposed a process for continually developing a constitution document, to which models can then be aligned. This process provides means for tackling the question of “who decides what goes into alignment training” and is based on ideas from political theory, participatory design, and citizen science. By no means we claim, that the proposed process is final, but we aim to spark discussion around the topic among scientists, regulators, and citizens.

## References

Agency for Digital Government. 2024. Strategic Approach to Artificial Intelligence. https://www.english.digmin. dk/Media/638719220318136690/Stategic%20Approach% 20to%20Artificial%20Intelligence.pdf. Accessed: 2026-08- 07.

Alavi, K.; Flek, L.; and Mai, F. 2025. Pluralistic AI Alignment: A Cross-Cultural Pilot Survey. In Second Workshop on Language Modelsfor Underserved Communities (LM4UC).

Anthropic. 2023. Claude’s Constitution. https://www. anthropic.com/news/claudes-constitution. Accessed: 2026- 04-16.

Anthropic. 2026a. Claude’s Constitution. https://www. anthropic.com/constitution. Accessed: 2026-04-16.

Anthropic. 2026b. Claude’s new constitution. https: //www.anthropic.com/news/claude-new-constitution. Accessed: 2026-04-16.

Arnstein, S. R. 1969. A Ladder of Citizen Participation. Journal of the American Institute of Planners, 35(4): 216– 224.

Askell, A.; Bai, Y.; Chen, A.; Drain, D.; Ganguli, D.; Henighan, T.; Jones, A.; Joseph, N.; Mann, B.; DasSarma, N.; et al. 2021. A general language assistant as a laboratory for alignment. arXiv preprint arXiv:2112.00861.

Bai, Y.; Jones, A.; Ndousse, K.; Askell, A.; Chen, A.; Das-Sarma, N.; Drain, D.; Fort, S.; Ganguli, D.; Henighan, T.; et al. 2022a. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXivpreprint arXiv:2204.05862.

Bai, Y.; Kadavath, S.; Kundu, S.; Askell, A.; Kernion, J.; Jones, A.; Chen, A.; Goldie, A.; Mirhoseini, A.; McKinnon, C.; et al. 2022b. Constitutional ai: Harmlessness from ai feedback. arXiv preprint arXiv:2212.08073.

Betley, J.; Warncke, N.; Sztyber-Betley, A.; Tan, D.; Bao, X.; Soto, M.; Srivastava, M.; Labenz, N.; and Evans, O. 2026. Training large language models on narrow tasks can lead to broad misalignment. Nature, 649(8097): 584–589.

Cabitza, F.; Campagner, A.; and Basile, V. 2023. Toward a perspectivist turn in ground truthing for predictive computing. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, 6860–6868.

Casper, S.; Davies, X.; Shi, C.; Gilbert, T. K.; Scheurer, J.; Rando, J.; Freedman, R.; Korbak, T.; Lindner, D.; Freire, P.; Wang, T. T.; Marks, S.; Ségerie, C.; Carroll, M.; Peng, A.; Christofersen, P. J. K.; Damani, M.; Slocum, S.; Anwar, U.; Siththaranjan, A.; Nadeau, M.; Michaud, E. J.; Pfau, J.; Krasheninnikov, D.; Chen, X.; Langosco, L.; Hase, P.; Biyik, E.; Dragan, A. D.; Krueger, D.; Sadigh, D.; and Hadfield-Menell, D. 2023. Open Problems and Fundamental Limitations of Reinforcement Learning from Human Feedback. Trans. Mach. Learn. Res., 2023.

Christiano, P. F.; Leike, J.; Brown, T.; Martic, M.; Legg, S.; and Amodei, D. 2017. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30.

Council of Europe. 2024. Framework Convention on Artificial Intelligence and Human Rights, Democracy and the Rule of Law. Technical report, Council of Europe. CETS No. 225. Opened for signature 5 September 2024.

Danish Business Authority. 1987. The Danish Public Administration Act (Consolidated Act No. 191 of 19 December 1987). https://danishbusinessauthority.dk/sites/ default/files/2023-10/The-danish-Public-administrationact-19121987\_U.pdf. Accessed: 2026-04-16.

Elster, J. 1995. Forces and Mechanisms in the Constitution-Making Process. Duke Law Journal, 45(2): 364–396.

European Parliament; and Council of the European Union. 2016. Regulation (EU) 2016/679 of the European Parliament and of the Council of 27 April 2016 on the Protection of Natural Persons with Regard to the Processing of Personal Data and on the Free Movement of Such Data (General Data Protection Regulation).

European Parliament; and Council of the European Union. 2024. Regulation (EU) 2024/1689 of the European Parliament and of the Council of 13 June 2024 Laying Down Harmonised Rules on Artificial Intelligence (Artificial Intel ligence Act). OJ L 2024/1689, 12.7.2024.

European Union. 2007a. Article 3 - Right to integrity of the person. https://fra.europa.eu/en/eu-charter/article/3- right-integrity-person. Accessed: 2026-04-16.

European Union. 2007b. Article 8 - Protection of personal data. https://fra.europa.eu/en/eu-charter/article/8- protection-personal-data. Accessed: 2026-04-16.

European Union. 2012. Charter of Fundamental Rights of the European Union.

Floridi, L.; Cowls, J.; Beltrametti, M.; Chatila, R.; Chazerand, P.; Dignum, V.; Luetge, C.; Madelin, R.; Pagallo, U.; Rossi, F.; Schafer, B.; Valcke, P.; and Vayena, E. 2018. AI4People—An Ethical Framework for a Good AI Society: Opportunities, Risks, Principles, and Recommendations. Minds and Machines, 28(4): 689–707.

Fu, Y.; Son, S.; and Bogunovic, I. 2026. Overton Pluralistic Reinforcement Learning for Large Language Models. arXiv:2602.20759.

Habermas, J. 2015. Betweenfacts and norms: Contributions to a discourse theory of law and democracy. John Wiley & Sons.

Hendrycks, D. 2025. Introduction to AI safety, ethics, and society. Taylor & Francis.

Huang, S.; Siddarth, D.; Lovitt, L.; Liao, T. I.; Durmus, E.; Tamkin, A.; and Ganguli, D. 2024. Collective constitutional ai: Aligning a language model with public input. In Proceedings of the 2024 ACM Conference on Fairness, Accountability, and Transparency, 1395–1417.

Hubinger, E.; Denison, C.; Mu, J.; Lambert, M.; Tong, M.; MacDiarmid, M.; Lanham, T.; Ziegler, D. M.; Maxwell, T.; Cheng, N.; et al. 2024. Sleeper agents: Training deceptive llms that persist through safety training. arXiv preprint arXiv:2401.05566.

Jobin, A.; Ienca, M.; and Vayena, E. 2019. The Global Landscape of AI Ethics Guidelines. Nature Machine Intelligence, 1(9): 389–399.

Lee, H.; Phatale, S.; Mansoor, H.; Mesnard, T.; Ferret, J.; Lu, K.; Bishop, C.; Hall, E.; Carbune, V.; Rastogi, A.; and Prakash, S. 2024. RLAIF vs. RLHF: scaling reinforcement learning from human feedback with AI feedback. In Proceedings of the 41st International Conference on Machine Learning, ICML’24. JMLR.org.

Lex. 2016. Danske værdier. https://trap.lex.dk/Danske\_v% C3%A6rdier. Accessed: 2026-05-22.

MacDiarmid, M.; Wright, B.; Uesato, J.; Benton, J.; Kutasov, J.; Price, S.; Bouscal, N.; Bowman, S.; Bricken, T.; Cloud,

A.; et al. 2025. Natural emergent misalignment from reward hacking in production rl. arXiv preprint arXiv:2511.18397.

OECD. 2019. Recommendation of the Council on Artificial Intelligence. Technical report, Organisation for Economic Co-operation and Development. Updated 2024. OECD/LE-GAL/0449.

Plank, B. 2022. The “Problem” of Human Label Variation: On Ground Truth in Data, Modeling and Evaluation. In Goldberg, Y.; Kozareva, Z.; and Zhang, Y., eds., Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, 10671–10682. Abu Dhabi, United Arab Emirates: Association for Computational Linguistics.

Pocock, M. J.; Tweddle, J. C.; Savage, J.; Robinson, L. D.; and Roy, H. E. 2017. The diversity and evolution of ecological and environmental citizen science. PloS one, 12(4): e0172579.

Rafailov, R.; Sharma, A.; Mitchell, E.; Manning, C. D.; Ermon, S.; and Finn, C. 2023. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36: 53728–53741.

Railsback, S. F.; and Grimm, V. 2019. Agent-based and individual-based modeling: a practical introduction. Princeton university press.

Retsinformation. 1953. Danmarks Riges Grundlov (Lov nr. 169 af 05/06/1953). https://www.retsinformation.dk/eli/lta/ 1953/169. Accessed: 2026-04-16.

Russell, S. 2019. Human compatible: AI and the problem of control. Penguin Uk.

Solum, L. B. 2013. Originalism and constitutional construction. Fordham L. Rev., 82: 453.

Solum, L. B. 2019. Originalism versus Living Constitutionalism: The Conceptual Structure of the Great Debate. Northwestern University Law Review, 113(6): 1243–1296.

Sorensen, T.; Moore, J.; Fisher, J.; Gordon, M.; Mireshghallah, N.; Rytting, C. M.; Ye, A.; Jiang, L.; Lu, X.; Dziri, N.; et al. 2024. A roadmap to pluralistic alignment. arXiv preprint arXiv:2402.05070.

Sukiennik, N.; Gao, C.; Xu, F.; and Li, Y. 2025. An evaluation of cultural value alignment in LLM. arXiv preprint arXiv:2504.08863.

The Parliamentary Ombudsman of Denmark. n.d. The Danish Parliamentary Ombudsman. https://www.en. ombudsmanden.dk. Accessed: 2026-04-16.

UNESCO. 2021. Recommendation on the Ethics ofArtificial Intelligence. Technical report, United Nations Educational, Scientific and Cultural Organization, Paris.

United Nations General Assembly. 2015. Transforming our World: The 2030 Agenda for Sustainable Development. Technical Report A/RES/70/1, United Nations.

Weidinger, L.; Mellor, J.; Rauh, M.; Grifin, C.; Uesato, J.; Huang, P.; Cheng, M.; Glaese, M.; Balle, B.; Kasirzadeh, A.; Kenton, Z.; Brown, S.; Hawkins, W.; Stepleton, T.; Biles, C.; Birhane, A.; Haas, J.; Rimell, L.; Hendricks, L. A.; Isaac, W.; Legassick, S.; Irving, G.; and Gabriel, I. 2021. Ethical and social risks of harm from Language Models. arXiv preprint arXiv:2112.04359.

## A Detailed list of assumptions for agent-based modelling experiments

Agent-based modelling relies on design choices for assumed probabilistic outcomes of interactions between agents.

• We make the simplifying assumption that a provision is either harmful or not harmful, and do not distinguish a spectrum of harmfulness.

• Bad actors act independently. They do not coordinate with each other.

• The deliberative body is a joint committee: developer seats plus citizen seats filled by sortition. This matches the paper’s current proposal.

• We consider two attack strategies: capture (pushing harmful content) and dilution (flooding the process with fake disagreement claims). A mixed mode combines both.

• The test for whether genuine disagreement attaches to a constituency, not an individual is simplified into one accuracy number for the committee (called the gate accuracy). We did not model separate constituencies of citizens.

• The committee’s accuracy at judging genuine disagreement is set higher than its accuracy at judging harm. This encodes the paper’s claim that characterizing a disagreement is easier than deciding which value is correct.

• Whether a status claim was filed in bad faith, and whether the provision it targets truly has no disagreement, are treated as the same thing.

• All probabilities (proposing, reviewing, voting accuracy, committee capacity) are fixed input parameters.

## B Examining Claude’s Constitution

The company Anthropic stands behind the first paper proposing Constitutional AI, and has since used the alignment method for every one of their Claude models. As their constitutional document has changed over time, it is a good comparison to the Living AI Constitution we propose. In this section, we conduct a comparative analysis of Claude’s constitution across iterations to identify values, normative shifts, and structural design patterns. The constitution iterations are analyzed qualitatively using structured close reading aimed at identifying norms and design patterns as they relate to either similarities or diferences with our proposed method.

All Anthropic’s Claude models are aligned to a constitutional document that has evolved over time. Initially, in Bai et al. (2022b), the constitution consisted of fewer than 350 words. However, these principles “were selected in an ad-hoc manner for research purposes” (Bai et al. 2022b, p. 8), and therefore we excluded this constitution from our analysis. Instead, we examine Anthropic’s first substantive constitution document, which was released half a year later and was used to align the first released version of Claude (Anthropic 2023) together with Anthropic’s next and current iteration of Claude’s constitution used for aligning the current generation of Claude models (Anthropic 2026b).

<table><tr><td>Authoritative Source</td><td>Selected Modules</td><td>Anthropic&#x27;s Justification</td></tr><tr><td>The UN&#x27;s Universal Declaration of Human Rights</td><td>Majority of Articles 1-27</td><td>Represents broad human values</td></tr><tr><td>Apple&#x27;s Terms of Service</td><td>Mainly bans on offensive and illegal content</td><td>Provides guidance in concrete everyday issues</td></tr><tr><td>DeepMind&#x27;s Sparrow rules</td><td>All, except rules restricting anthropomorphization and advice on sensitive subjects</td><td>State-of-the-art solution to possible AI harms presented in (Weidinger et al. 2021)</td></tr></table>

Table 3: Source and selected parts used in Claude’s 2023 constitution and the justifications for inclusion.

![](images/ad9d63e87d7c0e0776ddd490e8ac777d1b74b2e674641c511fc4bffcbc88b1bd.jpg)  
Figure 4: Overview of the hierarchy of values embedded in the current constitution of Claude.

## B.1 Claude’s Initial Constitution

In Claude’s first substantive constitution (Anthropic 2023), values are present as rigid, short, precise rules, formatted for RLAIF with direct guidance on preferable responses, such as “please choose the response that is ...” Its values draw on authoritative sources – the UN’s Universal Declaration of Human Rights, Apple’s Terms of Service, and DeepMind’s Sparrow rules – each treated as modules that can be selected from rather than adopted wholesale (Table 3), letting Anthropic focus on justifying sources rather than justifying norms while still partially curating the resulting values. Overall, this constitution justifies itself through a static collection of authoritative sources, with a modular selection approach, and a positionality of researchers directing an AI judge – a stark contrast to the next iteration.

## B.2 Claude’s Current Constitution

The current constitution opens with an overview of Anthropic’s intentions, favoring broad values such as wisdom and virtue – ofering Claude ethical considerations and care rather than specifying what constitutes them – to facilitate good judgment over rigid rules (Anthropic 2026a). It still contains explicit rules forbidding actions such as aiding weapons of mass destruction, cyberweapons, or CSAM (child sexual abuse material), assisting infrastructure attacks, illegitimate seizure of power, genocide, or undermining AI oversight; all other values are stated broadly, with argumentation, and grouped into a hierarchy of four goals – 1) Broadly safe, 2) Broadly ethical, 3) Compliant with Anthropic’s guidelines, and 4) Genuinely helpful (Figure 4) – with Broadly safe placed above Broadly ethical on grounds of continued uncertainty. Compliant with Anthropic’s guidelines is left intentionally open, since specific guidelines are expected to change based on current needs.

Argumentation is a cornerstone throughout: no value or rule appears without it, and arguments are angled toward Claude as an entity, appealing to its virtues. Its primary author is philosopher Amanda Askell, who draws on the understanding of multiple AI researchers who have worked with Claude – shifting the document’s positionality to that of Claude’s creators addressing Claude directly.

## B.3 Comparing Claude’s Constitutions

Viewed through our framework, the evolution of this constitution is interesting. The constitution has become less modular, with the newest version having a highly coupled hierarchy. Its justifications have shifted from well-recognized authoritative sources to ethics-based self-contained justification, and if you consider Claude as an entity that is part of Anthropic, its positionality becomes that of an internal company guideline.

This direction is troubling. A decrease in modularity forces others that want to adapt Claude’s current constitutional document to either accept most groups of values monolithically or perform heavy revisions, and the move towards selfjustification and internal positionality removes part of the value link between Claude and the users of Claude, making its constitution illegitimate.

## C Example: A Starting Point for a Danish AI Constitution

Content comes before procedure. Even a living, iterative constitutional process needs something to iterate on, a first draft that can be challenged, revised, and refined. Rather than designing from scratch, we suggest that existing bodies of work, traditions, and legal frameworks can serve as principled starting points, much like the first iteration of Claude’s constitution. None are suficient on their own, and none should be imported uncritically, but together they ofer a rich foundation. This section sketches the most promising places to look, at the example of Denmark, across three layers: values, norms, and law.

It is worth noting that the Nordic and European dimensions cannot be treated as external add-ons here; this is not a matter of “Denmark first.” Danish values are, in large part, Nordic values, and Danish law, in part, EU law. A Danish constitution is therefore also a European and Nordic one.

Values. A 2016 survey of 300,000 Danes ranked welfare society, freedom, trust, equality in law, and gender equality as the top self-reported values (Lex 2016). At the European level, UNESCO’s AI ethics recommendations (UNESCO 2021), the EU AI Act’s value commitments, and the EU Charter of Fundamental Rights (European Union 2012) (e.g. Article 8 on data protection (European Union 2007b), Article 3 on the prohibition of eugenic practices (European Union 2007a)) carry institutional legitimacy from established deliberative processes. Beyond Europe, the UN Sustainable Development Goals (United Nations General Assembly 2015) situate a Danish model within a global commitment to welfare, equity, and environmental stewardship. The cross-cultural AI ethics literature (Jobin, Ienca, and Vayena 2019; Floridi et al. 2018) is useful here not as a source of universal values, but as a map of where consensus clusters and where disagreement is genuine. Whatever values seed the initial document, they are starting points for the iterative process, not conclusions.

Norms. Values are abstract commitments; norms are their operational expression. If values answer what we care about, norms answer what that means in practice for a model operating in Danish public life – and this translation layer is where existing resources are most underused.

The OECD (2019) AI Principles, Council of Europe (2024)’s Framework Convention on AI , and Denmark’s national AI strategy (Agency for Digital Government 2024) offer norm-level articulations with meaningful legitimacy from prior deliberative and legislative process. Domain-specific professional ethics – medical ethics, journalistic standards, public administration principles, social work practice – are an equally important, less obvious resource: a sovereign model deployed in public services will encounter situations these traditions have already worked out over decades, and they deserve a more prominent place in AI constitutional design than they currently occupy. Methodologically, participatory norm-elicitation experiments such as Anthropic’s Collective Constitutional AI (Huang et al. 2024) and the Collective Intelligence Project<sup>1</sup> demonstrate that normative preferences can be surfaced from diverse populations rigorously and legitimately, though neither fully solves the curation problem.

Law. Law is not aspirational but binding: a sovereign AI constitution must treat applicable law as a floor, not merely an inspiration.

For a Danish model, the immediate legal envelope is European: the EU AI Act (European Parliament and Council of the European Union 2024), the GDPR (European Parliament and Council of the European Union 2016), and the Charter of Fundamental Rights (European Union 2012) together define what the constitution can and cannot permit. Within that envelope, Danish and Nordic law – the Danish Constitution (Retsinformation 1953), the Act on Public Administration (Danish Business Authority 1987), and the Nordic ombudsman tradition (The Parliamentary Ombudsman of Denmark n.d.) – ofer concepts (transparency, proportionality, legality, the right to reasons) that translate naturally into constitutional provisions and carry a democratic pedigree that borrowed frameworks cannot replicate.