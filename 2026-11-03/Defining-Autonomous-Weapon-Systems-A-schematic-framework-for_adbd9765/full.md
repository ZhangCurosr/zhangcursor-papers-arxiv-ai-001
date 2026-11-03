# Defining Autonomous Weapon Systems: A schematic framework for refined purpose-driven definitions

Björn Lundgren,<sup>∗</sup> bjorn.lundgren@fau.de<sup>1,2</sup> and Nathan G. Wood, nathan.wood@tuhh.de<sup>3,4,5</sup>

<sup>1</sup>Centre for Philosophy and AI Research, Friedrich-Alexander University Erlangen-Nuremberg, Germany <sup>2</sup>Institute for Futures Studies, Sweden

<sup>3</sup>Institute of Air Transportation Systems, Hamburg University of Technology, Germany

<sup>4</sup>Ethics + Emerging Sciences Group, California Polytechnic State University San Luis Obispo, United States of America

<sup>5</sup>Center for Environmental and Technology Ethics – Prague, Czechia

(Initial Draft) November 3, 2026

## Abstract

The literature on autonomous weapon systems (AWS) includes many proposed definitions for these (types of) systems, with core disagreements between scholars, governments, and international regulatory bodies. In this article, we present and critique prevailing (classes of) definitions, arguing that what is needed is not a singular definition, but rather a definitional framework which allows one to classify diferent sorts of AWS within the same conceptual family. Recognizing we cannot foresee all future AWS, we show how variations on a broadly accepted definition can be used to produce a definitional schema which can be easily modified in order to classify various concepts belonging to the class of AWS.

Keywords: Autonomous Weapon Systems; Autonomy; AI; Military Ethics; Arms Regulation; AI Governance; Responsible Innovation

## 1 Introduction

As of writing this, it has been over a decade since regulatory discussions around autonomous weapon systems (AWS) began in Geneva under the auspices of the United Nations Convention on Conventional Weapons (CCW),<sup>1</sup> and ethical discussions have been ongoing far longer.<sup>2</sup> In that time, many definitions of AWS have been put forward,<sup>3</sup> and while some have gained wider acceptance, there is still no broadly agreed view. Moreover, the range of existing definitions all admit weaknesses in need of addressing. This article seeks to remedy this by developing not a singular definition of AWS, but rather a conceptual definitional framework for defining specific (types of) autonomous systems used in various military roles. In doing so, the article pays heed to a core reality of autonomy in systems, namely that it can be leveraged for a variety of purposes, with a variety of technical, operational, legal, and ethical challenges, and with various attendant technical, operational, legal, and ethical approaches for mitigating those challenges.<sup>4</sup> Thus, given the need for specificity, the article builds toward a modular definitional schema where scholars, policy-makers, and defense professionals can “plug in” definitional components as needed, befitting both the system they are considering and their definitional needs. This latter concern is of relevance given that not all definitions serve the same purpose, and in a domain as politically charged and with as significant of consequences as this, being able to craft purpose-built definitions for specific applications holds immense value.

To develop these arguments, the article begins (Section 2) by canvassing general points on definitions, laying out the methodological assumptions guiding this work and tying these to the underlying purpose of the article. With these in place, Section 3 then presents and critiques two strands of definition one can find in the literature from scholars, states, and international bodies, what we call the “standard view” and the “demanding view”. Building from these critiques, Section 4 lays out base definitions of both “autonomous systems” and “autonomous weapon systems”, stressing that these are base definitions. That is, they are not to be treated as complete, as demarcating every relevant feature of some system, or as not needing further refinement. Indeed, Section 5, which forms the central result of this work, serves as an extended presentation of various core concepts which should be further indi cated in the definitions of particular autonomous systems in the military (and in other domains as well). This section concludes with general recommendations for constructing purpose-built definitions which include all relevant terms for a given system and use-case. The article concludes (Section 6) with a broad recapitulation of the need for a pragmatic and purpose-oriented definitional schema, rather than an intellectualized pursuit of “being right” at the cost of progress.

Before moving onto the arguments themselves though, there are two preliminary points to be addressed. First, it is worth clarifying and stressing at the outset that the purpose of this work is to add clarity to the ongoing debates surrounding autonomous (weapon) systems and to reduce the risks of scholars and practitioners “talking past one another”. It is not our aim to weigh in on any particular debates themselves though, nor to advocate for the benefits or risks of certain systems, their permissibility, or any related concern. That work is highly important, and it requires foundational definitions to be well-articulated, and it is this latter concern only which is addressed by the work here. Our aim is thus to develop a non-prejudicial, generally acceptable, and practically useful definitional schema, but not to advocate for anything beyond that.

Second, though we are discussing autonomous weapon systems in particular, the general points surrounding definitions and our schematic for constructing modular definitions which can be tailored to specific systems and use-cases holds general import. Thus, provided appropriate modifications are made to the list of pluggable modular terms, the overall schema and its motivating points can be ported to other technologies and domains of application. This is complementary to recent work such as that from Wood et al. (2026) which argues for the need for specificity, and Wood et al.’s “template for policy writing” on AI<sup>5</sup> fits neatly with the overall points developed here.

## 2 What’s In a Definition?

Not all definitions are created equal. If asked for a definition of a “yellow lab”, one could answer that this is an organism of the animal kingdom, of the chordata phylum, mammal class, carnivora order, canidae family, from the genus canis, species canis lupus, subspecies canis lupus familiaris, and breed Labrador Retriever. But also, since it is yellow, one must add that on top of these various taxonomic details, it is a specific set of characteristics regarding the MC1R, Agouti, and CBD103 genes that make it a “yellow” lab. Alternatively, one could define this as a dog that is traditionally used for retrieving hunted fowl. Both are fine definitions... in the right setting.

At a basic level, definitions can be built in what we might think of as being molded to science or engineering. A “scientific” definition may be seen as aiming to carve nature at the joints, capture the true state of things in some domain, or otherwise direct individuals toward knowledge or some ontological facts.<sup>6</sup> An “engineering” definition, on the other hand, is one which is aimed at being useful, at solving some set of problems humans have and providing a “good enough” approximation for some given aims.<sup>7</sup> There will be crossover between these two sorts of definitions as well, determined by use cases and the purpose of asking for a definition in the first place. For example, in most instances, the first definition of “yellow lab” above could be seen as almost purely scientific in nature, admitting little “usefulness”. However, if a geneticist is asking for the definition, then the taxonomy and particular genes involved in a yellow lab’s being yellow may be highly relevant for a particular aim or inquiry the geneticist has in mind. The reason for asking can thus determine the quality of a definition.

In this work, we take a broadly pragmatic approach that tries to balance these concerns, but errs on the side of viewing definitions through an engineering lens when conflicts might arise. This is partly because the definitions of man-made artifacts are bound to be at some point arbitrary anyway, and because the domain we have in mind, emerging technologies in the military domain, is one where definitions have extreme functional import for debates. There is thus value in looking at a definition not primarily as a marker of “what is”, but rather as a tool for facilitating discussion and debate, efective regulation, common understanding, etc. Moreover, given that technologies continuously develop, no definition of a technical artifact is apt to ever be properly complete, undermining the possibility for developing fully satisfactory scientific definitions.

As noted above, definitions can also serve a variety of purposes, grounding scientific or philosophical discussions, setting agreed terms for regulation/norms, or establishing enough understanding for technical documentation and training processes to be established. These diferent purposes will call for slightly diferent expressions of definitions, or for definitions which are more or less closely tied to what we take to be the ground truth. For example, Isaac Newton defined numerous terms which, in classical mechanics and everyday life, are very useful. However, those same definitions do not account for quantum or relativistic efects, and thus are technically wrong, scientifically speaking. On the other hand, quantum physics is not exactly the most practical discipline, and so it would be pragmatically wrong, from an engineering perspective, to rely on it for everyday engineering or design purposes. The simple fact is that a particular definition may be “good” or “bad” for a variety of reasons, and a central determinant of a definition’s “goodness” relates to what the definition is for in some instance.

Being overly bullish on the priority or value of any one particular definition makes one apt to overlook its shortcomings and the value of potential competitors for given tasks. These points complement recent work around explanations for AI-enabled and algorithmic systems, where explanations likewise are argued to be “inherently contextual”, relating to both the target group and task addressed by an explanation.<sup>8</sup> Explanations and definitions share many core elements, and while one can argue for “the” definition of some term, object, or entity, it is far more fruitful to allow for numerous definitions tailored to target groups and tasks.

If we aim to develop a base definition which can be refined and specified along broadly pragmatic grounds as needed, with the usefulness of that definition being a central criterion, a number of considerations must be taken into account. First, the base definition should be scientifically defensible while also understandable and acceptable to non-scientific audiences and stakeholders. Without this, it may be technically accurate, but unhelpful in application.

It should serve as a starting point for further discussion and debate, and not foreground or anticipate later conclusions we may hope to draw. Building on this, it should be neutral in three distinct respects. First, it should be non-prejudicial, meaning that the definition itself ought not unduly color or prejudge follow-on debates about the thing being defined. For example, in the debates around autonomous weapon systems, a view of “autonomy” which includes in situ or online learning as a necessary definitional component confers by fiat a host of arguably insurmountable challenges for AWS. This is because systems which can learn in real time and potentially alter their program almost certainly cannot be subjected to the necessary test, evaluation, verification, and validation required for weapon systems to be permissibly fielded in military contexts at all.<sup>9</sup> Thus, any view of autonomy that makes it impossible to control, adequately understand, or suficiently predict how a system will behave should be deemed prejudicial to knock-on discussions. In some instances, the “right” definition may by necessity color further debate, but when there are reasonable alternative definitions which do not include a prejudicial framing, these are to be preferred.<sup>10</sup>

Base definitions should also ideally be time-neutral, meaning that they are not designed to ensure that the defined systems (do not yet) exist. In some discussions of AWS, authors are keen to point out that the thing they are defining doesn’t exist yet (though they are often incorrect in their presentation; see Wood (2023a) and citations therein). In other definitions, such as that until recently provided by the United Kingdom’s Ministry of Defence (UK MoD), a highly strict definition has been given which allows some to, under that view, claim they do not possess autonomous weapons and do not plan to explore these.<sup>11</sup> That strict definition appears to be designed to satisfy a political aim (proclaiming their rejection of AWS, as defined by them), but does so at the cost of defining the term in a highly unhelpful manner (in virtue of the definition applying to a highly narrow class of entities, and one out of step with the current broad understanding).<sup>12</sup> Building definitions to satisfy an underlying commitment about whether or not some thing exists represents a type of prejudice harmful to efective discussion.

The third kind of neutrality in need of incorporation concerns use. Base definitions should be use-neutral, meaning they do not tie a broad class of technology to a narrow domain of application. I.e., one and the same base definition for “autonomous weapon systems” should be able to capture AWS used in anti-personnel or anti-materiel roles, with lethal or less-lethal armaments on board, used in purely defensive or potentially defensive/ofensive platforms, etc. Refinements on that base definition can pick out more specific platforms – Section 5 explores a variety of such refining categories – but the basic definition of AWS should not prejudge what more specific configurations of systems and capabilities fit within that. This is especially the case, given that some capabilities or use cases may be more ethically, legally, or operationally fraught than others, and so tying the base definition to one case (e.g., anti-personnel systems) will prejudice all following discussions.

To briefly recap, a base definition should be 1) as accurate as possible 2) while still being broadly useful, 3) acceptable to scientists and non-scientific stakeholder groups alike, 4) non-prejudicial in general terms, 5) time-neutral regarding whether the defined object exists (or must exist for the defining party’s aims), and 6) use neutral regarding whether the system must by necessity execute certain tasks or have certain target sets.

## 3 Existing Views and Their Limitations

With these considerations and desiderata for a “good” base definition in place, we may now critically evaluate two broad views of AWS proposed. The first we may think of as the “standard” view, shared by key state actors, international organizations, and many scholars. The second, which we call the “demanding” view, presents a much narrower view of AWS, and while not explicitly endorsed by as many groups or individuals, finds implicit voice in many arguments critical of AWS. The main aim of the criticisms leveled in this section is to draw out errors and points relevant for the creation of a better base definition upon which refinements may be layered, according to particular more specific systems or definitional use cases.

## 3.1 The Standard View

What we have termed the “standard” view is that prominently presented by the United States Department of Defense (US DoD), the International Committee of the Red Cross (ICRC), Human Rights Watch, and numerous other states, international organizations, non-governmental organizations, and scholars. This is the conception of AWS as weapon systems with autonomy in “critical functions” necessary for selecting and engaging targets,<sup>13</sup> and which can therefore select and engage targets without contemporaneous human input, oversight, or intervention.<sup>14</sup> As noted by Wood (2024), the “broad view is similarly shared by both scholars critical of AWS... and civil society groups”,<sup>15</sup> highlighting that this definition is not merely the preferred view for states and regulatory bodies, but sees broad acceptance across various domains. As an exemplar of the standard view, we may take the definition given by the US DoD:

A weapon system that, once activated, can select and engage targets without further intervention by an operator. This includes, but is not limited to, operator-supervised autonomous weapon systems that are designed to allow operators to override operation of the weapon system, but can select and engage targets without further operator input after activation.<sup>16</sup>

The standard view anchors a central aspect of AWS, that they are autonomous in core tasks relevant to a weapon system’s contribution to military operations. Beyond that, this view does not imply any other specific capabilities, limitations, or design choices though, allowing for AWS which are purely mechanical or which have the most sophisticated AI-enabled functions; which are transparent or opaque in their functioning; which are designed to be lethal or less lethal; or which exclusively target persons or objects. Given this breadth, there are many classes of system which have existed for decades which count as AWS under this definition, from anti-radiation missiles to missile interception systems, and even landmines.<sup>17</sup> This presents a basic challenge on its own, and there are three specific implications of the broad definition which undermine its usefulness for debates.

First, the debates around AWS and emerging technologies in the military more broadly are concerned with artifacts humans make and use. However, there are “weapon systems” which are autonomous and based centrally around biological entities. From the “bat bombs” of World War II<sup>18</sup> to modern combat-assault-dogs,<sup>19</sup> and many other animals besides have been used in war throughout the ages.<sup>20</sup> Taking for granted the system-of-systems understanding of modern military operations, where overall military potential is viewed as a result of the “interactions and synergism” <sup>21</sup> of distinct systems, each contributing to a larger aim, it is also possible to view individual human combatants as “autonomous weapon systems”, given that they are capable of selecting and engaging targets without further input and are systems within the larger system-of-systems driving modern warfare.<sup>22</sup> One could bite the bullet and claim that “autonomous weapon systems” can reasonably include biological entities, humans included, within its scope, but this would render the term highly unfit for the regulatory and doctrinal roles it is meant to fulfill.

A second concern is that the standard view’s focus on systems that can select and engage targets assumes a certain level of functionality in order for objects to count as AWS. Put diferently, it excludes systems that might be broken or improperly functioning. However, if systems are designed to “select and engage targets without further intervention” and normally can do that, temporary malfunctioning should not remove them from the definitional extension of AWS. This may seem a small quibble, but definitions set the standard for not just discussions, but also legal regulations, and a regulation governing AWS which assumes functionality will, on a strict reading, not cover any systems malfunctioning or temporarily limited. This is a problem.

Finally, the standard view is, quite simply, too broad in general terms. There are very many weapon systems which fall under this definition, used for an exceedingly wide array of tasks, and with widely disparate capabilities, use contexts, ethical and legal challenges, and other attendant concerns. Focusing on the mere capacity to “select and engage targets without further intervention by an operator” also implies that things which plausibly fall outside the traditional idea of warfighting would also count as AWS. For example, chatbots fine-tuned to carry out information operations and given agentic AI scafolding to allow them to independently engage with foreign civilians online could also be seen as AWS, simply because they could select targets (online users) and engage them (post messages tailored to them) without further input. In the evolving landscape of modern conflict, greater attention should be paid to the information domain (see Section 5.10 below), but capturing a missile-armed autonomous drone and a chatbot posting to social media under the same heading is clearly not helpful.

To sum, the standard view is very broad. For some applications, such as establishing fundamental norms or discussing elementary concerns over control and oversight, this breadth may be useful. But in the vast majority of use cases, the standard view will be too broad, covering such a wide array of systems that it loses its value and serves more to muddy discussion than to further it. For these reasons, a more precise view is worth pursuing.

## 3.2 The Demanding View

There are a variety of ways the standard view may be given more precision, or its extension limited. Across the literature, a standard approach is to simply add constraints or definitional points which make the definition of AWS more demanding. Above, we have already seen this with the UK MoD definition (now superseded) of AWS as systems “capable of understanding higher-level intent and direction”,<sup>23</sup> and the Canadian discussion of AWS admits similar addition of terms indicative of higher agency in AWS.<sup>24</sup> While there are numerous ways to develop a more demanding view, for the remainder of this section we will focus on the definition given by Taddeo and Blanchard (2022), as it represents one of the most fully developed, motivated, and defended instances of the “demanding” view. They define an autonomous weapon system as follows:

[A]n artificial agent which, at the very minimum, is able to change its own internal states to achieve a given goal, or set of goals, within its dynamic operating environment and without the direct intervention of another agent and may also be endowed with some abilities for changing its own transition rules without the intervention of another agent, and which is deployed with the purpose of exerting kinetic force against a physical entity (whether an object or a human being) and to this end is able to identify, select or attack the target without the intervention of another agent is an AWS. Once deployed, AWS can be operated with or without some forms of human control (in, on or out the loop). A lethal AWS is specific subset of an AWS with the goal of exerting kinetic force against human beings.<sup>25</sup>

Taddeo and Blanchard’s definition is unfortunately lengthy and complex, undermining its potential use in many domains where simplicity is a virtue. Even so, it gives much added specificity to the core conception of “autonomous weapon systems”. Inclusion of autonomous changing of internal states and transition rules are far more demanding standards than those present in the standard view, greatly limiting the extension of AWS. The stress on kinetic force also eliminates the possibility for systems usually wholly outside direct military operations – such as autonomous information warfare systems – to be considered AWS. Finally, addition of varying levels of control, lethality, and target class (personnel vs. materiel) as potential modifiers serve as useful ways of framing nuances in system design and deployment. Yet for all its added specificity, the demanding view presented by Taddeo and Blanchard critically fails for a number of reasons, with the failures indicating general issues apt to obtain for all such demanding views.

Before examining the individual distinct shortcomings of Taddeo and Blanchard’s definition, it is worth clarifying that in this section, we will not be evaluating their definition against their own aims<sup>26</sup> (though their definition does fail on some of their own aims; see discussion of value-neutrality below). Rather, we will focus on whether the extension of their definition sensibly tracks to what is broadly considered the “correct” extension of AWS. As artifacts do not fit as cleanly into taxonomic classes as natural and scientifically discernible entities do, some reference to general usage is an apt limiter of the success or failure of a definition. Moreover, whether the extension of a definition fits general usage is also critical to whether that definition is useful in terms of furthering, guiding, or grounding ongoing debates. If the extension of a definition departs widely from standard usage, then it is not a useful definition, and may indeed be a harmful one, in virtue of adding confusion to an already quite complex and multi-disciplinary enterprise with significant human consequences attending misunderstandings. Given these points, we will primarily critique the demanding view presented by Taddeo and Blanchard in light of how well it fits to the broad consensus in international debates concerning what are (not) AWS.

Problems arise already with the very first terms in Taddeo and Blanchard’s definition, as “artificial agent” renders the definition both too broad and too narrow. To broadness, human clones would arguably be “artificial” agents, given that they are artificially created beings. However, to treat clone as merely AWS would be technically, morally, and legally incorrect. Whether a human is the result of natural reproduction or some futuristic cloning process, such a human is still a human and therefore a full moral and legal agent, and should be treated as a fully-fledged combatant, with all the rights and responsibilities accorded that status. Even under the system-of-systems view presented above, where individual combatants may be viewed as subsystems in the larger military system, it is still awkward at the least, and likely simply wrong to view human combatants as “autonomous weapon systems”, as this greatly misrepresents their agency, ethical and legal responsibility, and capacity for novel action. To narrowness, artificial “agents” sets the bar for AWS too high, as agency can – and for ethical and legal reasons, should – be viewed as a thick concept that naturally rules out any but the most thoroughly advanced science fiction machine systems as “agents”. This is because agency implies the ability to take responsibility and to be reasonably held responsible for genuine choices one makes.<sup>27</sup> This is what distinguishes in a court of law humans of normal capacity from those who are (temporarily) insane or who are not fully cognitively capable (children or the mentally handicapped).<sup>28</sup> That distinction matters in ethics as well, and is a critical factor in the organization of military command structures and responsibility chains. Requiring that AWS be agents thus relegates the definition’s extension to far future science fiction, making it a useless conception for ongoing debates.

The demanding view presented by Taddeo and Blanchard also admits the same assumption of functionality found in the standard view, in light of picking out systems “able to change” internal states and “able to identify, select or attack” targets. As discussed above, this renders the definition too narrow, as systems designed to be autonomous but temporarily malfunctioning should not be excluded due to passing system limitations. The focus must be on design architectures and intended functionalities, not the precise capability set possessed by a system at a time of evaluation.

Focusing on systems “exerting kinetic force” is likewise too narrow, as modern warfare includes a variety of highly impactful non-kinetic forms of attack that may be carried out by autonomous systems. For example, electronic warfare (EW), cyber-attacks, or highly targeted information/cognitive warfare can be quite significant to ongoing military operations, with EW and cyber-attacks being central elements in modern drone-intensive and sensorlinked warfare. Limiting the extension of AWS to only those things that have immediate kinetic efects is thus unduly narrowing in general terms, and also out of touch with central realities of modern warfare and emerging autonomous systems design.<sup>29</sup>

Related to this, picking out “physical entities” as the sole target class of AWS is also unduly narrow. Modern warfare increasingly relies on digital infrastructures to underpin core military capabilities,<sup>30</sup> and while it is true that digital infrastructures have a physical side to them, there are potentially autonomous attacks that may be aimed solely at the digital aspects of such systems.<sup>31</sup> There are also attacks aimed not at systems themselves per se, but rather at capacities in systems. For example, jamming and spoofing attacks<sup>32</sup> aim to undermine a system’s ability to navigate or communicate, but do not necessarily target the platform itself. Indeed, many such attacks are area attacks, and so are not aimed at any one platform, but rather serve as capability-denial area attacks. It is hard to see how an electronic attack intended to undermine a capacity to navigate represents a strike on a “physical entity”.

Limiting the definition to systems “deployed with the purpose of” exerting force also conflates what is a weapon with what can be used as a weapon, making the definition too narrow and potentially too broad as well. It is too narrow because this restriction implies that autonomous weapon systems cease to be AWS any time they are deployed without an intent to exert force. For example, deploying missile-armed autonomous aerial drones for purely reconnaissance purposes would render them not AWS. Oddly, they would also potentially not be AWS only for that operation. This is clearly incorrect, as the purpose of a deployment should not alter the essential definition of the systems being deployed, especially considering that the purpose may change mid-deployment (say “reconnaissance-only” missile-armed drones identify enemy positions, and then are switched by human overseers into “reconnaissance and strike” mode).

Focus on the purpose of a deployment may be too broad as well, as it implies that some objects not normally considered weapons may be identified as AWS whenever they are used for attacks. For example, autonomous cars are not AWS, but if an insurgent were to modify/remove crash-avoidance systems and then use an autonomous car as a battering ram to strike the gates of a military base, it would, under this definition, be an autonomous weapon system for that operation. Perhaps such a use-determinant definition can have value, but it seems far more natural to say in such cases that “this is an autonomous car being used as an autonomous weapon system”, not simply that it “is an autonomous weapon system”. To be sure, there may be good arguments for both sides of this particular point, but the clarification just provided is much more parsimonious than having the extension of a term change according to momentary use. We would also stress that while “they are used as AWS” and “they are AWS” may seem a trivial and pedantic point to make, such pedantry is critical for legal regulations, lest norms governing weapons bleed over into purely civilian domains, or fail to govern the objects they are meant to oversee. In short, the precise extension of a definition can unduly limit or unfairly expand legal regimes, and so attention to that extension is merited.

The final and arguably most poignant concern with Taddeo and Blanchard’s view is that it simply is not neutral, in violation of both a central desiderata of good definitions discussed in Section 2 above and Taddeo and Blanchard’s own avowed aim to “identify essential aspects of AWS and then ofer a definition that provides a value-neutral ground to facilitate eforts to address the relevant ethical and legal problems”.<sup>33</sup> This is because, as discussed above, merely including in situ or online learning in the definition of AWS would render these systems incapable of being used in compliance with core rules of international law concerning necessary test and evaluation of new systems.<sup>34</sup> The demanding definition above goes far beyond in situ learning though, anchoring an ability to autonomously change internal states and potentially also programming as essential features of AWS. Quite bluntly, it would be fundamentally illegal to deploy any system which had such a capability, both because it could not be ensured that its use would not, “in some or all circumstances, be prohibited by this Protocol or by any other rule of international law”, in violation of Article 36 of Protocol I Additional to the Geneva Conventions (hereafter, AP I),<sup>35</sup> and because the potential unpredictability of such a system would render it a “method or means of combat which cannot be directed at a specific military objective” or a “method or means of combat the efects of which cannot be limited”, violating Article 51 of AP I.<sup>36</sup> A definition of AWS which renders their use necessarily incompatible with international humanitarian law is not a neutral definition.<sup>37</sup> Moreover, it is not a definition which facilitates debate or efforts to address ethical and legal concerns, as the only reasonable conclusion to draw from Taddeo and Blanchard’s view is that AWS, as understood by them, must be explicitly and comprehensively banned, as they could not be used in compliance with international humanitarian law.

The demanding view developed by Taddeo and Blanchard sufers from many faults specific to their particular formulation. Broadly though, demanding views will in general need to admit additional terms or definitional requirements in order to add stringency for limiting those definitions’ extensions. Every such addition will remove some (class of) systems which plausibly fit under the umbrella of “autonomous weapon systems” for some reasons or with respect to some use cases or purposes. Rather than trying to develop a singular unary definition that captures all plausible candidates of AWS (which can be important for regulatory, oversight, or doctrinal reasons), we should aim for a base definition which captures the essential features of AWS grounding broad-based fundamental norms, and then have a modular approach to constructing refined definitions that are system-, use-, and purpose-specific.

## 4 The Base Definition

A base definition of “autonomous weapon systems” must be broad enough to ground fundamental ethical, legal, doctrinal, and governance norms which span various systems and use cases. It must also not prejudge systems. And finally, it must be loose enough in its definitional stringency and structurally permissive in allowing other terms to be added into it as needed.

The standard view presented above meets most of these criteria, and while being too broad to be acceptable as the definition of AWS, it could serve well as a base definition, upon which additional elements are grafted to suit particular needs. However, as argued in Section 3.1, there are two specific elements of the standard view in need of addressing, namely that it includes biological entities as AWS, even potentially human combatants, and that it assumes that systems classified as AWS are properly functioning. Each of these must be addressed in a working base definition.

Given the ultimate aim of developing a modular framework, there is also value in beginning not with a base definition of “autonomous weapon systems”, but instead with the more general “autonomous systems”.<sup>38</sup> The ability to engage in warfighting functions – i.e., the term “weapons” – then represents the first basic module to be added. The base definition of “autonomous systems” may be augmented with non-weapon elements to serve in more specific definitions in other domains as well, or in non-combat aspects of the military such as bureaucratic, logistical, or other support roles.<sup>39</sup> This most basic definition may be given as follows:

Autonomous Systems (base definition): Machine-based systems that, if properly functioning, once activated, can identify and pursue (sets of) processes without contemporaneous human input which further a broader goal, task, or mission set by a human deployer.

By picking out a “machine-based system”, purely organic entities are excluded from the class of autonomous systems, and the assumption of proper functioning in this base definition removes the risk that malfunctioning systems are temporarily excluded from its extension. The selection of “machinebased” rather than simply “machine” also allows for inclusion of bio-robotic elements in systems without this automatically removing these from the base definition’s scope. Beyond these modifications, the base definition roughly follows the intent and framing of the standard view presented by the US DoD and the ICRC, but with respect to autonomous systems as such. Thus, the term tracks to autonomous execution of processes more generally, but not necessarily such processes with an explicitly warfighting function to them. For such a warfighting-centric base definition of “autonomous weapon systems”, we give the following:

Autonomous Weapon Systems (base definition): Machinebased weapon systems that, if properly functioning, once activated, can identify, select, and engage targets without contemporaneous human input.

As with the previous base definition, “machine-based” and “if properly functioning” serve to rule out those problem cases of the standard view discussed in 3.1 above. By adding identification, selection, and engagement of targets, the base definition of “autonomous systems” can be made more precise for the warfighting aspect of the military domain.

The reader might here object that the proposed base definition’s similarity to the standard view risks inheriting its weaknesses. In response, we stress that the core problem of the standard view was not that it was “wrong” per se, but rather that it was simply too broad.<sup>40</sup> The base definition thus takes the clarity and simplicity of the standard view, two virtues in a definition, and avoids the problems of over-broadness by anchoring that this is a base definition only. Thus, it only picks out the most basic and essential elements of an AWS, but does not articulate all or even necessarily the most important elements of some system for a given use-case or oversight aim. To achieve those more specific aims, the base definition must be augmented by modular terms picking out core concerns that may motivate ethical, legal, doctrinal, technological, or oversight debates. Yet even without those additions, central failings of the US DoD/ICRC standard view can be avoided, and the base definition serves as a useful grounding for discussion of mostfundamental regulations and oversight principles. For anything beyond those, added specificity will, however, be necessary.

## 5 A Conceptual Framework for Refined Definitions

Starting with a base definition of “autonomous systems”, “autonomous weapon systems” may be easily defined as a subcategory endowed with particular capabilities and fulfilling a particular role. The base definition of “autonomous weapon systems” given above is still too broad to be of use outside of mostfundamental regulatory, technical, or doctrinal discussions though. For any debates focusing on systems with specific levels of autonomy, used for particular combat functions, having additional capabilities, or otherwise drawing on more specific particularities of design and use, the base definition must be augmented with additional terms. The central aim of this section is to present a variety of such terms, focusing first on four elements which are apt to be central to core debates around AWS, after which we briefly canvas six narrower elements relevant for specific platforms or warfighting domains, and finally giving a brief guide to designing definitions for particular use cases. Having a catalog of such terms and a guide for how to leverage them for refined definitions is also critical, as without these, the base definition will not be useful for any purposes outside of establishing the most basic fundamental norms for AWS.<sup>41</sup>

It should be noted that while we present many terms which are natural candidates for refining the base definition in light of specific systems and use-cases, we do not claim our presentation to be complete. Thus, there may be other terms not captured by our discussion which can be included in a modular fashion to augment the base definition.<sup>42</sup> This is a central boon of having a modular conceptual framework for AWS rather a single unary definition treated as “the conception”. Given that emerging technologies are constantly changing, that warfare is driving rapid innovation, and that militaries are finding new ways of using existing technologies, it is preferable to have a framework for definitions rather than a single definition which is ill-suited to many use cases.

We also note that for reasons of space we will omit all discussion of systems which can be used as AWS, focusing instead on only those systems designed to be AWS in the course of their normal functioning.<sup>43</sup> Systems which can be used as AWS may also be captured under our definitional schema though by simply taking the base definition of “autonomous system” and adding to it that such systems are used – but not necessarily designed – to engage or attack some persons, objects, or capabilities. The modular schematic construction thus allows one to examine more than just AWS, and lays out refined definitions in a clear and traceable way, with all relevant points easily demarcated and able to be interrogated as needed.

## 5.1 Autonomy, State Transitions, and Self-Learning

One of the first and arguably most important elements to add to a base definition is what precise level of autonomy a system has, what internal states it can change on its own, and whether it has the capacity to add to, rewrite, or otherwise modify its own programming, approaches, or goals. Autonomy is moreover a critical term to have explicitly defined in a more refined definition of AWS (or autonomous systems generally) given that the term has distinctly diferent understandings in the philosophical/ethical/legal traditions compared to how it is used in engineering/computer science research. In the former, autonomy is deeply entwined with being a “first mover” or free agent that can make genuine decisions in the world, whereas the latter view focuses simply on independent execution of tasks without direct oversight or intervention. This distinction may be also captured in the diference between “making choices” and “executing processes”, with humans being endowed with the power to do the former, and machines only ever being able to do the latter (so far).<sup>44</sup> As succinctly put by Heather Rof, “[m]achine autonomy is therefore fundamentally diferent than philosophical or moral autonomy. Machine autonomy is merely the ability to act in the world without someone or something immediately directing that action”.<sup>45</sup>

In the literature on autonomous systems, difering views of autonomy have been put forward, with some focusing on scales of narrow machine autonomy rooted in levels of independence in task or process execution,<sup>46</sup> others focusing on more strict requirements such as being “capable of understanding higher-level intent and direction” as hurdles to be cleared before being considered “autonomous”.<sup>47</sup> A more general expression of autonomy which allows for the determination of difering levels and ranges of capability is given by Antsaklis (2020), who clarifies that “autonomy and autonomous imply that the system has the capability to accomplish certain goals”, with autonomy “always [being] with respect to certain goals”, and an “autonomous system always [having] a set of goals to be achieved and a control mechanism to achieve them. This implies that every autonomous system is a control system”.<sup>48</sup>

Now, scholars could (and likely will) continuously debate the exact contours of and distinctions between “automatic”, “automated”, and “autonomous” systems. But given that “there is no clear technical distinction between automated and autonomous systems, nor is there universal agreement on the meaning of these terms”,<sup>49</sup> it is far more useful to treat autonomy as a sliding scale, with merely (mechanistically) automatic entities at one end of the spectrum and full moral agents endowed with “first mover” capabilities at the other.<sup>50</sup> It should be settled though that machines do not ever “make decisions” in a philosophically, ethically, or legally relevant manner, and only ever “execute processes”. To that extent, all machines are “automatic” in the sense picked out by those authors wishing to draw hard distinctions between terms.

A further point worth noting is that while it would be prejudicial to define all AWS as inherently having in situ or online live learning (for reasons argued above), this capability can certainly be identified in refined definitions of particular types or classes of AWS. Moreover, such a refined definition is useful for picking out precisely these sorts of systems which likely would be illegal to use, and thus worth pre-emptively banning.

Regardless of one’s preferred views on what constitutes “autonomy” in systems, a refined modular conception of particular (classes of) AWS should clearly and explicitly identify what level of autonomy a system has, with the exact contours of autonomy levels or distinctions laid out following the presented definition.<sup>51</sup> An example refinement may look as follows:

Autonomous Weapon Systems (refined definition with autonomy): Machine-based weapon systems [with autonomy of level n] that, if properly functioning, once activated, can identify, select, and engage targets without contemporaneous human input.

## 5.2 F2T2EA

Related to the above, it is also critical to recognize that individual systems may have difering levels of autonomy in diferent tasks or areas of functioning. We should therefore be “thinking about ‘autonomous functions of systems, rather than characterizing an entire vehicle or system as ‘autonomous’ ”,<sup>52</sup> especially as autonomy may be used in isolated ways for an incredibly wide array of tasks, from streamlining mundane back-ofice aspects of military institutions to final kinetic engagement of enemy units.<sup>53</sup> One way to conceptualize the distinct tasks or functions augmented by autonomous capabilities is to look at the observe-orient-decide-act decision loop (OODA Loop) developed by U.S. Air Force Colonel John Boyd,<sup>54</sup> or to examine the more fine-grained breakdown of so-called “kill chains” through the F2T2EA model (find, fix, track, target, engage, assess).<sup>55</sup>

Considering the latter, each phase of the F2T2EA model may involve more discrete potentially autonomous capabilities as well. For example, the

“find” phase could involve systems which autonomously gather intelligence data, autonomously assess that data, autonomously flag objects/persons considered to be hostile, or some combination of these. Adding autonomy at diferent phases or for diferent functions can also implicate diferent ethical, legal, technical, or operational concerns. For example, autonomously tracking a target which a human has previously identified based on their own situational awareness is much less fraught than having a system that autonomously finds a target and then hands of tracking to a human agent. Full autonomy in engagement and assessment also raise numerous concerns not present with autonomous functions at other phases of the F2T2EA model. In short, it is not just the level of autonomy that matters, but also for what that autonomy is used. For refined definitions, this calls for an explicit mention of where autonomy factors into some system under consideration, both in terms of the functions where autonomy is present and the uses to which that system is put.

Autonomous Weapon Systems (refined definition with autonomy, functions): Machine-based weapon systems [with autonomy of level n] in [functions x, y, z] that, if properly functioning, once activated, can [execute functions x, y, z] without contemporaneous human input.

## 5.3 Lethality

In many discussions of AWS, lethality is a central element driving debate.<sup>56</sup> Now, there are fundamental issues with this category, given that lethality is generally a characteristic of use, and not of systems per se. Thus, a household hammer can be lethal or non-lethal in virtue of use, and there are few design decisions that could rule out its lethality. By the same token, high-explosive ordnance may be non-lethal, if it is directed purely at materiel and objects and not used against personnel or manned platforms. It is thus more correct to speak of systems designed to be lethal and those designed to be less lethal (as non-lethality generally cannot be guaranteed).<sup>57</sup> The only categories where lethality might be conclusively ruled out is for AWS operating purely in the electromagnetic environment (see Section 5.8), in the cyber domain (see Section 5.9), or in information/cognitive warfare systems (see Section 5.10).

The problems with lethality as a definitional marker notwithstanding, this is a characteristic regularly raised in debates on AWS, and its very close relationship with anti-personnel AWS also makes it a worthwhile feature to add. Any addition of this feature should highlight though that more/less lethality is as much derived from system use as it is from system design. It should also indicate that lethal AWS, that is, LAWS, are generally only to be LAWS because they are designed to be lethal, and not simply because they could be lethal in use.

Autonomous Weapon Systems (refined definition with autonomy, lethality): Machine-based weapon systems [with autonomy of level n] and [designed to be lethal/less-lethal] that, if properly functioning, once activated, can identify, select, and engage targets without contemporaneous human input, [resulting in death/not expected to result in death].

## 5.4 Anti-Personnel/Anti-Materiel

As “lethality” is related just as much to target classes (personnel vs. materiel) as it is to system design, this is a further core element which may feature in refined definitions. Whether a system is designed to target human beings also raises numerous ethical and legal worries, and poses unique engineering challenges as well. Anti-materiel systems, on the other hand, may often be designed so as to rely on very distinct markers of some platform type, and their ability to be deployed against platforms exclusively, potentially even platforms devoid of humans (incoming missiles, ordnance, or unmanned vehicles), makes their deployment less fraught and the need for regulation less acute. There are thus deep governance reasons to clearly distinguish the antipersonnel/anti-materiel character of certain AWS, should they be designed for one or another target class exclusively. Notably, purely anti-materiel systems may also be lethal, potentially by design, in virtue of being designed to destroy manned platforms, but the fact that they are by necessity targeting platforms greatly alters (and simplifies) the moral and legal considerations

being made.<sup>58</sup>

Autonomous Weapon Systems (refined definition with autonomy, functions, target class): Machine-based [antipersonnel/anti-materiel] weapon systems [with autonomy of level n] in [functions x, y, z] that, if properly functioning, once activated, can [execute functions $x , y , z ]$ against [individuals/objects] without contemporaneous human input.

## 5.5 Locomotion and Ofensive/Defensive Systems

The ofensive/defensive character of a system and whether it can move can also be relevant to regulatory and doctrinal discussions. While it is generally dificult to draw clean distinctions between “ofensive” and “defensive” weapon systems, as use will impact on this greatly, a system’s ability to move (or not) can significantly alter its efective ability to be deployed in ofensive or defensive roles. Given that defensive uses of force are generally easier to justify and more likely to generate greater latitude in decision-making, this factor may be relevant to build into certain refined definitions (based on use case). Thus, while there may be dificulties in establishing hard distinctions between ofensive and defensive weapons, the terms can ofer useful heuristics for debate, and locomotion represents a core capability of ofensive systems. Where AWS are fixed to points or other platforms, they will often be serving in defensive roles for those points/platforms, and both their static nature and greater likelihood to fulfill a defensive role may impact on regulation or doctrine.

## 5.6 Swarmed

While not yet a fully mature class of technology, there is increasing interest in swarming capabilities for AWS, especially aerial systems which may be designed to achieve mission success through saturation rather than precision and hard-to-intercept fires. Developers claim continuing progress in this domain,<sup>59</sup> and the trend in unmanned and autonomous aerial systems is toward greater platform integration and swarming capabilities. Swarming raises distinct ethical, legal, technical, and operational concerns though, and AWS which can execute swarming operations or designed to be swarmed systems should have that element picked out in refined definitions which drive debate around swarms/swarming.

## 5.7 Kinetic/Explosive

Systems which engage targets using kinetic or explosive force raise distinct challenges relating to whether they would “cause superfluous injury or unnecessary sufering”<sup>60</sup> and regarding the risks they pose to civilians. In particular, given that combatants are required to distinguish between civilians and combatants, directing attacks only at the latter;<sup>61</sup> that combatants must use methods and means of warfare which can be directed at military objectives;<sup>62</sup> and that combatants must take all feasible precautions to minimize harm to civilians and warn them of risks attending attacks,<sup>63</sup> kinetic/explosive methods and means of war create special challenges. This is because attacks causing kinetic/explosive efects may create immediate and potentially lethal risks to the civilian population. Picking out the (non-)kinetic/explosive character of AWS in refined definitions may thus be relevant to specific debates and potential regulatory eforts, especially those relating to the use of such weapons in urban environments.<sup>64</sup>

## 5.8 Electronic Warfare

Electronic warfare (EW) has become increasingly critical in modern war, 65 and emerging AWS, especially those designed to counter unmanned and autonomous systems, may increasingly feature EW capabilities or even be given

EW roles as their primary task.<sup>66</sup> Where kinetic attacks may directly cause risks of physical harm for combatants and civilians, EW interferes with the electromagnetic spectrum in which communications and data are transmitted, and through which core elements of navigation are relayed. This thus creates equally real risks for individuals, but mediated through failures of navigation, identification of (air)craft, or related concerns. Much EW is also dificult or impossible to tightly direct toward purely military objects, raising issues of distinction.<sup>67</sup> Given these distinct issues germane to EW, but less relevant or even nonexistent for other platforms, this may be a capability in need of explicit mention in refined definitions focusing on regulation around EW at large, and AWS capable of engaging in EW.

## 5.9 Cyber

Like EW, cyber-warfare does not immediately imply risks of harm to either combatants or civilians, but it can indeed lead to harm through the efects of cyber-attacks. Cyber-attacks can target both software and the hardware upon which software systems are built<sup>68</sup> and can be used to target civilian infrastructures<sup>69</sup> or feature as a core element of kinetic warfighting.<sup>70</sup> The dual-use nature of many critical infrastructures and the dificulty in conducting cyber-attacks that target purely military systems raises significant legal worries as well.<sup>71</sup> Furthermore, some forms of cyber-attack can be dificult to control once initiated, raising legal concerns over distinction and one’s ability to control the efects of attacks (as required by AP I, Art. 51.4.c).<sup>72</sup> For these reasons, AWS which are designed to execute attacks or carry out tasks in the cyber domain raise distinct worries and are worth having their cyber capabilities picked out in refined definitions dealing with the intersection of AWS and cyber-warfare.

## 5.10 Information/Cognitive Warfare

In addition to systems which are clearly weapons, some tactics of warfighting have nothing to do with exerting force on a battlefield, either directly or indirectly, and instead target the will and resolve of a population. Information operations can be carried out to sow uncertainty, entrench and exacerbate societal divisions, muddy efective response to aggression, or win over part of a population to an aggressor’s cause (to name just a few).<sup>73</sup> Cognitive warfare, on the other hand, can be leveraged to disrupt and undermine the very ways that a population thinks, making them untrusting of information, causing them to withdraw from complex issues, or exerting “reflexive control” to “prompt behavior that will give the principal party a competitive advantage”.<sup>74</sup> One may object to these being brought under the umbrella of autonomous weapon systems, but the increasing capability of large language models and agentic AI systems is allowing for information and cognitive warfare to be massively scaled up with very little cost,<sup>75</sup> and using fully autonomous systems which can directly engage a target population without direct human input or oversight.<sup>76</sup>

This is not to say that a definition of AWS need be so broad as to include these psychological aspects of modern warfighting, but to highlight that a modular definitional schema can accommodate such expansion and specificity as needed, or can be used to pick out more traditional kinetic systems with ease. In any event, the approach developed here creates clear, traceable, and useful routes for introducing or eliminating components from a particular definition, greatly improving over the status quo alternatives.

## 5.11 General Recommendations on Definition Construction

As noted by Dubber (2026a), “military autonomy spans domains (air, land, sea, space, cyber), functions (targeting, defence, logistics, intelligence, command support), levels (tactical, operational, strategic), and phases of conflict (cooperation, competition, crisis, conflict)” and “[e]ach combination of these dimensions presents a distinct profile of control, error, reversibility, and moral risk—and treating them as a single class obscures more than it reveals.”.<sup>77</sup> A unary conception of AWS is thus unfit for any but the most broad-based and fundamental of tasks, such as establishing basic understandings and grounding principles. For anything more specific – and there is much need for more specific rules and best practices in a host of use cases – refined definitions will be necessary. In developing such refinements, scholars, practitioners, and policy-makers should focus first and foremost on the use case they have in mind which requires a definition. As no single view is likely to be fully satisfactory, the central concern should be whether definitions are fit for purpose.

In crafting purpose-driven definitions, it should also be clear that various terms may be slotted into the base definition in diferent manners, as illustrated in Sections 5.1–5.4. There will not be one “correct” way to incorporate relevant terms in refined definitions, as the exact reason for having a definition and the stakeholder group who will use it can greatly impact on how best to develop a refined understanding. No matter how one goes about this though, it is centrally important that terms be viewed in a pragmatic fashion, as modules which can be leveraged in a “plug-and-play” fashion to serve discrete regulatory, doctrinal, or scholarly needs.

As argued at the outset of this article, man-made artifacts do not readily conform to a “scientific” definition, and in domains where the exact definition of a term can greatly color subsequent discourse, there is value in being agnostic and pragmatic about how precisely to define core terms. That is not to say that definitions do not matter or may be constructed simply as one sees fit. But recognizing the diversity of views on AWS (or autonomous systems, artificial intelligence, etc.), we should retain both enough intellectual humility to recognize our preferred definition may be reasonably objected to and enough humanistic feeling to put progress ahead of “being correct” on definitional matters.

This means that constructing definitions should be a group exercise carried out by relevant stakeholders in a domain. Such stakeholders should be open and honest about what matters to them in a precise definition, what concerns they have which they believe may be better discussed with a shared definition in mind, and what political and other aims they wish to pursue using that definition (as honesty on that front can prevent unnecessary con-

fusion and wasted time).

## 6 Conclusion

It has been over ten years since the Group of Governmental Experts on Lethal Autonomous Weapon Systems began its discussions, and still there is no broadly agreed consensus in the international community on how best to regulate AWS, nor even on what AWS are.<sup>78</sup> There are many reasons for this glacial and uncertain progress, but recurrent debates and relitigations of definitions has certainly not helped matters. Moreover, certain actors have added mud to the definitional waters simply for the sake of political ends,<sup>79</sup> with resulting intellectual debates faltering on core terms while technologies continue evolving with little explicit oversight.

As there is scant hope in trying to simply impose a definition on all parties and being met with agreement, and due to core limitations in any definition treated as the conception of AWS, we have argued that we should instead aim for a base definition to be refined with core terms as needed. Such a definitional schematic framework allows one to remain open to changes in technology, to fit definitions to discrete needs and systems, and to develop distinct definitions suited to diferent stakeholder groups. The guiding light in this approach is the blunt pragmatism embodied by the International Committee of the Red Cross, which wisely recalls that “unable to eliminate the scourge of war, one endeavors to master it and mitigate its efects”.<sup>80</sup> The point is not to have the “right” view. It is to have a view which is useful in helping us to master the scourge of war, to mitigate its efects.

## References

Altmann, J. and F. Sauer (2017). Autonomous weapon systems and strategic stability. Survival 59 (5), 117–142.

Antsaklis, P. (2020). Autonomy and metrics of autonomy. Annual Reviews in Control 49, 15–26.

Asaro, P. (2012). On banning autonomous weapon systems: Human rights, automation, and the dehumanization of lethal decision-making. International Review of the Red Cross 94 (886), 687–709.

Asaro, P. (2019). Algorithms of violence: Critical social perspectives on autonomous weapons. Social Research: An International Quarterly 86(2), 537– 555.

Asghari, H., N. Birner, A. Burchardt, D. Dicks, J. Faßbender, N. Feldhus, F. Hewett, V. Hofmann, M. C. Kettemann, W. Schulz, J. Simon, J. Stolberg-Larsen, and T. Züger (2021). What to explain when explaining is dificult? an interdisciplinary primer on XAI and meaningful information in automated decision-making. Technical report.

Blanchard, A. and M. Taddeo (2022). Predictability, distinction & due care in the use of lethal autonomous weapon systems. SSRN Electronic Journal.

Blanchard, A. and M. Taddeo (2023). Jus in bello necessity, the requirement of minimal force, and autonomous weapons systems. Journal of Military Ethics, 1–18.

Boothby, W. H. (2014a). The legal challenges of new technologies: An overview. In H. Nasu and R. McLaughlin (Eds.), New Technologies and the Law of Armed Conflict, pp. 21–28. The Hague, Netherlands: T.M.C. Asser Press.

Boothby, W. H. (2014b). Where do cyber hostilities fit in the international law maze. In H. Nasu and R. McLaughlin (Eds.), New Technologies and the Law of Armed Conflict, pp. 59–73. The Hague, Netherlands: T.M.C. Asser Press.

Boothby, W. H. (2016). Weapons and the Law of Armed Conflict (2 ed.). Oxford, UK: Oxford University Press.

Boulanin, V., N. Davison, N. Goussac, and M. P. Carlsson (2020). Limits on autonomy in weapon systems: Identifying practical elements of human control. Technical report, Stockholm International Peace Research Institute and International Committee of the Red Cross.

Bray, D. (2026). Hardware as the hidden battlefield. OODA Loop. https://oodaloop.com/analysis/disruptive-technology/hardware-asthe-hidden-battlefield/, Accessed: July 15, 2026.

Briefer, E. F., S. Haque, L. Baciadonna, and A. G. McElligott (2014). Goats excel at learning and remembering a highly novel cognitive task. Frontiers in Zoology 11 (1), 20.

Bukowski, M. F. (2025). Agents of Chaos: The Shadow Campaign Against the West. Warsaw, Poland: Casimir Pulaski Foundation.

Burton, I. and J. Straub (2019). Autonomous distributed electronic warfare system of systems. In 2019 14th Annual Conference System of Systems Engineering (SoSE), pp. 96–101. IEEE.

Caron, J.-F. (2020). Defining semi-autonomous, automated and autonomous weapon systems in order to understand their ethical challenges. Digital War 1 (1- 3), 173–177.

Casterline, K. A., N. J. Watkins, J. R. War, W. Li, and M. J. Thommana (2022). Applications of machine learning for electronic warfare emitter idenficiation and resource management. Johns Hopkins APL Technical Digest 36 (2), 121–140.

Caws, P. (1959). The functions of definition in science. Philosophy of Science 26 (3), 201–239.

Chen, P. and M. Unewisse (2017). SoS thinking: An approach to conceptualising and understanding military systems-of-systems. International Journal of System of Systems Engineering 8 (1), 74.

Chengeta, T. (2016). Measuring autonomous weapon systems against international humanitarian law rules. Journal of Law & Cyber Warfare 5, 66–146.

Corlett, J. A. (1996). Can terrorism be morally justified? Public Afairs Quarterly, 163–184.

Coufer, J. (1992). Bat Bomb: World War II’s other secret weapon. Austin, TX: University of Texas Press.

Crane, C. (2008). Bats over Tokyo. https://www.army.mil/article/6993/bats\_ over\_tokyo, Accessed: Aug. 19, 2026.

de Goeij, M. W. R. (2023). Reflexive control: Influencing strategic behavior. The US Army War College Quarterly: Parameters 53 (4).

Dubber, T. (2026a). Abhorrence and autonomy: On the place of public attitudes in the debate on militarised artificial intelligence. Philosophy & Technology 39(3).

Dubber, T. (2026b). Domain compression: The prospects of AI autonomy across warfighting domains. SSRN Electronic Journal.

Dubber, T. and S. Lazar (2025). Military AI cyber agents (MAICAs) constitute a global threat to critical infrastructure. arXiv.

Dym, C. L. (1994). Engineering Design. Cambridge, UK: Cambridge University Press.

Enck, R. E. (2012). The OODA loop. Home Health Care Management & Practice 24 (3), 123–124.

Ezenkwu, C. P. and A. Starkey (2019). Machine autonomy: Definition, approaches, challenges and research gaps. In Intelligent Computing-Proceedings of the Computing Conference, pp. 335–358. Springer.

Floridi, L. and M. Taddeo (Eds.) (2014). The Ethics of Information Warfare. Springer.

Frannsen, M., G.-J. Lokhorst, and I. van de Poel (2024). Philosophy of technology. The Stanford Encyclopedia of Philosophy. https://plato.stanford.edu/ archives/fall2024/entries/technology/.

Fredheim, R. and J. Pamment (2024). Assessing the risks and opportunities posed by AI-enhanced influence operations on social media. Place Branding and Public Diplomacy 21 (3), 319–326.

Ginet, C. (1997). Freedom, responsibility, and agency. The Journal of Ethics 1 (1), 85–98.

Gioe, D. V., R. Lovering, and T. Pachesny (2020). The Soviet legacy of Russian active measures: New vodka from old stills? International Journal of Intelligence and CounterIntelligence 33 (3), 514–539.

Goldstein, J. A., G. Sastry, M. Musser, R. DiResta, M. Gentzel, and K. Sedova (2023). Generative language models and automated influence operations: Emerging threats and potential mitigations. arXiv.

Group of Governmental Experts on Emerging Technologies in the Area of Lethal Autonomous Weapons Systems (2026). Report of the 2024-2025-2026 sessions of the Group of Governmental Experts on Emerging Technologies in the Area of Lethal Autonomous Weapons Systems.

Group of Governmental Experts on Lethal Autonomous Weapons Systems (2025). GGE on LAWS: Rolling text, status date: 12 may 2025.

Guetlein, M. A. (2005). Lethal autonomous weapons – ethical and doctrinal implications. Technical report, United States Naval War College.

Haidegger, T. (2019). Autonomy for surgical robots: Concepts and paradigms. IEEE Transactions on Medical Robotics and Bionics 1 (2), 65–76.

Hammerstrom, M. L. (2005). Ground Dog Day Lessons Don’t Have to be Relearned in the Use of Dogs in Combat. Ph. D. thesis, Naval Postgraduate School.

Hartmann, K. and K. Giles (2020). The next generation of cyber-enabled information warfare. In 2020 12th International Conference on Cyber Conflict (CyCon), pp. 233–250. IEEE.

Haugh, B. A., D. A. Sparrow, and D. M. Tate (2018). The status of test, evaluation, verification, and validation (TEV&V) of autonomous systems. Technical report, Institute for Defense Analysis.

Havlík, M. and J. Horáček (2026). War is a mind game: Countering weaponized information. Outlook: Strategy Series (1).

Heller, K. J. (2023). The concept of “the human” in the critique of autonomous weapons. Harvard National Security Journal 15 (1), 1–76.

Hibberd, F. J. (2019). What is scientific definition? The Journal of Mind and Behavior 40 (1), 29–52.

Hofman, F. (2025). Assessing “cognitive warfare”. Small Wars Journal. https: //smallwarsjournal.com/2025/11/14/assessing-cognitive-warfare/.

Hossain, N. U. I. (2018). A synthesis of definitions for systems engineering. In Proceedings of the International Annual Conference of the American Society for Engineering Management, pp. 1–10. American Society for Engineering Management (ASEM).

Human Rights Watch (2018). Heed the call: A moral and legal imperative to ban killer robots. Technical report, Human Rights Watch.

Human Rights Watch (2020). New weapons, proven precedent: Elements of and modes for a treaty on killer robots. Technical report, Human Rights Watch.

International Committee of the Red Cross (1987). Commentary on the Additional Protocols. Geneva, Switzerland: Martinus Nijhof Publishers.

International Committee of the Red Cross (2014). Autonomous weapons systems: Technical, military, legal and humanitarian aspects. Technical report, International Committee of the Red Cross.

International Committee of the Red Cross (2019). Autonomy, artificial intelligence and robotics: Technical aspects of human control. Technical report, International Committee of the Red Cross.

International Committee of the Red Cross (2021a). ICRC position and background paper on autonomous weapons systems. Technical report, International Committee of the Red Cross.

International Committee of the Red Cross (2021b). ICRC position on autonomous weapons systems. Technical report, International Committee of the Red Cross.

Jevglevskaja, N. (2018). Weapons review obligation under customary international law. International Law Studies 94, 186–221.

Jevglevskaja, N. (2021a). Challenges to Article 36 Reviews Posed by Autonomous Weapons Systems (AWS), pp. 207–238. Cambridge, UK: Cambridge University Press.

Jevglevskaja, N. (2021b). International Law and Weapons Review: Emerging Military Technology under the Law of Armed Conflict. Cambridge, UK: Cambridge University Press.

Johansmeyer, T. (2026). Small and Unthreatening. JSOU Report 26-10. MacDill Air Force Base, FL: Joint Special Operations University.

Kajander, A., R. Liivoja, and M. Naagel (2025). Cybersecurity of weapon systems: International law requirements and technical standards. Journal of Cybersecurity 11 (1), 1–15.

Kallioniemi, P. (2025). Beyond defence: A proactive strategy for the West in the information domain. Technical report, International Centre for Defence and Security, Tallinn, Estonia. https://icds.ee/en/beyond-defence-a-proactivestrategy-for-the-west-in-the-information-domain/.

Kennett, J. (2003). Agency and Responsibility: A common-sense moral psychology. Oxford, UK: Clarendon Press.

Knack, A., A. Bowden, N. Karner, L. Fletcher, R. Barrett-Taylor, O. Wickens, M. Welsh, and T. Whittaker (2025). Defence AI assurance: Identifying promising practice and a system card template for defence. Technical report.

Kobayashi, M. and P. F. Mellen (2009). Rubber bullet injury: Case report with autopsy observation and literature review. The American Journal of Forensic Medicine and Pathology 30 (3), 262–267.

Krewson, T. M. (2025). The brain is both the target and the weapon: The strategic necessity of NATO’s cognitive warfare concept. The Three Swords. https: //www.jwc.nato.int/our-work/three-swords-41/.

Krishnan, M. (2019). Against interpretability: a critical examination of the interpretability problem in machine learning. Philosophy & Technology 33 (3), 487–502.

Lehto, M. and G. Henselmann (2020). Non-kinetic warfare: The new game changer in the battle space. In B. K. Payne and H. Wu (Eds.), Proceedings of the 15th International Conference on Cyber Warfare and Security, pp. 316–325.

Lemish, M. G. (2008). War Dogs. Potomac Books Inc.

Lin, P., G. A. Bekey, and K. Abney (2008). Autonomous military robotics: Risk, ethics, and design. Technical report, DTIC Document.

Mantellassi, F. and G. P. Paoli (2024). Report: Cloud computing and international security: Risks, opportunities and governance challenges. Technical report, United Nations Institute for Disarmament Research (UNIDIR).

Manthorpe, W. H. (1996). The emerging joint system of systems: A systems engineering challenge and opportunity for APL. In Johns Hopkins APL Technical Digest, Volume 17, pp. 305–313.

McFarland, T. and Z. Assaad (2023). Legal reviews of in situ learning in autonomous weapons. Ethics and Information Technology 25 (1), 1–10.

Meerveld, H., L. Peperkamp, M. Šafář Postma, and R. Lindelauf (2025). Operationalising responsible AI in the military domain: A context-specific assessment. Ethics and Information Technology 27 (4).

Meerveld, H. W., R. Lindelauf, E. O. Postma, and M. Postma (2023). The irresponsibility of not using AI in the military. Ethics and Information Technology 25 (1), 14.

Michie, S., R. West, and J. Hastings (2019). Creating ontological definitions for use in science. Qeios.

Midson, D. (2014). Geography, territory, and sovereignty in cyber warfare. In H. Nasu and R. McLaughlin (Eds.), New Technologies and the Law of Armed Conflict, pp. 74–93. The Hague, Netherlands: T.M.C. Asser Press.

Moor, J. H. (2006). The nature, importance, and dificulty of machine ethics. IEEE Intelligent Systems 21 (4), 18–21.

Nowrot, K. (2015). Animals at war: The status of “animal soldiers” under international humanitarian law. Historical Social Research 40, 128–150.

Olejnik, L. (2024). Propaganda: From Disinformation and Influence to Operations and Information Warfare. CRC Press.

Owens, W. A. (1996). The emerging US system-of-systems. Strategic Forum (63), 1–4.

Pacholska, M. (2023). Military artificial intelligence and the principle of distinction: A state responsibility perspective. Israel Law Review 56 (1), 3–23.

Pacholska, M. (2024). Autonomous weapons. In B. Brożek, O. Kanevskaia, and P. Pałka (Eds.), Research Handbook on Law and Technology, pp. 392–407. Edward Elgar Publishing.

Quantum Systems (2024a). Airbus and Quantum Systems make progress in autonomous swarm technology. https://quantum-systems.com/news/quantumsystems-makes-progress-in-swarm-technology/, Accessed: Jan. 22, 2026.

Quantum Systems (2024b). Quantum Systems achieves milestone on the way to swarm technology: Successful test of several unmanned systems in KITU2 study. https://quantum-systems.com/news/ila\_swarming/, Accessed: Jan. 22, 2026.

Rof, H. (2024). Magnifying human confusion: Meaningful human control and the ongoing debate on autonomous weapons. The Rule of Law Post. https://www.penncerl.org/the-rule-of-law-post/magnifying-humanconfusion-meaningful-human-control-and-the-ongoing-debate-onautonomous-weapons/.

Rof, H. M. (2013). Killing in war: Responsibility, liability, and lethal autonomous robots. In F. Allhof, N. G. Evans, and A. Henschke (Eds.), Routledge Handbook

of Ethics and War: Just War Theory in the 21 $\mathcal { Z } \mathcal { 1 } ^ { s t }$ Century, pp. 352–364. New York, NY: Routledge.

Rof, H. M. (2014). The strategic robot problem: Lethal autonomous weapons in war. Journal of Military Ethics 13(3), 211–227.

Rogak, L. (2011). The Dogs of War. New York, NY: St. Martin’s Press.

Russo, S. (2026). Cognitive warfare is cheap – and that’s the problem. Small Wars Journal. https://smallwarsjournal.com/2026/01/30/cognitive-warfareis-cheap/.

Scharre, P. (2023). Four Battlegrounds. New York, NY: W. W. Norton & Company.

Scharre, P. and M. C. Horowitz (2015). An introduction to autonomy in weapons systems. Technical report, Center for a New American Security.

Schmitt, M. N. (2012a). “Attack” as a term of art in international law: The cyber operations context. In 2012 4th international conference on cyber conflict (CYCON 2012), pp. 1–11. IEEE.

Schmitt, M. N. (2012b). Autonomous weapon systems and international humanitarian law: A reply to the critics. Harvard National Security Journal Feature.

Schmitt, M. N. (Ed.) (2017). Tallinn Manual 2.0 on the International Law Applicable to Cyber Operations (2 ed.). Cambridge, UK: Cambridge University Press.

Schwenkenbecher, A. (2012). Terrorism: A Philosophical Enquiry. London, UK: Palgrave MacMillan.

Scriven, M. (1958). Definitions, explanations, and theories. Minnesota Studies in the Philosophy of Science 2, 99–195.

Shapiro, A. (2019). Autonomous Weapon Systems: Selected Implications for International Security and for Canada. Ottawa, CA.

Sharkey, N. (2007). Automated killers and the computing profession. Computer 40 (11), 124–123.

Sharkey, N. (2008). Grounds for discrimination: Autonomous robot weapons. RUSI Defence Systems 11 (2), 86–89.

Sharma, P., K. K. Sarma, and N. E. Mastorakis (2020). Artificial intelligence aided electronic warfare systems – recent trends and evolving applications. IEEE Access 8, 224761–224780.

Sillitto, H., J. Martin, D. McKinney, R. Griego, D. Dori, D. Krob, P. Godfrey, E. Arnold, and S. Jackson (2019). Systems engineering and system definitions. In INCOSE. sn.

Sparrow, R. (2007). Killer robots. Journal of Applied Philosophy 24 (1), 62–77.

Sparrow, R. (2016). Robots and respect: Assessing the case against autonomous weapon systems. Ethics & International Afairs 30 (01), 93–116.

Stop Killer Robots (2022). Negotiating a treaty on autonomous weapons systems: The way forward. Technical report, Stop Killer Robots.

Taddeo, M. (2014). Information warfare and just war theory. In L. Floridi and M. Taddeo (Eds.), The Ethics of Information Warfare, pp. 123–138. Springer.

Taddeo, M. and A. Blanchard (2022). A comparative analysis of the definitions of autonomous weapons systems. Science and Engineering Ethics 28 (5), 1–22.

Tatham, S. (2025). Information Operations. Havant, UK: Howgate Publishing.

Tiboris, M. (2014). Blaming the kids: Children’s agency and diminished responsibility. Journal of Applied Philosophy 31 (1), 77–90.

Tirpak, J. A. (2000). Find, fix, track, target, engage, assess. Air & Space Forces Magazine. https://www.airandspaceforces.com/article/0700find/, Accessed: Aug. 31, 2026.

Traina, C. L. H. (2009). Children and moral agency. Journal of the Society of Christian Ethics 29(2), 19–37.

Tsakanyan, V. (2026). How modern cyber strategies are leaving critical infrastructure in the shadows. Center for Cyber Diplomacy and International Security. https://cybercenter.space/2026/01/09/how-modern-cyber-strategiesare-leaving-critical-infrastructure-in-the-shadows/, Accessed: Jan. 23, 2026.

UK Ministry of Defence (2017). Joint doctrine publication 0-30.2: Unmanned aircraft systems. Technical report, United Kingdom Ministry of Defence.

UK Ministry of Defence (2022). Joint doctrine publication 0-30: UK air power. Technical report, United Kingdom Ministry of Defence.

US Department of Defense (2023). DoD Directive 3000.09. Technical report, United States Department of Defense.

Vakin, S. A., L. N. Shustov, and R. H. Dunwell (2001). Fundamentals of Electronic Warfare. Boston, MA: Artech House.

Varella, L. (2026). CCW report: Civil society perspectives on the Group of Governmental Experts of the Convention on Certain Convetional Weapons on Lethal Autonomous Weapons Systems, 31 August-4September 2026.

Verbruggen, M. (2022). No, not that verification: Challenges posed by testing, evaluation, validation and verification of artificial intelligence in weapon systems. In T. Reinhold and N. Schörnig (Eds.), Armament, Arms Control and Artificial Intelligence, Studies in Peace and Security, pp. 175–192. Springer.

Voloshchuk, V., S. Banakh, and O. Kovalchuk (2025). Artificial intelligence in military strategy: New approaches to command management. Aktual’ni Problemy Pravoznavstva (Current Problems of Law) (1), 69–74. https://appj.wunu.edu. ua/index.php/appj/article/view/2050/2077.

Waller, A. M. (1958). Dogs and National Defense. Department of the Army, Ofice of the Quartermaster General.

Williams, A. P. (2015). Defining autonomy in systems: Challenges and solutions. In A. P. Williams and P. D. Scharre (Eds.), Autonomous Systems: Issues for Defense Policymakers, pp. 27–62. The Hague, Netherlands: NATO Communications and Information Agency.

Wood, N. G. (2020). The problem with killer robots. Journal of Military Ethics 19(3), 220–240.

Wood, N. G. (2023a). Autonomous weapon systems: A clarification. Journal of Military Ethics 22(1), 18–32.

Wood, N. G. (2023b). Autonomous weapon systems and responsibility gaps: A taxonomy. Ethics and Information Technology 25 (1), 1–14.

Wood, N. G. (2024). Regulating autonomous and AI-enabled weapon systems: The dangers of hype. AI and Ethics 4, 805–817.

Wood, N. G. (2026a). Autonomous and AI-enabled systems: Extensions or replacements of human will and control? Ethics and Information Technology 28 (1), 1–16.

Wood, N. G. (2026b). Bombs, bots, and the principle of distinction: The law of armed conflict and contemporary warfare. Texas National Security Review 9 (1), 52–67.

Wood, N. G., S. Robbins, E. Zegarra Berodt, A. Graf von Westerholt, M. Behrndt, H. Budig, and D. Kloock-Schreiber (2026). Stop saying “AI”. Philosophy & Technology 39 (2), 1–35.

Yang, G.-Z., J. Cambias, K. Cleary, E. Daimler, J. Drake, P. E. Dupont, N. Hata, P. Kazanzides, S. Martel, R. V. Patel, V. J. Santos, and R. H. Taylor (2017). Medical robotics – regulatory, ethical, and legal considerations for increasing levels of autonomy. Science Robotics 2(4), 1–2.

Yu, A., I. Kolotylo, H. A. Hashim, and A. E. E. Eltoukhy (2025). Electronic warfare cyberattacks, countermeasures, and modern defensive strategies of UAV avionics: A survey. IEEE Access 13, 68660–68681.

Zając, M. (2022). Autonomous Weapon Systems from a Just War Theory Perspective. Ph. D. thesis, University of Warsaw.

Zając, M. (2026). Problem of (in)explainability in testing fully autonomous weapon systems for international humanitarian law compliance. Minds and Machines 36(3).

Zhang, C., L. Wang, R. Jiang, J. Hu, and S. Xu (2023). Radar jamming decisionmaking in cognitive electronic warfare: A review. IEEE Sensors Journal 23(11), 11383–11403.

Zilincik, S. (2026). Cognitive warfare: Some sceptical observations from general strategic theory and afective science. European Journal of International Security, 1–22.