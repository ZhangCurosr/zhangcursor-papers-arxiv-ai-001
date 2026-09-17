ARC Scales Al & Robot Mentorship into Rural Zones

# Teaching AI, Robotics, & Community: A Hubs-Based K–12 Education Framework for Reaching Rural Schools

Maxwell J. Jacobson, Gustavo Rodriguez-Rivera, Petros Drineas, and Yexiang Xue

Purdue University

jacobs57@purdue.edu, grr@purdue.edu, pdrineas@purdue.edu, yexiang@purdue.edu

## Abstract

K–12 robotics and AI education remains dificult to scale, especially in rural regions lacking sustained technical mentorship. Programs like FIRST provide competition pathways and instructional opportunities, but they do not eliminate the need for local programming and robotics expertise. We introduce AI, Robotics, & Community (ARC), a hubs-based framework where colleges train undergraduate mentors and host workshops for nearby K–12 teams. Mature school programs can become secondary hubs that support additional schools, creating a self-reinforcing education loop where mentorship reach propagates geographically and can even grow super-linearly. We first evaluate ARC through a trial deployment at one uni versity. The trial created three rural robotics teams. On fivepoint Likert surveys, mean increases in K–12 programming knowledge, resource access, and practice opportunities were 2.00, 2.25, and 1.25 points. Likewise, undergraduate confidence teaching technical concepts, adapting explanations, managing groups, and finding mentoring enjoyable and meaningful increased by 1.29, 1.14, 1.00, and 1.14 points. Additionally, we create a spatial Markov model of ARC’s growth and simulate it using the state of Indiana as a testbed. Under moderate conditions, we find that ARC reaches 74% of Indiana’s 1,925 public K–12 schools and produces 992 robotics programs after 40 years, compared with 161 projected under natural growth alone. Together, these results show ARC can create and support rural robotics programs, train undergraduate AI and robotics mentors, and potentially scale mentorship across a region.

## Introduction

Scaling AI and robotics literacy in K–12 education is dificult, especially in under-served rural regions. Robotics and AI are increasingly important to future economies, innovation, and civic life, but access to meaningful robotics and AI learning is uneven for schools that lack sustained technical support. Indiana provides an example of this access gap, where rural participation in the FIRST LEGO League (FLL) robotics competition has recovered much more slowly than urban participation following the 2020 remote season. Without early exposure, students may be less prepared to understand how AI and robots work and may become passive consumers of technology rather than active creators or critical evaluators.

Existing robotics programs provide a strong foundation for K–12 robotics education. In particular, the FIRST ecosystem ofers a progression of team based competitions, giving students opportunities to build, program, and compete with robots at increasing levels of complexity. Schools, teachers, and community volunteers already support these programs through local clubs and competitions. Related work has also developed curricula for teaching AI concepts and ethics in middle and high school (Williams, Kaputsos, and Breazeal 2021; Alvarez et al. 2022; Krakowski et al. 2022).

![](images/294f08b489337bb883563b732e5b3619c17f375977bdca66d931715848702ba6.jpg)  
Figure 1: Overview of the ARC hubs-based education framework. Primary hubs at colleges train undergraduate mentors in robotics and AI and host workshops for nearby K–12 robotics teams. As programs mature, some become secondary hubs that extend robotics and AI mentorship into rural areas and support additional nearby schools, allowing the network to propagate beyond the reach of universities.

The missing piece is not the absence of robotics programs or educational material, but the dificulty of scaling sustained technical mentorship. Many K–12 teams depend on a small number of volunteers, and geographically dispersed rural schools may have limited access to mentors with programming, robotics, or AI expertise. A school may have interested students and a willing teacher while still lacking the technical knowledge needed to start and sustain a program. This mentorship gap limits the formation, persistence, and growth of K–12 robotics teams. Students in under-supported regions therefore have fewer opportunities to develop programming, engineering, and AI literacy even when interest exists. At the same time, undergraduate computer science students often have few structured opportunities to deepen their own technical understanding through teaching and to apply their knowledge in real educational settings.

![](images/4f3b16c18c01f13707a9d8a913cf00c1328b05ac248e56ae0ac2818bc5d16422.jpg)  
Figure 2: Left: FIRST LEGO League participation in Indiana before and after the 2020 remote season participation crash. The left panel shows urban, rural, and total FLL organizations from 2015–2025. The maps show the geographic distribution of participating organizations in 2017 (the high point) and 2025 relative to Census urban areas and major STEM universities. Urban participation has gradually recovered, while rural participation has not seen a recovery. Right: ARC creates a self-reinforcing education loop. The primary loop (blue) shows how hubs host workshops that support K–12 schools, whose teams can mature into secondary hubs that expand the overall hub network and enable additional workshops. Zooming in, the undergraduate growth loop (red) shows how primary hubs create undergraduate mentors, who can become experienced mentors and later lead or support primary hubs, allowing those hubs to train additional undergraduates. Finally, the rural growth loop (green) shows how new secondary hubs expand geographic reach, and how that greater reach may yield additional secondary hubs. These loops integrate at multiple points, reinforcing themselves and each other to produce three desirable outcomes: more team support, more undergraduate growth, and greater reach into rural areas.

We propose the AI, Robotics, & Community education framework (ARC) for scaling K–12 robotics and AI mentorship. ARC organizes hands-on workshop events around regional hubs, allowing multiple teams in an area to receive direct technical mentorship. Each team works with a small number of mentors on the programming and robotics skills needed to make progress in the current competition. Workshop content is adapted to age and competition level, from programming and robot design for younger teams to autonomy, computer vision, and AI for advanced teams.

Primary hubs are colleges or other STEM institutions that run the ARC course, train undergraduate computer science students as robotics and AI mentors, and host workshops for nearby K–12 teams. An experienced mentor prepares the undergraduates in technical skills as well as the educational and interpersonal skills needed to teach younger students.

The undergraduates then lead the workshops under supervision. The course also exposes them to local research in AI and robotics and its efects on their communities. By helping students see technical mentorship as a meaningful role within their local community, ARC aims to encourage continued mentoring and volunteering after the course ends.

As K–12 teams become mature and self-sustaining, some can become secondary hubs that host workshops for nearby schools with limited assistance from the primary-hub network. Experienced K–12 students then become peer mentors for other teams. Secondary hubs allow technical support to extend beyond universities and farther into rural regions. More importantly, secondary hubs can help nearby teams mature into additional secondary hubs. Consequently, ARC creates a self-reinforcing education loop: primary hubs train undergraduates, undergraduates teach K–12 students at workshops, K–12 teams become self-sustaining, and mature teams can then host their own workshops. Because each new secondary hub can help create further hubs, ARC has the potential for multiplicative and super-linear growth in mentorship capacity and geographic reach.

ARC addresses the mentorship gap by allowing technical support to spread geographically rather than remain concentrated around universities and urban zones. Primary hubs support nearby schools, while mature K–12 programs can become secondary hubs capable of supporting schools farther into rural regions. Schools therefore gain access to technical knowledge while also developing the capacity to eventually provide that knowledge to others.

We evaluate the growth potential of ARC using a spatiallyexplicit Markov model that represents both school roboticsprogram development and each school’s relationship to the ARC network. In an Indiana simulation containing 1,925 public K–12 schools, a moderate set of ARC growth assumptions reaches approximately 74% of schools and produces 992 schools with robotics programs after 40 years, compared with only 161 programs under the ARC-of natural-growth baseline. Under the most optimistic assumptions considered, ARC reaches approximately 95% of the modeled state and produces approximately 1,638 school programs, or 85% of Indiana public K–12 schools, with clear super-linear growth as secondary hubs multiply. Even under the least optimistic case considered, the model predicted 338 programs, more than twice the predicted number without our program.

We also conduct an initial deployment of ARC through one primary hub supporting three new FLL teams. In an undergraduate mentors survey measuring several axes of growth in a 5-point Likert score (strongly disagree to strongly agree), confidence teaching technical concepts increased from $3 . 0 0 \pm 1 . 1 5 \tan 4 . 2 9 \pm 0 . 4 9$ , confidence adapting explanations increased from $3 . 2 9 \pm 1 . 1 1$ to $4 . 4 3 \pm 0 . { \bar { 5 } } 3 .$ , and connection to the local community increased from 1.86 ± 1.07 to $4 . 0 0 \pm 0 . 8 2$ . Parents reported similarly large changes for participating students, including increases in basic robot programming knowledge from 2.00 ± 1.41 to 4.00 ± 1.41 and access to needed programming resources from $2 . 0 0 \pm 0 . 8 2$ to $4 . 2 5 \pm 0 . 5 0$ . They also rated their children’s enjoyment of working with the undergraduate mentors at $4 . 7 \bar { 5 } \pm 0 . 5 0$ and mentor support at $4 . 5 0 \pm 0 . 5 8$ out of five. Together, the simulation and trial deployment provide evidence that ARC can both produce efective local mentorship and, through repeated hub formation, potentially scale that mentorship across an entire region.

## The Rural Robotics Access Gap

Programs like FIRST provide K–12 students with opportunities to learn programming, robotics, AI, and related skills through team-based competitions. FIRST provides several program levels – FIRST LEGO League (FLL) introduces younger students to robotics and programming, FIRST Tech Challenge (FTC) provides a more advanced robotics competition for middle- and high-school students, and FIRST Robotics Competition (FRC) provides a larger-scale competition for high-school students. Together, these form a path for students to learn increasingly advanced skills.

However, access to these programs is geographically uneven. The upper left panel of Figure 2 shows this pattern for FLL participation in Indiana. Participation fell sharply during the 2020 remote season in both urban and rural areas. Since then, urban participation has gradually recovered, with both new and returning schools participating. Rural participation has not shown the same recovery. In the 2025–2026 season, rural FLL participation remained close to its post-2020 level and well below its 2017 peak. The lower left panels of Figure 2 show the corresponding geographic pattern, with much of the recovered participation concentrated in and around urban areas and major STEM campuses.

One important mechanism behind this gap is access to mentor resources. By mentor resources, we mean people with enough robotics, AI, and programming knowledge to help teams start, develop technical skills, and continue participating over time. Mentorship has already been identified as an important constraint on school robotics programs. Teacher knowledge can afect the adoption and continued use of educational robotics (Stokes et al. 2023), while robotics preparation may be particularly limited among rural educators (Meador et al. 2025). This creates a dificult problem for rural schools. Even when students and teachers are interested in robotics, they may have limited access to the technical knowledge needed to sustain a program.

## A Hubs-Based K–12 Education Framework

The AI, Robotics, & Community (ARC) framework is a hubs-based approach for expanding access to robotics and AI mentorship across a region. This is accomplished via local workshop events run by hub locations. Primary hubs are colleges or other STEM institutions that train undergraduate mentors and host workshops for nearby K–12 robotics teams. As supported teams gain experience and become selfsustaining, some can become secondary hubs that host workshops for other nearby schools. ARC therefore provides a way for technical mentorship to spread geographically rather than remain concentrated around universities. Because secondary hubs can help create additional secondary hubs, this structure also has the potential for super-linear growth in mentorship resources and geographic reach, allowing rural schools to be eficiently serviced.

## Educational Workshops

Workshops are the basic unit through which ARC provides mentorship to K–12 teams. They are hands-on and mentorled, with several teams attending at the same time and each team working directly with a small group of mentors. Game arenas are provided so that students can work on their own robots and the tasks of the current year’s competition. Rather than teaching programming or robotics in the abstract, mentors focus on the skills that students need to make progress in the game. For example, a mentor may focus on teaching the concepts needed to make a robot navigate to a ball, and push it stably towards a goal. Skills like motor control, use of sensors, planning, and logic may all follow from this. Students plan robot behavior, write and debug programs, test their solutions on the game arena, and revise them with direct help from their mentors. Workshop content is adapted to the age, experience, and competition level of each team. For FLL teams, this primarily involves foundational robotics and programming skills. FTC and FRC teams can require more advanced robotics, autonomy, computer vision, and AI depending on the needs of the game and the team.

![](images/6cdd826a1cd6ad40381c1466efed029a0116a441ca98dc5f5056b7a47f285d71.jpg)  
Figure 3: Simplified transition diagram for the spatiallyexplicit Markov model. The two axes of the K–12 school grid represent underlying state variables: program maturity ∈ {no program, early program, mature program}, and ARC level ∈ {unreached, reached, supported, secondary hub}. Each transition is a probability of changing states by year end. Colleges are modeled separately as hubs or non-hubs. Reachability depends on geographic distance to a hub – the blue hub can support the green school, but not the gray one. Ablations can be tested by disabling model transitions (setting their probability to 0). For example, isolating the unreachable column (hatched) simulates expected growth without ARC.

The workshop format makes mentorship and resource sharing more practical across geographically widespread regions. Instead of requiring experienced mentors to travel repeatedly to many individual schools, several teams can travel to a nearby hub and receive focused support simultaneously. A limited number of mentors can support more teams while still giving each team direct, hands-on assistance.

## The ARC Course at Primary Hubs

Primary hubs are colleges or other institutions that run the ARC course and use trained undergraduate students to provide workshops. The course gives the hub a regular source of mentors, as well as a specific time and place for workshops to occur. An experienced mentor trains the undergraduate students and oversees their work with the K–12 teams. The undergraduates then lead the workshops, working directly with individual teams to teach programming, robotics, debugging, and other skills needed for the competition. Mentor preparation therefore includes both technical knowledge and the interpersonal and educational skills needed to communicate technical ideas to students at diferent K–12 levels. The course begins with focused preparation in robotics and programming, robotics education, youth protection, and methods for teaching programming. After this preparation, undergraduates lead supervised K–12 workshops, with short continuing seminars connecting their mentoring to AI, robotics, and community applications. Full course details and reproducibility recommendations are provided in Appendix H.

## Mature Schools as Secondary Hubs

Secondary hubs extend the same workshop model beyond the primary hubs. A secondary hub is a mature K–12 school with an active, self-sustaining robotics program and enough internal technical knowledge to help nearby teams. This builds on the existing FIRST culture of experienced teams helping less experienced teams. At a secondary-hub workshop, experienced K–12 students become the main peer mentors for participating teams. A teacher or a small number of experienced undergraduate mentors can provide oversight and additional technical support when needed.

Secondary hubs are especially important for extending ARC into rural areas. A school may be too far from a university for repeated participation at a primary hub, but close enough to another K–12 school to attend workshops there. As teams around a secondary hub receive mentorship and mature, some may eventually become secondary hubs themselves. Those new hubs can then support another set ofnearby schools in an expanded region of reachability. ARC can then expand through repeated local growth as well as the creation of additional university programs. The ability of secondary hubs to create more secondary hubs is the mechanism that gives the framework its potential for super-linear growth.

## Spatially-Explicit Markov Model of Growth

To examine how ARC could grow across a region, we model its development as a spatially-explicit Markov process with annual transitions. We consider a mixed-rural region containing n K–12 schools and m colleges or other institutions that could operate as primary hubs. The goal of ARC is to expand access to robotics and programming knowledge across the K–12 schools in this region. The model represents both the natural development of school robotics programs and the additional efects of ARC support and hub growth.

Each potential college primary hub p has a state $C _ { p } ( t ) \in$ {0, 1} indicating whether it is inactive or currently operating as an ARC primary hub. At each annual transition, an inactive college may become a primary hub, while an active primary hub may cease operating. Active primary hubs extend ARC’s geographic reach in large steps and contribute capacity for launching and supporting school programs. The model does not attempt to represent the internal process by which a college decides to begin or end participation. These changes are instead represented by transition probabilities that may vary across colleges and time.

K–12 schools have two state variables, shown in Figure 3. The FIRST program state $F _ { i } ( t ) \in \{ N , E , M \}$ describes whether school i has no program, an Early program, or a Mature program. The ARC state $L _ { i } ( t ) \in \{ U , R , S , H \}$ describes whether the school is Unreachable by the current hub network, Reachable, directly Supported, or itself a secondary Hub. The school program is the atomic unit of the model rather than an individual team. Not every combination of these variables is possible. A Supported school must have an active program, and a secondary Hub must have a Mature program, leaving nine legal school states. Annual transitions capture the ways in which both school programs and the ARC network can change. Programs can naturally start, mature, close, or backslide from Mature to Early. ARC can help launch a program at a reachable school, directly support an existing program, and increase the opportunity for a Mature school to become a secondary hub. Hubs themselves can also stop operating. Geography enters through a fixed relation specifying which schools each potential primary or secondary hub can serve. The active hub network therefore determines which schools are reachable, and new secondary hubs can expand that reach. Reachability alone does not alter a school’s natural program-development probabilities. Direct ARC support is what allows diferent maturation and persistence behavior. The complete transition model is given in Appendix A. A mean-field approximation that replaces exact geography and individual institutions with regional averages is provided in Appendix B – this is more analytically tractable but loses spatial clustering, overlapping coverage, and local capacity constraints.

![](images/9a7c367f170a809751d2b275e12fdded1843566cd5deed37f7b956e78e9b7863.jpg)

![](images/8c137dcd1ced42bc4038557a6a292111e9b2dfe21676fd1e5baa8fe750ee4559.jpg)

![](images/214a9c572acccbfcec9cf97fb0f1169564c61b89fd19826f31f7dd1ab41b2370.jpg)

![](images/8d7bbd7748489226ec86a19af61cc5736b2f9bfb08a2eef8645abb57f7305ec2.jpg)

![](images/18924c733a062dad8f1adb4b40578877357796e555b54a77cb1d6bf6cc6dbdfd.jpg)

![](images/9c633edbb1358184927e80c9a583e852e726c924c1f478951ca4f169cdd97b97.jpg)  
Figure 4: Simulation results for 12 optimism level ARC growth scenarios. Depending on these parameter sets, the model shows that superlinear program growth is possible. Top: the number of schools reached (left), programs existing (middle), and secondary hubs existing (right) over 40 year simulations. Shaded regions show 10th–90th percentile ranges over 1000 runs. The more optimistic scenarios show pronounced self-reinforcing takeofs in both secondary hubs and school programs. The less optimistic ones do not, but still produce much more rapid growth than projections without ARC. Bottom: model ablation at scenarios 1, 6, and 12. Primary hubs alone increase growth but remain baseline-shaped. Secondary hubs show accelerating growth, though slowly from a single starting hub. Full ARC uses both, enabling parallel expansion and the fastest growth.

For our analysis, we instantiate the full spatial model for Indiana, with 1,925 public K–12 schools and 32 colleges with computer science programs treated as potential primary hubs. Initial program states are based on observed FIRST participation, and ARC-specific quantities not yet directly measurable are varied across 12 optimism levels with monotonically more favorable assumptions. Each level is simulated for 40 years over 1,000 Monte Carlo runs. We also form three ablations by setting the incoming transition probabilities for selected parts of ARC to zero. ARC of disables ARC growth entirely, leaving only natural program dynamics. Primary hubs only disables secondary-hub formation while allowing additional primary hubs. Initial primary + secondary hubs starts from the initial primary hub and allows secondary hubs to form, but prevents additional primary hubs from activating. These isolate the contributions of the two hub-growth mechanisms from Full ARC. Complete parameterization and extended results are provided in Appendices C and E.

The full spatial simulations show substantial growth across the modeled range (Figure 4). After 40 years, ARC of produces a mean of 161 programs, while even optimism level 1 of Full ARC produces 338, more than twice as many. At level 6, the model produces 992 programs and reaches 1,415 schools. At level 12, 1,638 schools have programs and 1,829 are within ARC reach, approximately 85% and 95% of Indiana public K–12 schools, respectively. Thus the strongest modeled conditions produce more than ten times as many programs as natural growth alone. Level 6 reaches 61% of rural schools and yields 341 rural programs, compared with 60 under ARC of, and 118 with primary hubs alone.

The trajectory shapes support ARC’s proposed selfreinforcing growth mechanism. Lower optimism levels increase program growth without a strong takeof, while higher levels show accelerating growth as secondary hubs multiply. This is clearest in the secondary-hub and program trajectories in Figure 4, where the more favorable scenarios develop pronounced superlinear growth before eventually slowing as the finite school population begins to saturate.

The ablations show that this growth is not produced by either hub mechanism alone. Primary hubs only expands access across multiple regions but remains comparatively baseline-shaped, while Initial primary + secondary hubs can generate accelerating growth but must propagate outward from a single starting region. At optimism level 6, these models produce 323 and 551 programs after 40 years, respectively, compared with 992 under Full ARC. Primary hubs therefore provide geographically distributed starting points while secondary hubs recursively extend them, with the combined ARC design producing substantially stronger growth than either mechanism in isolation.

We additionally tested the ARC-specific assumptions using one-at-a-time sweeps and Morris sensitivity screening, with full methods and results in Appendix D. Both tests show that early program growth is driven primarily by primary-hub activation and secondary-hub formation from existing Mature programs, while later growth shifts toward Supported schools becoming new secondary hubs, together with direct support, reach, and capacity. At year 40, varying Supported secondary-hub formation, support capacity, and reach radius across their optimism-level ranges changed projected program counts by approximately 430, 338, and 302 schools, with Morris likewise ranking Supported secondary-hub formation as the strongest long-term driver.

## Related Work

University–K–12 service learning and near-peer mentorship are well-established approaches in computing and engineering education. Credit-bearing programs have taught undergraduates to provide sustained community service and mentor robotics teams (Coyle, Jamieson, and Oakes 2005; Yamamoto, Barker, and Voida 2023; Kafai et al. 2013; Karp et al. 2014; Bhounsule et al. 2017; Matar et al. 2024). Rural STEM ecosystem research extends this partnership model through university coordination, co-design, local leadership, and community ownership (National Academies of Sciences, Engineering, and Medicine 2025a; Kavanagh et al. 2022; Timko et al. 2023; Bhaduri et al. 2022; Froehlich and Raberger 2026). These fields motivate ARC’s credit-bearing mentor course and development of mentoring capacity within participating schools.

ARC focuses on the rural mentorship gap created when schools may lie beyond practical travel distance of a university. Primary hubs support nearby teams, while mature K–12 programs can become secondary hubs that serve additional schools. Through this, mentorship can propagate beyond the fixed catchments of colleges through a self-reinforcing loop. This transfer of ownership is consistent with broader accounts of educational scale (Coburn 2003; Morel et al. 2019; National Academies of Sciences, Engineering, and Medicine 2025b), while research on educational robotics, FIRST LEGO League, and AI literacy supports the workshop setting and instructional content ARC provides (Anwar et al. 2019; Grafin, Shefield, and Koul 2022; Long and Magerko 2020; Touretzky et al. 2019).

Undergrad Mentors Survey  
![](images/875abd2f2c4a3a1071922da08e489653fde313f8fb0d7894e34bfc890d068a53.jpg)  
Figure 5: Mean retrospective pre-then-post survey responses from undergraduate mentors (n = 7) and parents of participating FLL students (n = 4). Responses used a five-point Likert scale from Strongly Disagree to Strongly Agree. One undergraduate did not provide a pre-course leadership response, giving n = 6 for the paired leadership measure.

## Trial Deployment

This trial tested the fundamental components of ARC: a primary hub built around the ARC course and a recurring K–12 workshop. We did not attempt to test the full regional system, which would require several years for supported programs to mature and new hubs to form. Instead, the trial focused on whether undergraduate students could be quickly prepared as useful mentors, and whether a new robotics program could regularly attend and receive support through a primary hub. Establishing these components provides a foundation for larger deployments.

The trial was conducted at one large Indiana university located in a small urban area surrounded by substantial rural space. Three FLL teams from one school in a rural part of the surrounding county participated. This was the school’s first year with an FLL program, and their teachers had previously been uncertain about starting the program because of limited access to mentorship. The course instructor had 12 years of prior experience mentoring competitive high-school robotics teams, as well as prior university teaching experience.

Nine undergraduate students enrolled in the ARC course. The course met once each week for a two-hour block. The first four weeks focused on preparing the undergraduates as mentors. Afterward, approximately 90 minutes were used for a workshop with the three teams, while the remaining 30 minutes were used for continuing course seminars. The teams traveled to the university for the workshops, where the undergraduate mentors worked with them using LEGO robots and two provided FLL game arenas.

We evaluated the trial using anonymous end-of-program surveys with pre-then-post questions – respondents were asked after the program to rate their perceptions both before and after participation. This design can reduce response-shift efects when participation changes respondents’ understanding of the construct being rated, since both ratings use the same post-program frame of reference. However, retrospective ratings can also be afected by recall and current-state biases, and should therefore be interpreted as retrospective selfassessments rather than direct longitudinal measurements.

Questions used a five-point Likert scale from Strongly Disagree to Strongly Agree, along with several post-program and open-response questions. To keep the evaluation limited to adult participants, K–12 outcomes were assessed through parent reports rather than direct surveys of children. Seven of the nine undergraduates and four parents completed the surveys. Survey details are available in Appendix F.

The undergraduate survey measured outcomes related to mentoring, community connection, leadership, and interest in AI, robotics, and graduate study. Figure 5 shows the prethen-post results. Confidence teaching technical concepts increased from 3.00 ± 1.15 to $4 . 2 9 \overset { \cdot } { \pm } 0 . 4 9$ , while confidence adapting explanations increased from $3 . 2 9 \pm 1 . 1 1$ to $4 . 4 3 \pm 0 . 5 3$ . Confidence managing small groups increased from $3 . 0 0 \pm 1 . 1 5$ to $4 . 0 0 \pm 0 . { \bar { 5 } } 8$ . The largest change was in connection to the local community, which increased from $1 . 8 6 \pm 1 . 0 7$ to $4 . 0 0 \pm 0 . 8 2$ . Finding teaching and mentoring enjoyable and meaningful increased from $3 . 4 3 \pm 0 . 7 9$ to $4 . { \bar { 5 } } 7 \pm { \bar { 0 } } . 5 3$ . Interest in AI, robotics, and graduate study also increased, although the changes were smaller.

Parents reported similar positive changes in several outcomes for their children. Ratings of basic robot programming knowledge increased from 2.00 ± 1.41 to 4.00 ± 1.41, while access to needed programming resources increased from 2.00±0.82 to 4.25±0.50. Opportunities to practice programming and problem solving increased from $\bar { 3 } . 0 0 \pm \bar { 0 } . 8 2$ to $4 . 2 5 \pm 0 . 9 6$ , and enjoyment of robotics club increased from 3.25±0.50 to 4.25±1.50. Changes in plans to continue robotics and interest in further STEM education were smaller. Parents also rated their children’s experiences with the undergraduate mentors positively, with ratings of $4 . 7 5 \pm 0 . 5 0$ for enjoying work with the mentors, $4 . 5 0 \pm 0 . 5 8$ for feeling supported and encouraged, and 4.50 ± 0.58 for helping the team prepare for competition.

The three teams also competed at a 29-team regional FLL qualifier. During the competition, two reached the upper half of the field, including one that briefly reached the top quartile. Although their final standings were lower – two in the third quartile and one near the top of the fourth – this represents a strong first-season showing for three newly formed teams competing in their first tournament.

## Discussion & Lessons Learned

Evidence from the trial. The trial shows that undergraduates can be prepared within a short course to take on direct mentoring roles, and that new robotics programs can receive regular technical support through a primary hub. The undergraduate results also suggest benefits to mentor growth, particularly in teaching confidence and connection to local community. Parent responses indicate that the workshops provided useful programming resources and support to the participating teams. Given the small population, these results should be interpreted as initial evidence of feasibility and perceived benefit rather than precise estimates of program efects or a causal evaluation.

Lessons for mentor preparation. The trial identified two areas for improving mentor preparation: providing greater structure for novice mentors and better preparing them to manage participation within K–12 teams. Mentoring sessions were intentionally flexible because teams difered in pace, technical needs, and group dynamics, but five of six substantive improvement responses requested more structure, like clearer pacing, checkpoints, or milestones (Appendix G). Future deployments should pair flexible mentoring with clear milestone checklists and fallback activities, consistent with challenges reported in related servicelearning programs (Bhounsule et al. 2017; Froehlich and Raberger 2026). Mentors also sometimes struggled to maintain balanced participation when individual students dominated programming or robot work despite turn-taking and role rotation. Future preparation should include practical strategies for enforcing these structures respectfully, redirecting students, and coordinating with team coaches.

Implications for scaling ARC. The growth simulations suggest that ARC could scale substantially, with stronger assumptions producing progressively broader reach and more pronounced self-reinforcing growth, sometimes becoming superlinear. Structural assumptions make the model stronger evidence for growth shape than for exact long-horizon school counts. The ablations further show that the two hub mechanisms play complementary roles: primary hubs seed growth across broad regions, while secondary hubs allow mentorship to propagate recursively beyond them. Developing mature supported schools into durable secondary hubs should be an explicit objective of future ARC evaluations.

Future work. Future multi-site, multi-year deployments will expand the number of participating schools and primary hubs, follow supported programs as they mature, and observe the formation and persistence of secondary hubs. These observations will provide empirical estimates for the model’s transition parameters, while working in real communities to close the rural robotics access gap nationwide.

## References

Alvarez, L.; Gransbury, I.; Cateté, V.; Barnes, T.; Ledéczi, Á.; and Grover, S. 2022. A Socially Relevant Focused AI Curriculum Designed for Female High School Students. Proceedings ofAAAI Conference on AI, 36(11): 12698–12705.

Anwar, S.; Bascou, N. A.; Menekse, M.; and Kardgar, A. 2019. A Systematic Review of Studies on Educational Robotics. Journal of Pre-College Engineering Education Research, 9(2): 19–42.

Bhaduri, S.; Biddy, Q.; Elliott, C. H.; Jacobs, J. K.; Rummel, M.; Ristvey, J.; Sumner, T.; and Recker, M. 2022. Codesigning a Rural Research–Practice Partnership to Design and Support STEM Pathways for Rural Youth. Theory & Practice in Rural Education, 12(2): 45–70.

Bhounsule, P.; Chaney, D.; Claeys, L.; and Manteufel, R. D. 2017. Robotics Service Learning for Improving Learning Outcomes and Increasing Community Engagement. In Proceedings of the 2017 ASEE Gulf-Southwest Section Annual Conference. Dallas, Texas: American Society for Engineering Education.

Campolongo, F.; Cariboni, J.; and Saltelli, A. 2007. An Efective Screening Design for Sensitivity Analysis of Large Models. Environmental Modelling & Software, 22(10): 1509– 1518.

Coburn, C. E. 2003. Rethinking Scale: Moving Beyond Numbers to Deep and Lasting Change. Educational Researcher, 32(6): 3–12.

Coyle, E. J.; Jamieson, L. H.; and Oakes, W. C. 2005. EPICS: Engineering Projects in Community Service. International Journal ofEngineering Education, 21(1): 139–150.

Froehlich, D. E.; and Raberger, J. 2026. Sustainabilityby-Design in School–University Partnerships: The Case of the Teaching Clinic. School-University Partnerships, 19(2): 273–287.

Grafin, M.; Shefield, R.; and Koul, R. 2022. More than Robots: Reviewing the Impact of the FIRST LEGO League Challenge Robotics Competition on School Students’ STEM Attitudes, Learning, and Twenty-First Century Skill Development. Journalfor STEM Education Research, 5(3): 322– 343.

Kafai, Y. B.; Grifin, J.; Burke, Q.; Slattery, M.; Fields, D. A.; Powell, R. M.; Grab, M.; Davidson, S. B.; and Sun, J. S. 2013. A Cascading Mentoring Pedagogy in a CS Service Learning Course to Broaden Participation and Perceptions. In Proceedings of the 44th ACM Technical Symposium on Computer Science Education, 101–106. Association for Computing Machinery.

Karp, T.; Gale, R.; Tan, M.; and Burnham, G. 2014. Hosting a Pipeline ofK–12 Robotics Competitions at a College ofEngineering: A Review of Benefits and Challenges. International Journal for Service Learning in Engineering, Humanitarian Engineering and Social Entrepreneurship, 406–423.

Kavanagh, K.; DeWaters, J.; Rivera, S.; Richards, M. C.; Ramsdell, M.; and Galluzzo, B. 2022. A University– Community Partnership Model to Support Rural STEM Teaching and Student Engagement. Theory & Practice in Rural Education, 12(2): 229–248.

Krakowski, A.; Greenwald, E.; Hurt, T.; Nonnecke, B.; and Cannady, M. 2022. Authentic Integration of Ethics and AI through Sociotechnical, Problem-Based Learning. Proceedings ofAAAI Conference on AI, 36(11): 12774–12782.

Long, D.; and Magerko, B. 2020. What Is AI Literacy? Competencies and Design Considerations. In Proceedings of the 2020 CHI Conference on Human Factors in Computing Systems, 1–16. Association for Computing Machinery.

Matar, S.; Wimer, B.; Shuvo, M. I. R.; Mahmud, S.; Kim, J.-H.; Novak, E.; and Borgerding, L. 2024. Cascade Mentoring Experience to Engage High School Learners in AI and Robotics Through Project-Based Learning. In Intelligent Human Computer Interaction, volume 14531 of Lecture Notes in Computer Science, 279–294. Springer.

Meador, A.; Raygoza, A.; Irvin, M.; Starrett, A.; Quiroz, B.; and Cartif, B. 2025. A Cross-Case Analysis of Rural Robotics Teacher Leaders’ Identity Development. Frontiers in Education, 10: 1578584.

Morel, R. P.; Coburn, C. E.; Catterson, A. K.; and Higgs, J. 2019. The Multiple Meanings of Scale: Implications for Researchers and Practitioners. Educational Researcher, 48(6): 369–377.

Morris, M. D. 1991. Factorial Sampling Plans for Preliminary Computational Experiments. Technometrics, 33(2): 161–174.

National Academies of Sciences, Engineering, and Medicine. 2025a. K–12 STEM Education and Workforce Development in Rural Areas. Washington, DC: The National Academies Press.

National Academies of Sciences, Engineering, and Medicine. 2025b. Scaling and Sustaining Pre-K–12 STEM Education Innovations: Systemic Challenges, Systemic Responses. Washington, DC: The National Academies Press.

Stokes, A.; Aurini, J.; Rizk, J.; Gorbet, R.; and McLevey, J. 2023. Using Robotics to Support the Acquisition of STEM and 21st-Century Competencies: Promising (and Practical) Directions. Canadian Journal ofEducation /Revue canadienne de l’éducation, 45(4): 1141–1170.

Timko, G.; Harris, M.; Hayde, D.; and Peterman, K. 2023. Sustainable Development of Community-Supported STEM-Learning Ecosystems in Rural Areas of the United States. Community Development Journal, 58(3): 492–511.

Touretzky, D. S.; Gardner-McCune, C.; Breazeal, C.; Martin, F.; and Seehorn, D. 2019. A Year in K–12 AI Education. AI Magazine, 40(4): 88–90.

Williams, R.; Kaputsos, S. P.; and Breazeal, C. 2021. Teacher Perspectives on How To Train Your Robot: A Middle School AI and Ethics Curriculum. Proceedings ofAAAI Conference on AI, 35(17): 15678–15686.

Yamamoto, F. R.; Barker, L.; and Voida, A. 2023. CISing Up Service Learning: A Systematic Review of Service Learning Experiences in Computer and Information Science. ACM Transactions on Computing Education, 23(3).

## A. Spatially-Explicit Markov Model of Growth Details

## Model Overview & State Space

The ARC growth model is a discrete-time stochastic model of individual K–12 school programs and potential college hubs. One time step represents one year. The state at year t describes the system during that year, and one transition produces the complete state for year $t + 1$ . Geography is represented explicitly, so whether ARC can reach a school depends on the locations of the active hubs.

The unit of the school model is a school program. A school has an active program if at least one relevant FIRST team is registered to it. If a school has multiple teams, its program state is based on its most developed current team. Team count is not itself part of the state.

For school $i = 1 , \ldots , n ,$ where n is the number of K–12 schools, let $F _ { i } ( t ) \in \{ N , E , M \}$ denote its FIRST program state, where N is No program, E is Early, and M is Mature. Let $L _ { i } ( t ) \in \{ U , R , S , { \dot { H } } \}$ denote its ARC state, where U is Unreachable, R is Reachable but not directly supported, S is Supported, and H is a secondary Hub. The complete state of school i is $X _ { i } ( t ) = ( F _ { i } ( t ) , L _ { i } ( t ) )$

Only nine combinations are valid. Table 1 defines these school states.

<table><tr><td>State</td><td>Program</td><td>ARC status</td></tr><tr><td>NU</td><td>No program</td><td>Unreachable</td></tr><tr><td>NR</td><td>No program</td><td>Reachable</td></tr><tr><td>EU</td><td>Early</td><td>Unreachable</td></tr><tr><td>ER</td><td>Early</td><td>Reachable</td></tr><tr><td>ES</td><td>Early</td><td>Supported</td></tr><tr><td>MU</td><td>Mature</td><td>Unreachable</td></tr><tr><td>MR</td><td>Mature</td><td>Reachable</td></tr><tr><td>MS</td><td>Mature</td><td>Supported</td></tr><tr><td>MH</td><td>Mature</td><td>Secondary hub</td></tr></table>

Table 1: Valid K–12 school states in the full spatial model.

There are no NS, NH, or EH states. Supported schools must have an active program, and secondary hubs must have a Mature program. ARC assistance before program creation is represented by the NR state and the ARC launch probability defined below.

For potential primary-hub college $p = 1 , \ldots , m$ , where m is the number of potential primary-hub colleges, $C _ { p } ( t ) \in \{ 0 , 1 \}$ denotes whether the college is operating as an ARC primary hub. In particular, $C _ { p } ( t ) = 1$ means college p is operating as a primary hub. The complete regional state is $Y ( t ) = ( \overbar { X } _ { 1 } ( t ) , \cdot \cdot \cdot , X _ { n } ( t ) , C _ { 1 } ( t ) , \cdot \cdot \cdot , \overbar { C } _ { m } ( t ) )$

Several transition rules are fixed by the model. A school with no program can start only as Early, so there is no direct $N  M$ transition. If a program closes, any later restart begins again as Early. Reachability alone does not change program maturation, program-closure, or program-backslide probabilities, so EU and ER use the same program dynamics, as do MU and MR. Direct support may change these dynamics. A school can become a secondary hub only if it begins the year with a Mature program and remains Mature through the transition. Thus an Early program may mature in one step, but it cannot mature and become a hub in the same step. Programs may also undergo program closure or backslide from Mature to Early, Supported schools may lose support or reachability, and primary and secondary hubs may stop operating.

## Parameters and Calculated Quantities

Table 2 defines the parameters and inputs used by the full model. A superscript B denotes unsupported program dynamics, S denotes directly Supported program dynamics, and H denotes secondary-hub program dynamics.

The parameters in Table 2 are combined into a small number of calculated quantities used in the transition probabilities. These calculations follow the model’s annual update: first program-state changes, then the hub network and geographic reach, then direct ARC support.

For program-state changes, the probability that a program remains at its current development level is whatever probabilit remains after its possible forward or backward transitions:

$$
\begin{array} { r } { r _ { E } ^ { B } = 1 - \delta _ { E } ^ { B } - \mu _ { E } ^ { B } , \qquad r _ { E } ^ { S } = 1 - \delta _ { E } ^ { S } - \mu _ { E } ^ { S } , \qquad r _ { M } ^ { B } = 1 - \delta _ { M } ^ { B } - \nu _ { M } ^ { B } , \qquad r _ { M } ^ { S } = 1 - \delta _ { M } ^ { S } - \nu _ { M } ^ { S } , \qquad r _ { M } ^ { H } = 1 - \delta _ { M } ^ { H } - \nu _ { M } ^ { H } . } \end{array}
$$

Essentially, each r is the probability that the program stays at the same development level for another year. For an Early program, this means it neither undergoes program closure nor matures. For a Mature program, it means it neither undergoes program closure nor backslides to Early. For a secondary hub, $r _ { M } ^ { H }$ refers to the probability that its program remains Mature. Whether the school also remains a secondary hub is handled separately by $\rho _ { i } ( t )$

A Reachable school with no program can start either naturally or through ARC. ARC attempts an assisted program launch only when a natural start does not occur, so the total probability that an NR school enters an Early program is

$$
s _ { i } ( t ) = q _ { i } ^ { B } ( t ) + \left( 1 - q _ { i } ^ { B } ( t ) \right) \ell _ { i } ( t ) .
$$

<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td> $q _ { i } ^ { B } ( t )$ </td><td>Natural  $N  E$  start or restart probability for school i</td></tr><tr><td> $\ell _ { i } ( t )$ </td><td>Probability that ARC launches an Early program at an NR school after no natural start</td></tr><tr><td> $\mu _ { E } ^ { B }$ </td><td>Early → Mature probability for an unsupported program</td></tr><tr><td> $\delta _ { E } ^ { B }$   $^ { \upsilon } _ { _ { S } }$ </td><td>Program-closure probability for an unsupported Early program</td></tr><tr><td> $\mu _ { E } ^ { \backprime }$ </td><td>Early → Mature probability for a Supported school</td></tr><tr><td> $\delta _ { E } ^ { S }$ </td><td>Program-closure probability for a Supported Early program</td></tr><tr><td> $\nu _ { M } ^ { B }$ </td><td>Mature → Early program-backslide probability for an unsupported program</td></tr><tr><td> $\delta _ { M } ^ { B }$ </td><td>Program-closure probability for an unsupported Mature program</td></tr><tr><td> $\nu _ { M } ^ { s }$ </td><td>Mature → Early program-backslide probability for a Supported school</td></tr><tr><td> $\delta _ { M } ^ { S }$ </td><td>Program-closure probability for a Supported Mature program</td></tr><tr><td> $\nu _ { M } ^ { H }$ </td><td>Mature → Early program-backslide probability for a secondary hub</td></tr><tr><td> $\delta _ { M } ^ { H }$ </td><td>Program-closure probability for a secondary hub</td></tr><tr><td> ${ \chi _ { i } ^ { B } } ( t )$ </td><td>Probability that an unsupported school with a Mature program becomes a secondary hub</td></tr><tr><td> $\chi _ { i } ^ { S } ( t )$ </td><td>Probability that a Supported school with a Mature program becomes a secondary hub</td></tr><tr><td> $\rho _ { i } ( t )$ </td><td>Probability that an existing secondary hub remains a secondary hub if its program remains Mature</td></tr><tr><td> $\kappa _ { p } ( t )$ </td><td>Probability that inactive potential primary hub  $p$  begins operating</td></tr><tr><td> $\xi _ { p } ( t )$ </td><td>Probability that active primary hub p stops operating</td></tr><tr><td> $b _ { h i }$ </td><td>Binary indicator that primary or secondary hub  $h$  can geographically serve school i</td></tr><tr><td> $w _ { i } ^ { n e w } ( t + 1 )$ </td><td>Probability that an active, reachable, non-hub school receives new direct ARC support</td></tr><tr><td> $w _ { i } ^ { r e t } ( t + 1 )$ </td><td>Probability that an active, reachable, non-hub school retains existing direct ARC support</td></tr></table>

Table 2: Parameters and structural inputs of the full spatial model.

Its probability of remaining without a program is $1 - s _ { i } ( t ) = ( 1 - q _ { i } ^ { B } ( t ) ) ( 1 - \ell _ { i } ( t ) )$

The next part of the update determines the hub network and, from it, which schools are geographically reachable. For compact notation, let $J _ { i } ( t ) = \mathbf { 1 } \bar { \{ X _ { i } ( t ) = \mathrm { M H } \} }$ , so $J _ { i } ( t ) = 1$ when school i is a secondary hub and 0 otherwise. The fixed quantity $b _ { h i }$ equals one when hub h can geographically serve school i and zero otherwise, where $h$ may refer to either a primary or secondary hub. In the reachability calculation, $p$ indexes potential primary-hub colleges and $j$ indexes K–12 schools. Here, the + superscript indicates reachability for year $t + 1$ after the next-year hub states have been determined:

$$
g _ { i } ^ { + } = 1 - \prod _ { p = 1 } ^ { m } \left( 1 - b _ { p i } C _ { p } ( t + 1 ) \right) \prod _ { \stackrel { j = 1 } { j \neq i } } ^ { n } ( 1 - b _ { j i } J _ { j } ( t + 1 ) ) .
$$

Thus $g _ { i } ^ { + } = 1$ if at least one active primary hub or another active secondary hub can serve school $i ,$ and $g _ { i } ^ { + } = 0$ otherwise. The exclusion $j \neq i$ prevents a school from making itself reachable through its own hub status.

Finally, an active non-hub school that is reachable may receive direct ARC support. Its final ARC status is divided between Unreachable, Reachable but unsupported, and Supported according to

$$
1 - g _ { i } ^ { + } , \qquad g _ { i } ^ { + } ( 1 - w _ { i } ) , \qquad g _ { i } ^ { + } w _ { i } ,
$$

where $w _ { i } = w _ { i } ^ { r e t } ( t + 1 )$ if the school was already in a Supported state and $w _ { i } = w _ { i } ^ { n e w } ( t + 1 )$ otherwise. A school with no program cannot be Supported, so a no-program destination is divided only between NU and NR.

The model requires $\chi _ { i } ^ { S } ( t ) \geq \chi _ { i } ^ { B } ( t )$ , so a Supported school with a Mature program is at least as likely to become a secondary hub as an otherwise comparable unsupported school with a Mature program. It also requires $w _ { i } ^ { r e t } ( t + 1 ) \geq w _ { i } ^ { n e w } ( t + 1 )$ , so retaining an existing direct-support relationship is at least as likely as creating a new one. All input probabilities lie in $[ 0 , 1 ] ,$ with

$$
\delta _ { E } ^ { B } + \mu _ { E } ^ { B } \leq 1 , \qquad \delta _ { E } ^ { S } + \mu _ { E } ^ { S } \leq 1 , \qquad \delta _ { M } ^ { B } + \nu _ { M } ^ { B } \leq 1 , \qquad \delta _ { M } ^ { S } + \nu _ { M } ^ { S } \leq 1 , \qquad \delta _ { M } ^ { H } + \nu _ { M } ^ { H } \leq 1 .
$$

These constraints ensure that the corresponding r probabilities are nonnegative.

The quantities $\ell _ { i } ( t ) , w _ { i } ^ { n e w } ( t + 1 )$ , and $w _ { i } ^ { r e t } ( t + 1 )$ describe the chances that ARC launches a new program, begins direct support, or continues direct support at a school. A particular implementation may specify these probabilities directly or determine them from explicit limits on how many schools can be launched or supported. If only a fixed number of launch or support spaces are available, the implementation must also specify how the schools receiving those spaces are selected.

## Annual Transition Process

Each $t \to t + 1$ update produces one new state for every school and college. The calculation proceeds in five steps.

1. Each school receives a program outcome: No program, Early, or Mature. The probabilities depend on its state at year t. An NR school additionally has the ARC launch opportunity $\ell _ { i } ( \dot { t } )$

<table><tr><td>NU</td></tr><tr><td>To Probability</td></tr><tr><td>NU  $( 1 - q _ { i } ^ { B } ) ( 1 - g _ { i } ^ { + } )$ </td></tr><tr><td>NR  $( 1 - q _ { i } ^ { B } ) g _ { i } ^ { + }$ </td></tr><tr><td>EU  $q _ { i } ^ { B } ( 1 - g _ { i } ^ { + } )$ </td></tr><tr><td>ER</td></tr><tr><td> $q _ { i } ^ { B } g _ { i } ^ { + } ( 1 - w _ { i } ^ { n e w } )$  ES  $q _ { i } ^ { B } g _ { i } ^ { + } w _ { i } ^ { n e w }$ </td></tr></table>

<table><tr><td colspan="2">NR</td></tr><tr><td>To</td><td>Probability</td></tr><tr><td>NU</td><td> $( 1 - q _ { i } ^ { B } ) ( 1 - \ell _ { i } ) ( 1 - g _ { i } ^ { + } )$ </td></tr><tr><td>NR</td><td> $( 1 - q _ { i } ^ { B } ) ( 1 - \ell _ { i } ) g _ { i } ^ { + }$ </td></tr><tr><td>EU</td><td> $s _ { i } ( 1 - g _ { i } ^ { + } )$ </td></tr><tr><td>ER</td><td> $s _ { i } g _ { i } ^ { + } ( 1 - w _ { i } ^ { n e w } )$ </td></tr><tr><td>ES</td><td> $s _ { i } g _ { i } ^ { + } w _ { i } ^ { n e w }$ </td></tr></table>

<table><tr><td colspan="2">ER</td></tr><tr><td>To</td><td>Probability</td></tr><tr><td>NU</td><td> $\delta _ { E } ^ { B } ( 1 - g _ { i } ^ { + } )$ </td></tr><tr><td>NR</td><td> $\delta _ { E } ^ { B } g _ { i } ^ { + }$ </td></tr><tr><td>EU</td><td> $r _ { E } ^ { B } ( \dot { 1 } - g _ { i } ^ { + } )$ </td></tr><tr><td>ER</td><td> $r _ { E } ^ { B } g _ { i } ^ { + } ( 1 - w _ { i } ^ { n e w } )$ </td></tr><tr><td>ES</td><td> $r _ { E } ^ { B } g _ { i } ^ { + } w _ { i } ^ { n e w }$ </td></tr><tr><td>MU</td><td> $\mu _ { E } ^ { B } ( 1 - g _ { i } ^ { + } )$ </td></tr><tr><td>MR</td><td> $\mu _ { E } ^ { B } g _ { i } ^ { + } ( 1 - w _ { i } ^ { n e w } )$ </td></tr><tr><td>MS</td><td> $\mu _ { E } ^ { B } g _ { i } ^ { + } w _ { i } ^ { n e w }$ </td></tr></table>

<table><tr><td colspan="2">ES</td></tr><tr><td>To</td><td>Probability</td></tr><tr><td>NU</td><td> $\delta _ { E } ^ { S } ( 1 - g _ { i } ^ { + } )$ </td></tr><tr><td>NR</td><td> $\delta _ { E } ^ { S } g _ { i } ^ { + }$ </td></tr><tr><td>EU</td><td> $r _ { E } ^ { S } ( 1 - g _ { i } ^ { + } )$ </td></tr><tr><td>ER</td><td> $r _ { E } ^ { S } g _ { i } ^ { + } ( 1 - w _ { i } ^ { r e t } )$ </td></tr><tr><td>ES</td><td> $r _ { E } ^ { S } g _ { i } ^ { + } w _ { i } ^ { r e t }$ </td></tr><tr><td>MU</td><td> $\mu _ { E } ^ { S } ( 1 - g _ { i } ^ { + } )$ </td></tr><tr><td>MR</td><td> $\mu _ { E } ^ { S } g _ { i } ^ { + } ( 1 - w _ { i } ^ { r e t } )$ </td></tr><tr><td>MS</td><td> $\mu _ { E } ^ { S } g _ { i } ^ { + } w _ { i } ^ { r e t }$ </td></tr></table>

$$
\delta _ { M } ^ { B } ( 1 - g _ { i } ^ { + } )
$$

$$
\delta _ { M } ^ { B } g _ { i } ^ { + }
$$

<table><tr><td colspan="2">EU</td></tr><tr><td>To</td><td>Probability</td></tr><tr><td>NU</td><td> $\delta _ { E } ^ { B } ( 1 - g _ { i } ^ { + } )$ </td></tr><tr><td>NR</td><td> $\delta _ { E } ^ { B } g _ { i } ^ { + }$ </td></tr><tr><td>EU</td><td> $r _ { E } ^ { B } ( 1 - g _ { i } ^ { + } )$ </td></tr><tr><td>ER</td><td> $r _ { E } ^ { B } g _ { i } ^ { + } ( 1 - w _ { i } ^ { n e w } )$ </td></tr><tr><td>ES</td><td> $r _ { E } ^ { B } g _ { i } ^ { + } w _ { i } ^ { n e w }$ </td></tr><tr><td>MU</td><td> $\mu _ { E } ^ { B } ( 1 - g _ { i } ^ { + } )$ </td></tr><tr><td>MR</td><td> $\mu _ { E } ^ { B } g _ { i } ^ { + } ( 1 - w _ { i } ^ { n e w } )$ </td></tr><tr><td>MS</td><td> $\mu _ { E } ^ { B } g _ { i } ^ { + } w _ { i } ^ { n e w }$ </td></tr></table>

$$
\nu _ { M } ^ { B } ( \dot { 1 } - g _ { i } ^ { + } )
$$

$$
\nu _ { M } ^ { B } g _ { i } ^ { + } ( 1 - w _ { i } ^ { n e w } )
$$

$$
\nu _ { M } ^ { B } g _ { i } ^ { + } w _ { i } ^ { n e w }
$$

$$
r _ { M } ^ { \widetilde { B } } ( 1 - \dot { \chi } _ { i } ^ { B } ) ( 1 - g _ { i } ^ { + } )
$$

$$
r _ { M } ^ { B } ( 1 - \chi _ { i } ^ { B } ) g _ { i } ^ { + } ( 1 - w _ { i } ^ { n e w } )
$$

$$
r _ { M } ^ { B } ( 1 - \chi _ { i } ^ { B } ) g _ { i } ^ { + } w _ { i } ^ { n e w }
$$

$$
r _ { M } ^ { B } \chi _ { i } ^ { B }
$$

$$
\delta _ { M } ^ { B } ( 1 - g _ { i } ^ { + } )
$$

$$
\delta _ { M } ^ { B } g _ { i } ^ { + }
$$

$$
\nu _ { M } ^ { B } ( \dot { 1 } - g _ { i } ^ { + } )
$$

$$
\nu _ { M } ^ { B } g _ { i } ^ { + } ( 1 - w _ { i } ^ { n e w } )
$$

$$
\nu _ { M } ^ { B } g _ { i } ^ { + } w _ { i } ^ { n e w }
$$

$$
r _ { M } ^ { \widetilde { B } } ( 1 - \dot { \chi } _ { i } ^ { B } ) ( 1 - g _ { i } ^ { + } )
$$

$$
r _ { M } ^ { B } ( 1 - \chi _ { i } ^ { B } ) g _ { i } ^ { + } ( 1 - w _ { i } ^ { n e w } )
$$

<table><tr><td colspan="2">MS</td></tr><tr><td>To</td><td>Probability</td></tr><tr><td>NU</td><td> $\delta _ { M } ^ { S } ( 1 - g _ { i } ^ { + } )$ </td></tr><tr><td>NR</td><td> $\delta _ { M } ^ { S } g _ { i } ^ { + }$ </td></tr><tr><td>EU</td><td> $\nu _ { M } ^ { S } ( \dot { 1 } - g _ { i } ^ { + } )$ </td></tr><tr><td>ER</td><td> $\nu _ { M } ^ { S } g _ { i } ^ { + } ( 1 - w _ { i } ^ { r e t } )$ </td></tr><tr><td>ES</td><td> $\nu _ { M } ^ { S } g _ { i } ^ { + } w _ { i } ^ { r e t }$ </td></tr><tr><td>MU</td><td> $r _ { M } ^ { S } ( \ r _ { 1 } - \chi _ { i } ^ { S } ) ( \ r _ { 1 } - g _ { i } ^ { + } )$ </td></tr><tr><td>MR</td><td> $r _ { M } ^ { S } ( 1 - \chi _ { i } ^ { S } ) g _ { i } ^ { + } ( 1 - w _ { i } ^ { r e t } )$ </td></tr><tr><td>MS</td><td> $r _ { M } ^ { S } ( 1 - \chi _ { i } ^ { S } ) g _ { i } ^ { + } w _ { i } ^ { r e t }$ </td></tr><tr><td>MH</td><td> $r _ { M } ^ { S } \chi _ { i } ^ { S }$ </td></tr></table>

Table 3: Complete school transition probabilities. Each small table is labeled by its source state and gives every legal destination from that state. Time arguments are omitted for readability.

$$
r _ { M } ^ { B } ( 1 - \chi _ { i } ^ { B } ) g _ { i } ^ { + } w _ { i } ^ { n e w }
$$

$$
\delta _ { M } ^ { H } ( 1 - g _ { i } ^ { + } )
$$

$$
\delta _ { M } ^ { H } g _ { i } ^ { + }
$$

$$
\nu _ { M } ^ { H } ( \dot { 1 } - g _ { i } ^ { + } )
$$

$$
\nu _ { M } ^ { H } g _ { i } ^ { + } ( 1 - w _ { i } ^ { n e w } )
$$

$$
\nu _ { M } ^ { H } g _ { i } ^ { + } w _ { i } ^ { n e w }
$$

$$
r _ { M } ^ { B } \chi _ { i } ^ { B }
$$

$$
r _ { M } ^ { H } ( \ r _ { 1 } - \rho _ { i } ) ( \ r _ { 1 } - g _ { i } ^ { + } )
$$

$$
r _ { M } ^ { H } ( 1 - \rho _ { i } ) g _ { i } ^ { + } ( 1 - w _ { i } ^ { n e w } )
$$

$$
r _ { M } ^ { H } ( 1 - \rho _ { i } ) g _ { i } ^ { + } w _ { i } ^ { n e w }
$$

$$
r _ { M } ^ { H } \rho _ { i }
$$

2. Potential primary hubs update using $\kappa _ { p } ( t )$ and $\xi _ { p } ( t )$ . Schools that began the year with a Mature program and remain Mature may become secondary hubs using $\chi _ { i } ^ { B } ( t )$ or $\chi _ { i } ^ { S } ( t )$ . Existing secondary hubs whose programs remain Mature remain secondary hubs with probability $\rho _ { i } ( t )$

3. The resulting primary and secondary-hub network is used to calculate $g _ { i } ^ { + }$ for every non-hub school.

4. Active reachable non-hub schools receive or retain direct ARC support according to $w _ { i } ^ { n e w } ( t + 1 )$ or $w _ { i } ^ { r e t } ( t + 1 )$

5. The program outcome and ARC outcome are combined into one of the nine valid states in Table 1.

These are dependencies within one annual transition. For example, a school that begins the year with an Early program cannot become a hub at that boundary even if its program becomes Mature.

For each transition, the source is the school’s state at year t and the destination is its state at year t + 1. A school transition probability contains a program factor, an optional hub factor, and a final reach/support factor. Table 3 gives all 70 legal school transitions. Any source–destination pair not listed has probability zero

For each starting state, the probabilities of all possible next states sum to one. For an Early program, the possible program outcomes are program closure, remaining Early, or maturation. For a Mature program, they are program closure, backsliding to Early, or remaining Mature. If an unsupported or Supported school with a Mature program remains Mature, its probability is further divided between becoming a secondary hub and remaining a non-hub school. If an existing secondary hub remains Mature, its probability is further divided between remaining a secondary hub and ceasing to operate as one. Each active non-hub branch is then divided between the three ARC outcomes because

$$
( 1 - g _ { i } ^ { + } ) + g _ { i } ^ { + } ( 1 - w _ { i } ) + g _ { i } ^ { + } w _ { i } = 1 .
$$

Potential primary hubs have four transitions:

$$
\begin{array} { r l r } { \operatorname* { P r } ( C _ { p } ( t + 1 ) = 1 \mid C _ { p } ( t ) = 0 ) = \kappa _ { p } ( t ) , } & { } & { \operatorname* { P r } ( C _ { p } ( t + 1 ) = 0 \mid C _ { p } ( t ) = 0 ) = 1 - \kappa _ { p } ( t ) , } \\ { \operatorname* { P r } ( C _ { p } ( t + 1 ) = 0 \mid C _ { p } ( t ) = 1 ) = \xi _ { p } ( t ) , } & { } & { \operatorname* { P r } ( C _ { p } ( t + 1 ) = 1 \mid C _ { p } ( t ) = 1 ) = 1 - \xi _ { p } ( t ) . } \end{array}
$$

By default, the random program, hub, and support events for diferent schools and colleges are drawn independently once the current system state and their required inputs are fixed. The schools are nevertheless linked through geography because realized hub outcomes determine the reachability of other schools. If a concrete implementation has a fixed number of launch or support spaces, it must also specify how the schools receiving those spaces are selected.

## Initialization and Model Outputs

A full spatial-model run requires the set of schools, an initial valid state $X _ { i } ( 0 )$ for each school, the set of potential primary-hub colleges, an initial value $C _ { p } ( 0 )$ for each college, the geographic relation $b _ { h i } .$ , the model parameters, and any rules used to select schools when launch or support capacity is limited. The model does not assume a universal initial distribution. These inputs are properties of the region being modeled.

For any school state s, let $\begin{array} { r } { \mathsf { \bar { Z } } _ { s } ( t ) = \sum _ { i = 1 } ^ { n } \mathbf { 1 } \{ X _ { i } ( t ) = s \} } \end{array}$ denote the number of schools in that state.

The total number of schools with active FIRST programs is

$$
P ( t ) = Z _ { \mathrm { E U } } ( t ) + Z _ { \mathrm { E R } } ( t ) + Z _ { \mathrm { E S } } ( t ) + Z _ { \mathrm { M U } } ( t ) + Z _ { \mathrm { M R } } ( t ) + Z _ { \mathrm { M S } } ( t ) + Z _ { \mathrm { M H } } ( t ) .
$$

The total number of schools within the ARC network, including Reachable, Supported, and secondary-hub schools, is

$$
A ( t ) = Z _ { \mathrm { N R } } ( t ) + Z _ { \mathrm { E R } } ( t ) + Z _ { \mathrm { E S } } ( t ) + Z _ { \mathrm { M R } } ( t ) + Z _ { \mathrm { M S } } ( t ) + Z _ { \mathrm { M H } } ( t ) .
$$

The number of directly Supported schools is $S ( t ) = Z _ { \mathrm { E S } } ( t ) + Z _ { \mathrm { M S } } ( t )$ , the number of secondary hubs is $H ( t ) = Z _ { \mathrm { M H } } ( t )$ and the number of active primary hubs is $\begin{array} { r } { C ( t ) = \dot { \sum _ { p = 1 } ^ { m } } C _ { p } ( \dot { t } ) . } \end{array}$

## B. Mean Field Approximation Details

The mean-field model is an approximation of the spatially-explicit model in Appendix A. It uses the same nine school states, the same two primary-hub college states, and the same allowed transitions. The diference is that individual schools and their exact geographic relationships are replaced by regional averages. Instead of sampling a state for every school, the model tracks the expected number of schools in each state.

Let $Z _ { s } ( t )$ denote the expected number of schools in school state s at year $t ,$ where $s \in$ {NU, NR, EU, ER, ES, MU, MR, MS, MH}. These expected counts satisfy $\begin{array} { r } { \sum _ { s } Z _ { s } ( t ) ~ = ~ n . } \end{array}$ . Because they are expectations, they may be fractional. For example, $\dot { Z } _ { \mathrm { M H } } ( t ) = 5 \dot { 5 } . 6$ means that the model predicts an average of 55.6 secondary hubs across comparable realizations, not that a physical system contains a fraction of a hub.

Similarly, let $C _ { 0 } ( t )$ and $C _ { 1 } ( t )$ denote the expected numbers of inactive and active potential primary hubs, with $\begin{array} { r l } { C _ { 0 } ( t ) + C _ { 1 } ( t ) = } \end{array}$ m.

## Mean-Field Replacements

Most parameters from the full model are unchanged. The program maturation, program-closure, and program-backslide probabilities $\mu , \delta , \nu .$ , and their calculated retention probabilities r retain the same definitions. The mean-field approximation replaces school-specific and geographically-specific quantities with the regional quantities shown in Table 4.

<table><tr><td>Full model</td><td>Mean field</td><td>Meaning</td></tr><tr><td> $q _ { i } ^ { B } ( t )$ </td><td> $q ^ { B }$ </td><td>Regional natural start probability</td></tr><tr><td> $\ell _ { i } ( t )$ </td><td> $\bar { \ell } _ { t }$ </td><td>Regional ARC launch probability</td></tr><tr><td> ${ \chi _ { i } ^ { B } } ( t )$ </td><td> $\nu ^ { B }$   $\chi ^ { B }$ </td><td>Unsupported secondary-hub formation probability</td></tr><tr><td> $\chi _ { i } ^ { S } ( t )$ </td><td> $^ { \prime \star } s$   $\chi ^ { \scriptscriptstyle \mathscr { s } }$ </td><td>Supported secondary-hub formation probability</td></tr><tr><td> $\rho _ { i } ( t )$ </td><td> $\rho$ </td><td>Secondary-hub retention probability</td></tr><tr><td> $\kappa _ { p } ( t )$ </td><td> $\kappa$ </td><td>Primary-hub activation probability</td></tr><tr><td> $\xi _ { p } ( t )$ </td><td> $\xi$ </td><td>Primary-hub stopping probability</td></tr><tr><td> $b _ { p i }$ </td><td> $c _ { C }$ </td><td>Average coverage from one primary hub</td></tr><tr><td> $b _ { j i }$ </td><td> $c _ { H }$ </td><td>Average coverage from one secondary hub</td></tr><tr><td> $g _ { i } ^ { + }$ </td><td> $\bar { g } _ { t + 1 }$ </td><td>Regional probability of being reachable</td></tr><tr><td> $w _ { i } ^ { n e w }$ </td><td> $\bar { w } ^ { n e w }$ </td><td>New direct-support probability</td></tr><tr><td> $w _ { i } ^ { r e t }$ </td><td> $\hat { w } ^ { r e t }$ </td><td>Direct-support retention probability</td></tr></table>

Table 4: Quantities replaced by regional averages in the mean-field approximation.

The mean-field model also uses $L ,$ the expected maximum number of successful new school-program launches that one active hub can produce in a year, and $K ,$ , the expected number of schools that one active hub can directly support.

## Mean-Field Annual Update

The annual update follows the same order as the full model. Program states change first, followed by hub formation, reachability, direct support, and the final school-state update.

A Reachable school with no program can still start naturally with probability $q ^ { B }$ . ARC can attempt to launch a program among the remaining NR schools. The expected number of such schools available for an ARC launch is $( 1 - q ^ { B } ) Z _ { \mathrm { N R } } ( \overline { { t } } )$ . The hubs active during year t can together produce up to $L [ C _ { 1 } ( t ) + Z _ { \mathrm { M H } } ( t ) ]$ expected launches. The resulting ARC launch probability is therefore

$$
\bar { \ell } _ { t } = \operatorname* { m i n } \left( 1 , \frac { L [ C _ { 1 } ( t ) + Z _ { \mathrm { M H } } ( t ) ] } { ( 1 - q ^ { B } ) Z _ { \mathrm { N R } } ( t ) } \right) .
$$

If the denominator is zero, $\bar { \ell } _ { t } = 0$ . As in the full model, launches during year t use hubs that are already active during that year. Expected primary-hub counts update as

$$
C _ { 1 } ( t + 1 ) = C _ { 1 } ( t ) ( 1 - \xi ) + C _ { 0 } ( t ) \kappa , \qquad C _ { 0 } ( t + 1 ) = C _ { 0 } ( t ) ( 1 - \kappa ) + C _ { 1 } ( t ) \xi .
$$

The expected number of secondary hubs at year $t + 1$ is

$$
Z _ { \mathrm { M H } } ( t + 1 ) = Z _ { \mathrm { M H } } ( t ) r _ { M } ^ { H } \rho + [ Z _ { \mathrm { M U } } ( t ) + Z _ { \mathrm { M R } } ( t ) ] r _ { M } ^ { B } \chi ^ { B } + Z _ { \mathrm { M S } } ( t ) r _ { M } ^ { S } \chi ^ { S } .
$$

The three terms represent existing secondary hubs that remain Mature and continue operating, unsupported schools with Mature programs that become secondary hubs, and Supported schools with Mature programs that become secondary hubs. As in the full model, a school that begins the year with an Early program cannot become a secondary hub during the same transition.

Exact geographic coverage is then replaced by average coverage. Let $c _ { C }$ be the probability that one active primary hub covers a representative school and $c _ { H }$ the corresponding probability for one secondary hub. Mean-field reachability is

$$
\bar { g } _ { t + 1 } = 1 - ( 1 - c _ { C } ) ^ { C _ { 1 } ( t + 1 ) } ( 1 - c _ { H } ) ^ { Z _ { \mathrm { M H } } ( t + 1 ) } .
$$

This is the main spatial approximation in the model. The full model determines whether each specific school is inside the service area of each specific hub. The mean-field model instead spreads each hub’s expected coverage across the whole region. It therefore does not preserve geographic clustering or overlapping service areas. In particular, it can make geographic reach expand faster than the full spatial model when many real hubs would cover the same schools or remain concentrated in one part of the region.

Direct support then uses $\bar { w } ^ { n e w }$ for schools that were not previously Supported and $\bar { w } ^ { r e t }$ for schools that were already in a Supported state. Let $A _ { n e w }$ and $A _ { r e t }$ denote the corresponding expected masses of active, reachable, non-hub schools. The expected number of direct-support assignments is

$$
D _ { \mathrm { s u p p o r t } } = \bar { w } ^ { n e w } A _ { n e w } + \bar { w } ^ { r e t } A _ { r e t } .
$$

The expected available support capacity is

$$
Q _ { \mathrm { s u p p o r t } } = K [ C _ { 1 } ( t + 1 ) + Z _ { \mathrm { M H } } ( t + 1 ) ] .
$$

The selected support probabilities are checked against $D _ { \mathrm { s u p p o r t } } \leq Q _ { \mathrm { s u p p o r t } }$ . This capacity check does not change the transition probabilities automatically.

## State-Mass Update

The mean-field model uses the same 70 legal school transitions listed in Table 3. Their probabilities are obtained by replacing the school-specific quantities with the corresponding mean-field quantities in Table 4. For example, $\mathrm { P r } ( \mathrm { M S }  \mathrm { M H } ) \stackrel { \bullet } { = } \stackrel { S } { r } _ { M } ^ { S } \chi ^ { \dot { S } }$ and $\mathrm { P r } ( \mathrm { M S } \to \mathrm { M R } ) = r _ { M } ^ { S } ( 1 - \chi ^ { S } ) \bar { g } _ { t + 1 } ( 1 - \bar { w } ^ { r e t } )$

Let $\bar { p } _ { s  v } ( t )$ denote the resulting mean-field transition probability from state s to state v. The expected number of schools in destination state v is then

$$
Z _ { v } ( t + 1 ) = \sum _ { s } Z _ { s } ( t ) \bar { p } _ { s \to v } ( t ) ,
$$

where the sum is over all nine school states. Because the transition probabilities from each starting state sum to one, the total expected school count remains $\begin{array} { r } { \sum _ { s } Z _ { s } ( t ) = n } \end{array}$ at every year.

A mean-field run therefore begins with expected counts for the nine school states and the two primary-hub college states. After initialization, the equations above deterministically produce the expected state counts for each subsequent year. Unlike the full spatial model, the mean-field model does not sample individual schools or produce realization-to-realization variation.

## C. Spatially-Explicit Model Analysis of Indiana

## Quantities Held Fixed Across Scenarios

All 12 Indiana optimism levels use the same 2025–2026 public K–12 school universe and the same natural FIRST program dynamics. Private K–12 schools are excluded from the modeling denominator to simplify data collection. The potential-college hub population is not restricted to public institutions. A school is counted as having a FIRST program when at least one FIRST LEGO League Challenge, FIRST Tech Challenge, or FIRST Robotics Competition team is linked to that school. Program maturity is the maturity of the school’s most-developed current team.

<table><tr><td>Quantity</td><td>Fixed value</td><td>Rationale / calculation</td><td>Source or basis</td></tr><tr><td>n</td><td>1,925</td><td>Indiana public schools serving at least one K-12 grade in the 2025–2026 directory; PK-only sites are excluded.</td><td>Indiana Department of Education, 2025–2026 Indiana School Directory.ª IPEDS 2023-2024 completions and institutional</td></tr><tr><td>m</td><td>32</td><td>Conventional Indiana campuses reporting at least three bachelor&#x27;s-level completions in IPEDS CIP family 11 during 2023–2024. Public and private institutions are included; online and extension-only institutions are ex-</td><td>locations.b</td></tr><tr><td>W</td><td>3 seasons</td><td>cluded. A team is Mature after the same team identity is linked to the same school in three consecutive FIRST seasons. A school is Mature if at least one of its current teams is Mature. A closure resets this history, implementing</td><td>Fixed operational definition for mapping observed FIRST records into model states.</td></tr><tr><td> $q ^ { B }$ </td><td>0.01229</td><td>the model&#x27;s no-memory restart assumption. Natural N → E start probability. Pooled 2021–2025 estimate:  $( 8 9 - 1 ) / 7 1 \bar { 6 } 2 = 0 . 0 1 2 2 8 7 .$  , subtracting the</td><td>FIRST Team &amp; Event Search linked to IDOE public schools.c</td></tr><tr><td> $\mu _ { E } ^ { B }$ </td><td>0.28947</td><td>one known 2025 ARC-created school start. Natural  $E \to M$  maturation probability: 55/190 Early</td><td>Same linked 2021–2025 public-school panel.</td></tr><tr><td> $\delta _ { E } ^ { B }$ </td><td>0.23684</td><td>school-years. Natural Early-program closure probability: 45/190</td><td>Same linked 2021–2025 public-school panel.</td></tr><tr><td> $\nu _ { M } ^ { B }$ </td><td>0.00575</td><td>Early school-years. Natural M → E backslide probability: 2/348 Mature school-years. This is observable because maturity fol- lows the most-developed team&#x27;s continuous tenure at</td><td>Same linked 2021–2025 public-school panel.</td></tr><tr><td> $\delta _ { M } ^ { B }$ </td><td>0.09770</td><td>that school. Natural Mature-program closure probability: 34/348</td><td>Same linked 2021–2025 public-school panel.</td></tr><tr><td> $p _ { \mathrm { r e q } } ^ { \mathrm { r e t } }$ </td><td>1</td><td>Mature school-years. Every previously Supported school that remains an eligible reachable active non-hub school requests con- tinued support and receives priority over new support</td><td>ARC operating assumption supplied from the trial.</td></tr><tr><td></td><td></td><td>requests. Actual assignment remains subject to avail- able capacity at a covering hub. No accepted FIRST program link in 2025.</td><td></td></tr><tr><td>N(0) E(0)</td><td>1,790 schools 44 schools</td><td>At least one accepted current FIRST program link, but no current team at that school has met the three-season</td><td>Conservative FIRST–IDOE school linkage. Conservative FIRST-IDOE school linkage.</td></tr><tr><td>M(0)</td><td>91 schools</td><td>maturity rule. At least one current team identity is linked to that</td><td>Conservative FIRST–IDOE school linkage.</td></tr><tr><td>C1(0)</td><td>1 college</td><td>school in each of the final three consecutive seasons. One active primary hub at the start of the 2025–2026</td><td>ARC trial fact.</td></tr><tr><td> $C _ { 0 } ( 0 )$ </td><td>31 colleges</td><td>analysis. Remaining potential primary hubs:  $m - C _ { 1 } ( 0 ) = 3 2 -$ </td><td>Derived from m and the trial initial condition.</td></tr><tr><td>MH(0)</td><td>0 schools</td><td>1. No secondary school hubs at the initial boundary.</td><td>ARC trial fact.</td></tr></table>

<sup>a</sup> Indiana Department of Education, Indiana School Directory: https://www.in.gov/doe/it/data-center-and-reports/  
<sup>b</sup> Integrated Postsecondary Education Data System (IPEDS), National Center for Education Statistics: https://nces.ed.gov/ipeds/  
<sup>c</sup> FIRST Team & Event Search: https://www.firstinspires.org/team-event-search

The natural transition estimates pool the four post-COVID year-to-year transitions 2021–2022 through 2024–2025. FIRST organization records are linked conservatively to current IDOE public-school names using normalized school-name and locality information; ambiguous multi-school, community, home-school, and non-public organizations are not silently assigned to a public school. The resulting 2025 linked count of 135 public schools with at least one FIRST program should therefore be read as a conservative school-linked count, not as a claim that every unmatched FIRST organization is non-public.

## Quantities in Scenarios

Parameters describing ARC efectiveness cannot yet be estimated reliably from the single-year trial. We therefore evaluate 12 coordinatewise ordered optimism levels. Level 1 uses the least favorable assumptions considered, Level 6 provides the moderate anchor used for the main comparison, and Level 12 uses the most favorable assumptions considered. Levels 2–5 interpolate linearly between Levels 1 and 6, while Levels 7–11 interpolate linearly between Levels 6 and 12. These levels are not confidence intervals or fitted estimates. Natural program dynamics and the underlying school and college populations remain fixed across all 12 levels.

<table><tr><td>Quantity</td><td>Level 1</td><td>Level 6</td><td>Level 12</td><td>Interpretation / construction</td></tr><tr><td>Reach radius</td><td>15 mi</td><td>20 mi</td><td>25 mi</td><td>Maximum great-circle distance over which an active hub can make ARC resources reach- able to a school.</td></tr><tr><td>L</td><td>3</td><td>3</td><td>3</td><td>Maximum number of successful new school-program launches allocated to each hub active at the start of a year. Held fixed so stronger levels do not gain from greater per-hub launch</td></tr><tr><td>K</td><td>3</td><td>6</td><td>10</td><td>throughput. Expected number of direct-support slots per active hub. Three distinct schools are opera- tionally feasible with demonstrated trial resources; 6–10 represents the estimated scalable</td></tr><tr><td> $p _ { \mathrm { r e q } } ^ { \mathrm { n e w } }$ </td><td>0.05</td><td>0.10</td><td>0.15</td><td>range. Annual probability that an eligible active, reachable, non-supported school requests new direct ARC support. Whether the request is filled depends on local hub capacity.</td></tr><tr><td> $\mu _ { E } ^ { S }$ </td><td>0.28947</td><td>0.36184</td><td>0.43421</td><td>Supported Early-program maturation. The three anchors use 1.0, 1.25, and 1.5 times the observed natural maturation rate.</td></tr><tr><td> $\delta _ { E } ^ { S }$ </td><td>0.21316</td><td>0.16579</td><td>0.11842</td><td>Supported Early-program closure. The natural closure rate is multiplied by 0.9, 0.7, and 0.5.</td></tr><tr><td> $\nu _ { M } ^ { S }$ </td><td>0.00517</td><td>0.00402</td><td>0.00287</td><td>Supported Mature-program backslide, using the same 0.9, 0.7, and 0.5 multipliers.</td></tr><tr><td> $\delta _ { M } ^ { S }$ </td><td>0.08793</td><td>0.06839</td><td>0.04885</td><td>Supported Mature-program closure, using the same multipliers.</td></tr><tr><td> $\nu _ { M } ^ { H }$ </td><td>0.00172</td><td>0.00115</td><td>0.00057</td><td>Hub-program backslide, using 0.30, 0.20, and 0.10 times the observed natural Mature backslide rate.</td></tr><tr><td> $\delta _ { M } ^ { H }$ </td><td>0.02931</td><td>0.01954</td><td>0.00977</td><td>Hub-program closure, using 0.30, 0.20, and 0.10 times the observed natural Mature</td></tr><tr><td> $\chi ^ { B }$ </td><td>0.00125</td><td>0.0030</td><td>0.0035</td><td>closure rate. Annual probability that an eligible Mature, non-supported program becomes a secondary</td></tr><tr><td> $\chi ^ { S }$ </td><td>0.010</td><td>0.025</td><td>0.030</td><td>hub. Annual hub-formation probability for a Supported Mature program. Support is assumed</td></tr><tr><td> $\rho$ </td><td>0.96</td><td>0.97</td><td>0.98</td><td>to make deliberate hub training substantially more likely. Probability that a secondary hub continues hosting, conditional on its program remaining</td></tr><tr><td>κ</td><td>0.012</td><td>0.017</td><td>0.020</td><td>Mature. Primary-hub activation probability. With 31 initially inactive colleges, these correspond</td></tr><tr><td>ξ</td><td>0.05</td><td>0.045</td><td>0.040</td><td>to approximately 0.37, 0.53, and 0.62 expected new primary hubs in the first year. Primary-hub cessation probability, corresponding to mean uninterrupted active periods of approximately 20, 22, and 25 years.</td></tr></table>

Geographic coverage in the full model is calculated directly from the coordinates of individual schools and hubs. School coordinates come primarily from NCES EDGE 2024–2025, with documented Census and NCES-derived fallback geocodes for unmatched directory entries; college coordinates come from IPEDS. At each optimism level, the reach radius defines a fixed hub-to-school coverage relation, and a school is reachable only when at least one hub that covers it is active. The mean-field coverage fractions $c _ { C }$ and $c _ { H }$ in Appendix B are derived from the same radii but are not inputs to the spatial simulation.

The Supported-program rates are constructed from the empirically estimated natural rates. Level 1 assumes no improvement in maturation and only a small reduction in failure, while Levels 6 and 12 progressively increase maturation and reduce closure and backslide. Hub persistence is represented by both program stability and continued hosting. At Levels 1, 6, and 12, the resulting one-year probabilities that an active secondary hub remains a hub are approximately 0.930, 0.950, and 0.970, corresponding under constant rates to expected uninterrupted active-hub durations of approximately 14.3, 20.0, and 33.2 years.

The remaining quantities describe ARC operations and replication and therefore cannot yet be estimated from historical natural-growth data. In particular, $\chi ^ { B }$ and $\chi ^ { S }$ control secondary-hub formation, with $\chi ^ { S } > \chi ^ { \tilde { B } }$ representing the hypothesis that a school already receiving direct ARC support is more likely to be trained into a hub. L controls successful program creation and is held fixed at three at all levels, while K governs local support capacity and $p _ { \mathrm { r e q } } ^ { \mathrm { n e w } }$ controls how often eligible schools request new direct support.

The spatial simulation enforces both launch and support capacity locally. Reachable No Program schools that do not start naturally are randomly ordered and assigned to their nearest covering hub with an available launch slot, using hubs active at the start of the year. After primary- and secondary-hub transitions determine the next-year network, continuing Supported schools receive priority for direct-support capacity. New support requesters are then randomly ordered and assigned to their nearest covering hub with an available slot. Thus the $p _ { \mathrm { r e q } }$ quantities govern requests, while the $w _ { i }$ quantities give the resulting support-assignment probabilities. When an intermediate optimism level gives a fractional value of K, each active hub receives either $\lfloor K \rfloor \mathbf { \bar { o r } } \lceil K \rceil$ slots through randomized rounding so that its expected capacity equals K.

## Scenario Initial States

Because reachability depends on the optimism level’s radius, the full nine-state initialization difers slightly across levels even though the underlying $\dot { N } / E / M$ program counts are identical. The known ARC-supported pilot school is Early and reachable at all 12 levels. The table below gives the three anchor initializations; intermediate levels are calculated directly from their corresponding radii rather than by interpolating state counts.

<table><tr><td>State</td><td>Level 1</td><td>Level 6</td><td>Level 12</td></tr><tr><td> $N _ { U }$ </td><td>1,761</td><td>1,750</td><td>1,732</td></tr><tr><td> $N _ { R }$ </td><td>29</td><td>40</td><td>58</td></tr><tr><td> $E _ { U }$ </td><td>43</td><td>43</td><td>43</td></tr><tr><td> $E _ { R }$ </td><td>0</td><td>0</td><td>0</td></tr><tr><td> $E _ { S }$ </td><td>1</td><td>1</td><td>1</td></tr><tr><td> $M \boldsymbol { \upsilon }$ </td><td>82</td><td>82</td><td>79</td></tr><tr><td> $M _ { R }$ </td><td>9</td><td>9</td><td>12</td></tr><tr><td> $M _ { S }$ </td><td>0</td><td>0</td><td>0</td></tr><tr><td> $M _ { H }$ </td><td>0</td><td>0</td><td>0</td></tr></table>

At every level, the school-state counts sum to $n = 1 , 9 2 5 .$ , with the same marginal program totals $N ( 0 ) = 1 , 7 9 0 , E ( 0 ) = 4 4$ and $M ( 0 ) = 9 1$ . Only the initial reachable/unreachable partition changes with the assumed geographic radius. The college initialization remains $\dot { C } _ { 1 } ( 0 ) = 1$ and $C _ { 0 } ( 0 ) = 3 1$ at all 12 levels.

## D. Parameter Sensitivity Analysis

Several ARC-specific parameters cannot yet be estimated reliably from the trial deployment. We therefore tested how strongly projected outcomes depend on the parameter ranges represented by the 12 optimism levels. The main questions are which assumptions most strongly control projected growth and whether the same major drivers remain important across diferent combinations of assumptions.

We used two complementary sensitivity tests: one-at-a-time (OAT) sweeps and Morris screening (Morris 1991; Campolongo, Cariboni, and Saltelli 2007). OAT shows the direction and size of the efect produced by changing one factor around the level-6 scenario. Morris instead tests each factor under many diferent combinations of the other assumptions, showing which factors have the greatest influence more broadly. Using both lets us compare an easily interpreted scenario sweep with a globa sensitivity test.

For OAT, we varied one sensitivity factor through optimism levels 1–12 while holding all other factors at level 6. We tested 11 factors: nine individual parameters and two parameter bundles. The bundles were Supported-program closure/backslide $( \delta _ { E } ^ { S } , \nu _ { M } ^ { S } , \delta _ { M } ^ { S } )$ and secondary-hub closure/backslide $( \nu _ { M } ^ { H } , \delta _ { M } ^ { H } )$ , because the rates within each bundle already vary together through a common multiplier in the scenario construction. Changing reach radius R also recalculates the geographic relationships and initial reachable states derived from that radius. We measured the resulting change in projected outcomes at years 10, 20, and 40.

Morris repeatedly changes one factor by a fixed step and records how much the output changes, while starting from many diferent combinations of the other factors. We rank factors using $\mu ^ { * }$ , the average absolute size of these changes, so larger values indicate greater overall influence. We also calculate σ, which becomes large when a factor’s efect varies substantially across diferent surrounding assumptions. We used six grid levels and 30 Morris trajectories. Each parameter setting was averaged over 128 paired Monte Carlo realizations, using the same random-number streams within each comparison to reduce simulation noise.

The two tests produced closely aligned results. For projected schools with programs, both ranked $\kappa , \chi ^ { B } , \bar { w } ^ { \mathrm { n e w } } , \chi ^ { S }$ , and the Supported closure/backslide bundle as the five strongest factors at year 10, in that order. At year 20, both gave the same leading six: $\mathbf { \bar { \chi } } _ { \mathcal { X } } ^ { S } , \kappa , \mathbf { \chi } ^ { B } , \bar { w } ^ { \mathrm { n e w } }$ , K, and R. At year 40, they retained the same leading six factors, with only $\chi ^ { B }$ and $\bar { w } ^ { \mathrm { n e w } }$ exchanging positions. Reach radius R was the strongest factor for the number of schools within ARC reach at years 10, 20, and 40, while κ was consistently strongest for primary-hub growth. Secondary-hub growth was driven most strongly by $\chi ^ { B }$ at year 10 and by $\chi ^ { S }$ at years 20 and 40.

Together, these results show how ARC’s growth drivers shift as the network develops. Early expansion is driven primarily by establishing additional primary hubs and converting existing Mature programs into secondary hubs, while later growth depends increasingly on Supported programs becoming further secondary hubs. The close agreement between OAT and Morris shows that these same major drivers remain important across many surrounding assumptions. Overall, ARC scales when it can successfully create and sustain new hubs, with direct support, geographic reach, and local capacity determining how far that hub-driven growth can propagate.

Table 5: One-at-a-time sensitivity of projected schools with programs, $P ( t )$ . Each factor was varied from optimism level 1 to level 12 while all other factors remained fixed at level 6. Values report the change in projected programs from the level-1 to level-12 setting for that factor.
<table><tr><td>Sensitivity factor</td><td>Year 10</td><td>Year 20</td><td>Year 40</td></tr><tr><td>Supported secondary-hub formation  $( \chi ^ { S } )$ </td><td>7.4</td><td>70.2</td><td>429.6</td></tr><tr><td>Direct-support capacity per hub (K)</td><td>5.5</td><td>51.1</td><td>338.4</td></tr><tr><td>Reach radius (R)</td><td>5.1</td><td>42.4</td><td>302.3</td></tr><tr><td>New support-request probability  $( p _ { \mathrm { r e q } } ^ { \mathrm { n e w } } )$ </td><td>9.1</td><td>52.4</td><td>216.4</td></tr><tr><td>Unsupported secondary-hub formation  $( \chi ^ { B } )$ </td><td>14.4</td><td>57.9</td><td>213.3</td></tr><tr><td>Primary-hub activation (κ)</td><td>19.6</td><td>63.7</td><td>165.9</td></tr><tr><td>Supported closure/backslide bundle  $( \delta _ { E } ^ { S } , \nu _ { M } ^ { S } , \delta _ { M } ^ { S } )$ </td><td>6.5</td><td>32.9</td><td>120.5</td></tr><tr><td>Secondary-hub closure/backslide bundle  $( \nu _ { M } ^ { H } , \delta _ { M } ^ { H } )$ </td><td>1.7</td><td>17.6</td><td>109.7</td></tr><tr><td>Secondary-hub retention (ρ)</td><td>1.4</td><td>12.9</td><td>93.8</td></tr><tr><td>Supported Early-program maturation  $( \mu _ { E } ^ { S } )$ </td><td>1.4</td><td>8.8</td><td>31.7</td></tr><tr><td>Primary-hub cessation (ξ)</td><td>2.4</td><td>10.7</td><td>30.3</td></tr></table>

Table 6: Morris sensitivity of projected schools with programs, $P ( t )$ . Values are $\mu ^ { * }$ , the average absolute change produced by a Morris step on the common normalized optimism-level scale. Larger values indicate greater influence across the tested combinations of assumptions. Bold values mark the strongest factor at each year.
<table><tr><td>Sensitivity factor</td><td>Year 10</td><td>Year 20</td><td>Year 40</td></tr><tr><td>Supported secondary-hub formation  $( \chi ^ { S } )$ </td><td>7.3</td><td>66.4</td><td>380.6</td></tr><tr><td>Direct-support capacity per hub (K)</td><td>4.6</td><td>43.8</td><td>279.4</td></tr><tr><td>Reach radius (R)</td><td>4.7</td><td>37.9</td><td>251.6</td></tr><tr><td>New support-request probability  $( p _ { \mathrm { r e q } } ^ { \mathrm { n e w } } )$ </td><td>8.1</td><td>45.4</td><td>174.3</td></tr><tr><td>Unsupported secondary-hub formation  $( \chi ^ { B } )$ </td><td>12.8</td><td>49.0</td><td>183.7</td></tr><tr><td>Primary-hub activation (κ)</td><td>16.8</td><td>53.3</td><td>138.6</td></tr><tr><td>Supported closure/backslide bundle  $( \delta _ { E } ^ { S } , \nu _ { M } ^ { S } , \delta _ { M } ^ { S } )$ </td><td>5.3</td><td>28.7</td><td>114.0</td></tr><tr><td>Secondary-hub closure/backslide bundle  $( \nu _ { M } ^ { H } , \delta _ { M } ^ { H } )$  1</td><td>1.8</td><td>15.8</td><td>91.9</td></tr><tr><td>Secondary-hub retention (ρ)</td><td>1.0</td><td>10.4</td><td>74.6</td></tr><tr><td>Supported Early-program maturation  $( \mu _ { E } ^ { S } )$ </td><td>0.9</td><td>5.5</td><td>25.8</td></tr><tr><td>Primary-hub cessation (ξ)</td><td>2.4</td><td>10.4</td><td>29.8</td></tr></table>

## E. Extended Spatial Simulation Results

<table><tr><td colspan="4">Primary Secondary</td></tr><tr><td>Scenario</td><td>Programs</td><td>Reachable</td><td>hubs</td><td>hubs</td></tr><tr><td>ARC off</td><td>161.4</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Level 1</td><td>338.0</td><td>592.3</td><td>5.7</td><td>4.8</td></tr><tr><td>Level 2</td><td>393.1</td><td>733.2</td><td>6.2</td><td>8.1</td></tr><tr><td>Level 3</td><td>477.1</td><td>898.1</td><td>6.7</td><td>13.9</td></tr><tr><td>Level 4</td><td>601.0</td><td>1,063.4</td><td>7.2</td><td>23.7</td></tr><tr><td>Level 5</td><td>772.8</td><td>1,241.1</td><td>7.6</td><td>40.0</td></tr><tr><td>Level 6</td><td>992.2</td><td>1,415.1</td><td>8.0</td><td>67.2</td></tr><tr><td>Level 7</td><td>1,134.0</td><td>1,516.9</td><td>8.4</td><td>89.1</td></tr><tr><td>Level 8</td><td>1,271.7</td><td>1,609.2</td><td>8.6</td><td>117.1</td></tr><tr><td>Level 9</td><td>1,393.8</td><td>1,682.7</td><td>8.9</td><td>150.3</td></tr><tr><td>Level 10</td><td>1,491.9</td><td>1,743.6</td><td>9.2</td><td>187.9</td></tr><tr><td>Level 11</td><td>1,572.8</td><td>1,791.9</td><td>9.6</td><td>229.6</td></tr><tr><td>Level 12</td><td>1,638.0</td><td>1,829.1</td><td>9.9</td><td>276.4</td></tr></table>

Table 7: Mean Indiana public-school outcomes after 40 years in the full spatial model. Each ARC optimism level is averaged over 1,000 Monte Carlo realizations.

![](images/f7bde426c74391e058f3846388aedcd1257f6acb56b2ce7044a0e5982e8a0c65.jpg)

![](images/d0768f689c523d0df7f1d4af7154e3021412ae804bac1fb924d8eb35c9b0b634.jpg)  
Figure 6: Program growth for all 1,925 modeled schools (left) and the 719 rural schools (right) across all 12 optimism levels. Solid lines show Full ARC means over 1,000 Monte Carlo runs, with levels 1, 6, and 12 emphasized. Dashed lines show ARC of. Rural schools lie outside the 2020 Census urban-area boundaries used in Figure 2.

<table><tr><td></td><td></td><td colspan="2">Year 10</td><td colspan="2">Year 20</td><td colspan="2">Year 40</td></tr><tr><td>Level</td><td>Model</td><td>Rural</td><td>Urban</td><td>Rural</td><td>Urban</td><td>Rural</td><td>Urban</td></tr><tr><td>1</td><td>ARC off</td><td>52.9</td><td>100.2</td><td>58.0</td><td>100.1</td><td>59.9</td><td>101.5</td></tr><tr><td>1</td><td>Full ARC</td><td>67.4</td><td>130.8</td><td>89.6</td><td>167.0</td><td>115.0</td><td>223.0</td></tr><tr><td></td><td>Primary hubs only Initial primary +</td><td>64.6</td><td>124.4</td><td>80.6</td><td>147.6</td><td>93.1</td><td>174.0</td></tr><tr><td></td><td>secondary hubs</td><td>59.3</td><td>113.4</td><td>68.7</td><td>122.3</td><td>75.5</td><td>133.9</td></tr><tr><td>6</td><td>Full ARC</td><td>82.9</td><td>150.8</td><td>147.3</td><td>262.2</td><td>340.9</td><td>651.2</td></tr><tr><td></td><td>Primary hubs only Initial primary +</td><td>72.2</td><td>130.2</td><td>95.9</td><td>165.0</td><td>118.3</td><td>204.9</td></tr><tr><td></td><td>secondary hubs</td><td>68.5</td><td>124.0</td><td>99.0</td><td>170.5</td><td>189.9</td><td>360.6</td></tr><tr><td>12</td><td>Full ARC</td><td>98.1</td><td>173.7</td><td>247.3</td><td>426.6</td><td>597.4</td><td>1,040.5</td></tr><tr><td></td><td>Primary hubs only Initial primary +</td><td>78.1</td><td>137.7</td><td>109.9</td><td>184.7</td><td>143.1</td><td>241.2</td></tr><tr><td></td><td>secondary hubs</td><td>77.0</td><td>135.0</td><td>152.2</td><td>253.1</td><td>448.0</td><td>794.1</td></tr></table>

Table 8: Mean numbers of rural and urban schools with FIRST programs after 10, 20, and 40 years, over 1,000 runs per scenario. All scenarios start with 34 rural and 101 urban programs. At level 6, Full ARC yields 341 rural programs after 40 years, compared with 118 with primary hubs only and 190 with the initial primary hub plus secondary hubs.

## F. Survey Information

The evaluation used two anonymous end-of-program surveys: one for undergraduate mentors and one for parents or guardians of participating K–12 students. All respondents were at least 18 years old. Participation was voluntary, respondents could skip questions or stop at any time, and no names or other identifying information were collected. Surveys were administered via Qualtrics.

## Undergraduate Survey

All nine students enrolled in the ARC course were eligible to participate, and seven completed the survey. The survey was distributed after the course ended and final grades had been submitted. Respondents rated both their perceived state before the course and their perceived state after the course during the same survey administration.

At the beginning of the survey, students were told the purpose of the study, that participation was voluntary and anonymous, and that participation would not afect their grades or academic standing. They were told that they could skip questions or stop at any time. Respondents confirmed that they were at least 18 years old and agreed to participate before beginning the survey. Question Types. The survey contained 35 items after the consent screen. Eighteen items formed nine retrospective before-andafter pairs covering teaching, communication, group management, community connection, mentoring, leadership, and interest in AI, robotics, and graduate school. These items used a five-point Likert scale. The survey also included nine post-course rating questions, four background questions, three short-response questions, and one gold-standard attention check instructing respondents to select 4. All seven respondents answered the attention check correctly

Demographics. The students came from several undergraduate years and included both computer science and engineering students. The class also included a mix of genders, countries of origin, and primary languages. Exact demographic information in these categories was not collected. Of the seven survey respondents, six selected “CS/AI/DS Major” and one selected “Other.” Two were first-year students, one was a second-year student, three were fourth-year students, and one did not report a year. Four reported prior FIRST experience and five reported prior teaching or mentoring experience.

![](images/b0229d28c75c488e1fdc662f007e58c0c945decdf40a7923b8792916ab7f70bc.jpg)  
Table 9: Undergraduate survey questions and possible responses.

## Parent Proxy Survey

K–12 students were not surveyed directly. Parents or legal guardians instead reported on their child’s experience after the program. Four parents completed the survey. Parents could answer using their own observations, conversations with their child, or both.

At the beginning of the survey, parents were told that the study examined how participation in the AI and robotics outreach workshops may influence students’ interest, confidence, and engagement in STEM, programming, and basic AI concepts. They were told that the survey was anonymous and voluntary, that they could skip questions or stop at any time, and that no data would be collected directly from children. Respondents confirmed that they were at least 18 years old and agreed to participate before beginning the survey.

Question Types. The survey contained 22 items after the consent screen. Twelve items formed six retrospective before-and-after pairs covering basic robot programming knowledge, access to programming resources, opportunities to practice programming and problem solving, enjoyment of robotics club, plans to continue robotics, and interest in further STEM education. These items used a five-point Likert scale from Strongly Disagree to Strongly Agree. Five additional Likert questions asked about the child’s experience with the mentors and the efects of mentoring. The survey also contained two gold-standard attention checks, two short-response questions, and one question asking whether the parent had discussed the survey with their child. Al four respondents answered both attention checks correctly. One parent reported discussing the survey with their child before answering. Two reported that they had not discussed it with their child. One reported that they had not discussed it with thei child but had attended the competition near the end of the program.

Demographics. Detailed demographic information about the parents or children was not collected. All children were in 5th grade attending the rural elementary school focused on in the trial deployment, and were members of the school robotics club.

<table><tr><td>Survey question</td><td>Possible answers</td></tr><tr><td>Before the program, my child knew how to do basic robot programming.</td><td>Likert (1–5)</td></tr><tr><td>After the program, my child knew how to do basic robot programming.</td><td>Likert (1–5)</td></tr><tr><td>Before the program, my child&#x27;s team had access to the programming resources they needed.</td><td>Likert (1–5)</td></tr><tr><td>After the program, my child&#x27;s team had access to the programming resources they needed.</td><td>Likert (1–5)</td></tr><tr><td>Before the program, my child had opportunities to practice programming and problem-solving skills.</td><td>Likert (1–5)</td></tr><tr><td>After the program, my child had opportunities to practice programming and problem-solving skills.</td><td>Likert (1–5)</td></tr><tr><td>Before the program, my child enjoyed their robotics club.</td><td>Likert (1–5)</td></tr><tr><td>After the program, my child enjoyed their robotics club.</td><td>Likert (1–5)</td></tr><tr><td>For this question, please select “Agree.&quot;</td><td>Likert (1–5)</td></tr><tr><td>Before the program, my child planned to continue robotics club.</td><td>Likert (1–5)</td></tr><tr><td>After the program, my child planned to continue robotics club.</td><td>Likert (1–5)</td></tr><tr><td>Before the program, my child considered pursuing further education in science, robotics, or programming</td><td>Likert (1–5)</td></tr><tr><td>After the program, my child considered pursuing further education in science, robotics, or programming.</td><td>Likert (1–5)</td></tr><tr><td>My child enjoyed working with the university mentors.</td><td>Likert (1–5)</td></tr><tr><td>My child felt supported and encouraged by the university mentors.</td><td>Likert (1–5)</td></tr><tr><td>The mentoring helped my child&#x27;s FLL team prepare for competition.</td><td>Likert (1–5)</td></tr><tr><td>The mentoring improved my child&#x27;s ability to solve problems and try new ideas.</td><td>Likert (1–5)</td></tr><tr><td>The mentoring improved teamwork on my child&#x27;s FLL team.</td><td>Likert (1–5)</td></tr><tr><td>For this question, please select &quot;Disagree.&quot;</td><td>Likert (1–5)</td></tr><tr><td>How do you feel this program affected your child&#x27;s interest in robotics and STEM going forward?</td><td>Short Response</td></tr><tr><td>Did you discuss this survey with your child?</td><td>[Yes, before answering | No | No, but I at-</td></tr><tr><td>Please share anything meaningful your child said about the mentors, the program, or the FLL season. Do not use names or personal details.</td><td>tended the competition] Short Response</td></tr></table>

Table 10: Parent-proxy survey questions and possible responses.

## G. Survey Response Data

Because of the small number of respondents, Tables 11 and 12 report the complete individual Likert responses underlying the retrospective pre–post results. Responses are listed in a fixed respondent order so that before and after values remain paired. All items use a 1–5 scale from Strongly Disagree to Strongly Agree.

Table 13: Undergraduate mentor open-response survey data.
<table><tr><td>Survey question</td><td>Response</td></tr><tr><td>Please write what you think this course did well.</td><td>Introducing me to more applications of AI and robotics within the community, it gave me a lot more experience mentoring than I previously had.</td></tr><tr><td></td><td>Multiple mentors per group worked well because it was hard to have one keep all the kids together in one area and not have at least one of them roaming.</td></tr><tr><td rowspan="4"></td><td>The facilitation was fantastic. The fact that the kids actually came here, we had two tables and several robots to work with, and everyone was able to go through the process smoothly is already a success. I also think the integration of mentoring and guest lectures from faculty at the forefront of their respective fields exposes students to a diverse range of new perspectives.</td></tr><tr><td>I think this course connected to the community well.</td></tr><tr><td>Definitely made me feel more connected to the local community I thought this course did a great job helping students prepare for the FLL competition</td></tr><tr><td>structure worked well, especially with hands-on sessions followed by guest lectures. Those talks helped me understand how AI and robotics appear in everyday life, and I realized that building LEGO robots and teaching them simple commands is the first step toward how these systems function in the real world.</td></tr><tr><td rowspan="6">Please write any suggestions you have to improve the course.</td><td>The guest lecturers sharing their research was really informative. Some of the planning for activities and group structures seems like it should be a little</td></tr><tr><td>more concrete, which probably results from the fact it's a new course. More structure for the mentoring to make better use of the limited time with the students and strategies to be able to make the students want to do something instead</td></tr><tr><td>of forcing them to do something like strategy before jumping in. It would also be nice to have a bigger space because the room we are in got really crowded and of course would be nice if we could have the table set up permanently. I think it should have been clearer from the beginning that we were preparing the</td></tr><tr><td>students for a serious competition. We at as well as the teachers at learned a lot from this first year and I expect the team will do much better next year, especially considering how late in the season we started. I would say that providing more time with the children on the robotics teams to</td></tr><tr><td>work on the robots more would be beneficial. In addition, having them focus on developing a plan and not developing randomly in every direction would also help. Mock competitions would also be highly beneficial.</td></tr><tr><td>Sometimes activities felt rushed or uneven, so having weekly goals or mini milestones when working with the kids and their robots would make it easier for both students and mentors to stay on track. We did this towards the end, but starting off with that could possibly work.</td></tr><tr><td rowspan="2">Please share any meaningful stories you have from your interactions with the FLL students. Please do not use real names or personal information. as simply as I could, and I believe he gained a bit more of an understanding regarding</td><td>The sessions with the students could have been geared more towards evaluation and feedback instead of building and programming. I think it was the third session with them, and I was supervising as the team rebuilt their robot, and while they were building, one of them began to ask me about different concepts regarding gear ratios and getting a certain part to move faster. I explained it</td></tr><tr><td>that, which I found very meaningful. Also, by the end of the competition, we managed to get one of the children who had been trying to do more than he should to take a step back to help his other team members learn as he guided them through different parts.</td></tr><tr><td></td><td>One student was clearly very invested in the competition, and also really wanted to do things his way. On the day of the competition, when things weren't working quite as he planned, I encouraged him to try something different, and collaborate with one of his teammates who wanted to help. Regardless of their final performance, seeing them work together for the good of the team showed clear growth.</td></tr><tr><td>Survey question Response</td><td></td></tr><tr><td></td><td>The kids told me after their last time in the room at that they didn't think they would be there without me or my help. It really made me feel connected, and more “a part of their team" than I had previously.</td></tr><tr><td></td><td>After the competition. One of the kid I've been working with told me that he will see me next year, it showed me that he not only learned the technical skills — he also felt supported and wanted to keep going. That small moment made the whole experience worthwhile.</td></tr><tr><td></td><td>I really enjoyed watching how their thinking process worked. If something did not go as planned, they would detect their mistake, try again, and sometimes even draw out where they wanted the robot to go to fix the issue. Seeing them do that made me realize strategies I had not considered before.</td></tr></table>

Note. Black bars indicate text removed for anonymity. An em dash indicates that no response was provided.

Table 14: Parent open-response survey data.
<table><tr><td>Survey question</td><td>Response</td></tr><tr><td>Please share anything meaningful your child said about the mentors, the program, or the FLL season. Do not use names or personal details.</td><td></td></tr><tr><td></td><td>My son was impressed with his mentors and enjoyed working with them. He loved being with the mentors and his time at</td></tr><tr><td>How do you feel this program af-</td><td>Positively</td></tr><tr><td>fected your child&#x27;s interest in robotics and STEM going forward?</td><td>My son has always had a strong interest in STEM and robotics and really enjoyed</td></tr><tr><td></td><td>working with his mentors and was sad the program had to end. I think it was</td></tr><tr><td></td><td>a great experience for the kids to be able to work with college students and learn from their experience and just have the opportunity to communicate with a young adult.</td></tr><tr><td></td><td>Unfortunately my child was very interested in programming but did not get the chance.</td></tr><tr><td></td><td>That had nothing to do with the mentors.</td></tr></table>

Note. Black bars indicate text removed for anonymity. An em dash indicates that no response was provided.

Table 11: Complete undergraduate retrospective pre–post Likert responses (n = 7). Entries within each cell correspond to respondents U1–U7 in the same order. An em dash indicates a skipped response.
<table><tr><td>Measure</td><td>Before</td><td>After</td></tr><tr><td>Teaching technical concepts</td><td>2, 4, 4, 4, 3, 1, 3</td><td>4, 4, 5, 4, 5, 4, 4</td></tr><tr><td>Adapting explanations</td><td>3, 3, 4, 4, 2, 2, 5</td><td>4, 4, 5, 4, 4, 5, 5</td></tr><tr><td>Small-group management</td><td>4,2, 4, 3, 3,1, 4</td><td>4, 3, 5, 4, 4, 4, 4</td></tr><tr><td>Community connection</td><td>1, 1, 4, 2, 1, 2, 2</td><td>3, 3, 5, 4, 5, 4, 4</td></tr><tr><td>Teaching/mentoring meaning</td><td>2, 3, 4, 4, 4, 4, 3</td><td>4, 5, 5, 4, 5, 5, 4</td></tr><tr><td>Leadership confidence</td><td>3, 1, 4, −, 3, 2, 4</td><td>4, 1, 5, 3, 4, 4, 4</td></tr><tr><td>AI interest</td><td>1,3,5,3,5,3,5</td><td>2, 3, 5,4,5, 4, 5</td></tr><tr><td>Robotics interest</td><td>5, 4, 5, 4, 5, 2, 4</td><td>5, 4, 5, 4, 5, 4, 5</td></tr><tr><td>Graduate-school interest</td><td>4,1, 5, 3, 4, 3,5</td><td></td></tr><tr><td></td><td></td><td>4, 3, 5, 3, 5,4,5</td></tr></table>

Table 12: Complete parent-proxy retrospective pre–post Likert responses (n = 4). Entries within each cell correspond to respondents P1–P4 in the same order.
<table><tr><td>Measure</td><td>Before</td><td>After</td></tr><tr><td>Robot programming knowledge</td><td>1,4,2, 1</td><td>4,5,2,5</td></tr><tr><td>Access to programming resources</td><td>2,3,2,1</td><td>4,5,4,4</td></tr><tr><td>Programming/problem-solving practice</td><td>3,4,2,3</td><td>4,5,3,5</td></tr><tr><td>Enjoyed robotics club</td><td>4,3,3,3</td><td>5,5,2,5</td></tr><tr><td>Planned to continue robotics club</td><td>4,5,3,3</td><td>4,5,2,5</td></tr><tr><td>Further STEM/programming education</td><td>3,5,3,2</td><td>3,5,2,4</td></tr></table>

## H. ARC Course Details and Reproducibility

The ARC course was a three-credit undergraduate course meeting once per week for a two-hour block. Mentoring was the central course activity. Mentoring and participation accounted for 70% of the course grade and required consistent attendance, communication with team coaches, and contribution to the teams’ progress. A final essay accounted for 20% and a final presentation for 10%. Optional biweekly reflections asked students to connect course topics to their experiences and communities. These were eligible for extra credit. The final assignment required a 4–8 page essay and an 8–10 minute presentation connecting course topics with the student’s mentoring experience. Students could write either a persuasive essay on AI, robotics, and community or a short research proposal addressing a community-relevant problem.

The first four weeks prepared the undergraduates before they began mentoring K–12 teams. Students were introduced to the FLL platform by building and programming small robots, learned about robotics education and FIRST, completed required youth-protection training, and practiced methods for teaching programming. The first K–12 workshop occurred in week five. After workshops began, approximately 90 minutes of each weekly meeting were devoted to direct mentoring and approximately 30 minutes to continuing course material. Table 15 summarizes the course structure.

A primary hub does not need to reproduce the trial course lecture-for-lecture. The essential components are mentor preparation followed by recurring, supervised K–12 workshops. At least one instructor or other experienced robotics mentor is needed to prepare the undergraduate mentors and remain available during workshops. This role cannot be replaced entirely by fixed lesson material because teams difer in technical problems, pace, competition readiness, and group dynamics. The experienced mentor must be able to help when a team or novice mentor encounters a problem that was not anticipated in the course material.

The physical requirements depend strongly on the robotics program being supported. For the FLL deployment studied here, a hub needs FLL robot kits, devices capable of programming them, the current competition field and game materials, suficient working space, supervision, and the required youth-protection procedures. The trial provided two FLL game arenas so that multiple teams could test robots during the same workshop. The same course structure can be used with FTC or FRC teams, but those programs require progressively greater hardware, workspace, fabrication and maintenance resources, and transportation support. For FTC and FRC, basic safety training should also be mandatory for all mentors, as these programs commonly use power tools.

The continuing seminar component is intentionally portable. The trial used both instructor-led and guest seminars to connect robotics mentoring with AI and robotics research and with problems afecting the surrounding community. Another university does not need the same lecturers or presentation materials. Local faculty, researchers, robotics organizations, and community partners can instead provide topics relevant to that region. These topics should focus on real research or development in AI or robotics, and how that development can impact local communities.

<table><tr><td>Week</td><td>Phase</td><td>Topic / activity</td></tr><tr><td>1</td><td>Preparation</td><td>Course introduction; FLL kits; build a small mobile robot.</td></tr><tr><td></td><td>Preparation</td><td>Robotics education and FIRST; youth-protection training; FLL programming</td></tr><tr><td></td><td>Preparation</td><td>Practical methods for teaching programming; FLL programming.</td></tr><tr><td>234</td><td>Preparation</td><td>Practical teaching exercise.</td></tr><tr><td>5</td><td>Workshops begin</td><td>Workshop preparation and first K-12 mentoring session.</td></tr><tr><td>6-10</td><td>Workshops + seminars</td><td>Recurring mentoring; seminars on citizen science, robotics safety and ethics, FTC/FRC, human-robot teaming, and AI literacy.</td></tr><tr><td>11-14</td><td>Workshops + seminars</td><td>Continued mentoring; later topics included autonomous systems, AI safety and explainability, and preparation for competition and final work.</td></tr><tr><td>15-16</td><td>Synthesis</td><td>Final-project support and student presentations.</td></tr></table>

Table 15: Structure of the ARC course during the initial deployment. Seminar topics can be replaced with locally relevant AI and robotics topics while retaining the mentor-training and workshop structure.