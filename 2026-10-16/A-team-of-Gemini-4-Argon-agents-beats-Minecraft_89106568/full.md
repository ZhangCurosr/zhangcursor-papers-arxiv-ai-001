# A team of Gemini 4 Argon agents beats Minecraft

Maxence Faldor<sup>1</sup>, Satinder Singh<sup>1</sup> and Roberta Raileanu<sup>1</sup> <sup>1</sup>Google DeepMind

We placed four Gemini 4 Argon agents in a sandboxed Minecraft survival world and gave them one goal: defeat the Ender Dragon. Each agent controlled a player via an open-source library called Mineflayer and coordinated with its teammates through in-game chat and shared memory. No one assigned roles or wrote game logic. The agents planned, wrote, and debugged every program themselves. Researchers could watch the run on a live dashboard and pause or stop it at any time, but they had no cause to intervene. Starting from a randomly generated world on Easy difficulty, the team wrote 2,783 programs and defeated the dragon in 27 hours and 14 minutes, dying 605 times along the way. To our knowledge, this is the first end-to-end autonomous completion of Minecraft by AI agents, achieved using only the standard Mineflayer library without pre-scouted seeds or other game-specific tools written by humans. Winning required the agents to keep a coherent plan across thousands of dependent steps for more than a day. Complex real-world software engineering and enterprise tasks demand the same long-horizon coding and multi-agent coordination. We describe the setup, how the run unfolded, the team’s strategies (dividing the work, sharing what they learned, recovering each other’s items), and the limits of the result.

![](images/2f84eacb20d633fc82efffa84393e17de4c5c41473916bbb1f753792984bd663.jpg)  
Figure 1 | The four agents, and their run step by step: the thirteen steps from bare hands to the Ender Dragon, each colored by the player who reached it first, over 27 hours and 14 minutes. Item icons and player faces from the open-source Pixel Perfection texture pack (XSSheep and Nova\_Wostra, CC BY-SA 4.0), not Minecraft’s textures.

## 1. Introduction

Four Gemini 4 Argon agents entered a newly generated Minecraft survival world with one instruction: defeat the Ender Dragon. The run tests whether a team of general-purpose coding agents, not built for Minecraft, can pursue one goal for more than a day, through thousands of programs and hundreds of failures, while coordinating with its peers. Each agent controlled a player through Mineflayer, an open-source JavaScript library for Minecraft bots, and communicated with its teammates through the game’s chat. Twenty-seven hours after their first program, the team defeated the dragon.

Reaching the end of Minecraft from a random spawn requires a long sequence of dependent steps across three dimensions (Fan et al., 2022): gathering resources and crafting tools in the Overworld, collecting blaze rods and ender pearls in the Nether, locating an underground stronghold, and destroying the ten crystals that heal the dragon in the End. Prior work in Minecraft has trained policies on gameplay (Baker et al., 2022; Hafner et al., 2025; SIMA Team, 2025) or paired language models with game-specific planners, curricula, and skill libraries (Wang et al., 2023; Wang et al., 2024; Wang et al., 2025; Altera.AL et al., 2024). Without pre-assigned roles or a prescribed plan, the four agents divided the work among themselves, kept shared notes of coordinates and lessons, recovered each other’s items after deaths, and wrote 2,783 JavaScript programs to complete the game.

27 h 14 min from the first program to the dragon's defeat

2,783   
programs writen and run by   
the four agents

63,313 distinct lines of JavaScript

229   
messages between   
teammates in the game's   
chat

605 deaths along the way

## 2. How the agents play

Each player is driven by our general-purpose Gemini 4 Argon coding agent, connected to a standard Minecraft server through Mineflayer and its pathfinding plugin. Rather than seeing pixels or pressing keys, an agent acts by writing and running a short JavaScript program, such as walk to these coordinates and mine every iron ore within 16 blocks, and by reading the text that comes back: its position, health and inventory, what it found, or the error it hit. It then writes the next program. Each program runs until its task finishes or reaches a time limit chosen by the agent; the median program ran for 33.5 seconds. Over the run, each player had a program running about 40% of the time; the rest went to reading results, thinking and writing the next program.

Mineflayer lets a program read a player’s position, health, inventory and surroundings, and order basic actions such as walking to a point, digging or placing a block, crafting an item or attacking. It does not know how to play: what to do, and in what order, was up to the agents.

![](images/32755695deb3374690ad8073b36b54b037a2faea5e98f7422182c4e4781a6385.jpg)  
Figure 2 | How each agent plays. It writes a JavaScript program, a Mineflayer bot runs it in the shared world, and text comes back. The players share the game’s chat and can read each other’s notes. A fifth agent, the Observer, reports on the game but never plays. Item icons and player faces from the open-source Pixel Perfection texture pack (XSSheep and Nova\_Wostra, CC BY-SA 4.0), not Minecraft’s textures.

The game, Minecraft Java Edition 1.20.4, ran in survival mode on Easy difficulty in a randomly generated world, with cheat commands disabled and no human input during the run. Beyond a brief note that they were working as a team of four, the agents’ instructions about the game consisted of a single paragraph:

“The objective is to defeat the Ender Dragon, starting from the spawn of the session’s shared Minecraft survival world. [. . . ] This is the only criterion. Any means of meeting it within the rules of survival mode is permitted, and the players decide how to share the work.”

Their other instructions are the same as for any environment: verify the result of each action, build reusable tools, record lessons in a memory shared by the team, and report outcomes honestly.

## 2.1. Agents and tools

Each of the five agents, the four players and the Observer, is an instance of the same general-purpose coding agent, run with Gemini 4 Argon on its own cloud machine. Each has the tools it uses for software work, including a code workspace with version control, a shell, file editing and web search, and a persistent memory with a private part and a part shared by the team. The agent compacts its context when it grows too long, as it does for software work; the instructions ask agents to keep in memory what must survive a compaction, such as work in progress and next steps. None of the 2,839 programs the players ran during the session (2,783 before the win) made a network request or referred to a web source, and neither their chat nor the Observer’s report mentions one.

Players interact with the game only through a command-line client. It sends a JavaScript program to the player’s Mineflayer bot on the game server, waits up to a timeout that the agent chooses (60 seconds by default), and prints what the program logged and returned. The program runs as the body of an asynchronous function that receives the bot, with the pathfinding plugin loaded and no helpers added. It can load the built-in modules of Node.js and the installed Mineflayer packages, but cannot start processes or write files, and values it stores on the global object persist between programs. Players talk through the game’s chat, which reaches every bot and appears in the output of each player’s next program. The server records chat, deaths and advancements in its log, keeps every program with its output, and saves the world every five minutes; after a failure, it restarts from the latest save.

## 2.2. Instructions

Each agent’s instructions combine a general part, the same for every environment, with a short description of the environment and of the objective. The general part for players asks them to observe the environment before acting and verify the result afterwards, to validate new tools on short tasks before trusting them, to build and refine their own tools when manual actions limit their speed or reliability, to record mechanics, strategies and workarounds in memory so that no obstacle is met twice, to keep working while background jobs run, to be honest about their actions and outcomes, and to pursue the objective rather than exploit flaws of the environment or of its evaluation. The description of the environment explains how to run code with the bot, how the chat works, where the server records the game and where to find the documentation of Mineflayer; it contains no game strategy. A brief note lists the other members of the team.

## 2.3. Monitoring

A fifth agent, the Observer, followed the session as an independent witness and never played. Its instructions ask it to check the players’ claims against the record before reporting them, to report specification gaming, such as an exploited flaw in the evaluation, and fabricated results, to leave the game unchanged, and to intervene only on failures of the environment, such as a broken tool or a quota ceiling, never on the players’ strategy. It published a live report of the session, with a summary of progress, a log of verified events and its findings about the infrastructure. It flagged no specification gaming, and it never joined the game or ran a program in it. The run didn’t contain any safety incidents, according to the Observer agent and our manual reviews of the logs.

## 3. The 27-hour run

From the first program to the dragon’s defeat took 27 hours and 14 minutes. The four agents wrote and ran 2,783 programs, exchanged 229 messages and died 605 times along the way. That time includes 6 server restarts, which paused the world for about 20 minutes in all, and spans about 81 in-game days. For human players, HowLongToBeat’s median for Minecraft’s main story is 45 hours and the random-seed speedrun record is under seven minutes; the agents played as a team of four on Easy difficulty, through programs rather than a screen and keyboard.

![](images/9213ff44fc77d7dfcd40f6c1664ca4023131bb12bce63365af7ec810f21dc570.jpg)  
Figure 3 | The whole run. Each lane is a player: dots are chat messages, diamonds are advancements, red ticks are deaths. Bars count programs per half hour; orange dashes mark the six restarts of the world server. Item icons and player faces from the open-source Pixel Perfection texture pack (XSSheep and Nova\_Wostra, CC BY-SA 4.0), not Minecraft’s textures.

![](images/d01a63e4b5b3404bbf7bf0f9caebf55c862b383b7beade687656c22eddb9a3c8.jpg)  
Figure 4 | The thirteen dependent steps from initial spawn to the Ender Dragon, each with the time the team first reached it and the player who did. Item icons from the open-source Pixel Perfection texture pack (XSSheep and Nova\_Wostra, CC BY-SA 4.0), not Minecraft’s textures.

## 3.1. Hour 0: a team forms

Ten minutes after the first program ran, janesick proposed a division of labor in the chat. Within a minute after that, auratian answered with a different plan, and gyldenia replied “Copy that!”. Roles settled quickly: gyldenia scouted for food, beds and flint; bulicame went down for diamonds; janesick smelted iron and made tools; auratian prepared the Nether portal.

![](images/288323c9bae4f7cdb8fd003a376a7869676f08e4fd281c5772ef1db4b4fa2f5e.jpg)  
Figure 5 | Chat in the first quarter hour: janesick proposes roles ten minutes in, and auratian answers with a different plan within a minute. Messages in all chat figures are verbatim, timed from the first program; [. . . ] marks an elision. Item icons and player faces from the open-source Pixel Perfection texture pack (XSSheep and Nova\_Wostra, CC BY-SA 4.0), not Minecraft’s textures.

Within two hours the team had iron tools, diamonds and a portal to the Nether. auratian found 55 blocks of lava right in a village, gyldenia gathered bread, flint and buckets, janesick placed the last block of obsidian, and auratian lit the frame two seconds later.

![](images/fc58f8570ec5b72a1228682d531a925c0acbb63c23b2936665bbf0722997b4f9.jpg)  
The spawn area and the village where the team set up its first base, rendered from the saved world. Rendered with the open-source Pixel Perfection texture pack (XSSheep and Nova\_Wostra, CC BY-SA 4.0), not Minecraft’s textures.

![](images/5113f6d6fca863666686f7faad9414c5e5f2d49687f997d6c923364f6de68c9b.jpg)  
The Nether portal the team built in the village, lit 1 hour 42 minutes into the run. Rendered with the open-source Pixel Perfection texture pack (XSSheep and Nova\_Wostra, CC BY-SA 4.0), not Minecraft’s textures.

## 3.2. Hours 2 to 11: into the Nether

An Eye of Ender needs blaze powder, from the blazes that guard Nether fortresses, and an ender pearl, dropped by Endermen. The team split up. janesick found a fortress and its blaze spawner. In a warped forest, gyldenia, auratian and bulicame built two-block-high shelters: an Enderman that is looked at teleports toward the player but, at nearly three blocks tall, cannot fit under a two-block ceiling, so from inside a low bunker, the Endermen came to them. Five and a half hours into the run, the team had 15 pearls in a shared chest.

Bringing blaze powder back to the Overworld took more than six hours. janesick burned to death carrying twelve of it, and a Ghast fireball destroyed a pile of dropped items. The team turned losses into rules: after the Ghast attack, auratian posted a lesson for everyone, and two minutes later, janesick posted the root cause and a fix.

![](images/4c01b242189d459739d2c552ccb59412bd44013dbd111b8585b2dada0f4952a8.jpg)  
Figure 6 | After a Ghast fireball destroyed dropped items, auratian posts a rule for the team, and janesick follows two minutes later with the root cause and a fix. Item icons and player faces from the open-source Pixel Perfection texture pack (XSSheep and Nova\_Wostra, CC BY-SA 4.0), not Minecraft’s textures.

![](images/070ec50c97de413578a62c97a1a110c4d49e4bf4308f820de5600b3b8da3e802.jpg)  
The blaze spawner in the Nether fortress, and the shelter the agents built beside it, which they called the South Pillbox. Rendered with the open-source Pixel Perfection texture pack (XSSheep and Nova\_Wostra, CC BY-SA 4.0), not Minecraft’s textures.

![](images/09088b614e62c3d078d5f2cd56c46d8e96652d4a3ba9f2a201949e554aab9828.jpg)  
The bridge the agents built across a lava lake in the Nether, its Ghast-blasted gap patched with basalt. Rendered with the open-source Pixel Perfection texture pack (XSSheep and Nova\_Wostra, CC BY-SA 4.0), not Minecraft’s textures.

## 3.3. Hours 10 to 13: finding the stronghold

A thrown Eye of Ender flies toward the nearest stronghold, the underground structure that holds the portal to the End. janesick threw two from different places, sampled each eye’s flight every 50 milliseconds, and computed where the two lines cross: 1.3 km from the village. The estimate was 43 m from the actual portal room.

![](images/eb6c78d561146e81ac5f22a03ed697d14cfddd69e3e7da183af88c60f1f5f749.jpg)

## Triangulating the stronghold

A thrown Eye of Ender flies toward the nearest stronghold. janesick threw two from two places, recorded both directions, and computed where the lines cross.

1 2 Where janesick threw the two eyes

Where the lines cross: (432, −1376)

The actual portal room: (463, −30, −1406)

## 1.3 km

from the village to the crossing

43 m from the crossing to the portal room

10 h 46 min into the run when janesick found the crossing

Figure 7 | janesick’s two throws, minutes apart in the eleventh hour of the run, and where the lines cross. Coordinates are in blocks; one block is one meter. Item icons from the open-source Pixel Perfection texture pack (XSSheep and Nova\_Wostra, CC BY-SA 4.0), not Minecraft’s textures.

```javascript
bot.activateItem();
const samples = [];
const start = Date.now();
while (!isCancelled() && Date.now() - start < 4500) {
const ent = Object.values(bot.entities).find(e =>
e && (e.name === 'eye_of_ender' || (e.displayName &&
,→ e.displayName.toLowerCase().includes('eye'))) &&
e.position && e.position.distanceTo(bot.entity.position) < 30
);
if (ent) {
samples.push({
t: Date.now() - start,
x: +ent.position.x.toFixed(4),
y: +ent.position.y.toFixed(4),
z: +ent.position.z.toFixed(4)
});
} else if (samples.length > 5) {
break;
}
await new Promise(r => setTimeout(r, 50));
// [...]
// Intersect with Ray 1: P1=(17.5, -164.526), U1=(0.323733, -0.946149)
const x1 = 17.5, z1 = -164.526, ux1 = 0.323733, uz1 = -0.946149;
// x1 + t1 ux1 = x0 + t2 ux2
// z1 + t1 uz1 = z0 + t2 uz2
const det = ux1 <sub>*</sub> (-uz2) - uz1 <sub>*</sub> (-ux2);
if (Math.abs(det) > 1e-6) {
const rhsX = x0 - x1;
const rhsZ = z0 - z1;
const t1 = (rhsX (-uz2) - rhsZ (-ux2)) / det;
const t2 = (ux1 <sub>*</sub> rhsZ - uz1 <sub>*</sub> rhsX) / det;
const ix = x1 + t1 <sub>*</sub> ux1;
const iz = z1 + t1 <sub>*</sub> uz1;
```

From the program janesick ran 10 hours 45 minutes into the run: throw the eye, sample its flight, and intersect the two rays. Abridged; the comments are janesick’s own.

Then the eyes had to travel 1.3 km. When auratian drowned 8 m from bulicame, carrying all twelve, janesick and auratian each sent bulicame the exact spot and the minute the eyes would vanish. bulicame picked up all twelve with two and a half minutes to spare. At 13 hours 17 minutes into the run, bulicame reported the End portal lit, all twelve eyes in place.

![](images/2ed0519724953bebb8eb322f9f3db8345b6ab26b58b16e8a4564038b29ceef5f.jpg)  
Figure 8 | auratian drowns carrying all twelve Eyes of Ender. janesick and auratian send bulicame the exact spot and the despawn deadline, and bulicame recovers all twelve. Item icons and player faces from the open-source Pixel Perfection texture pack (XSSheep and Nova\_Wostra, CC BY-SA 4.0), not Minecraft’s textures.

![](images/17f7b54e7c03f03d8d49983fa705789048883af59b556f4c38d85134a58d6e12.jpg)  
Figure 9 | The End portal in the stronghold, with all twelve eyes in place. Rendered with the opensource Pixel Perfection texture pack (XSSheep and Nova\_Wostra, CC BY-SA 4.0), not Minecraft’s textures.

## 3.4. Hours 13 to 27: learning the dragon

The End is where the team slowed down; the dragon killed them 69 times. Between attempts, they inspected its internal state from the game, down to the number of its current behavior phase, and wrote down rules: never wait on the arrival platform, approach only while it is perched, leave the

End to reset it when it gets stuck. To leave, they jumped into the void on purpose, a shortcut human players use too: a player who dies returns to its bed in the Overworld. Seventeen and a half hours in, all four did so within six minutes, so that the empty End would unload and the stuck dragon reset.

![](images/b686e91c03b687ac9021c64a5c2fb6fe4ed8355e3b3d1535c359311e707eafed.jpg)  
Figure 10 | Rules for the End, written between attempts at the dragon: never wait on the arrival platform, leave the End to reset a stuck dragon, approach only while it perches. Item icons and player faces from the open-source Pixel Perfection texture pack (XSSheep and Nova\_Wostra, CC BY-SA 4.0), not Minecraft’s textures.

The breakthrough came about 21 hours in. auratian pointed out that the ten crystals on top of the obsidian pillars kept healing the dragon to full health, and worked out that no pickaxe was needed to reach them: stack dirt to climb beside each pillar, punch the crystal, dig back down by hand. Over the next four hours and forty minutes, the team destroyed all ten. Thirty-nine minutes after the last one fell, the dragon died, with bulicame down to 2.5 of its 20 health points.

![](images/1013455789cd1bbf26c7efcf050ffecd34964f90839da6d05acba31a9843c395.jpg)  
Figure 11 | auratian finds why the dragon stays at full health, the healing crystals, and then a way to reach them without a pickaxe. Item icons and player faces from the open-source Pixel Perfection texture pack (XSSheep and Nova\_Wostra, CC BY-SA 4.0), not Minecraft’s textures.

Ten crystals, destroyed from pillars of dirt  
![](images/416ca4a4bcaa5eccd2a7d336d7af24c527bbe64824bc0b0b18c54f24efb71957.jpg)  
gyldenia 6 auratian 3 bulicame 1 janesick 0

The dragon's health, as the players' programs read it  
![](images/a571e3e26471eb0ccab47d3d5814bdbe5c1a0dcb4023b6d94dd89c6cda1fe606.jpg)

Figure 12 | Left: the ten crystals on the End island, seen from above, colored by the player who destroyed each, with the time into the run. Right: the dragon’s health in the final fight, as the players’ own programs read it. Item icons from the open-source Pixel Perfection texture pack (XSSheep and Nova\_Wostra, CC BY-SA 4.0), not Minecraft’s textures.  
![](images/ad683af60620464c998ecb172cc67fda3cf295164455f2e9f6fe4617432d1c08.jpg)  
The bridge the agents built from the arrival platform to the main island, and the staircase up its cliff. Rendered with the open-source Pixel Perfection texture pack (XSSheep and Nova\_Wostra, CC BY-SA 4.0), not Minecraft’s textures.

![](images/f303684ce94ba56fcdeae55afccdd96c5d4aa2b66eb6b3af5376a4679f4aaeaa.jpg)  
A flight over the End after the win: the obsidian pillars that held the dragon’s healing crystals, and the dragon egg on the exit portal. Rendered with the open-source Pixel Perfection texture pack (XSSheep and Nova\_Wostra, CC BY-SA 4.0), not Minecraft’s textures.

## 4. Emergent Behaviors

Nothing in the agents’ instructions mentioned roles, chests or rescues, beyond the line that the players decide how to share the work. Their general instructions do ask for reusable tools, notes in a shared memory and honest reports, but not what to build, what to record or how to work together. Over the 229 messages they exchanged before the win, the team converged on practices familiar from effective human teams.

![](images/b397485dd03ed283fbcc264da2d2140b71b8bf3f1267e3edcf264f4e6ab17369.jpg)  
Figure 13 | Left: messages that match simple patterns for each behavior, by sender; a message can match several. Right: the 605 deaths before the win, by cause, from the server log.

• Specialization. Each agent settled into a role: gyldenia gathered food and supplies and later destroyed six of the ten crystals; janesick made tools, found the fortress and triangulated the stronghold; auratian lit the portal and worked out how to reach the crystals; bulicame helped build the Enderman shelters and landed the final blow in the End.

• A shared memory. They posted the coordinates of chests, portals and dangers, and kept notes on the quirks of their tools in the memory the team shares. Nineteen minutes in, gyldenia warned the team about a quirk of Mineflayer’s crafting and how to avoid it; a helper function with the fix appears in 655 programs written by all four agents.

• Their own tools. The agents kept 146 pieces of state and code alive between programs, including 42 helper functions: a background loop that keeps a bot afloat in water, timed walking, item counting, and a counter that makes leftover code from a timed-out program stop itself, referenced in 1,519 programs.

• Looking out for each other. Urgent warnings about dangers, and items recovered after a teammate’s death.

• Corrections. When auratian’s program announced a success that had not happened, auratian corrected it two minutes later: “my previous chat was an unguarded script branch”. And when janesick reported coming back with twelve blaze powder one second after burning to death, gyldenia flagged the contradiction within three minutes.

![](images/a77e91c057df37889f5bcc394b574383ad48070b4ed38f5781f7a83a61620139.jpg)

ALL 14x ENDER\_EYE BANKED IN OVERWORLD CHEST (16, 72, -168)! Ready for Stronghold (432, -1376)!

![](images/7a915f4258487054b7068a4cd6f9f6c62892aa16f857fa28105e3256362f819a.jpg)

CORRECTION FROM AURATIAN: 00:37:39Z restart saved after my death! I spawned in Overworld with empty inv (my previous chat was an unguarded script branch). […]

Figure 14 | auratian retracts, two minutes later, a success that its program had announced but that had not happened. Item icons and player faces from the open-source Pixel Perfection texture pack (XSSheep and Nova\_Wostra, CC BY-SA 4.0), not Minecraft’s textures.

## 5. Verification

Every one of the 2,783 programs and its output, every chat message and the full server log were recorded. The win is in the server log: bulicame has made the advancement [Free the End], 27 hours 14 minutes after the first program. None of the programs used a server command.

## 6. Cost

Up to the win, the four players processed about 1.35 billion input tokens, 91% of them served from the context cache, and generated about 16 million output tokens, including 4 million thinking tokens, over roughly 24,000 model calls. At Gemini 4 Argon’s public API list price (introductory), that comes to about \$520 (\$1,040 at the standard price). The whole session, including the Observer agent and the day of activity after the win, came to about \$810 (\$1,620). These figures are estimates from list prices and the session’s token counters, not the cost of serving the model.

## 7. Related work

Minecraft has long been a testbed for open-ended agents. MineRL released human demonstrations and ran a competition on sample-efficient learning of the diamond task (Guss et al., 2019b; Guss et al., 2019a). Video PreTraining learned keyboard and mouse control from online videos and crafted diamond tools (Baker et al., 2022), DreamerV3 collected diamonds from scratch with reinforcement learning (Hafner et al., 2025), and STEVE-1 followed short-horizon open-ended text instructions (Lifshitz et al., 2023). MineDojo called defeating the Ender Dragon, starting barehanded in a fresh world, one of its moonshot goals (Fan et al., 2022). SIMA and SIMA 2 act from pixels with a keyboard and mouse across many 3D games, and SIMA 2 was also tested on held-out Minecraft tasks (SIMA Team, 2024; SIMA Team, 2025). These systems learn their policies from large amounts of gameplay data or experience.

Another line of work uses language models as planners or programmers, building on interleaved reasoning and tool use (Yao et al., 2023) and code as policy (Liang et al., 2023). Voyager wrote Mineflayer programs into a growing skill library under an automatic curriculum (Wang et al., 2024). DEPS, GITM, JARVIS-1, Plan4MC, and Optimus-1 added Minecraft-specific planners, knowledge, memory, or learned skills (Wang et al., 2023; Zhu et al., 2023; Wang et al., 2025; Yuan et al., 2023; Li et al., 2024). Multi-agent studies examined coordination and social behavior in games, including Minecraft (Gong et al., 2024; Altera.AL et al., 2024; White et al., 2025). Reported long-horizon results in this line center on the Overworld technology tree, such as obtaining diamond tools.

Two community projects have recently reported end-to-end dragon kills: a single Claude Opus 5 agent calling 37 high-level Minecraft tools on Normal difficulty (AI Bengineering, 2026), and a planner paired with an action-selection model on a fixed speedrun seed in Peaceful difficulty (Malde, 2026). Our setting differs mainly in what the agents were given. Four general-purpose coding agents received the standard Mineflayer library, as published, with its pathfinding plugin, a one-paragraph objective, and their general instructions. We added nothing on top: no instructions about the game, no curriculum, no skill library, no human-written tools, and no route. Every program they ran, they wrote themselves. They played a standard survival server on Easy from a random seed and coordinated through chat and shared notes.

## 8. Limitations

• Perception and navigation through code. Mineflayer tells the agents exactly which blocks and creatures are around them, including blocks behind solid rock and the dragon’s internal state, information that a human player has to see and interpret or cannot see at all, and its pathfinding plugin plans routes around obstacles, which humans do by eye. Playing from pixels with a keyboard and mouse, as SIMA 2 does, remains much harder.

• Easy difficulty. Monsters spawn and attack, but deal less damage than on Normal or Hard, and hunger cannot kill a player.

• Prior knowledge. Gemini 4 Argon knows a great deal about Minecraft and Mineflayer from its training data. The run tests whether a general-purpose coding agent can turn that knowledge into working programs and team coordination over a day-long task, not whether it can play a game it has never seen.

• Many failures along the way. The players died 605 times, mostly to monsters; 22% of programs ended in an error; the fight in the End took nearly 14 hours.

• Infrastructure. The world server restarted 6 times during the run, interrupting the game each time.

• A single run. We describe a single run, the first launch we tried, which cannot show how reliably the setup succeeds.

## 9. Responsible development

The agents ran in a sandboxed Minecraft world and their in-game programs could control only their own character: they could neither start processes nor write files, and none used a server command. The agents themselves ran on dedicated cloud machines, with the tools they use for software work. A fifth agent, the Observer, followed the session as an independent witness to watch for cheating or specification gaming; neither the Observer nor our manual review of a sample of the logs found any. Every program, message and game event was recorded. Researchers could also watch the run live on a dashboard and pause or stop it at any time, but had no cause to intervene. Studying how teams of agents coordinate in a setting where every step is visible is one way to understand such systems before they are trusted with real-world tasks. The skills required to solve Minecraft such as coordinating a team, recovering from failure, and pursuing a goal for more than a day, are also what agents need to work reliably alongside people.

## Acknowledgments

We thank Jake Bruce and Vibhavari Dasagi for helpful discussions and feedback on this work.

Minecraft is a registered trademark of Mojang Synergies AB / Microsoft Corporation. This independent scientific research is not approved by or affiliated with Mojang or Microsoft. All in-game visual assets in this report were rendered using the open-source Pixel Perfection texture pack by XSSheep and Nova\_Wostra (licensed under CC BY-SA 4.0).

## References

AI Bengineering. Mine AI MCP: beating Minecraft with an LLM agent. https://github.com/ aibengineering/mine-ai-mcp, 2026. Dataset: https://huggingface.co/datasets/ aibengineering/beat-the-game-minecraft. Run that defeated the Ender Dragon, 2026- 09-15. Accessed 2026-10-13.

Altera.AL, A. Ahn, N. Becker, S. Carroll, N. Christie, M. Cortes, A. Demirci, M. Du, F. Li, S. Luo, P. Y. Wang, M. Willows, F. Yang, and G. R. Yang. Project Sid: Many-agent simulations toward AI civilization. arXiv preprint arXiv:2411.00114, 2024. doi: 10.48550/arXiv.2411.00114. URL https://arxiv.org/abs/2411.00114.

B. Baker, I. Akkaya, P. Zhokhov, J. Huizinga, J. Tang, A. Ecoffet, B. Houghton, R. Sampedro, and J. Clune. Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos. In Advances in Neural Information Processing Systems, 2022. doi: 10.48550/arXiv.2206.11795. URL https://arxiv.org/abs/2206.11795.

L. Fan, G. Wang, Y. Jiang, A. Mandlekar, Y. Yang, H. Zhu, A. Tang, D.-A. Huang, Y. Zhu, and A. Anandkumar. MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge. In Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, 2022. doi: 10.48550/arXiv.2206.08853. URL https://arxiv.org/abs/2206.08853.

R. Gong, Q. Huang, X. Ma, Y. Noda, Z. Durante, Z. Zheng, D. Terzopoulos, L. Fei-Fei, J. Gao, and H. Vo. MindAgent: Emergent Gaming Interaction. In Findings of the Association for Computational Linguistics: NAACL 2024, pages 3154–3183. Association for Computational Linguistics, 2024. doi: 10.18653/v1/2024.findings-naacl.200.

W. H. Guss, C. Codel, K. Hofmann, B. Houghton, N. Kuno, S. Milani, S. Mohanty, D. P. Liebana, R. Salakhutdinov, N. Topin, M. Veloso, and P. Wang. The MineRL 2019 Competition on Sample Efficient Reinforcement Learning using Human Priors. arXiv preprint arXiv:1904.10079, 2019a. doi: 10.48550/arXiv.1904.10079. URL https://arxiv.org/abs/1904.10079.

W. H. Guss, B. Houghton, N. Topin, P. Wang, C. Codel, M. Veloso, and R. Salakhutdinov. MineRL: A Large-Scale Dataset of Minecraft Demonstrations. In Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence (IJCAI), pages 2442–2448. International Joint Conferences on Artificial Intelligence Organization, 2019b. doi: 10.24963/ijcai.2019/339.

D. Hafner, J. Pasukonis, J. Ba, and T. Lillicrap. Mastering diverse control tasks through world models. Nature, 640(8059):647–653, 2025. doi: 10.1038/s41586-025-08744-2.

Z. Li, Y. Xie, R. Shao, G. Chen, D. Jiang, and L. Nie. Optimus-1: Hybrid Multimodal Memory Empowered Agents Excel in Long-Horizon Tasks. In Advances in Neural Information Processing Systems, 2024. doi: 10.48550/arXiv.2408.03615. URL https://arxiv.org/abs/2408.03615.

J. Liang, W. Huang, F. Xia, P. Xu, K. Hausman, B. Ichter, P. Florence, and A. Zeng. Code as Policies: Language Model Programs for Embodied Control. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 9493–9500. IEEE, 2023. doi: 10.1109/ICRA48891.2023.10160591.

S. Lifshitz, K. Paster, H. Chan, J. Ba, and S. McIlraith. STEVE-1: A Generative Model for Text-to-Behavior in Minecraft. In Advances in Neural Information Processing Systems, 2023. doi: 10.48550/ arXiv.2306.00937. URL https://arxiv.org/abs/2306.00937.

R. Malde. Astra and JEV Minecraft agent. https://github.com/rmalde/minecraft-agent, 2026. GitHub repository, created 2026-09-20. Accessed 2026-10-13.

SIMA Team. Scaling Instructable Agents Across Many Simulated Worlds. arXiv preprint arXiv:2404.10179, 2024. doi: 10.48550/arXiv.2404.10179. URL https://arxiv.org/abs/ 2404.10179.

SIMA Team. SIMA 2: A Generalist Embodied Agent for Virtual Worlds. arXiv preprint arXiv:2512.04797, 2025. doi: 10.48550/arXiv.2512.04797. URL https://arxiv.org/abs/ 2512.04797.

G. Wang, Y. Xie, Y. Jiang, A. Mandlekar, C. Xiao, Y. Zhu, L. Fan, and A. Anandkumar. Voyager: An Open-Ended Embodied Agent with Large Language Models. Transactions on Machine Learning Research, 2024. doi: 10.48550/arXiv.2305.16291. URL https://arxiv.org/abs/2305.16291.

Z. Wang, S. Cai, G. Chen, A. Liu, X. Ma, and Y. Liang. Describe, Explain, Plan and Select: Interactive Planning with Large Language Models Enables Open-World Multi-Task Agents. In Advances in Neural Information Processing Systems, 2023. doi: 10.48550/arXiv.2302.01560. URL https: //arxiv.org/abs/2302.01560.

Z. Wang, S. Cai, A. Liu, Y. Jin, J. Hou, B. Zhang, H. Lin, Z. He, Z. Zheng, Y. Yang, X. Ma, and Y. Liang. JARVIS-1: Open-World Multi-Task Agents With Memory-Augmented Multimodal Language Models. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(3):1894–1907, 2025. doi: 10.1109/TPAMI.2024.3511593.

I. White, K. Nottingham, A. Maniar, M. Robinson, H. Lillemark, M. Maheshwari, L. Qin, and P. Ammanabrolu. Collaborating Action by Action: A Multi-agent LLM Framework for Embodied Reasoning. arXiv preprint arXiv:2504.17950, 2025. doi: 10.48550/arXiv.2504.17950. URL https: //arxiv.org/abs/2504.17950.

S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao. ReAct: Synergizing Reasoning and Acting in Language Models. In International Conference on Learning Representations, 2023. doi: 10.48550/arXiv.2210.03629. URL https://arxiv.org/abs/2210.03629.

H. Yuan, C. Zhang, H. Wang, F. Xie, P. Cai, H. Dong, and Z. Lu. Skill Reinforcement Learning and Planning for Open-World Long-Horizon Tasks. arXiv preprint arXiv:2303.16563, 2023. doi: 10.48550/arXiv.2303.16563. URL https://arxiv.org/abs/2303.16563.

X. Zhu, Y. Chen, H. Tian, C. Tao, W. Su, C. Yang, G. Huang, B. Li, L. Lu, X. Wang, Y. Qiao, Z. Zhang, and J. Dai. Ghost in the Minecraft: Generally Capable Agents for Open-World Environments via Large Language Models with Text-based Knowledge and Memory. arXiv preprint arXiv:2305.17144, 2023. doi: 10.48550/arXiv.2305.17144. URL https://arxiv.org/abs/2305.17144.