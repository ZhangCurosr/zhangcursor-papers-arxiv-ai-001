# A Head Camera Is (Almost) All You Need: An Electronics-Free UMI Glove Labelled and Observed by One Moving Camera

Yosub Shin University of Hawaii at Manoa yosubs@hawaii.edu

Yuan Liu Independent hiveyuan@gmail.com

Igor Molybog University of Hawaii at Manoa molybog@hawaii.edu

Abstract: Handheld demonstration devices decouple data collection from the robot, but conventional designs burden the hand with a wrist camera, cables and a pose tracker (SLAM, an IMU, a VR controller, or a fixed external camera that the collector’s body occludes), and burden the pipeline with as many data streams to timestamp, synchronise and cross-calibrate. We show that a single egocentric camera (\$122) worn on the collector’s head replaces both the camera and the tracker, and the streams with one. It reads printed fiducials on an electronicsfree glove (\$22) against paper tiles taped around the workspace, giving the 6-DoF gripper pose and the jaw width from the same frames the policy will see; at deployment the same camera rides a low-cost (\$120) arm beside the robot that mimics the collector’s head from the robot’s joint state. Against the robot’s encoders the labels are accurate to 6.5 mm and $1 . 6 ^ { \circ }$ median with 93–100 % frame coverage, and a policy trained on them instead of encoders pays 3 % in validation loss. Adding 194 glove demonstrations to 48 teleoperated episodes lifts rollout success from 10/25 to 17/25, level with 172 teleoperated episodes alone (18/25), and the full co-trained head-camera policy succeeds in 20/25 against 15/25 for the wristcamera recipe, whose policies drop the object when the goal leaves their view, and 12/25 for a fixed camera.

![](images/9860165e5dba352e08c1511d6132d1fe2467d751db8a0f2d0d859507f738592e.jpg)

![](images/a936d58e2db1bf5c3888fd051fc0daf73bfd711fbeb253d3c2b801a7977347e9.jpg)

![](images/7942e229ab69890ba017e89306e9830058d81481950be06f8d032d4a774e29cd.jpg)  
Figure 1: One camera, two mounts. (a) The collector wears the camera and holds the electronicsfree glove over the tiled workspace. (b) The camera’s own frame: tiles (green), ball (red) and jaw tags (blue, yellow) give the pose and the jaw width. (c) The same camera on a six-joint positioner beside the robot during a rollout. The wrist fisheye on the glove and the fixed camera on the post serve the ablations only.

## 1 The idea

Teleoperation occupies a robot and an operator for every demonstration. Handheld devices decouple collection from the robot: with ours a collector records 2.0–3.1 demonstrations per minute against 1.6 by teleoperation on the same task (§2), in parallel and without robot time, with a \$22 printed glove, a \$122 wearable camera and four paper tiles. The glove is purely mechanical, a replica of the robot’s gripper carrying fiducials and nothing else, and the one head-worn camera is at once the policy’s observation and the source of the gripper’s pose and jaw width, so there is no second sensor stream to synchronise or calibrate against.

Every UMI-style device answers two questions: what does the policy see, and how is the gripper tracked? The conventional answer puts both on the wrist. SLAM on the wrist camera [1, 2] needs a mapping pass and textured scenes and loses track under occlusion or fast motion; a visual-inertial sensor adds a second device [3]; a VR controller adds weight (119 g of YUBI’s 319 g [4, 5]) and external tracking infrastructure [6, 7]; a fixed overhead camera [8] avoids drift but ties collection to a mount and loses the hand behind the body in 8–22 % of frames (§2). Each tracker is a second stream beside the camera, with its own clock and calibration to reconcile, and the wrist camera itself adds cable and weight while its view is tied to the hand: even through UMI’s 155<sup>◦</sup> fisheye the frame holds what the jaws face, so a drop-off bin leaves it once the hand lifts or turns [9].

We answer both with one head-worn camera, addressing two challenges this workshop names [10]: tracking without SLAM and the human-to-robot viewpoint gap. Gaze precedes the hand [11], so the head keeps the gripper at arm’s length, unoccluded by the torso, with the workspace around it. From that vantage an 11-face fiducial ball gives the pose, tags on the jaws the opening, and four AprilTag tiles [12] the world frame. At deployment the camera rides a small printed servo arm beside the robot that imitates head motion from the robot’s joint state, identically in teleoperation and rollouts, so glove, teleoperated and rollout data share one viewpoint. EgoMimic [13] co-trains on egocentric human video with the hand pose estimated from the video; our collector holds a replica of the robot’s gripper, so contact and jaw width are the robot’s. Active-vision teleoperation [14, 15] steers a robot camera from a VR headset; our positioner needs neither operator nor headset.

## 2 System

Collection. The glove is an unpowered printed replica of the YAM gripper [16] from Shin [8], stock fingers on a rack-and-pinion, with an 11-face 15 mm ArUco dodecahedron [17, 18] on the back of the hand. Fingertip tags facing a wrist camera [1] are edge-on from the head, so each jaw carries a printed wedge, the tail, with 12 mm AprilTags on five faces beside a fixed tag on the gripper back: one face is always visible, and the tail-to-back distance along the jaw axis gives the width. Parts cost \$22. The collector wears an Arducam B0587 4K camera (\$122) on a \$10 GoPro head strap (103 g worn), recording at 15 Hz with a short exposure, aimed once per session so the frame centre coincides with the collector’s own (App. A). Four 60 mm AprilTag tiles define the workspace frame, placed where the gaze lands during grasping so that two stay in view, with no other tag in the scene. Continuous recording with keypress episode boundaries gives 2.0–3.1 pick-and-place demonstrations per minute (median 6–8 s) against 1.6 by teleoperation on the same arm (median 14 s).

Labels from the head camera. Each frame yields (i) the camera pose from the tiles, (ii) the gripper pose in the camera frame from PnP over the visible ball faces, and (iii) the jaw width from the tails by a per-unit linear map (0.9–3.8 mm holdout RMS on calipered spacers). Composing (i) and (ii) gives the tool pose in the tile frame; a one-time link to the robot base, fitted with the ball mounted on the YAM (App. A), turns it into joint-space actions comparable to teleoperation. Two mechanisms keep the camera localised under head motion: the tile layout is refit from the multi-tile frames of the footage itself (four-tile reprojection 3.4 px to 0.5 px RMS), and frames with a single tile are chained to the previous accepted pose under velocity bounds rather than discarded. Dropouts over 0.3 s split an episode and segments under 1 s are dropped: 0.7–7.4 % of frames are cut per session,

Table 1: The eight policies. Teleop labels are encoders except arm 6. Val = ACT loss on 6 held-out teleoperated episodes with encoder targets. Rollouts: 25 placements, one trial each, 60 s budget; n = no grasp, d = dropped before the bin, b = dropped at the bin, t = timeout.
<table><tr><td># Arm</td><td></td><td>Glove labels</td><td>Observation</td><td>Teleop + glove</td><td>Val</td><td>Success</td><td>Fails</td></tr><tr><td>1</td><td>teleop only</td><td></td><td>head cam</td><td>172</td><td>0.318</td><td>18/25</td><td>n5 d1 t1</td></tr><tr><td>2</td><td>device</td><td>head cam</td><td>head cam</td><td>172 + 194</td><td>0.263</td><td>20/25</td><td>n5</td></tr><tr><td>3</td><td>teleop only, low budget</td><td></td><td>head cam</td><td>48</td><td>0.464</td><td>10/25</td><td>n15</td></tr><tr><td>4</td><td>device, low budget</td><td>head cam</td><td>head cam</td><td>48 + 194</td><td>0.295</td><td>17/25</td><td>n8</td></tr><tr><td>5</td><td>wrist-camera UMI</td><td>fixed cam</td><td>wrist fisheye</td><td>168 + 253</td><td>0.257</td><td>15/25</td><td>n6 d3 b1</td></tr><tr><td>6</td><td>head-cam teleop labels</td><td></td><td>head cam</td><td>172 (head labels)</td><td>0.328</td><td>15/25</td><td>n10</td></tr><tr><td>7</td><td>fixed camera as system</td><td>fixed cam</td><td>fixed cam</td><td> $1 7 2 + 2 5 5$ </td><td>0.256</td><td>12/25</td><td>n12 b1</td></tr><tr><td>8</td><td>head labels, wrist view</td><td>head cam</td><td>wrist fisheye</td><td> $1 6 8 + 1 9 2$ </td><td>0.252</td><td>14/25</td><td>n3 d7 b1</td></tr></table>

200 segments from 198 demonstrations. A fixed external camera on the same demonstrations loses   
8–22 % to body occlusion and fragments them into 259 segments.

The robot’s head. A six-joint SO-101 follower arm [19] (six Feetech STS3215 servos, \$122 in parts) stands beside the YAM and carries the same camera on a GoPro-pattern mount in place of its gripper (Fig. 1c). Its kinematics are fitted once from 60 tile-verified poses (7 mm $/ 0 . 7 ^ { \circ }$ residual). At each session start it parks at the collector’s median head pose and registers against the tiles; thereafter it tracks the gripper from the YAM’s joint encoders alone, transparent to the policy: a fixed-gain controller holds the gripper’s bearing and a ridge-regression residual fitted on about 100 glove episodes reproduces head motion relative to the hand (46 mm / 5.4<sup>◦</sup> median head-pose error on held-out episodes against 87 mm / 7.0<sup>◦</sup> for a static camera; App. A). The camera follows the gripper with 0.4 s lag and 16 mm median tile-referenced error, and frames it where the collector did (normalised gripper centroid 0.37/0.62 against 0.38/0.60).

Datasets and training. Teleoperation under the positioner gave 178 episodes (38.6k frames; 6 held out for validation); four glove sessions gave 198 demonstrations, 200 segments (50.0k frames; 6 held out). All policies are ACT [20] in LeRobot [21], 50k steps, joint-space actions, best checkpoint by validation loss on the held-out teleoperated episodes. Following Shin [8], glove timelines are resampled to teleoperation pace and widths mapped into the gripper command space, with the closed state anchored to the teleoperators’ minimum commanded squeeze (0.22 of stroke rather than the 0.25 median): the head-view policies released no block during the carry in 120 rollouts.

## 3 Experiments

Design. Eight policies (Table 1) span three views: the head camera, a wrist fisheye and a fixed external camera. Teleoperated actions come from the encoders except in arm 6, and glove labels from the head camera unless stated. The teleoperator watched the head-camera preview and all three cameras recorded every episode, so the arms share demonstration trajectories and differ only in observation and data composition. Arms 1–4 test co-training at two teleoperation budgets (172 and 48 episodes); arm 6 tests head-camera labels in place of encoders; arms 5, 7 and 8 test the view: arm 5 is the wrist-camera UMI recipe [1] with a fisheye added to the glove for this ablation only, arm 7 a fixed camera (10 % fewer glove frames to body occlusion), and arm 8 head labels on the wrist view, separating the view from the labels.

Label accuracy. With the ball on the YAM and the encoders as reference, the chain places the tool point within 6.5 mm median (14.1 mm p90) and 1.55<sup>◦</sup> (2.70<sup>◦</sup>) over 5 065 frames. The jaw width, checked on 112 teleoperated episodes (24.6k frames), is within 3.5 % of stroke at the median (7.9 % p90), 2.3 % in the closed state that decides a grasp, with no lag and a label on every frame in 9 of 10 episodes; the error is flat over workspace, distance, elevation and viewing angle (2.8–4.8 % median per bin; App. B). A policy trained on these labels instead of encoders (arm 6 against arm 1), scored on the same encoder-labelled held-out episodes, loses 3 % in validation loss (0.328 against 0.318) and is within noise on the robot (15/25 against 18/25; 5 placements won to 8 lost, exact McNemar $p = 0 . 5 8 )$ .

Co-training and the teleoperation budget. Co-training across sources has precedent [22, 13]. Adding the 194 glove segments to 172 teleoperated episodes lowers the robot-validation loss from 0.318 to 0.263 (arm 2 against 1); added to 48 episodes, from 0.464 to 0.295 (arm 4 against 3). By log-linear interpolation of the teleop-only curve (Fig. A1) the glove set is worth 65–100 teleoperated episodes, 2–3 glove demonstrations per teleoperated one, and the 48-episode mix lands at the level of 96–172 teleoperated episodes; the gain shrinks with budget (0.075 at 24 episodes, 0.011 at 96), so glove data stretches a small teleoperated set rather than replacing the last few dozen. On the robot the 48-episode mix succeeds in 17/25 against 10/25 for 48 teleoperated episodes alone (9 placements won to 2 lost, $p = 0 . 0 6 5 )$ and 18/25 for all 172 (4 to 5). Every failure of both 48-episode arms is a missed grasp; the difference is recovery: in scene 5 the mix missed, backed off and re-grasped where the teleop-only policy never closed in.

Which camera. Offline the four co-trained views (arms 2, 5, 7, 8) sit within 0.011 of each other (0.252–0.263), and offline loss is a weak proxy for rollouts [23]. The rollouts separate them: head camera 20/25, wrist fisheye 15/25, fixed camera 12/25, head labels on the wrist view 14/25. On shared placements the head camera wins 6/1, 11/3 and 8/2 against them $( p = 0 . 1 3 , 0 . 0 6 , 0 . 1 1 )$ , so no pair clears 0.05 at 25 placements, but no arm beats arm 2 on more than three. The failure modes split by view: head-camera policies fail by never grasping, wrist-camera policies by dropping the block mid-carry (12 of the 21 failures of arms 5 and 8, against 1 of 45 for the head-camera arms, a slip at lift-off). In the operator’s notes every wrist-view drop came while the fisheye looked away from the bin and none with the bin in view (Fig. A3); the head camera keeps gripper and bin in one frame through the carry. Same view, other labels (arm 8 against 5): 14/25 against 15/25, indistinguishable. Same labels, other view (8 against 2): 20/25 falls to 14/25 and the no-grasps become drops. The fixed camera (arm 7) is the weakest and fails like the head view, by not grasping (12 of 13); it was mounted low because the positioner took the high vantage point, so its 10 % occlusion is an upper bound. Median time to success is 16–19 s for every arm: the views differ in whether the task completes, not in how fast.

## 4 Limitations and what is next

The tile layout is refined by median single-tile poses rather than a full bundle adjustment [24], which should tighten it further. A label needs a tile in view as well as the glove, so coverage depends on where the tiles are placed; we followed the placement rules of §2 but tried only one layout. The positioner lags 0.4 s and hits joint limits on wide sweeps; its viewpoint model is a linear fit on one session with no tuning against rollouts, and the jaws were often occluded by the robot’s wrist-camera bracket, which the head-camera system does not need and which stayed on only for the wrist-view ablations. Removing it is the cheapest next gain. The head camera sees the collector’s forearm and fingers, so glove-only policies do not transfer (validation 0.52 against 0.26 co-trained), as with the fixed-camera device. Controls not run: a higher fixed camera; teleoperation under a wrist-camera preview, since the operator watched the head view; a second seed, so the paired tests sit at $p \geq 0 . 0 6$ with a consistent ranking; and a second collector, as one person with one glove recorded every session. A workspace away from the robot needs no robot-side calibration: the tile-to-base link, fitted here with the robot holding the ball, can be left free per session as UMI-style pipelines do with relative trajectories; the collector only needs tiles taped down with one, preferably two, in view. Nothing is single-handed: a second glove with its own tag ids is read by the same camera, and one positioner serves both arms. Hardware [8] and tools [25] are MIT-licensed; the robot-side calibration and dataset tools are not yet released. Next are the bimanual version, multi-tile workspaces, in-thewild sessions and a second task.

## References

[1] C. Chi, Z. Xu, C. Pan, E. Cousineau, B. Burchfiel, S. Feng, R. Tedrake, and S. Song. Universal manipulation interface: In-the-wild robot teaching without in-the-wild robots. In Robotics: Science and Systems (RSS), 2024.

[2] C. Campos, R. Elvira, J. J. G. Rodr´ıguez, J. M. M. Montiel, and J. D. Tardos. ORB-SLAM3:´ An accurate open-source library for visual, visual-inertial, and multimap SLAM. IEEE Transactions on Robotics, 37(6):1874–1890, 2021.

[3] Zhaxizhuoma, K. Liu, C. Guan, Z. Jia, Z. Wu, X. Liu, T. Wang, S. Liang, P. Chen, P. Zhang, H. Song, D. Qu, D. Wang, Z. Wang, N. Cao, Y. Ding, B. Zhao, and X. Li. Fastumi: A scalable and hardware-independent universal manipulation interface with dataset. arXiv preprint arXiv:2409.19499, 2024.

[4] T. Ohkawa, J. Arima, Y. Noguchi, M. Tateno, M. Sugiura, T. Okubo, K. Ikeuchi, Y. Shin, H. Nishizawa, N. Kanazawa, Y. Wakayama, D. Fukunaga, K. Makihara, T. Motoda, F. Erich, Y. Domae, T. Matsushima, Y. Okumatsu, and K. Ota. Yubi: Yielding universal bidigital interface for bimanual dexterous manipulation at scale. arXiv preprint arXiv:2606.10244, 2026.

[5] vovw. tinyumi: a hand-worn, 3d-printable gripper based on yam-umi. https://github. com/vovw/tinyumi, 2026.

[6] A. Banaszczyk, M. Łysakowski, M. R. Nowicki, P. Skrzypczynski, and S. K. Tadeja. How´ accurate is the positioning in VR? using motion capture and robotics to compare positioning capabilities of popular VR headsets. arXiv preprint arXiv:2412.06116, 2024.

[7] A. Carnevale, I. Mannocchi, M. S. Hadj Sassi, M. Carli, G. De Luca, U. G. Longo, V. Denaro, and E. Schena. Virtual reality for shoulder rehabilitation: Accuracy evaluation of Oculus Quest 2. Sensors, 22(15):5511, 2022.

[8] Y. Shin. Yam-umi: hand-worn data collection device for the i2rt yam linear gripper. https: //github.com/YosubShin/yam-umi, 2026.

[9] K. Hsu, M. J. Kim, R. Rafailov, J. Wu, and C. Finn. Vision-based manipulators need to also see from their hands. In International Conference on Learning Representations (ICLR), 2022.

[10] AIRoA. The UMI arena: CoRL 2026 workshop. https://umi-arena.airoa.io/, 2026.

[11] M. F. Land and M. Hayhoe. In what ways do eye movements contribute to everyday activities? Vision Research, 41(25–26):3559–3565, 2001.

[12] E. Olson. AprilTag: A robust and flexible visual fiducial system. In IEEE International Conference on Robotics and Automation (ICRA), pages 3400–3407, 2011.

[13] S. Kareer, D. Patel, R. Punamiya, P. Mathur, S. Cheng, C. Wang, J. Hoffman, and D. Xu. EgoMimic: Scaling imitation learning via egocentric video. In IEEE International Conference on Robotics and Automation (ICRA), pages 13226–13233, 2025.

[14] I. Chuang, A. Lee, D. Gao, M.-M. Naddaf-Sh, and I. Soltani. Active vision might be all you need: Exploring active vision in bimanual robotic manipulation. In IEEE International Conference on Robotics and Automation (ICRA), pages 7952–7959, 2025.

[15] X. Cheng, J. Li, S. Yang, G. Yang, and X. Wang. Open-TeleVision: Teleoperation with immersive active visual feedback. In Conference on Robot Learning (CoRL), volume 270 of Proceedings ofMachine Learning Research, pages 2729–2749, 2024.

[16] I2RT Robotics. Yam 6-dof arm. https://i2rt.com/products/yam-6-dof-arm, 2026.

[17] S. Garrido-Jurado, R. Munoz-Salinas, F. J. Madrid-Cuevas, and M. J. Mar˜ ´ın-Jimenez. Auto-´ matic generation and detection of highly reliable fiducial markers under occlusion. Pattern Recognition, 47(6):2280–2292, 2014.

[18] P.-C. Wu, R. Wang, K. Kin, C. Twigg, S. Han, M.-H. Yang, and S.-Y. Chien. DodecaPen: Accurate 6DoF tracking of a passive stylus. In ACM Symposium on User Interface Software and Technology (UIST), pages 365–374, 2017.

[19] TheRobotStudio and Hugging Face. SO-ARM100 / SO-101: a low-cost 3d-printed robot arm. https://github.com/TheRobotStudio/SO-ARM100, 2025.

[20] T. Z. Zhao, V. Kumar, S. Levine, and C. Finn. Learning fine-grained bimanual manipulation with low-cost hardware. Robotics: Science and Systems (RSS), 2023.

[21] R. Cadene, S. Alibert, A. Soare, Q. Gallouedec, A. Zouitine, and T. Wolf. Lerobot: Stateof-the-art machine learning for real-world robotics in pytorch. https://github.com/ huggingface/lerobot, 2024.

[22] Z. Fu, T. Z. Zhao, and C. Finn. Mobile ALOHA: Learning bimanual mobile manipulation using low-cost whole-body teleoperation. In Conference on Robot Learning (CoRL), volume 270 of Proceedings ofMachine Learning Research, pages 4066–4083, 2024.

[23] A. Mandlekar, D. Xu, J. Wong, S. Nasiriany, C. Wang, R. Kulkarni, L. Fei-Fei, S. Savarese, Y. Zhu, and R. Mart´ın-Mart´ın. What matters in learning from offline human demonstrations for robot manipulation. In Conference on Robot Learning (CoRL), volume 164 of Proceedings ofMachine Learning Research, pages 1678–1690, 2021.

[24] B. Triggs, P. F. McLauchlan, R. I. Hartley, and A. W. Fitzgibbon. Bundle adjustment – a modern synthesis. In Vision Algorithms: Theory and Practice, volume 1883 of Lecture Notes in Computer Science, pages 298–372, 2000.

[25] Y. Shin. forward-cam-umi: fiducial labelling, recording and dataset tools for fixed and headworn cameras. https://github.com/YosubShin/forward-cam-umi, 2026.

[26] M. Shah. Solving the robot-world/hand-eye calibration problem using the Kronecker product. Journal ofMechanisms and Robotics, 5(3), 2013.

## A Details

Camera mounting and aiming. The Arducam B0587 sits on a \$10 GoPro head strap through a printed GoPro-pattern adapter (103 g worn, plus 37 g of USB cable). With the live preview open, the collector looks at a point 0.5–1 m away and tilts the mount until the centre of the frame coincides with the centre of their own view, so the camera looks where the head looks. The positioner carries the same pattern: the SO-101 follower’s gripper bracket is removed and its gripper servo reused as a wrist-yaw joint, so the one camera moves between head and robot without re-mounting.

Calibration. With the ball mounted on the YAM’s gripper, the flange-to-ball link is solved as a robot-world hand-eye problem (OpenCV’s calibrateRobotWorldHandEye, Shah [26]), and the tile-world-to-base link is a trimmed Kabsch fit of the head-camera tool point (tiles × ball) to forward kinematics over one 5 065-frame session (91 % inliers). The tile refit reduces four-tile reprojection error from 3.4 px to 0.5 px RMS and the disagreement between single-tile solves of the same frame from 10–13 mm to 4–9 mm. Single-tile frames are chained under velocity bounds of 0.6 m/s and 120 <sup>◦</sup>/s.

Positioner control. Forward kinematics gives the gripper position; a fixed-gain controller holds its initial visual bearing, and the residual model adds the head’s displacement relative to the hand. The model is a ridge regression on nine pose features plus the grasp bearing, a bearing history and the distance, fitted on about 100 glove episodes of one session; on 15 held-out episodes it predicts the collector’s head pose to 46 mm $/ 5 . 4 ^ { \circ }$ median (81 mm $/ 9 . 6 ^ { \circ } \ \mathsf { p 9 0 } )$ , against 87 mm $/ 7 . 0 ^ { \circ }$ for a camera held at the median head pose. Commands are low-pass filtered, rate-limited to $5 0 ^ { \circ } / \mathrm { s }$ and bounded to a safe volume.

## B Jaw-width accuracy by condition

On 112 teleoperated episodes (24.6k frames) the head-camera width is compared with the encoder value. Binned by workspace quartile, by distance to the camera (0.4–0.8 m), by the camera’s elevation $( 4 0 – 8 7 ^ { \circ } )$ and by the angle between the line of sight and the jaw axis, every bin reads 2.8–4.8 % of stroke median and 6.8–8.5 % p90, worst when the camera looks across the jaws rather than along them. Shifting the label by one frame in either direction raises the error, so there is no lag. The residual is a systematic under-read of $4 \mathrm { - } 6 \%$ at full opening, a scale error of the per-session fit (alternate-frame holdout) rather than noise. The fixed camera of Shin [8] read the same gripper at 4.4 % median with 94 % coverage.

## C Rollout protocol and figures

Each of the 25 placements was set once and re-placed to a saved reference image for every arm (ghost alignment), with the arm order shuffled per scene and a 60 s budget; the operator scored success, no grasp, dropped before the bin, dropped at the bin or timeout. In scene 5, arm 8 released the block mid-carry with the block outside its view, as if it no longer registered the object in hand; arm 5 shows the same pattern in its three drops. Head-camera arms that fail do so before contact, stopping short of the block or closing beside it, and then hold until the budget expires.

![](images/678e1148562faba4b6d7389300c322938d5a0f799f09ec4b55d9ac9d0c31a518.jpg)  
Figure A1: Robot validation loss against the number of teleoperated episodes, with and without the 194 glove segments. Solid: head-camera-labelled teleoperation (round-2 chain, four budgets); dashed: encoder-labelled teleoperation (final design, two budgets). The glove set moves every budget to the level of 2–4× as many teleoperated episodes.

![](images/b91f7b8395d4b7d3796822ab4cef6bf517ad6be8f4e4cf3fc2b5388c1fdd4d6f.jpg)  
Figure A2: The 25 placements over the fixed camera’s view. Each disc is the successes of the eight arms at that placement; the arrow points to the scene’s bin.

![](images/2d0bef6372d571136190a18ebecbad0e3fd453875fd942e525055ad9aab8a2a1.jpg)  
Figure A3: One placement (scene 16), the four camera arms, at 2, 6, 10 and 14 s, from the fixed workspace camera with each policy’s own view inset. During the carry the wrist view (rows 2 and 4) holds the block and the far wall and nothing else; the head view (row 1) keeps gripper and bin in one frame. Arm 8 opened with the bin out of its view.