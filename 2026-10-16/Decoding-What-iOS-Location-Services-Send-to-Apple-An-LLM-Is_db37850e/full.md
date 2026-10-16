# Decoding What iOS Location Services Send to Apple: An LLM Is an Unreliable Analyst but a Useful Toolmaker

D.J.Leith

Trinity College Dublin, Ireland

Abstract—Reverse engineering the data formats used by mobile software is skilled and slow, and using AI to assist with it is attractive. We evaluate this for the protobuf messages that an iPhone sends to Apple’s location services. Working from Apple’s binaries, we use two frontier models (Claude Opus 4.8, Codex GPT Sol 5.6) to reconstruct the schemas of protobufs observed in network traffic. The great majority of the schemas contain substantial errors, and these are hard to detect because the schemas are internally consistent. Direct AI use is therefore unsuitable for forensic and privacy analysis. AI can, however, write reliable analysis tools when their output can be validated. We use Claude to write a Python tool, Extractor, that reliably recovers protobuf schemas from Apple binaries. We validate it against binaries compiled from schemas with known answers, previously published schemas, and network measurements. We demonstrate the use of Extractor to decode and analyse the location services network traffic of a jailbroken iPhone 8 (iOS 16.7.11).

Keywords: Mobile OS Privacy, iPhone, Mobile Location Data, AI, Network Measurement

## I. INTRODUCTION

Mobile location data is sensitive. Knowing where a person is at night and during the day usually reveals where they live and work, and visits to a clinic, a place of worship or a protest can reveal health, belief and political activity. Handset collection of location data has repeatedly proved unexpected, as shown by the penalties and settlements imposed on Google over its location tracking on Android [1, 2]. Previous measurement studies of location data have mostly focussed on Android and on third-party apps [3, 4]. To our knowledge there is no previous systematic measurement study of the location data that iOS itself sends to Apple that decodes the message contents. Apple’s public documentation does not describe network connections, endpoints or message contents.

Measuring what a phone sends requires understanding what the data means. The traffic can be decrypted on a jailbroken handset, but it is binary-encoded protobuf for which the message schemas are not public, and so they must be recovered from the software that sends it. Apple’s code is mostly Objective-C (ObjC) compiled to ARM64. Decompilers give crude output, and in practice the analysis is manual inspection of assembly. Reverse engineering of this kind is skilled, labour intensive and slow, and it is a barrier to independent scrutiny of what phones send. Using AI to assist with it, or even to replace it, is therefore attractive. Whether this is viable matters for forensic and privacy analysis generally, and not only for the present study.

## Our main contributions are:

1. Direct AI reverse engineering is not viable. We use two frontier models (Claude Opus 4.8, Codex GPT Sol 5.6) to reconstruct the schemas of protobufs observed in our traffic. The great majority of the schemas contain substantial errors, and these are hard to detect because the schemas are internally consistent. Direct AI use is therefore unsuitable for forensic and privacy analysis.

2. AI-written analysis tools are viable when they can be validated. We use Claude to write a Python tool, apple\_protobuf\_extractor (“Extractor”), that recovers protobuf schemas from Apple’s binaries. We can make this reliable because we are able to validate the output. Namely, we reverse engineer the template that Apple’s protobuf classes follow, compile protobufs with known schemas to ARM64 with a compiler of the same era as the Apple code, and require Extractor to recover them exactly. We also check that (i) the recovered schemas decode our captured traffic to sensible values and (ii) they match previously published hand-decoded schemas. This validation needs reverse engineering and measurement of its own, so it is not free, but it is much less work than recovering the schemas by hand.

3. Measurements of the location data sent to Apple. Using Extractor to decode the traffic of an iPhone 8 (iOS 16.7.11), we find that: (i) no message carries a direct identifier of the user or handset<sup>1</sup>, (ii) with location on, the handset sends its precise position to Apple in reverse geocoding and Wi-Fi fingerprint lookups which carry the entry and exit times of visits, in Wi-Fi and cell lookups, and in batched uploads of GPS fixes and Wi-Fi and cell observations, (iii) usage telemetry carries session identifiers that persist for at least 106 days and so link a handset’s records over months, and (iv) with location off, the MAC address of the connected Wi-Fi access point, a network identifier, is still sent, which locates the handset to tens of metres if Apple knows where that access point is. Apple’s documentation does not mention (iii) or (iv).

Extractor (plus associated tool ios\_decoder and the extracted location services protobuf descriptors) are available on GitHub at https://github.com/doug-leith/ios-protobufdecoding. Our AI analysis data (Section III), including the prompts used, and our network measurement data (Section V) are available on GitHub at https://github.com/doug-leith/ioslocation-services-study-data.

The rest of the paper is organised as follows. We review related work, then evaluate direct use of two LLMs, Claude and Codex, for reverse engineering and describe Extractor and its validation. We then apply Extractor to Apple’s location services: we describe the experimental setup and the architecture of the services, and analyse the location data sent, the sensor data uploads, the telemetry and the identifiers used, before comparing our observations with Apple’s public documentation. We end with the limitations of our measurements and our conclusions.

## II. RELATED WORK

LLMs for reverse engineering. LLMs have been applied to decompilation [5] and to recovering types and symbols from stripped binaries [6], including Objective-C method signatures in macOS frameworks [7]. Benchmarks report that the best agents solve only 27% to 32% of realistic binary-analysis tasks [8], and that decompiled code can pass tests yet diverge in behaviour [9], consistent with our finding that errors are hard to spot.

Recovering protobuf schemas. Existing tools recover schemas from binaries that embed a serialised descriptor [10, 11], or infer them from traces [12], and neither applies to Apple’s PBCodable classes. We are not aware of prior work that recovers protobuf schemas from compiled PBCodable code and checks them against compiled ground truth.

Network traffic analysis. A number of large-scale measurement studies have used network traffic analysis to detect transmissions to tracking services by Android apps, and to investigate the data transmitted by mobile apps. These use rooted phones and a man-in-the-middle attack to intercept and decrypt HTTPS traffic leaving the phone. Typically, apps are installed from the Google Play store, run for a short period with no user interaction, and the network traffic is recorded and analysed for transmission of sensitive information such as location and personal identifiers. See, for example, [13, 14, 15, 16, 17, 18, 19, 20, 21].

Mobile OS-level telemetry. While there has been a fair number of measurement studies on app telemetry, mostly telemetry sent by third-party SDKs, there has been much less work on data transmitted by the mobile OS itself. The earliest work on Android appears to be [16] with follow-on studies by [17, 18, 19]. The earliest work on Apple iphones appears to be [17], with more recent work focussing on the Apple App Store [22].

Apple location services. Apple’s Wi-Fi positioning service has been reverse engineered from captured traffic and used to look up access point locations [23], and Rye and Levin show that it can be used to learn the locations of over 2 billion access points [24]. A repository that cites that paper [25] adds protobuf field names recovered from CoreLocationProtobuf, tools for the tile and cell-tower lookups, and an informal technical report. The report records traffic from locationd, geod and geoanalyticsd, but most payloads are not decoded and it reaches conclusions about the data sent without recovering the message formats. We are not aware of a systematic measurement of what these daemons send that decodes the payloads, nor of one that covers geod and the CoreRoutine traffic. On-device forensic work documents what iOS stores about location [26, 27] (the CoreRoutine databases, and retention of Significant Locations), which is the on-device counterpart of the uploads that we observe.

Anonymity of location data. Removing identifiers does not make a location trace anonymous: a few coarse spatiotemporal points can single out most individuals [28, 29, 30]. We therefore report the session and network identifiers, as well as the location accuracy.

## III. LLM REVERSE ENGINEERING IS TOO LOSSY

## A. Evaluation Setup

We evaluate how accurately two frontier LLMs (Claude Opus 4.8, Codex GPT Sol 5.6) reconstruct Apple PBCodable protobuf descriptors directly from Mach-O disassembly.

Starting from one seed protobuf message class, the LLM is asked to recover the name, wire type, number and label (optional/required/repeated) for every field, and to recurse into every protobuf message- or enum-typed field until every reachable type has been recovered. An orchestrating Python script assigns recovery of each message- or enumtype to a fresh sub-agent with its own clean context window<sup>2</sup>. The prompt instructs the LLM to leave a field marked // UNRESOLVED: <reason> rather than guess when disassembly genuinely does not reveal something, and to keep a field as bytes if that is its genuine wire type rather than invent a message type for it. The script maintains a running record of message types already reconstructed, so each message- or enum-type is only reconstructed once even if it occurs multiple times within the seed message and its submessages.

The evaluation is blind. Each sealed LLM runs in a sandbox that denies it the ground-truth .proto files, any prior session transcript, and web access. It can use only the disassembly tool and read only the raw dyld shared cache. Grading happens afterwards, outside the sandbox, by diffing the LLM’s reconstructed protobuf descriptor field-by-field against the schema produced by our independently validated Extractor tool, which the LLM never sees during the run.

TABLE I: Fields recovered correctly (number, type and name).
<table><tr><td rowspan=1 colspan=1>Field type</td><td rowspan=1 colspan=1>Fields</td><td rowspan=1 colspan=1>Claude</td><td rowspan=1 colspan=1>Codex</td></tr><tr><td rowspan=1 colspan=1>Scalar</td><td rowspan=1 colspan=1>4,151</td><td rowspan=1 colspan=1>3,575 (86.1%)</td><td rowspan=1 colspan=1>3,283 (79.1%)</td></tr><tr><td rowspan=1 colspan=1>Enum: right enumall values correct</td><td rowspan=1 colspan=1>993993</td><td rowspan=1 colspan=1>807 (81.3%)775 (78.0%)</td><td rowspan=1 colspan=1>602 (60.6%)575 (57.9%)</td></tr><tr><td rowspan=1 colspan=1>Message: right messagewhole subtree correct</td><td rowspan=1 colspan=1>2,6442,644</td><td rowspan=1 colspan=1>2,337 (88.4%)1,528 (57.8%)</td><td rowspan=1 colspan=1>1,923 (72.7%)1,374 (52.0%)</td></tr></table>

## B. Results

As seed messages we use 17 GeoServices and CoreLocation-Protobuf request and response protobufs observed in network traffic. Each seed is analysed separately, so a message or enum type that occurs in several seeds is recovered afresh in each, and its fields are counted once per seed. The 17 seeds contain 7,788 fields (4,987 distinct) in 1,627 message types. Of these fields, 4,151 have a scalar type (integer, boolean, floating point, string or bytes), 993 an enum type and 2,644 a message type.

We compare each LLM descriptor with Extractor’s by following field numbers outward from the seed message, as a decoder does, rather than by matching type names. A field is correct if the LLM descriptor has a field with the same number at the same place in the message tree (that is, reached from the seed message by the same sequence of field numbers), with the same type and the same name. For a scalar field the type must be identical, including signedness and encoding (int32, uint32, sint32, fixed32), since these decode the same bytes to different values. For an enum- or messagetyped field we report two levels: the field refers to the right enum or message, and, more strictly, every value of that enum, or every field of that message and of all messages below it, is also correct.

Where an LLM and Extractor disagree we checked the binary (the writer function each field’s writeTo: calls, the instance variable (ivar) type encoding and the enum value-string tables) by a mixture of scripted and manual analysis. Of the 413 such disagreements in which the LLM gives a type, Extractor is found to be correct. We also picked 50 fields at random where Claude, Codex and Extractor agree and manually analysed them. In every case the manual analysis also agreed.

1) Analysis Of The Errors: Table II breaks down the observed errors in more detail. This shows that many of the errors are due to fields being “cut off”. That is, an error higher up on the path from the seed message makes everything below it unreachable. A handful of errors is responsible for the cutoffs, e.g. Claude’s placeholder for PDPlace.requestData cuts off 697 fields, and Codex’s wrong field numbers in PDPlaceRequest 1,725.

Table II also shows that the two models fail in different ways.

• Claude. Most of its errors at the field itself are integer signedness, mainly int32 for uint32. These decode identically until a value reaches 2<sup>31</sup>. It also leaves 29 message-typed fields as opaque bytes. For 52 enum fields the type name resolves to nothing: most are short, scope-relative names such as Type that are never defined; the rest are placeholder names.

TABLE II: What a decoder does with the bytes of each field that is not correct (counts are fields, each counted once per seed it occurs in). Bold rows group the errors by consequence, with an example; the indented rows give the type of error behind each. Apart from descriptors that do not load, and the occasional message whose bytes no longer parse, every error decodes without complaint.

<table><tr><td></td><td>Claude</td><td>Codex</td></tr><tr><td>Descriptor does not load: it names a type it never defines</td><td>67</td><td>24</td></tr><tr><td>e.g. Codex gives CoarseLocationMetadata.type an enum</td><td></td><td></td></tr><tr><td>type Type that it never defines</td><td></td><td></td></tr><tr><td>Enum type never defined</td><td>52</td><td>24</td></tr><tr><td>Message type never defined</td><td>13</td><td>0</td></tr><tr><td>Type of a scalar field never defined</td><td>2</td><td>0</td></tr><tr><td>Value silently dropped, or left as raw bytes</td><td>38</td><td>61</td></tr><tr><td>e.g. Claude types the latitude LatLngE7.1atE7</td><td></td><td></td></tr><tr><td>(sfixed32) as int32, so a decoder skips it Message left as bytes</td><td></td><td></td></tr><tr><td>Field omitted</td><td>29</td><td>1</td></tr><tr><td>Incompatible wire type</td><td>1</td><td>52</td></tr><tr><td>Enum and message confused</td><td>5</td><td>4</td></tr><tr><td>Wrong value or wrong enum name</td><td>3</td><td>4</td></tr><tr><td>e.g. Claude types</td><td>56</td><td>31</td></tr><tr><td>CdmaCellNeighbor.pnoffset (sint32)</td><td></td><td></td></tr><tr><td>as int32, so —5 decodes as 9 A different enum</td><td></td><td></td></tr><tr><td>sint read as int</td><td>24</td><td>24</td></tr><tr><td>A different message</td><td>15</td><td>0</td></tr><tr><td></td><td>10</td><td>4</td></tr><tr><td>Same encoding, different meaning (e.g.</td><td>3</td><td>1</td></tr><tr><td>double for sfixed64) Strings or bytes read as packed integers</td><td></td><td></td></tr><tr><td>Integer read as an enum</td><td>3</td><td>0</td></tr><tr><td>Wrong only when the top bit is set (signed vs</td><td>1 128</td><td>2 3</td></tr><tr><td>unsigned) e.g. Claude types</td><td></td><td></td></tr><tr><td>ChargerPlug.maximumPower (uint32) as int32, so 3,000,000,000 would decode as -1,294,967,296</td><td></td><td></td></tr><tr><td>Right value, label lost or wrong e.g. Codex types MotionActivity.type as</td><td>50</td><td>136</td></tr><tr><td>int 32, so a number is shown instead of the activity name</td><td></td><td></td></tr><tr><td>Enum read as a plain integer</td><td>6</td><td>90</td></tr><tr><td>Wrong field name</td><td>28</td><td>46</td></tr><tr><td>Placeholder field name</td><td>16</td><td>0</td></tr><tr><td>Wrong at the field itself (all of the above)</td><td>339</td><td>255</td></tr><tr><td>Cut off by an earlier error: never decoded, or</td><td>730</td><td></td></tr><tr><td>decoded under the wrong schema</td><td></td><td>1,725</td></tr><tr><td>e.g. Codex swaps fields 1, 2 and 8 of</td><td></td><td></td></tr><tr><td>PDPlaceRequest, so none of the 34 captured</td><td></td><td></td></tr><tr><td>requests parses</td><td></td><td></td></tr><tr><td>Recovered but on the wrong field, correct there</td><td>4</td><td>1,680</td></tr><tr><td>Recovered but on the wrong field, wrong there</td><td>0</td><td></td></tr><tr><td></td><td></td><td>43</td></tr><tr><td></td><td></td><td></td></tr><tr><td>Never recovered</td><td>726</td><td>2</td></tr><tr><td>Total not correct (of 7,788 fields)</td><td>1,069</td><td>1,980</td></tr></table>

• Codex. It types 90 enum fields as plain integers, which decode the number but lose its meaning, and omits 50 enum fields altogether.

Within enums that are otherwise identified correctly, Claude misnumbers 224 values in 11 enums and Codex 197 values in 7. For example, Claude omits value 35 (DIRECTIONS) of DailyUseSummaryUseEventType and shifts the 161 values after it down by one. Codex swaps values 3 and 12 of MapRegionSourceType. Every such value decodes from the wire to the wrong name.

![](images/b74f770c106dfb01f82bae10d558af12147764c6c485114df9b7b38c98d018e8.jpg)  
Fig. 1: Cumulative share of field-level errors against cumulative share of fields, with message types sorted descending by number of errors. Cut-off fields are excluded; the random curves are the mean of 2,000 random placements of the same errors. Half the errors lie in message types holding about a tenth of the fields, about twice as concentrated as random placement.

A further 28 fields (Claude, plus 16 with placeholder names) and 46 (Codex) have the right type but the wrong name, e.g. Codex names the two fields of Attribution that hold an attribution’s name and URL logo and logoChecksum.

Neither model adds spurious field numbers to a message it identifies correctly. Both add types that are not part of the seed:

• Claude. 24 types (115 fields). 22 of these (113 fields) are real GeoServices types reached through a wrong reference. For example, it substitutes the 30-field routeattributes message for the WiFiESS.attributes enum, together with the options messages below it, and the licence-plate region types (GEOLPRRegion and others, which it names Region, RegionInfo and so on) for the tile-region structure in Resources. One is an empty message standing in for an enum. One, a 2-field TileType, has no counterpart in the binary.

• Codex. Four types (8 fields). Three are real types and one is an empty message standing in for an enum.

Apart from Claude’s 2-field TileType and the two empty stand-ins, every type the models add exists in the binary, but belongs elsewhere in the schema.

2) Errors Tend To Concentrate: The errors are concentrated in part of the schema, but not in a few messages. Figure 1 shows that half of them lie in message types holding about a tenth of the fields, roughly twice as concentrated as if the same number of errors had been placed at random. Even so, they touch some 200 message types, most of them with a single error. A single error makes every message above it incorrect, so these errors break far more subtrees than they touch directly.

3) Variance Across Seed Protobufs: Figure 2 shows the error rates for each of the seed protobufs. Each seed is analysed independently of the others, so this gives an indication of the level of variability that can be expected from using an LLM to reconstruct a protobuf. The reconstruction rates generally vary between about 85% and 100% but can be much worse, e.g. Claude’s GEOTileGroup is entirely wrong (Claude types all nine of the message’s uint32 fields as int32 and replaces its three sub-message types with placeholders).

4) Decoding Real Traffic: We also decoded real traffic with the recovered descriptors and compared the result with Extractor’s decoding of the same bytes. We collected 1,057 instances of the seed messages from the iPhone 8 captures, including instances nested inside other messages, for 15 of the 17 seeds. The two exceptions:

• GEOMapServiceTraits is held only by on-device objects, such as the XPC request between an app and geod, so it never appears on the network.

• GEOMuninVersion occurs only inside a part of Resources that none of the captured payloads contains.

Real traffic exercises only a small part of each schema. Of the 7,407 fields in these 15 seeds, only 806 (10.9%) ever occur in the captures. On those 806 fields the recovered descriptors are more accurate than over the whole schema: 94.2% correct for Claude and 95.9% for Codex, against 85.8% and 73.7% over all fields of the same seeds. Errors therefore concentrate in rarely used fields, and testing a recovered descriptor against captured traffic would miss almost 90% of the fields it gets wrong.

Where errors do fall on fields used in practice, the decoder reports nothing, but the decoded data is wrong:

• Lost data. In Claude’s descriptor for GEOResources, two fields refer to the wrong message type (GEOResource.region to the licence-plate Region message and GEOTileGroup.tileSets to TileSet). Decoding the eight captured Resources payloads silently loses 70% of the decoded values and leaves a further 6% as undecoded bytes.

• Wrong values. Claude types CLPLocation.appBundleIdIndices, a list of strings, as a list of integers, so the string "0" decodes as the number 48. Codex types the photo count photosAvailable as a boolean.

• Parse failure. The only error that announces itself is Codex’s PDPlaceRequest field-number error, whose descriptor fails to parse any of the 34 captured messages.

5) Compute Cost: As well as being error prone, using AI to recover protobuf descriptors is computationally expensive. Recovering the 17 seed descriptors took 2,078 sub-agents for Claude and 2,333 for Codex, one for each message or enum type the model dispatched. Figure 3(a) shows how the cost of one sub-agent depends on the size of the message type it recovers. There is a fixed cost of around 0.48 million tokens (Claude) and 0.28 million (Codex), plus a cost that grows with roughly the square root of the number of fields<sup>3</sup>.

![](images/75c4cb244fc7db664152c6bd373fc1b88a031f4ed703cf4079366da1cb33f30b.jpg)  
Fig. 2: Outcome for every field of each seed, as a percentage of the seed’s fields (field count in brackets). 12 (Claude) and 14 (Codex) of the 17 seeds are at least 90% correct; each model’s losses are dominated by one error high in the message tree that cuts off a large subtree (Claude: GEOPDPlaceResponse; Codex: GEOLogMessageCollectionRequest).

Figure 3(b) shows that total tokens per seed protobuf rise almost in proportion to the number of message and enum types in the seed (fitted slope 0.90 for Claude and 0.96 for Codex), at a median of 0.82 million tokens per type for Claude and 0.52 million for Codex. The two largest seeds, GEOPDPlaceResponse (925 reachable types) and GEOLogMessageCollectionRequest (1,172), account for 80% of Claude’s tokens and 85% of Codex’s.

Some seeds lie well above the dotted lines because a model dispatched more sub-agents than the seed has types: a wrong type reference pulls in types that do not belong to the seed, and each is recovered by its own subagent. For example, GEOTileGroup reaches four types. Codex dispatched four sub-agents for it. Claude dispatched 16, 15 of them for types outside the seed, reached through its three placeholder group\_ types. Similarly, GEOWiFiQualityServiceResponse reaches 33 types. Claude’s substitution of the route-attributes message for the WiFiESS.attributes enum pulls in that message and the 27 types below it, and its use of the GEOTimeRange class for an inline time-range structure adds one more, so 29 of Claude’s 51 sub-agents for this seed recovered types outside it. Errors therefore raise cost as well as lowering accuracy.

We do not count 94 Claude sub-agent attempts (2.9 million tokens) and three Codex attempts that failed with infrastructure errors and were retried.

Although wall clock time is only an indirect measure of compute cost, to put the token counts in context, recovering the 17 seed protobufs took roughly five days of wall-clock time. In comparison, Extractor recovers the schemas of the whole GeoServices and CoreLocationProtobuf images (1,805 message types, which include the 17 seeds and every type they reach) in about 2 minutes (126 s and 4 s) on one core of an Apple M3 Max laptop, with no model calls. Its output is identical to the reference schemas used above.

6) Choice Of Prompt: The above results are for a recursive prompt, where each message type is decoded by its own agent. We also investigated decoding each seed protobuf using a single agent. However, we found this gave far worse performance. With the single-agent approach, only 696 of the 7,788 fields (8.9%) are correct, mostly because the agent stops before reaching most of the nested messages (6,360 fields are never recovered), and only 59% of the fields it does reach are correct. This may be due to the single agent exhausting its context window.

## C. Discussion

1) Direct AI Analysis Is Too Unreliable: This data indicates that direct protobuf reconstruction by Claude and Codex is far too unreliable for forensic and privacy analysis applications. Not only are the error rates too high, but the great majority of the errors cannot be easily caught since the protobuf descriptors generated are internally consistent: correct protobuf syntax, plausible field names, a type that compiles and decodes real messages. Nothing in the descriptor itself flags the error. In our captures only one error, Codex’s field numbers in PDPlaceRequest, makes decoding fail; every other error decodes without complaint, and almost 90% of the wrongly recovered fields never occur in the captured traffic at all. Catching these errors requires an external reference such as the ground-truth schema, or some other independent evidence.

![](images/5379d36a341e0f8f8a1ea0ec4453ed7bdba57cf421cc5221c171ec23d5e61c96.jpg)

![](images/d4b849beb0c64d4ef7b22e873dd564471ec4f3f127185640e22b92b3b5da73dc.jpg)  
Fig. 3: Tokens used to recover descriptors. (a) One point per sub-agent that recovered a message type, against the number of fields Extractor gives that type; lines are medians within size bins and bands the interquartile range. Cost per sub-agent grows only with about the square root of the number of fields, from a large fixed cost. (b) One point per seed, against the number of message and enum types reachable from the seed; dotted lines are tokens proportional to the number of types. The cost of a seed is set by how many types it reaches.

2) Fair Comparison: Regarding independent evidence, Apple does not publish its ground-truth schemas. With hindsight, after development of Extractor, we can create our own groundtruth schemas and associated binaries. But these are a byproduct of the detailed reverse engineering carried out while building Extractor, and so they are not “free”: giving them to the direct-AI agents would hand them the tool’s output and would not yield a fair comparison. Similarly, the network measurements reported later are another source of external evidence that could be fed back to the agents. However, these were not used during the development of Extractor, only for its subsequent validation. Additionally, as already noted above, their coverage of the protobuf fields, and so the feedback they could provide, is limited.

## IV. PROTOBUF EXTRACTOR TOOL

While using AI to analyse binaries directly to extract protobuf schemas is too error prone, AI can nevertheless be used to develop a reliable Python tool to carry out this work. In this section we describe such a tool, apple\_protobuf\_extractor (“Extractor”). Extractor is a 21,000-line Python tool, with a comparably sized test suite, built entirely by Claude. None of the code or tests is humanauthored.

## A. From Descriptor To Assembly: A Worked Example

![](images/ca6c3e2c27520954169966170511370724d4cd4aa3cf55d6fbec8daba15a315e.jpg)  
Fig. 4: Example protobuf message, TestMessage.

![](images/0123b67afca9cf9d63c1fbf84cb86ead3e83be713df682779577dde0e062054c.jpg)  
Fig. 5: ObjC runtime representation of TestMessage.

Consider the protobuf message in Figure 4. Figure 5 shows how this would be implemented by Apple as an ObjC class<sup>4</sup>. This class has one ivar per field, a \_has bitfield struct recording which optional fields are set, and a writeTo: method that calls a field-writing helper, once per set field, passing the field number as a literal argument. A production Apple binary carries only the compiled result of code such as this. The compiled binary has no field names, no field numbers and no indication that a given ivar even corresponds to a protobuf field. When compiled, the method above becomes the code shown in Figure 6.

The recovery task is to invert this. From the raw instruction stream alone, the tool must identify: (a) that \_first/\_second/\_third are protobuf fields at all, distinguished from scratch or internal state by the ivar-offset idiom (ldrsw from a named \_OBJC\_IVAR\_\$\_Class.\_ivar symbol); (b) each field’s name (from which ivar-offset feeds the load immediately before the call); (c) each field’s number (from the mov w1,#N/mov w2,#N immediately preceding the bl, not from source or declaration order: fields 5, 2, 9 above are deliberately non-monotonic, to show that a tool that used ivar-declaration-order numbering would fail); (d) each field’s type (from which PBWrite<sub>\*</sub>Field helper is called).

```asm
ldrsw x21, [x8, #ivar_off(_has)]
ldrb w8, [x20, x21] ; w8 = _has bitfield byte
tbnz w8, #0, L_first ; has_first?
tbnz w8, #1, L_second ; has_second?
tbz w8, #2, L_epilogue ; !has_third -> done
L_third:
ldrsw x8, [x8, #ivar_off(_third)]
ldr s0, [x20, x8] ; s0 = self->_third (a float, not x1)
mov x0, x19 ; x0 = writer
mov w1, #0x9 ; w1 = field number
; (float/double only: w1, not w2)
bl _PBWriteFloatField
b L_epilogue
L_first:
ldrsw x8, [x8, #ivar_off(_first)]
ldr x1, [x20, x8] ; x1 = self->_first (an NSString )
mov x0, x19
mov w2, #0x5
bl _PBWriteStringField
ldrb w8, [x20, x21]
tbz w8, #1, L_third ; re-test has_second/has_third
b L_second
L_second:
ldrsw x8, [x8, #ivar_off(_second)]
ldr w1, [x20, x8] ; w1 = self->_second (a plain int32)
mov x0, x19
mov w2, #0x2
bl _PBWriteInt32Field
ldrb w8, [x20, x21]
tbnz w8, #2, L_third
b L_epilogue
```  
Fig. 6: Compiled -Os ARM64 for the writeTo: method in Figure 5 (prologue, epilogue and retain/release housekeeping omitted).

Extractor implements this by walking the method’s controlflow graph from entry, recording every PBWrite<sub>\*</sub>Fieldfamily call site reached, and binding, for each one, a field number to the ivar name that fed the value load.

## B. Complications Beyond The Simple Case

1) A Full Control-Flow-Graph Walk Is Needed: The example above already shows why a textual scan is insufficient: the compiler does not lay the three write blocks out in source order, and does not reach a given block only by falling straight through to it. A tool scanning top-to-bottom for mov w1,#N/mov w2,#N; bl PBWrite<sub>\*</sub>Field would still find all three calls in a function this small, but in physical-address order (9, 5, 2), with no principled way to confirm each call is genuinely reachable, or to exclude a similarly-shaped but dead call left over elsewhere in the binary. Extractor therefore performs an explicit reachability walk over the method’s basic blocks from its single entry point, so every recovered field write is guaranteed reachable, and later analysis stages can reuse the same walk to resolve values that depend on which block a load appears in, not merely that it appears somewhere in the function body.

2) Repeated (Message-Typed) Fields: A repeated Msg field = N field is stored as a plain, untyped array. Nothing about the array itself records what kind of message its elements are. A class states this fact in one of two other places, and the tool must find whichever one a given class actually uses:

1. A small helper method the class defines purely to advertise its element type. This is the easy case: the tool just reads off what class the method returns.

2. readFrom:: the element type has to be inferred from readFrom:, the method that parses an incoming byte stream back into an object (the reverse of writeTo:). readFrom: routes each incoming field to per-field handler code, and the tool follows that same routing to see which other class’s parsing method a given field’s handler calls.

The second, harder route might seem the natural one to look for, but it is the less common one in practice. A third, rarer route resolves the element type by tracing which of the class’s own array-adding methods the routing logic calls directly. It is used when neither of the above applies.

3) Distinguishing Plain Ints From Sint/Fixed Variants: An ObjC int might be encoded on the wire as an int32/int64, a zigzag-encoded sint32/sint64, or a fixed-width fixed32/fixed64. ObjC storage does not distinguish these, so the tool works it out from the write call itself, but the details depend on where the binary lives.

PBDataWriter, the class that builds the wire bytes, gives each wire-type family its own method, e.g. writeInt32:, writeFixed32:. In a standalone binary, writeTo: calls these as Objective-C messages, so the method name says the type outright: no disassembly is needed, just read the selector. Inside the dyld shared cache (where most Apple frameworks reside), the same call is implemented as an unnamed C function call. In this case, the tool follows the call and reads the first few instructions of whatever it finds to decide on the wire type, e.g. a zigzag encoder shifts the value left by one bit before anything else.

4) Enum Values: Another Lookup, In One Of Several Shapes: An enum field’s value names, like the field names themselves, are not stored as data. Many classes have a small companion method, named after the field itself (e.g. a status field’s values are named by a statusAsString: method), that takes the stored integer and returns the matching name string. In this case, the tool recovers the field’s full number-to-name table by decoding that method. The compiler lays this lookup out differently depending on how the values are distributed, so the tool needs a different decoder for each shape:

1. Densely packed values (e.g. 0 to 7): the compiler builds a small jump table, one entry per value, each pointing at a tiny block of code that returns that value’s name. The tool walks the table block by block.

2. Widely spaced values (e.g. 20, 30): rather than a table, the compiler writes a sequence of individual “is the value 20? is it 30? . . . ” checks. The tool finds and reads each check in turn.

3. Both at once: a single field can use both shapes for different parts of its range, so full recovery needs both decoders to run on the same field and combine their results.

4. No companion method: some fields have no dedicated method like statusAsString:. Fortunately, Apple’s runtime also gives every class a separate debugging method that builds a dictionary of every field’s current value, by name, and so the tool reads the enum names there instead. The dense/sparse split above still applies.

## C. Program Architecture

The codebase splits into two layers. macho/ is a generalpurpose Mach-O and ARM64 binary-analysis layer with no protobuf-specific knowledge: memory.py maps a loaded image (a standalone binary or a slice of the dyld shared cache) and resolves chained-fixup pointers;<sup>5</sup> arm64decode.py decodes individual raw instruction words (movz, adrp, cmp, ldr, conditional branches, and so on); disasm.py iterates an instruction stream; classlist.py and strings.py enumerate ObjC classes, ivars, methods and string tables from a binary’s metadata sections; dyld\_cache.py, local\_symbols.py and cache\_classlist.py handle shared-cache-specific image lookup and symbolication, used only for cache-native runs.

proto/ is the protobuf-recovery layer, built on top of macho/, summarised in Table III.

## D. Code Development

The full AI session transcripts, the complete commit history and AI-authored process artefacts (a running to-do list, a heuristics registry, its own limitations note) generated during the development of Extractor are available at https://github.com/doug-leith/extractor-code-development. The development process, including the AI errors made during it and the reliability of the AI’s claims about its own work, is analysed in [31].

## E. Testing

1) Using Ground Truth From A Reverse-Engineered Template: Apple protobuf schemas are not publicly available, and so we cannot verify Extractor output by comparison against them. Nevertheless, we can still validate Extractor’s output as follows.

Apple’s ObjC protobuf implementation uses a separate class for each protobuf message and follows a fixed template as illustrated in Figure 5, repeated across every class. Having reverse-engineered that template closely enough to reproduce it, we can author our own protobuf descriptors with a known schema, write the equivalent ObjC, compile it to an ARM64 Mach-O binary, and run Extractor on it. Because the schema was authored by us, any mismatch between the recovered output and the authored ground truth is a tool defect. By testing against a variety of protobuf descriptors we can exercise Extractor’s code thoroughly (see below). Descriptors are chosen to (i) trigger a specific code shape, facilitating testing, e.g. an 8-field message compiles to a jump-table dispatch, a smaller one compiles to a compare chain, and (ii) reproduce Apple’s field and enum names and values.

The compiler used also affects the code shapes observed in the ground-truth tests. Using Apple clang 11.0.0 (build clang-1100.0.33.8, bundled with Xcode 11.1) with the -Os optimisation flag reproduces the branch and jump-table shapes seen in the iPhone 8 binaries. For the more recent OS versions Apple clang 15.0.0 (clang-1500.3.9.4, Xcode 15.4) reproduces the patterns seen in Apple binaries<sup>6</sup>.

The Extractor test suite therefore consists of two kinds of tests. The first are ground-truth tests against binaries derived from known protobufs, and the second are regression tests against Apple binaries. The latter are regression tests because the true protobuf schemas are unknown, and so the tests can only confirm the tool’s output has not changed, not that it was correct to begin with.

Table IV shows the statement and branch coverage of the ground-truth tests. Note that much of Extractor’s codebase by line count is either standard disassembler components not specific to the task, or plumbing that the synthetic fixtures are not meant to exercise. The data in Table IV is therefore restricted to: (i) the modules that implement the protobufspecific recovery work, and (ii) within each module to the functions that actually decode instructions or Mach-O structures into a recovered name, number, type or table entry (excluding validation, defensive and orchestration code). The code doing the actual binary analysis is well exercised by the ground-truth tests, with overall statement coverage of 83% and branch coverage of 71%.

2) Using Manually Reverse-Engineered Descriptors: In addition, Extractor’s output is compared against Apple protobuf descriptors from iCloud, iMessage and iOS location services already manually reverse engineered and published by other researchers (see Section VIII-A1 in the Additional Material for details). In summary, the field numbers agree wherever both define a field, and in every case where the published descriptor disagrees with Extractor on a field’s type, meaning or structure, the iPhone’s binaries agree with Extractor.

3) Using Network Measurements: We take two more steps to check Extractor’s accuracy and robustness.

Firstly, Extractor is coded defensively, configured to fail loudly and to always flag uncertainty to the user, e.g. if it returns a field’s type as bytes because it is unsure about its real type. For all of the 889 (message, field) pairs observed being transmitted over the network (see below), Extractor flags no uncertainty<sup>7</sup>, 15 fields are unknown but none is an Extractor miss (12 are newer fields declared in iOS 27, one is a field of an opaque pass-through class, and two are fields in server responses that neither the iPhone 8 nor the iOS 27 binaries declare).

Secondly, the extracted protobuf descriptors are used to decode serialised Apple protobufs observed being transmitted over the network, and we confirm that (i) decoding succeeds, which confirms that the descriptor field wire types are compatible with the observed protobuf, and (ii) the decoded field values are consistent with the extracted field names, e.g. a latitude field has a reasonable value given a phone’s known location. Of the 889 (message, field) pairs observed, 297 are numeric or string fields confirmed to have values consistent with the field name, 98 are enums with values consistent with the expected range, 76 are booleans (trivially consistent), 313 are sub-message containers, and 105 have no applicable check.

TABLE III: proto/ module responsibilities.
<table><tr><td rowspan=1 colspan=1>Module</td><td rowspan=1 colspan=1>Responsibility</td></tr><tr><td rowspan=1 colspan=1>protocore.py</td><td rowspan=1 colspan=1>ivar type encoding to proto scalar/message type and label</td></tr><tr><td rowspan=1 colspan=1>writeto.py</td><td rowspan=1 colspan=1>writeTo: CFG walk, field number/name binding (Section IV-B1)</td></tr><tr><td rowspan=1 colspan=1>dispatch.py</td><td rowspan=1 colspan=1>readF rom: field-dispatch decoding, used for repeated-element typing (Section IV-B2)</td></tr><tr><td rowspan=1 colspan=1>repeated.py</td><td rowspan=1 colspan=1>repeated-field element type/ivar resolution (Section IV-B2)</td></tr><tr><td rowspan=1 colspan=1>enums.py</td><td rowspan=1 colspan=1>both enum-recovery mechanisms across dense/sparse/hybrid shapes (Section IV-B4)</td></tr><tr><td rowspan=1 colspan=1>write_helpers.py</td><td rowspan=1 colspan=1>classifies a writeTo: helper&#x27;s wire-type family from its tag-building instructions (Section IV-B3)</td></tr><tr><td rowspan=1 colspan=1>extractor.py</td><td rowspan=1 colspan=1>orchestrator:  walks theclasslist,  invokes  the  modules  above  per  class,  assemblesProtoMessage/ProtoField records and assigns each field number&#x27;s confidence tier</td></tr><tr><td rowspan=1 colspan=1>models.py</td><td rowspan=1 colspan=1>plain data classes (ProtoField, ProtoMessage) shared across modules</td></tr><tr><td rowspan=1 colspan=1>emitter.py</td><td rowspan=1 colspan=1>renders recovered records as . proto text</td></tr><tr><td rowspan=1 colspan=1>config.py, defaults.py</td><td rowspan=1 colspan=1>class name to proto package/prefix mapping</td></tr><tr><td rowspan=1 colspan=1>enums_debug.py</td><td rowspan=1 colspan=1>opt-in diagnostic sink (-debug-enums), dormant unless explicitly enabled, recording why a field&#x27;s enumrecovery failed</td></tr><tr><td rowspan=1 colspan=1>protoclass_heuristic.py</td><td rowspan=1 colspan=1>a retired heuristic class detector, kept only as a fail-loud safety net if a class is ever routed through it insteadof the current structural detector</td></tr></table>

TABLE IV: Statement and branch coverage of the specialised recovery code.
<table><tr><td rowspan=1 colspan=1>Module</td><td rowspan=1 colspan=1>Recovers</td><td rowspan=1 colspan=1>Statement coverage</td><td rowspan=1 colspan=1>Branch coverage</td></tr><tr><td rowspan=1 colspan=1>protocore.py</td><td rowspan=1 colspan=1>ivar type/label decoding</td><td rowspan=1 colspan=1>89.2% (33/37)</td><td rowspan=1 colspan=1>87.5% (21/24)</td></tr><tr><td rowspan=1 colspan=1>dispatch.py</td><td rowspan=1 colspan=1>readF rom: field-dispatch decoding for repeated-element typing(Section IV-B2)</td><td rowspan=1 colspan=1>88.4% (274/310)</td><td rowspan=1 colspan=1>78.5% (135/172)</td></tr><tr><td rowspan=1 colspan=1>writeto.py</td><td rowspan=1 colspan=1>writeTo: field number/name binding (Section IV-B1)</td><td rowspan=1 colspan=1>84.0% (247/294)</td><td rowspan=1 colspan=1>70.1% (143/204)</td></tr><tr><td rowspan=1 colspan=1>write_helpers.py</td><td rowspan=1 colspan=1>wire-type family classification (Section IV-B3)</td><td rowspan=1 colspan=1>83.0% (44/53)</td><td rowspan=1 colspan=1>73.3% (22/30)</td></tr><tr><td rowspan=1 colspan=1>enums.py</td><td rowspan=1 colspan=1>enum value-table recovery (Section IV-B4)</td><td rowspan=1 colspan=1>80.2% (482/601)</td><td rowspan=1 colspan=1>67.0% (233/348)</td></tr><tr><td rowspan=1 colspan=1>repeated.py</td><td rowspan=1 colspan=1>repeated-field element resolution (Section IV-B2)</td><td rowspan=1 colspan=1>77.9% (127/163)</td><td rowspan=1 colspan=1>67.6% (69/102)</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>82.8% (1207/1458)</td><td rowspan=1 colspan=1>70.8% (623/880)</td></tr></table>

4) iOS And macOS Versions Tested: Extractor has been tested on dyld shared caches from an iPhone 8 (iOS 16.7.11; cache UUID f203ff1d-3ce6-3f7a-9849-791b1c1f4b8b), macOS 14.7 (UUID 8047f785-c408-34e1-a0a8-38ee34ab89f4), and the recent versions iOS 27.0 (UUID 9416075f-d579-30bc-a976- 2b2b9a39270e) and macOS 27.0 (UUID ea2c265e-297c-39c2- 8646-7d8a2dff648a).

## F. Other Protobuf Implementations

In addition to Apple’s ObjC PBCodable implementations, Extractor recovers schemas for swift-protobuf (including Apple’s InternalSwiftProtobuf and its newer packed name-map format) and for the full C++ libprotobuf runtime, and detects, but does not decode, Google’s ObjC GPBMessage, C++ libprotobuf-lite and protobuf-c. All of the messages in the location services network traffic we decode are PBCodable.

## V. APPLYING EXTRACTOR TO APPLE LOCATION SERVICES

## A. Experimental Setup

1) Hardware And Software Used: We use an Apple iPhone 8 running iOS 16.7.11<sup>8</sup>, jailbroken using Palera1n v2.2.1<sup>9</sup> and with the Ellekit<sup>10</sup> Objective-C function hooker. We intercept network connections using mitmdump 13.0.0.dev. We bypass SSL certificate pinning within handset processes using a modified version of SSL KillSwitch<sup>11</sup>.

2) Test Protocol: The test protocol used is similar to that in [22]. In summary, we carry out experiments with a privacyconscious but busy/non-technical user, who when asked does not select options that share data with Apple but otherwise leaves handset settings at their default value. We use a two-step process to jailbreak the phone and allow decryption of HTTPS traffic. Firstly, we (i) jailbreak the phone using Palera1in with the Sileo package manager, (ii) use Sileo to download and install Ellekit and openssh, (iii) connect to the handset via ssh and install the Objective-C injection script used to bypass SSL certificate pinning. Then we insert a SIM with no credit (data access is disabled), factory reset the phone and the user goes through the onboarding process without using a network connection, selecting the settings shown in Table XIII (Additional Material). We jailbreak the phone again using Palera1in. Ellekit, openssh and the SSL bypass script are all left installed by the factory reset and become activated again after this second jailbreak without the need for a network connection. The user connects to the test Wi-Fi network and installs the mitmproxy CA cert as a trusted certificate by downloading it from the special URL http://mitm.it supported by mitmdump, enabling it in the phone settings and setting it to be trusted. At this point system HTTPS network connections can now be intercepted and successfully decrypted, and we record all handset network traffic.

The rest of the experiment is divided into three parts. In the first part the user is not logged in to an Apple account and the handset location setting is off. In the second part the handset location setting is turned on and the user logs in to an Apple account. In the third part, the phone SIM is replaced with one that has credit (so data connectivity is active), the Apple App Store is used to install the OpenVPN app and the handset settings are configured to send all network traffic via the VPN. The handset is then taken on four car commuter journeys, namely morning and evening for two days. The Apple Maps app is briefly opened to confirm the reported handset location is accurate, and then closed.

3) Decrypting HTTPS Connections: In the first two parts of the experiment we route handset traffic via a Wi-Fi access point (AP) that we control, configure this AP to use mitmdump as a proxy [32] and adjust the firewall settings to redirect all Wi-Fi HTTP/HTTPS traffic to mitmdump so that the proxying is transparent to the handset. When a process running on the handset starts a new network connection, the mitmdump proxy pretends to be the destination server and presents a fake certificate for the target server. This allows mitmdump to decrypt the traffic. It then creates an onward connection to the actual target server and acts as an intermediary, relaying requests and their replies between the handset process and the target server while logging the traffic. In the third part of the experiment, handset traffic is routed via a VPN server which is similarly configured to use mitmdump.

System processes typically carry out checks on the authenticity of server certificates received when starting a new connection and abort the connection when these checks fail. This happens even when the mitmproxy CA cert is installed on the handset as a trusted certificate. To bypass these checks, the iPhone is jailbroken, Ellekit is installed and a custom Objective-C injection script is used to bypass SSL certificate pinning within handset processes.

4) ios\_decoder: Extractor recovers the protobuf descriptors from Apple binaries. We developed a second tool ios\_decoder that applies them to network captures: it reads a saved mitmproxy capture offline and for each connection prints the decoded HTTP request and response.

In more detail, within ios\_decoder a table maps each connection, identified by host and URL path, to its format. For a protobuf payload the entry gives the message name<sup>12</sup>, and the descriptor recovered by Extractor is used to decode the body with the standard protobuf library. The entry also gives the framing, e.g. the length-prefixed ARPC envelope that wraps many Apple requests. Other payloads are decoded by format: property lists, JSON, map tiles and so on. HTTP headers and URL query parameters are decoded in the same way, with the values the table identifies as coordinates or tile numbers converted to the area they denote.

Of the 1,500 geod and locationd connections we observe, 42% carry protobuf bodies. These include all of the location uploads, the telemetry, A/B experiment and dispatcher exchanges, wloc lookups and the Wi-Fi quality lookups and tiles. The remainder are mainly tile and resource downloads. 5) Measurement Data & Additional Material: Tables XIV and XV (Additional Material) detail the network connections that we observe being made by geod and locationd, together with their estimated purpose and a summary of their content.

## B. Architecture Of Apple Location Services

There is no public Apple documentation on the internal operation of its location services, and only very limited prior reverse engineering work has been published. The following is based on analysis of the geod, locationd and GeoServices binary code (class and string names, selective disassembly of stripped binaries) combined with decoding of observed network transmissions (the network traffic analysis is covered in detail later and only summarised here).

Location services on iOS are centred on two system daemons, geod and locationd, which make almost all the network connections analysed in this paper, and GeoServices, a large private framework (about 27 MB of code and 2,597 Objective-C classes) that both daemons link to. See Figure 7.

geod is accessed via XPC and acts as a caching gateway between clients and Apple’s geo servers. It serves map tiles, other map resources (stylesheets and icons), place data and place images from its caches where it can, and makes network requests on the client’s behalf where it cannot. Other geod services that can make network requests include reverse geocoding, point-of-interest lookup, routing directions, ETA and transit updates and Wi-Fi quality lookups. geod provides a logging funnel, batching and uploading usage and log messages that clients report to it. It also maintains shared local state and logic, including: GeoServices configuration, country code (determined from the SIM or a geo-IP lookup), the A/B experiment assignment, request counters/throttling and offline map subscriptions. geod contacts some Apple servers without a client request, including the geo\_manifest server, and can preload tiles.

Only a few network endpoints used by geod are hard-coded in GeoServices. These include: (i) the defaults plist URL configuration.ls.apple.com/config/defaults, (ii) the geo\_manifest URL<sup>13</sup> and (iii) the geo-IP country lookup URL gspe1-ssl.ls.apple.com/pep/gcc. The others are supplied by the responses from the geo\_manifest and defaults endpoints.

![](images/888964a7d51b36e106777718b903cc5192db14600ab8c0b876634f88aa8a71eb.jpg)  
Fig. 7: Architecture of iOS location services. GeoServices (purple) is a single private framework loaded into locationd, geod and every process that uses it. Processes reach locationd through CoreLocation and geod through GeoServices, which MapKit and CoreLocation’s geocoder also use. Thin arrows are XPC, thick arrows are HTTPS, and dashed black arrows are hand-offs between the daemons. In our captures two processes bypass geod and connect to the geo servers themselves (dashed amber arrow).

locationd hosts the phone location providers (GNSS, Wi-Fi, cell, BLE) and is accessed by apps via the public Core-Location framework, which communicates with locationd over XPC. It also performs a range of other on-device processing, including geofencing, significant-location and visit monitoring, motion classification, health and fitness metrics (e.g. VO max, gait, fall detection), and Find My tracking avoidance. It downloads data from Apple for its own use, including: (i) tiles of cell identifiers and Wi-Fi MAC addresses with their locations from cl2.apple.com, presumably so that the phone location can be estimated on the handset from the cells and access points it observes, (ii) elevation and radio signal environment tiles from cl3.apple.com and (iii) GNSS assistance data (satellite orbit predictions) from iphone-ld.apple.com, which it sends to the GPS chipset. It can also transmit a Wi-Fi MAC address, cell identifier, or the handset location to Apple server gs-loc.apple.com/clls/wloc, which responds with the locations of nearby Wi-Fi networks and cell towers. locationd also reports (“harvests”) phone sensor data to Apple servers, either itself or, for the real-time traffic probe, through geod (Section V-D).

The clients of geod are system daemons, services and extensions that run without user action. They do not contact Apple’s geo servers themselves: their GeoServices code passes requests to geod over XPC, and geod sends them to Apple. The clients are: routined, “a per-user daemon that learns historical location patterns of a user and predicts future visits to locations”<sup>14</sup>, which apparently takes its locations from locationd and reaches geod through its XPC helper service com.apple.CoreRoutine.helperservice (“CoreRoutine” for short); the Weather widget; analyticsd, which “collects diagnostics and usage data locally for users that have opted in”; networkserviceproxy, a transparent network proxy for Apple system services; kbd, the keyboard daemon; the CoreLocation permission-prompt plugins; and com.apple.wifi.ThreeBarsXPCService, a prefetch and cache service for Wi-Fi network quality data. We infer the purpose of the clients without manual pages from their binaries.

Two further processes are not geod clients for network requests: they connect to Apple’s geo servers directly. photoanalysisd, the photo library analysis agent (it supports Memories, People and photo search), sends reverse geocoding and point-of-interest lookups. Each carries a precise location, which in our captures is the handset position. The request code is in the MediaMiningKit framework, which links GeoServices, and the server URLs come from geod’s geo\_manifest. photoanalysisd reports its requests to geod’s telemetry. mapspushd, the Maps application services daemon, fetches Maps announcements and sends no location. It is a client of locationd.

## C. Analysis Of Location Data Sent

The following location data is observed being sent to Apple by geod, locationd and photoanalysisd:

1. Coarse location is sent by both geod and locationd when location is on. Tile downloads identify the area around the handset to within about 1 to 5 km for the finest tiles.

2. Precise location is sent by geod, locationd and photoanalysisd when location is on. locationd sends Wi-Fi MAC addresses and cell identifiers for lookup and uploads GPS locations accurate to about 5 m. geod sends the real-time traffic probe (see below) and, on behalf of CoreRoutine, lookups that carry a precise location. photoanalysisd sends precise locations directly to Apple’s geo servers.

3. Uploads are delayed. Apart from the real-time traffic probe, which is sent within a minute, uploads are sent hours after the data was collected, and up to about 2 days after.

4. Switching location off removes most location data but not the MAC address of the connected access point. The Wi-Fi quality service sends it even when location is off, which locates the handset to tens of metres if Apple holds the position of the access point. The location that geod sends for its other clients varies (see Table V).

![](images/a1b7c0f05b4b48f9ebf59a799722596f6dc97a90c9e2d5daf58a3edb554e154d.jpg)  
(a) Tile styles 1 and 11

![](images/cd585d8bd25700b5050aae55ca24629d4b79ab2666ce234d200e329353b90aa0.jpg)  
(b) Tile styles 20 and 68

![](images/5f83fa0bb7cc09f2295c495c64341ae0abb92b73b78e6fbf5ecd3e0541d4557c.jpg)  
(c) Tile style 62

![](images/4007dcc078f5199bdefc362b33ce29feadb5f16bedf5e3e1ba616e2dd779a296.jpg)  
(d) Wi-Fi tiles  
Fig. 8: Visualisation of highest zoom-level tiles of each style downloaded by geod (the style is specified in the download request (inspection of the GeoServices code maps the style number to enum values e.g. style=1 is VECTOR\_STANDAR D, style=11 is VECTOR\_BUILDINGS). The star in each plot marks the handset location.

1) Coarse Location: Both geod and locationd download tiles of map, Wi-Fi, cell and other data, and each request identifies an area. geod names a tile by its X, Y and zoom Z values in the standard Web Mercator scheme, which divides the Earth into $2 ^ { Z } \times 2 ^ { Z }$ tiles. A tile is about 10 km wide at zoom 12, 5 km at zoom 13 and 1 km at zoom 15 at the equator; about 60% of that at the latitude of Ireland.

locationd instead puts the latitude and longitude of the tile’s south-west corner in the URL path. In cl2.apple.com/1/v1 /143/173/1433500\_1738000.gz, the value 1433500 equals (latitude+90)\*10000 and 1738000 equals (longitude+180)\*10000, giving latitude 53.35<sup>◦</sup> and longitude −6.20<sup>◦</sup>. Its Wi-Fi and signal environment tiles are about 5.6 km by 3.3 km in size, its elevation tiles about 11 km by 7 km, and its GSM and LTE cell tiles about 220 km by 130 km.

Figure 8 and, in the Additional Material, Figure 12 show the smallest tiles that each process downloads, with the handset location marked by a star. When a single tile is downloaded it contains the handset (Figure 8(c)), so its location is revealed to an accuracy of about the tile width. When several tiles are downloaded they are centred on the handset (Figures 8(a),(b),(d) and 12(a),(b)), so averaging their positions locates it more accurately than the tile width<sup>15</sup>.

With location off, geod still downloads map tiles but only at zoom levels 2 and 3, each covering thousands of km, which reveals no more than the location that can be inferred from the handset IP address.

2) Location Sent For geod Clients: geod makes requests for several clients, and each request names its client. Table V shows the location carried by each client’s requests, and by the direct requests of photoanalysisd and mapspushd. Apart from CoreRoutine, the Wi-Fi service and photoanalysisd, the location sent is coarse or absent. The Wi-Fi service sends no coordinates, but the MAC address of the connected access point locates the handset to tens of metres if Apple holds the position of the access point.

TABLE V: Location sent to Apple’s geo servers, by client. The first seven clients send through geod, which names the client in each request. The last two, marked direct, connect to the servers themselves.

<table><tr><td rowspan=1 colspan=1>Client</td><td rowspan=1 colspan=1>Request</td><td rowspan=1 colspan=1>Location sent</td></tr><tr><td rowspan=1 colspan=1>Weather widget</td><td rowspan=1 colspan=1>Reverse geocoding</td><td rowspan=1 colspan=1>City-level (one fixed point sev-eral kilometres from the handset;location-off experiment only)</td></tr><tr><td rowspan=1 colspan=1>analyticsd</td><td rowspan=1 colspan=1>Reverse geocoding</td><td rowspan=1 colspan=1>Coordinates rounded to 1 to 2decimal places (about 1 to 10 km)</td></tr><tr><td rowspan=1 colspan=1>network-serviceproxy</td><td rowspan=1 colspan=1>Reverse geocoding</td><td rowspan=1 colspan=1>Reported accuracy about 3.7 km</td></tr><tr><td rowspan=1 colspan=1>kbd</td><td rowspan=1 colspan=1>Geocoding</td><td rowspan=1 colspan=1>Postal district, no coordinates</td></tr><tr><td rowspan=1 colspan=1>CoreLocation  promptplugins</td><td rowspan=1 colspan=1>Dataset status check</td><td rowspan=1 colspan=1>None (IP address only)</td></tr><tr><td rowspan=1 colspan=1>Wi-Fi         service(ThreeBars-XPCService)</td><td rowspan=1 colspan=1>Wi-Fi quality lookup</td><td rowspan=1 colspan=1>MAC address of the connectedaccess point, no coordinates</td></tr><tr><td rowspan=1 colspan=1>CoreRoutine</td><td rowspan=1 colspan=1>Reverse geocoding;Wi-Fi   fingerprintlookup</td><td rowspan=1 colspan=1>Precise (reported accuracy 5.6 to30 m in the lookups that stateone); also nearby access pointsand visit times</td></tr><tr><td rowspan=1 colspan=1>photoanalysisd(direct)</td><td rowspan=1 colspan=1>Batch reverse geocod-ing; spatial lookup</td><td rowspan=1 colspan=1>Precise (7 decimal places; handsetposition in our captures); searchradius 1 to 20 km</td></tr><tr><td rowspan=1 colspan=1>mapspushd(direct)</td><td rowspan=1 colspan=1>Announcements fetch</td><td rowspan=1 colspan=1>None</td></tr></table>

3) Locations Identified By Wi-Fi Or Cell Identifiers: locationd sends a Wi-Fi MAC address or a cell tower identifier to gs-loc.apple.com/clls/wloc (wloc for short), and the response lists nearby access points or towers and their positions (Figure 9; Figure 13 in the Additional Material shows a cell example). Provided that Apple knows the position of the access point or tower, the identifier locates the handset to tens of metres (Wi-Fi) or hundreds of metres (cell). We observe 37 wloc requests, all with location on. One request instead carries the handset’s precise latitude and longitude and the response lists 400 nearby access points. The Wi-Fi quality lookup (gsp85- ssl.ls.apple.com/wi fi\_request), sent by geod for the Wi-Fi prefetch service com.apple.wifi.ThreeBarsXPCService, sends the MAC address of the access point that the handset is using, and we observe this even with location off.

4) Precise Location: We call a location precise when its coordinates have not been coarsened, which is Apple’s term for the opposite of an approximate location. Precise locations are sent in all five sensor data uploads (Section V-D), in the wloc request that carries the handset position and in two kinds of lookup request that geod makes for CoreRoutine (all observed only with location on):

![](images/f95adaf20a2f18bdc6468246e8aef34c25d5267344d685734a2c392a83782893.jpg)  
Fig. 9: Example showing transmission of a Wi-Fi MAC address to Apple server gs-loc.apple.com/clls/wloc.

• Reverse geocoding requests to gsp57-ssl-revgeo.ls.app le.com/dispatcher.arpc, sent by geod, carry a precise location, and the response includes the street address (Figure 10). The request names CoreRoutine as the client.

• Wi-Fi fingerprint lookups to gsp57-ssl-locus.ls.apple.co m/dispatcher.arpc, sent by geod for CoreRoutine, carry 94 locations, the MAC addresses and signal strengths of nearby access points and the entry and exit times of a visit. The purpose of the request is not clear.

![](images/b81d430e73ed402194ac1eb297d0c1fabe4eba9dd7bbc537dd56414e0bb21687.jpg)  
Fig. 10: Example showing transmission of a precise lat/lon location to Apple server gsp57-ssl-revgeo.ls.apple.com/dispa tcher.arpc.

5) Location Off: During the 27 hours of the experiment with location off, locationd made 7 connections (GNSS assistance data and Bluetooth configuration) and sent no location. geod made 299 connections, including configuration and resource downloads, telemetry, A/B experiment assignment, map tile downloads at zoom 2 and 3, one weather widget lookup with a city-level location and 97 Wi-Fi quality lookups. Each Wi-Fi quality lookup sends the MAC address of the access point that the handset is connected to. wloc lookups, locationd tile downloads, higher zoom geod tile downloads and all uploads occur only with location on. Switching location off therefore limits the location sent to city level, apart from the MAC address of the connected access point.

## D. Apple’s “Harvesting” Of Phone Sensor Data

locationd collects data from the handset sensors and uploads it to Apple servers, either itself or, for the real-time traffic probe, through geod. The locationd code refers to this collection as “harvesting”. In brief:

1. A wide range of sensor data is collected. This includes cell and Wi-Fi observations, GPS locations with satellite measurements, pressure, temperature, compass attitude, step counts and motion state.

2. The user is not asked about it. No consent is requested during setup, and declining to share iPhone Analytics does not stop it.

3. We observe it only in the drive experiment, and not always with an app. No uploads occur with location off, nor in the static experiment with location on. The GPS uploads (/pds/pd, rtloc) cover only the minutes when the Apple Maps app was open and name it as the app using location. One trace upload (/hvr/trc), however, contains GPS fixes that name no app and inspection of the locationd code suggests that it can start GNSS for harvesting without an app requesting location<sup>16</sup>.

4. Data is held and sent in batches. Uploads are up to 13 hours old when sent, apart from the real-time traffic probe sent by geod, which is sent within a minute.

Table VI lists the uploads that we observe. The data is paired with the handset location, which has accuracy from 5 m (GPS) to 400 m (cell and Wi-Fi) for most records and up to 2.6 km for some in the trace uploads. Uploads include the handset hardware model and OS build and, in some, the identifier of the app using location and a tripId value that links them (Section V-F), but we did not observe any user or device identifiers.

The rtloc upload, sent by geod, carries 66 locations at 1 s intervals with accuracy about 5 m and the motion state (Figure 14 in the Additional Material). The locations are those that locationd also uploads in /pds/pd, which carries 99 locations over 7.5 minutes with the bundle identifier of the app using location (Figure 11). Code inspection indicates that locationd passes the locations to GeoServices, which hands them to geod for upload. The tripId value is the same in both uploads, so they can be linked (Section V-F).

Collection and transmission are decoupled. The rtloc upload follows the last location by 42 s. The other uploads are later: observations in pbcwloc are up to 12 hours old, /pds/pd and /hvr/trc 13 hours<sup>17</sup>, and the CoreRoutine Wi-Fi fingerprint lookups are about 2 days after the visit. Apple therefore receives a history of where the handset has been, not only its current position. We have not established what triggers each upload.

TABLE VI: Sensor data uploads observed. All but rtloc are sent by locationd.
<table><tr><td rowspan=1 colspan=1>Endpoint</td><td rowspan=1 colspan=1>Data</td><td rowspan=1 colspan=1>Uploads observed</td></tr><tr><td rowspan=1 colspan=1>gsp10-ssl.apple.com/hcy/pbcwloc</td><td rowspan=1 colspan=1>LTE cell (MCC, MNC, TAC, cell ID, band, signal strength) and Wi-Fi (MAC address,channel, signal strength, scan time) observations, each with the handset location andmotion state</td><td rowspan=1 colspan=1>5 in two batches 12 hours apart, data upto 12 hours old</td></tr><tr><td rowspan=1 colspan=1>gsp10-ssl.apple.com/pds/pd</td><td rowspan=1 colspan=1>GPS locations with altitude, accuracy, speed, course and motion state, app identifier,tripId</td><td rowspan=1 colspan=1>1, with 99 locations 13 hours old</td></tr><tr><td rowspan=1 colspan=1>gsp9-ssl.apple.com/hvr/v2/rtloc (sent bygeod)</td><td rowspan=1 colspan=1>As /pds/pd</td><td rowspan=1 colspan=1>1, with 66 locations 42 s old</td></tr><tr><td rowspan=1 colspan=1>gsp10-ssl.apple.com/hvr/trc</td><td rowspan=1 colspan=1>Pressure, temperature, locations with GNSS satellite reports (satellite identifier, signalstrength, azimuth, elevation), compass attitude, pedometer, motion state, app identifier</td><td rowspan=1 colspan=1>2, with 317 and 864 events, up to 13 hoursold</td></tr><tr><td rowspan=1 colspan=1>gsp10-ssl.apple.com/psr</td><td rowspan=1 colspan=1>Barometric pressure with the GPS fix at the time of the reading</td><td rowspan=1 colspan=1>3, one with an empty body</td></tr></table>

![](images/6a7a6e009d02bf0ebe21d289f966823bf76e4aab6ed008d139bab9abc6404700.jpg)  
Fig. 11: Example showing transmission of GPS fixes, with the handset motion state, to Apple server gsp10-ssl.apple.com/p ds/pd. The two fixes shown are also sent to gsp9-ssl.apple.c om/hvr/v2/rtloc (Figure 14).

Presumably Apple uses the data to build and refresh its Wi-Fi, cell, traffic and barometric databases, but we have not established this. The user is not asked about this collection during setup (Table XIII in the Additional Material), and the iPhone Analytics option, which the test user set to “Don’t Share”, does not prevent it. We did not test whether switching off a setting under Location Services (for example Routing and Traffic) stops it, so we cannot say whether any opt out exists short of switching location off, which our measurements suggest stops it.

## E. Telemetry & Experiments

geod sends usage records to Apple and receives an A/B experiment assignment from it. In brief:

1. Telemetry is sent with location off and with iPhone Analytics declined. The test user chose “Don’t Share” at setup.

2. The records describe each request that geod makes and the client that made it.

3. A daily record carries an identifier that persists for months.

4. The handset is assigned to A/B experiments, and the bucket identifier is echoed to Apple in later requests.

1) Telemetry: geod sends batches of log records to gsp64-ssl.ls.apple.com/hvr/v3/use. The request is a protobuf and the response is an empty acknowledgement. We observe 88 connections in the 27 hours with location off, 52 in the 16 days with location on and the handset stationary, and 37 in the drive experiment.

Most records describe a network request made by geod. Each gives the network service used (for example manifest, resource download, A/B assignment, Wi-Fi quality or place data), the application that requested it, its duration, the request and response sizes, the HTTP status or error code, the server IP address and port and the tile set identifier. The requesting application identifier can include the calling class and method. For example, a CoreRoutine request to identify the place of a visit from a Wi-Fi fingerprint is recorded with the method name. Records also carry the handset model, OS version, country code, carrier name, connectivity type (Wi-Fi or cellular) and whether cellular data is allowed. The network event records contain no location.

Once a day geod also sends a daily use summary and a device sync status record, which the protobuf labels FIFTEEN\_MONTH\_SESSION and FIFTEEN\_MONTH\_USER\_SESSION. Both carry session IDs (Section V-F), and the labels suggest that Apple retains the records for 15 months. We did not observe daily records in the 27 hours with location off.

The other telemetry records also carry a session ID (see Section V-F).

2) A/B Experiments: geod sends a request to gsp-ssl.ls.apple.com/ab.arpc containing a random guid value, the locale, OS version and country code, and receives an A/B experiment bucket identifier for each of four experiment namespaces (Parsec, Siri, RAP and

Maps). The bucket identifier is the same for all four. The response contains a refresh interval of 4 hours (most of the requests that we observe are 4 hours apart) and a branch expiration time of 30 days, presumably the lifetime of the assignment. The handset echoes the bucket identifier to Apple in dispatcher.arpc requests, presumably so that Apple can attribute behaviour to an experiment group.

## F. Analysis Of Identifiers Used

We distinguish three kinds of identifier. A “direct identifier” identifies the individual person or handset, e.g. an Apple Account ID or a device serial number. A “session identifier” is a random value created by the handset’s software and reused across records for a period of time, so that records can be associated with one another. Examples below include the telemetry session IDs, the A/B guid and the tripId. A “network identifier” comes from the network the handset is using and not from its software, e.g. the IP address or the MAC address of a nearby Wi-Fi access point. It potentially links connections for as long as the handset stays on that network. 1) Direct Identifiers: A direct identifier sent alongside data often makes that data “personal data”<sup>18</sup>. We did not observe any direct identifiers in geod or locationd connections, including when the user is logged in to an Apple account.

2) Session Identifiers: A session identifier is more complicated since its impact depends, amongst other things, on: (i) the data that it links together, and (ii) the lifetime of the identifier (which affects the duration over which data is linked together). For example, linking location data together over several days might easily identify an individual, e.g. from where they spend the night (which shows where they live) and where they spend the day (which shows where they work). A short-lived session identifier, e.g. one that persists for a few seconds or minutes, is generally much less of a concern than a long-lived identifier which persists for days or months. Additionally, we might be concerned about whether a session identifier can be easily linked to a direct identifier, e.g. if the two are sent in the same network connection. However, no direct identifiers are sent in geod or locationd connections and so this is not a concern here.

Table VII lists the identifiers that we observe in geod or locationd connections.

Session IDs in the daily records. The daily device sync status record lists two session IDs, the sessionId and preSyncSessionId, both created on 19 February at 04:31 UTC (createTime). Both are still in use in the records sent on 4 and 5 June, 106 days later. The FIFTEEN\_MONTH\_SESSION summary carries a third ID that is unchanged from 19 February to 5 June. We do not know when these values change, but the record labels suggest that Apple retains the records for 15 months.

Session IDs in telemetry. Each record in geod’s usage telemetry (Section V-E) carries a session ID and its age (ageOfSessionIdInSeconds). The IDs come from 11 processes<sup>19</sup> and several persist for over two weeks: those of geod, the Wi-Fi quality service and one permission-prompt plugin are each used in records sent over 16 to 17 days.

TABLE VII: Identifiers observed in geod and locationd connections and whether Apple’s documentation describes them. Lifetimes are measured from the times at which records are sent, and from the session age field in the telemetry.
<table><tr><td rowspan=1 colspan=1>Identifier</td><td rowspan=1 colspan=1>Lifetime observed</td><td rowspan=1 colspan=1>Documented</td></tr><tr><td rowspan=1 colspan=1>Daily record sessionIDs</td><td rowspan=1 colspan=1>At least 106 days</td><td rowspan=1 colspan=1>No; the 15-monthlabel is not men-tioned</td></tr><tr><td rowspan=1 colspan=1>Session ID in geodtelemetry</td><td rowspan=1 colspan=1>20 IDs across 11 processes. Max-imum age per ID 3 to 29 days.Used over 16 to 17 days bygeod, the Wi-Fi quality serviceand a prompt plugin</td><td rowspan=1 colspan=1>No</td></tr><tr><td rowspan=1 colspan=1>dispatcher.arpcsessionId</td><td rowspan=1 colspan=1>10 IDs across the clients, noneseen for more than 10 seconds</td><td rowspan=1 colspan=1>No</td></tr><tr><td rowspan=1 colspan=1>A/B guid</td><td rowspan=1 colspan=1>At least 1.8 days; a differentvalue in the drive experiment</td><td rowspan=1 colspan=1>No</td></tr><tr><td rowspan=1 colspan=1>A/B bucketId</td><td rowspan=1 colspan=1>Assigned by server: 7428 from17 February to 6 March, 12296on 4 and 5 June</td><td rowspan=1 colspan=1>No</td></tr><tr><td rowspan=1 colspan=1>tripId</td><td rowspan=1 colspan=1>One trip</td><td rowspan=1 colspan=1>Yes, rotates whenthe trip ends</td></tr><tr><td rowspan=1 colspan=1>MAC address of con-nected access point</td><td rowspan=1 colspan=1>Same address in all 184 Wi-Fiquality lookups (17 February to 6March); also seen in three otherendpoints up to 5 June, 108 days</td><td rowspan=1 colspan=1>No</td></tr><tr><td rowspan=1 colspan=1>MAC addresses ofneighbouring accesspoints</td><td rowspan=1 colspan=1>Sent  in  wloc  lookups,pbcwloc uploads and theCoreRoutine Wi-Fi fingerprintlookups; six access points occurin both an upload and a lookupsent up to 12 hours apart</td><td rowspan=1 colspan=1>No</td></tr><tr><td rowspan=1 colspan=1>IP address</td><td rowspan=1 colspan=1>Not recorded (Apple sees it inevery connection)</td><td rowspan=1 colspan=1>Yes, for approxi-mate location</td></tr></table>

Session IDs in dispatcher.arpc requests. Each client sends a session ID in its dispatcher.arpc requests. For the system processes it differs from their telemetry session ID. We observe 10 such IDs. Each is shared by requests made close together (the three Wi-Fi fingerprint lookups share one) and none is seen for more than 10 seconds, but we have too few requests to establish how long they last.

A/B experiment identifiers. The guid value sent to gsp-ssl.ls.apple.com/ab.arpc is random. We observe one value in all 12 requests of the first two parts of the experiment (17 to 19 February) and a different value in the 7 requests of the drive experiment (4 to 5 June), so it can link requests to that endpoint over at least 1.8 days. The A/B experiment bucketId (Section V-E) is assigned by the server and so is probably shared by many handsets, but we cannot establish this from measurements on a single handset. The service tag in dispatcher.arpc requests is a random UUID that differs for each connection.

Trip identifier. The tripId value is a random UUID sent in both the rtloc upload by geod and the /pds/pd upload by locationd made 13 hours later, so it links the transmissions of the two processes even when there is a delay between uploads. We observe a single trip. Inspection of the locationd code indicates that it creates a new value when no location has been collected for 15 minutes, or when a trip lasts more than 3 hours<sup>20</sup>.

While the telemetry records contain no location, they can potentially be matched to the requests that do. A record for a CoreRoutine request gives the request’s size in bytes, its type and the method that made it. For each of the nine CoreRoutine requests in our captures, exactly one record has a size equal to the request’s length and was sent within 15 minutes after it (5,393, 3,372 and 997 bytes for the three Wi-Fi fingerprint lookups). The match needs no shared identifier, and Apple receives both from the same IP address. The telemetry’s session ID attaches to a request that carries a precise location, and since the telemetry session IDs persist for several days, they could link requests made days apart. We do not know whether Apple makes this association, and indeed it seems unlikely that it does, but we note that the possibility exists. This would be mitigated by Apple using shorter-lived telemetry session IDs.

3) Network Identifiers: The IP address is necessarily revealed in every network connection, and potentially links all connections made from the same network<sup>21</sup>. The MAC address of the connected access point is sent in every Wi-Fi quality lookup, with the same address in all of the lookups in the location-off experiment, and it appears in four endpoints over 108 days (190 requests). The MAC addresses of neighbouring access points are sent in wloc lookups (164 distinct addresses over the 29 requests that carry a MAC address), in the pbcwloc uploads and in CoreRoutine’s Wi-Fi fingerprint lookups. The same access point therefore links these records: six appear in both the pbcwloc uploads and the Wi-Fi fingerprint lookups, which were sent up to 12 hours apart.

4) Values Shared By Many Handsets: Requests carry the handset model (iPhone10,1, or D20AP in uploads), OS build, locale and country code. The accessKey URL parameter and the maps-auth-token header sent in tile and resource downloads are not individual device identifiers (see Section VIII-B in the Additional Material).

## G. Comparison With Apple’s Public Documentation

Apple’s public documentation of Location Services does not describe network connections, endpoints or message contents, so it cannot be compared with our measurements connection by connection. It does make statements about what data is sent to Apple and about the identifiers attached to it, and these can be checked.

We compare our observations with the Location Services Privacy Overview white paper of November 2019 [33], and with the Apple Privacy Policy and the “Location Services & Privacy”, “Apple Maps & Privacy” and “Device Analytics &

TABLE VIII: Summary of the comparison of Apple’s statements with our observations.
<table><tr><td rowspan=1 colspan=1>Apple statement</td><td rowspan=1 colspan=1>Assessment</td></tr><tr><td rowspan=1 colspan=1>1. Wi-Fi and cell submissions are anonymous, encryptedand use a rotating identifier</td><td rowspan=1 colspan=1>Incomplete</td></tr><tr><td rowspan=1 colspan=1>2. Routing and Traffic data is sent with an identifier thatrotates at the end of a trip</td><td rowspan=1 colspan=1>Consistent for the iden-tifier; incomplete for thedata</td></tr><tr><td rowspan=1 colspan=1>3. With Location Services off, apps cannot access loca-tion</td><td rowspan=1 colspan=1>Incomplete</td></tr><tr><td rowspan=1 colspan=1>4. Device Analytics sharing can be disabled altogether</td><td rowspan=1 colspan=1>Not addressed</td></tr><tr><td rowspan=1 colspan=1>5. Significant Locations are encrypted end to end; on-device processing means Apple does not learn the pat-tern of life</td><td rowspan=1 colspan=1>Not addressed (encryp-tion); contradicted (pat-tern of life)</td></tr><tr><td rowspan=1 colspan=1>6. Maps identifiers rotate or are longer-lived</td><td rowspan=1 colspan=1>Identifier lifetimes aregiven only for Maps: Notaddressed</td></tr><tr><td rowspan=1 colspan=1>7. Precise location is collected only to support services,or where the user agrees</td><td rowspan=1 colspan=1>Not assessed</td></tr></table>

Privacy” pages<sup>22</sup> These pages are not versioned and apply to every supported device. We cannot check statements about server-side handling, such as aggregation.

For each statement we give its context, what we observe, and an assessment: consistent; incomplete (the statement is not contradicted but omits what we observe); not addressed (the documentation does not cover what we observe); or contradicted. Omissions are not contradictions, and we report a contradiction only where a statement is incompatible with what we observe. Table VIII summarises the assessments and Table VII lists the identifiers that we observe.

1. Wi-Fi and cell location submissions. Apple says: the data is sent “in an anonymous and encrypted form”, and only when Location Services is on (Location Services page). The white paper describes this as crowdsourcing to improve Apple’s database of Wi-Fi hotspot and cell tower locations, and says that “the submission uses a random, rotating identifier so that Apple can’t tie multiple submissions together”. We observe: five pbcwloc uploads, sent only with location on. We take “encrypted” to mean TLS, since every payload that we decode is a plain protobuf inside the TLS session, so Apple’s servers can read it. The request bodies contain no identifier, so none rotates. Each record pairs the MAC address of an access point or the identity of a cell with the handset location (accuracy 400 m or better) and a timestamp. The uploads share a source IP address, and six of their access points recur in CoreRoutine’s Wi-Fi fingerprint lookups sent up to 12 hours apart. Assessment: incomplete. The uploads carry no direct identifier, which seems an improvement on the rotating identifier that Apple describes. However, “anonymous” and “Apple can’t tie multiple submissions together” seem too strong since the connections contain values that could link submissions.

2. Routing and Traffic. Apple says: while in transit the handset sends “GPS data, travel speed and direction, and barometric pressure”, anonymously (Location Services page). The white paper adds that the identifier “rotates at the conclusion of a trip” and is not tied to an Apple ID. We observe: only the drive experiment produces these uploads. The rtloc and /pds/pd uploads carry a random tripId, and the /hvr/trc and /psr uploads carry no identifier of any kind. None carries an Apple Account identifier. The tripId rotates after a 15-minute gap or a 3-hour trip (Section V-F), which is close to the “conclusion of a trip”. The uploads also carry data that the description does not list: motion state, altitude, the bundle identifier of the app using location, GNSS satellite reports, compass attitude, pedometer data and temperature, and they are up to 13 hours old when sent. Assessment: consistent for the identifier; incomplete for the data.

3. Location Services off. Apple says: when Location Services is off, “apps can’t access the user’s Location Services data at any time” (white paper), and apps and websites “may still use other information, such as the IP address” (Location Services page). Both statements are about apps and websites, and neither mentions Apple’s own daemons. The white paper separately notes that “the networks users connect to” “might be used to track location”, in a passage on apps that scan for Wi-Fi networks and Bluetooth beacons to get around location controls. We observe: in the 27 hours with location off, geod makes 299 connections. They include 97 Wi-Fi quality lookups, each carrying the MAC address of the connected access point, which locates the handset to tens of metres if Apple holds the position of the access point, and a weather widget lookup with a city-level location. Assessment: incomplete. Apple’s statements about apps are not contradicted, but they say nothing about what its own daemons send.

4. Device Analytics. Apple says: users can “disable the sharing of Device Analytics altogether” (Device Analytics page). The page does not say which Apple services this setting covers. We observe: the test user chose “Don’t Share” at setup, and geod nevertheless sends 88 telemetry connections in the 27 hours with location off. Assessment: not addressed, because we cannot tell whether this telemetry is Device Analytics.

5. Significant Locations. Apple says: Significant Locations & Routes “are synced between your devices using end-to-end encryption and cannot be read by Apple” (Location Services page). The white paper adds that processing them on the device avoids “Apple or third parties learning sensitive details about your pattern of life”. We observe: CoreRoutine, which apparently implements Significant Locations, sends requests to Apple, through geod, that are not end-to-end encrypted, so Apple’s servers can read them. Its reverse geocoding requests carry precise locations. Its Wi-Fi fingerprint lookups carry 94 timestamped locations, the MAC addresses of nearby access points and the entry and exit times of a visit. Apple therefore receives where the handset was and for how long. Assessment: the encryption statement concerns the synchronised store and says nothing about these requests, so it is not addressed. The claim that Apple does not learn the pattern of life is contradicted for the visits that we observe, assuming that CoreRoutine’s requests are part of Significant Locations.

6. Identifier lifetimes. Apple says: the Apple Maps page, which concerns the Maps app, says that individual usage metrics are “associated with an identifier that rotates multiple times per hour”. It adds that “Longer-lived identifiers” are used for “application performance, aggregate usage metrics” and to maintain the integrity of the service, and that these are not attached to an Apple Account. It gives no lifetime for the longer-lived identifiers. No other page that we examined gives identifier lifetimes. We observe: telemetry session IDs that persist for days to months, an A/B experiment guid that persists for at least 1.8 days, and the connected access point’s MAC address, which appears in four endpoints over 108 days (Table VII). The session IDs come from 11 system processes and are sent without Maps being used, including in the location-off experiment, so they are not Maps data. Assessment: not addressed. The Maps page concerns Maps and so says nothing about these identifiers or their lifetimes.

7. Apple Privacy Policy. We do not assess its statement that Apple collects “precise location only to support services such as Find My or where you agree for region-specific services”. It is too open-ended to test: “such as” leaves the list of services open and “where you agree” is not defined. We observe precise locations sent only with Location Services on, which is consistent with a reading in which switching it on is the agreement, but the policy does not say so and the per-service settings that Apple documents suggest that agreement may be intended per service. We did not test those settings (Routing and Traffic, Improve Location Accuracy and Significant Locations), so we do not know whether they stop the uploads and requests that we observe, nor whether the telemetry and the Wi-Fi quality lookups have a control at all.

## VI. AI USE

Summary of main AI use:

• Section III: Claude Opus 4.8, Codex GPT Sol 5.6 used for recovery of protobuf descriptors from Apple binaries.

• Section IV: Claude Opus 4.8/Sonnet 5 used to develop Extractor.

• Section V: Claude Sonnet 5 used to: (i) assist with ObjC binary code analysis to trace the lifecycle of the GUID, Maps-Auth-Token, accessKey, service\_tag, session\_id, clientRevisions, abClientMetadata, Gloria tile ID (X-tilekey header) values, i.e. how they are calculated and when they change (validated using network measurements and by human code review); (ii) assist with code analysis of URL location encoding (e.g. cl2.apple.com/1/v1/{lat0}/{lon0}/{lat}\_{lon (validated by human code review and known handset location); (iii) assist with decoding headers of Wi-Fi, GSM fine cell, DEM (Digital Elevation Model), HRSE (High-Resolution Signal Environment) tiles to extract tile geographic boundaries (the download URLs also often specify this information, so this is primarily a cross-check).

## VII. LIMITATIONS

The measurements reported here are inevitably limited in nature. In our experiments we only make fairly minimal use of the handset, although we do change the handset location and take it for a drive. We also assume a privacy-conscious user who answers “no” when asked to share data with Apple and who uses essentially no Apple location services<sup>23</sup>. Behaviour that is infrequent, or that depends on usage we did not carry out, may not be observed. This means that the data collection that we report here is likely an underestimate: absence of evidence is not evidence of absence. We observe only what is sent, not how Apple stores or uses it.

In our tests we use a jailbroken handset. This is necessary to allow decryption of the HTTPS network traffic, as discussed in Section V-A3. It is possible that the behaviour of a jailbroken handset differs from that of a non-jailbroken one. However, we have seen no evidence of this.

The decoding of the traffic depends on Extractor. Its validation tests use binaries compiled from an ObjC template that we reverse engineered, and so share our assumptions about Apple’s code, and for Apple’s own binaries the true schemas are unknown. We mitigate this by checking that decoded messages are consistent with the field names and with the handset’s known position.

Probably the main limitation of our measurements is the use of an older version of iOS. This is unavoidable since there are no jailbreaks for later versions. Later versions may make different connections or send different data, and we cannot test this. However, we can compare message formats and this shows that the protobuf fields we observe in iPhone 8 (iOS 16.7.11) traffic are still declared, unchanged, in later releases. We ran Extractor on the GeoServices and CoreLocationProtobuf images of iOS 27 and macOS 27 and compared the recovered descriptors with those of iOS 16: all 739 geo and clp message fields seen in our captures (188 messages) have the same field number, name and type in both. Later releases add fields (177 in iOS 27, 173 in macOS 27, to these 188 messages) but remove none of the observed ones.

## VIII. CONCLUSION

We measure the location data that an iPhone sends to Apple and, because the message formats are undocumented, evaluate how far AI can help to recover them.

Asking two frontier models to reconstruct Apple’s protobuf schemas directly from its binaries gets most individual fields right but only three of the 17 schemas entirely right for each model. The errors cannot be detected without an independent reference: they leave the schemas internally consistent and fall mostly on rarely used fields, so testing against captured traffic does not reveal them. Having Claude write a tool whose output can be validated is, in contrast, reliable and much less work than recovering the schemas by hand. The condition is that ground truth exists that does not come from the model.

Here we construct it by reverse engineering the template that Apple’s protobuf classes follow and compiling schemas whose answers we know.

The decoded traffic shows that no message carries a direct identifier of the user or handset. With location on, however, the handset sends its precise position to Apple, in reverse geocoding and Wi-Fi fingerprint lookups that carry the entry and exit times of visits, in Wi-Fi and cell lookups, and in batched uploads of GPS fixes and of Wi-Fi and cell observations. Usage telemetry carries session identifiers that persist for at least 106 days and so can link a handset’s records over months, and the MAC address of the connected access point, a network identifier, is sent when location is off. Apple’s documentation does not mention the last two. Our measurements are of one handset used in a limited way, so our account of the data sent is likely an underestimate.

We recommend that Apple (i) document the telemetry session identifiers and limit their lifetime, (ii) state which data is sent when location is off, or stop sending it, (iii) make clear that Significant Locations processing involves requests to Apple’s servers that carry precise locations and visit times, and (iv) ask users before “harvesting” phone sensor data.

## REFERENCES

[1] Australian Competition and Consumer Commission, “Google LLC to pay \$60 million for misleading representations,” Media release, August 2022. [Online]. Available: https://www.accc.gov.au/media-release/googl e-llc-to-pay-60-million-for-misleading-representations

[2] Texas Attorney General, “Attorney General Ken Paxton Secures Historic \$1.375 Billion Settlement with Google Related to Texans’ Data Privacy Rights,” Press release, May 2025. [Online]. Available: https://www.texasattor neygeneral.gov/news/releases/attorney-general-ken-pax ton-secures-historic-1375-billion-settlement-google-rel ated-texans-data

[3] N. Sivan, R. Bitton, and A. Shabtai, “Analysis of Location Data Leakage in the Internet Traffic of Android-based Mobile Devices,” in 22nd International Symposium on Research in Attacks, Intrusions and Defenses (RAID 2019), 2019, pp. 243–260. [Online]. Available: https://www.usenix.org/conference/raid2019/p resentation/sivan

[4] A. Girish, J. Reardon, J. Tapiador, S. Matic, and N. Vallina-Rodriguez, “Your Signal, Their Data: An Empirical Privacy Analysis of Wireless-scanning SDKs in Android,” Proceedings on Privacy Enhancing Technologies, vol. 2025, no. 3, pp. 361–381, 2025. [Online]. Available: https://arxiv.org/abs/2503.15238

[5] H. Tan, Q. Luo, J. Li, and Y. Zhang, “LLM4Decompile: Decompiling Binary Code with Large Language Models,” in Proc Conference on Empirical Methods in Natural Language Processing (EMNLP), 2024, pp. 3473– 3487.

[6] D. Xie, Z. Zhang, N. Jiang, X. Xu, L. Tan, and X. Zhang, “ReSym: Harnessing LLMs to Recover Variable and Data Structure Symbols from Stripped Binaries,” in Proc ACM SIGSAC Conference on Computer and Communications Security (CCS), 2024, pp. 4554–4568.

[7] A. Kharlamova, Y. Sun, and T. Yu, “Exposing Hidden Interfaces: LLM-Guided Type Inference for Reverse Engineering macOS Private Frameworks,” arXiv:2601.01673, 2026. [Online]. Available: https: //arxiv.org/abs/2601.01673

[8] J. Spence, N. Assaderaghi, F. Xiao, J. Zhu, N. Ravi, X. Qi, M. Jagielski, R. A. Popa, E. Wallace, G. Wei, Y. Ding, and Z. Zhang, “The Next Challenge for Agentic Cybersecurity: A Realistic, Contamination-Free Reverse Engineering Benchmark,” arXiv:2608.11469, 2026. [Online]. Available: https://arxiv.org/abs/2608.114 69

[9] C. Liu, E. Raff, and K. Micinski, “When LLM Decompilers Recompile More and Preserve Less,” arXiv:2609.05370, 2026. [Online]. Available: https: //arxiv.org/abs/2609.05370

[10] M. Moulinier, “pbtk: Protobuf Toolkit,” GitHub repository. [Online]. Available: https://github.com/m arin-m/pbtk

[11] A. Tetelman, “protodump: Extract Protobuf Definitions From Compiled Binaries,” GitHub repository. [Online]. Available: https://github.com/arkadiyt/protodump

[12] NCC Group, “Blackbox Protobuf,” GitHub repository. [Online]. Available: https://github.com/nccgroup/blackb oxprotobuf

[13] K. Kollnig, R. Binns, P. Dewitte, M. Van Kleek, G. Wang, D. Omeiza, H. Webb, and N. Shadbolt, “A fait accompli? an empirical study into the absence of consent to Third-Party tracking in android apps,” in Seventeenth Symposium on Usable Privacy and Security (SOUPS 2021). USENIX Association, Aug. 2021, pp. 181–196. [Online]. Available: https://www.usenix.org/c onference/soups2021/presentation/kollnig

[14] T. T. Nguyen, M. Backes, N. Marnau, and B. Stock, “Share first, ask later (or never?) studying violations of GDPR’s explicit consent in android apps,” in 30th USENIX Security Symposium (USENIX Security 21). USENIX Association, Aug. 2021, pp. 3667–3684. [Online]. Available: https://www.usenix.org/conference/ usenixsecurity21/presentation/nguyen

[15] K. Kollnig, A. Shuba, R. Binns, M. Van Kleek, and N. Shadbolt, “Are iphones really better for privacy? a comparative study of ios and android apps,” Proceedings on Privacy Enhancing Technologies, vol. 2022, no. 2, pp. 6–24, 2022.

[16] D. J. Leith and S. Farrell, “Contact Tracing App Privacy: What Data Is Shared By Europe’s GAEN Contact Tracing Apps,” in Proc IEEE INFOCOM, 2021, pp. 1–10.

[17] D. J. Leith, “Mobile Handset Privacy: Measuring the Data iOS and Android Send to Apple and Google,” in Proc SecureComm, Security and Privacy in Communication Networks, LNICST 399, 2021, pp. 231–251.

[18] H. Liu, P. Patras, and D. J. Leith, “ On the data privacy practices of Android OEMs,” PLoS ONE, vol. 18, no. 1, p. e0279942, 2023.

[19] D. J. Leith, “What Data Do the Google Dialer and Messages Apps on Android Send to Google?” in Proc SecureComm, Security and Privacy in Communication Networks, LNICST 462, 2022, pp. 549–568.

[20] A. Razaghpanah, R. Nithyanand, N. Vallina-Rodriguez, S. Sundaresan, M. Allman, C. Kreibich, and P. Gill, “Apps, Trackers, Privacy, and Regulators: A Global Study of the Mobile Tracking Ecosystem,” in Proc NDSS, 2018.

[21] K. Kollnig, A. Shuba, M. Van Kleek, R. Binns, and N. Shadbolt, “Goodbye Tracking? Impact of iOS App Tracking Transparency and Privacy Labels,” in Proc ACM Conference on Fairness, Accountability, and Transparency (FAccT), 2022, pp. 508–520.

[22] D.J.Leith, “Poor Privacy Practices Of The Apple App Store: Cookies, Advertising and Tracking Of Users,” arXiv:2610.05475, 2026. [Online]. Available: https://arxiv.org/abs/2610.05475

[23] hubert3, “iSniff GPS,” GitHub repository (presented at Black Hat USA, July 2012), 2012. [Online]. Available: https://github.com/hubert3/iSniff-GPS

[24] E. Rye and D. Levin, “Surveilling the masses with wifi-based positioning systems,” in 2024 IEEE Symposium on Security and Privacy (SP), 2024, pp. 2831–2846.

[25] Antonio Cheong (acheong08), “apple-corelocationexperiments,” GitHub repository, including techreport.md (undated). [Online]. Available: https: //github.com/acheong08/apple-corelocation-experiments

[26] S. Edwards, “On the Tenth Day of APOLLO, My True Love Gave to Me: An Oddly Detailed Map of My Recent Travels (iOS Location Analysis),” mac4n6.com blog, December 2018. [Online]. Available: http://www.mac4n6.com/blog/2018/12/23/on-the-tenth-d ay-of-apollo-my-true-love-gave-to-me-an-oddly-detaile d-map-of-my-recent-travels-ios-location-analysis

[27] V. Katalov, “Apple Probably Knows What You Did Last Summer,” Elcomsoft blog, June 2018. [Online]. Available: https://blog.elcomsoft.com/2018/06/apple-pro bably-knows-what-you-did-last-summer/

[28] Y.-A. de Montjoye, C. A. Hidalgo, M. Verleysen, and V. D. Blondel, “Unique in the Crowd: The Privacy Bounds of Human Mobility,” Scientific Reports, vol. 3, p. 1376, 2013.

[29] H. Zang and J. Bolot, “Anonymization of Location Data Does Not Work: A Large-Scale Measurement Study,” in Proc ACM MobiCom, 2011, pp. 145–156.

[30] P. Golle and K. Partridge, “On the Anonymity of Home-/Work Location Pairs,” in Proc Pervasive Computing, LNCS 5538, 2009, pp. 390–397.

[31] D.J.Leith, “Between the Commits: Process, Error, and Claim Reliability in a Wholly AI-Authored Codebase,” arXiv:2609.29744, 2026. [Online]. Available: https://arxiv.org/abs/2609.29744

[32] A. Cortesi, M. Hils, T. Kriechbaumer, and contributors, “mitmproxy: A free and open source interactive HTTPS proxy,” 2020. [Online]. Available: https://mitmproxy.org/

[33] Apple Inc., “Location Services Privacy Overview,” November 2019. [Online]. Available: https://www.apple. com/privacy/docs/Location\_Services\_White\_Paper\_Nov \_2019.pdf

[34] OpenBubbles, “rustpush,” GitHub repository, commit cb2361c (4 October 2026), 2026. [Online]. Available: https://github.com/OpenBubbles/rustpush

[35] J. Spangenberger, “iOS Forensics: VMP4 File format,” digitalforensics.io blog post, September 2020. [Online]. Available: https://www.digitalforensics.io/ios-forensics-v mp4-file-format/

[36] K. Sulayman, “vmp4-dump: Dumps Apple VMP4 files, used by geod, GeoServices, Maps, and other Apple apps,” GitHub repository, 2022. [Online]. Available: https://github.com/19h/vmp4-dump

## ADDITIONAL MATERIAL

## A. Comparison With Published Protobuf Descriptors

This section compares the descriptors that Extractor recovers with descriptors published by other researchers and written by hand, for Apple’s location services [25] and for CloudKit and iMessage [34]. The field numbers agree wherever both define a field, and in every case where the published descriptor disagrees with Extractor on a field’s type, meaning or structure, the iPhone’s binaries agree with Extractor.

1) Location Protobuf Descriptors: The repository accompanying [25] contains three protobuf descriptors written by hand from captured traffic. They cover the Wi-Fi and cell positioning request and response (wloc), the pbcwloc request, and the Wi-Fi tile response (/wifi\_request\_tile). We compare these with the descriptors that Extractor recovers from the iPhone 8 binaries. Messages are paired by endpoint and field number: their AppleWLoc (a single message used for both directions) with ALSLocationRequest and ALSLocationResponse in locationd, their PbcWlocRequest with CLPCellWifiCollectionRequest in CoreLocationProtobuf, and their WifiTile with GEOWiFiQualitySearchResultTile in GeoServices. Both sets of descriptors are then used to decode the same captured traffic: 66 wloc requests, 66 wloc responses, 34 pbcwloc requests and 226 Wi-Fi tiles.

Every captured body decodes under Extractor’s descriptors with no unknown fields. The published descriptors leave unknown fields in every message type. They omit the wloc request fields that carry cell towers (1), the Wi-Fi band (31) and the altitude scale (32), the channel and Z-axis harvest count of each access point (21, 22), the app bundle identifier in each pbcwloc Wi-Fi entry (5), and several fields of the tile entries.

The field numbers agree wherever both descriptors define a field. Where they differ is in the declared types and in the field names. Since the protobuf wire format does not distinguish int32 from int64 or uint32 for small values, and carries no names, these differences cannot be resolved from traffic alone. We therefore resolve each one against the iPhone 8 binaries. The serialisation code in writeTo: calls a writer routine for each field (for example PBDataWriterWriteInt32Field or PBDataWriterWriteUint32Field) and the Objective-C ivar encoding records the stored type (i for int, I for unsigned int, Q for unsigned long long, B for bool). Table IX lists every declared-type difference. In every case the binary agrees with Extractor.

The first row is an encoding difference rather than a width difference: sint32 uses zigzag encoding, so an sint32 value of N is encoded as 2N and is received as 2N by a reader that expects int32. The latitude and longitude fields (ALSLocation 1 and 2) are int64 in both. The last row is a difference in wire type: the quality field of a tile entry (field 4 of WiFiBSS) is a repeated message, WiFiQualityProfile, which contains a repeated WiFiQualityIndex of a type and a value. The published descriptor defines it as two int64 fields (varint rather than length-delimited), so the real contents are left as unknown fields. The ivar encoding records only that the field is an array of objects, not the class of the elements. Extractor infers the class from the code that reads the field (Section IV-B2), and the 226 captured tiles all decode under it with no unknown fields.

The remaining differences are in the field names. Most are cosmetic and describe the same value (for example bssid and macID, wifi\_devices and wirelessAPs, horizontal\_accuracy and accuracy), so Table X lists only those names that imply a different meaning or that the published descriptor leaves unnamed. Several of the published names for fields of the wloc location message are those of the corresponding CLPLocation fields, but ALSLocation is a different class with only ten data fields, and its fields 7, 11, 12 and 13 are confidence, score, reach and infoMask. The published message also defines fields 8 to 10 and 14 to 31, which do not exist in ALSLocation (the class has ten data ivars and writeTo: writes only tags 1 to 7 and 11 to 13), and none of them occurs in the 2,142 locations in the captured responses. In the CoreLocationProtobuf MotionActivity message, writeTo: writes the type ivar as field 1 and confidence as field 2, the reverse of the published names. The published DeviceType message (operating system, then model) matches the ALSMeta message that the wloc request carries, but pbcwloc uses CLPMeta, which has the product identifier first and the software version second. The last column of Table X gives values observed in the captured traffic. For fields 11 and 12 and for the pbcwloc device information they are consistent with the binary name and not with the published one, and the tile key equals the key requested. The motion activity values do not distinguish the two definitions, so for that message we rely on the binary.

TABLE IX: Differences in declared field type between the published descriptors and Extractor, with the writer routine called in writeTo: and the ivar encoding in the iPhone 8 binary. The binary agrees with Extractor in every row.
<table><tr><td rowspan=1 colspan=1>Message (class)</td><td rowspan=1 colspan=1>Field(s)</td><td rowspan=1 colspan=1>Published</td><td rowspan=1 colspan=1>Extractor</td><td rowspan=1 colspan=1>Binary</td></tr><tr><td rowspan=1 colspan=1>wloc request (ALSLocationRequest)</td><td rowspan=1 colspan=1>3,4</td><td rowspan=1 colspan=1>sint32</td><td rowspan=1 colspan=1>int32</td><td rowspan=1 colspan=1>Int32Field, i</td></tr><tr><td rowspan=1 colspan=1>wloc request (ALSLocationRequest)</td><td rowspan=1 colspan=1>25</td><td rowspan=1 colspan=1>optional</td><td rowspan=1 colspan=1>repeated</td><td rowspan=1 colspan=1>array ivar, Submessage</td></tr><tr><td rowspan=1 colspan=1>LTE cell tower(ALSLteCellTower)</td><td rowspan=1 colspan=1>1-4, 6, 7</td><td rowspan=1 colspan=1>uint32</td><td rowspan=1 colspan=1>int32</td><td rowspan=1 colspan=1>Int32Field, i</td></tr><tr><td rowspan=1 colspan=1>wloc location (ALSLocation)</td><td rowspan=1 colspan=1>3-7, 12</td><td rowspan=1 colspan=1>int64</td><td rowspan=1 colspan=1>int32</td><td rowspan=1 colspan=1>Int32Field, i</td></tr><tr><td rowspan=1 colspan=1>wloc location (ALSLocation)</td><td rowspan=1 colspan=1>11, 13</td><td rowspan=1 colspan=1>int64</td><td rowspan=1 colspan=1>uint32</td><td rowspan=1 colspan=1>Uint32Field, I</td></tr><tr><td rowspan=1 colspan=1>pbcwloc Wi-Fi entry(CLPWifiAPLocation)</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>int32</td><td rowspan=1 colspan=1>enum</td><td rowspan=1 colspan=1>i, has scanTypeAsString:</td></tr><tr><td rowspan=1 colspan=1>pbcwloc location(CLPLocation)</td><td rowspan=1 colspan=1>16,17</td><td rowspan=1 colspan=1>int32</td><td rowspan=1 colspan=1>bool</td><td rowspan=1 colspan=1>B</td></tr><tr><td rowspan=1 colspan=1>tile (GEOWiFiQualitySearchResultTile)</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>int64</td><td rowspan=1 colspan=1>uint64</td><td rowspan=1 colspan=1>Q; decodes as varint</td></tr><tr><td rowspan=1 colspan=1>tile entry (WiFiBSS)</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>pair of int64</td><td rowspan=1 colspan=1>repeated WiFiQualityProfile</td><td rowspan=1 colspan=1>array of objects</td></tr></table>

The Extractor results are for iOS 16.7.11, whereas the published descriptors are for later iOS versions: the repository’s code identifies itself to Apple as iOS 17.5 and its report analyses iOS 18.6.2.

2) CloudKit Protobuf Descriptors: CloudKit is the protocol used by iCloud and iMessage to store and synchronise data. The iMessage client library rustpush [34] contains a protobuf descriptor file for it, cloudkit.proto, with 153 messages. Field names such as unk2 and comments such as always 0 indicate that it was written by hand from observed traffic. We exclude three messages of the iCloud content-storage framework MMCS, which use protobuf-c, an encoding that Extractor does not support. We compare the remaining 150 messages with the descriptors that Extractor recovers from the iPhone 8, counting the generic SignedInfo as the three messages it stands for in the binary (152 in all). 142 of the 152 pair with an Extractor message and ten have no counterpart. Of the 142 pairs, 56 pair under a name that differs from the rustpush name (for example rustpush AssetUploadData is CKDPAsset in the binary), which we identified from similar names or from matching field numbers and types. Of the ten unmatched, two (FileToken and FileTokens) are defined by rustpush but are unused, five are empty in rustpush or were added after iOS 16 and are in iOS 27, where Extractor recovers them and their fields, and three are in no binary that we examined.

The 142 pairs share 592 fields. The field numbers and wire types agree wherever both descriptors define a field, with one exception: SupportsSOS, which rustpush defines with a boolean as field 2 and the binary with an enumeration as field 1. Four differ in repeated or singular label, for example the published descriptor has a single asset in AssetUpload where the binary has an array. 387 of the 592 agree in number, name, type and label. 147 differ in name and 74 in declared type. Table XI lists the type differences with the evidence. In every case the binary agrees with Extractor.

Most of these are places where the published descriptor is less specific than Extractor (an integer where the class has an enumeration, or bytes where the field is a nested message or a string) or where it uses an integer of a different width or sign. These decode identically on the wire for values that fit both types. The last row is an error in the published descriptor, since a repeated string cannot decode the field’s contents.

Of the 147 name differences, 7 are unk placeholders that Extractor names (for example unk1 is deviceSoftwareIsAppleInternal), four are spelling errors in the published descriptor (lattitude, evaulationType, isAnnonymousInvitedParticipant, annonymousPublicAccess), and the rest are different wordings. For OTBottle, the keys that the published descriptor names escrowedSigningKey, escrowedEncryptionKey, peerSigningKey and peerEncryptionKey are named with the suffix SPKI in the binary (the standard X.509 SubjectPublicKeyInfo encoding of a public key), for example escrowedSigningSPKI, and ciphertext is named contents.

The descriptors differ in coverage. Extractor recovers 184 fields that rustpush does not define. Rustpush defines 31 fields that are not in the iOS 16.7.11 binaries. We repeated the comparison on iOS 27.0.1, where 27 of the 31 appear, so they were added after iOS 16. The other four (operationCost and fields 3, 7 and 12 of Asset) are in none of the binaries we searched.

3) iMessage Protobuf Descriptors: The iMessage client library rustpush [34] defines the outer envelope of an iMessage in a small descriptor file, ids.proto, with five messages (one nested). We compare it with the descriptors that Extractor recovers from the MessageProtection framework of the iPhone 8, where the corresponding classes have the prefix NGMPB. Four of the five messages pair with an Extractor message (PreKeyData is NGMPBPublicDevicePrekey,

TABLE X: Field names in the published descriptors that imply a different meaning from the iPhone 8 binary, or that are left unnamed. The binary name is the name of the ivar written for that field number, which Extractor reports unchanged. The last column gives values observed in the captured traffic.
<table><tr><td rowspan=1 colspan=1>Message</td><td rowspan=1 colspan=1>Fieldno.</td><td rowspan=1 colspan=1>Published name</td><td rowspan=1 colspan=1>Binary name</td><td rowspan=1 colspan=1>Meaning</td><td rowspan=1 colspan=1>Observed values</td></tr><tr><td rowspan=1 colspan=1>wloc location(ALSLocation)</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>unknown_value4</td><td rowspan=1 colspan=1>locationType</td><td rowspan=1 colspan=1>unnamed</td><td rowspan=1 colspan=1>0 or 3, 2,001 locations (3 in 550)</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>speed</td><td rowspan=1 colspan=1>confidence</td><td rowspan=1 colspan=1>differs</td><td rowspan=1 colspan=1>not observed in responses</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>motion_activity_type</td><td rowspan=1 colspan=1>score</td><td rowspan=1 colspan=1>differs</td><td rowspan=1 colspan=1>1 to 948 (median 63), 2,001 locations</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>motion_activity_confidence</td><td rowspan=1 colspan=1>reach</td><td rowspan=1 colspan=1>differs</td><td rowspan=1 colspan=1>0 to 70,502 (median 435), 2,001locations</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>provider</td><td rowspan=1 colspan=1>infoMask</td><td rowspan=1 colspan=1>differs</td><td rowspan=1 colspan=1>not observed in responses</td></tr><tr><td rowspan=1 colspan=1>pbcwloc motion activity(CLPMotionActivity)</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>confidence</td><td rowspan=1 colspan=1>type</td><td rowspan=1 colspan=1>differs(swapped)</td><td rowspan=1 colspan=1>1, 2,3</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>activity</td><td rowspan=1 colspan=1>confidence</td><td rowspan=1 colspan=1>differs(swapped)</td><td rowspan=1 colspan=1>0,2</td></tr><tr><td rowspan=1 colspan=1>pbcwloc deviceinformation (CLPMeta)</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>operating_system</td><td rowspan=1 colspan=1>productId</td><td rowspan=1 colspan=1>differs</td><td rowspan=1 colspan=1>D20AP,D53gAP</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>model</td><td rowspan=1 colspan=1>softwareVersion</td><td rowspan=1 colspan=1>differs</td><td rowspan=1 colspan=1>iPhone OS16.7.11/20H360,iPhone OS16.1.1/20B101</td></tr><tr><td rowspan=1 colspan=1>tile (GEOWiFiQualitySearchResultTile)</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>unknown1</td><td rowspan=1 colspan=1>tileKey</td><td rowspan=1 colspan=1>unnamed</td><td rowspan=1 colspan=1>equals the request X-tilekeyheader in 226 of 226 tiles</td></tr></table>

TABLE XI: Differences in declared field type between the rustpush descriptors and Extractor for the 142 paired CloudKit messages, with the evidence in the iPhone 8 binaries. The binary agrees with Extractor in all 74 cases.
<table><tr><td rowspan=1 colspan=1>Fields</td><td rowspan=1 colspan=1>rustpush</td><td rowspan=1 colspan=1>Extractor</td><td rowspan=1 colspan=1>n</td><td rowspan=1 colspan=1>Binary</td></tr><tr><td rowspan=1 colspan=1>enumerations, e.g.RequestOperation.Header35, Subscription 2,ShareInfo 3, ResetRequest 1</td><td rowspan=1 colspan=1>int32 or uint32</td><td rowspan=1 colspan=1>enum</td><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>the class has a &lt;field&gt;AsString: method, or theSwift type is an enumeration</td></tr><tr><td rowspan=1 colspan=1>nested messages, e.g.SubscriptionNotification1, DispositionType 1-5</td><td rowspan=1 colspan=1>bytes</td><td rowspan=1 colspan=1>message</td><td rowspan=1 colspan=1>18</td><td rowspan=1 colspan=1>ivars of a message class(@&quot;CKDPSubscriptionNotificationAlert&quot;),or arrays written by the routine that writes messagefields</td></tr><tr><td rowspan=1 colspan=1>Asset 4, 14, 18</td><td rowspan=1 colspan=1>uint64</td><td rowspan=1 colspan=1>int64</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>ivar q (signed)</td></tr><tr><td rowspan=1 colspan=1>TlkShare 2, 3, 5, 10</td><td rowspan=1 colspan=1>uint64 or uint32</td><td rowspan=1 colspan=1>int64</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>Swift type Int 64</td></tr><tr><td rowspan=1 colspan=1>EncryptedAttributes 3-10,14, 15</td><td rowspan=1 colspan=1>uint64, uint32 orint32</td><td rowspan=1 colspan=1>int32, int64 oruint32</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>ivars i, q or I, as recovered by Extractor (all 10)</td></tr><tr><td rowspan=1 colspan=1>AssetUploadData 4,AssetInfo 3, MMCSHeaders 2</td><td rowspan=1 colspan=1>uint32, uint64 orint32</td><td rowspan=1 colspan=1>int64 or uint64</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>ivars q, q and Q</td></tr><tr><td rowspan=1 colspan=1>RequestOperation.Header29, 34, ZoneDeleteRequest 2</td><td rowspan=1 colspan=1>int32 or uint32</td><td rowspan=1 colspan=1>bool</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>ivar B</td></tr><tr><td rowspan=1 colspan=1>SubscriptionNotification3, EncryptedAttributes 11</td><td rowspan=1 colspan=1>bytes</td><td rowspan=1 colspan=1>string</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>array written by the string routine; ivar NSString</td></tr><tr><td rowspan=1 colspan=1>RecordSaveRequest 3</td><td rowspan=1 colspan=1>repeated string</td><td rowspan=1 colspan=1>repeated message</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>array written by the message routine, not the stringroutine</td></tr></table>

OuterMessage is NGMPBOuterMessage, InnerMessage is NGMPBInnerMessage and the nested NgmPublicIdentity is NGMPBPublicDeviceIdentity). The fifth, KtLoggableData, has no counterpart: the iPhone 8 code receives it as an opaque NSData blob and passes it to a KTLoggableData class that stores it as bytes.

The four pairs share 12 fields (Table XII). The field numbers, wire types and declared types agree in every case, including the uint32 counter and the double timestamp, and the writeTo: code of the iPhone 8 classes writes exactly these field numbers. The only differences are in 6 field names. The binary names describe the role of the field (for example encryptedPayload, ephemeralPubKey and keyValidator), where rustpush uses payload, key and validator. This is the one set of published descriptors that we compare with no difference in type.

## B. Access Key And Maps-Auth-Token Generation

Resource requests to gspe21-ssl.ls.apple.com carry an accessKey query parameter, and tile requests to gspe19 -ssl.ls.apple.com/tile.vf carry a Maps-Auth-Token header. The handset computes both from a salt string hard-coded in the GeoServices framework and the authToken value T that it receives from gspe35-ssl.ls.apple.com/geo\_manifest/dynam ic/config. The accessKey also depends on the path of the requested resource, the current time and a random nonce, and the Maps-Auth-Token on the version of the requested map style. None of these inputs is specific to the handset or the user.

In tests we found that the same T is sent to different iPhones, both when they have the same IP address and when they have different ones, and that different iPhones send the same Maps-Auth-Token values. Neither value therefore acts as an individual device identifier.

TABLE XII: Fields of the iMessage envelope in the rustpush descriptor and in the iPhone 8 binary. Field numbers and types agree for all 12 fields, and only the names differ.
<table><tr><td rowspan=1 colspan=1>Binary class (rustpush message)</td><td rowspan=1 colspan=1>Field</td><td rowspan=1 colspan=1>rustpush name</td><td rowspan=1 colspan=1>Binary name</td><td rowspan=1 colspan=1>Type</td></tr><tr><td rowspan=1 colspan=1>NGMPBPublicDevicePrekey(PreKeyData)</td><td rowspan=1 colspan=1>123</td><td rowspan=1 colspan=1>keysignaturetimestamp</td><td rowspan=1 colspan=1>prekeyprekeySignaturetimestamp</td><td rowspan=1 colspan=1>bytesbytesdouble</td></tr><tr><td rowspan=1 colspan=1>NGMPBPublicDeviceIdentity(NgmPublicIdentity)</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>publicKey</td><td rowspan=1 colspan=1>signingKey</td><td rowspan=1 colspan=1>bytes</td></tr><tr><td rowspan=1 colspan=1>NGMPBInnerMessage(InnerMessage)</td><td rowspan=1 colspan=1>12399</td><td rowspan=1 colspan=1>messagecounterktGossipDatadebugInfo</td><td rowspan=1 colspan=1>messagecounterktGossipDatadebugInfo</td><td rowspan=1 colspan=1>bytesuint32bytesbytes</td></tr><tr><td rowspan=1 colspan=1>NGMPBOuterMessage(OuterMessage)</td><td rowspan=1 colspan=1>12399</td><td rowspan=1 colspan=1>payloadkeysignaturevalidator</td><td rowspan=1 colspan=1>encryptedPayloadephemeralPubKeysignaturekeyValidator</td><td rowspan=1 colspan=1>bytesbytesbytesbytes</td></tr></table>

## C. X-tilekey Header Decoding

A Gloria tile ID is a 64-bit unsigned integer E that encodes a Web Mercator tile’s grid coordinates $( x , y )$ and zoom level z in a single value. The three decoding steps are described below.

1) Zoom Recovery: During encoding, a sentinel bit is placed at bit position 2z:

$$
E = m \mid 4 ^ { z }
$$

where m is the Morton code and $4 ^ { z } ~ = ~ 2 ^ { 2 z }$ . Because the Morton code for a zoom-z tile uses only bits $0 \ldots 2 z - 1 { \mathrm { . } }$ the most-significant set bit of E is always exactly bit 2z.

Let $\nu = \lfloor \log _ { 2 } E \rfloor$ be the zero-indexed position of the mostsignificant set bit of E. Then

$$
z = \left\lfloor { \frac { \nu } { 2 } } \right\rfloor
$$

2) Sentinel Removal: Since $4 ^ { z }$ is now known, the raw Morton code is recovered by XOR-clearing the sentinel bit:

$$
m = E \oplus 4 ^ { z }
$$

No masking of lower bits is needed: the Morton code carries no information at or above position 2z by construction.

3) Morton Code De-Interleaving: The Morton code is formed by interleaving the bits of x and y:

$$
m = \operatorname { s p r e a d } ( x ) \mid \left( \operatorname { s p r e a d } ( y ) \ll 1 \right)
$$

where spread expands a 32-bit integer so each bit occupies an even-indexed position $( 0 , 2 , 4 , \ldots )$

Decoding uses the inverse compact operation, which gathers the even-indexed bits of a 64-bit word back into the contiguous low-order bits of a 32-bit result:

$$
x = \mathrm { c o m p a c t } ( m ) , \qquad y = \mathrm { c o m p a c t } ( m \gg 1 )
$$

4) The Compact Operation: Compact proceeds as an initial mask followed by five mask-and-fold stages. Define the 64-bit masks

$$
M _ { k } = { \frac { 2 ^ { 6 4 } - 1 } { 2 ^ { 2 ^ { k + 1 } } - 1 } } \left( 2 ^ { 2 ^ { k } } - 1 \right) , \quad k = 0 , \ldots , 5 ,
$$

in which each block of $2 ^ { k + 1 }$ bits has its low $2 ^ { k }$ bits set, or, in hexadecimal:

$$
M _ { 0 } = 0 \mathbf { x } 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 \quad ( \mathrm { e v e n \ b i t s \ o n l y } )
$$

$$
M _ { 1 } = 0 { \bf x } 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 \quad ( . . . 0 0 1 1 0 0 1 1 _ { 2 } )
$$

$$
M _ { 2 } = 0 \mathbf { x } 0 \mathbf { F } 0 \mathbf { F } 0 \mathbf { F } 0 \mathbf { F } 0 \mathbf { F } 0 \mathbf { F } 0 \mathbf { F } 0 \mathbf { F } 0 \mathbf { F } 0 \mathbf { F } 0 \mathbf { F } 1 1 1 _ { 2 } )
$$

$$
M _ { 3 } = { 0 } { \bf { x } } { 0 } 0 0 \mathrm { { F F } } 0 0 \mathrm { { F F } } 0 0 \mathrm { { F F } } 0 0 \mathrm { { F F } }
$$

$$
M _ { 4 } = 0 { \bf x } 0 0 0 0 \mathrm { F F F } 0 0 0 0 \mathrm { F F F }
$$

$$
M _ { 5 } = 0 { \bf x } 0 0 0 0 0 0 0 0 \bf { F } \mathrm { F F F F F F F F }
$$

For input n (either m or $m \gg 1 )$ :

$$
n  n \ \& \ M _ { 0 }\tag{1}
$$

$$
n  ( n | n \gg 1 ) \ \& \ M _ { 1 }\tag{2}
$$

$$
n  ( n | n \gg 2 ) \ \& \ M _ { 2 }\tag{3}
$$

$$
n  ( n \mid n \gg 4 ) \ \& \ M _ { 3 }\tag{4}
$$

$$
n  ( n | n \gg 8 ) \ \& \ M _ { 4 }\tag{5}
$$

$$
n  ( n \mid n \gg 1 6 ) \ \& \ M _ { 5 }\tag{6}
$$

Step (1) isolates the target bits (even-indexed for x, oddindexed already shifted to even for $y )$ . Each subsequent step folds adjacent pairs of already-gathered bits downward by a doubling shift, progressively packing them into the low 32 bits. The final mask $M _ { 5 }$ discards the upper 32 bits, leaving a clean 32-bit result.

5) Result: The decoded tile coordinates $( x , y , z )$ are:

• z: the Web Mercator zoom level; the tile space is a $2 ^ { z } \times 2 ^ { z }$ grid.

$x \in [ 0 , 2 ^ { z } )$ : tile column, increasing west to east.

$y \in [ 0 , 2 ^ { z } )$ : tile row, increasing north to south (standard Web Mercator convention).

<table><tr><td>Onboarding screen</td><td>Option selected</td></tr><tr><td>Language</td><td>&quot;English&quot;, &quot;Ireland&quot;</td></tr><tr><td>Quick Start</td><td>&quot;Set Up Manually&quot;</td></tr><tr><td>Written and Spoken Lan-</td><td>&quot;Continue&quot;</td></tr><tr><td>guages Choose a Wi-Fi Network</td><td>&quot;Continue without wifi&quot;, &quot;Continue&quot;24</td></tr><tr><td>Data &amp; Privacy</td><td>&quot;Continue&quot;</td></tr><tr><td>Touch ID</td><td>&quot;Set up Touch ID later&quot;</td></tr><tr><td>Create a Passcode</td><td>&quot;Passcode options&quot;, &quot;Don&#x27;t use passcode&quot; &quot;Don&#x27;t Transfer Apps &amp; Data&quot;</td></tr><tr><td>Apps &amp; Data</td><td>&quot;Forgot password or don&#x27;t have an Apple</td></tr><tr><td>Apple ID</td><td>ID&quot;, &quot;Set up later in Settings&quot;</td></tr><tr><td>Terms and Conditions</td><td>&quot;Agree&quot; &quot;Only Download Automatically&quot;</td></tr><tr><td>Update your iPhone Auto- matically</td><td></td></tr><tr><td>iMessage &amp; FaceTime</td><td>&quot;Not now&quot;</td></tr><tr><td>Location Services</td><td>&quot;Disable Location Services&quot;</td></tr><tr><td>Siri</td><td>&quot;Set Up Later in Settings&quot;</td></tr><tr><td>Screen Time</td><td>&quot;Set Up Later in Settings&quot;</td></tr><tr><td>iPhone Analytics</td><td>&quot;Don&#x27;t Share&quot;</td></tr><tr><td>Appearance</td><td>&quot;Continue&quot;</td></tr><tr><td>Display Zoom</td><td>&quot;Continue&quot;</td></tr><tr><td>Welcome to iPhone</td><td>&quot;Get Started&quot;</td></tr></table>

TABLE XIII: Settings selected during handset onboarding following a factory reset.

## D. Connections Made By geod And locationd

## E. Third-Party Claims About /hvr/v3/use

The technical report accompanying [25] states that gsp64-ssl.ls.apple.com/hvr/v3/use receives opaque binary data that includes the user’s home location, the open apps, and IP addresses and ports, and that some of the requests are encrypted. The report does not decode the payloads. Our decoded records (Section V-E) account for most of these observations. The “open apps” correspond to the identifier of the app or daemon that triggered each geod request (for example com.apple.wifi.ThreeBarsXPCService), which is not a list of the apps that are open. The IP addresses and ports are those of the Apple server that the request was sent to (for example 17.253.77.204:443, in 15 records), and not the handset’s connections. The requests that appear encrypted are ARPC bodies compressed with LZ4, which have high entropy until they are decompressed. We did not observe a home location. The Maps user record has the fields homeCountryCode and homeMetroRegion, but both read “UNKNOWN” because our test user has no home set, and the fields indicate that a populated value would be a country and metro region and not a street address.

## F. Handset Onboarding Settings

Table XIII lists the settings that the test user selected during onboarding after the factory reset.

## G. Further Figures

Figure 12 shows the smaller tiles (about 3–11 km across) that locationd downloads (Section V-C).

![](images/f13bde108548a3fbbe84c6e08094a0183ed85c2add0f9da1285f9be57c20c5f8.jpg)  
(a) DEM tile

![](images/dc97266f18acecdd4dc8c3691f1322e61e96609e7ad07cb4bea8b9d0987f616b.jpg)  
(b) Wi-Fi tile

![](images/3365722018e3a18b812c3ac41bcd898b76c43438a74924902fd64f59e61eb5b8.jpg)  
(c) HRSE tile  
Fig. 12: Visualisation of smaller tiles (≈3–11 km across) downloaded by locationd. The star in each plot marks the handset location.

Figure 13 shows a cell tower identifier sent to wloc.

![](images/2f15c89da94640a985bb91a22f3a493453260dc8628f5e7b91745458ba8b0568.jpg)  
Fig. 13: Example showing transmission of a cell tower identifier to Apple server gs-loc.apple.com/clls/wloc.

Figure 14 shows the real-time traffic probe sent by geod. It carries the same GPS fixes as the /pds/pd upload in Figure 11.

## H. Examples Of geod Network Connections

I. HTTP Headers And Query Parameters In geod Connections

![](images/2fa5558acb88fd5e4b882d10dbd472bfbdc5845c6e01ce5a78ff0439fd395c36.jpg)  
Fig. 14: Example showing transmission of device GPS locations to Apple server gsp9- ssl.apple.com/hvr/v2/rtloc. The locations are carried in a serialised CoreLocationProtobuf message, shown decoded. Enumeration names are shortened.

![](images/4a61be645bfc3262fefa9aafb8f6d630aca9cae425a9f3b0c290d31379af9d57.jpg)  
Fig. 15: Example showing a connection to the Apple A/B experiment endpoint (ab\_experiment\_url), gsp-ssl.ls.ap ple.com/ab.arpc.

![](images/7d076bdf36d207d5a79daa941f1d95b1d1e7116d620800abeb88c1729fd26233.jpg)  
Fig. 16: Example showing transmission of telemetry data to Apple server gsp64-ssl.ls.apple.com/hvr/v3/use.

TABLE XIV: Network connections made by the iOS geod process.
<table><tr><td>Endpoint</td><td>Purpose</td><td>Request content</td><td>Response content</td><td>Loc. sent</td><td>When loc. off?</td></tr><tr><td>configuration.ls.apple.com/con fig/defaults</td><td>Settings</td><td>URL params: OS version, hardware</td><td>XML plist: feature flags, tile- type switches, endpoint URLs, per- country feature maps</td><td>No</td><td>√</td></tr><tr><td>gspe35-ssl.1s.apple.com/geo_ manifest/dynamic/config</td><td>Settings</td><td>URL params: country code, OS ver- sion, hardware</td><td>Protobuf: geod resources, incl. map tiles, fonts, icons, textures available, endpoint URLs</td><td>Country code</td><td>√</td></tr><tr><td>gspe1-ssl.1s.apple.com/pep/gcc</td><td>IP country lookup</td><td>Standard HTTP headers only; no body</td><td>Plain text: Country code</td><td>No</td><td></td></tr><tr><td>gsp-ssl.ls.apple.com/ab.arpc</td><td>A/B experiments</td><td>ARPC protobuf: session GUID, lo- cale, OS version, country code</td><td>Protobuf: bucket ID for namespaces (parsec, Siri, RAP, mapsAb)</td><td>Country code</td><td>√</td></tr><tr><td>gsp64-ssl.ls.apple.com/hvr/v3 /use</td><td>Telemetry</td><td>ARPC protobuf: batched per-request log records with app ID, HW/OS, connectivity type, carrier, country code, latency, HTTP status, server</td><td>Empty ARPC acknowledgement (0- byte payload)</td><td>Carrier; country code</td><td>√</td></tr><tr><td>gsp9-ssl.apple.com/hvr/v2/rtlo C</td><td>Real-time traffic probe</td><td>IP:port, tile-set ID ARPC protobuf: sequence of times- tamped records with handset GPS lo- cation, accuracy, handset motion type</td><td>Empty ARPC acknowledgement (0- byte payload)</td><td>Precise (GPS, ≈5 m accuracy)</td><td></td></tr><tr><td>gsp-ssl.ls.apple.com/dispatcher .arpc</td><td>Rev. geocoding, geocoding</td><td>ARPC protobuf: caller (com.apple.weather.widget, analyticsd, com.apple.networkserviceproxy, com.apple.kbd or com.apple.corelocation.CoreLocationTemporaryPreciseAuthPromptPlugin),</td><td>Protobuf: reverse-geocoded street address, bounding polygons, place- data</td><td>Coarse or none</td><td>√</td></tr><tr><td>gsp57-ssl-revgeo.ls.apple.com /dispatcher.arpc</td><td>Rev. geocoding</td><td>or none, place-data ARPC protobuf: caller com.apple.CoreRoutine.helperservice, HW/OS, locale, GPS coordinates,</td><td>Protobuf: reverse-geocoded street address, bounding polygons, place- data</td><td>Precise (full- precision</td><td></td></tr><tr><td>gsp57-ssl-locus.ls.apple.com/ dispatcher.arpc</td><td>Wi-Fi fingerprint lookup</td><td>place-data ARPC protobuf: caller com.apple.CoreRoutine.helperservice, HW/OS, locale, session ID, timestamped locations, MAC address</td><td>Status only: no result</td><td>Precise (reported accuracy 5.6 to 30 m)</td><td></td></tr><tr><td>gsp85-ssl.ls.apple.com/wifi_re quest</td><td>Wi-Fi quality lookup</td><td>APs, visit entry/exit times ARPC protobuf: BSSID of currently associated AP, locale, country code</td><td>Protobuf: quality status for submit- ted BSSID</td><td>BSSID (current Wi-Fi AP)</td><td>V</td></tr><tr><td>gspe85-ssl.ls.apple.com/wifi request_tile</td><td>Wi-Fi quality tile download</td><td>X-tilekey header: Gloria tile key²</td><td>Protobuf tile: per-AP records with packed BSSID (48-bit int), lat/lon, popularity score, auth type, public-</td><td>Coarse (tile area</td><td></td></tr><tr><td>gspe19-ssl.ls.apple.com/tile.vf</td><td>Map resource</td><td>URL params or maps-tile-* headers: Mercator tile coords x, y, z4</td><td>network flag VMP4 file [35, 36]? map features, POIs etc.</td><td>Coarse (tile areaß</td><td></td></tr><tr><td>gspe79-ssl.1s.apple.com/65/v1/ zoom8/&lt;key&gt;.tar.gz</td><td>Coarse location tile</td><td>URL path: Gloria tile key²</td><td>Coarse location grid cells</td><td>Coarse (tile area)</td><td></td></tr><tr><td>gspe79-ssl.1s.apple.com/72/v2/ 72/&lt;key&gt;</td><td>Network interface selection (Wi-</td><td>URL path: Gloria tile key²</td><td>Tiles with network data: downlink throughput, interface mask, quality percentile, tower hash</td><td>Coarse (tile area)</td><td></td></tr><tr><td>gspe21-ssl.ls.apple.com/styles heet/*.styl</td><td>Fi/LTE/5G) VectorKit stylesheet download</td><td>URL param; version in filename</td><td>VectorKit rendering rules (colours, stroke widths, label fonts, icon zoom ranges, POI busyness hooks)</td><td>No</td><td></td></tr><tr><td>gspe21-ssl.1s.apple.com (/xml/, /other/, /icon/, /config/)</td><td>Map resource downloads</td><td>URL param; version in filename</td><td>JSON/plist/protobuf/XML: POI type maps, country feature lists, localisation data, altitude refs, publisher attribution, EV/vehicle</td><td>No</td><td>√</td></tr><tr><td>mesu.apple.com/assets/.../Ge oPolygonDataAssets.xml</td><td>Settings</td><td>OS version</td><td>transit, POI, localised XML manifest of available polygon</td><td>No</td><td>V</td></tr></table>

<sup>1</sup> By caller: city-level (one fixed point several kilometres from the handset) for com.apple.weather.widget; coordinates rounded to 1 to 2 decimal places for analyticsd; reported accuracy of about 3.7 km for com.apple.networkserviceproxy; postal district, no coordinates, for com.apple.kbd; IP address only for com.apple.corelocation.CoreLocationTemporaryPreciseAuthPromptPlugin.  
<sup>2</sup> Encoding lat/lon/zoom ≈1–2 km at zoom 13; e.g. 81607936 → 53.32°N, 6.31°W  
<sup>3</sup> Maximum tile zoom observed is 13, corresponding to a tile size of ≈0.05<sup>◦</sup> or ≈5 km.  
<sup>4</sup> Maps to lat/lon/zoom, e.g. x=1976, y=1328, z=12 (zoom) → 52.1°N, 6.33°W  
<sup>5</sup> All successful tile.vf responses in our iPhone 8 captures begin with the bytes VMP4 (after optional gzip), followed by a table of typed sections, some of which are zlib compressed, as in the on-device map cache files described in [35]. We do not decode the contents.  
<sup>6</sup> Maximum zoom level observed is 15, corresponding to a tile size of ≈0.01<sup>◦</sup> or ≈1 km.  
<sup>7</sup> Zoom 8, corresponding to a tile size of ≈1.4<sup>◦</sup> or ≈150 km.

TABLE XV: HTTPS connections made by the iOS locationd process.
<table><tr><td>Endpoint</td><td>Purpose</td><td></td><td>Request content</td><td>Response content</td><td>Loc. sent</td><td>When</td><td>loc. off?</td></tr><tr><td>gs-loc.apple.com /clls/wloc</td><td>Wi-Fi/cell lookup</td><td>location</td><td>Request contains a cell tower iden- tity (MCC, MNC, cell ID) or a Wi- Fi MAC address (in one request, the handset lat/lon).</td><td>Response is a list of cell towers or Wi-Fi MAC addresses and their lat/lon positions.</td><td>Precise Fi/cell coverage unrounded in</td><td>(Wi- tower area;</td><td></td></tr><tr><td>gsp10-ssl.apple.com/hcy/pbcwloc</td><td>Cell/Wi-Fi observation upload</td><td></td><td>Cell tower (MCC, MNC, TAC, cell ID, signal strength) or Wi-Fi (MAC address, channel, signal strength) records, each with the handset lo-</td><td>Acknowledgement only</td><td>Precise 400m)</td><td>(10-</td><td></td></tr><tr><td>gsp10-ssl.apple.com/pds/pd</td><td>GPS fix upload</td><td></td><td>GPS fixes (lat/lon, altitude, accu- racy, speed, course, motion state), app bundle ID, trip ID</td><td>Acknowledgement only</td><td>Precise ≈5 m)</td><td>(GPS,</td><td></td></tr><tr><td>gsp10-ssl.apple.com/hvr/trc</td><td>Sensor trace upload</td><td></td><td>Pressure, temperature, locations with GNSS satellite data, compass attitude, pedometer, motion state, app bundle ID</td><td>Empty</td><td>Precise and fixes)</td><td>(GPS network</td><td></td></tr><tr><td>gsp10-ssl.apple.com/psr</td><td>Pressure upload</td><td></td><td>Barometric pressure reading and timestamp, with the GPS fix at the time of the reading (lat/lon, altitude, accuracy)</td><td>Empty</td><td>Precise ≈5m)</td><td>(GPS,</td><td></td></tr><tr><td>cl2.apple.com/1/v1/ {1at0}/{1on0}/{1at}_{1on}.gz</td><td>Wi-Fi tile download</td><td></td><td>Tile SW corner in URL path</td><td>A tile containing Wi-Fi MAC ad- dresses and lat/lon positions</td><td>Coarse (tile size, ≈5 km)</td><td></td><td></td></tr><tr><td>cl2.apple.com/3/v1/ {1at0}/{1on0}/ {1at}_{lon}_{sec_key}.gz</td><td>GSM cellular download</td><td>tile</td><td>Tile SW corner in URL path, sec_key = (MCC《16)|MNC encodes mobile carrier</td><td>GSM cell identifiers, locations and coverage footprints for the specified carrier network</td><td>Coarse (tile size, ≈130km)</td><td></td><td></td></tr><tr><td>cl2.apple.com/4/v1/ {1at0}/{1on0}/ {lat}_{lon}_{sec_key}.gz</td><td>LTE cellular tile down- load</td><td></td><td>Tile SW corner in URL path, sec_key = (MCC《16)|MNC encodes mobile carrier</td><td>LTE cell identifiers (TAC and Global CellID) and their locations for the specified carrier network</td><td>Coarse (tile size, ≈130km)</td><td></td><td></td></tr><tr><td>cl3.apple.com/2/v1/ {1at0}/{1on0}/{lat}_{lon}.gz</td><td>High-Resolution Signal (HRSE) tile download</td><td>Environment</td><td>Tile SW corner in URL path</td><td>RF signal environment class²at fine resolution over the tile area</td><td>Coarse (tile size, ≈5 km)</td><td></td><td></td></tr><tr><td>cl3.apple.com/3/v2/ {1at0}/{1on0}/{lat}_{lon}.gz</td><td>Digital model download</td><td>elevation (DEM) tile</td><td>Tile SW corner in URL path1</td><td>Altitudes on a grid across the tile (likely used for GPS altitude correc- tion and barometric fusion)</td><td>Coarse (tile size, ≈7km)</td><td></td><td></td></tr><tr><td>c12.apple.com/{6,7,8,11}/v1/ 0/0/0000000_0000000.gz</td><td>Global cellular download</td><td>tile</td><td>Path 0/0/0000000_0000000 is a single tile covering all coordinates. so no location information is con- veyed</td><td>Global cellular tile: /6 → list of 87K GSM MCC/MNC/CelIID en- tries and corresponding lat/lon lo- cations, /7 → list of CDMA Base</td><td></td><td></td><td></td></tr><tr><td>cl3.apple.com: cldavl.avl.gz</td><td>Tile indexes DEM, HRSE,</td><td>availability for</td><td></td><td>Cell IDs (eNB-ID+antenna index), /11 → list of 5G NR Cell IDs. Availability bitmaps indicating which geographic tiles have server-</td><td></td><td></td><td></td></tr><tr><td>clPressureavl.avl.gz cl3.apple.com:</td><td>tile sets Global coarse signal-</td><td>barometric-pressure</td><td></td><td>Global raster map of RF signal</td><td></td><td></td><td></td></tr><tr><td>clsef.bin.gz cl4.apple.com/2/m2/a.1//</td><td>Indoor positioning</td><td>environment map</td><td></td><td>environment2 covering the Earth at 0.05° resolution (≈5.5 km per cell at the equator).</td><td></td><td></td><td></td></tr><tr><td>availability.pb.gz iphone-ld.apple.com</td><td>venue availability GNSS orbit assistance</td><td></td><td></td><td>List of venue bounding boxes (NW/SE lat/lon pairs) for regions where indoor map tiles exist</td><td></td><td></td><td></td></tr><tr><td>/xtra/xtra3grej.bin /xtra/xtra3int.bin</td><td>data XTRA, Apple)</td><td>(Qualcomm proxied by</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>configuration.apple.com /.../bt/bcwv.plist /.../bt/bcw.plist.gz</td><td>vehicle allowlist</td><td>Bluetooth hands-free</td><td></td><td>bcwv.plist: XML plist contain- ing only a version integer (15); bcw.plist.gz: gzip plist with</td><td></td><td>√</td><td></td></tr></table>

<sup>1</sup> lat/lon in URL are encoded as lat=(latitude + 90) × 10000, lon=(longitude + 180) × 10000, lat0/lon0 are the first three digits of lat/lon.  
<sup>2</sup> Signal environment types are: Rural, Urban, DenseUrban, DenseUrbanCanyon, Marine, Foliage

TABLE XVI: HTTP query parameters observed in geod network traffic.
<table><tr><td rowspan=1 colspan=1>Endpoint</td><td rowspan=1 colspan=1>Parameter</td><td rowspan=1 colspan=1>Example value</td><td rowspan=1 colspan=1>Description</td></tr><tr><td rowspan=3 colspan=1>configuration.ls.apple.com/config/defaults</td><td rowspan=1 colspan=1>os</td><td rowspan=1 colspan=1>ios</td><td rowspan=1 colspan=1>Operating system identifier</td></tr><tr><td rowspan=1 colspan=1>os_version</td><td rowspan=1 colspan=1>16.7.11</td><td rowspan=1 colspan=1>OS version string</td></tr><tr><td rowspan=1 colspan=1>hardware</td><td rowspan=1 colspan=1>iPhone10,1</td><td rowspan=1 colspan=1>Hardware model identifier</td></tr><tr><td rowspan=7 colspan=1>gspe35-ssl.1s.apple.com/geo_manifest/dynamic/config</td><td rowspan=1 colspan=1>application</td><td rowspan=1 colspan=1>geod</td><td rowspan=1 colspan=1>Client application name</td></tr><tr><td rowspan=1 colspan=1>application_version</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Application version</td></tr><tr><td rowspan=1 colspan=1>country_code</td><td rowspan=1 colspan=1>IE</td><td rowspan=1 colspan=1>Country code</td></tr><tr><td rowspan=1 colspan=1>hardware</td><td rowspan=1 colspan=1>iPhone10,1</td><td rowspan=1 colspan=1>Handset model</td></tr><tr><td rowspan=1 colspan=1>os</td><td rowspan=1 colspan=1>ios</td><td rowspan=1 colspan=1>Operating system</td></tr><tr><td rowspan=1 colspan=1>os_build</td><td rowspan=1 colspan=1>20H360</td><td rowspan=1 colspan=1>OS build</td></tr><tr><td rowspan=1 colspan=1>os_version</td><td rowspan=1 colspan=1>16.7.11</td><td rowspan=1 colspan=1>OS version</td></tr><tr><td rowspan=10 colspan=1>gspe19-ssl.ls.apple.com/tile.vf</td><td rowspan=1 colspan=1>style</td><td rowspan=1 colspan=1>62, 68, 73</td><td rowspan=1 colspan=1>Map tile style ID</td></tr><tr><td rowspan=1 colspan=1>size</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>Tile size (Retina factor)</td></tr><tr><td rowspan=1 colspan=1>scale</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>Display scale</td></tr><tr><td rowspan=1 colspan=1>V</td><td rowspan=1 colspan=1>18946531</td><td rowspan=1 colspan=1>Map data version number</td></tr><tr><td rowspan=1 colspan=1>Z</td><td rowspan=1 colspan=1>2, 9, 12</td><td rowspan=1 colspan=1>Zoom level (XYZ tile scheme)</td></tr><tr><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>1, 246, 1976</td><td rowspan=1 colspan=1>Tile X coordinate</td></tr><tr><td rowspan=1 colspan=1>y</td><td rowspan=1 colspan=1>1, 165, 1328</td><td rowspan=1 colspan=1>Tile Y coordinate</td></tr><tr><td rowspan=1 colspan=1>vlang</td><td rowspan=1 colspan=1>en-GB</td><td rowspan=1 colspan=1>Display language</td></tr><tr><td rowspan=1 colspan=1>preflight</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>Request token</td></tr><tr><td rowspan=1 colspan=1>accessKey</td><td rowspan=1 colspan=1>Base64 token</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=3 colspan=1>gspe19-ssl.ls.apple.com/asset/v3/material/&lt;id&gt;</td><td rowspan=1 colspan=1>style</td><td rowspan=1 colspan=1>60</td><td rowspan=1 colspan=1>Map tile style ID</td></tr><tr><td rowspan=1 colspan=1>preflight</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>Request token</td></tr><tr><td rowspan=1 colspan=1>accessKey</td><td rowspan=1 colspan=1>Base64 token</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>gspe21-ssl.1s.apple.com/{type}/{resource}</td><td rowspan=1 colspan=1>accessKey</td><td rowspan=1 colspan=1>Base64 token</td><td rowspan=1 colspan=1>Used for all static asset types: font, icon,texture, stylesheet, xml, other</td></tr></table>

TABLE XVII: Non-standard HTTP headers observed in geod network connections. Standard headers (Accept, User-Agent, etc.) are omitted.
<table><tr><td rowspan=1 colspan=1>Header</td><td rowspan=1 colspan=1>Description / example value</td></tr><tr><td rowspan=1 colspan=1>maps-auth-token</td><td rowspan=1 colspan=1>HMAC of the tile version v (e.g. v=18965870); see Section VIII-B</td></tr><tr><td rowspan=1 colspan=1>x-tilekey</td><td rowspan=1 colspan=1>Integer tile key, e.g. 81607936</td></tr><tr><td rowspan=1 colspan=1>maps-tile-x / -y / -z</td><td rowspan=1 colspan=1>XYZ tile grid coordinates (column, row, zoom level), e.g. x=3, y=2, z=3</td></tr><tr><td rowspan=1 colspan=1>maps-tile-style</td><td rowspan=1 colspan=1>Tile parameters, e.g. style=20&amp;size=2&amp;scale=0&amp;v=18965870. The headers carrythe same fields as the tile . vf query parameters of Table XVI; styles 1, 11, 20,30 and 53 and zoom levels 3 to 15 occur in them</td></tr><tr><td rowspan=1 colspan=1>x-msg-priority</td><td rowspan=1 colspan=1>Small integer, either 5 or 8</td></tr><tr><td rowspan=1 colspan=1>x-os-version</td><td rowspan=1 colspan=1>iOS version and build string, 16.7.11.20H360</td></tr><tr><td rowspan=1 colspan=1>maps-tile-reason</td><td rowspan=1 colspan=1>Fixed value, PROACTIVE</td></tr></table>