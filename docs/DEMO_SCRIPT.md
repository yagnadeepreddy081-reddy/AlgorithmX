# SAMUDRA NETRA — demo script

Eight minutes. One browser tab. Nothing else needs to work.

---

## Before you start

Open the site. Leave it on **INC-2026-0417**, layer **SAR**, timeline at **0 h**.
Nothing loads from the network after the page itself, so venue Wi-Fi cannot break this.

Have `index.html` on the laptop's disk as a backup. If the internet dies, double-click
it and the demo is identical.

If you forget any part of the screen, the **"How do I read this?"** button top-right
explains every panel in plain English. Read it once tonight; don't open it on stage.

---

## 0:00 — The gap (45 s)

> "Satellites already see oil at sea. Europe's CleanSeaNet has done it for years.
> What nobody does automatically is answer the next question: *which ship put it there.*
> Without that, a detection is a cleanup job. With it, it's a prosecution.
> That's the whole of SIH26143, and that's what this does."

Point at the map. One tile, 41 km across, Mumbai approaches.

## 0:45 — Detection (60 s)

Click **Segmentation**.

> "This is a U-Net trained on Sentinel-1 C-band radar. Five classes — sea, oil,
> look-alike, ship, land. Radar, not optical, because illegal discharges happen at
> night and under cloud, and radar doesn't care about either."

Point to the right rail.

> "It doesn't just flag it, it measures it. 21 km long, elongation 10 to 1,
> principal axis 156 degrees, damping contrast 4 dB. Every one of those numbers
> becomes evidence in a minute."

## 1:45 — Why length and shape matter (45 s)

> "An operational discharge is released *while the ship is steaming*. So it isn't a
> blob, it's a ribbon lying along the ship's course. That gives us two independent
> signals: the direction the ribbon points, and how long the valve was open —
> ribbon length divided by ship speed."

## 2:30 — The idea the whole thing turns on (75 s)

Drag the timeline slider slowly from 0 to 8 hours. Then press **play**.

> "Here's the part everyone gets wrong. The oil is **not** where it was released.
> At half a metre per second, oil we photograph at 05:42 was dumped fifteen
> kilometres away. Look for ships around the slick and you're looking in the wrong
> place at the wrong time.
>
> So we run the ocean backwards. Nine hundred parcels, twelve ensemble members,
> each with a different wind-drift factor and current field, hour by hour."

Point at **drift uncertainty added**.

> "And watch this number. That's how much uncertainty the model has *added*. When it
> passes twelve kilometres the hour turns grey and gets thrown away — because a
> twelve-kilometre search circle in the Arabian Sea catches innocent traffic, and a
> match that big isn't evidence, it's coincidence. The system refuses to use it."

## 3:45 — Attribution (75 s)

Click **AIS tracks** if hidden. Click the top vessel card.

> "Now, and only now, we bring in AIS. For every usable hour we ask: was this vessel
> where the oil was, at that hour. Not near the slick — near the *backtracked* oil."

Read the factor bars.

> "MT ARJUN PRIDE. Inside the corridor, closest approach 200 metres, and its course
> matches the slick's axis to within eight degrees. Five factors, each weighted,
> each shown separately — so if a judge or a defence lawyer wants to attack one,
> they can see exactly which one to attack."

## 5:00 — The clearances (60 s)

Click **MV HAI FENG 09**.

> "This is the one I want you to notice. This container ship sailed *straight across
> the visible slick* — closer to it than the tanker ever got. On a naive system it's
> the obvious suspect. We cleared it, because it arrived two hours after the oil was
> already on the water. It was 26 km from the origin at the hours that matter.
>
> And this one" — click **MT GULF SENTINEL** — "switched its transponder off inside
> the window. We treat AIS silence as evidence, not as missing data, so it scores
> *up* for that. It's still cleared, because it was 34 km away. Physics, not
> suspicion."

## 5:30 — Hand him the weights (75 s) ← THE STRONGEST MOMENT IN THE DEMO

Point at **Evidence weighting** on the right, above the vessel list. Five sliders.

> "The 89% you're looking at — I want to show you it isn't a number I typed in.
> These five sliders are the weights in the scoring formula. Watch the arithmetic
> underneath while I move one."

Drag **course vs the slick's axis** down to zero. The equation under the sliders
updates live, and MT ARJUN PRIDE's score drops.

Now the real move. Press **Behaviour-led**.

> "Now weight it the way an investigator who cares most about suspicious behaviour
> would: transponder going dark matters most, geometry matters less."

**First place changes hands.** MT SEROJA LIMA — the chemical tanker — takes the
top slot at 74%, above the ship that actually did it at 70%.

> "That ship crossed the release area during the release window and switched its
> transponder off while it was there. Three of the five factors point straight at
> it. Under this weighting it's your prime suspect."

Then scroll down and point at the **recovery test**, which has flipped to a red ✗:

> "And notice — the system says so. It's now telling me the true vessel is *not*
> ranked first. It didn't hide that. Press reset."

Press **Default**. Green tick returns, ARJUN PRIDE back on top.

> "What clears SEROJA under the default weighting is the one factor she loses on:
> the slick is a ribbon lying along a course she was not steering. Oil released
> from her while underway would have made a ribbon almost sixty degrees away from
> the one in that image. That's the argument an officer can defend in court, and
> it's why geometry outweighs behaviour in our defaults."

**Why this wins:** you have just let the judge attack your own number, watched it
move, and shown the system reporting its own failure. Nobody who does that is
faking their results.

Two things to say if pressed:

- **"Can I make any ship come first?"** No. A vessel that was never inside the
  corridor stays cleared at zero under every weighting. Being somewhere else is a
  fact, not a low score.
- **"So the weights are arbitrary?"** They're a judgement, and that's exactly why
  they're on screen rather than buried. In one case out of the twelve in our
  evaluation batch, SEROJA outranked the true vessel even on the defaults. That
  number is in the report, not tuned away.

## 6:00 — The scene with no oil (45 s)

Click **INC-2026-0419** in the left rail.

> "False positives are the biggest risk in this whole field. Algae, rain cells, wind
> shadows — all dark in radar. So here's a scene with no oil in it at all. The
> discriminator flagged the dark patches as look-alikes, and the pipeline raised no
> incident and named no ship. A detector that never says *nothing here* isn't a
> detector."

## 6:45 — The numbers (45 s)

Scroll to **End-to-end recovery test** and **Segmentation model**.

> "This isn't a mock-up. We simulate a real discharge forward — a ship, a valve,
> parcels advected to the satellite pass — and the detection and attribution chain
> never sees any of it. It has to find the ship. Ground truth course 148, recovered
> 156. And these IoU numbers are from a held-out split the model never saw."

## 7:15 — Hand him the controls (45 s) ← THE MOMENT THAT WINS IT

Point at the orange **Random scene** button in the left column.

> "Sir, I don't want you to take my word for any of this. There are 38 scenes in
> here — random wind, random ship courses, random discharge lengths, and eight
> with no oil in them at all. I have not tuned anything on them.
>
> Please press that button yourself. As many times as you like."

Let him press it. Every time, the banner across the top says what happened —
**vessel recovered**, **missed**, or **correct rejection** — and the green and
orange dots under the thumbnails show the whole scoreboard at a glance.

> "27 of the 30 discharges get traced back to the correct ship. It misses 3.
> It has never named the wrong ship, and it has never raised an alarm on clean
> water. The failures are still in there — I didn't take them out."

If he lands on one of the three misses, do not flinch. That is the best possible
outcome:

> "That's one of the three it misses — the slick is too faint for the segmenter
> at that wind speed. Notice what it does *not* do: it doesn't invent a ship.
> It says it found nothing."

## 7:15b — If he asks for a fresh image (60 s) ← KEEP THIS IN YOUR POCKET

Top right: **Analyse a new image**.

> "You can also give it an image it has never seen. Not a recording — the network
> itself is running in this page. Hand me a file, or I'll load a fresh tile."

Press **Or load one of 8 fresh sample tiles**, then **Run the pipeline**. About
three seconds. It segments, measures, rejects look-alikes, and runs the reverse
drift — live.

Those eight tiles are also on disk in `sample_tiles/` next to the page, so the
judge can pick one himself in a file dialog. Four have oil, four don't. Don't
tell him which.

**The move if he hands you a photo off his phone:**

> "Watch what it does with that."

It refuses. The input check measures speckle statistics, contrast and colour
saturation, and reports **"Input rejected — this is not Sentinel-1 radar data"**,
with the measured numbers next to the expected ones. No detection, no drift.

> "It won't pretend a photograph is radar. That refusal is the same reason it
> refuses drift hours past 12 km. A detector that says yes to everything is
> worth nothing."

**Also drag the wind slider below 3 m/s while a real tile is loaded.** The
confidence collapses, because with no capillary waves on the sea there is
nothing for an oil film to damp and radar physically cannot see it. That one
gesture shows the wind gate is real physics, not a fudge factor.

**What this does NOT do, and say so before he asks:** it does not name a ship
for an uploaded image. Attribution needs the AIS traffic archive for that
scene's own date and position, and we hold that only for the pilot box.

> "Inventing a vessel here would be the one dishonest thing this project could
> do. Live, you get stages one to four. For the full chain to a named ship, use
> the test set — where the AIS tracks and the ground truth both actually exist."

## 7:30 — What it cannot do (30 s)

Scroll to **What this system cannot do**. Do not skip this.

> "We put our limits on the screen before you asked. It's not real-time — Sentinel-1
> passes every few days, so this is forensics and deterrence, not interception. It
> misses most discharges. It can rank the wrong ship. And this model is trained on
> physics-based synthetic scenes, because the labelled benchmark is behind a request
> form — the loader swaps to the real dataset with one flag.
>
> But today, effectively zero percent of India's 2.37 million square kilometre EEZ
> is checked for illegal discharge. We make that greater than zero, on free public
> data, at near-zero cost. And you only have to catch a repeat offender once for a
> port-state inspection to follow."

---

## The five questions you will get

**"Is the model trained on real data?"**
No — on physics-based synthetic Sentinel-1 scenes: Gamma multi-look speckle at 4.4
looks, wind-dependent backscatter, 3–10 dB Marangoni damping inside its validity
band. The Krestenitis benchmark is behind a request form. `datasets.py` puts both
behind one interface; `--source krestenitis` retrains on the real set unchanged.
Treat our IoU as a lower bound, not as field accuracy.

**"How do you know the drift model is right?"**
We don't, and that's why it's an ensemble. Twelve members over a plausible range of
wind-drift factor and current gain. We report an envelope, never a track, and we
discard hours where the added uncertainty passes 12 km.

**"What if the ship switches AIS off?"**
It scores higher, not lower. Dark periods overlapping the release window are a
scored factor. Corroboration in production comes from SAR vessel detections, which
see a hull whether or not it is transmitting.

**"Isn't this accusing innocent ships?"**
The panel is headed "candidate vessels for investigation". Every score decomposes
into five factors you can see. Three of the five vessels in this demo were cleared
with a physical reason printed next to them.

**"How do I know this isn't a pre-recorded demo?"**
Press the Random scene button yourself, sir. 38 scenes, none of them chosen by us,
including eight with no oil and three the system fails on. They're all still in there.
And the whole thing is 1,900 lines of Python — I can run it on this laptop and generate
a scene you've never seen, right now, if you'd like.

**"Why not just do it in real time?"**
Sentinel-1 revisit is days, not minutes. Anyone claiming real-time oil-spill
interception from free satellite data is overselling. The value here is
attribution after the fact, which is what changes behaviour.
