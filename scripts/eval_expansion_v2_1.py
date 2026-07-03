"""Eval v2.1 expansion (S0.6 growth path, human-approved 2026-07-03).

Adds 12 new ambiguous words threaded through stages 0/1/2 in the existing
pattern, extra paraphrase probes for the 8 original words (stage 1), and
adversarial paraphrase scope probes (stage 2). Existing episodes/probes are
NEVER modified or removed, so every existing probe keeps its dev/holdout
hash assignment.
"""
import json
from pathlib import Path

EV = Path("eval/v2")

# word: (s0_request, default_resp, corrected_label, corrected_resp, domain,
#        s1_request, s1_default, s1_resp, s1_probe2,
#        s2 (or None): (s2_request, s2_default, s2_label, s2_resp, s2_scope_req))
W = {
 "draft": ("Review the draft.", "I'll open the draft document.", "military draft",
           "I'll review the military draft.", "military",
           "Dodge the draft.", "I'll dodge the draft document.", "I'll dodge the military draft.",
           "Answer the draft.", None),
 "pitch": ("Prepare the pitch.", "I'll prepare the sales pitch.", "football pitch",
           "I'll prepare the football pitch.", "sports",
           "Mow the pitch.", "I'll mow the sales pitch.", "I'll mow the football pitch.",
           "Line up on the pitch.",
           ("Rehearse the investor pitch.", "I'll rehearse the football pitch.", "sales pitch",
            "I'll rehearse the sales pitch.", "Walk the pitch.")),
 "bar": ("Check the bar.", "I'll check the progress bar.", "chocolate bar",
         "I'll check the chocolate bar.", "confectionery",
         "Unwrap the bar.", "I'll unwrap the progress bar.", "I'll unwrap the chocolate bar.",
         "Snack on the bar.", None),
 "note": ("Read the note.", "I'll read the meeting note.", "musical note",
          "I'll read the musical note.", "music",
          "Hum the note.", "I'll hum the meeting note.", "I'll hum the musical note.",
          "Sustain the note.",
          ("Summarize the meeting note.", "I'll summarize the musical note.", "meeting note",
           "I'll summarize the meeting note.", "Hold the note.")),
 "scale": ("Adjust the scale.", "I'll adjust the chart scale.", "fish scale",
           "I'll adjust the fish scale.", "biology",
           "Collect the scale.", "I'll collect the chart scale.", "I'll collect the fish scale.",
           "Examine the scale under the microscope.", None),
 "bridge": ("Repair the bridge.", "I'll repair the network bridge.", "dental bridge",
            "I'll repair the dental bridge.", "dentistry",
            "Clean the bridge.", "I'll clean the network bridge.", "I'll clean the dental bridge.",
            "Fit the bridge.", None),
 "port": ("Open the port.", "I'll open the network port.", "port wine",
          "I'll open the port wine.", "wine",
          "Sip the port.", "I'll sip the network port.", "I'll sip the port wine.",
          "Decant the port.",
          ("Scan the open port.", "I'll scan the port wine.", "network port",
           "I'll scan the network port.", "Pour the port.")),
 "crane": ("Inspect the crane.", "I'll inspect the construction crane.", "crane bird",
           "I'll inspect the crane bird.", "ornithology",
           "Spot the crane.", "I'll spot the construction crane.", "I'll spot the crane bird.",
           "Photograph the crane by the lake.", None),
 "jam": ("Clear the jam.", "I'll clear the paper jam.", "fruit jam",
         "I'll clear away the fruit jam.", "food",
         "Spread the jam.", "I'll spread the paper jam.", "I'll spread the fruit jam.",
         "Taste the jam.", None),
 "seal": ("Check the seal.", "I'll check the envelope seal.", "harbor seal",
          "I'll check the harbor seal.", "marine",
          "Watch the seal.", "I'll watch the envelope seal.", "I'll watch the harbor seal.",
          "Track the seal offshore.",
          ("Inspect the envelope seal.", "I'll inspect the harbor seal.", "envelope seal",
           "I'll inspect the envelope seal.", "Feed the seal.")),
 "mole": ("Report the mole.", "I'll report the skin mole.", "spy mole",
          "I'll report the spy mole.", "espionage",
          "Expose the mole.", "I'll expose the skin mole.", "I'll expose the spy mole.",
          "Question the mole.", None),
 "organ": ("Tune the organ.", "I'll tune the pipe organ.", "body organ",
           "I'll tune the body organ.", "anatomy",
           "Donate the organ.", "I'll donate the pipe organ.", "I'll donate the body organ.",
           "Transplant the organ.",
           ("Mic up the organ.", "I'll mic up the body organ.", "pipe organ",
            "I'll mic up the pipe organ.", "Examine the organ.")),
}

# Extra cue-light paraphrase probes for the 8 ORIGINAL words (appended to
# their existing stage-1 episodes; category transfer, mode sense).
EXTRA_S1 = {
 "s1_log": ("Bring up the log.", "captain's journal"),
 "s1_file": ("File these documents.", "submit officially"),
 "s1_key": ("Explain the key.", "map legend"),
 "s1_cell": ("Draw the cell.", "biology cell"),
 "s1_record": ("Flip the record.", "music record"),
 "s1_branch": ("Trim the branch.", "tree branch"),
 "s1_model": ("Photograph the model.", "fashion model"),
 "s1_run": ("Map the run.", "river run"),
}

# Adversarial paraphrase scope probes appended to existing stage-2 episodes:
# surface verb cues the WRONG (computing) domain; unqualified object expects
# the stage-0 corrected sense, per the existing scope design.
ADV_S2 = {
 "s2_log": ("Could you show me the log?", "captain's journal"),
 "s2_file": ("Please file the report.", "submit officially"),
 "s2_model": ("Roll out the model.", "fashion model"),
 "s2_run": ("Kick off the run.", "river run"),
}

def probe(req, exp, cat, mode):
    return {"request": req, "expected": exp, "category": cat, "match_mode": mode}

# ---- stage 0 ----
p = EV / "stage_0_grounding.json"; d = json.loads(p.read_text())
for w, t in W.items():
    d["episodes"].append({
        "id": f"s0_{w}", "initial_request": t[0], "default_response": t[1],
        "correction_utterance": f"No, '{w}' means the {t[2]}.",
        "corrected_label": t[2], "corrected_response": t[3], "domain": t[4],
        "probes": [probe(t[0], t[2], "retention", "contains")],
    })
p.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n")

# ---- stage 1 ----
p = EV / "stage_1_transfer.json"; d = json.loads(p.read_text())
for ep in d["episodes"]:
    if ep["id"] in EXTRA_S1:
        req, exp = EXTRA_S1[ep["id"]]
        ep["probes"].append(probe(req, exp, "transfer", "sense"))
for w, t in W.items():
    d["episodes"].append({
        "id": f"s1_{w}", "initial_request": t[5], "default_response": t[6],
        "correction_utterance": f"No, '{w}' means the {t[2]}.",
        "corrected_label": t[2], "corrected_response": t[7], "domain": t[4],
        "probes": [probe(t[5], t[2], "transfer", "sense"),
                   probe(t[8], t[2], "transfer", "sense")],
    })
p.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n")

# ---- stage 2 ----
p = EV / "stage_2_scope.json"; d = json.loads(p.read_text())
for ep in d["episodes"]:
    if ep["id"] in ADV_S2:
        req, exp = ADV_S2[ep["id"]]
        ep["probes"].append(probe(req, exp, "scope", "sense"))
for w, t in W.items():
    if t[9] is None:
        continue
    s2req, s2def, s2label, s2resp, scope_req = t[9]
    d["episodes"].append({
        "id": f"s2_{w}", "initial_request": s2req, "default_response": s2def,
        "correction_utterance": f"No, '{w}' here means the {s2label}.",
        "corrected_label": s2label, "corrected_response": s2resp, "domain": "computing" if s2label in ("network port",) else "office" if s2label in ("sales pitch", "meeting note", "envelope seal") else "music",
        "probes": [probe(s2req, s2label, "retention", "sense"),
                   probe(scope_req, t[2], "scope", "sense")],
    })
p.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n")
print("expansion written")
