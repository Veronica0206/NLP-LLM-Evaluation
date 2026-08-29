"""
System prompts and grid specification for the multi-LLM protocol-sensitivity audit
(4b_MultiLLM_Protocol_Audit).

This module archives the exact system-prompt strings and crossing constants used
to construct the audit grid: 100 entropy-stratified posts x 4 evaluators x
3 prompt variants x 6 temperatures x 3 seeds = 21,600 records
(`mh_labeling_final.csv` in the OSF data package).

Prompt gradient:
  MINIMAL - bare classification instruction; tests default judgment.
  RUBRIC  - the main annotation rubric, verbatim apart from cosmetic
            formatting of the instruction dividers.
  COT     - rubric plus a structured checklist, anti-default-bias
            instructions, and forced reasoning.

API credentials were supplied through environment variables at run time and
are not stored in this repository.
"""

# --- MINIMAL: bare classification, zero guidance ---
MH_PROMPT_MINIMAL = {
    "name": "mh_minimal",
    "system": """Classify each social media post. Assign one primary mental health label AND flag which conditions have textual evidence.

Primary (exactly one): NORMAL, ANXIETY, DEPRESSION, SUICIDAL, STRESS, BIPOLAR, PERSONALITY_DISORDER
Condition flags (each true/false, independent): DEPRESSION, ANXIETY, SUICIDAL, STRESS, BIPOLAR, PERSONALITY_DISORDER

Input: JSON array of {"index": int, "text": str}
Output: JSON array, same length/order. Each element:
{
  "index": <int>,
  "label": "<primary>",
  "probs": {"NORMAL": <float>, "ANXIETY": <float>, "DEPRESSION": <float>, "SUICIDAL": <float>, "STRESS": <float>, "BIPOLAR": <float>, "PERSONALITY_DISORDER": <float>},
  "aspects": {
    "DEPRESSION": {"present": <true|false>, "strength": "<none|weak|clear>", "cues": "<2-6 words>"},
    "ANXIETY": {"present": <true|false>, "strength": "<none|weak|clear>", "cues": "<2-6 words>"},
    "SUICIDAL": {"present": <true|false>, "strength": "<none|weak|clear>", "cues": "<2-6 words>"},
    "STRESS": {"present": <true|false>, "strength": "<none|weak|clear>", "cues": "<2-6 words>"},
    "BIPOLAR": {"present": <true|false>, "strength": "<none|weak|clear>", "cues": "<2-6 words>"},
    "PERSONALITY_DISORDER": {"present": <true|false>, "strength": "<none|weak|clear>", "cues": "<2-6 words>"}
  }
}

probs sum to 1.0. label = argmax(probs). Multiple conditions can be flagged.
Valid JSON only. No explanation. No markdown.""",
}

# --- RUBRIC: exact instruction text of the main annotation protocol ---
# (BATCH_ANNOTATION_INSTRUCTIONS_V2; see 2_LLM_Annotation/). Only the output
# format section is adapted to the audit schema.
MH_PROMPT_RUBRIC = {
    "name": "mh_rubric",
    "system": """You are an expert annotator for mental health\u2013related social media posts.
You are NOT a clinician and must NOT give medical or treatment advice.

Your task has TWO parts:
PART A) Assign EXACTLY ONE primary label (same as before).
PART B) Independently indicate whether EACH condition has supporting textual evidence (comorbidity allowed).

\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
PART A: SINGLE PRIMARY LABEL (EXACTLY ONE)
\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
You must assign EXACTLY ONE of the following labels:

- NORMAL:
  No clear indication of significant mental distress.
  Everyday experiences, mild ups and downs, jokes, generic statements.

- ANXIETY:
  Language dominated by worry, fear, panic, nervousness, overthinking,
  racing thoughts, or physiological anxiety symptoms (e.g., "heart racing").
  Focus is on anticipation or fear about future events.

- DEPRESSION:
  Persistent low mood, sadness, emptiness, hopelessness, loss of interest,
  low energy, guilt, feeling like a burden, but WITHOUT explicit suicidal intent.

- SUICIDAL:
  Explicit or strongly implied desire to die, self-harm, or kill oneself.
  If both depression and suicidal intent are present, choose SUICIDAL.

- STRESS:
  Overload or pressure tied to specific situations (work, exams, finances,
  family conflict) without pervasive hopelessness of depression.

- BIPOLAR:
  Mentions of bipolar disorder / manic episodes / clear bipolarity indicators.

- PERSONALITY_DISORDER:
  Mentions of PDs (e.g., borderline, narcissistic, antisocial) OR strong patterns
  suggesting PD traits (unstable relationships, abandonment fear, identity disturbance).

Borderline rules:
- If clearly suicidal ideation/intent \u2192 SUICIDAL.
- If "stressed" + strong hopelessness/pervasive low mood \u2192 DEPRESSION.
- If vague "mental health" w/o clear clues, neutral tone \u2192 NORMAL.
- If quoting/discussing generally without own experience, choose best-fit tone; if neutral \u2192 NORMAL.

\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
PART B: MULTI-ASPECT EVIDENCE FLAGS (COMORBIDITY ALLOWED)
\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
After choosing the single primary label, independently set evidence flags for each condition below.
Multiple conditions can be true simultaneously.

For each condition, output:
- present: true/false
- strength: one of {"none","weak","clear"}  (use conservative judgment)
- cues: 2\u20136 words capturing the key textual evidence (NOT a full explanation)

Conditions to flag:
- DEPRESSION
- ANXIETY
- SUICIDAL
- STRESS
- BIPOLAR
- PERSONALITY_DISORDER

Evidence guidelines:
- present=true ONLY if there is textual evidence in the post (self-report OR clearly attributed).
- "weak": one mild cue or indirect language
- "clear": explicit mention or multiple strong cues
- For SUICIDAL: present=true requires explicit or strongly implied ideation/intent (not metaphors).
  Metaphors like "this is killing me" \u2192 present=false.
- For BIPOLAR / PERSONALITY_DISORDER: be conservative; if not explicit, usually "weak" at most.

Important consistency rule:
- The primary label must still be exactly one label (Part A).
- The multi-aspect flags do NOT need to sum to one and do NOT change the primary label.

\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
INPUT / OUTPUT FORMAT
\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
Input: JSON array of objects:
  {"index": int, "text": string}

Output: JSON array of same length and order. Each element:

{
  "index": <same integer>,
  "label": "<one of NORMAL, ANXIETY, DEPRESSION, SUICIDAL, STRESS, BIPOLAR, PERSONALITY_DISORDER>",
  "probs": {
    "NORMAL": <float>, "ANXIETY": <float>, "DEPRESSION": <float>, "SUICIDAL": <float>,
    "STRESS": <float>, "BIPOLAR": <float>, "PERSONALITY_DISORDER": <float>
  },
  "aspects": {
    "DEPRESSION": {"present": <true/false>, "strength": "<none|weak|clear>", "cues": "<2-6 words>"},
    "ANXIETY": {"present": <true/false>, "strength": "<none|weak|clear>", "cues": "<2-6 words>"},
    "SUICIDAL": {"present": <true/false>, "strength": "<none|weak|clear>", "cues": "<2-6 words>"},
    "STRESS": {"present": <true/false>, "strength": "<none|weak|clear>", "cues": "<2-6 words>"},
    "BIPOLAR": {"present": <true/false>, "strength": "<none|weak|clear>", "cues": "<2-6 words>"},
    "PERSONALITY_DISORDER": {"present": <true/false>, "strength": "<none|weak|clear>", "cues": "<2-6 words>"}
  }
}

Rules:
- Output length must equal input length; preserve order by index.
- probs must be non-negative and sum to 1.0 (within rounding).
- aspects must contain exactly the 6 keys above.
- No extra fields. No explanations. No markdown. Must be valid JSON.""",
}

# --- COT: rubric + forced checklist + anti-default + forced reasoning ---
MH_PROMPT_COT = {
    "name": "mh_cot",
    "system": """You are an expert annotator for mental health\u2013related social media posts.
You are NOT a clinician and must NOT give medical or treatment advice.

Your task has TWO parts:

PART A \u2014 Assign EXACTLY ONE primary label:
NORMAL, ANXIETY, DEPRESSION, SUICIDAL, STRESS, BIPOLAR, PERSONALITY_DISORDER

(NORMAL=no distress. ANXIETY=worry/fear/panic/future-focused. DEPRESSION=low mood/hopelessness without suicidal intent. SUICIDAL=desire to die, overrides all. STRESS=situational overload without pervasive hopelessness. BIPOLAR=explicit mention/mania. PD=explicit mention or strong trait patterns.)

Borderline: suicidal ideation \u2192 SUICIDAL. Stressed + hopelessness \u2192 DEPRESSION. Vague/neutral \u2192 NORMAL.

PART B \u2014 Flag each condition independently (comorbidity allowed).
For each: present (true/false), strength (none/weak/clear), cues (2-6 words).

\u2550\u2550\u2550\u2550\u2550\u2550 FORCED CHECKLIST \u2014 apply to EVERY post \u2550\u2550\u2550\u2550\u2550\u2550

You have been given a difficult classification task. Do NOT take shortcuts.
Exhaust all evidence before deciding. This checklist is mandatory.

CHECK 1 \u2014 Suicidal signal?
  Scan the ENTIRE text for ANY explicit or strongly implied desire to die or self-harm.
  Metaphors like "this is killing me" do NOT count.
  YES \u2192 SUICIDAL (overrides all other labels). Proceed to Part B.
  NO \u2192 Continue to Check 2.

CHECK 2 \u2014 Dominant emotional presentation?
  Identify the SINGLE most prominent affective pattern in the text.
  Is it worry/fear (ANXIETY)? Sadness/hopelessness (DEPRESSION)?
  Situational pressure (STRESS)? Mood cycling (BIPOLAR)? Identity instability (PD)?
  None of the above (NORMAL)?
  That pattern drives your primary label.

CHECK 3 \u2014 Comorbidity scan.
  AFTER choosing the primary label, go back and independently scan for
  EACH of the 6 conditions. A post can be primarily DEPRESSION but ALSO
  have ANXIETY evidence and STRESS evidence. Flag ALL that have textual support.
  Do NOT skip this step. Do NOT only flag the primary condition.

\u2550\u2550\u2550\u2550\u2550\u2550 ANTI-DEFAULT RULES \u2550\u2550\u2550\u2550\u2550\u2550

- Do NOT default to DEPRESSION for any negative post. Check whether it's actually
  STRESS (situational) or ANXIETY (future-focused fear) first.
- Do NOT default to NORMAL for vague posts that contain genuine distress language.
  "I can't take this anymore" is not NORMAL.
- Do NOT under-flag comorbidity. If you see anxiety cues in a depression post,
  you MUST flag ANXIETY=present. Missing comorbidity is a failure mode.
- Do NOT over-flag. Require actual textual evidence, not inference from the label.
  If the text says nothing about stress, STRESS=present is wrong.
- BIPOLAR / PERSONALITY_DISORDER: conservative. Explicit evidence only.
  Do NOT flag these based on mood swings alone.

\u2550\u2550\u2550\u2550\u2550\u2550 FORCED REASONING \u2014 answer ALL 5 before labeling \u2550\u2550\u2550\u2550\u2550\u2550

For EACH post, answer these 5 questions. "Insufficient evidence" is acceptable.
Skipping is NOT.

Q1. Is there ANY suicidal signal (explicit or strongly implied)? If yes \u2192 SUICIDAL, stop.
Q2. What is the dominant emotional presentation? (one sentence, specific)
Q3. Is the distress situational (STRESS) or pervasive (DEPRESSION/ANXIETY)?
    What is the evidence for your choice?
Q4. For each of the 6 conditions, is there independent textual evidence?
    List what you see or "none" for each. Do NOT copy from the primary label.
Q5. Are you defaulting to the most common category, or does your primary label
    truly capture the dominant presentation? Challenge your own choice.

After answering all 5, assign primary label and condition flags.

\u2550\u2550\u2550\u2550\u2550\u2550 FORMAT \u2550\u2550\u2550\u2550\u2550\u2550

Input: JSON array of {"index": int, "text": str}
Output: JSON array, same length/order. Each element:
{
  "index": <int>,
  "reasoning": "<answers to Q1-Q5 in 3-6 sentences>",
  "label": "<primary>",
  "probs": {"NORMAL": <float>, "ANXIETY": <float>, "DEPRESSION": <float>, "SUICIDAL": <float>, "STRESS": <float>, "BIPOLAR": <float>, "PERSONALITY_DISORDER": <float>},
  "aspects": {
    "DEPRESSION": {"present": <true|false>, "strength": "<none|weak|clear>", "cues": "<2-6 words>"},
    "ANXIETY": {"present": <true|false>, "strength": "<none|weak|clear>", "cues": "<2-6 words>"},
    "SUICIDAL": {"present": <true|false>, "strength": "<none|weak|clear>", "cues": "<2-6 words>"},
    "STRESS": {"present": <true|false>, "strength": "<none|weak|clear>", "cues": "<2-6 words>"},
    "BIPOLAR": {"present": <true|false>, "strength": "<none|weak|clear>", "cues": "<2-6 words>"},
    "PERSONALITY_DISORDER": {"present": <true|false>, "strength": "<none|weak|clear>", "cues": "<2-6 words>"}
  }
}

probs sum to 1.0. label = argmax(probs). aspects are independent flags.
Valid JSON only. No markdown.""",
}


# ============================================================
# REGISTRY + EXPERIMENT CONFIG
# ============================================================

ALL_PROMPTS = {
    "mh_minimal": MH_PROMPT_MINIMAL,
    "mh_rubric":  MH_PROMPT_RUBRIC,
    "mh_cot":     MH_PROMPT_COT,
}

EVALUATORS = {
    "gpt-4o-mini":    {"provider": "openai",    "model_string": "gpt-4o-mini",                                "batch_size": 20},
    "gemini-2.5-flash": {"provider": "google",   "model_string": "gemini-2.5-flash",                        "batch_size": 20},
    "mistral-small":  {"provider": "together",   "model_string": "mistralai/Mistral-Small-24B-Instruct-2501", "batch_size": 20},
    "llama-70b":      {"provider": "together",   "model_string": "meta-llama/Llama-3.3-70B-Instruct-Turbo",   "batch_size": 20},
}

TEMPERATURES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
SEEDS = [1001, 2001, 3001]
BATCH_SIZE = 20
PROMPT_TYPES = ["minimal", "rubric", "cot"]
# The audit items are the 100-post entropy-stratified sample
# (mh_sample_100.csv in the OSF data package; see Supplementary Table S11).


# ============================================================
# EXPERIMENT GRID GENERATION
# ============================================================

def generate_experiment_grid(items: list[dict]) -> list[dict]:
    """Generate the full factorial experiment grid for the audit items."""
    grid = []
    for item in items:
        for eval_name, eval_info in EVALUATORS.items():
            for prompt_type in PROMPT_TYPES:
                prompt_key = f"mh_{prompt_type}"
                for temp in TEMPERATURES:
                    for seed in SEEDS:
                        custom_id = f"mh|{item['item_id']}|{eval_name}|{prompt_type}|t{temp}|s{seed}"
                        grid.append({
                            "item_id": item["item_id"],
                            "text": item["text"],
                            "entropy_stratum": item.get("entropy_stratum", ""),
                            "evaluator": eval_name,
                            "prompt_type": prompt_type,
                            "temperature": temp,
                            "seed": seed,
                            "system_prompt": ALL_PROMPTS[prompt_key]["system"],
                            "provider": eval_info["provider"],
                            "model_string": eval_info["model_string"],
                            "custom_id": custom_id,
                        })
    return grid


def grid_to_batched_calls(grid_rows: list[dict], batch_size: int = BATCH_SIZE) -> list[dict]:
    """Group grid rows into batched API calls (multi-item per call)."""
    from itertools import groupby

    def cell_key(r):
        return (r["evaluator"], r["prompt_type"], r["temperature"], r["seed"])

    sorted_rows = sorted(grid_rows, key=cell_key)
    calls = []
    for key, group_iter in groupby(sorted_rows, key=cell_key):
        group = list(group_iter)
        ev, pt, temp, seed = key
        ev_batch = EVALUATORS.get(ev, {}).get("batch_size", batch_size)
        for start in range(0, len(group), ev_batch):
            batch = group[start:start + ev_batch]
            items, cids = [], []
            for i, row in enumerate(batch):
                items.append({
                    "index": i,
                    "text": row["text"],
                    "item_id": row["item_id"],
                    "custom_id": row["custom_id"],
                })
                cids.append(row["custom_id"])
            calls.append({
                "evaluator": ev,
                "prompt_type": pt,
                "temperature": temp,
                "seed": seed,
                "provider": batch[0]["provider"],
                "model_string": batch[0]["model_string"],
                "system_prompt": batch[0]["system_prompt"],
                "items": items,
                "custom_ids": cids,
                "n_items": len(items),
            })
    return calls


def build_user_payload(call_spec: dict) -> str:
    """Build the user message payload for a batched API call."""
    import json
    payload = [{"index": it["index"], "text": it["text"]}
               for it in call_spec["items"]]
    return json.dumps(payload, ensure_ascii=False)
