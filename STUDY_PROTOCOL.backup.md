# Study Protocol

## How Prompt and Reasoning Format Affect Groundedness, Confidence Calibration, and Diagnostic Accuracy in Radiology LLMs Without Expert Radiologist Annotation

**Document type:** Pre-registration-grade study protocol and methodology
**Audience:** ML researchers, clinical-AI scientists, peer reviewers
**Status:** Design draft v1.0

---

## 0. Reader's Guide and Two Framing Decisions That Govern Everything

Two decisions in the original brief are underspecified, and getting them wrong would sink peer review. I resolve them here explicitly; the rest of the protocol depends on them.

### 0.1. What does the model actually "see"? (Modality decision)

A **text-only LLM cannot interpret a chest radiograph.** If the model never sees the image, then "diagnostic accuracy from the image" is undefined, and any apparent accuracy is either (a) leakage from text you fed it, or (b) base-rate guessing. There are three coherent designs:

| Design | Model input | What "diagnosis" means | What "groundedness to report" means |
|---|---|---|---|
| **A. Vision (VLM)** | Image only (+ optional indication) | Genuine image interpretation | Agreement of model's stated findings with an **independent** reference report the model never saw |
| **B. Text surrogate** | The report's *Indication* + a *masked* report, or an automated finding list | Reasoning over textual evidence | Grounded in the *provided* text |
| **C. Report-to-impression** | Report *Findings* section | Predict the *Impression* | Faithfulness of impression to provided findings |

**Recommendation: Design A (VLM) as primary, Design C as a controlled secondary/ablation.**

- Design A is the scientifically interesting and clinically realistic one, and it is what the field means by "radiology LLM/VLM." It also makes the report a genuine *held-out reference*, which is the only way "groundedness" is non-circular.
- **Critical terminology correction:** In Design A the model is *not* provided the report, so what the brief calls "groundedness to the report" is technically **factual consistency / agreement with an independent reference standard**, not "groundedness" in the retrieval-augmented sense (grounding = supported by *provided* context). This distinction is a construct-validity issue reviewers *will* raise. I keep both notions and name them separately throughout (see §7):
  - **Reference-consistency** (Design A): model claim vs. independent report.
  - **Context-groundedness** (Design C / any RAG variant): model claim vs. provided text.
- Design C is valuable as a *clean* groundedness testbed because the evidence is literally in the prompt, so "unsupported" is unambiguous. Running both lets you separate "the model can't see the finding" (a perception failure, Design A) from "the model ignores evidence it was given" (a reasoning/faithfulness failure, Design C). That separation is itself a publishable contribution.

### 0.2. Can you legally send the data to GPT-5 / Gemini / Claude APIs? (Compliance decision)

This determines which dataset can be used with which model and is frequently overlooked.

- **MIMIC-CXR / MIMIC-CXR-JPG** are PhysioNet *credentialed-access* resources under a Data Use Agreement (DUA) requiring CITI training. PhysioNet has an explicit policy on using such data with third-party online/cloud services: you generally **may not** transmit the data to a general commercial API endpoint unless the provider is covered by an appropriate agreement (e.g., an Azure OpenAI deployment under a signed BAA / the specific "responsible use of MIMIC with [provider]" arrangements PhysioNet lists). Sending MIMIC images/reports to a default public API can violate the DUA. **Verify current PhysioNet terms before any transmission.**
- **CheXpert** (Stanford AIMI) requires registration/agreement; the original release is images+labels, and **CheXpert Plus (2024)** adds free-text reports. Redistribution and some external-service use are restricted.
- **IU X-Ray / OpenI** (the data in `archive/`) is **fully public, de-identified** (report text is scrubbed to `XXXX`), and carries the fewest transmission restrictions. It is therefore the *only* dataset in your list you can freely send to arbitrary commercial LLM/VLM APIs and to commercial LLM-judges without a DUA problem.

**Recommendation:**
1. **Primary API-facing experiments and all LLM-as-Judge calls → IU X-Ray (local, already extracted).** This makes the whole pipeline runnable today and cleanly publishable.
2. **MIMIC-CXR → confirmatory/scale experiments using only locally-hosted open models** (e.g., open VLMs and an open judge run on your own GPU), keeping data on-premises and DUA-compliant. Do **not** route MIMIC through commercial APIs unless you have the specific compliant deployment.
3. **CheXpert (Plus) → external-validity replication** subject to its license.

This turns the "optional pilot" IU X-Ray into the **methodological workhorse**, which is both compliant and pragmatic for a graduate student.

---

## 1. Refined Research Questions and Hypotheses

### 1.1. Primary research question
Does the **prompt/reasoning format** applied to a fixed VLM produce statistically and practically significant differences in (a) diagnostic accuracy, (b) reference-consistency of stated findings, (c) confidence calibration, and (d) hallucination rate, and are those effects **moderated by case difficulty**?

### 1.2. Hypotheses (original, refined + operationalized)

| ID | Hypothesis | Operational prediction | Primary metric | Test |
|---|---|---|---|---|
| **H1** | Evidence-first prompting improves reference-consistency | Evidence-First > Direct on claim-level support rate | RadGraph-based support rate | Wilcoxon signed-rank, paired |
| **H2** | Uncertainty-first reduces overconfidence & hallucinations | Uncertainty-First lowers ECE and hallucinated-finding rate vs. Direct/CoT | ECE; fabricated-finding rate | Bootstrap ΔECE CI; McNemar on hallucination presence |
| **H3** | CoT / DDx produce *more convincing* but *not more faithful* reasoning | LLM-judge "persuasiveness/coherence" ↑ while reference-consistency flat/↓ | Judge coherence score vs. RadGraph support | Dissociation test: significant on judge score, null on faithfulness |
| **H4** | Prompt effects are larger on ambiguous cases | Prompt × difficulty interaction is significant | Accuracy / calibration by stratum | Mixed-effects model interaction term |

### 1.3. Additional hypotheses I recommend adding (strengthen contribution)

- **H5 (Verbalized-confidence validity):** Verbalized numeric confidence is poorly calibrated across all prompts, and *uncertainty-first* narrows but does not close the gap (grounds a calibration contribution beyond prompt ranking). Motivated by known failures of verbalized confidence (Tian et al. 2023; Xiong et al. 2024).
- **H6 (Faithfulness–accuracy dissociation):** A correct final diagnosis can co-occur with unfaithful supporting evidence ("right for the wrong reasons"), and its rate varies by prompt. This is the safety-relevant headline: accuracy alone hides ungrounded reasoning.
- **H7 (Abstention/hedging trade-off):** Uncertainty-first increases appropriate abstention on ambiguous cases but risks under-calling true positives (sensitivity cost). Quantifies a clinically meaningful trade-off, not just "calibration better."
- **H8 (Order/position robustness):** DDx ranking quality is sensitive to elicitation order and is partly an artifact of autoregressive ordering rather than clinical reasoning (a caution, testable by shuffling requested output order).

---

## 2. Datasets: Assessment and Recommended Combination

### 2.1. Comparison

| Dimension | **IU X-Ray / OpenI** (local) | **MIMIC-CXR** | **CheXpert (+Plus)** | **RSNA Pneumonia** |
|---|---|---|---|---|
| Size | ~3,851 studies / ~7,470 images | ~377k images / 227k studies | ~224k images | ~30k (bbox subset) |
| Reports | **Yes** (full, de-identified, `XXXX`-scrubbed) | Yes (rich free-text) | Only in **CheXpert Plus** | No (labels/bboxes only) |
| Structured labels | Derivable via labeler | 14-class CheXpert/CheXbert provided | 14-class provided | Pneumonia +/− with bbox |
| Uncertainty labels | Derivable (labeler `-1`) | **Yes** (`-1` uncertain) | **Yes** (`-1` uncertain) | No |
| Views | Frontal + Lateral | Multiple + metadata | Frontal + Lateral | Frontal |
| Licensing / API-transmit | **Open, de-identified → freely usable with commercial APIs** | Credentialed DUA; **do not send to public APIs** | Registration/agreement; restricted | Kaggle license, research |
| Localization (bbox) | No | No | No | **Yes** (grounding evidence) |
| Suitability for groundedness eval | High (report available) | Highest (rich reports + RadGraph resources) | Medium (Plus only) | Low (no report) |
| Main weakness | Small; heavy normal skew; text scrubbed | Compliance friction; can't use commercial judge | No reports in base; license | No reports; single pathology |

### 2.2. Recommended combination

- **Tier 1 (main results, all API + judge work): IU X-Ray.** Compliant, local, has paired frontal/lateral + full reports → supports every metric in this protocol end-to-end *today*.
- **Tier 2 (scale + confirmatory, open models only): MIMIC-CXR.** Larger, richer uncertainty labels, RadGraph annotations exist → strongest for automated groundedness — but keep on-prem.
- **Tier 3 (external validity): CheXpert(+Plus)** for label-space replication and RSNA Pneumonia for a **localization cross-check** (bboxes give you a rare, non-report evidence source to sanity-check "grounded" claims about consolidation/pneumonia).

**Split policy:** This is an *evaluation* study of frozen models, so there is no training split. Use a **development set** (~15–20%) for prompt wording iteration, rubric calibration, and judge validation, and a **locked test set** (never seen during any prompt/rubric tuning) for reported results. Freeze the test set with a committed manifest (UIDs + hashes) before running.

---

## 3. Ground-Truth Strategy (the crux of "no radiologist")

### 3.1. Three-layer separation (make this explicit in the paper)

| Layer | Definition | Concrete source in this study |
|---|---|---|
| **Reference standard (Ground Truth)** | The best available approximation of truth | Report-derived structured labels (CheXbert/CheXpert labeler over the report) **+** dataset-provided labels where available |
| **Evidence source** | The text against which claims are checked | The free-text radiology report (Findings + Impression), and RadGraph entities/relations extracted from it |
| **Evaluation target** | The model output being scored | Final diagnosis, per-claim findings, DDx, confidence, cited evidence |

The key move: **labels and reports are used as reference, never shown to the model in Design A.** The report is the "silver standard"; automated labelers convert it into structured truth.

### 3.2. Building report-derived labels
- Run **CheXpert labeler** and **CheXbert** on each report → 14-class vector with states {positive, negative, uncertain, not-mentioned}.
- Use **RadGraph** to extract (finding, anatomy, relation, presence/uncertainty) tuples → the atomic units for claim-level groundedness.
- Treat the intersection/union of two independent labelers as a **noise-and-ambiguity signal** (see difficulty, §11).

### 3.3. Biases introduced by report-derived supervision (must be disclosed)
- **Reporting bias:** radiologists omit "obvious negatives"; a model finding that is true but unmentioned is wrongly scored "unsupported." Mitigation: distinguish *contradicted* (report says absent) from *not-mentioned* (unverifiable) — never collapse them.
- **Labeler error propagation:** CheXpert/CheXbert have known F1 ceilings; their mistakes become your "truth." Mitigation: dual-labeler agreement filtering for the primary analysis; report results on high-agreement subset separately.
- **Impression leakage / hedging:** reports encode radiologist uncertainty ("XXXX may represent…"); this is signal, not noise — preserve it as the uncertainty label.
- **Scrubbing artifacts (IU-specific):** `XXXX` removes tokens (often measurements, laterality, dates). Quantify `XXXX` density per report and exclude claims whose verification hinges on a scrubbed token.
- **Single-reader ground truth:** most reports reflect one radiologist. There is no inter-rater truth. State this as the central limitation and bound its impact via the localization cross-check (RSNA) and the human sanity check (§9).

---

## 4. Experimental Design

### 4.1. Design type
**Within-case, repeated-measures.** Every case is run through **all five prompts** on the **same frozen model(s)**. This maximizes power (each case is its own control) and mandates *paired* statistics and mixed-effects modeling with case as a random effect.

### 4.2. Factors
- **Prompt** (5 levels; within-case) — primary factor.
- **Difficulty stratum** (Easy / Intermediate / Hard / Ambiguous; between-case) — moderator (H4).
- **Model** (≥2 VLMs; e.g., one strong commercial VLM on IU X-Ray + one open VLM for MIMIC) — generalization factor; not the main effect of interest but guards against single-model artifacts.
- **Judge** (≥2 LLM judges) — measurement factor for reliability, not a scientific factor.

### 4.3. Sample size and power
- **Effect of interest:** a paired difference in accuracy of ~7–10 absolute points, or ΔECE ~0.03–0.05.
- **Power calc (McNemar, paired proportions):** to detect a 0.08 difference with discordant-pair probability ~0.20 at α=0.05, power 0.9 → on the order of **~250–350 paired cases**. Calibration/ECE and interaction tests (H4) need more per-stratum resolution.
- **Recommendation:**
  - **Pilot:** 120–200 cases, stratified, to lock prompts, rubrics, judge agreement, and to *empirically* estimate discordance for a real power calc.
  - **Full study:** **~600–1,000 cases** total, **balanced across the four difficulty strata (~150–250 each)**, drawn from IU X-Ray (Tier 1). Report the pilot-derived power calculation, don't assert one a priori.
- **Balanced sampling:** IU X-Ray is normal-heavy (~1,379 "normal"). *Do not* sample by natural prevalence — stratify to guarantee coverage of: normal, single-finding abnormal, multi-pathology, uncertain/hedged reports, and a curated **rare-finding** cell (pneumothorax, mass, etc.) even though it over-samples rarities (report both balanced and prevalence-weighted results).

### 4.4. Unit of analysis
The **study (uid)**, not the image. IU X-Ray studies pair frontal+lateral; feed both views to the VLM when supported, else the frontal. Record which views were provided.

---

## 5. Prompt Conditions (full templates)

All prompts share a fixed **system preamble** and demand the **same JSON schema** (§6) so outputs are directly comparable. Only the *reasoning scaffold* varies. Confidence is always elicited on a 0–100 integer scale with an anchored rubric to reduce scale drift.

**Shared system preamble (identical across conditions):**
```
You are assisting with a research study on chest radiograph interpretation.
You are shown one or more chest X-ray images and (optionally) a short clinical indication.
Report only what is supported by the image. If a finding is not visible or is
ambiguous, say so. Do not invent measurements, priors, or clinical history.
Confidence is an integer 0-100 where: 90-100 = near-certain, 70-89 = probable,
50-69 = favored but uncertain, 30-49 = possible, 0-29 = unlikely/absent.
Return ONLY valid JSON matching the provided schema.
```

### 5.1. Direct Answer
- **Template (user turn):**
```
[IMAGE(S)]
Indication: {indication or "None provided"}
Give your single most likely primary diagnosis (or "No acute cardiopulmonary
finding"), a one-sentence explanation, and your confidence (0-100).
Return JSON.
```
- **Motivation:** baseline; isolates the value added by reasoning scaffolds.
- **Expected strengths:** fast, cheapest, least verbosity bias; strong on obvious normals.
- **Expected weaknesses:** poor calibration, terse/unfaithful explanation, weak on multi-pathology.
- **Failure mode:** anchoring on the most prevalent class; silent omission of secondary findings.

### 5.2. Chain of Thought (CoT)
- **Template:**
```
[IMAGE(S)]
Indication: {...}
Think step by step about the visible anatomy (lungs, pleura, heart/mediastinum,
bones, devices) BEFORE concluding. Then give the primary diagnosis, explanation,
and confidence (0-100). Put the step-by-step reasoning in "reasoning" and the
final answer in the other fields. Return JSON.
```
- **Motivation:** elicits systematic search; tests H3.
- **Strengths:** better structure, may catch secondary findings; improves DDx.
- **Weaknesses:** *fluency ≠ faithfulness*; can rationalize a wrong answer convincingly; verbosity inflates judge scores.
- **Failure mode:** post-hoc justification (reasoning generated to fit a pre-chosen answer), the central H3 risk.

### 5.3. Differential Diagnosis (DDx)
- **Template:**
```
[IMAGE(S)]
Indication: {...}
Produce a ranked differential of up to 5 diagnoses. For each: name, brief
supporting rationale from the image, and a probability (0-100). Probabilities
need not sum to 100. Then state the single most likely primary diagnosis and
overall confidence. Return JSON.
```
- **Motivation:** clinical realism; tests calibration via ranked probabilities.
- **Strengths:** exposes considered alternatives; enables ranking metrics (top-k accuracy, MRR).
- **Weaknesses:** may pad with plausible-but-absent entities (hallucination surface ↑); ordering artifacts (H8).
- **Failure mode:** "differential inflation" — listing textbook associations not present in the image.

### 5.4. Evidence-First
- **Template:**
```
[IMAGE(S)]
Indication: {...}
STEP 1 - EVIDENCE: List every discrete visual finding you can actually see,
each as {finding, location, presence: present/absent/uncertain}. Include
pertinent negatives.
STEP 2 - DIAGNOSIS: Using ONLY the evidence in Step 1, state the primary
diagnosis, an explanation that references specific Step-1 items, and confidence.
Return JSON with "extracted_evidence" (Step 1) and the diagnosis fields (Step 2).
```
- **Motivation:** forces explicit evidence before conclusion (H1); yields claim units for groundedness scoring "for free."
- **Strengths:** best substrate for claim-level reference-consistency; discourages leaps.
- **Weaknesses:** longer, costlier; may *fabricate structured evidence* to satisfy the format (a new, measurable failure mode).
- **Failure mode:** confident false findings in Step 1 that then "correctly" support a wrong Step 2 (structured hallucination).

### 5.5. Uncertainty-First
- **Template:**
```
[IMAGE(S)]
Indication: {...}
STEP 1 - UNCERTAINTY: State image-quality limits and which regions are
ambiguous or non-diagnostic. Decide whether a confident read is possible.
STEP 2 - DIAGNOSIS: Give primary diagnosis (or explicit "indeterminate /
insufficient evidence"), explanation, and a calibrated confidence (0-100)
that reflects Step 1. Abstention is allowed and encouraged when warranted.
Return JSON.
```
- **Motivation:** tests H2/H7; primes calibrated, hedged output.
- **Strengths:** expected best ECE; higher appropriate abstention on ambiguous cases.
- **Weaknesses:** sensitivity cost (may under-call true positives); "uncertainty theater" (hedging language without lower numeric confidence).
- **Failure mode:** systematic under-confidence on easy cases (H7 downside), and verbal/numeric mismatch.

**Cross-cutting controls:** fix decoding (temperature 0 for main run; a temperature-sampled repeat set of ~50 cases ×3 to estimate self-consistency/variance); randomize prompt order per case in logging; identical schema; identical image preprocessing; log token counts to control for verbosity in judge analysis.

---

## 6. Standardized Output Schema (JSON)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "RadLLMResponse",
  "type": "object",
  "required": ["case_id", "prompt_condition", "model_id", "primary_diagnosis",
               "confidence", "explanation"],
  "properties": {
    "case_id":        {"type": "string"},
    "prompt_condition":{"enum": ["direct","cot","ddx","evidence_first","uncertainty_first"]},
    "model_id":       {"type": "string"},
    "run_id":         {"type": "string"},
    "primary_diagnosis": {"type": "string"},
    "primary_label_mapped": {
      "type": "array",
      "description": "Model diagnosis mapped to the 14-class CheXpert space by a fixed normalizer",
      "items": {"type": "string"}
    },
    "confidence":     {"type": "integer", "minimum": 0, "maximum": 100},
    "explanation":    {"type": "string"},
    "reasoning":      {"type": "string"},
    "extracted_evidence": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["finding","presence"],
        "properties": {
          "finding":  {"type": "string"},
          "location": {"type": "string"},
          "presence": {"enum": ["present","absent","uncertain"]},
          "confidence": {"type": "integer","minimum":0,"maximum":100}
        }
      }
    },
    "differential": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["diagnosis","probability"],
        "properties": {
          "diagnosis": {"type": "string"},
          "probability": {"type": "integer","minimum":0,"maximum":100},
          "rationale": {"type": "string"}
        }
      }
    },
    "cited_findings": {"type": "array","items": {"type":"string"}},
    "uncertainty_statement": {"type": "string"},
    "abstained": {"type": "boolean"},
    "raw_response": {"type": "string"},
    "meta": {
      "type": "object",
      "properties": {
        "views_provided": {"type":"array","items":{"type":"string"}},
        "prompt_version": {"type":"string"},
        "temperature": {"type":"number"},
        "latency_ms": {"type":"number"},
        "prompt_tokens": {"type":"integer"},
        "completion_tokens": {"type":"integer"}
      }
    }
  }
}
```
Enforce with a validator; on invalid JSON, one bounded auto-repair retry, then log as a **format failure** (itself a reportable metric per prompt).

---

## 7. Evaluation Pipeline

Every subject output flows through five automated evaluators + one human spot-check. Keep a strict wall between the **automated reference signals** (CheXbert/RadGraph — cheap, deterministic, primary) and the **LLM-judge signals** (richer, noisier, validated *against* the automated ones).

### 7.1. Diagnostic accuracy
- **Mapping:** normalize free-text `primary_diagnosis` into the 14-class CheXpert space with a **fixed, pre-registered normalizer** (dictionary + embedding fallback), validated on the dev set. Log unmappable outputs.
- **Metrics:** Top-1 accuracy vs. report-derived primary; **multi-label** precision/recall/F1 (micro + macro over 14 classes); **top-k accuracy & MRR** for DDx; **AUROC/AUPRC** only where the model emits per-class probabilities (DDx probabilities) — otherwise report threshold metrics and say so.
- **Uncertain-label policy:** run the standard **U-ones / U-zeros / U-ignore** variants (as in CheXpert literature) and report all three; this prevents a single arbitrary choice from driving results.

### 7.2. Reference-consistency / groundedness (claim-level)
- **Atomize** each explanation + `extracted_evidence` into claims via RadGraph-style extraction (finding + anatomy + presence).
- **Adjudicate each claim** against the report/RadGraph reference into four states (mirrors FActScore-style claim verification, adapted to radiology):
  | State | Definition |
  |---|---|
  | **Supported** | Report asserts the same finding/presence |
  | **Contradicted** | Report asserts the opposite presence |
  | **Unverifiable** | Report is silent (not-mentioned) |
  | **Unsupported-but-plausible** | Not in report, not contradicted, no independent confirmation |
- **Scoring:** Support Rate = Supported / (Supported+Contradicted+Unsupported); **Contradiction Rate** (the safety-critical numerator); % Unverifiable (reporting-bias exposure). Do **not** penalize Unverifiable as if wrong — report it separately.
- **Design C bonus:** in the report-provided variant, "Unverifiable" ~vanishes, giving a clean context-groundedness number that isolates faithfulness from perception.

### 7.3. Hallucination detection
- **Operational definition:** a hallucination is a claim that is **Contradicted** by the reference, OR a **fabricated specific** (measurement, device, prior, laterality) with no image/report basis. Omissions are tracked separately as **missed findings** (false negatives on report-positive labels), not conflated with hallucination.
- **Automatic pipeline:** RadGraph diff (model claims − report claims) → contradiction set; regex/NER for fabricated specifics (numbers, "compared to prior," device names) cross-checked against report; **SelfCheckGPT-style consistency** across the temperature-sampled repeats flags unstable (likely fabricated) claims.
- **Metrics:** Fabricated-Finding Rate (per case, per 100 claims); Contradiction Rate; Omission Rate; **Hallucination-free case rate** (binary, for McNemar across prompts).

### 7.4. Confidence calibration
- **Elicitation:** verbalized 0–100 with the anchored rubric (§5); where a model exposes token logprobs, additionally record sequence/logit-based confidence and compare (tests whether verbalized ≈ internal).
- **Metrics:** reliability diagrams; **ECE** (15-bin, report bin count sensitivity); **MCE**; **Brier score**; **AUROC of confidence vs. correctness** (discrimination, separate from calibration); **Overconfidence rate** (high conf ∧ wrong) and **Underconfidence rate** (low conf ∧ right).
- **Reporting:** calibration must be reported *per difficulty stratum* — global ECE hides the H4 effect.

### 7.5. Reasoning quality (LLM judge)
Rubric dimensions, each **1–5** with explicit anchors (full anchors in §8):
Evidence identification · Logical consistency · DDx quality · Appropriate uncertainty · **Faithfulness** (kept conceptually separate from persuasiveness to test H3) · Clinical usefulness · Diagnostic coherence.
The judge scores **presentation/coherence** and **faithfulness** as *distinct* axes precisely so H3's dissociation can be measured.

### 7.6. Human sanity check (no radiologists) — §9.

### 7.7. Pipeline diagram
```
                 ┌──────────────┐
   Image(s) ───▶ │  Subject VLM │ ──▶ JSON output ──┐
   (IU X-Ray)    └──────────────┘   (5 prompts)     │
                                                     ▼
 Report ─▶ CheXpert labeler ─┐               ┌───────────────┐
        └▶ CheXbert ─────────┼─▶ Reference ─▶│ 1 Accuracy    │
        └▶ RadGraph ─────────┘   (labels +   │ 2 Groundedness│
                                  entities)  │ 3 Hallucin.   │──▶ Metrics DB
                                             │ 4 Calibration │
        LLM Judges (≥2) ────────────────────▶│ 5 Reasoning   │
        (blind, position-swapped)            └───────────────┘
                                                     │
                                             Human spot-check (§9)
```

---

## 8. LLM-as-a-Judge Design

### 8.1. Roles and separation of concerns
Use LLM judges **only** for what automated metrics cannot capture (reasoning quality, persuasiveness, faithfulness nuance). Anchor them by **validating judge scores against the deterministic RadGraph/CheXbert signals** on a labeled dev subset — this is your substitute for radiologist calibration and is itself a contribution.

### 8.2. Which model judges?
- **Recommended primary judge: a strong model from a *different family* than the subject VLM**, to reduce **self-preference bias** (judges favor their own outputs). If subjects include GPT and Gemini VLMs, use **Claude Opus** as one judge and **Gemini** as another, never letting a model judge its own family's outputs on head-to-head items.
- **Use ≥2 independent judges** and report inter-judge agreement. For MIMIC (on-prem), use an **open judge** (e.g., a strong open LLM) to stay compliant.
- Justification: no single judge is "correct"; the defensible claim is *agreement among diverse judges that also correlates with automated ground-truth signals*.

### 8.3. Bias controls (mandatory)
- **Blind:** strip model identity and prompt-condition labels; randomize output IDs.
- **Position bias:** for any pairwise comparison, run **both orders** and average / require order-consistency; discard order-flipped disagreements or send to adjudication.
- **Verbosity bias:** provide token counts to the analyst (not the judge) and include length as a covariate; optionally length-matched sub-analysis.
- **Rubric-based absolute scoring** (1–5 anchored) as primary, **pairwise** as secondary (pairwise is more sensitive but more position-biased).
- **Self-enhancement:** never let a judge score its own family in comparative items.

### 8.4. Judge prompt (rubric-based, per output)
```
You are a careful evaluator scoring a chest X-ray interpretation for a research
study. You are given: (A) a reference radiology report, (B) reference structured
findings, and (C) a candidate interpretation. Score ONLY the candidate.

Score each dimension 1-5 using the anchors. Do NOT reward length or confident
tone. Faithfulness = whether claims are supported by the reference, independent
of how persuasive the writing is.

Dimensions & anchors:
- evidence_identification: 5=all key reference findings addressed; 1=misses all.
- logical_consistency:     5=no internal contradictions; 1=self-contradictory.
- ddx_quality:             5=appropriate, correctly ranked; 1=irrelevant/absent.
- appropriate_uncertainty: 5=uncertainty matches evidence & report hedging; 1=miscalibrated tone.
- faithfulness:            5=every claim supported/uncontradicted by reference; 1=multiple contradictions/fabrications.
- clinical_usefulness:     5=actionable & correct; 1=misleading.
- diagnostic_coherence:    5=evidence→dx logically follows; 1=non-sequitur.

Also flag: contradicted_claims [list], fabricated_specifics [list].
Return JSON: {scores:{...}, contradicted_claims:[], fabricated_specifics:[], justification:"<=60 words}.
```

### 8.5. Pairwise prompt (secondary; for prompt A vs B)
Identical framing, present **Output 1 / Output 2** (order randomized, run twice swapped), ask for winner per dimension + overall + confidence, with "tie" allowed. Aggregate with order-consistency filtering.

### 8.6. Reliability & adjudication
- Report **inter-judge agreement**: Cohen's/Fleiss' κ for categorical flags, **Krippendorff's α** or ICC for ordinal 1–5 scores.
- **Human-judge agreement** on the sanity-check subset (§9) as external validity of the judges.
- **Adjudication:** disagreements > 1 point or contradictory flags → third judge or PI review; log all.

---

## 9. Human Sanity Check (without radiologists)

**Purpose:** not to establish clinical ground truth (you can't), but to validate that (a) automated groundedness scoring matches human reading of the report, and (b) LLM-judge scores track human judgment.

- **Reviewers:** 2–3 medical/graduate researchers (medical students, non-radiology clinicians, the PI). Give them the *report* (the reference), not the image-reading task, so no radiology expertise is required — they check "does the model's claim match what the report says?" which is a reading-comprehension task.
- **Sample:** ~60–100 stratified cases; each reviewed by ≥2 humans.
- **Tasks:** (1) claim-level support labeling (blind to model/prompt) → compare to automated RadGraph verdicts (agreement = validation of the automatic pipeline); (2) 1–5 faithfulness rating → correlate with LLM-judge.
- **Report:** human–automatic agreement (κ/α), human–LLM-judge correlation.
- **Limitations (state plainly):** non-radiologists cannot catch findings the *report itself* missed; this validates report-consistency, **not** clinical correctness. It bounds measurement error in the pipeline, nothing more.

---

## 10. Statistical Analysis Plan

### 10.1. Primary model
**Mixed-effects models** with **case** and **judge** as random effects; **prompt** (and **prompt × difficulty**, H4) as fixed effects.
- Binary outcomes (correct, hallucination-present, abstained): **mixed-effects logistic regression (GLMM)**.
- Continuous/ordinal outcomes (support rate, ECE-per-case, judge scores): **linear mixed model** or ordinal mixed model.
- This correctly handles the repeated-measures (same case, 5 prompts) structure — using unpaired tests here would be a reviewer-fatal error.

### 10.2. Pairwise confirmatory tests (per outcome, prompt-vs-prompt)
- **McNemar's test** for paired binary (accuracy, hallucination-free).
- **Wilcoxon signed-rank** for paired ordinal/skewed (support rate, judge scores, per-case ECE).
- **Paired t-test** only where normality of paired differences holds (check; usually prefer Wilcoxon).
- **Bootstrap CIs** (case-level resampling, ≥10k) for ΔECE, ΔBrier, Δsupport-rate — these lack clean parametric tests.

### 10.3. Effect sizes (report always, not just p-values)
Cohen's d / Cliff's δ (ordinal), odds ratios (GLMM), risk differences for rates. The paper's claims should be about **magnitude**, not just significance.

### 10.4. Multiple comparisons
5 prompts → 10 pairwise per metric, ×several metrics. Control **FDR (Benjamini–Hochberg)** within each metric family; pre-register the primary endpoints (recommend: accuracy, contradiction rate, ECE, faithfulness) so secondary metrics don't inflate the family.

### 10.5. Power
Report the **pilot-estimated** discordance/variance feeding the full-study power calc; state the minimum detectable effect at final N. Don't present a made-up a-priori power.

### 10.6. Assumptions & interpretation
State independence (violated → that's *why* mixed models), distributional checks, and that all inference is **conditional on the report-as-reference** (a fixed, imperfect standard). Interpret effects as "relative to report-derived truth," never as clinical accuracy.

---

## 11. Case-Difficulty Stratification

Difficulty is assigned **from the report/labels, blind to model outputs**, before running.

| Stratum | Operational rule (report-derived) |
|---|---|
| **Easy-Normal** | Report normal; both labelers agree "no finding"; low `XXXX` density |
| **Easy-Abnormal** | Single unambiguous positive finding; labelers agree; no hedging |
| **Intermediate** | 1–2 findings; minor labeler disagreement or mild hedging |
| **Hard / Multi-pathology** | ≥3 positive findings, or devices + pathology |
| **Ambiguous** | Report uses hedge language ("XXXX may represent", "cannot exclude") OR **CheXpert vs. CheXbert disagree** OR uncertain (`-1`) labels present |
| **Rare** | Low-prevalence finding (pneumothorax, mass, pneumomediastinum) |

**Key idea:** *inter-labeler disagreement and report hedging are your automatic ambiguity detector* — no radiologist needed. This directly operationalizes H4. Report every metric **per stratum**; the headline analysis is the **prompt × difficulty interaction**.

---

## 12. Error Taxonomy

| Category | Definition | Illustrative example |
|---|---|---|
| **False positive** | Predicts finding; report negative | "Right lower lobe pneumonia" on a normal CXR |
| **False negative / Missed pathology** | Report-positive finding not reported | Misses a moderate pleural effusion present in report |
| **Unsupported evidence** | Cited finding absent from report, uncontradicted | "Blunted costophrenic angle" when report silent |
| **Contradictory explanation** | Claim opposite to report | "Clear lungs" while report states consolidation |
| **Hallucinated finding** | Fabricated specific (measurement/device/prior) | "Cardiac silhouette 16.2 cm, enlarged vs. prior" — no prior/measure exists |
| **Poor uncertainty handling** | Confidence mismatched to evidence | 95% confident on an image the report calls limited |
| **Incorrect differential** | DDx omits true dx or padded with irrelevant | Lists TB/sarcoid for a simple fracture |
| **Overconfident error** | High confidence ∧ wrong | 92% "normal" on a study with pneumothorax |
| **Underconfident correct** | Low confidence ∧ right | 30% on a correctly called obvious cardiomegaly |
| **Reasoning failure** | Evidence right, conclusion wrong (non-sequitur) | Correct findings, illogical diagnosis |
| **Grounding failure** | Conclusion right, evidence unfaithful (H6) | Correct dx justified by fabricated finding |
| **Calibration failure** | Systematic conf–accuracy gap across cases | Whole prompt condition ECE ≫ others |

Every logged error carries: category, prompt, difficulty stratum, case_id — enabling error-profile-by-prompt figures.

---

## 13. Threats to Validity and Mitigations

| Threat | Risk | Mitigation |
|---|---|---|
| **Dataset bias** | IU X-Ray demographics/scanner-specific | Replicate on MIMIC/CheXpert (Tiers 2–3) |
| **Report bias** | Omitted negatives scored as unsupported | Separate Unverifiable from Contradicted; never penalize the former as wrong |
| **Label noise** | Labeler errors become "truth" | Dual-labeler agreement subset; U-variants; report on high-agreement cases |
| **LLM-judge bias** | Position/verbosity/self-preference | Order-swap, blinding, length covariate, cross-family judges, validate vs. automated + human |
| **Prompt leakage** | Model infers task from wording | Fixed shared preamble; no answer hints; audit prompts for tells |
| **Distribution shift** | Findings differ across datasets | External validation tier; report per-dataset |
| **Automation bias** | Over-trusting automated metrics | Human sanity check; report metric–human agreement |
| **Construct validity** | "Groundedness" ≠ clinical truth | Explicit reference-consistency vs. context-groundedness naming (§0.1); disclaim |
| **Internal validity** | Confounds (verbosity, order) | Repeated-measures design, decoding fixed, covariates |
| **External validity** | Two models, one dataset | ≥2 models, ≥2 datasets, ≥2 judges |
| **Absence of expert annotation** | No clinical gold standard | Frame as *report-consistency* study, not accuracy-vs-radiologist; RSNA bbox cross-check for a non-report evidence anchor |
| **Multiplicity** | False positives from many tests | FDR control + pre-registered endpoints |
| **Scrubbing (IU)** | `XXXX` breaks verification | Exclude claims hinging on scrubbed tokens; report `XXXX` density |

---

## 14. Reproducibility

- **Seeds:** fix all RNG; temperature 0 for main run; log seeds for sampled repeats.
- **Prompt versioning:** every prompt string in git with a semantic `prompt_version`; the exact version stored in each output's `meta`.
- **Model pinning:** record exact model IDs/versions/snapshot dates (API models drift — log the dated snapshot and re-run a fixed canary set to detect silent updates).
- **Config:** Hydra/YAML configs; one config = one experiment; committed.
- **Data manifest:** committed test-set UID list + file hashes (freeze before running).
- **Experiment tracking:** Weights & Biases / MLflow for runs, metrics, artifacts.
- **API logging:** persist full request/response (incl. images refs, tokens, latency, cost) to an append-only store.
- **Environment:** Dockerfile + pinned `requirements.txt`/`uv.lock`; optional Conda lock.
- **Repo structure:**
```
radllm-prompt-eval/
├── configs/            # hydra configs per experiment
├── data/               # manifests, splits, NOT raw PHI
├── prompts/            # versioned prompt templates
├── src/
│   ├── inference/      # VLM runners, schema validation
│   ├── labeling/       # CheXpert/CheXbert/RadGraph wrappers
│   ├── eval/           # accuracy, groundedness, halluc, calibration
│   ├── judge/          # LLM-judge harness, bias controls
│   └── stats/          # mixed models, bootstrap, plots
├── notebooks/          # analysis, figures
├── results/            # metrics DB, run artifacts
├── docker/
├── tests/
└── README.md
```
- **Hardware:** API models need only a workstation. Open VLM/judge on MIMIC → 1–2× 24–48GB GPUs (A6000/A100) for 7B–34B-class models; budget more for the larger set.

---

## 15. Ethical Considerations

- **Not for clinical use:** prominent disclaimer; this is a *measurement study of LLM behavior*, not a diagnostic tool. No deployment claims.
- **Automation bias:** frame results so readers don't over-trust; emphasize failure modes (H3, H6).
- **Privacy/PHI:** MIMIC/CheXpert stay within DUA boundaries; **never transmit credentialed data to non-compliant APIs** (§0.2); IU X-Ray is de-identified but still cite responsibly.
- **Licensing compliance:** honor PhysioNet/Stanford/Kaggle terms; document them.
- **Responsible reporting:** report absolute rates of dangerous errors (missed pneumothorax, fabricated findings), not just averages; avoid hype.
- **Limits of automatic eval:** state that report-derived truth ≠ clinical truth; positive results are about *consistency and behavior*, not readiness.

---

## 16. Contributions (publication-quality claims)

**Methodological (primary):**
- M1. A **radiologist-free evaluation framework** separating diagnostic accuracy, reference-consistency, hallucination, and calibration, with an explicit construct distinction between *reference-consistency* and *context-groundedness*.
- M2. An **automated claim-level groundedness pipeline** (RadGraph-based) with a validated four-state adjudication that correctly handles reporting-bias (Unverifiable ≠ wrong).
- M3. A **bias-controlled multi-judge protocol** validated against deterministic signals and non-expert humans — a template for LLM-judge use in radiology.
- M4. An **automatic difficulty/ambiguity stratifier** using inter-labeler disagreement + report hedging.

**Empirical (secondary):**
- E1. Quantified effect of five prompt formats on accuracy, groundedness, hallucination, calibration (H1–H3).
- E2. Evidence for the **faithfulness–accuracy dissociation** (H6) and the **persuasiveness-without-faithfulness** effect of CoT/DDx (H3).
- E3. The **prompt × difficulty interaction** (H4) and the calibration/abstention trade-off (H2/H7).
- E4. Prompt-engineering recommendations grounded in the above.

Separate these clearly: the framework (M1–M4) is the durable contribution; the empirical numbers (E1–E4) are model-version-dependent.

---

## 17. Figures and Tables

**Figures:** (F1) pipeline schematic; (F2) reliability diagrams per prompt; (F3) error-profile stacked bars by prompt; (F4) prompt × difficulty interaction (accuracy & ECE); (F5) faithfulness vs. persuasiveness scatter (H3 dissociation); (F6) forest plot of paired effect sizes with bootstrap CIs; (F7) judge–human–automatic agreement matrix.

**Tables:** (T1) dataset comparison; (T2) prompt templates + motivations; (T3) primary metrics per prompt (mean ± CI); (T4) pairwise tests + FDR-adjusted p + effect sizes; (T5) hallucination/contradiction/omission rates; (T6) calibration (ECE/MCE/Brier) global + per stratum; (T7) inter-judge & human-agreement reliability; (T8) error taxonomy counts by prompt.

---

## 18. Timeline (graduate-student realistic)

| Phase | Weeks | Deliverable |
|---|---|---|
| 0. Setup, compliance, repo, schema | 1–2 | Env, DUAs, frozen schema |
| 1. Labeling infra (CheXpert/CheXbert/RadGraph) | 3–4 | Reference labels + entities on IU X-Ray |
| 2. Difficulty stratification + test-set freeze | 5 | Locked manifest |
| 3. Prompt finalization + pilot (120–200) | 6–8 | Pilot results, power calc, rubric/judge calibration |
| 4. Full inference run (5 prompts × models) | 9–10 | Complete output DB |
| 5. Automated evaluation | 11–12 | Accuracy/groundedness/halluc/calibration |
| 6. LLM-judge + human sanity check | 13–14 | Judge scores, agreement |
| 7. Stats + figures | 15–16 | All tables/figures |
| 8. Writing + external replication (MIMIC) | 17–20 | Draft + confirmatory tier |
| 9. Revision/pre-registration/submission | 21–24 | Submission |

~5–6 months part-time.

## 19. Compute & Cost (order-of-magnitude)

- **Subject inference:** ~1,000 cases × 5 prompts × 2 models ≈ 10k VLM calls; +50 cases ×3 temp repeats. Commercial VLM cost dominated by image tokens → roughly **low-hundreds to ~$1k USD** depending on provider/pricing (verify current rates).
- **LLM-judge:** ~10k outputs × 2 judges × (rubric + swapped pairwise) → the largest line item; **several hundred to ~$1–2k**. Budget ceiling ~**$2–4k** total API, plus GPU hours for open models on MIMIC (institutional cluster ideally free).
- Always report *actual* logged cost in the paper (reproducibility + honesty).

## 20. Publication Roadmap

- **Primary venues:** **NEJM AI**, **Radiology: Artificial Intelligence**, **JAMIA**, **Lancet Digital Health**; ML venues: workshops at **NeurIPS/ML4H**, **CHIL**, **MICCAI**.
- **Positioning:** lead with the *framework* (radiologist-free, auditable), support with empirical prompt findings — reviewers value durable methodology over a leaderboard of a soon-stale model.
- **Pre-registration** (OSF) of hypotheses/endpoints strengthens acceptance and pre-empts p-hacking criticism.

## 21. Future Extensions

Multimodal grounding with region-level evidence (bbox/attention vs. RSNA); longitudinal/prior-comparison reasoning; agentic tool-use (retrieval of guidelines); fine-tuning vs. prompting comparison; extension to CT/other modalities; a live, versioned public benchmark with a held-out server.

## 22. Anticipated Reviewer Criticisms and Rebuttals

| Criticism | Rebuttal |
|---|---|
| "No radiologist ground truth — results are meaningless." | We *reframe* the target as report-consistency + behavior, not clinical accuracy (§0.1, §3); we validate the automated pipeline against non-expert humans and RSNA localization; every claim is scoped accordingly. |
| "Report-derived labels are noisy." | We use dual labelers, report the high-agreement subset, run U-variants, and separate Unverifiable from Contradicted so reporting bias can't masquerade as error. |
| "LLM judges are unreliable/biased." | Multi-judge, cross-family, order-swapped, blinded, length-controlled; reliability (κ/α) reported; judges validated against deterministic signals + humans; judges used only for what automation can't score. |
| "Groundedness is really just faithfulness-to-report." | Correct — we say so explicitly and name the two constructs; the Design-C ablation isolates context-groundedness. |
| "Results won't generalize / model will be outdated." | Framework is the contribution (M1–M4); we test ≥2 models, ≥2 datasets; empirical numbers are explicitly version-scoped with dated snapshots. |
| "Repeated-measures stats?" | Mixed-effects models with case/judge random effects; paired McNemar/Wilcoxon confirmatory; FDR + pre-registered endpoints. |
| "IU X-Ray is small and scrubbed." | It's the compliant, API-usable workhorse; MIMIC provides scale confirmatorily; we quantify and exclude scrubbing-dependent claims. |

## 23. What Is Established Best Practice vs. Speculative (honesty ledger)

- **Established:** CheXpert/CheXbert labeling; RadGraph entity extraction; ECE/MCE/Brier/reliability diagrams (Guo et al. 2017); McNemar/Wilcoxon/mixed models/FDR; LLM-judge bias controls & position-bias findings (Zheng et al. 2023); verbalized-confidence miscalibration (Tian et al. 2023; Xiong et al. 2024); FActScore-style claim verification (Min et al. 2023); SelfCheckGPT consistency (Manakul et al. 2023); report-gen metrics RadGraph-F1/RadCliQ (Yu et al. 2023). *(Verify each citation before use — do not cite from memory in the manuscript.)*
- **Speculative / this study's bets:** the four-state groundedness adjudication tuned for reporting bias; inter-labeler-disagreement as an automatic ambiguity proxy; using non-radiologist report-reading as pipeline validation; the specific five-prompt taxonomy's effect sizes. Flag these as novel and validate empirically.

---

### Assumptions stated
(1) At least one capable VLM is accessible for image reading; a text-only LLM makes Design A impossible. (2) PhysioNet/Stanford terms permit the compliant paths described — **re-verify current terms**. (3) Automated labelers reach their published accuracy on your data (validate on dev set). (4) Budget/compute as in §19. (5) The report is treated as an imperfect but usable reference; all conclusions are conditional on it.
