---
name: midterm-md-ppt
description: Use this skill to generate a markdown document for a 5-slide midterm check-in presentation. The output must include slide content, speaker notes, and concrete image file paths from the repository to support team collaboration.
---
You are an assistant that generates a READY-TO-USE markdown document for a 5-slide project check-in presentation.

This markdown will be used directly by a team to create slides.

---

## 🔴 HARD REQUIREMENTS

- EXACTLY 5 slides
- Output MUST be in MARKDOWN format
- Each slide must include:
  - Title
  - Bullet points
  - Speaker notes
  - Image section
- Images MUST use repository file paths when possible

---

## 🧠 STEP 1: Analyze repository

Search for:

- README.md
- model implementations
- experiment scripts
- results
- logs
- plots
- images

Specifically look for image files:

- .png
- .jpg
- .jpeg

Common folders:

- figures/
- results/
- outputs/
- plots/
- assets/

---

## 🧠 STEP 2: Identify available visuals

For each image found:

- record its relative path
- infer what it represents

Example:

- results/loss_curve.png → training curve
- outputs/sample_prediction.png → qualitative result

---

## 🧠 STEP 3: Generate 5-slide content

Follow strict midterm structure:

1. Recap & scope change
2. Algorithms & baselines
3. Results & analysis
4. Challenges
5. Next steps

---

## 📄 OUTPUT FORMAT (STRICT)

# Project Midterm Check-in Slides

---

## Slide 1 — <Title>

### Content

- bullet
- bullet
- bullet

### Image

Path: `<relative/path/to/image.png>`

Description:
<what this image shows>

If no image available:
Path: `[IMAGE TO GENERATE]`
Suggestion: <what to create>

### Speaker Notes

<what to say>

---

(repeat for all 5 slides)

---

## 📊 Image Usage Rules

- ALWAYS try to use real images from the repo
- NEVER invent fake file paths
- If unsure, DO NOT guess — use placeholder
- Prefer:
  - training curves
  - result tables (can be text)
  - prediction examples
  - pipeline diagrams

---

## 🧠 CRITICAL THINKING RULES

Slides MUST include reasoning:

- why this method
- why results look like this
- why issues happened
- why next steps matter

---

## 🚫 DO NOT

- hallucinate images
- invent results
- use vague placeholders
- skip image section

---

## 📌 FINAL SECTION

At the end, include:

## Assets to Prepare

List:

- missing images to generate
- plots to export
- tables to create

---

## Collaboration Notes

Explain:

- which slides need more work
- which teammate can fill gaps
