# Sai Polymath Data Thesis

Status: active data-design objective. This document changes how Sai selects and
mixes candidate data. It does not itself admit a source or authorize 4B
training.

## Decision

Sai rejects both of these shortcuts:

1. `web == bad`, therefore minimize all web data;
2. successful models used 75--85% web, therefore Sai should copy that ratio.

Web is a **reservoir**, not a quota. It contains valuable contemporary language,
world knowledge, dialogue, practical instruction, journalism, culture, and
technical explanation, but also duplication, SEO, corruption, and synthetic
slop. Sai will retain valuable web documents and discard the rest without
protecting a predeclared web percentage.

The corpus is optimized for a broad, interconnected model of the world. A model
that knows only one technical specialty is not the target. Literature, history,
art, philosophy, biography, law, economics, psychology, sociology, journalism,
and ordinary human life are capability-bearing data, not decorative leftovers.
They sit alongside mathematics, science, engineering, code, and reference
knowledge as first-class parts of the model's education.

## Optimize functions, not source labels

Every retained document must serve at least one explicit function:

- **linguistic range**: natural English across registers, voices, genres, and
  situations;
- **world knowledge**: reliable facts, entities, events, places, institutions,
  practices, and everyday mechanisms;
- **conceptual instruction**: definitions, explanations, examples, exercises,
  misconceptions, and prerequisite-aware exposition;
- **formal structure**: mathematics, proofs, logic, scientific notation, and
  symbolic transformations;
- **procedural competence**: code, debugging, tests, manuals, specifications,
  experiments, and reproducible workflows;
- **human understanding**: narrative, dialogue, motives, emotion, culture,
  ethics, institutions, and social consequences;
- **cross-domain bridges**: documents that accurately connect concepts from
  otherwise separated areas;
- **expert depth**: advanced material that extends an already represented
  prerequisite chain;
- **reasoning practice**: grounded transformations or problems whose result can
  be checked.

Source type remains provenance metadata. It is not a proxy for usefulness. A
paper can be noisy, a forum answer can be excellent, a textbook can be wrong,
and a terse specification can be extraordinarily valuable.

## The Polymath Graph

Sai will represent the selected pool as a multi-label graph rather than one
scalar quality ranking.

### Document metadata

The next labeling schema extends the existing quality, English, domain,
difficulty, prerequisite burden, curriculum phase, pedagogical role, concepts,
risks, evidence spans, and confidence fields with:

- information density;
- factual/reference value;
- educational value;
- reasoning density;
- technical depth;
- formatting/extraction quality;
- human-expression and social-world value;
- style and discourse form;
- likely organic, synthetic, or transformed provenance;
- cross-domain bridge concepts;
- novelty relative to the selected pool.

One comprehensive frontier-model judgment may produce these semantic fields in
bulk. Deterministic checks separately establish facts that prose alone cannot:
source identity, applicable license, exact and near duplication, secrets,
benchmark contamination, byte integrity, and repeated exposure.

### Selection objective

For a candidate document `d` and already selected pool `S`, selection is driven
by marginal utility rather than an upstream dataset name:

```text
utility(d | S) =
    reliability(d)
  * usable_information(d)
  * curriculum_fit(d | S)
  + concept_novelty(d | S)
  + cross_domain_bridge_value(d | S)
  + style_coverage_gain(d | S)
  - duplicate_exposure(d | S)
  - unresolved_risk(d)
```

This expression defines the accounting categories; its numerical weights must
be frozen prospectively before a final stream is selected. No high aggregate
score may override a contamination, provenance, or integrity veto.

## Spiral progression, not four isolated piles

The model should encounter concepts in prerequisite order, but it must continue
rehearsing earlier knowledge while advancing. Sai therefore uses a spiral:

1. **Grounding** -- language, everyday mechanisms, core facts, elementary
   quantitative ideas, introductory science and computation, foundational
   stories, history, and social concepts.
2. **Breadth** -- broad reference knowledge, literature and humanities,
   explanatory science, programming foundations, civic and economic systems,
   and varied natural discourse.
3. **Integration** -- textbooks, worked problems, documentation, comparative
   history, causal explanations, larger programs, and explicit bridges among
   domains.
4. **Reasoning and depth** -- proofs, advanced exercises, research, standards,
   expert technical material, and grounded multi-domain synthesis.

Every later phase retains a declared rehearsal share from earlier phases.
Advanced documents are admitted only when their prerequisite concepts already
have sufficient earlier exposure. Easy material does not disappear merely
because advanced material begins.

## Human-world breadth is mandatory

The final pool must report independent coverage for at least:

- literature, narrative, drama, poetry, rhetoric, and criticism;
- visual art, architecture, music, design, and aesthetics;
- history, biography, geography, archaeology, and religion/mythology;
- philosophy, ethics, logic, law, civics, and political thought;
- economics, finance, organizations, and public policy;
- psychology, sociology, anthropology, education, and communication;
- journalism, dialogue, correspondence, practical life, hobbies, and culture;
- mathematics, computing, engineering, natural sciences, and medicine.

These are coverage dimensions, not mutually exclusive token buckets. A history
of cryptography can support history, mathematics, politics, and computing at
once. Such accurate bridges are particularly valuable.

## Acquisition policy

Current priority order is:

1. preserve brutally filtered broad English rather than preserving a web ratio;
2. acquire open textbooks, reference works, public-domain long-form writing,
   and authored pedagogical sequences;
3. acquire science PDFs and arXiv as expert material, but pair them with
   introductory and intermediate explanations;
4. acquire provenance-preserving code together with documentation, issues,
   pull requests, tests, notebooks, and technical discussion;
5. expand mathematics across exposition, proofs, exercises, worked solutions,
   formal material, and verified code-math interactions;
6. explicitly acquire literature, arts, humanities, social sciences,
   biographies, and culturally situated natural language;
7. use grounded synthetic transformations later to repair measured gaps,
   create prerequisite-aware explanations, and produce verifiable cross-domain
   exercises--not to flood the linguistic foundation with generic model prose.

FineWeb-Edu, Dolma, Nemotron, books, papers, and code repositories are candidate
reservoirs. None receives a protected percentage because a competitor used it.

## What the published recipes do and do not establish

[SmolLM3](https://huggingface.co/blog/smollm3) shows that a progressive
web/code/math schedule can produce a strong 3B model, but its `web` category
combines multiple web corpora and multilingual material. It does not establish
that 75--85% FineWeb is optimal for an English-first Sai model.

[OLMo 3](https://allenai.org/blog/olmo3) supports broad pretraining followed by
harder math, code, science, instruction, and reading-comprehension material. It
is evidence for stage separation and strong cross-corpus curation, not a proof
that its source proportions are universal.

[DataComp-LM](https://arxiv.org/abs/2406.11794) provides strong evidence that
selection quality can outperform much larger uncurated token budgets. It does
not imply that one scalar quality score is sufficient.

Recent [OLMo capability-tracing
work](https://allenai.org/blog/olmo-capability-tracing) found that narrative,
interpersonal, literature, social-life, customer-support, and Q&A data were
especially influential for social reasoning, and that removing influential
literature data damaged SocialIQA more than removing random literature. This
supports treating human expression as capability-bearing evidence while
stopping short of claiming that literature alone guarantees general reasoning.

## Success criterion

Sai's data advantage is not `more tokens` or `less web`. It is a higher fraction
of unique, reliable, capability-bearing tokens arranged so that prerequisites,
breadth, integration, and expert depth reinforce one another. The final stream
must publish coverage, connection, difficulty, style, uniqueness, and repeated-
exposure ledgers in addition to conventional source percentages.
