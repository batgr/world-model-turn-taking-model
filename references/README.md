# Research references

Zotero is the bibliographic source of truth for this project. This directory
defines how papers, reports and other research resources are organised and how
the curated Zotero collection is exported into the repository.

The repository must not become a second manually maintained reference manager.
Bibliographic metadata is edited in Zotero; project hypotheses, interpretations
and decisions live in the project documentation.

## Zotero collection structure

Create one top-level collection:

```text
Turn-WM
├── 00 Core
├── 01 Turn-taking and HRI
├── 02 JEPA and representation learning
├── 03 World models and planning
├── 04 Actions and inverse dynamics
├── 05 Multimodal and social cues
├── 06 Evaluation and probing
├── 07 Datasets and benchmarks
├── 08 Regularization and normalization
└── 99 Inbox
```

A paper may belong to several collections. Collections answer "where should I
browse?"; tags answer "why does this matter to the project?".

## Tags

Use a small controlled vocabulary instead of creating a new tag for every
paper.

```text
status/to-read
status/reading
status/read

role/core
role/method
role/evidence
role/alternative
role/negative-result
role/future

version/v1
version/v2
version/v3-actions
version/v3-planning
version/v4-hierarchical

topic/turn-taking
topic/jepa
topic/world-model
topic/planning
topic/action-grounding
topic/inverse-dynamics
topic/latent-actions
topic/multimodal
topic/probing
topic/domain-generalization
topic/sigreg
topic/normalization
```

When a source is tied to a concrete project hypothesis, also add the
hypothesis id from `docs/research_program.md`, for example
`hyp/H-V2-01`.

## Item-note template

For papers that affect the project, keep one short Zotero note with this
structure:

```text
Project relevance
- Why is this source in the library?

Supported claim
- What concrete claim or method does the source support?

Limits
- What does it not establish for our conversational setting?

Project implication
- Which architecture choice, metric, hypothesis or experiment does it affect?

Linked hypothesis / experiment
- H-...
- V...

Decision impact
- none / motivates test / supports decision / contradicts decision
```

Do not copy large paper summaries into the repository. Zotero notes hold the
paper-level reading record; the research program records only the implications
for this project.

## Better BibTeX export

Install Better BibTeX for Zotero, then export the top-level `Turn-WM`
collection as **Better BibTeX** to:

```text
references/library.bib
```

Enable **Keep updated** so changes to the Zotero collection update the export
automatically. Sort the export by citation key to keep git diffs stable. Do
not export attachments into the repository.

`references/library.bib` is therefore a generated, reproducible export. It
may be committed, but it must not be edited by hand. Zotero remains the source
of truth.

The Zotero Web API also supports BibTeX/BibLaTeX export, so a later CI or
remote-sync workflow can use the API if local auto-export becomes
inconvenient. Do not add API keys to the repository.

## Intake workflow

When a new resource looks useful:

1. Add it to `Turn-WM/99 Inbox`.
2. Add a DOI/arXiv identifier and verify metadata/version.
3. Read enough to classify its role and add topic/version tags.
4. If it changes a project hypothesis, add the corresponding `hyp/H-...` tag
   and update `docs/research_program.md`.
5. Move it to the relevant collection(s).
6. Only create/update a decision record when project evidence is sufficient to
   accept or reject an architectural/experimental choice.

A paper can motivate a hypothesis. It cannot, by itself, establish a result in
this project.

## Initial seed

These are the sources already used in the project or directly tied to planned
experiments. Import them into Zotero rather than recreating their metadata in
`library.bib` manually.

| Resource | Identifier | Project role |
| --- | --- | --- |
| LeJEPA | arXiv:2511.08544 | SIGReg / JEPA foundations |
| LeWorldModel | arXiv:2603.19312 | primary model and training reference |
| V-JEPA 2 | arXiv:2506.09985 | action-conditioned JEPA and planning |
| What Drives Success in Physical Planning with Joint-Embedding Predictive World Models? | arXiv:2512.24497 | planning / window and representation evidence |
| DINO-WM | arXiv:2411.04983 | planning on pretrained latent features |
| FF-JEPA | arXiv:2606.09311 | long-horizon latent planning |
| Hierarchical Planning with Latent World Models | arXiv:2604.03208 | planned hierarchical architecture |
| Learning Latent Action World Models In The Wild | arXiv:2601.05230 | continuous/constrained latent actions |
| Delta-JEPA: Learning Action-Sensitive World Models via Latent Difference Decoding | arXiv:2606.31232 | action-sensitive transition geometry |
| Toward Physically Grounded JEPA World Models for Goal-Conditioned Robotic Planning | arXiv:2609.03565 | IDM, state alignment and transition-subspace analysis |
| D-JEPA: A Decision-Aligned Latent World Model | arXiv:2609.24749 | prediction vs decision-alignment gap |
| Temporally Centered SIGReg Improves Multi-Task LeWorldModel Learning | arXiv:2607.26924 | possible follow-up if lambda trade-off persists |
| Voice Activity Projection | arXiv:2205.09812 | turn-taking future activity modelling |
| Turn-taking in Conversational Systems and Human-Robot Interaction: A Review | DOI:10.1016/j.csl.2020.101178 | HRI framing and action semantics |
| Applying General Turn-taking Models to Conversational HRI | arXiv:2501.08946 | transfer of turn-taking models to robots |
| Designing and Interpreting Probes with Control Tasks | ACL Anthology D19-1275 | probe controls and shortcut analysis |
| Information-Theoretic Probing for Linguistic Structure | ACL 2020 | representation probing methodology |
| Understanding intermediate layers using linear classifier probes | arXiv:1610.01644 | linear-probe foundations |

The seed is deliberately small. Zotero can contain additional resources that
are interesting but not yet tied to a project hypothesis.
