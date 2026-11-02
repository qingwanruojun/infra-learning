---
name: experiment-report-editor
description: Review an existing coding experiment against its code, IR, artifacts, and logs; then write a natural Chinese learning report and organize a dayN delivery with code, note, and a root README. Use when the user asks to revise an experimental learning report or package the finished experiment. Do not use for designing or running a new benchmark.
---

# Experiment report editor

Use the user's requested scope and format. This skill is for reporting **existing** work; it does not authorize new device runs, changing experimental code, publishing, or pushing Git commits.

## Establish the evidence before writing

- Read the current package, the previous report, and one or two earlier reports the user considers good. Keep their voice and level of detail; do not copy claims or structure blindly.
- Check code, commands, raw logs, generated IR/artifacts, exit status, and dates. Separate four stages: IR conversion, device compilation, NPU execution, and numerical validation. An output file or a success string alone does not prove a later stage.
- When an experiment was revised, identify which logs are current and which are historical. Preserve useful failed attempts, but mark their role. Say explicitly when review was offline rather than a fresh device run.
- Write down the few material gaps before drafting: missing check, narrow input scope, compiler workaround, unsupported performance claim, or an inference presented as observation. Fix prose directly; propose a code change only if it is within the user's request.

## Write for a reader

- Open with one or two numbered contributions: concrete change, observable outcome, and the key limit. Then explain the experiment in the order a colleague would ask about it: what was tried, what happened, what broke, how it was handled, and what remains untested.
- Prefer natural first-person Chinese for a personal learning report. Use paragraphs for the story and short tables only for numbers or evidence mappings. Avoid a long audit checklist, repeated “通过” rows, or inflated claims like “全链路” when the run used different inputs or a manual toolchain splice.
- Put a nearby log or code path beside each important number. Distinguish measured result, source/document explanation, and inference in the sentence rather than adding boilerplate labels everywhere.
- Describe current limits precisely: a single op through a fusion pipeline does not establish multi-op fusion; a crash on sampled inputs does not establish failure on every input; no profiler data means no performance conclusion.

## Package only when requested

For a day-level handoff, a useful default is `dayN/code/` for scripts, source, IR, binaries and raw logs; `dayN/note/` for the main report, technical notes and worklog; and one root `dayN/README.md` for orientation, reproduction, environment, evidence index, and limits. Adapt to the existing repository convention. Preserve unrelated user changes.

After moving files, update scripts and document paths, inspect the final tree, and verify referenced files and key artifacts. A Feishu-ready copy should keep the main report's body identical and change presentation only. Do not claim an online Feishu document was created unless it was actually written there.

Report exactly what was verified and what remains uncommitted or unpublished. Do not commit or push solely because a report was packaged.
