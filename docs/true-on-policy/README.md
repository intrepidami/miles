---
title: True On Policy Notes
description: Compact working notes for true-on-policy related code reading and changes.
---

# True On Policy Notes

This directory is a lightweight workspace for compressed engineering notes around
true-on-policy work.

## Rules

- Keep notes concise and actionable.
- Record conclusions, file ownership, risks, and next steps.
- Do not store raw scratchpad reasoning.
- Prefer repository-relative file paths so notes stay stable across machines.

## Files

- `overview-zh.md` - 中文说明：实现了什么、怎么实现、有什么用（相比 main 的完整变更说明）。
- `rules.md` - standing working rules for this investigation thread.
- `candidate-scripts.md` - closest existing scripts for true-on-policy test work.
- `worklog.md` - chronological summaries of investigations and decisions.
- `code-map.md` - compact map of relevant modules and ownership boundaries.
- `tools.md` - changed files, launch flags, and usage for logprob/hidden-state consistency metrics.

## Usage Contract

This directory is the primary reference source for all true-on-policy work.

- Read relevant files here before starting any new task in this area.
- Record every new finding, decision, or rule here immediately after established.
- Use concise, factual language. No reasoning narrative.
- Files here take precedence over memory when they conflict.
