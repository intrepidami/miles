---
title: True On Policy Working Rules
description: Standing rules for maintaining true-on-policy investigation notes and changes.
---

# Working Rules

## Workflow

- Before every action, list the plan explicitly.

## Documentation

- `docs/true-on-policy/` is the primary reference source. Read it before starting any task in this area.
- Record every new finding, decision, or rule immediately after established.
- Use concise, factual language. No reasoning narrative.
- Files here take precedence over memory when they conflict.
- Tell the user when notes are updated.

## Code Changes

- Make the smallest possible change. Prefer zero source modifications.
- Add new files (tests, wrappers, configs) before touching existing source.
- Avoid refactors unless they directly reduce the change needed for true-on-policy testing.

## Test Work

- Focus test scripts on training/inference log-prob consistency.
- Prefer tests that hit the framework's existing true-on-policy assertion path instead of duplicating the checker.
- Keep smoke/debug modes short enough for quick iteration, then scale only when the signal requires it.

## Git

- Commit after each minimal complete change.
- Keep each commit focused on one logical change.
- Use commit messages that describe the repository change only.
- Do not include assistant/collaboration wording in commit messages.

## Skills And Hooks

- Generate a skill only when a repeated workflow needs reusable instructions beyond this repository.
- Generate a hook only when a repeated check or automation should run without manual prompting.
- Tell the user before or when adding or updating any skill or hook.
- Keep skill/hook changes separate from ordinary code or docs commits.
