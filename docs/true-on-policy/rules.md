---
title: True On Policy Working Rules
description: Standing rules for maintaining true-on-policy investigation notes and changes.
---

# Working Rules

## Documentation

- Store compressed engineering notes under `docs/true-on-policy/`.
- Record conclusions, file relationships, risks, decisions, and next steps.
- Do not store raw scratchpad reasoning.
- Tell the user when these notes are updated.

## Code Changes

- Prefer the smallest effective code change that exercises the required behavior.
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
