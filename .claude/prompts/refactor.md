# Refactoring Workflow

## Principles

1. **Green before refactor** — All tests must pass before starting
2. **One thing at a time** — Small, focused changes
3. **Run tests after each change** — Catch regressions immediately
4. **Preserve behavior** — Refactor doesn't change external behavior
5. **Document why** — Commit messages explain the refactor

## Steps

1. **Verify current state** — Run tests, ensure all pass
2. **Identify scope** — What exactly is being refactored?
3. **Plan the refactor** — Write to `docs/planning/` if complex
4. **Get user approval** — Iterate on plan
5. **Refactor in stages** — Small commits
6. **Run tests after each stage** — Green light before next stage
7. **Update docs if needed** — Especially if public API changes

## Common Refactors

### Extract Function
- Identify repeated logic
- Extract to named function
- Update callers
- Test still passes

### Rename
- Find all usages (`rg <old_name>`)
- Update in single commit
- Test still passes

### Split Module
- Create new file
- Move relevant code
- Update imports
- Test still passes

### Change Signature
- Update function definition
- Update all callers
- Update tests
- Update docs

## Commit Message Format
```
refactor(scope): describe what changed and why

Optional detail paragraph.

BREAKING CHANGE: if applicable, describe migration path.
```

Examples:
- `refactor(grids/resample): extract _validate_data helper`
- `refactor(io/hdf5): split reader.py into reader.py + parser.py`
- `refactor(core/schema): rename validate_dataset to validate`