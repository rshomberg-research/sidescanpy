# Feature Planning Template

Use this workflow for planning new features or major refactors.

## Steps

1. **Understand requirements** — Ask user clarifying questions
2. **Read existing code** — Use Read/Grep to understand current state
3. **Write plan** — Save to `../notes/sidescanpy/<feature-name>.md` (parent workspace, not this repo)
4. **Get user approval** — Iterate on plan
5. **Implement in stages** — One logical change at a time
6. **Update docs** — Reflect changes in relevant docs

## Plan Template

```markdown
# <Feature Name>

**Version:** <target version>
**Status:** planning / in-progress / complete
**Started:** <date>

## Overview
<1-2 paragraph summary>

## Motivation
<why this is needed>

## Design Decisions
<key architectural choices>

## Implementation Plan

### Stage 1: <name>
- Files affected
- Changes
- Tests needed
- Acceptance criteria

### Stage 2: ...

## Risks & Mitigations
<what could go wrong, how to handle>

## Open Questions
<things to resolve before implementation>
```
