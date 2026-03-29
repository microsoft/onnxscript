---
name: multi-agent-coordination
description: How to coordinate multiple AI agents working on the same Git repository simultaneously. Covers worktree isolation, commit coordination, verification gates, context management, and failure recovery patterns. Use this skill when managing parallel workstreams across multiple agents on a shared codebase.
---

# Multi-Agent Coordination

Practical guide for a project lead coordinating 10–17 AI agents working on a single codebase simultaneously. Each section documents a real failure mode encountered in practice and the pattern that prevents it.

---

## 1. Worktree Isolation (CRITICAL)

**Problem**: Multiple agents sharing a single working directory caused file contamination. One agent's uncommitted changes were overwritten by another checking out a different branch. Work was lost and had to be redone.

**Solution**: Each agent gets its own git worktree.

```bash
# Setup — run this before delegating work to an agent
git worktree add .worktrees/<role>-<agent-id> <branch> 2>/dev/null || true
cd .worktrees/<role>-<agent-id>
git fetch origin && git checkout <branch> && git pull
```

**Rules**:
- NEVER let two agents work in the same directory.
- Even "read-only" exploration can cause issues if an agent runs tests that generate artifacts or cache files.
- The main working directory (`/home/.../repo`) is reserved for the lead. Agents always use their worktree.
- Worktrees live at `.worktrees/<role>-<agent-id>` — predictable names make auditing easy.

**Cleanup**: Remove worktrees after agents are done with `git worktree remove .worktrees/<role>-<agent-id>`. Do NOT clean up while agents are still active — you will destroy their workspace.

---

## 2. Commit Coordination Protocol

Multiple agents committing to the same branch requires discipline to avoid divergence.

**Rules**:
1. **Pull before commit**: Always `git pull origin <branch> --rebase` immediately before committing.
2. **Push immediately after commit**: Don't let commits sit unpushed. Other agents may be waiting on your changes or about to conflict with yours.
3. **Atomic commits**: Each commit must be self-contained and leave tests passing. Never commit half-finished work.
4. **Descriptive commit messages**: Include what was fixed and the relevant test outcome. Other agents (and the lead) need to understand the diff at a glance.
5. **Conflict resolution**: If `pull --rebase` fails, the agent should report the conflict back to the lead rather than force-pushing or making arbitrary merge decisions.

**Pattern for each agent commit**:
```bash
git pull origin <branch> --rebase     # sync first
git add <specific-files>              # never git add -A (picks up others' work)
git commit -m "fix: ..."
git push origin <branch>              # push immediately
```

---

## 3. Verification Gates

Checks at each stage prevent problems from compounding across agents.

### Before starting work
- Verify the branch exists and is up to date.
- Run the relevant tests to establish a baseline — know what was already failing before you touched anything.
- Check for uncommitted changes left by previous agents in your worktree.

### After each commit
- Run the affected tests immediately (don't batch; catch regressions early).
- Verify the commit landed on the correct branch (`git log --oneline -3`).
- Confirm the push succeeded.

### Before final review (lead audit)
- Check ALL worktrees for unpushed commits or uncommitted changes.
- Verify the total commit count matches expectations.
- Run the full test suite from a clean checkout (not a worktree).
- Confirm no worktree has diverged from the remote branch.

### Before PR creation
- Triple review: code correctness, critical logic, readability — ideally by three separate agents.
- All review findings addressed.
- Final test run passes.
- PR description updated with an accurate scorecard of what changed.

---

## 4. Task Dependency Management (DAG)

Track task dependencies as a directed acyclic graph (DAG).

**State machine**: `pending → ready → running → done`

A task becomes `ready` only when ALL its dependencies are `done`.

**Dependency rules**:
- Review tasks depend on ALL implementation tasks (can't review code that isn't written).
- PR creation depends on ALL reviews passing.
- Integration tests depend on implementation being complete.
- Exploration/investigation tasks have no dependencies — launch them first.

**Anti-pattern**: Circular dependencies. Reviews depend on implementation, not vice versa. If a review finds a bug, create a new implementation task — don't loop the dependency graph.

---

## 5. Parallel Execution Patterns

### Safe to parallelize
- Independent model/file fixes (different files, no shared state)
- Code review + readability review + critical review (all read-only)
- Investigation and exploration tasks
- Golden data generation for different models
- Any tasks touching completely separate files

### Must be serialized
- Two agents editing the same file
- Commit + push sequences within the same agent (not cross-agent)
- Tasks with explicit DAG dependencies (review after implementation)
- Branch cleanup before final review

### Optimal parallelism
4–6 agents working simultaneously is the sweet spot. Beyond that, coordination overhead increases and merge conflicts become more likely. Scale back if you see frequent push failures or agents blocking each other.

---

## 6. Agent Role Specialization

Match the agent role to the task type. Mismatched roles waste agent capacity.

| Role | Best For | Avoid |
|------|----------|-------|
| **Architect** | Investigation, analysis, design decisions, understanding existing code | Simple mechanical fixes |
| **Developer** | Implementation, bug fixes, refactoring | Open-ended exploration |
| **QA Tester** | Verification, auditing, smoke tests, checking other agents' work | New implementation |
| **Code Reviewer** | Correctness, logic bugs, API contracts | Style opinions |
| **Critical Reviewer** | Security, edge cases, invariant violations | Routine fixes |
| **Readability Reviewer** | Naming, clarity, documentation | Implementation work |

**Delegation tip**: Developers should receive clear instructions — specific files to change, specific test commands to run, and a concrete definition of "done." Architects are better suited for ambiguous investigation tasks.

---

## 7. Communication Patterns

**Agent → Lead**: Status updates after significant steps, completion summaries, blocker notifications. Don't wait until fully done — send progress updates so the lead can coordinate.

**Lead → Agent**: Task delegation with full context (worktree path, setup commands, what to change, test commands, commit message format, definition of done).

**Agent → Agent**: Coordinate through the lead. Agents should not directly orchestrate other agents unless explicitly set up as a sub-lead.

**Anti-patterns**:
- Sending new tasks to a context-saturated agent. If they don't respond correctly, use a different agent.
- Vague task prompts ("fix the MLP stuff"). Be specific about files, methods, and expected outcomes.
- Assuming agents share context. Each agent starts fresh — always include relevant background in the task prompt.

---

## 8. Failure Recovery

| Failure | Recovery |
|---------|----------|
| **Lost work** | Check worktrees, `git stash list`, `git reflog`. Most work is recoverable. |
| **Broken tests** | Have the responsible agent fix it immediately. Don't leave broken tests for later. |
| **Merge conflict** | Have the agent that caused the conflict resolve it — they have the most context. |
| **Agent stuck/looping** | Terminate and create a fresh agent. Don't try to unstick a saturated agent. |
| **Wrong branch** | Cherry-pick commits to the correct branch (`git cherry-pick <sha>`). Don't redo work. |
| **Agent pushed to wrong branch** | Revert the wrong-branch commit, cherry-pick to the right branch. Coordinate with lead before force-pushing. |
| **Diverged worktree** | `git fetch origin && git rebase origin/<branch>` from the worktree. |

---

## 9. Single-Branch Strategy

For large multi-agent efforts, use **one shared feature branch** rather than one branch per agent.

**Why**: Avoids complex multi-branch merge scenarios at the end. All agents commit to the same branch via their isolated worktrees.

**Trade-offs**:
- More `pull --rebase` cycles per agent.
- Occasional push conflicts (recoverable with rebase).
- BUT: much simpler final state, single PR, linear history.

**Alternative**: Separate branches per agent merged via separate PRs — only viable when features are truly independent and don't share files.

---

## Checklist: Delegating a Task to an Agent

Before sending a task, verify your prompt includes all of the following:

1. ☐ **Worktree path** — where the agent should work (`.worktrees/<role>-<id>`)
2. ☐ **Setup commands** — `conda activate`, `git fetch`, `git checkout`, `git pull`
3. ☐ **What to change and why** — specific files, functions, or test cases
4. ☐ **Test commands** — what to run after changes to verify correctness
5. ☐ **Commit message format** — so the history is readable
6. ☐ **Push reminder** — explicit instruction to push after committing
7. ☐ **Definition of done** — what "finished" looks like (tests passing, specific output, etc.)

**Example delegation prompt**:
```
Work in .worktrees/developer-<id>. Setup:
  git fetch origin && git checkout justinchu/fix-model-parity && git pull

Fix the Phi-2 MLP weight mismatch: replace the gated MLP (gate_proj/up_proj/down_proj)
with FCMLP (fc1/fc2) in src/mobius/models/phi.py. The HF weights use fc1/fc2.

After changes, run:
  python -m pytest tests/build_graph_test.py -k "phi" -q

Commit message: "fix: use FCMLP for Phi-1/2 model (HF uses fc1/fc2 not gated MLP)"
Push to justinchu/fix-model-parity immediately after committing.
Done = tests pass, push confirmed.
```
