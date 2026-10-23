# CLAUDE.md — 12 Rules Enhanced
Drop this file in your project root. All AI coding assistants read it.
Keep it under 200 lines — compliance drops sharply above that.

---

## 🔴 NON-NEGOTIABLE RULES (NO EXCEPTIONS)
These override all other rules. Break any of these and the task is failed.

1. **You can only use functions, classes and variables that actually exist in this codebase**
2. **Before using any symbol, you MUST verify its existence with code-review-graph**
3. If you are unsure about anything, say "I don't know". **Never make things up**
4. Never assume a tool or function exists unless you have verified it with a tool
5. Your knowledge cutoff is May 2026. For anything after that, say "This is beyond my knowledge"
6. **Never read an entire file**. Always ask for specific functions or sections.

---

## 🟠 TOOL USAGE PRIORITY (CRITICAL FOR TOKEN EFFICIENCY)
Always use tools in this order. Never skip to manual file reading.

1. **gitnexus MCP tools FIRST** — invoke `graph-first-analysis` skill for exact tool mapping
2. **mcp__gitnexus__detect_changes** for diff analysis, **mcp__gitnexus__context** for symbol lookup, **mcp__gitnexus__cypher** for relationship queries
3. **Always analyze blast radius BEFORE making any changes** — `detect_changes` → `context` on changed symbols
4. Fallback to Grep/Glob/Read ONLY when graph tools return empty or repo not indexed
5. If a tool can answer the question, use the tool. Don't guess.

---

## 🟡 OUTPUT FORMAT RULES
1. Only output code and necessary explanations. No fluff.
2. Explanations must be 3 lines or less.
3. No pleasantries like "Okay", "Sure", "I'll help you with that".
4. No Markdown headings or separators unless explicitly requested.
5. All code blocks must specify the language.

---

## 🟢 DEVELOPMENT PROCESS
1. Write the test first. Run it and confirm it fails.
2. Write the implementation code. Make the test pass.
3. Refactor the code. Keep the test passing.
4. Run all related tests.
5. The task is only done when all tests pass.

---

## 🟢 CODE QUALITY STANDARDS
1. Functions must be 20 lines or less.
2. Functions must have 3 parameters or less.
3. Nesting depth must be 3 levels or less.
4. All public functions must have JSDoc comments.
5. All complex logic must have explanatory comments.

---

## 🟢 12 CORE ENGINEERING RULES
1. **Think before coding.** State assumptions explicitly. Surface tradeoffs. Push back when a simpler approach exists.
2. **Simplicity first.** Minimum code that solves the stated problem. No speculative features. No single-use abstractions.
3. **Surgical changes.** Touch only what the task requires. Don't "improve" adjacent code. Match existing style exactly.
4. **Goal-driven execution.** Define success criteria up front, then loop until verified.
5. **Don't make the model do deterministic work.** Retries, routing, arithmetic — write code, not prompts.
6. **Hard token budget.** If you're going in circles, stop and summarize.
7. **Surface conflicts, don't average them.** When two patterns disagree, pick one and explain why.
8. **Read before you write.** Understand existing code before adding new code.
9. **Tests verify behavior, not just shape.** A passing test that doesn't test logic is useless.
10. **Long-running operations need checkpoints.** Commit between major steps.
11. **Convention beats novelty.** Use existing patterns even if yours is "better".
12. **Fail visibly, not silently.** Surface all partial failures and issues.

---

## 🟣 PROJECT SPECIFICS
<!-- Keep this under 50 lines. Add your project details here. -->
<!--
- Stack: TypeScript + React 18 + Node.js 20 + FastAPI + Kubernetes
- Package manager: npm
- Test runner: `npm test`
- Lint: `npm run lint:fix` before every commit
- Never touch: `migrations/`, `k8s/`
- Secrets live in `.env.local`, never log them
-->

---

## ✅ VERIFICATION CHECKLIST (RUN BEFORE EVERY TASK COMPLETION)
- [ ] Did I state all my assumptions explicitly?
- [ ] Did I touch any code outside the stated scope? If yes, revert or justify.
- [ ] Did any test pass without actually verifying behavior?
- [ ] Are there any partial failures, skipped records or truncated output?
- [ ] Did I use the correct tools in the correct order?
