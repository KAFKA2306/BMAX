# Repository Agent Contract

## Mission

Own the Bitcoin-treasury convertible-bond benchmark and its reproducible contract/model evidence. Turn SEC/issuer terms and explicitly declared market/model assumptions into comparable issuer/issue records and a trustworthy benchmark product.

## Canonical authority

- Contract terms come from SEC/issuer primary filings and must retain issue identity, filing/source URL, dates, units/currency and observed time.
- Keep `contract_term`, `market_observation` and `model_output` separate in data and UI.
- Model scenarios are not observed prices, fair values, recommendations, or financing-success probabilities.
- Do not duplicate Bitcoin treasury holdings or general market facts owned by another finance repository; reference versioned upstream artifacts when needed.

## Autonomous execution

1. Inspect current `main`, README, open Issues/PRs, benchmark data/schema, model engine, workflows and deployed `/benchmark` surface.
2. Resume the existing canonical workline before adding another dataset, service layer, workflow or Issue.
3. Prefer verified issue/issuer coverage, complete reproducible scenarios, field-level evidence, production benchmark usability, or a measured conversion/reliability blocker.
4. Reuse/delete/consolidate before adding infrastructure. Do not build SaaS/backends unless current evidence shows the existing static product cannot satisfy the bounded outcome.
5. For implementation changes, run the smallest relevant model/data tests and verify the exact reviewed revision.
6. Stop at the verified fixed point. Do not extend a completed benchmark decision into speculative analytics or sales tooling.

## Branch lifecycle

- Aside from the default branch and unavoidable platform-managed/protected branches, a persistent branch is permitted only while it is the head branch of a currently open PR.
- Creating a work branch creates an obligation to open or reuse its canonical PR immediately; do not use branches as backlog, continuation state, backup, archive, or evidence storage.
- After a PR is merged or closed, delete its head branch after verifying PR/main state. A branch with no open PR is an orphan and must be deleted.
- Before and after work, compare repository branches with open PR heads. Do not report cleanup/fixed point while an orphan task branch remains.
- If the available tool cannot delete a branch, record that as a tooling blocker and do not claim cleanup complete. Never create another orphan branch as a workaround.

## Merge and release are separate

### PR merge conditions

A PR may merge when the repository-local benchmark/data/model contract is correct on the exact head revision: primary contract evidence and model boundaries remain valid, relevant deterministic tests pass, generated benchmark artifacts are reproducible when affected, and no unresolved review or correctness blocker remains.

A deployed `/benchmark` URL, production analytics, qualified inquiry, live market observation after merge, or commercial outcome is **not** a merge condition unless the PR specifically changes the release mechanism and pre-merge validation of that mechanism belongs to the bounded change.

### Product release conditions

Release is a separate post-merge decision. Treat the benchmark product as released only after the merged `main` revision and actual public release surface are read back and verified, including deployment identity, material benchmark flow, published artifacts, and rollback path when applicable. Commercial/user outcomes may be measured after release but are not proof of repository-local merge correctness.

A merged PR does not prove product release. A release blocker does not retroactively invalidate a correctly merged repository change. Report merge and release independently.

## Boundaries

- Unknown contract provisions remain unknown; distinguish confirmed absence from unverified.
- Do not infer credit spreads, liquidity, option exercise, conversion, or realized financing outcomes.
- Do not execute securities transactions, financing, transfers, or account actions.
- Local/browser events are not durable business analytics unless a canonical collection surface proves them.
- Unobserved CI, deployment, user, inquiry, or revenue outcomes remain unverified.

## Completion report

Report material Before -> After coverage/capability, primary evidence and canonical artifact, Issue/PR/commit/check evidence, then report `merged` and `released` separately with direct evidence for each. Include branch cleanup state, commercial outcome only when observed, complexity/manual work removed, and the remaining blocker.