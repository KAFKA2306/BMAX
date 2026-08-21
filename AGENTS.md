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
5. For implementation changes, run the smallest relevant model/data tests, verify the reviewed revision, and read back production when deployment is in scope.
6. Stop at the verified fixed point. Do not extend a completed benchmark decision into speculative analytics or sales tooling.

## Boundaries

- Unknown contract provisions remain unknown; distinguish confirmed absence from unverified.
- Do not infer credit spreads, liquidity, option exercise, conversion, or realized financing outcomes.
- Do not execute securities transactions, financing, transfers, or account actions.
- Local/browser events are not durable business analytics unless a canonical collection surface proves them.
- Unobserved CI, deployment, user, inquiry, or revenue outcomes remain unverified.

## Completion report

Report material Before -> After coverage/capability, primary evidence and canonical artifact, Issue/PR/commit/check/deployment evidence when applicable, complexity/manual work removed, commercial outcome only when observed, and the remaining blocker.