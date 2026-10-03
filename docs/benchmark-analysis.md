# Day 17 benchmark analysis

## Evidence boundary

This is a candidate offline result, not expert/domain approval or production-readiness evidence. The benchmark used the frozen Vietnamese datasets in `data/`, deterministic offline agent paths, the same turns for both agents, fresh recall threads, and a temporary state directory per suite. Token counts use the project heuristic estimator (`approximately characters / 4`), not a provider tokenizer.

## Measured results

### Standard Benchmark

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 1255 | 12954 | 0.00 | 0.00 | 0 | 0 |
| Advanced | 2263 | 26557 | 1.00 | 1.00 | 382 | 0 |

### Long-Context Stress Benchmark

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 328 | 22476 | 0.00 | 0.00 | 0 | 0 |
| Advanced | 440 | 14422 | 1.00 | 1.00 | 253 | 4 |

## Interpretation

- Fact: Advanced reached full measured recall on both suites, while Baseline reached zero on fresh-thread recall. This is the expected consequence of writing stable facts to `User.md`; Baseline only retains messages keyed to the original thread.
- Fact: Advanced processed 8,054 fewer prompt-estimate tokens than Baseline in the stress suite, a reduction of approximately 35.8% (`(22476 - 14422) / 22476`). Four compactions were recorded in that run.
- Fact: Advanced used more generated-agent tokens than Baseline in both suites. In the standard suite it also processed more prompt-estimate tokens because persistent profile context is carried on every turn and the short conversations did not trigger compaction.
- Inference: Compact memory is most valuable when the thread is long enough for unbounded history to dominate the prompt. It is not expected to win every short-conversation cost comparison.
- Fact: Advanced created persistent memory growth while Baseline created none. This is a real storage/maintenance cost and creates a need for correction handling, confidence policy or future memory decay.
- Limitation: `Response quality` is a transparent expected-string heuristic, not a human or expert judgment. A score of `1.00` means all expected strings appeared, not that the answer was natural or factually complete beyond the benchmark gold strings.

## Reproduction

From the repository root:

```powershell
python src/benchmark.py
pytest src/test_agents.py -v
```

In this workspace, the system `python`/`pytest` command was unavailable and the repository virtual-environment executable could not start directly. The available Python 3.12 runtime ran the repository's pytest 9.1.1 package from `.venv` with third-party plugin autoload disabled; result: `4 passed`. The benchmark command completed offline with exit code 0. No dependency was installed and no live API call was made.

## Remaining limitations

- Live provider constructors are lazy-smoke tested only; credentials, network behavior and provider-specific response quality remain unknown.
- The token estimator is intentionally heuristic, so the measured numbers are for relative comparison within this lab, not billing or latency claims.
- The current profile extractor is rule-based and dataset-oriented. Broader languages, ambiguous facts and sophisticated memory decay remain out of scope.
- No human domain reviewer has approved the result. The deliverable remains `candidate`; the recorded review is a bounded coordinator self-review, not an independent or production-readiness approval.
