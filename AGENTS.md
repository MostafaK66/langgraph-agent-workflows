# Engineering contract

- Support Python 3.11 and 3.12 with production code in `src/`.
- Keep graph definitions, node logic, integrations, and CLI concerns separate.
- Never perform network, model, environment, or database work at import time.
- Inject model, search, input/output, and persistence boundaries.
- Treat model and search output as untrusted; validate before using it.
- Keep state updates deterministic and avoid mutating caller-owned lists or messages.
- Add strict type hints and deterministic offline tests for behavior changes.
- Run `make quality` before committing and report only checks actually executed.
- Never commit credentials, checkpoints, databases, IDE metadata, or environments.
