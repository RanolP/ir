## [0.3.1-pre] - 2026-03-03

### Bug Fixes

- Auto CPU fallback when Metal context creation fails in sandboxed environments ([`299780a`](https://github.com/vlwkaos/ir/commit/299780a))
- Open search connections immutable/read-only to avoid WAL shm writes in sandbox ([`aa307c2`](https://github.com/vlwkaos/ir/commit/aa307c2))

### Other

- Move config and data to XDG-style `~/.config/ir` (cross-platform, sandbox-accessible) ([`295b3bb`](https://github.com/vlwkaos/ir/commit/295b3bb))
- DSPy optimizer: add structured logging, `--resume` flag, ollama smoke-test ([`2fdd0e9`](https://github.com/vlwkaos/ir/commit/2fdd0e9))
