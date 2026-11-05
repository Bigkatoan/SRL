# Security Policy

## Supported versions

SRL is pre-1.0. Security fixes are made on `main` and shipped in the next
release; only the latest release receives fixes.

## Reporting a vulnerability

Please **do not open a public issue** for security problems.

Report privately via GitHub's
[private vulnerability reporting](https://github.com/Bigkatoan/SRL/security/advisories/new)
or by email to **bigkatoan6969@gmail.com**. Include a description, steps to
reproduce, affected versions and, if possible, a suggested fix.

You can expect an acknowledgement within 7 days. Once a fix is available we
will publish an advisory and credit you unless you prefer to stay anonymous.

## Scope notes

- **Checkpoints are code.** `srl.utils.checkpoint.CheckpointManager.load`
  uses `torch.load(..., weights_only=False)` for `.pt` files, which can execute
  arbitrary code from a malicious file. Only load checkpoints you trust; prefer
  the `.safetensors` format where possible.
- YAML configs are parsed with `yaml.safe_load`.
