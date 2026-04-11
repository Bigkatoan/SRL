# Changelog

All notable changes to SRL are documented in this file.

The format follows Keep a Changelog and the project uses Semantic Versioning as a target release model.

## [Unreleased]

### Added
- Structured CLI documentation page.
- Limitations page for current declarative and deployment boundaries.
- Structured ROS 2 YAML schema support in the config layer.
- Shared observation remapping utility used across training, runtime model execution, and ROS 2 inference.
- Initial GitHub Actions workflows for tests and linting.

### Changed
- Top-level package imports are now lazy, so CLI help paths do not fail early on heavyweight runtime imports.
- ROS 2 inference now uses the same observation remapping rules as the training/runtime path.

### Fixed
- `python -m srl.cli.train --help` no longer fails immediately because of eager algorithm imports.
- `python -m srl.cli.visualize --help` no longer fails immediately because of eager utility imports.

## [0.1.0] - 2026-04-12

### Added
- Initial release of SRL with PPO, SAC, DDPG, TD3, A2C, and A3C.
- YAML-driven model building with flow graphs, encoders, heads, and multimodal support.
- Isaac Lab integration, benchmark scripts, checkpointing, and ROS 2 Python API.

[Unreleased]: https://github.com/Bigkatoan/SRL/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Bigkatoan/SRL/releases/tag/v0.1.0