# Contributing to Talk to Text

Thanks for contributing. This document defines the workflow, quality bar, and review expectations for this repository.

## Scope

Contributions are welcome for:

- Bug fixes
- UI/UX improvements
- Platform compatibility (macOS, Windows, Linux)
- Performance improvements in preprocessing/transcription flow
- Packaging/release automation
- Documentation improvements

## Before You Start

1. Check existing issues and pull requests to avoid duplicate work.
2. For non-trivial changes, open an issue first and propose approach/scope.
3. Keep pull requests focused. Large mixed changes are harder to review and maintain.

## Development Setup

### Prerequisites

- Python `3.13`
- `pipenv`
- `git`

Optional local tools:

- `ffmpeg` (if unavailable, app can download a local copy on first run)
- `cmake` and build toolchain (required when bootstrapping `whisper.cpp` locally)

### Install dependencies

```bash
pipenv install
```

### Run the app

UI mode:

```bash
make ui
```

CLI mode:

```bash
make run
```

Direct module invocation:

```bash
pipenv run python -m transcriber.ui_app
pipenv run python -m transcriber.main -v
```

## Project Layout

- `transcriber/ui_app.py`: Qt desktop UI entrypoint (`PySide6`)
- `transcriber/main.py`: CLI entrypoint for directory batch transcription
- `transcriber/preprocessing/`: ffmpeg acquisition + audio normalization
- `transcriber/transcription/`: whisper.cpp bootstrap + transcription execution
- `transcriber/utils/`: constants, file handling, hardware/model selection logic
- `.github/workflows/`: CI/CD and release workflows

## Contribution Guidelines

### Coding style

- Preserve existing coding style and naming conventions.
- Prefer explicit types and clear function boundaries.
- Keep functions cohesive and focused on a single responsibility.
- Avoid introducing heavy dependencies without strong justification.
- Keep user-facing logs actionable and specific.

### Backward compatibility

- Do not break current CLI flags, file naming behavior, or default runtime directories without clear migration notes.
- If behavior changes are required, include docs updates in the same PR.

### Error handling

- Fail fast for invalid inputs.
- Return actionable error messages (include failing command/context when useful).
- Keep cancellation/interrupt behavior intact (`InterruptedError` paths).

## Testing Expectations

There is currently no formal automated test suite in this repository. Every functional PR should include manual validation notes in the PR description.

Minimum smoke checks:

1. UI launches successfully (`make ui`).
2. CLI launches successfully (`make run` or direct module command).
3. At least one short media file is processed end-to-end.
4. Transcript file is produced at expected output path.
5. No obvious regressions in setup/bootstrap flow (ffmpeg, whisper.cpp, model checks).

If your change touches packaging/release:

1. Validate workflow YAML syntax.
2. Confirm artifact naming conventions remain stable.
3. Confirm release trigger remains tag-based (`v*`).

## Pull Request Process

1. Create a branch from the latest default branch.
2. Implement and keep commits focused.
3. Update docs for any behavior/configuration changes.
4. Open a PR with:
   - Problem statement
   - Proposed solution
   - Validation steps and results
   - Risks and rollback considerations

PRs may be asked to split if they combine unrelated changes.

## Commit Message Guidance

Use clear, imperative commit messages. Examples:

- `fix: handle missing ffmpeg archive extension on macOS`
- `docs: clarify release artifact naming`
- `ui: improve queue status rendering`

## Reporting Bugs

When opening issues, include:

- OS + architecture
- App run mode (UI or CLI)
- Reproduction steps
- Expected vs actual behavior
- Relevant logs or stack traces
- Sample file type involved (if media-specific)

For support-style questions, use `SUPPORT.md` guidance.

## License

By contributing, you agree your contributions are released under the repository license in `/Users/premkumar/Code/talk-to-text/LICENSE` (The Unlicense).
