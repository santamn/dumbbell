# dumbbell

## Project Overview

This dumbbell is a Brownian motion simulation program implemented in Rust and CUDA.

## Project Structure

```
src/
├── main.rs         # CLI (run / animate subcommands)
├── config.rs       # Loading TOML configuration and expanding it into parameter combinations
├── simulation.rs   # CPU implementation of the physical model (used for animation and the CPU version)
├── simulation.cu   # GPU implementation of the physical model (CUDA kernels)
├── runner.rs       # Running all cases and writing out results (GPU/CPU backends)
└── renderer.rs     # Animation display (egui/eframe)
docs/
├── model.md        # Derivation and formulation of the physical model
└── architecture.md # Hardware configuration of the execution machine
```

## Physical Model of the Simulation

In the simulation, particles undergoing Brownian motion are treated as structures in which two points are connected by a rigid rod (i.e., dumbbell-shaped particles). The detailed physical model is described in [model.md](docs/model.md). Refer to it as needed.

## About Animation

- The physical model of particles in the animation must always match the physical model of particles in the simulation
  - Whenever you resolve a semantic difference between the two physical models to keep them in sync, you must explicitly declare that you have done so

## Computer Architecture

The architecture of the machine used for GPU-based computation is described in [architecture.md](docs/architecture.md). Understanding the machine architecture is important when performing optimizations and similar work.

## General Coding Guide

- Do not preserve backward compatibility. Remove obsolete paths instead of adding compatibility layers, fallbacks, or migrations.
- Grow the system in layers. Start from the smallest version that works end to end, and add each new capability on top of a product that already works. Never trade a working product for unfinished complexity.
- Keep components modular and concerns clearly separated.
- Follow functional programming style.
  - Prefer to make data immutable.
  - Specify three components: Actions, Calculation, Data (This principle is written in the book "Grokking Simplicity"). Specifically, carefully isolate Actions.
    - Actions: Depend on how many times or when it is run. Also called functions with side-effects, side-effecting functions, impure functions. Examples: Send an email, read from a database, including I/O operations.
    - Calculations: Computations from input to output. Also called pure functions, mathematical functions. Examples: Find the maximum number, check if an email address is valid.
    - Data: Facts about events. Examples: The email address a user gave us, the dollar amount read from a bank's API.

## Command-line tools

### Installed tools

The following are installed in this environment. Prefer them over the standard Unix equivalents.

- `ast-grep` — Syntax-aware code search and rewriting. Use when regex is too fragile. See the ast-grep skill for rule syntax.
- `ax` — HTTP fetching and HTML extraction. Use instead of `curl` plus a throwaway parsing script. Run `ax agent-context` to learn it.
- `fd` — File and directory search. Use instead of `find`.
- `rg` (ripgrep) — Text and regex search. Use instead of `grep -r`.
- `sem` — Entity-level diff, blame, and impact analysis (functions, classes). Use instead of `git diff` and `git blame`. See the sem skill for details.

### Rules that override your defaults

- Before changing or removing a function signature, always check the blast radius with `sem impact`.
- When reporting how much changed, do not count `+`/`-` lines from `git diff`. Use the entity counts from `sem diff`.
- When an `ast-grep` pattern fails to match, do not guess at rewrites. Dump the parsed AST with `ast-grep run --lang <lang> --pattern '<pattern>' --debug-query=ast` (`--lang` is required), then fix the pattern.
- Before reading a large source file in full, get its structure with `ast-grep outline <path>`.
- When working with HTML or an API, reach for `ax` before writing a Python or Node script.
