# dumbbell

## Project Overview

This dumbbell is a Brownian motion simulation program implemented in Rust and CUDA.

## Physical Model of the Simulation

In the simulation, particles undergoing Brownian motion are treated as structures in which two points are connected by a rigid rod (i.e., dumbbell-shaped particles). The detailed physical model is described in [model.md](docs/model.md). Refer to it as needed.

## Computer Architecture

The architecture of the machine used for GPU-based computation is described in [architecture.md](docs/architecture.md). Understanding the machine architecture is important when performing optimizations and similar work.

## Code Quality Practices

- Keep complexity under control through appropriate abstraction, concretization, and use of libraries
  - Remove code and libraries that are no longer needed
- When using a library, consult its documentation and use it correctly
  - Refer to the documentation for how to specify library versions
  - Unless the documentation instructs otherwise, use the latest stable version
- Always attach comments in Japanese explaining the meaning of functions, structs, and any other semantically cohesive pieces of code
- Refactor code following established best practices such as those in *The Art of Readable Code* to improve readability
  - In particular, after making substantial additions or changes to the codebase, refactor the entire codebase in light of those changes

## About Animation

- The physical model of particles in the animation must always match the physical model of particles in the simulation
  - Whenever you resolve a semantic difference between the two physical models to keep them in sync, you must explicitly declare that you have done so

## Command line tools

You can use the following command-line tools to perform searches, edits, and code analysis with AI agents more efficiently and quickly.

- ripgrep (`rg`): Fast text/regex search across the repo. Prefer this over `grep -r`. Respects .gitignore by default. Examples: `rg 'pattern'`, `rg -n --glob '*.ts' 'foo'`, `rg -l 'TODO'` (files only), `rg -F 'literal string'` (no regex).
- fd (`fdfind`): Fast file/directory finder. Prefer this over `find`. Respects .gitignore, case-insensitive smart matching. Examples: `fd config`, `fd -e py` (by extension), `fd -t d src` (directories only), `fd -H` (include hidden).
- ax: Local HTTP and HTML I/O for coding agents. One command instead of curl + throwaway Python. Run `ax agent-context` to learn it — use it instead of throwaway scripts.
- ast-grep (`sg`): Structural (AST-based) code search and rewrite. Use when text regex is too fragile — matching syntax, not strings. Examples: `sg -p 'console.log($ARG)' -l ts` (search), `sg -p 'foo($A)' -r 'bar($A)' -U` (rewrite in place). `$VAR` matches one node, `$$$` matches many.
- sem: Semantic code search — finds code by meaning, not exact text. Use for "where is X handled?"-style questions when you don't know the identifier names. Example: `sem "retry logic for failed uploads"`. Fall back to `rg` once you know the concrete symbol.