# dummbell

## プロジェクト概要

この dumbell は Rust と CUDA を用いて実装されたブラウン運動のシミュレーションプログラムである。

## シミュレーションの物理モデル

シミュレーションではブラウン運動を行う粒子は、2つの点が剛体の棒で結合された構造（=ダンベル型粒子）として扱われる。詳細な物理モデルについては [model.md](docs/model.md) に書かれている。必要があれば参照せよ。

## 計算機アーキテクチャ

GPU を用いた計算を行う際に使用する計算機のアーキテクチャについては [architecture.md](docs/architecture.md) に書かれている。最適化などを行う際は計算機のアーキテクチャを理解しておくことは重要である。

## コード品質向上施作

- 適切な抽象化・具象化・ライブラリの利用によって複雑さを抑えること
  - 不要になったコードやライブラリは削除すること
- ライブラリを使用する際はそのライブラリのドキュメントを参照し、適切な使い方をすること
  - ライブラリのバージョン指定の方法はドキュメントを参照すること
  - 特にドキュメントに指示がない場合は最新の安定版を使用すること
- 関数・構造体・その他意味的にまとまりのあるコード片に対して、必ずその意味を説明するコメントを日本語で付与すること
- コードの可読性を向上させるために、『リーダブルコード』などの既存のベストプラクティスを意識してコードのリファクタリングを行うこと
  - 特に、一定以上コードベースに大きな追加・変更を加えた際は、その変更を踏まえてコードベース全体のリファクタリングを行うこと

## animation について

- animation における粒子の物理モデルと simulation における粒子の物理モデルは常に一致するようすること
  - この同期を行うために両者の物理モデルにおける意味的な差分を解消した際は、必ずそのことを宣言すること

## Command line tools

You can use the following command-line tools to perform searches, edits, and code analysis with AI agents more efficiently and quickly.

- ripgrep (`rg`): Fast text/regex search across the repo. Prefer this over `grep -r`. Respects .gitignore by default. 
  - Examples:
    - `rg 'pattern'`
    - `rg -n --glob '*.ts' 'foo'`
    - `rg -l 'TODO'` (files only)
    - `rg -F 'literal string'` (no regex)
- fd (`fdfind`): Fast file/directory finder. Prefer this over `find`. Respects .gitignore, case-insensitive smart matching. 
  - Examples: 
    - `fdfind config`
    - `fdfind -e py` (by extension)
    - `fdfind -t d src` (directories only)
    - `fdfind -H` (include hidden)
- ax: Local HTTP and HTML I/O for coding agents. One command instead of curl + throwaway Python. Run `ax agent-context` to learn it — use it instead of throwaway scripts.
- ast-grep (`sg`): Structural (AST-based) code search and rewrite. Use when text regex is too fragile — matching syntax, not strings. 
  - Examples: 
    - `sg -p 'console.log($ARG)' -l ts`: search
    - `sg -p 'foo($A)' -r 'bar($A)' -U`: rewrite in place. `$VAR` matches one node, `$$$` matches many.
- sem: Semantic version control on top of git. Parses code with tree-sitter and diffs, blames, and analyzes impact at the entity level (functions, classes, methods, structs) instead of lines. Prefer this over `git diff` / `git blame` when you care about *which entities* changed rather than which lines. Works in any git repo with no setup.
  - Examples:
    - `sem diff`
    - `sem diff --staged`
    - `sem diff --format json`
    - What breaks if an entity changes (dependencies & dependents): `sem impact simulate_step`
    - `sem blame src/simulation.rs`
    - `sem log simulate_step`
    - `sem entities src/`
    - Token-budgeted context (entity + deps + dependents) for LLMs: `sem context simulate_step --budget 4000`
