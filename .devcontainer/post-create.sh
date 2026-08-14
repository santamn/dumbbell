#!/usr/bin/env bash
set -euo pipefail

# --- Rust 製ツール ---
cargo fetch
command -v rg       >/dev/null || cargo install ripgrep --locked
command -v fd       >/dev/null || cargo install fd-find --locked
command -v ast-grep >/dev/null || cargo install ast-grep --locked
command -v sem      >/dev/null || cargo install --git https://github.com/Ataraxy-Labs/sem --locked sem-cli

# ツール類の共通インストール先 (devcontainer.json でボリュームとして永続化されている)
TOOLS_DIR="$HOME/.local"

# --- ax ---
if ! command -v ax >/dev/null; then
  curl -fsSL https://ax.yusuke.run/install | env AX_INSTALL_DIR="$TOOLS_DIR/bin" sh
fi

# --- mold ---
if ! command -v mold >/dev/null; then
  # 最新リリースのダウンロードURLを取得
  MOLD_URL=$(curl -fsSL https://api.github.com/repos/rui314/mold/releases/latest |
    rg -N -o '"browser_download_url": "([^"]*x86_64-linux\.tar\.gz)"' -r '$1' || true)

  if [ -n "$MOLD_URL" ]; then
    curl -fsSL "$MOLD_URL" -o mold.tar.gz
    tar -C "$TOOLS_DIR" --strip-components=1 -xzf mold.tar.gz
    rm mold.tar.gz
  else
    echo "Failed to fetch mold release URL."
  fi
fi

sem setup
