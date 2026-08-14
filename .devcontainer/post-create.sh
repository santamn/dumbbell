#!/usr/bin/env bash
set -euo pipefail

# --- Rust 製ツール ---
cargo fetch

command -v rg       >/dev/null || cargo install ripgrep --locked
command -v fd       >/dev/null || cargo install fd-find --locked
command -v ast-grep >/dev/null || cargo install ast-grep --locked
command -v sem      >/dev/null || cargo install --git https://github.com/Ataraxy-Labs/sem --locked sem-cli

# --- ax ---
if ! command -v ax >/dev/null; then
  curl -fsSL https://ax.yusuke.run/install | sudo env AX_INSTALL_DIR=/usr/bin sh
fi

# --- mold ---
# 最新リリースのダウンロードURLを取得
MOLD_URL=$(curl -s https://api.github.com/repos/rui314/mold/releases/latest | grep -oP '"browser_download_url": "\K(.*x86_64-linux\.tar\.gz)(?=")')

if [ -n "$MOLD_URL" ]; then
    curl -fsSL "$MOLD_URL" -o mold.tar.gz
    # /usr/local 配下に展開 (bin, lib, man などが適切に配置されます)
    sudo tar -C /usr/local --strip-components=1 -xzf mold.tar.gz
    rm mold.tar.gz
else
    echo "Failed to fetch mold release URL."
fi

sem setup
