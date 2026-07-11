//! エントリポイント。
//!
//! TOML設定ファイルを読み込み、サブコマンドに応じて
//! 一括シミュレーション(run)またはアニメーション表示(animate)を実行する。

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod config;
mod renderer;
mod runner;
mod simulation;

/// ダンベル型粒子のブラウン運動シミュレータ
#[derive(Parser)]
#[command(version, about)]
struct Cli {
    /// 設定ファイル(TOML)のパス
    #[arg(short, long, default_value = "config.toml", global = true)]
    config: PathBuf,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// 設定された全パラメータ組み合わせのシミュレーションを実行する(既定)
    Run,
    /// 1粒子のアニメーションをGUIで表示する
    Animate,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = config::Config::load(&cli.config)?;

    match cli.command.unwrap_or(Command::Run) {
        Command::Run => runner::run_all(&config, &cli.config),
        Command::Animate => renderer::run_animation(&config),
    }
}
