use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// TOML形式の設定ファイルに対応する構造体。
/// `f`, `c_1`, `c_2`, `l` はリストで指定し、その全組み合わせがシミュレーションされる。
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// 時間刻み幅 Δt
    pub delta_t: f64,
    /// 総シミュレーション時間 T
    pub time: f64,
    /// 壁のばね定数 K
    pub k: f64,
    /// アンサンブルサイズ(1ケースあたりの粒子数)
    pub ensemble_size: u32,
    /// 全ケースの結果をまとめて出力するフォルダ
    pub output_dir: PathBuf,
    /// 一定外力 f(x方向)のリスト
    pub f: Vec<f64>,
    /// 電場パラメータ C1 = βEp/l̃ のリスト
    pub c_1: Vec<f64>,
    /// 電場パラメータ C2 = ΔαE/p のリスト
    pub c_2: Vec<f64>,
    /// 棒の長さ l のリスト
    pub l: Vec<f64>,
    /// GPU実行に関する設定(省略可)
    #[serde(default)]
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))] // CPUビルドでは参照されない
    pub gpu: GpuConfig,
}

/// GPU実行に関する任意設定
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(not(feature = "gpu"), allow(dead_code))] // CPUビルドでは参照されない
pub struct GpuConfig {
    /// 使用するGPUのID(省略時は搭載されている全GPU)
    pub ids: Option<Vec<usize>>,
    /// 1つのGPUで同時に実行するケース数(省略時はA100での実測に基づく既定値)
    pub tasks_per_gpu: Option<usize>,
}

impl Config {
    /// TOMLファイルを読み込み、値の妥当性を検証する
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("設定ファイル {} を読み込めません", path.display()))?;
        let config: Config = toml::from_str(&text)
            .with_context(|| format!("設定ファイル {} の形式が不正です", path.display()))?;

        ensure!(config.delta_t > 0.0, "delta_t は正の値にしてください");
        ensure!(config.time > 0.0, "time は正の値にしてください");
        ensure!(config.k >= 0.0, "k は非負の値にしてください");
        ensure!(
            config.ensemble_size > 0,
            "ensemble_size は1以上にしてください"
        );
        for (name, list) in [
            ("f", &config.f),
            ("c_1", &config.c_1),
            ("c_2", &config.c_2),
            ("l", &config.l),
        ] {
            ensure!(!list.is_empty(), "{name} には1つ以上の値を指定してください");
        }
        // 上限は初期配置の計算(ω(x) - l/2 > 0)が退化しない範囲
        ensure!(
            config.l.iter().all(|&l| l > 0.0 && l < 0.5),
            "l は 0 < l < 0.5 の範囲で指定してください"
        );

        Ok(config)
    }

    /// 総シミュレーションステップ数 T/Δt
    pub fn steps(&self) -> u64 {
        (self.time / self.delta_t).round() as u64
    }

    /// パラメータリストの全組み合わせ(直積)をケースとして展開する
    pub fn cases(&self) -> Vec<Case> {
        let mut cases = Vec::new();
        for &l in &self.l {
            for &c_1 in &self.c_1 {
                for &c_2 in &self.c_2 {
                    for &f in &self.f {
                        cases.push(Case { f, c_1, c_2, l });
                    }
                }
            }
        }
        cases
    }
}

/// 1回のシミュレーションに対応するパラメータの組
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Case {
    pub f: f64,
    pub c_1: f64,
    pub c_2: f64,
    pub l: f64,
}

impl Case {
    /// このケースの結果を保存するフォルダ名(例: "f10_c1-2_c2-0.5_l0.05")
    pub fn dir_name(&self) -> String {
        format!("f{}_c1-{}_c2-{}_l{}", self.f, self.c_1, self.c_2, self.l)
    }

    /// パラメータの組から再現性のある乱数シードを導出する(FNV-1aハッシュ)。
    /// 実行順序やケースの分割方法に依存せず、同じパラメータには常に同じシードが与えられる。
    pub fn seed(&self) -> u64 {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for bits in [
            self.f.to_bits(),
            self.c_1.to_bits(),
            self.c_2.to_bits(),
            self.l.to_bits(),
        ] {
            hash ^= bits;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash
    }
}
