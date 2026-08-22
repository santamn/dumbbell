use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Deserializer};
use std::path::{Path, PathBuf};

/// 数値または "1/3" のような分数文字列を受け付けて f64 のリストにデシリアライズする
fn deserialize_frac_vec<'de, D>(deserializer: D) -> Result<Vec<f64>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum FracValue {
        Number(f64),
        Text(String),
    }

    Vec::<FracValue>::deserialize(deserializer)?
        .into_iter()
        .map(|v| match v {
            FracValue::Number(n) => Ok(n),
            FracValue::Text(s) => parse_fraction(&s).map_err(serde::de::Error::custom),
        })
        .collect()
}

/// "分子/分母" 形式、または通常の数値文字列を f64 に変換する
fn parse_fraction(s: &str) -> Result<f64, String> {
    match s.split_once('/') {
        Some((num, den)) => {
            let num: f64 = num
                .trim()
                .parse()
                .map_err(|_| format!("不正な分数表記です: \"{s}\""))?;
            let den: f64 = den
                .trim()
                .parse()
                .map_err(|_| format!("不正な分数表記です: \"{s}\""))?;
            ensure_nonzero_denominator(den, s)?;
            Ok(num / den)
        }
        None => s
            .trim()
            .parse()
            .map_err(|_| format!("不正な数値です: \"{s}\"")),
    }
}

fn ensure_nonzero_denominator(den: f64, original: &str) -> Result<(), String> {
    if den == 0.0 {
        Err(format!("分母が0の分数です: \"{original}\""))
    } else {
        Ok(())
    }
}

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
    /// アンサンブルサイズ(最終的に必要なサンプル数)
    pub ensemble_size: u32,
    /// 既に計算済みのサンプル数。`current_sample_size..ensemble_size` の範囲だけが計算される。
    /// サンプルを後から追加する際は、この値を前回の ensemble_size に書き換えて再実行する
    pub current_sample_size: u32,
    /// 全ケースの結果をまとめて出力するフォルダ
    pub output_dir: PathBuf,
    /// 一定外力 f(x方向)のリスト(負の値も可)
    pub f: Vec<f64>,
    /// 電場パラメータ C1 = βEp/l̃ のリスト("1/3" のような分数表記も可)
    #[serde(deserialize_with = "deserialize_frac_vec")]
    pub c_1: Vec<f64>,
    /// 電場パラメータ C2 = ΔαE/p のリスト("1/3" のような分数表記も可)
    #[serde(deserialize_with = "deserialize_frac_vec")]
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
        ensure!(
            config.current_sample_size < config.ensemble_size,
            "current_sample_size ({}) は ensemble_size ({}) より小さくしてください。追加するサンプルがありません",
            config.current_sample_size,
            config.ensemble_size
        );
        for (name, list) in [
            ("f", &config.f),
            ("c_1", &config.c_1),
            ("c_2", &config.c_2),
            ("l", &config.l),
        ] {
            ensure!(!list.is_empty(), "{name} には1つ以上の値を指定してください");
        }
        ensure!(
            config.l.iter().all(|&l| l > 0.0),
            "l は正の値を指定してください"
        );

        Ok(config)
    }

    /// 総シミュレーションステップ数 T/Δt
    pub fn steps(&self) -> usize {
        (self.time / self.delta_t).round() as usize
    }

    /// 今回計算するサンプル数(= ensemble_size - current_sample_size)。
    /// load() で current_sample_size < ensemble_size を検証済みなので必ず1以上になる
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))] // CPUビルドでは範囲を直接使う
    pub fn sample_count(&self) -> u32 {
        self.ensemble_size - self.current_sample_size
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

/// 表示用に、値を小数点以下6桁に丸めた上で末尾の余分な0や"."を取り除いて整形する
fn format_param(value: f64) -> String {
    let rounded = format!("{value:.6}");
    let trimmed = rounded.trim_end_matches('0').trim_end_matches('.');
    if trimmed.is_empty() || trimmed == "-" {
        "0".to_string()
    } else {
        trimmed.to_string()
    }
}

/// 1回のシミュレーションに対応するパラメータの組
#[derive(Debug, Clone, Copy)]
pub struct Case {
    pub f: f64,
    pub c_1: f64,
    pub c_2: f64,
    pub l: f64,
}

impl Case {
    /// 進捗表示用のケース名(例: "f-10_c1-2_c2-0.5_l-0.05")。
    /// "1/3" のような分数指定で無限小数になった値も、小数点以下6桁に丸めて短く表示する。
    pub fn label(&self) -> String {
        format!(
            "f-{}_c1-{}_c2-{}_l-{}",
            format_param(self.f),
            format_param(self.c_1),
            format_param(self.c_2),
            format_param(self.l)
        )
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

#[cfg(test)]
mod tests {
    use super::*;

    /// テスト用の設定TOML(current_sample_size を差し替えられるようにしてある)
    fn config_text(current_sample_size: u32) -> String {
        format!(
            r#"
delta_t = 2e-7
time = 10.0
k = 1.5e6
ensemble_size = 100
current_sample_size = {current_sample_size}
output_dir = "data/"
f = [10.0, -5.0]
c_1 = ["1/3"]
c_2 = [0.0]
l = [0.04]
"#
        )
    }

    #[test]
    fn parses_fraction_and_negative_values() {
        let config: Config = toml::from_str(&config_text(0)).unwrap();
        assert_eq!(config.f, vec![10.0, -5.0]);
        assert_eq!(config.c_1, vec![1.0 / 3.0]);
        assert_eq!(config.sample_count(), 100);
    }

    /// 追記分のサンプルが残っていない設定は読み込み時に弾かれる
    #[test]
    fn rejects_exhausted_sample_range() {
        let path = std::env::temp_dir().join("dumbbell_exhausted_config.toml");
        std::fs::write(&path, config_text(100)).unwrap();
        let err = Config::load(&path).unwrap_err();
        std::fs::remove_file(&path).ok();
        assert!(err.to_string().contains("current_sample_size"));
    }

    /// 未計算分だけが計算対象になる
    #[test]
    fn counts_only_remaining_samples() {
        let config: Config = toml::from_str(&config_text(30)).unwrap();
        assert_eq!(config.sample_count(), 70);
    }

    #[test]
    fn label_stays_short_for_repeating_decimals() {
        let case = Case {
            f: 10.0,
            c_1: 1.0 / 3.0,
            c_2: -1.0 / 3.0,
            l: 0.04,
        };
        assert_eq!(case.label(), "f-10_c1-0.333333_c2--0.333333_l-0.04");
    }
}
