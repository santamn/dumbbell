//! パラメータ全組み合わせのシミュレーション実行と結果の書き出し。
//!
//! ケースをジョブキューに積み、GPUごとに複数のワーカースレッドが取り出して実行する。
//! 1ケースは数分かかる大きな計算なので、スレッドがGPUの完了を同期的に待つ
//! 単純な構成で十分であり、非同期ランタイムは使用しない。

use crate::config::{Case, Config};
use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::mpsc::Sender;
use std::time::{Duration, Instant};

/// 1サンプル(1粒子)分の生データ。
/// 集計は別スクリプトが行うため、プログラムはこの生データをCSVへ書き出す
#[derive(Debug, Clone, Copy)]
pub struct Sample {
    /// サンプルID。0始まりの通し番号で、乱数ストリームと1:1に対応する。
    /// 追記実行時は current_sample_size から続きの番号が振られる
    pub id: u32,
    /// 開始時のx座標
    pub start_x: f64,
    /// 終了時のx座標
    pub end_x: f64,
}

impl Sample {
    /// x方向の変位 Δx
    fn displacement(&self) -> f64 {
        self.end_x - self.start_x
    }
}

/// 1ケース分の計算結果
pub struct CaseResult {
    /// ケース一覧の中での位置
    pub index: usize,
    /// 今回計算したサンプル(id の昇順)
    pub samples: Vec<Sample>,
    /// このケースの計算に要した実時間(進捗表示用)
    pub elapsed: Duration,
}

/// 1ケースのシミュレーションから得られる統計量
#[derive(Debug, Clone, Copy)]
pub struct CaseStatistics {
    /// 平均変位 ⟨Δx⟩
    pub mean_displacement: f64,
    /// 非線形移動度 μ = ⟨v⟩ / f
    pub nonlinear_mobility: f64,
    /// 有効拡散係数 D_eff = (⟨Δx²⟩ - ⟨Δx⟩²) / (2T)
    pub effective_diffusion: f64,
    /// 平均初通過時間(周期1を進むのにかかる平均時間)= 1 / |⟨v⟩|
    pub mean_first_passage_time: f64,
}

impl CaseStatistics {
    /// 変位の1次・2次モーメントから統計量を計算する
    fn from_moments(mean_disp: f64, mean_sq_disp: f64, time: f64, force: f64) -> Self {
        let mean_speed = mean_disp / time;
        Self {
            mean_displacement: mean_disp,
            nonlinear_mobility: mean_speed / force,
            effective_diffusion: (mean_sq_disp - mean_disp * mean_disp) / (2.0 * time),
            mean_first_passage_time: 1.0 / mean_speed.abs(),
        }
    }

    /// サンプル列から統計量を計算する。
    /// 追記実行では「今回計算した分」だけが対象になるため、全サンプルを通した正しい集計は
    /// CSVを読む別スクリプトが行う
    fn from_samples(samples: &[Sample], time: f64, force: f64) -> Self {
        let (sum, sq_sum) = samples.iter().fold((0.0, 0.0), |(sum, sq_sum), sample| {
            let dx = sample.displacement();
            (sum + dx, sq_sum + dx * dx)
        });
        let n = samples.len() as f64;
        Self::from_moments(sum / n, sq_sum / n, time, force)
    }
}

/// 追記モードでファイルを開く。新規作成したときだけヘッダ行を書き込む
fn open_append(path: &Path, header: &str) -> Result<File> {
    let is_new = !path.exists();
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("{} を開けません", path.display()))?;
    if is_new {
        writeln!(file, "{header}")?;
    }
    Ok(file)
}

/// 2つのパスが同じ実体を指しているかを判定する。
/// 存在しないパスは canonicalize に失敗するため、その場合は「別物」として扱う
fn is_same_file(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}

/// 設定された全ケースを実行し、ケースごとのサンプルCSV・まとめファイル(summary.dat)・
/// 設定のコピーを書き出す。既存の出力フォルダにはサンプルが追記される
pub fn run_all(config: &Config, config_path: &Path) -> Result<()> {
    let total_start = Instant::now();
    let cases = config.cases();

    // ケース名はサンプルCSVのファイル名になるため、衝突すると別ケースのサンプルが
    // 同じファイルに混ざってしまう。label() は小数点以下6桁に丸めるので、
    // パラメータの重複指定や、丸めた結果一致してしまう組をここで弾く
    let mut labels = HashSet::new();
    for case in &cases {
        anyhow::ensure!(
            labels.insert(case.label()),
            "ケース名 {} が重複しています。f, c_1, c_2, l に同じ値、または小数点以下6桁で一致する値が含まれていないか確認してください",
            case.label()
        );
    }

    // サンプルを後から追加できるよう、出力フォルダは既に存在していてもよい
    std::fs::create_dir_all(&config.output_dir).with_context(|| {
        format!(
            "出力フォルダ {} を作成できません",
            config.output_dir.display()
        )
    })?;

    // 再現性のため、使用した設定ファイルをそのまま出力フォルダへコピーする。
    // 出力先の config.toml をそのまま指定して再実行された場合、コピーすると
    // 自分自身を空にしてしまうためスキップする
    let config_copy = config.output_dir.join("config.toml");
    if !is_same_file(config_path, &config_copy) {
        std::fs::copy(config_path, &config_copy)?;
    }

    // 全ケースの結果をまとめるファイル(完了したケースから順に追記される)
    let mut summary = open_append(
        &config.output_dir.join("summary.dat"),
        "# f c_1 c_2 l mean_disp mu d_eff mfpt",
    )?;

    let progress = ProgressBar::new(cases.len() as u64).with_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) {msg}")
            .unwrap()
            .progress_chars("=>-"),
    );
    // バーは放っておくと更新イベント(inc等)まで一度も描画されない。ケースは数分〜数十分
    // かかるため、定期tickで実行開始直後から表示し、経過時間も更新され続けるようにする
    progress.enable_steady_tick(Duration::from_millis(500));

    // ワーカーが計算したケース単位の結果を受け取るチャネル
    let (tx, rx) = std::sync::mpsc::channel();
    let workers = backend::spawn_workers(config, &cases, tx)?;

    // 全ワーカーが送信側を手放すまで、完了したケースから順に結果を書き出す
    for CaseResult {
        index,
        samples,
        elapsed,
    } in rx
    {
        let case = cases[index];

        // サンプル単位の生データをケースごとのCSVへ追記する。
        // 途中で中断してもここまでの結果が残るよう、ケースごとにflushする
        let mut csv = open_append(
            &config.output_dir.join(format!("{}.csv", case.label())),
            "id,start_x,end_x,dx,time",
        )?;
        for sample in &samples {
            writeln!(
                csv,
                "{},{},{},{},{}",
                sample.id,
                sample.start_x,
                sample.end_x,
                sample.displacement(),
                config.time,
            )?;
        }
        csv.flush()?;

        let stats = CaseStatistics::from_samples(&samples, config.time, case.f);
        writeln!(
            summary,
            "{} {} {} {} {} {} {} {}",
            case.f,
            case.c_1,
            case.c_2,
            case.l,
            stats.mean_displacement,
            stats.nonlinear_mobility,
            stats.effective_diffusion,
            stats.mean_first_passage_time,
        )?;
        summary.flush()?;

        progress.println(format!(
            "完了: {} ({:.1}秒)",
            case.label(),
            elapsed.as_secs_f64()
        ));
        progress.set_message(format!("完了: {}", case.label()));
        progress.inc(1);
    }

    // ワーカーのエラー(CUDA失敗など)をここで回収する
    for worker in workers {
        worker.join().expect("ワーカースレッドが異常終了")?;
    }

    let total_elapsed = total_start.elapsed();
    progress.finish_with_message(format!(
        "全{}ケース完了 ({:.1}秒) → {}",
        cases.len(),
        total_elapsed.as_secs_f64(),
        config.output_dir.display()
    ));
    Ok(())
}

/// GPUバックエンド: ケースをジョブキューから取り出し、CUDAカーネルで計算するワーカー群
#[cfg(feature = "gpu")]
mod backend {
    use super::*;
    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread::JoinHandle;

    /// CUDAカーネルのブロックあたりスレッド数。
    /// ブロックを小さくして数を増やした方が全SMに均等に行き渡るため、
    /// A100での実測では256より128の方が速い
    const BLOCK_SIZE: u32 = 128;
    /// GPUあたりの同時実行ケース数の既定値。
    /// 1ケース(数百ブロック)ではA100を使い切れないため、複数ケースを別ストリームで
    /// 同時に流して占有率を上げる。実測では4ケース同時で飽和スループットの約95%に達し、
    /// それ以上増やしても1ケースあたりの所要時間が延びるだけで得られる向上はわずか
    const DEFAULT_TASKS_PER_GPU: usize = 4;

    /// build.rsがnvccでコンパイルしたCUDAカーネル(CUBIN形式)
    const KERNEL_IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/simulation.cubin"));

    /// GPUごとに tasks_per_gpu 個のワーカースレッドを起動する。
    /// 各ワーカーは専用のCUDAストリームを持ち、共有キューからケースを取り出しては
    /// カーネルを実行し、結果を results 経由でメインスレッドへ送る
    pub fn spawn_workers(
        config: &Config,
        cases: &[Case],
        results: Sender<CaseResult>,
    ) -> Result<Vec<JoinHandle<Result<()>>>> {
        let gpu_ids = match &config.gpu.ids {
            Some(ids) => ids.clone(),
            None => (0..CudaContext::device_count()? as usize).collect(),
        };
        anyhow::ensure!(!gpu_ids.is_empty(), "使用可能なGPUがありません");
        let tasks_per_gpu = config.gpu.tasks_per_gpu.unwrap_or(DEFAULT_TASKS_PER_GPU);

        // 全ワーカーで共有する「次に実行するケース番号」
        let next_case = Arc::new(AtomicUsize::new(0));

        let mut workers = Vec::new();
        for gpu_id in gpu_ids {
            let ctx =
                CudaContext::new(gpu_id).with_context(|| format!("GPU {gpu_id} の初期化に失敗"))?;
            let module = ctx.load_module(cudarc::nvrtc::Ptx::from_binary(KERNEL_IMAGE.to_vec()))?;

            for _ in 0..tasks_per_gpu {
                let ctx = ctx.clone();
                let module = module.clone();
                let cases = cases.to_vec();
                let config = config.clone();
                let next_case = next_case.clone();
                let results = results.clone();

                workers.push(std::thread::spawn(move || -> Result<()> {
                    let stream = ctx.new_stream()?;
                    let func = module.load_function("simulate")?;
                    // 出力用バッファ: サンプルごとの [開始x, 終了x]。
                    // 大きさはケースによらず一定なので一度だけ確保する
                    let count = config.sample_count();
                    let mut samples_dev = stream.alloc_zeros::<f64>(2 * count as usize)?;

                    loop {
                        // キューからケースを1つ取り出す(なくなったら終了)
                        let index = next_case.fetch_add(1, Ordering::Relaxed);
                        let Some(case) = cases.get(index).copied() else {
                            return Ok(());
                        };

                        // カーネルが全要素を書き込むため、ここでのゼロ初期化は不要
                        let case_start = Instant::now();
                        unsafe {
                            stream
                                .launch_builder(&func)
                                .arg(&case.seed())
                                .arg(&config.steps())
                                .arg(&config.current_sample_size)
                                .arg(&count)
                                .arg(&config.delta_t)
                                .arg(&config.k)
                                .arg(&case.l)
                                .arg(&case.f)
                                .arg(&case.c_1)
                                .arg(&case.c_2)
                                .arg(&mut samples_dev)
                                .launch(LaunchConfig {
                                    grid_dim: (count.div_ceil(BLOCK_SIZE), 1, 1),
                                    block_dim: (BLOCK_SIZE, 1, 1),
                                    shared_mem_bytes: 0,
                                })?;
                        }
                        stream.synchronize()?;

                        // [開始x, 終了x] の並びを Sample へ組み替える(idは current_sample_size 始まり)
                        let samples = stream
                            .clone_dtoh(&samples_dev)?
                            .chunks_exact(2)
                            .enumerate()
                            .map(|(i, pair)| Sample {
                                id: config.current_sample_size + i as u32,
                                start_x: pair[0],
                                end_x: pair[1],
                            })
                            .collect();
                        results
                            .send(CaseResult {
                                index,
                                samples,
                                elapsed: case_start.elapsed(),
                            })
                            .ok();
                    }
                }));
            }
        }
        Ok(workers)
    }
}

/// CPUバックエンド: GPUなしでも動作確認できるよう、rayonで粒子を並列計算する。
/// GPU版に比べ桁違いに遅いため、小規模な検証用
#[cfg(not(feature = "gpu"))]
mod backend {
    use super::*;
    use crate::simulation::{ModelParams, Particle};
    use rand::{SeedableRng, rngs::SmallRng};
    use rayon::prelude::*;
    use std::thread::JoinHandle;

    /// 1本のワーカースレッドがケースを順に処理する(ケース内部はrayonで並列)
    pub fn spawn_workers(
        config: &Config,
        cases: &[Case],
        results: Sender<CaseResult>,
    ) -> Result<Vec<JoinHandle<Result<()>>>> {
        let config = config.clone();
        let cases = cases.to_vec();

        Ok(vec![std::thread::spawn(move || -> Result<()> {
            let steps = config.steps();
            for (index, case) in cases.iter().enumerate() {
                let case_start = Instant::now();
                let params = ModelParams {
                    delta_t: config.delta_t,
                    spring_k: config.k,
                    length: case.l,
                    force_x: case.f,
                    c_1: case.c_1,
                    c_2: case.c_2,
                };

                // 未計算のサンプルだけを独立にシミュレートする。
                // サンプルIDがそのまま乱数シードに入るため、追記分は既存分と別の乱数列になる
                let samples = (config.current_sample_size..config.ensemble_size)
                    .into_par_iter()
                    .map(|id| {
                        let rng = SmallRng::seed_from_u64(case.seed().wrapping_add(id as u64));
                        let mut particle = Particle::new(rng, params);
                        let start_x = particle.state().position.x;
                        particle.advance(steps);
                        Sample {
                            id,
                            start_x,
                            end_x: particle.state().position.x,
                        }
                    })
                    .collect();

                results
                    .send(CaseResult {
                        index,
                        samples,
                        elapsed: case_start.elapsed(),
                    })
                    .ok();
            }
            Ok(())
        })])
    }
}
