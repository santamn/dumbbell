//! パラメータ全組み合わせのシミュレーション実行と結果の書き出し。
//!
//! ケースをジョブキューに積み、GPUごとに複数のワーカースレッドが取り出して実行する。
//! 1ケースは数分かかる大きな計算なので、スレッドがGPUの完了を同期的に待つ
//! 単純な構成で十分であり、非同期ランタイムは使用しない。

use crate::config::{Case, Config};
use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use serde::Serialize;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::mpsc::Sender;

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
}

/// 設定された全ケースを実行し、ケースごとのフォルダと全体のまとめファイルに結果を書き出す
pub fn run_all(config: &Config, config_path: &Path) -> Result<()> {
    let cases = config.cases();
    std::fs::create_dir_all(&config.output_dir).with_context(|| {
        format!(
            "出力フォルダ {} を作成できません",
            config.output_dir.display()
        )
    })?;

    // 再現性のため、使用した設定ファイルをそのまま出力フォルダへコピーする
    std::fs::copy(config_path, config.output_dir.join("config.toml"))?;

    // 全ケースの結果をまとめるファイル(完了したケースから順に追記される)
    let mut summary = File::create(config.output_dir.join("summary.dat"))?;
    writeln!(summary, "# f c_1 c_2 l mean_disp mu d_eff mfpt")?;

    let progress = ProgressBar::new(cases.len() as u64).with_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) {msg}")
            .unwrap()
            .progress_chars("=>-"),
    );

    // ワーカーが計算した (ケース番号, モーメント) を受け取るチャネル
    let (tx, rx) = std::sync::mpsc::channel();
    let workers = backend::spawn_workers(config, &cases, tx)?;

    // 全ワーカーが送信側を手放すまで、完了したケースから順に結果を書き出す
    for (index, mean_disp, mean_sq_disp) in rx {
        let case = cases[index];
        let stats = CaseStatistics::from_moments(mean_disp, mean_sq_disp, config.time, case.f);
        write_case_result(config, &case, &stats)?;
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

        progress.set_message(format!("完了: {}", case.dir_name()));
        progress.inc(1);
    }

    // ワーカーのエラー(CUDA失敗など)をここで回収する
    for worker in workers {
        worker.join().expect("ワーカースレッドが異常終了")?;
    }

    progress.finish_with_message(format!(
        "全{}ケース完了 → {}",
        cases.len(),
        config.output_dir.display()
    ));
    Ok(())
}

/// ケースごとのフォルダに、再現用の設定と統計量を書き出す
fn write_case_result(config: &Config, case: &Case, stats: &CaseStatistics) -> Result<()> {
    /// ケースフォルダに書き出す再現用設定(このファイルだけで1ケースを再現できる)
    #[derive(Serialize)]
    struct CaseRecord {
        delta_t: f64,
        time: f64,
        k: f64,
        ensemble_size: u32,
        f: f64,
        c_1: f64,
        c_2: f64,
        l: f64,
        seed: u64,
    }

    let dir = config.output_dir.join(case.dir_name());
    std::fs::create_dir_all(&dir)?;

    let record = CaseRecord {
        delta_t: config.delta_t,
        time: config.time,
        k: config.k,
        ensemble_size: config.ensemble_size,
        f: case.f,
        c_1: case.c_1,
        c_2: case.c_2,
        l: case.l,
        seed: case.seed(),
    };
    std::fs::write(dir.join("config.toml"), toml::to_string(&record)?)?;

    let mut result = File::create(dir.join("result.dat"))?;
    writeln!(result, "# mean_disp mu d_eff mfpt")?;
    writeln!(
        result,
        "{} {} {} {}",
        stats.mean_displacement,
        stats.nonlinear_mobility,
        stats.effective_diffusion,
        stats.mean_first_passage_time,
    )?;
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
        results: Sender<(usize, f64, f64)>,
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
                    // 出力用バッファ: [Σ Δx, Σ (Δx)²]
                    let mut out = stream.alloc_zeros::<f64>(2)?;

                    loop {
                        // キューからケースを1つ取り出す(なくなったら終了)
                        let index = next_case.fetch_add(1, Ordering::Relaxed);
                        let Some(case) = cases.get(index).copied() else {
                            return Ok(());
                        };

                        stream.memset_zeros(&mut out)?;
                        unsafe {
                            stream
                                .launch_builder(&func)
                                .arg(&case.seed())
                                .arg(&config.steps())
                                .arg(&config.ensemble_size)
                                .arg(&config.delta_t)
                                .arg(&config.k)
                                .arg(&case.l)
                                .arg(&case.f)
                                .arg(&case.c_1)
                                .arg(&case.c_2)
                                .arg(&mut out)
                                .launch(LaunchConfig {
                                    grid_dim: (config.ensemble_size.div_ceil(BLOCK_SIZE), 1, 1),
                                    block_dim: (BLOCK_SIZE, 1, 1),
                                    shared_mem_bytes: 0,
                                })?;
                        }
                        stream.synchronize()?;

                        let sums = stream.clone_dtoh(&out)?;
                        let n = config.ensemble_size as f64;
                        results.send((index, sums[0] / n, sums[1] / n)).ok();
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
        results: Sender<(usize, f64, f64)>,
    ) -> Result<Vec<JoinHandle<Result<()>>>> {
        let config = config.clone();
        let cases = cases.to_vec();

        Ok(vec![std::thread::spawn(move || -> Result<()> {
            let steps = config.steps() as usize;
            for (index, case) in cases.iter().enumerate() {
                let params = ModelParams {
                    delta_t: config.delta_t,
                    spring_k: config.k,
                    length: case.l,
                    force_x: case.f,
                    c_1: case.c_1,
                    c_2: case.c_2,
                };

                // 各粒子を独立にシミュレートし、変位の総和と二乗和を求める
                let (sum, sq_sum) = (0..config.ensemble_size)
                    .into_par_iter()
                    .map(|i| {
                        let rng = SmallRng::seed_from_u64(case.seed().wrapping_add(i as u64));
                        let mut particle = Particle::new(rng, params);
                        let start_x = particle.state().position.x;
                        particle.advance(steps);
                        let dx = particle.state().position.x - start_x;
                        (dx, dx * dx)
                    })
                    .reduce(|| (0.0, 0.0), |(a, aa), (b, bb)| (a + b, aa + bb));

                let n = config.ensemble_size as f64;
                results.send((index, sum / n, sq_sum / n)).ok();
            }
            Ok(())
        })])
    }
}
