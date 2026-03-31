pub struct Statistics {
    pub effective_diffusion: f64,
    pub first_passage_time: f64,
    pub nonlinear_mobility: f64,
}

#[allow(unused_imports)]
pub use backend::statistics;
#[cfg(feature = "gpu")]
pub use backend::sweep_statistics;

#[cfg(feature = "gpu")]
mod backend {
    use super::{Statistics, diffusion, nonlinear_mobility};
    use crate::simulation::{ENSEMBLE_SIZE, TIME};
    use cudarc::driver::{CudaContext, CudaModule, LaunchConfig, PushKernelArg};
    use std::hash::{DefaultHasher, Hash, Hasher};
    use std::sync::Arc;
    use tokio::sync::mpsc::Receiver;
    use tokio::task::spawn_blocking;

    /// 単一の外力に対してGPUを用いてアンサンブル平均を計算する関数
    #[allow(dead_code)]
    pub async fn statistics(
        device: Arc<CudaContext>,
        module: Arc<CudaModule>,
        length: f64,
        force: f64,
    ) -> Statistics {
        let mut hasher = DefaultHasher::new();
        length.to_bits().hash(&mut hasher);
        force.to_bits().hash(&mut hasher);
        let seed = hasher.finish();

        let inv_length = 1.0 / length;

        let (disp_sum, sq_disp_sum) = spawn_blocking(move || {
            let func = module.load_function("displacements_sum").unwrap();
            let stream = device.default_stream();

            let mut dev_disp = stream.alloc_zeros::<f64>(1).unwrap();
            let mut dev_sq_disp = stream.alloc_zeros::<f64>(1).unwrap();

            let block_size = 256;
            let grid_size = ENSEMBLE_SIZE.div_ceil(block_size as u64) as u32;
            let cfg = LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                stream
                    .launch_builder(&func)
                    .arg(&seed)
                    .arg(&length)
                    .arg(&inv_length)
                    .arg(&force)
                    .arg(&mut dev_disp)
                    .arg(&mut dev_sq_disp)
                    .launch(cfg)
            }
            .unwrap();

            let host_disp = stream.clone_dtoh(&dev_disp).unwrap();
            let host_sq_disp = stream.clone_dtoh(&dev_sq_disp).unwrap();
            (host_disp[0], host_sq_disp[0])
        })
        .await
        .unwrap();

        let mean_displacement = disp_sum / ENSEMBLE_SIZE as f64;
        let mean_square_displacement = sq_disp_sum / ENSEMBLE_SIZE as f64;
        let mean_speed = mean_displacement / TIME;

        Statistics {
            effective_diffusion: diffusion(mean_displacement, mean_square_displacement, TIME),
            first_passage_time: 1.0 / mean_speed.abs(),
            nonlinear_mobility: nonlinear_mobility(mean_speed, force),
        }
    }

    /// GPUを用いて指定された範囲の外力（1からmax_forceまで順逆両方）のシミュレーション結果を一括で非同期計算し、
    /// 結果をストリーム（Receiver）として順番に返す関数。元の BulkBuffer に相当する高速化・スレッドブロック局所化を行います。
    pub async fn sweep_statistics(
        device: Arc<CudaContext>,
        module: Arc<CudaModule>,
        length: f64,
        max_force: usize,
    ) -> Receiver<(usize, Statistics, Statistics)> {
        // 結果を受け取るためのチャネルを作成
        let (tx, rx) = tokio::sync::mpsc::channel(max_force.max(1));

        // メモリの確保とGPUへのカーネル投入、同期待機処理はTokioのワーカースレッドをブロックしないよう `spawn_blocking` で行う
        spawn_blocking(move || {
            let func = module.load_function("displacements_sum").unwrap();
            let inv_length = 1.0 / length;

            // 前方と後方のシミュレーション管理のための構造
            struct TaskData {
                stream: Arc<cudarc::driver::CudaStream>,
                dev_disp: cudarc::driver::CudaSlice<f64>,
                dev_sq_disp: cudarc::driver::CudaSlice<f64>,
                force: f64,
                seed: u64,
            }

            let block_size = 256;
            let grid_size = ENSEMBLE_SIZE.div_ceil(block_size as u64) as u32;
            let cfg = LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            // 個別のCUDAストリームとデバイスメモリの一括事前確保
            let mut forward_tasks = Vec::with_capacity(max_force);
            let mut backward_tasks = Vec::with_capacity(max_force);

            for i in 1..=max_force {
                let f_force = i as f64;
                let b_force = -(i as f64);

                let mut hasher = DefaultHasher::new();
                length.to_bits().hash(&mut hasher);
                f_force.to_bits().hash(&mut hasher);
                let f_seed = hasher.finish();

                let mut hasher = DefaultHasher::new();
                length.to_bits().hash(&mut hasher);
                b_force.to_bits().hash(&mut hasher);
                let b_seed = hasher.finish();

                // 独立したストリームを作成し、非同期のストリームオーダー・メモリ確保を利用する (非常に軽量)
                let f_stream = device.new_stream().unwrap();
                let f_dev_disp = f_stream.alloc_zeros::<f64>(1).unwrap();
                let f_dev_sq_disp = f_stream.alloc_zeros::<f64>(1).unwrap();

                let b_stream = device.new_stream().unwrap();
                let b_dev_disp = b_stream.alloc_zeros::<f64>(1).unwrap();
                let b_dev_sq_disp = b_stream.alloc_zeros::<f64>(1).unwrap();

                forward_tasks.push(TaskData {
                    stream: f_stream,
                    dev_disp: f_dev_disp,
                    dev_sq_disp: f_dev_sq_disp,
                    force: f_force,
                    seed: f_seed,
                });

                backward_tasks.push(TaskData {
                    stream: b_stream,
                    dev_disp: b_dev_disp,
                    dev_sq_disp: b_dev_sq_disp,
                    force: b_force,
                    seed: b_seed,
                });
            }

            // GPUの各ストリームに一気にカーネルを非同期に投入する
            // ここは関数が即座にリターンするためブロックされない
            for (f_task, b_task) in forward_tasks.iter_mut().zip(backward_tasks.iter_mut()) {
                unsafe {
                    f_task
                        .stream
                        .launch_builder(&func)
                        .arg(&f_task.seed)
                        .arg(&length)
                        .arg(&inv_length)
                        .arg(&f_task.force)
                        .arg(&mut f_task.dev_disp)
                        .arg(&mut f_task.dev_sq_disp)
                        .launch(cfg)
                }
                .unwrap();

                unsafe {
                    b_task
                        .stream
                        .launch_builder(&func)
                        .arg(&b_task.seed)
                        .arg(&length)
                        .arg(&inv_length)
                        .arg(&b_task.force)
                        .arg(&mut b_task.dev_disp)
                        .arg(&mut b_task.dev_sq_disp)
                        .launch(cfg)
                }
                .unwrap();
            }

            // 全てのカーネルがスケジュールされた後、順次デバイスから結果をホスト（CPU）に転送し、チャネルを通じて通知する
            // (Tokioスレッドではなく、この spawn_blocking されたOSスレッドだけが同期待機を行う)
            for (idx, (f_task, b_task)) in
                forward_tasks.iter().zip(backward_tasks.iter()).enumerate()
            {
                // clone_dtoh を呼ぶことで当該ストリームの完了を待機する
                let f_disp = f_task.stream.clone_dtoh(&f_task.dev_disp).unwrap()[0];
                let f_sq_disp = f_task.stream.clone_dtoh(&f_task.dev_sq_disp).unwrap()[0];

                let b_disp = b_task.stream.clone_dtoh(&b_task.dev_disp).unwrap()[0];
                let b_sq_disp = b_task.stream.clone_dtoh(&b_task.dev_sq_disp).unwrap()[0];

                let f_mean_disp = f_disp / ENSEMBLE_SIZE as f64;
                let f_mean_sq_disp = f_sq_disp / ENSEMBLE_SIZE as f64;
                let f_mean_speed = f_mean_disp / TIME;

                let b_mean_disp = b_disp / ENSEMBLE_SIZE as f64;
                let b_mean_sq_disp = b_sq_disp / ENSEMBLE_SIZE as f64;
                let b_mean_speed = b_mean_disp / TIME;

                let forward_stat = Statistics {
                    effective_diffusion: diffusion(f_mean_disp, f_mean_sq_disp, TIME),
                    first_passage_time: 1.0 / f_mean_speed.abs(),
                    nonlinear_mobility: nonlinear_mobility(f_mean_speed, f_task.force),
                };

                let backward_stat = Statistics {
                    effective_diffusion: diffusion(b_mean_disp, b_mean_sq_disp, TIME),
                    first_passage_time: 1.0 / b_mean_speed.abs(),
                    nonlinear_mobility: nonlinear_mobility(b_mean_speed, b_task.force),
                };

                let i = idx + 1;
                // Channelが閉じられても問題ないようにエラーは無視 (途中でキャンセルされた場合など)
                let _ = tx.blocking_send((i, forward_stat, backward_stat));
            }
        });

        rx
    }
}

#[cfg(not(feature = "gpu"))]
mod backend {
    use super::{Statistics, diffusion, nonlinear_mobility};
    use crate::simulation::{ENSEMBLE_SIZE, Particle, STEPS, TIME};
    use nalgebra::Vector2;
    use rand::{SeedableRng, rngs::SmallRng};
    use rayon::prelude::*;

    pub fn statistics(length: f64, force: f64) -> Statistics {
        let force_vec = Vector2::new(force, 0.0);
        let (mean_displacement, mean_square_displacement) = (0..ENSEMBLE_SIZE)
            .into_par_iter()
            .map(|i| {
                let rng = SmallRng::seed_from_u64(i);
                let mut particle = Particle::new(rng, length, force_vec);
                let start = particle.now().position.x;
                let delta_x = particle.nth(STEPS).unwrap().position.x - start;

                (delta_x, delta_x * delta_x)
            })
            .reduce_with(|(a, aa), (x, xx)| (a + x, aa + xx))
            .map(|(sum, sq_sum)| (sum / ENSEMBLE_SIZE as f64, sq_sum / ENSEMBLE_SIZE as f64))
            .unwrap();

        let mean_speed = mean_displacement / TIME;

        Statistics {
            effective_diffusion: diffusion(mean_displacement, mean_square_displacement, TIME),
            first_passage_time: 1.0 / mean_speed,
            nonlinear_mobility: nonlinear_mobility(mean_speed, force),
        }
    }
}

/// 有効拡散係数 D_eff = (⟨x^2⟩ - ⟨x⟩^2) / (2t)
fn diffusion(mean_disp: f64, mean_sq_disp: f64, time: f64) -> f64 {
    (mean_sq_disp - mean_disp * mean_disp) / (2.0 * time)
}

/// 非線形移動度 μ = ⟨v⟩ / F
fn nonlinear_mobility(mean_speed: f64, force: f64) -> f64 {
    mean_speed / force
}

/// 整流尺度 α = |μ - μ_rev| / (μ + μ_rev)
#[allow(dead_code)]
pub fn alpha(forward_mobility: f64, backward_mobility: f64) -> f64 {
    (forward_mobility - backward_mobility).abs() / (forward_mobility + backward_mobility)
}
