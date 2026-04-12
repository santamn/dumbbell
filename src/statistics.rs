pub struct Statistics {
    pub effective_diffusion: f64,
    pub first_passage_time: f64,
    pub nonlinear_mobility: f64,
}

#[allow(unused_imports)]
pub use backend::statistics;

#[cfg(feature = "gpu")]
mod backend {
    use super::{Statistics, diffusion, nonlinear_mobility};
    use crate::simulation::{BLOCK_SIZE, ENSEMBLE_SIZE, TIME};
    use cudarc::driver::{CudaContext, CudaModule, LaunchConfig, PushKernelArg};
    use std::hash::{DefaultHasher, Hash, Hasher};
    use std::ops::RangeInclusive;
    use std::sync::Arc;
    use std::sync::OnceLock;
    use tokio::sync::Mutex;
    use tokio::sync::mpsc::Receiver;

    static CU_ALLOC_MUX: OnceLock<Mutex<()>> = OnceLock::new();

    /// GPUを用いて指定された範囲の外力のシミュレーション結果を一括で非同期計算し、結果をストリーム（Receiver）として順番に返す関数
    pub async fn statistics(
        ctx: Arc<CudaContext>,
        module: Arc<CudaModule>,
        length: f64,
        forces: RangeInclusive<usize>,
    ) -> Receiver<(f64, Statistics, Statistics)> {
        // 結果を受け取るためのチャネルを作成
        let (tx, rx) = tokio::sync::mpsc::channel(forces.end() - forces.start() + 1);
        // Tokioの軽量タスク（JoinSetを用いた並行Spawn）としてスケジューリング
        let mut join_set = tokio::task::JoinSet::new();

        // 各外力に対してシミュレーションをSpawn
        forces
            .into_iter()
            .map(|i| {
                (
                    tx.clone(),
                    i as f64,
                    (ctx.clone(), module.clone()),
                    (ctx.clone(), module.clone()),
                )
            })
            .for_each(|(tx, force, (f_ctx, f_mod), (b_ctx, b_mod))| {
                // 外力ごとに個別の短命なTokioタスクを生やし、順方向と逆方向のシミュレーションを同時に待機
                join_set.spawn(async move {
                    let (forward_stat, backward_stat) = tokio::join!(
                        simulate_single_case(f_ctx, f_mod, length, force),
                        simulate_single_case(b_ctx, b_mod, length, -force)
                    );
                    // 計算が終了したものからチャネルに結果を流し込むため、呼び出し元でも即座にファイル書き込み等の逐次処理が可能
                    let _ = tx.send((force, forward_stat, backward_stat)).await;
                });
            });

        // 全てのタスクが完了するまでJoinSetを待機（リソースリークを防ぐため、裏で全ての完了を保証）
        tokio::spawn(async move { while join_set.join_next().await.is_some() {} });

        rx
    }

    /// 単一のシミュレーションに対するCUDAシミュレーションと非同期メモリ転送を司る関数
    async fn simulate_single_case(
        ctx: Arc<CudaContext>,
        module: Arc<CudaModule>,
        length: f64,
        force: f64,
    ) -> Statistics {
        // パラメーターをハッシュ化してシード値を生成
        let mut hasher = DefaultHasher::new();
        length.to_bits().hash(&mut hasher);
        force.to_bits().hash(&mut hasher);
        let seed = hasher.finish();

        // 同時に数百のタスクがCUDAの初期化処理に入って不安定になることを防ぐため、Mutexによってストリームやメモリ空間の作成を同期的に行う
        let _guard = CU_ALLOC_MUX.get_or_init(|| Mutex::new(())).lock().await;
        // 専用のストリーム（非同期実行キュー）を作成
        let stream = ctx.new_stream().unwrap();
        // GPU上のデバイスメモリを確保
        let mut dev_disp = stream.alloc_zeros::<f64>(1).unwrap();
        let mut dev_sq_disp = stream.alloc_zeros::<f64>(1).unwrap();
        // メモリ確保が完了したらロックを解放して他のタスクがCUDA処理に入れるようにする
        drop(_guard);

        // カーネルの非同期実行をストリームへ投入
        let func = module.load_function("displacements_sum").unwrap();
        unsafe {
            stream
                .launch_builder(&func)
                .arg(&seed)
                .arg(&length)
                .arg(&(1.0 / length)) // 逆数を事前に計算して渡すことで、カーネル内での除算を乗算に変換し高速化
                .arg(&force)
                .arg(&mut dev_disp)
                .arg(&mut dev_sq_disp)
                .launch(LaunchConfig {
                    block_dim: (BLOCK_SIZE, 1, 1),
                    grid_dim: (ENSEMBLE_SIZE.div_ceil(BLOCK_SIZE), 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .unwrap();

        // わずか16バイトの転送であるため、spawn_blockingを用いてOSスレッド上で同期コピー（完了待機）を行う
        // これにより、イベントのポーリングによるReadLockの枯渇を防ぐ
        let (h_disp, h_sq_disp) = tokio::task::spawn_blocking(move || {
            let h_disp = stream.clone_dtoh(&dev_disp).unwrap();
            let h_sq_disp = stream.clone_dtoh(&dev_sq_disp).unwrap();
            (h_disp[0], h_sq_disp[0])
        })
        .await
        .unwrap();

        let mean_displacement = h_disp / ENSEMBLE_SIZE as f64;
        let mean_square_displacement = h_sq_disp / ENSEMBLE_SIZE as f64;
        let mean_speed = mean_displacement / TIME;

        Statistics {
            effective_diffusion: diffusion(mean_displacement, mean_square_displacement, TIME),
            first_passage_time: 1.0 / mean_speed.abs(),
            nonlinear_mobility: nonlinear_mobility(mean_speed, force),
        }
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
