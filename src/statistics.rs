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
    use cudarc::driver::{
        CudaContext, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
    };
    use std::hash::{DefaultHasher, Hash, Hasher};
    use std::ops::RangeInclusive;
    use std::sync::{Arc, OnceLock};
    use tokio::sync::{Mutex, mpsc::Receiver, mpsc::Sender};

    static CU_ALLOC_MUX: OnceLock<Mutex<()>> = OnceLock::new();

    /// GPUデバイスからのストリームと各タスク用メモリ領域を使い回すための構造体
    struct CudaWorker {
        stream: Arc<CudaStream>,
        dev_disp: CudaSlice<f64>,
        dev_sq_disp: CudaSlice<f64>,
    }

    impl CudaWorker {
        fn memset_zeros(&mut self) {
            self.stream.memset_zeros(&mut self.dev_disp).unwrap();
            self.stream.memset_zeros(&mut self.dev_sq_disp).unwrap();
        }
    }

    /// GPUリソース（CudaWorker）をプールする構造体
    #[derive(Clone)]
    struct CudaWorkerPool {
        sender: Sender<CudaWorker>,
        receiver: Arc<Mutex<Receiver<CudaWorker>>>,
    }

    impl CudaWorkerPool {
        async fn new(ctx: Arc<CudaContext>, count: usize) -> Self {
            let (sender, receiver) = tokio::sync::mpsc::channel(count);
            let receiver = Arc::new(Mutex::new(receiver));

            // CU_ALLOC_MUXを用いて排他的に複数ワーカーを一括生成
            // これによりハングアップを防ぎつつ、初期化オーバーヘッドを1回で済ませる
            {
                let _guard = CU_ALLOC_MUX.get_or_init(|| Mutex::new(())).lock().await;
                for _ in 0..count {
                    let stream = ctx.new_stream().unwrap();
                    let dev_disp = stream.alloc_zeros::<f64>(1).unwrap();
                    let dev_sq_disp = stream.alloc_zeros::<f64>(1).unwrap();
                    sender
                        .send(CudaWorker {
                            stream,
                            dev_disp,
                            dev_sq_disp,
                        })
                        .await
                        .unwrap();
                }
            }

            Self { sender, receiver }
        }

        /// プールからワーカーを取得する非同期関数
        async fn acquire(&self) -> CudaWorker {
            let mut rx = self.receiver.lock().await;
            rx.recv().await.expect("ワーカーの取得に失敗")
        }

        /// ワーカーをプールに返却する非同期関数
        async fn release(&self, worker: CudaWorker) {
            self.sender
                .send(worker)
                .await
                .expect("ワーカーの返却に失敗");
        }

        // 全タスク完了後のGPUリソース一括解放
        async fn destroy(self) {
            let mut rx = self.receiver.lock().await;
            rx.close(); // 受信側を明示的に閉じ、ワーカーを取り出せないようにする

            let mut workers = Vec::new();
            while let Some(worker) = rx.recv().await {
                workers.push(worker); // キュー内の不要になった全てのワーカーを安全なメモリへ移す
            }

            // 安全な一括破棄（排他制御）
            let _guard = CU_ALLOC_MUX.get().unwrap().lock().await;
            tokio::task::spawn_blocking(move || {
                drop(workers); // 完全にOSスレッド上で直列に行われるため、ドライバロック競合が起きずハングアップしなくなる
            })
            .await
            .unwrap();
        }
    }

    /// GPUを用いて指定された範囲の外力のシミュレーション結果を一括で非同期計算し、結果をストリームとして順番に返す関数
    pub async fn statistics(
        ctx: Arc<CudaContext>,
        module: Arc<CudaModule>,
        length: f64,
        forces: RangeInclusive<usize>,
    ) -> Receiver<(f64, Statistics, Statistics)> {
        // 結果を受け取るためのチャネルを作成
        let (tx, rx) = tokio::sync::mpsc::channel(forces.end() - forces.start() + 1);

        // GPUリソースプール（CudaWorker）の作成
        // キュー容量が実質的なSemaphore(並行制限)として機能する
        let pool = CudaWorkerPool::new(ctx.clone(), 128).await;

        // Tokioの軽量タスク（JoinSetを用いた並行Spawn）としてスケジューリング
        let mut join_set = tokio::task::JoinSet::new();

        // 各外力に対してシミュレーションをSpawn
        forces
            .into_iter()
            .map(|i| {
                (
                    tx.clone(),
                    i as f64,
                    module.clone(),
                    module.clone(),
                    pool.clone(),
                )
            })
            .for_each(|(tx, force, f_mod, b_mod, pool)| {
                // 外力ごとに個別の短命なTokioタスクを生やし、順方向と逆方向のシミュレーションを同時に待機
                join_set.spawn(async move {
                    let (forward_stat, backward_stat) = tokio::join!(
                        simulate_single_case(f_mod, length, force, pool.clone()),
                        simulate_single_case(b_mod, length, -force, pool)
                    );
                    // 計算が終了したものからチャネルに結果を流し込むため、呼び出し元でも即座にファイル書き込み等の逐次処理が可能
                    let _ = tx.send((force, forward_stat, backward_stat)).await;
                });
            });

        tokio::spawn(async move {
            // 全てのタスクが完了するまでJoinSetを待機（リソースリークを防ぐため、裏で全ての完了を保証）
            while join_set.join_next().await.is_some() {}
            // 全タスク完了後のGPUリソース一括解放
            pool.destroy().await;
        });

        rx
    }

    /// 単一のシミュレーションに対するCUDAシミュレーションと非同期メモリ転送を司る関数
    async fn simulate_single_case(
        module: Arc<CudaModule>,
        length: f64,
        force: f64,
        pool: CudaWorkerPool,
    ) -> Statistics {
        // パラメーターをハッシュ化してシード値を生成
        let mut hasher = DefaultHasher::new();
        length.to_bits().hash(&mut hasher);
        force.to_bits().hash(&mut hasher);
        let seed = hasher.finish();

        // プールから使用するワーカーを取得
        let mut worker = pool.acquire().await;
        // 次のカーネル実行のためデバイスメモリを 0 に初期化
        worker.memset_zeros();

        // カーネルの非同期実行をストリームへ投入
        let func = module.load_function("displacements_sum").unwrap();
        unsafe {
            worker
                .stream
                .launch_builder(&func)
                .arg(&seed)
                .arg(&length)
                .arg(&(1.0 / length)) // 逆数を事前に計算して渡すことで、カーネル内での除算を乗算に変換し高速化
                .arg(&force)
                .arg(&mut worker.dev_disp)
                .arg(&mut worker.dev_sq_disp)
                .launch(LaunchConfig {
                    block_dim: (BLOCK_SIZE, 1, 1),
                    grid_dim: (ENSEMBLE_SIZE.div_ceil(BLOCK_SIZE), 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .unwrap();

        // わずか16バイトの転送であるため、spawn_blockingを用いてOSスレッド上で同期コピー（完了待機）を行う
        // これにより、イベントのポーリングによるReadLockの枯渇を防ぐ
        let (worker, h_disp, h_sq_disp) = tokio::task::spawn_blocking(move || {
            let h_disp = worker.stream.clone_dtoh(&worker.dev_disp).unwrap();
            let h_sq_disp = worker.stream.clone_dtoh(&worker.dev_sq_disp).unwrap();
            // 結果と所有権をそのまま一緒に返すためキューは一切ロック処理が不要
            (worker, h_disp[0], h_sq_disp[0])
        })
        .await
        .unwrap();

        // 使用し終わったワーカーをプールに返却
        pool.release(worker).await;

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
