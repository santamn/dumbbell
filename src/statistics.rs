const ENSEMBLE_SIZE: u64 = 30_000; // アンサンブル平均のサンプル数

pub struct Statistics {
    pub effective_diffusion: f64,
    pub first_passage_time: f64,
    pub nonlinear_mobility: f64,
}

pub use backend::statistics;

#[cfg(feature = "gpu")]
mod backend {
    use super::{ENSEMBLE_SIZE, Statistics, diffusion, nonlinear_mobility};
    use crate::simulation::{DELTA_T, K, NOISE_SCALE, STEPS, TIME};
    use std::ffi::c_void;
    use std::future::Future;
    use std::pin::Pin;
    use std::task::{Context, Poll};

    const TOTAL_GPUS: u64 = 3; // 使用するGPUの数

    unsafe extern "C" {
        unsafe fn start_calculate_displacement_sum_on_gpu(
            device_id: u64,
            k: f64,
            delta_t: f64,
            noise_scale: f64,
            steps: u64,
            ensemble_size: u64,
            seed: u64,
            length: f64,
            force_x: f64,
        ) -> *mut c_void;

        unsafe fn query_calculate_displacement_sum_on_gpu(
            handle: *mut c_void,
            total_displacement: *mut f64,
            total_square_displacement: *mut f64,
        ) -> std::ffi::c_int;
    }

    /// GPUタスクの完了を待機するためのFuture構造体
    struct GpuTaskFuture {
        handle: Option<*mut c_void>,
    }

    impl Future for GpuTaskFuture {
        type Output = (f64, f64);

        fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
            let mut disp_sum = 0.0;
            let mut sq_disp_sum = 0.0;

            unsafe {
                let handle = self.handle.unwrap();
                // CUDAのイベント状態をポーリングして確認
                let status = query_calculate_displacement_sum_on_gpu(
                    handle,
                    &mut disp_sum,
                    &mut sq_disp_sum,
                );

                if status == 1 {
                    // 計算が完了していれば結果を返す
                    Poll::Ready((disp_sum, sq_disp_sum))
                } else if status == -1 {
                    // エラー発生時のフォールバック処理
                    panic!("CUDA error during query_calculate_displacement_sum_on_gpu");
                } else {
                    // 未完了の場合、ビジーループを避けるために10ミリ秒待機してから再ポーリングするようにWakerを登録
                    let waker = cx.waker().clone();
                    tokio::spawn(async move {
                        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                        waker.wake();
                    });
                    Poll::Pending
                }
            }
        }
    }

    unsafe impl Send for GpuTaskFuture {}

    /// GPUを用いてアンサンブル平均を計算し、非線形移動度、整流尺度、有効拡散係数を求める
    pub fn statistics(length: f64, force: f64) -> Statistics {
        // Tokioのシングルスレッドランタイムを生成して、現在のスレッド上で非同期処理をブロック実行
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                // 各GPUでの計算を非同期タスクとして開始する
                let mut futures = Vec::new();
                for device_id in 0..TOTAL_GPUS {
                    let handle = unsafe {
                        start_calculate_displacement_sum_on_gpu(
                            device_id,
                            K,
                            DELTA_T,
                            NOISE_SCALE,
                            STEPS as u64,
                            ENSEMBLE_SIZE / TOTAL_GPUS, // 各GPUで処理するサンプル数
                            1 + device_id,              // 各GPUに与えるシード値
                            length,
                            force,
                        )
                    };
                    futures.push(GpuTaskFuture {
                        handle: Some(handle),
                    });
                }

                // 全てのGPUタスクの完了を並行して待機
                let results = futures::future::join_all(futures).await;

                // 各GPUの計算結果を合算する
                let (mean_displacement, mean_square_displacement) = results
                    .into_iter()
                    .fold((0.0, 0.0), |(a, aa), (d, sq)| (a + d, aa + sq));

                let mean_displacement = mean_displacement / ENSEMBLE_SIZE as f64;
                let mean_square_displacement = mean_square_displacement / ENSEMBLE_SIZE as f64;

                let mean_speed = mean_displacement / TIME;

                Statistics {
                    effective_diffusion: diffusion(
                        mean_displacement,
                        mean_square_displacement,
                        TIME,
                    ),
                    first_passage_time: 1.0 / mean_speed,
                    nonlinear_mobility: nonlinear_mobility(mean_speed, force),
                }
            })
    }
}

#[cfg(not(feature = "gpu"))]
mod backend {
    use super::{ENSEMBLE_SIZE, Statistics, diffusion, nonlinear_mobility};
    use crate::simulation::{Particle, STEPS, TIME};
    use nalgebra::Vector2;
    use rand::{SeedableRng, rngs::SmallRng};
    use rayon::prelude::*;

    /// アンサンブル平均を用いて、非線形移動度、整流尺度、有効拡散係数を計算する
    pub fn statistics(length: f64, force: f64) -> Statistics {
        let force_vec = Vector2::new(force, 0.0);
        let (mean_displacement, mean_square_displacement) = (0..ENSEMBLE_SIZE)
            .into_par_iter()
            .map(|i| {
                let rng = SmallRng::seed_from_u64(i);
                let mut particle = Particle::new(rng, length, force_vec);
                let start = particle.now().position.x;
                let delta_x = particle.nth(STEPS).unwrap().position.x - start; // 移動距離

                (delta_x, delta_x * delta_x) // 変位, 二乗変位
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

#[allow(dead_code)]
pub fn particle_distribution() {
    todo!("アンサンブル平均により、粒子の位置分布を推定する機能を実装する予定")
}

/// 非線形移動度 μ(f) = ⟨v⟩/|f|
fn nonlinear_mobility(mean_speed: f64, force: f64) -> f64 {
    mean_speed / force
}

/// 有効拡散係数 D_eff = (⟨x²⟩ - ⟨x⟩²)/2t
fn diffusion(mean_displacement: f64, mean_square_displacement: f64, time: f64) -> f64 {
    (mean_square_displacement - mean_displacement * mean_displacement) / (2.0 * time)
}
