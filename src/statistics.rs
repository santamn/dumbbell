const ENSEMBLE_SIZE: u64 = 30_000; // アンサンブル平均のサンプル数

pub struct Statistics {
    pub effective_diffusion: f64,
    pub first_passage_time: f64,
    pub nonlinear_mobility: f64,
}

pub use backend::{statistics, statistics_async};

#[cfg(feature = "gpu")]
mod backend {
    use super::{ENSEMBLE_SIZE, Statistics, diffusion, nonlinear_mobility};
    use crate::simulation::{DELTA_T, K, NOISE_SCALE, STEPS, TIME};

    unsafe extern "C" {
        unsafe fn calculate_displacement_sum_on_gpu(
            device_id: u64,
            k: f64,
            delta_t: f64,
            noise_scale: f64,
            steps: u64,
            ensemble_size: u64,
            seed: u64,
            length: f64,
            force_x: f64,
            total_displacement: *mut f64,
            total_square_displacement: *mut f64,
        );

        unsafe fn calculate_displacement_sum_on_gpu_async(
            device_id: u64,
            k: f64,
            delta_t: f64,
            noise_scale: f64,
            steps: u64,
            ensemble_size: u64,
            seed: u64,
            length: f64,
            force_x: f64,
            rust_callback: unsafe extern "C" fn(f64, f64, *mut std::ffi::c_void),
            user_data: *mut std::ffi::c_void,
        );
    }

    /// GPUを用いてアンサンブル平均を計算し、非線形移動度、整流尺度、有効拡散係数を求める
    pub fn statistics(length: f64, force: f64) -> Statistics {
        let mut disp_sum = 0.0;
        let mut sq_disp_sum = 0.0;

        // 全てのアンサンブルを1つのGPU（device_id = 0）で行い、結果を同期的に待つ
        unsafe {
            calculate_displacement_sum_on_gpu(
                1,
                K,
                DELTA_T,
                NOISE_SCALE,
                STEPS as u64,
                ENSEMBLE_SIZE,
                1,
                length,
                force,
                &mut disp_sum as *mut _,
                &mut sq_disp_sum as *mut _,
            );
        }

        let mean_displacement = disp_sum / ENSEMBLE_SIZE as f64;
        let mean_square_displacement = sq_disp_sum / ENSEMBLE_SIZE as f64;
        let mean_speed = mean_displacement / TIME;

        Statistics {
            effective_diffusion: diffusion(mean_displacement, mean_square_displacement, TIME),
            first_passage_time: 1.0 / mean_speed.abs(),
            nonlinear_mobility: nonlinear_mobility(mean_speed, force),
        }
    }

    use std::ffi::c_void;
    use tokio::sync::oneshot;

    /// CUDA側の非同期処理（計算・コピー）が完了した際に呼び出されるコールバック関数。
    /// Cの関数ポインタとして渡すため `extern "C"` を指定し、ユーザーデータとして
    /// `oneshot::Sender` の生のポインタを一緒に受け取ります。
    unsafe extern "C" fn gpu_done_callback(
        disp_sum: f64,
        sq_disp_sum: f64,
        user_data: *mut c_void,
    ) {
        // user_data として渡された生ポインタから Box<Sender> を復元し、リソースの所有権を取り戻す
        let sender = unsafe { Box::from_raw(user_data as *mut oneshot::Sender<(f64, f64)>) };
        // async 側の rx.await で待機しているタスクへ計算結果を送信して起床させる
        let _ = sender.send((disp_sum, sq_disp_sum));
    }

    /// GPUを用いてアンサンブル平均を計算する非同期（async）バージョンの関数。
    /// この関数は呼び出されると直ちにGPUに計算を投げ、完了まで現在のTokioタスクを
    /// （OSスレッドをブロックすることなく）完全にスリープさせます。
    pub async fn statistics_async(device_id: u64, length: f64, force: f64) -> Statistics {
        // 結果を受け取るための1回限りの通信チャネル(oneshot)を作成
        let (tx, rx) = oneshot::channel::<(f64, f64)>();
        // Sender をヒープ(Box)に置き、所有権を放棄して生ポインタに変換する
        // （C言語側のコールバックに持たせるため）
        let tx_ptr = Box::into_raw(Box::new(tx)) as *mut c_void;

        unsafe {
            // C側の非同期関数を呼び出す。GPUへのコマンド送信をスケジュールするだけで、
            // 関数自体は即座にリターンし、ブロックしない
            calculate_displacement_sum_on_gpu_async(
                device_id,
                K,
                DELTA_T,
                NOISE_SCALE,
                STEPS as u64,
                ENSEMBLE_SIZE,
                1,
                length,
                force,
                gpu_done_callback,
                tx_ptr,
            );
        }

        // CUDA計算が終わって gpu_done_callback が呼ばれるまで、Tokioタスクを非同期待機(await)させる。
        // この間、CPUは他の並行するタスク（別のGPUへの操作など）を処理できるため無駄がない
        let (disp_sum, sq_disp_sum) = rx.await.expect("GPU 完了コールバックの受信に失敗しました");

        let mean_displacement = disp_sum / ENSEMBLE_SIZE as f64;
        let mean_square_displacement = sq_disp_sum / ENSEMBLE_SIZE as f64;
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

    pub async fn statistics_async(_device_id: u64, length: f64, force: f64) -> Statistics {
        // CPU fallback for async. Just spawn blocking.
        tokio::task::spawn_blocking(move || statistics(length, force))
            .await
            .unwrap()
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
