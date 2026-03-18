#[cfg(not(feature = "cuda"))]
use crate::simulation::Particle;
use crate::simulation::{DELTA_T, K, NOISE_SCALE, STEPS, TIME};
#[cfg(not(feature = "cuda"))]
use nalgebra::Vector2;
#[cfg(not(feature = "cuda"))]
use rand::{SeedableRng, rngs::SmallRng};
use rayon::prelude::*;

const ENSEMBLE_SIZE: u64 = 30_000; // アンサンブル平均のサンプル数
#[cfg(feature = "cuda")]
const TOTAL_GPUS: u64 = 3; // 使用するGPUの数

pub struct Statistics {
    pub effective_diffusion: f64,
    pub first_passage_time: f64,
    pub nonlinear_mobility: f64,
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    unsafe fn run_simulation_cuda(
        k: f64,
        delta_t: f64,
        noise_scale: f64,
        device_id: u64,
        seed: u64,
        length: f64,
        force_x: f64,
        steps: u64,
        ensemble_size: u64,
        total_displacement: *mut f64,
        total_square_displacement: *mut f64,
    );
}

#[cfg(feature = "cuda")]
pub fn statistics(length: f64, force: f64) -> Statistics {
    // 各GPUでの計算を並列で実行する
    let (mean_displacement, mean_square_displacement) = (0..TOTAL_GPUS)
        .into_par_iter()
        .map(|device_id| {
            let mut local_disp = 0.0;
            let mut local_sq_disp = 0.0;

            unsafe {
                run_simulation_cuda(
                    K,
                    DELTA_T,
                    NOISE_SCALE,
                    device_id,
                    1 + device_id, // GPUごとにシードを変える
                    length,
                    force,
                    STEPS as u64,
                    ENSEMBLE_SIZE / TOTAL_GPUS, // 各GPUで処理するサンプル数
                    &mut local_disp as *mut _,
                    &mut local_sq_disp as *mut _,
                );
            }

            (local_disp, local_sq_disp)
        })
        .reduce_with(|(d1, sq1), (d2, sq2)| (d1 + d2, sq1 + sq2))
        .map(|(sum, sq_sum)| (sum / ENSEMBLE_SIZE as f64, sq_sum / ENSEMBLE_SIZE as f64))
        .unwrap();

    let mean_speed = mean_displacement / TIME;

    Statistics {
        effective_diffusion: diffusion(mean_displacement, mean_square_displacement, TIME),
        first_passage_time: 1.0 / mean_speed,
        nonlinear_mobility: nonlinear_mobility(mean_speed, force),
    }
}

#[cfg(not(feature = "cuda"))]
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
