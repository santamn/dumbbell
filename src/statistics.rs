use crate::simulation::{Particle, Real, STEPS, TIME};
use nalgebra::Vector2;
use rand::{SeedableRng, rngs::SmallRng};
use rayon::prelude::*;

const ENSEMBLE_SIZE: u64 = 30_000; // アンサンブル平均のサンプル数

pub struct Statistics {
    pub effective_diffusion: Real,
    pub first_passage_time: Real,
    pub nonlinear_mobility: Real,
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    unsafe fn run_simulation_cuda(
        device_id: u64,
        seed: u64,
        length: Real,
        force_x: Real,
        steps: u64,
        ensemble_size: u64,
        total_displacement: *mut Real,
        total_square_displacement: *mut Real,
    );
}

#[cfg(feature = "cuda")]
/// アンサンブル平均を用いて、非線形移動度、整流尺度、有効拡散係数を計算する
pub fn statistics(length: Real, force: Real) -> Statistics {
    let mut total_displacement = 0.0;
    let mut total_square_displacement = 0.0;

    let total_gpus = 4;
    let per_gpu_size = ENSEMBLE_SIZE / total_gpus;

    for device_id in 0..total_gpus {
        unsafe {
            run_simulation_cuda(
                device_id,
                12345 + device_id,
                length,
                force,
                STEPS as u64,
                per_gpu_size,
                &mut total_displacement as *mut _,
                &mut total_square_displacement as *mut _,
            );
        }
    }

    let mean_displacement = total_displacement / ENSEMBLE_SIZE as Real;
    let mean_square_displacement = total_square_displacement / ENSEMBLE_SIZE as Real;

    let mean_speed = mean_displacement / TIME;
    let mu = nonlinear_mobility(mean_speed, force);

    Statistics {
        effective_diffusion: diffusion(mean_displacement, mean_square_displacement, TIME),
        first_passage_time: 1.0 / mean_speed,
        nonlinear_mobility: mu,
    }
}

#[cfg(not(feature = "cuda"))]
/// アンサンブル平均を用いて、非線形移動度、整流尺度、有効拡散係数を計算する
pub fn statistics(length: Real, force: Real) -> Statistics {
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
        .map(|(sum, sq_sum)| (sum / ENSEMBLE_SIZE as Real, sq_sum / ENSEMBLE_SIZE as Real))
        .unwrap();

    let mean_speed = mean_displacement / TIME;
    let mu = nonlinear_mobility(mean_speed, force);

    Statistics {
        effective_diffusion: diffusion(mean_displacement, mean_square_displacement, TIME),
        first_passage_time: 1.0 / mean_speed,
        nonlinear_mobility: mu,
    }
}

#[allow(dead_code)]
pub fn particle_distribution() {
    todo!("アンサンブル平均により、粒子の位置分布を推定する機能を実装する予定")
}

/// 非線形移動度 μ(f) = ⟨v⟩/|f|
fn nonlinear_mobility(mean_speed: Real, force: Real) -> Real {
    mean_speed / force.abs()
}

/// 有効拡散係数 D_eff = (⟨x²⟩ - ⟨x⟩²)/2t
fn diffusion(mean_displacement: Real, mean_square_displacement: Real, time: Real) -> Real {
    (mean_square_displacement - mean_displacement * mean_displacement) / (2.0 * time)
}
