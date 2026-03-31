pub struct Statistics {
    pub effective_diffusion: f64,
    pub first_passage_time: f64,
    pub nonlinear_mobility: f64,
}

pub use backend::statistics;

#[cfg(feature = "gpu")]
mod backend {
    use super::{Statistics, diffusion, nonlinear_mobility};
    use crate::simulation::{ENSEMBLE_SIZE, TIME};
    use cudarc::driver::{CudaContext, CudaModule, LaunchConfig, PushKernelArg};
    use std::hash::{DefaultHasher, Hash, Hasher};
    use std::sync::Arc;
    use tokio::task::spawn_blocking;

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
