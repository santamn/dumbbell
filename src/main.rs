#[cfg(feature = "gpu")]
use cudarc::driver::{CudaContext, CudaModule};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use nalgebra::Vector2;
use rand::{SeedableRng, rngs::SmallRng};
use renderer::SimApp;
use simulation::{DELTA_T, K, Particle, STEPS};
use statistics::{alpha, statistics};
use std::fs::File;
use std::io::Write;
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Semaphore;

mod renderer;
mod simulation;
mod statistics;

const GPU_IDS: [u64; 3] = [3, 1, 2];

fn main() {
    let lengths = [0.03, 0.04, 0.05, 0.06, 0.07, 0.09, 0.1];
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        calculate_statistics(&lengths).await;
    });
}

#[allow(dead_code)]
#[cfg(feature = "gpu")]
async fn calculate_statistics(lengths: &[f64]) {
    let m = MultiProgress::new();

    let semaphores: Vec<Arc<Semaphore>> = GPU_IDS
        .iter()
        .map(|_| Arc::new(Semaphore::new(4)))
        .collect();

    let devices: Vec<(Arc<CudaContext>, Arc<CudaModule>)> = GPU_IDS
        .iter()
        .map(|&id| {
            let ctx = CudaContext::new(id as usize).unwrap();
            let ptx = cudarc::nvrtc::Ptx::from_binary(
                include_bytes!(concat!(env!("OUT_DIR"), "/simulation.cubin")).to_vec(),
            );
            let module = ctx.load_module(ptx).unwrap();
            (ctx, module)
        })
        .collect();

    let style = ProgressStyle::default_bar()
        .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) - {msg}")
        .unwrap()
        .progress_chars("#>-");

    futures::future::join_all(lengths.iter().enumerate().map(|(i, &length)| {
        let index = i % GPU_IDS.len();
        let semaphore = semaphores[index].clone();

        let device_info = devices[index].clone();
        let current_gpu_id = GPU_IDS[index];

        let pb = m.add(ProgressBar::new(100));
        pb.set_style(style.clone());
        pb.set_message(format!("length: {:.2} (GPU {})", length, current_gpu_id));

        tokio::spawn(async move {
            let _permit = semaphore.acquire().await.unwrap();
            record_statistics(device_info, current_gpu_id, length, pb).await;
        })
    }))
    .await;
}

#[cfg(feature = "gpu")]
async fn record_statistics(
    device_info: (Arc<CudaContext>, Arc<CudaModule>),
    device_id: u64,
    length: f64,
    pb: ProgressBar,
) {
    let path = Path::new("data")
        .join(format!("new_K_{}", K))
        .join(format!("len_{:.2}", length));
    std::fs::create_dir_all(&path).expect("ディレクトリの作成に失敗");

    let mut config = File::create(path.join("config.txt")).unwrap();
    writeln!(config, "時間の刻み幅: {}", DELTA_T).unwrap();
    writeln!(config, "バネ定数: {}", K).unwrap();
    writeln!(config, "棒の長さ: {}", length).unwrap();

    let mut mu_dat = File::create(path.join("mu.dat")).unwrap();
    let mut d_dat = File::create(path.join("d_eff.dat")).unwrap();
    let mut time_dat = File::create(path.join("time.dat")).unwrap();
    let mut alpha_dat = File::create(path.join("alpha.dat")).unwrap();

    for i in 1..=100 {
        let (forward, backward) = {
            let (ctx1, mod1) = device_info.clone();
            let (ctx2, mod2) = device_info.clone();
            tokio::join!(
                statistics(ctx1, mod1, length, i as f64),
                statistics(ctx2, mod2, length, -(i as f64))
            )
        };

        writeln!(
            mu_dat,
            "{} {} {}",
            i, forward.nonlinear_mobility, backward.nonlinear_mobility
        )
        .unwrap();
        writeln!(
            d_dat,
            "{} {} {}",
            i, forward.effective_diffusion, backward.effective_diffusion
        )
        .unwrap();
        writeln!(
            time_dat,
            "{} {} {}",
            i, forward.first_passage_time, backward.first_passage_time
        )
        .unwrap();
        writeln!(
            alpha_dat,
            "{} {}",
            i,
            alpha(forward.nonlinear_mobility, backward.nonlinear_mobility)
        )
        .unwrap();

        pb.inc(1);
    }

    pb.finish_with_message(format!("length: {:.2} (GPU {}) 完了", length, device_id));
}

#[allow(dead_code)]
fn single_particle_simulation(seed: u64, rod_length: f64, force: Vector2<f64>) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut particle = Particle::new(&mut rng, rod_length, force);
    let start = particle.now().position.x;
    let time = Instant::now();
    println!("変位: {}", particle.nth(STEPS).unwrap().position.x - start);
    println!("計算時間: {:.3?}", time.elapsed());
}

#[allow(dead_code)]
fn run_animation(
    seed: u64,
    sample_stride: usize,
    x_range: Range<f64>,
    rod_length: f64,
    force: Vector2<f64>,
) {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1000.0, 720.0])
            .with_resizable(false),
        ..Default::default()
    };
    let app = SimApp::new(seed, sample_stride, x_range, rod_length, force);

    eframe::run_native(
        "Brownian Motion Viewer",
        options,
        Box::new(|_| Ok(Box::new(app))),
    )
    .expect("eframe failed to start");

    std::process::exit(0);
}
