#[cfg(feature = "gpu")]
use cudarc::driver::{CudaContext, CudaModule};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use nalgebra::Vector2;
use rand::{SeedableRng, rngs::SmallRng};
use renderer::SimApp;
use simulation::{DELTA_T, K, Particle, STEPS};
use statistics::alpha;
#[cfg(feature = "gpu")]
use statistics::statistics;
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

// GPU 3の性能が最も良いので、GPU 3を優先的に使うようにGPUのIDを指定する
const GPU_IDS: [usize; 3] = [3, 1, 2];

fn main() {
    let lengths = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1];
    // Tokio のランタイム（非同期実行エンジン）を明示的に立ち上げる
    let rt = tokio::runtime::Runtime::new().unwrap();
    // 立ち上げたエンジンの上で非同期のメイン処理を実行し、全て終わるまで同期的にブロックして待つ
    rt.block_on(async {
        calculate_statistics(&lengths).await;
    });
}

#[allow(dead_code)]
#[cfg(feature = "gpu")]
async fn calculate_statistics(lengths: &[f64]) {
    // プログレスバーを作成
    let m = MultiProgress::new();

    // 各GPUごとに同時に実行できるシミュレーションのケース数を制限するためのセマフォ
    // 大量のシミュレーションが一気にGPUに積まれてVRAM不足になるのを防ぐ
    let semaphores: Vec<Arc<Semaphore>> = GPU_IDS
        .iter()
        .map(|_| Arc::new(Semaphore::new(1)))
        .collect();

    // 各GPUのコンテキストとモジュールを事前に作成しておき、シミュレーションタスクに渡すためのタプルを作成
    let devices: Vec<(Arc<CudaContext>, Arc<CudaModule>)> = GPU_IDS
        .iter()
        .map(|&id| {
            let ctx = CudaContext::new(id).unwrap();
            let ptx = cudarc::nvrtc::Ptx::from_binary(
                include_bytes!(concat!(env!("OUT_DIR"), "/simulation.cubin")).to_vec(),
            );
            let module = ctx.load_module(ptx).unwrap();

            (ctx, module)
        })
        .collect();

    // プログレスバーのスタイルを定義
    let style = ProgressStyle::default_bar()
        .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) - {msg}")
        .unwrap()
        .progress_chars("=>-");

    // 発行したすべてのシミュレーションが完了するのを待機する
    futures::future::join_all(lengths.iter().enumerate().map(|(i, &length)| {
        // lengthを順番に取り出し、シミュレーションをGPUにラウンドロビン方式で割り当てる
        let index = i % GPU_IDS.len();
        let semaphore = semaphores[index].clone();

        let pb = m.add(ProgressBar::new(100));
        pb.set_style(style.clone());
        pb.set_message(format!("length: {:.2} (GPU {})", length, GPU_IDS[index]));

        let device = devices[index].clone();
        tokio::spawn(async move {
            // このGPUに割り当てられた実行枠を取得するまで待機
            let _permit = semaphore.acquire().await.unwrap();
            record_statistics(device, length, pb).await;
            // ブロックを抜けると _permit がドロップされ、次のシミュレーションがこのGPUで実行可能になる
        })
    }))
    .await;
}

#[cfg(feature = "gpu")]
async fn record_statistics(
    device: (Arc<CudaContext>, Arc<CudaModule>),
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

    let (ctx, module) = device;
    let device_id = ctx.ordinal();

    // 全ての計算を並列に GPU に投げ、完了したものから受け取る流し込み処理
    let mut rx = statistics(ctx, module, length, 1..=100).await;
    while let Some((force, forward, backward)) = rx.recv().await {
        writeln!(
            mu_dat,
            "{} {} {}",
            force, forward.nonlinear_mobility, backward.nonlinear_mobility
        )
        .unwrap();
        writeln!(
            d_dat,
            "{} {} {}",
            force, forward.effective_diffusion, backward.effective_diffusion
        )
        .unwrap();
        writeln!(
            time_dat,
            "{} {} {}",
            force, forward.first_passage_time, backward.first_passage_time
        )
        .unwrap();
        writeln!(
            alpha_dat,
            "{} {}",
            force,
            alpha(forward.nonlinear_mobility, backward.nonlinear_mobility)
        )
        .unwrap();

        // 1ケース完了ごとにプログレスバーを1つ進める
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
