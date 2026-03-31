use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use nalgebra::Vector2;
use rand::{SeedableRng, rngs::SmallRng};
use renderer::SimApp;
use simulation::{DELTA_T, K, Particle, STEPS};
use statistics::{GpuResources, alpha, statistics};
use std::fs::File;
use std::io::Write;
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tokio::{sync::Semaphore, task::JoinSet};

mod renderer;
mod simulation;
mod statistics;

// GPU 3の性能が最も良いので、GPU 3を優先的に使うようにGPUのIDを指定する
const GPU_IDS: [u64; 3] = [3, 1, 2];

fn main() {
    let lengths = [0.03, 0.04, 0.05, 0.06, 0.07, 0.09, 0.1];
    // Tokio のランタイム（非同期実行エンジン）を明示的に立ち上げる
    let rt = tokio::runtime::Runtime::new().unwrap();
    // 立ち上げたエンジンの上で非同期のメイン処理を実行し、全て終わるまで同期的にブロックして待つ
    rt.block_on(async {
        calculate_statistics(&lengths).await;
    });
}

#[allow(dead_code)]
async fn calculate_statistics(lengths: &[f64]) {
    let m = MultiProgress::new();

    // 各GPUごとに同時に実行できるシミュレーションのケース数を制限するためのセマフォ
    // 大量のシミュレーションが一気にGPUに積まれてVRAM不足になるのを防ぐ
    let semaphores: Vec<Arc<Semaphore>> = GPU_IDS
        .iter()
        .map(|_| Arc::new(Semaphore::new(4)))
        .collect();

    let style = ProgressStyle::default_bar()
        .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) - {msg}")
        .unwrap()
        .progress_chars("#>-");

    // 発行したすべてのシミュレーションが完了するのを待機する
    futures::future::join_all(lengths.iter().enumerate().map(|(i, &length)| {
        // lengthを順番に取り出し、シミュレーションをGPUにラウンドロビン方式で割り当てる
        let index = i % GPU_IDS.len(); // 0, 1, 2, 0, 1, 2...
        let semaphore = semaphores[index].clone();

        let pb = m.add(ProgressBar::new(100));
        pb.set_style(style.clone());
        pb.set_message(format!("length: {:.2} (GPU {})", length, GPU_IDS[index]));

        tokio::spawn(async move {
            // このGPUに割り当てられた実行枠を取得するまで待機
            let _permit = semaphore.acquire().await.unwrap();
            record_statistics(GPU_IDS[index], length, pb).await;
            // ブロックを抜けると _permit がドロップされ、次のシミュレーションがこのGPUで実行可能になる
        })
    }))
    .await;
}

#[allow(dead_code)]
async fn record_statistics(device_id: u64, length: f64, pb: ProgressBar) {
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

    // 外力1~100をそれぞれ順方向と逆方向の両方に印加するシミュレーションを非同期で計算するタスクを作成
    // 100個の力に対して順方向と逆方向の両方を計算するので、200個分のバッファが必要
    let mut gpu_resources = GpuResources::new(device_id, 200);
    let mut set: JoinSet<_> = (1..=100)
        .map(|i| {
            // closureの外でポインタを取得しておくことで、gpu_resourcesそのものがasync blockにmoveされるのを防ぐ
            let forward_ptr = gpu_resources.get_pointers(2 * i - 2);
            let backward_ptr = gpu_resources.get_pointers(2 * i - 1);

            async move {
                let (forward, backward) = tokio::join!(
                    statistics(device_id, length, i as f64, forward_ptr),
                    statistics(device_id, length, -(i as f64), backward_ptr)
                );
                (i, forward, backward)
            }
        })
        .collect();

    // 生成したタスクが完了したものから順に取り出し、ファイルに書き込む
    while let Some(Ok((i, forward, backward))) = set.join_next().await {
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

    // 全ての計算が終わった後にバッファを（OSスレッドをブロックさせずに）安全に解放
    gpu_resources.dispose().await;

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
