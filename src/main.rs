use nalgebra::Vector2;
use rand::{SeedableRng, rngs::SmallRng};
use renderer::SimApp;
use simulation::{DELTA_T, K, Particle, STEPS};
use statistics::{alpha, statistics_async};
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

const GPU_IDS: [u64; 3] = [1, 2, 3];

fn main() {
    let lengths = [0.03, 0.04, 0.05, 0.06, 0.07, 0.09, 0.1];
    // Tokio のランタイム（非同期実行エンジン）を明示的に立ち上げる
    let rt = tokio::runtime::Runtime::new().unwrap();
    // 立ち上げたエンジンの上で、非同期のメイン処理を実行し、全て終わるまで同期的にブロックして待つ
    rt.block_on(async {
        calculate_statistics(&lengths).await;
    });
}

// TODO: 全体での進捗率を表示する機能を追加する

#[allow(dead_code)]
async fn calculate_statistics(lengths: &[f64]) {
    // 各GPUごとに同時に実行できるシミュレーションのケース数を制限するためのセマフォ
    // 大量のシミュレーションが一気にGPUに積まれてVRAM不足になるのを防ぐ
    let semaphores: Vec<Arc<Semaphore>> = GPU_IDS
        .iter()
        .map(|_| Arc::new(Semaphore::new(2)))
        .collect();

    // 発行したすべてのシミュレーションが完了するのを待機する
    futures::future::join_all(lengths.iter().enumerate().map(|(i, &length)| {
        // lengthを順番に取り出し、シミュレーションをGPUにラウンドロビンで割り当てる
        let index = i % GPU_IDS.len(); // 0, 1, 2, 0, 1, 2...
        let semaphore = semaphores[index].clone();

        tokio::spawn(async move {
            // このGPUに割り当てられた実行枠を取得するまで待機
            let _permit = semaphore.acquire().await.unwrap();
            record_statistics(GPU_IDS[index], length).await;
            // ブロックを抜けると _permit がドロップされ、次のシミュレーションがこのGPUで実行可能になる
        })
    }))
    .await;
}

#[allow(dead_code)]
async fn record_statistics(device_id: u64, length: f64) {
    let path = Path::new("data")
        .join(format!("K_{}", K))
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
    let mut set: JoinSet<_> = (1..=100)
        .map(|i| async move {
            let (forward, backward) = tokio::join!(
                statistics_async(device_id, length, i as f64),
                statistics_async(device_id, length, -(i as f64))
            );
            (i, forward, backward)
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
    }
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
