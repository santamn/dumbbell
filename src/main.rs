use nalgebra::Vector2;
use rand::{SeedableRng, rngs::SmallRng};
use renderer::SimApp;
use simulation::{DELTA_T, K, Particle, STEPS};
use statistics::{statistics, statistics_async};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Semaphore;

mod renderer;
mod simulation;
mod statistics;

// #[tokio::main] マクロにより、プログラム全体を非同期ランタイム上で動作させる
#[tokio::main]
async fn main() {
    // 測定したい length のリスト（自由に種類を追加可能）
    let lengths = vec![0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10];

    // 利用可能な GPU ID のリスト
    let gpu_ids = [1, 2, 3];

    // 各GPUごとに同時に実行できる「length」の数を制限するためのセマフォ（例: 2個まで）
    // （これがないと数十種類のlengthが一気にA100に積まれてVRAM不足になるのを防ぐ）
    let semaphores: Vec<Arc<Semaphore>> = gpu_ids
        .iter()
        .map(|_| Arc::new(Semaphore::new(2)))
        .collect();

    let mut tasks = Vec::new();

    // length のリストを順番に取り出し、GPU にラウンドロビン（順番）で割り当てる
    for (i, &length) in lengths.iter().enumerate() {
        let gpu_index = i % gpu_ids.len(); // 0, 1, 2, 0, 1, 2...
        let device_id = gpu_ids[gpu_index];
        let sem_clone = semaphores[gpu_index].clone();

        // フォルダ名を動的に生成
        let folder_name = format!("len{:.2}_K1.5e6", length);

        let task = tokio::spawn(async move {
            // このGPUに割り当てられた実行枠（許可証）を取得するまで待機
            let _permit = sem_clone.acquire().await.unwrap();

            println!(
                "> 開始: length = {:.2} (担当 GPU ID: {})",
                length, device_id
            );
            record_statistics_async(&folder_name, length, device_id).await;
            println!(
                "< 完了: length = {:.2} (担当 GPU ID: {})",
                length, device_id
            );

            // ブロックを抜けると _permit がドロップされ、次の length がこのGPUで実行可能になる
        });

        tasks.push(task);
    }

    // 発行したすべてのタスク（全 length の測定）が完了するのを待機する
    futures::future::join_all(tasks).await;
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

#[allow(dead_code)]
fn record_statistics(folder_name: &str, length: f64) {
    let path = Path::new("data").join(folder_name);
    std::fs::create_dir_all(&path).expect("Failed to create data directory");

    let mut config = File::create(path.join("config.txt")).unwrap();
    writeln!(config, "時間の刻み幅: {}", DELTA_T).unwrap();
    writeln!(config, "バネ定数: {}", K).unwrap();
    writeln!(config, "棒の長さ: {}", length).unwrap();

    let mut mu_writer = BufWriter::new(
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(path.join("mu.dat"))
            .unwrap(),
    );
    let mut d_writer = BufWriter::new(
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(path.join("d_eff.dat"))
            .unwrap(),
    );
    let mut time_writer = BufWriter::new(
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(path.join("time.dat"))
            .unwrap(),
    );
    let mut alpha_writer = BufWriter::new(
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(path.join("alpha.dat"))
            .unwrap(),
    );

    let start = Instant::now();
    for i in 1..=100 {
        let forward = statistics(length, i as f64);
        let backward = statistics(length, -(i as f64));

        writeln!(
            mu_writer,
            "{} {} {}",
            i, forward.nonlinear_mobility, backward.nonlinear_mobility
        )
        .unwrap();
        writeln!(
            d_writer,
            "{} {} {}",
            i, forward.effective_diffusion, backward.effective_diffusion
        )
        .unwrap();
        writeln!(
            time_writer,
            "{} {} {}",
            i, forward.first_passage_time, backward.first_passage_time
        )
        .unwrap();
        writeln!(
            alpha_writer,
            "{} {}",
            i,
            (forward.nonlinear_mobility - backward.nonlinear_mobility).abs()
                / (forward.nonlinear_mobility + backward.nonlinear_mobility)
        )
        .unwrap();

        mu_writer.flush().unwrap();
        d_writer.flush().unwrap();
        time_writer.flush().unwrap();
        alpha_writer.flush().unwrap();
    }

    writeln!(config, "計算時間: {:.3?}", start.elapsed()).unwrap();
}

#[allow(dead_code)]
async fn record_statistics_async(folder_name: &str, length: f64, device_id: u64) {
    let path = Path::new("data").join(folder_name);
    std::fs::create_dir_all(&path).expect("Failed to create data directory");

    let mut config = File::create(path.join("config.txt")).unwrap();
    writeln!(config, "時間の刻み幅: {}", DELTA_T).unwrap();
    writeln!(config, "バネ定数: {}", K).unwrap();
    writeln!(config, "棒の長さ: {}", length).unwrap();

    let mut mu_writer = BufWriter::new(
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(path.join("mu.dat"))
            .unwrap(),
    );
    let mut d_writer = BufWriter::new(
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(path.join("d_eff.dat"))
            .unwrap(),
    );
    let mut time_writer = BufWriter::new(
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(path.join("time.dat"))
            .unwrap(),
    );
    let mut alpha_writer = BufWriter::new(
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(path.join("alpha.dat"))
            .unwrap(),
    );

    let start = Instant::now();

    // 1から100までの force に対する、forward と backward 両方の非同期タスクを一気に生成する（まだ待機しない）
    let mut futures = Vec::new();
    for i in 1..=100 {
        futures.push(async move {
            let f = statistics_async(device_id, length, i as f64);
            let b = statistics_async(device_id, length, -(i as f64));
            // 同じ i に対する forward と backward も並行して待つ
            let (forward, backward) = tokio::join!(f, b);
            (i, forward, backward)
        });
    }

    // 作成した 100回分(×2) のタスクをGPUへ一気に投げて、すべて完了するまで並行して待機する
    let results = futures::future::join_all(futures).await;

    // 全て計算が終わったら、結果を順番にファイルへ書き出す
    for (i, forward, backward) in results {
        writeln!(
            mu_writer,
            "{} {} {}",
            i, forward.nonlinear_mobility, backward.nonlinear_mobility
        )
        .unwrap();
        writeln!(
            d_writer,
            "{} {} {}",
            i, forward.effective_diffusion, backward.effective_diffusion
        )
        .unwrap();
        writeln!(
            time_writer,
            "{} {} {}",
            i, forward.first_passage_time, backward.first_passage_time
        )
        .unwrap();

        let alpha_forward = (2.0 * forward.effective_diffusion) / forward.nonlinear_mobility;
        let alpha_backward = (2.0 * backward.effective_diffusion) / (-backward.nonlinear_mobility);

        writeln!(alpha_writer, "{} {} {}", i, alpha_forward, alpha_backward).unwrap();

        println!(
            "[Device {}] length: {}, force: {}, elapsed: {:.2?}",
            device_id,
            length,
            i,
            start.elapsed()
        );
    }

    writeln!(config, "計算時間: {:.3?}", start.elapsed()).unwrap();
}
