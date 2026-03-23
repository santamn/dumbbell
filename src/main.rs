use nalgebra::Vector2;
use rand::{SeedableRng, rngs::SmallRng};
use renderer::SimApp;
use simulation::{DELTA_T, K, Particle, STEPS};
use statistics::statistics;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::ops::Range;
use std::path::Path;
use std::time::Instant;

mod renderer;
mod simulation;
mod statistics;

fn main() {
    record_statistics("len0.02_K1.5e6", 0.02);
}

#[allow(dead_code)]
fn single_particle_simulation(seed: u64, rod_length: f64, force: Vector2<f64>) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut particle = Particle::new(&mut rng, rod_length, force);
    let start = particle.now().position.x;
    let time = Instant::now();
    println!("変位: {}", particle.nth(STEPS).unwrap().position.x - start);
    println!("計算時間: {:.3?}秒", time.elapsed());
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

    writeln!(config, "計算時間: {:.3?}秒", start.elapsed()).unwrap();
}
