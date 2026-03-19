use nalgebra::Vector2;
use renderer::SimApp;
use simulation::{DELTA_T, K, Real};
use statistics::statistics;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::ops::Range;
use std::path::Path;

mod renderer;
mod simulation;
mod statistics;

fn main() {
    run_animation(0, 1000, 0.0..5.0, 0.08, Vector2::new(1.0, 0.0));
}

#[allow(dead_code)]
fn run_animation(
    seed: u64,
    sample_stride: usize,
    x_range: Range<Real>,
    rod_length: Real,
    force: Vector2<Real>,
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
fn record_statistics(folder_name: &str, length: Real) {
    let path = Path::new("data").join(folder_name);
    std::fs::create_dir_all(&path).expect("Failed to create data directory");

    let mut config = File::create(path.join("config.txt")).unwrap();
    writeln!(config, "時間の刻み幅: {}", DELTA_T).unwrap();
    writeln!(config, "バネ定数: {}", K).unwrap();
    writeln!(config, "棒の長さ: {}", length).unwrap();

    let mut mu_writer = BufWriter::new(File::create(path.join("mu.dat")).unwrap());
    let mut d_writer = BufWriter::new(File::create(path.join("d_eff.dat")).unwrap());
    let mut time_writer = BufWriter::new(File::create(path.join("time.dat")).unwrap());
    let mut alpha_writer = BufWriter::new(File::create(path.join("alpha.dat")).unwrap());

    let start = std::time::Instant::now();

    for i in 1..=100 {
        let forward = statistics(length, i as Real);
        let backward = statistics(length, -(i as Real));

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
    }

    writeln!(config, "計算時間: {:.2?}秒", start.elapsed()).unwrap();
}
