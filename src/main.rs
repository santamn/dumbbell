use nalgebra::Vector2;
use renderer::SimApp;
use simulation::Real;
use std::ops::Range;

mod renderer;
mod simulation;

fn main() {
    run_animation(0, 1000, -1.0..3.0, 0.08, Vector2::new(1.0, 0.0));
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
}
