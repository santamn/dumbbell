use nalgebra::Vector2;

mod renderer;
mod simulation;

fn main() {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1000.0, 720.0])
            .with_resizable(false),
        ..Default::default()
    };

    eframe::run_native(
        "Dumbbell Particle Brownian Motion Viewer",
        options,
        Box::new(|_| {
            Ok(Box::new(renderer::SimApp::new(
                0,
                1000,
                -1.0..3.0,
                0.08,
                Vector2::new(1.0, 0.0),
            )))
        }),
    )
    .expect("eframe failed to start");
}
