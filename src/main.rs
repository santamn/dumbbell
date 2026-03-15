use nalgebra::{Point2, Vector2};
use rand::{Rng, RngExt, SeedableRng, rngs::SmallRng};
use rand_distr::StandardNormal;
use std::collections::VecDeque;
use std::f64::consts::{PI, TAU};
use std::ops::Add;

type Real = f64; // 計算の精度を決める型

const fn const_sqrt(x: Real) -> Real {
    assert!(
        0.0 <= x && x <= 1.0,
        "sqrt_constは0以上1以下の値に対してのみ定義されます"
    );

    let mut guess = 1.0;
    let mut iterations = 0;
    while iterations < 32 {
        guess = 0.5 * (guess + x / guess);
        iterations += 1;
    }
    guess
}

const PI_SQUARED: Real = PI * PI; // π^2

const PARTICLES: u64 = 30_000; //                  アンサンブル平均に用いる粒子数  3×10^4
const TIME: Real = 10.0; //                        総シミュレーション時間         10.0
const DELTA_T: Real = 2e-7; //                     時間刻み幅                2.0×10^-7
const NOISE_SCALE: Real = const_sqrt(DELTA_T); //  ブラウン運動のノイズの大きさ  
const STEPS: usize = (TIME / DELTA_T) as usize; // ステップ数        T / Δt = 5×10^7

#[derive(Debug, Clone, Copy, PartialEq)]
struct State {
    position: Point2<Real>,
    angle: Real,
}

impl State {
    fn new<R>(rng: &mut R, length: Real) -> Self
    where
        R: Rng + ?Sized,
    {
        let x = rng.random_range(0.0..1.0);
        let limit = omega(&x) - length * 0.5;
        Self {
            position: Point2::new(x, rng.random_range(-limit..limit)),
            angle: rng.random_range(0.0..TAU),
        }
    }
}

impl Add<(Vector2<Real>, Real)> for State {
    type Output = Self;

    fn add(self, other: (Vector2<Real>, Real)) -> Self {
        Self {
            position: self.position + other.0,
            angle: self.angle + other.1,
        }
    }
}

trait Wall {
    const SIGN: Real; // 壁の向き (上壁なら1.0、下壁なら-1.0)
}

struct Ceiling;
impl Wall for Ceiling {
    const SIGN: Real = 1.0;
}

struct Floor;
impl Wall for Floor {
    const SIGN: Real = -1.0;
}

fn main() {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "Dumbbell Brownian Motion Viewer",
        options,
        Box::new(|_cc| Ok(Box::new(SimApp::default()))),
    )
    .expect("eframe failed to start");
}

/// ブラウン運動の軌跡を生成
fn brown_motion_iter<R: Rng>(
    rng: &mut R,
    force: Vector2<Real>,
    length: Real,
) -> impl Iterator<Item = State> {
    std::iter::successors(Some(State::new(rng, length)), move |&particle| {
        let xi_x = rng.sample(StandardNormal);
        let xi_y = rng.sample(StandardNormal);
        let xi_phi = rng.sample::<Real, _>(StandardNormal);

        let (s, c) = particle.angle.sin_cos();
        let e_phi = Vector2::new(-s, c);
        let h = 0.5 * length * Vector2::new(c, s);
        let (p1, p2) = (particle.position + h, particle.position - h);
        let (f1, f2) = (force + repulsion(&p1), force + repulsion(&p2));

        Some(
            particle
                + (
                    0.5 * (f1 + f2) * DELTA_T + Vector2::new(xi_x, xi_y) * NOISE_SCALE,
                    ((f1 - f2).dot(&e_phi) * DELTA_T + 2.0 * xi_phi * NOISE_SCALE) / length,
                ),
        )
    })
    .take(STEPS)
}

fn brown_motion_step<R: Rng>(
    rng: &mut R,
    particle: State,
    force: Vector2<Real>,
    length: Real,
) -> State {
    let xi_x = rng.sample(StandardNormal);
    let xi_y = rng.sample(StandardNormal);
    let xi_phi = rng.sample::<Real, _>(StandardNormal);

    let (s, c) = particle.angle.sin_cos();
    let e_phi = Vector2::new(-s, c);
    let h = 0.5 * length * Vector2::new(c, s);
    let (p1, p2) = (particle.position + h, particle.position - h);
    let (f1, f2) = (force + repulsion(&p1), force + repulsion(&p2));

    particle
        + (
            0.5 * (f1 + f2) * DELTA_T + Vector2::new(xi_x, xi_y) * NOISE_SCALE,
            ((f1 - f2).dot(&e_phi) * DELTA_T + 2.0 * xi_phi * NOISE_SCALE) / length,
        )
}

struct SimApp {
    rng: SmallRng,
    state: State,
    force: Vector2<Real>,
    rod_length: Real,
    running: bool,
    steps_per_frame: usize,
    frame_stride: usize,
    trail_capacity: usize,
    trail: VecDeque<Point2<Real>>,
    sampled_step_count: usize,
    total_steps: usize,
    seed: u64,
    wall_samples: Vec<(Real, Real)>,
}

impl Default for SimApp {
    fn default() -> Self {
        let seed = 0;
        let mut rng = SmallRng::seed_from_u64(seed);
        let rod_length = 0.02;
        let state = State::new(&mut rng, rod_length);
        Self {
            rng,
            state,
            force: Vector2::zeros(),
            rod_length,
            running: true,
            steps_per_frame: 300,
            frame_stride: 8,
            trail_capacity: 2_000,
            trail: VecDeque::with_capacity(2_000),
            sampled_step_count: 0,
            total_steps: 0,
            seed,
            wall_samples: sample_walls(512),
        }
    }
}

impl SimApp {
    fn reset(&mut self) {
        self.rng = SmallRng::seed_from_u64(self.seed);
        self.state = State::new(&mut self.rng, self.rod_length);
        self.trail.clear();
        self.sampled_step_count = 0;
        self.total_steps = 0;
    }

    fn advance(&mut self) {
        if self.total_steps >= STEPS {
            self.running = false;
            return;
        }

        for _ in 0..self.steps_per_frame {
            if self.total_steps >= STEPS {
                self.running = false;
                break;
            }

            self.state = brown_motion_step(&mut self.rng, self.state, self.force, self.rod_length);
            self.total_steps += 1;
            self.sampled_step_count += 1;
            if self.sampled_step_count >= self.frame_stride {
                self.sampled_step_count = 0;
                self.trail.push_back(self.state.position);
                if self.trail.len() > self.trail_capacity {
                    let _ = self.trail.pop_front();
                }
            }
        }
    }

    fn world_to_screen(rect: egui::Rect, x: Real, y: Real) -> egui::Pos2 {
        let x_t = ((x as f32) / 1.0).clamp(0.0, 1.0);
        let y_min = -2.0f32;
        let y_max = 2.0f32;
        let y_t = (((y as f32) - y_min) / (y_max - y_min)).clamp(0.0, 1.0);
        egui::pos2(
            rect.left() + x_t * rect.width(),
            rect.bottom() - y_t * rect.height(),
        )
    }

    fn draw_scene(&self, ui: &mut egui::Ui, rect: egui::Rect) {
        let painter = ui.painter_at(rect);

        let stroke_wall = egui::Stroke::new(1.5, egui::Color32::from_rgb(120, 180, 220));
        let mut upper = Vec::with_capacity(self.wall_samples.len());
        let mut lower = Vec::with_capacity(self.wall_samples.len());
        for &(x, y) in &self.wall_samples {
            upper.push(Self::world_to_screen(rect, x, y));
            lower.push(Self::world_to_screen(rect, x, -y));
        }
        painter.add(egui::Shape::line(upper, stroke_wall));
        painter.add(egui::Shape::line(lower, stroke_wall));

        if self.trail.len() >= 2 {
            let mut points = Vec::with_capacity(self.trail.len());
            for point in &self.trail {
                points.push(Self::world_to_screen(rect, point.x, point.y));
            }
            painter.add(egui::Shape::line(
                points,
                egui::Stroke::new(1.0, egui::Color32::from_rgb(255, 170, 90)),
            ));
        }

        let (s, c) = self.state.angle.sin_cos();
        let half = 0.5 * self.rod_length;
        let p1 = Point2::new(
            self.state.position.x + half * c,
            self.state.position.y + half * s,
        );
        let p2 = Point2::new(
            self.state.position.x - half * c,
            self.state.position.y - half * s,
        );
        let p1s = Self::world_to_screen(rect, p1.x, p1.y);
        let p2s = Self::world_to_screen(rect, p2.x, p2.y);
        painter.line_segment(
            [p1s, p2s],
            egui::Stroke::new(3.0, egui::Color32::from_rgb(230, 90, 60)),
        );
        let center = Self::world_to_screen(rect, self.state.position.x, self.state.position.y);
        painter.circle_filled(center, 3.5, egui::Color32::WHITE);
    }
}

impl eframe::App for SimApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("controls").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui
                    .button(if self.running { "Pause" } else { "Play" })
                    .clicked()
                {
                    self.running = !self.running;
                }
                if ui.button("Reset").clicked() {
                    self.reset();
                }
                ui.add(egui::Slider::new(&mut self.steps_per_frame, 1..=5000).text("steps/frame"));
                ui.add(egui::Slider::new(&mut self.frame_stride, 1..=128).text("sample stride"));
                ui.add(egui::Slider::new(&mut self.trail_capacity, 100..=10_000).text("trail"));
            });
            ui.horizontal(|ui| {
                ui.label(format!("step = {} / {}", self.total_steps, STEPS));
                ui.label(format!(
                    "x = {:.4}, y = {:.4}",
                    self.state.position.x, self.state.position.y
                ));
                ui.label(format!("phi = {:.3}", self.state.angle));
            });
        });

        if self.running {
            self.advance();
            ctx.request_repaint();
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            let desired = ui.available_size();
            let (rect, _resp) = ui.allocate_exact_size(desired, egui::Sense::hover());
            self.draw_scene(ui, rect);
        });
    }
}

fn sample_walls(samples: usize) -> Vec<(Real, Real)> {
    (0..=samples)
        .map(|i| {
            let x = i as Real / samples as Real;
            (x, omega(&x))
        })
        .collect()
}

/// 境界へのめり込みに対する反発力
fn repulsion(point: &Point2<Real>) -> Vector2<Real> {
    const K: Real = 1.5e6; // 反発力の強さの定数 5.0×10^6

    K * if point.y > omega(&point.x) {
        perpendicular_foot::<Ceiling>(point) - point
    } else if point.y < -omega(&point.x) {
        perpendicular_foot::<Floor>(point) - point
    } else {
        Vector2::zeros()
    }
}

/// pointから境界への垂線の足を求める関数
fn perpendicular_foot<W: Wall>(point: &Point2<Real>) -> Point2<Real> {
    std::iter::successors(Some(point.x), |x| {
        let h = W::SIGN * omega(x) - point.y;
        let p = W::SIGN * omega_derivative(x);

        // 点(x_0,y_0)から曲線y=f(x)への垂線の足を求める方程式: (x-x_0, f(x)-y_0)^t ・ (1, f'(x))^t = 0
        // ニュートン法の更新式: x_{n+1} = x_n - (x_n - x_0 + f'(x_n)*(f(x_n) - y_0)) / (1 + f'(x_n)^2 - f"(x_n)*(f(x_n) - y_0))
        let d = (x - point.x + p * h) / (1.0 + p * p - W::SIGN * omega_derivative_second(x) * h);

        (d.abs() > Real::EPSILON).then_some(x - d)
    })
    .last()
    .map(|x| Point2::new(x, omega(&x)))
    .unwrap()
}

/// omega(x) = sin(2πx) + 0.25sin(4πx)+ 1.12 = sin(2πx) + 0.5sin(2πx)cos(2πx) + 1.12
#[inline]
fn omega(x: &Real) -> Real {
    let (s, c) = (TAU * x).sin_cos();
    s + 0.5 * s * c + 1.12
}

/// omega'(x) = 2πcos(2πx) + πcos(4πx) = 2πcos(2πx) + π(2cos^2(2πx) - 1) = 2π cos(2πx)(cos(2πx) + 1) - π
#[inline]
fn omega_derivative(x: &Real) -> Real {
    let c = (TAU * x).cos();
    TAU * c * (c + 1.0) - PI
}

/// omega"(x) = - 4π^2sin(2πx) - 4π^2sin(4πx) = - 4π^2sin(2πx) - 8π^2sin(2πx)cos(2πx) = - 4π^2sin(2πx)(1 + 2cos(2πx))
#[inline]
fn omega_derivative_second(x: &Real) -> Real {
    let (s, c) = (TAU * x).sin_cos();
    -4.0 * PI_SQUARED * s * (1.0 + 2.0 * c)
}
