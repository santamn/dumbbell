//! シミュレーションのアニメーション表示。
//!
//! 1粒子のブラウン運動をリアルタイムに描画する。シード値・外力 f・C1・C2・棒の長さ l は
//! GUIから変更でき、f/C1/C2/l は実行中の粒子に即座に反映される(シードはReset時に反映)。
//! カメラは x 方向にのみ粒子を追いかける。ブラウン運動の細かな揺れに画面が
//! 追従するとガタつくため、デッドゾーン+一次遅れ(指数平滑)で滑らかに追従させる。
//!
//! ※ egui の標準フォントは日本語グリフを含まないため、UIのラベルは英語表記とする。

use crate::config::Config;
use crate::simulation::{ModelParams, Particle, State, omega};
use anyhow::Result;
use eframe::egui::{
    self, CentralPanel, Color32, Context, DragValue, Panel, Pos2, Rect, Sense, Shape, Slider,
    Stroke, Ui,
};
use nalgebra::Point2;
use rand::{SeedableRng, rngs::SmallRng};
use std::f64::consts::TAU;

/// 表示を保証する x 方向の最小範囲(カメラ中心 ± この値)。ズーム倍率の決定に使い、
/// ウィンドウが横長の場合は実際の可視範囲はこれより広くなる
const VIEW_HALF_WIDTH: f64 = 1.5;
/// 表示する y 方向の範囲(チャネルの最大幅 max ω ≈ 2.221 を覆う)
const Y_MAX: f64 = 2.3;
/// カメラのデッドゾーン: 粒子が画面中央から±この距離に収まっている間はカメラを動かさない
const CAMERA_DEAD_ZONE: f64 = 0.6;
/// カメラ追従の時定数 [秒]。大きいほどゆったり追いかける
const CAMERA_TIME_CONSTANT: f64 = 0.5;
/// 境界線を折れ線近似するときの x 方向サンプリング間隔
const BOUNDARY_SAMPLING_STRIDE: f64 = 0.002;

/// アニメーションウィンドウを開き、閉じられるまでブロックする
pub fn run_animation(config: &Config) -> Result<()> {
    // 初期パラメータは設定ファイルの各リストの先頭値を使う(GUIで変更可能)
    let params = ModelParams {
        delta_t: config.delta_t,
        spring_k: config.k,
        length: config.l[0],
        force_x: config.f[0],
        c_1: config.c_1[0],
        c_2: config.c_2[0],
    };
    let app = SimApp::new(0, params, config.steps());

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1100.0, 680.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Dumbbell Brownian Motion Viewer",
        options,
        Box::new(|_| Ok(Box::new(app))),
    )
    .map_err(|e| anyhow::anyhow!("アニメーションの起動に失敗: {e}"))
}

/// シミュレーションの可視化を管理するアプリケーション構造体
struct SimApp {
    particle: Particle<SmallRng>, // シミュレーション本体
    initial_state: State,         // 初期状態(ゴースト表示と情報表示に使う)
    seed: u64,                    // GUIで編集中のシード値(Resetで反映)
    total_steps: usize,           // 総ステップ数 T/Δt
    current_step: usize,          // 現在のステップ数
    steps_per_frame: usize,       // 1フレームに進めるステップ数
    running: bool,                // アニメーションが進行中かどうか
    camera_x: f64,                // カメラ中心の x 座標
    trail: Vec<Point2<f64>>,      // 重心の軌跡(ワールド座標)
}

impl SimApp {
    fn new(seed: u64, params: ModelParams, total_steps: usize) -> Self {
        let particle = Particle::new(SmallRng::seed_from_u64(seed), params);
        let initial_state = particle.state();
        Self {
            particle,
            initial_state,
            seed,
            total_steps,
            current_step: 0,
            steps_per_frame: 1_000,
            running: true,
            camera_x: initial_state.position.x,
            trail: vec![initial_state.position],
        }
    }

    /// 現在のシード値とパラメータでシミュレーションを最初からやり直す
    fn reset(&mut self) {
        let params = self.particle.params;
        self.particle = Particle::new(SmallRng::seed_from_u64(self.seed), params);
        self.initial_state = self.particle.state();
        self.current_step = 0;
        self.running = true;
        self.camera_x = self.initial_state.position.x;
        self.trail = vec![self.initial_state.position];
    }

    /// カメラを粒子に追従させる。デッドゾーン内では動かさず、
    /// はみ出した分だけを一次遅れで滑らかに詰めることで画面のガタつきを防ぐ
    fn follow_particle(&mut self, frame_dt: f64) {
        let offset = self.particle.state().position.x - self.camera_x;
        let overshoot = offset.abs() - CAMERA_DEAD_ZONE;
        if overshoot > 0.0 {
            let alpha = 1.0 - (-frame_dt / CAMERA_TIME_CONSTANT).exp();
            self.camera_x += overshoot * offset.signum() * alpha;
        }
    }

    /// ワールド座標から画面座標への変換を表すクロージャ(y軸は上向きが正)と、
    /// 画面の左右端まで届く x 方向の可視半幅(ワールド座標)を返す
    fn world_to_screen(&self, rect: Rect) -> (impl Fn(Point2<f64>) -> Pos2, f64) {
        let scale = (rect.width() / (2.0 * VIEW_HALF_WIDTH) as f32)
            .min(rect.height() / (2.0 * Y_MAX) as f32);
        let visible_half_width = (0.5 * rect.width() / scale) as f64;
        let center = rect.center();
        let camera_x = self.camera_x;
        (
            move |p: Point2<f64>| center + egui::vec2((p.x - camera_x) as f32, -p.y as f32) * scale,
            visible_half_width,
        )
    }
}

impl eframe::App for SimApp {
    /// 毎フレームのロジック更新(描画前に呼ばれる)。
    /// シミュレーションの前進とカメラ追従を行い、描画は `ui` に任せる
    fn logic(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
        if self.running {
            // シミュレーションを進める
            let n = self
                .steps_per_frame
                .min(self.total_steps - self.current_step);
            self.particle.advance(n);
            self.current_step += n;
            self.trail.push(self.particle.state().position);
            if self.current_step >= self.total_steps {
                self.running = false;
            }

            // カメラを粒子に追従させる(フレーム時間で平滑化)
            let frame_dt = ctx.input(|i| i.stable_dt).min(0.1) as f64;
            self.follow_particle(frame_dt);

            ctx.request_repaint();
        }
    }

    /// 毎フレームのUI更新。GUI操作の反映と描画を行う
    fn ui(&mut self, ui: &mut Ui, _frame: &mut eframe::Frame) {
        // 操作パネル(上部)
        Panel::top("controls").show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui
                    .button(if self.running { "Pause" } else { "Resume" })
                    .clicked()
                {
                    self.running = !self.running;
                }
                if ui.button("Reset").clicked() {
                    self.reset();
                }
                ui.separator();

                // シードはResetを押したときに反映される
                ui.label("seed:");
                ui.add(DragValue::new(&mut self.seed).speed(1));
                ui.separator();

                // f, C1, C2 は実行中の粒子に即座に反映される
                let params = &mut self.particle.params;
                ui.label("f:");
                ui.add(DragValue::new(&mut params.force_x).speed(0.5));
                ui.label("C1(=βEp/l):");
                ui.add(DragValue::new(&mut params.c_1).speed(0.1));
                ui.label("C2(=ΔαE/p):");
                ui.add(DragValue::new(&mut params.c_2).speed(0.01));
                ui.label("l:");
                ui.add(
                    DragValue::new(&mut params.length)
                        .speed(0.005)
                        .range(0.001..=1.0), // 0除算を防ぐため正の値に制限
                );
                ui.separator();

                ui.add(
                    Slider::new(&mut self.steps_per_frame, 1..=100_000)
                        .logarithmic(true)
                        .text("steps/frame"),
                );
            });

            // 初期状態と現在の状態の表示
            ui.horizontal(|ui| {
                let now = self.particle.state();
                ui.label(format!(
                    "step {} / {}  (t = {:.4})",
                    self.current_step,
                    self.total_steps,
                    self.current_step as f64 * self.particle.params.delta_t,
                ));
                ui.separator();
                ui.label(format!(
                    "initial: x = {:.4}, y = {:.4}, Φ = {:.4}",
                    self.initial_state.position.x,
                    self.initial_state.position.y,
                    self.initial_state.angle.rem_euclid(TAU),
                ));
                ui.separator();
                ui.label(format!(
                    "current: x = {:.4}, y = {:.4}, Φ = {:.4}",
                    now.position.x,
                    now.position.y,
                    now.angle.rem_euclid(TAU),
                ));
            });
        });

        // 描画パネル(中央)
        CentralPanel::default().show(ui, |ui| {
            let (rect, _) = ui.allocate_exact_size(ui.available_size(), Sense::empty());
            let painter = ui.painter_at(rect);
            let (to_screen, visible_half_width) = self.world_to_screen(rect);

            // チャネル境界の描画(画面の左右いっぱいまで。カメラが動くため毎フレーム可視範囲を計算する)
            let samples = (2.0 * visible_half_width / BOUNDARY_SAMPLING_STRIDE) as usize;
            let (upper, lower): (Vec<Pos2>, Vec<Pos2>) = (0..=samples)
                .map(|i| {
                    let x =
                        self.camera_x - visible_half_width + i as f64 * BOUNDARY_SAMPLING_STRIDE;
                    let y = omega(x);
                    (to_screen(Point2::new(x, y)), to_screen(Point2::new(x, -y)))
                })
                .unzip();
            let wall_stroke = Stroke::new(1.4, Color32::from_rgb(120, 180, 220));
            painter.add(Shape::line(upper, wall_stroke));
            painter.add(Shape::line(lower, wall_stroke));

            // 軌跡の描画
            let trail_points: Vec<Pos2> = self.trail.iter().map(|&p| to_screen(p)).collect();
            painter.add(Shape::line(
                trail_points,
                Stroke::new(1.0, Color32::from_rgb(255, 170, 90)),
            ));

            // 初期状態のゴースト表示(半透明の棒)
            let (init_plus, init_minus) = self.initial_state.endpoints(self.particle.params.length);
            painter.line_segment(
                [to_screen(init_plus), to_screen(init_minus)],
                Stroke::new(3.0, Color32::from_rgba_unmultiplied(160, 160, 160, 120)),
            );
            painter.circle_filled(
                to_screen(self.initial_state.position),
                1.5,
                Color32::from_rgba_unmultiplied(220, 220, 220, 120),
            );

            // 現在の粒子の描画(棒と重心)
            let (p_plus, p_minus) = self.particle.endpoints();
            painter.line_segment(
                [to_screen(p_plus), to_screen(p_minus)],
                Stroke::new(3.0, Color32::from_rgb(230, 90, 60)),
            );
            painter.circle_filled(
                to_screen(self.particle.state().position),
                1.5,
                Color32::WHITE,
            );
        });
    }
}
