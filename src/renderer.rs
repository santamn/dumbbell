use crate::simulation::{Real, STEPS, Trajectory, omega};
use eframe::egui::{
    self, CentralPanel, Color32, Context, Pos2, Rect, Sense, Stroke, TopBottomPanel,
};
use nalgebra::{Point2, Vector2};
use rand::{SeedableRng, rngs::SmallRng};
use std::cell::OnceCell;
use std::f64::consts::TAU;
use std::ops::Range;

const LOCAL_MINIUM_POINT: Real = -0.190359162688;
const Y_MAX: Real = 2.23;
const Y_MIN: Real = -2.23;
const BOUNDARY_SAMPLING_STRIDE: Real = 0.001;

/// シミュレーションの可視化を管理するアプリケーション構造体
pub struct SimApp {
    boundary: OnceCell<(Vec<Pos2>, Vec<Pos2>)>, // チャネルの境界線
    current_step: usize,                        // 現在のシミュレーション内部ステップ
    sample_stride: usize,                       // 1フレームで進める内部ステップ数
    trail: Vec<Pos2>,                           // 粒子の軌跡を保存するバッファ
    trajectory_iter: Trajectory<SmallRng>,      // 粒子軌跡を逐次生成するイテレータ
    x_range: Range<Real>,                       // 描画するx座標の範囲
}

impl SimApp {
    pub fn new(
        seed: u64,
        sample_stride: usize,
        x_range: Range<Real>,
        rod_length: Real,
        force: Vector2<Real>,
    ) -> Self {
        Self {
            boundary: OnceCell::new(),
            current_step: 0,
            sample_stride,
            trail: Vec::with_capacity(STEPS / sample_stride + 1),
            trajectory_iter: Trajectory::new(SmallRng::seed_from_u64(seed), rod_length, force),
            x_range: (x_range.start - LOCAL_MINIUM_POINT).floor() + LOCAL_MINIUM_POINT
                ..(x_range.end - LOCAL_MINIUM_POINT).ceil() + LOCAL_MINIUM_POINT,
        }
    }

    /// シミュレーション上の座標を表示画面上の座標系に変換する
    fn screen_position(&self, rect: Rect, point: Point2<Real>) -> Pos2 {
        let real_w = (self.x_range.end - self.x_range.start) as f32;
        let real_h = (Y_MAX - Y_MIN) as f32;
        let scale = (rect.width() / real_w).min(rect.height() / real_h);

        let center_x = 0.5 * (self.x_range.start + self.x_range.end) as f32;
        rect.center() + egui::vec2(point.x as f32 - center_x, point.y as f32) * scale
    }
}

impl eframe::App for SimApp {
    /// 毎フレームのUI更新。入力反映と描画を担当する
    fn update(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
        // シミュレーションの状態をsample_stride分だけ進める
        let state = self
            .trajectory_iter
            .nth((self.sample_stride - 1).min(self.current_step))
            .unwrap();

        // 現在のシミュレーションの状態を表示するUIパネルを上部に配置
        TopBottomPanel::top("controls").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(format!("step = {} / {}", self.current_step, STEPS));
                ui.label(format!(
                    "force = ({:.3}, {:.3})",
                    self.trajectory_iter.force.x, self.trajectory_iter.force.y
                ));
                ui.label(format!(
                    "position = ({:.4} {:.4})",
                    state.position.x, state.position.y
                ));
                ui.label(format!("Φ = {:.4} [rad]", state.angle.rem_euclid(TAU)));
            });
        });

        // 粒子の軌跡を描画する中央のパネルを配置
        CentralPanel::default().show(ctx, |ui| {
            let (rect, _) = ui.allocate_exact_size(ui.available_size(), Sense::empty());
            let painter = ui.painter_at(rect);

            // 境界線の座標は一度のみ計算する
            let (upper_boundary, lower_boundary) = self.boundary.get_or_init(|| {
                (0..=((self.x_range.end - self.x_range.start) / BOUNDARY_SAMPLING_STRIDE) as usize)
                    .map(|i| {
                        let x = self.x_range.start + i as Real * BOUNDARY_SAMPLING_STRIDE;
                        let y = omega(x);
                        (
                            self.screen_position(rect, Point2::new(x, y)),
                            self.screen_position(rect, Point2::new(x, -y)),
                        )
                    })
                    .unzip()
            });

            // 上下の境界線の描画
            let stroke_wall = Stroke::new(1.4, Color32::from_rgb(120, 180, 220));
            upper_boundary.windows(2).for_each(|window| {
                painter.line_segment([window[0], window[1]], stroke_wall);
            });
            lower_boundary.windows(2).for_each(|window| {
                painter.line_segment([window[0], window[1]], stroke_wall);
            });

            // 軌跡は履歴バッファを細線で連結して表示する
            self.trail.push(self.screen_position(rect, state.position));
            let stroke_trail = Stroke::new(1.0, Color32::from_rgb(255, 170, 90));
            self.trail.windows(2).for_each(|window| {
                painter.line_segment([window[0], window[1]], stroke_trail);
            });

            // 粒子の描画
            let (s, c) = state.angle.sin_cos();
            let h = 0.5 * self.trajectory_iter.length * Vector2::new(c, s);
            let (p1, p2) = (state.position + h, state.position - h);
            // 棒
            painter.line_segment(
                [
                    self.screen_position(rect, p1),
                    self.screen_position(rect, p2),
                ],
                Stroke::new(3.0, Color32::from_rgb(230, 90, 60)),
            );
            // 粒子の中心
            painter.circle_filled(
                self.screen_position(rect, state.position),
                1.5,
                Color32::WHITE,
            );
        });

        if self.current_step < STEPS {
            self.current_step += self.sample_stride;
            ctx.request_repaint();
        }
    }
}
