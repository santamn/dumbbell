//! ダンベル型粒子のブラウン運動モデル(CPU実装)。
//!
//! docs/model.md の無次元化された確率微分方程式を予測子・修正子法で数値積分する。
//! GPU実装(simulation.cu)と同一の物理モデルであり、
//! 物理モデルを変更する場合は必ず両者を同期させること。

use nalgebra::{Point2, Vector2};
use rand::{Rng, RngExt};
use rand_distr::StandardNormal;
use std::f64::consts::{PI, TAU};

/// 壁の垂線の足を求めるニュートン法の反復回数(model.md で規定)
const NEWTON_ITERATIONS: usize = 5;

/// チャネルの境界を表すトレイトと、その実装としての天井と床の構造体
trait Wall {
    const SIGN: f64; // 壁の上下を表す定数(上壁: 1, 下壁: -1)
}

struct Ceiling;
impl Wall for Ceiling {
    const SIGN: f64 = 1.0;
}

struct Floor;
impl Wall for Floor {
    const SIGN: f64 = -1.0;
}

/// モデルの無次元パラメータ一式
#[derive(Debug, Clone, Copy)]
pub struct ModelParams {
    /// 時間刻み幅 Δt
    pub delta_t: f64,
    /// 壁のばね定数 K
    pub spring_k: f64,
    /// 棒の長さ l
    pub length: f64,
    /// 一定外力 f(x方向)
    pub force_x: f64,
    /// 電場パラメータ C1 = βEp/l̃
    pub c_1: f64,
    /// 電場パラメータ C2 = ΔαE/p
    pub c_2: f64,
}

/// 粒子の状態(重心座標と棒の角度)
#[derive(Debug, Clone, Copy)]
pub struct State {
    pub position: Point2<f64>,
    pub angle: f64,
}

impl State {
    /// 初期状態を乱数で生成する。x は 1 周期弱の範囲、y はチャネル内部、角度は一様分布
    pub fn new<R: Rng>(rng: &mut R, length: f64) -> Self {
        let x = rng.random_range(-0.1..0.7);
        let limit = omega(x) - length * 0.5;
        Self {
            position: Point2::new(x, rng.random_range(-limit..limit)),
            angle: rng.random_range(0.0..TAU),
        }
    }

    /// この状態における棒の両端の座標 (X+, X-) を返す
    pub fn endpoints(&self, length: f64) -> (Point2<f64>, Point2<f64>) {
        let (s, c) = self.angle.sin_cos();
        let half = 0.5 * length * Vector2::new(c, s);
        (self.position + half, self.position - half)
    }
}

/// 1個のダンベル型粒子。乱数生成器を内部に保持し、`step` で1ステップずつ時間発展する
#[derive(Debug)]
pub struct Particle<R: Rng> {
    rng: R,
    /// モデルパラメータ。アニメーションからの動的な変更を許すため公開している
    pub params: ModelParams,
    state: State,
}

impl<R: Rng> Particle<R> {
    pub fn new(mut rng: R, params: ModelParams) -> Self {
        let state = State::new(&mut rng, params.length);
        Self { rng, params, state }
    }

    pub fn state(&self) -> State {
        self.state
    }

    /// 棒の両端の座標 (X+, X-) を返す
    pub fn endpoints(&self) -> (Point2<f64>, Point2<f64>) {
        self.state.endpoints(self.params.length)
    }

    /// 予測子・修正子法で1ステップ時間発展させる
    pub fn step(&mut self) {
        let ModelParams { delta_t, .. } = self.params;
        let noise_scale = delta_t.sqrt(); // Wiener過程の増分の標準偏差 √Δt

        // Wiener過程の増分(標準正規乱数 × √Δt)。予測子・修正子で共通して使う
        let xi_x: f64 = self.rng.sample(StandardNormal);
        let xi_y: f64 = self.rng.sample(StandardNormal);
        let xi_phi: f64 = self.rng.sample(StandardNormal);
        let noise_position = Vector2::new(xi_x, xi_y) * noise_scale;
        let noise_angle = 2.0 * xi_phi * noise_scale / self.params.length;

        let anchor = self.state;

        // 予測子: Euler–Maruyama 法の1ステップ
        let (drift_position, drift_angle) = drift(&anchor, &self.params);
        let predicted = State {
            position: anchor.position + drift_position * delta_t + noise_position,
            angle: anchor.angle + drift_angle * delta_t + noise_angle,
        };

        // 修正子: ドリフトを評価し直し、予測子に適用する
        let (drift_position, drift_angle) = drift(&predicted, &self.params);
        self.state = State {
            position: anchor.position + drift_position * delta_t + noise_position,
            angle: anchor.angle + drift_angle * delta_t + noise_angle,
        };
    }

    /// nステップまとめて時間発展させる
    pub fn advance(&mut self, n: usize) {
        for _ in 0..n {
            self.step();
        }
    }
}

/// 指定した状態におけるドリフト項(重心・角度それぞれの力・トルクによる決定論的な変化率)を返す。
/// 予測子・修正子法では始状態と予測状態の両方でこの関数を呼び、ドリフトを評価し直す
fn drift(state: &State, params: &ModelParams) -> (Vector2<f64>, f64) {
    let ModelParams {
        spring_k,
        length,
        force_x,
        c_1,
        c_2,
        ..
    } = *params;
    let (s, c) = state.angle.sin_cos();

    // 棒の両端に働く壁からの反発力
    let (p_plus, p_minus) = state.endpoints(length);
    let (f_plus, f_minus) = (repulsion(spring_k, &p_plus), repulsion(spring_k, &p_minus));

    // 電場が電気双極子に及ぼすトルク: 2 C1 cosΦ (1 + C2 sinΦ)
    let torque = 2.0 * c_1 * c * (1.0 + c_2 * s);
    // 外積 n × (f+ − f−)
    let cross = (f_plus - f_minus).dot(&Vector2::new(-s, c));

    let drift_position = Vector2::new(force_x, 0.0) + 0.5 * (f_plus + f_minus);
    let drift_angle = (torque + cross) / length;
    (drift_position, drift_angle)
}

/// 壁への沈み込みに対する反発力(ペナルティ法): f = K((x*, y*) - (px, py))
fn repulsion(spring_k: f64, point: &Point2<f64>) -> Vector2<f64> {
    spring_k
        * if point.y > omega(point.x) {
            perpendicular_foot::<Ceiling>(point) - point
        } else if point.y < -omega(point.x) {
            perpendicular_foot::<Floor>(point) - point
        } else {
            Vector2::zeros() // チャネル内部なら力は働かない
        }
}

/// 点から壁 y = ±ω(x) へ下ろした垂線の足をニュートン法で求める。
/// g(x) = (x - px) + φ'(x)(φ(x) - py) = 0 を初期値 x0 = px から固定回数反復して解く
fn perpendicular_foot<W: Wall>(point: &Point2<f64>) -> Point2<f64> {
    let mut x = point.x;
    for _ in 0..NEWTON_ITERATIONS {
        let diff = W::SIGN * omega(x) - point.y; // φ(x) - py
        let phi_p = W::SIGN * omega_derivative(x); // φ'(x)
        let phi_pp = W::SIGN * omega_derivative_second(x); // φ''(x)

        // ニュートン法の更新式: x ← x - g(x) / g'(x)
        x -= (x - point.x + phi_p * diff) / (1.0 + phi_p * phi_p + phi_pp * diff);
    }
    Point2::new(x, W::SIGN * omega(x))
}

/// チャネル上壁の形状 ω(x) = sin(2πx) + 0.25sin(4πx) + 1.12 = sin(2πx)(0.5cos(2πx) + 1) + 1.12
#[inline]
pub fn omega(x: f64) -> f64 {
    let (s, c) = (TAU * x).sin_cos();
    s + 0.5 * s * c + 1.12
}

/// ω'(x) = 2πcos(2πx) + πcos(4πx) = 2πcos(2πx)(cos(2πx) + 1) - π
#[inline]
fn omega_derivative(x: f64) -> f64 {
    let c = (TAU * x).cos();
    TAU * c * (c + 1.0) - PI
}

/// ω''(x) = -4π²sin(2πx) - 4π²sin(4πx) = -(2π)²sin(2πx)(1 + 2cos(2πx))
#[inline]
fn omega_derivative_second(x: f64) -> f64 {
    let (s, c) = (TAU * x).sin_cos();
    -TAU * TAU * s * (1.0 + 2.0 * c)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// チャネル内部の点には反発力が働かないこと
    #[test]
    fn no_repulsion_inside_channel() {
        let force = repulsion(1.5e6, &Point2::new(0.25, 0.0));
        assert_eq!(force, Vector2::zeros());
    }

    /// ニュートン法で求めた垂線の足が停留条件 g(x*) = 0 をほぼ満たすこと
    #[test]
    fn perpendicular_foot_satisfies_stationarity() {
        let point = Point2::new(0.3, omega(0.3) + 0.01); // 上壁を少し越えた点
        let foot = perpendicular_foot::<Ceiling>(&point);
        let g = (foot.x - point.x) + omega_derivative(foot.x) * (omega(foot.x) - point.y);
        assert!(g.abs() < 1e-10, "g(x*) = {g}");
    }

    /// 反発力がめり込みを解消する向き(壁の内側向き)であること
    #[test]
    fn repulsion_pushes_back_into_channel() {
        let above = Point2::new(0.3, omega(0.3) + 0.01);
        assert!(repulsion(1.5e6, &above).y < 0.0);

        let below = Point2::new(0.3, -omega(0.3) - 0.01);
        assert!(repulsion(1.5e6, &below).y > 0.0);
    }

    /// 電場トルクの平衡点: Φ = π/2(双極子が電場と平行)でトルクが消えること
    #[test]
    fn electric_torque_vanishes_at_equilibrium() {
        let torque = |phi: f64, c_1: f64, c_2: f64| {
            let (s, c) = phi.sin_cos();
            2.0 * c_1 * c * (1.0 + c_2 * s)
        };
        assert!(torque(std::f64::consts::FRAC_PI_2, 3.0, 0.5).abs() < 1e-12);
        // Φ = 0 では正のトルク(Φ = π/2 へ引き込む向き)
        assert!(torque(0.0, 3.0, 0.5) > 0.0);
    }
}
