use nalgebra::{Point2, Vector2};
use rand::{Rng, RngExt};
use rand_distr::StandardNormal;
use std::f64::consts::{PI, TAU};
use std::ops::Add;

/// 定数文脈で平方根を計算するためのニュートン法による近似関数
const fn const_sqrt(x: f64) -> f64 {
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

pub const DELTA_T: f64 = 2e-7; //                      シミュレーションの時間刻み
pub const K: f64 = 1.5e6; //                           壁の反発力の強さ
pub const NOISE_SCALE: f64 = const_sqrt(DELTA_T); //   ブラウン運動のノイズのスケール
pub const TIME: f64 = 10.0; //                         総シミュレーションの時間
pub const STEPS: usize = (TIME / DELTA_T) as usize; // シミュレーションの総ステップ数

// チャネルの境界を表すトレイトと、その実装としての天井と床の構造体
trait Wall {
    const SIGN: f64; // 壁の上下を表す定数（上壁: 1, 下壁: -1）
}

struct Ceiling;
impl Wall for Ceiling {
    const SIGN: f64 = 1.0;
}

struct Floor;
impl Wall for Floor {
    const SIGN: f64 = -1.0;
}

/// 粒子の状態を表す構造体
#[derive(Debug, Clone, Copy)]
pub struct State {
    pub position: Point2<f64>,
    pub angle: f64,
}

impl State {
    pub fn new<R: Rng>(rng: &mut R, length: f64) -> Self {
        let x = rng.random_range(-0.1..0.7);
        let limit = omega(x) - length * 0.5;
        Self {
            position: Point2::new(x, rng.random_range(-limit..limit)),
            angle: rng.random_range(0.0..TAU),
        }
    }
}

impl Add<(Vector2<f64>, f64)> for State {
    type Output = Self;

    fn add(self, other: (Vector2<f64>, f64)) -> Self {
        Self {
            position: self.position + other.0,
            angle: self.angle + other.1,
        }
    }
}

/// 粒子を表す構造体。内部で乱数生成器を保持し、イテレータとして粒子の軌跡を逐次生成する
#[derive(Debug)]
pub struct Particle<R: Rng> {
    rng: R,
    is_first: bool,
    length: f64,
    force: Vector2<f64>,
    state: State,
}

impl<R: Rng> Particle<R> {
    pub fn new(mut rng: R, length: f64, force: Vector2<f64>) -> Self {
        let state = State::new(&mut rng, length);
        Self {
            rng,
            is_first: true,
            length,
            force,
            state,
        }
    }

    pub fn endpoints(&self) -> (Point2<f64>, Point2<f64>) {
        let (s, c) = self.state.angle.sin_cos();
        let h = 0.5 * self.length * Vector2::new(c, s);
        (self.state.position + h, self.state.position - h)
    }

    pub fn force(&self) -> Vector2<f64> {
        self.force
    }

    pub fn now(&self) -> State {
        self.state
    }
}

impl<R: Rng> Iterator for Particle<R> {
    type Item = State;

    fn next(&mut self) -> Option<Self::Item> {
        if self.is_first {
            self.is_first = false;
            return Some(self.state);
        }

        let xi_x = self.rng.sample(StandardNormal);
        let xi_y = self.rng.sample(StandardNormal);
        let xi_phi = self.rng.sample::<f64, _>(StandardNormal);

        let (s, c) = self.state.angle.sin_cos();
        let h = 0.5 * self.length * Vector2::new(c, s);
        let e_phi = Vector2::new(-s, c);
        let (p1, p2) = (self.state.position + h, self.state.position - h);
        let (f1, f2) = (repulsion(&p1), repulsion(&p2));

        self.state = self.state
            + (
                (self.force + 0.5 * (f1 + f2)) * DELTA_T + Vector2::new(xi_x, xi_y) * NOISE_SCALE,
                ((f1 - f2).dot(&e_phi) * DELTA_T + 2.0 * xi_phi * NOISE_SCALE) / self.length,
            );

        Some(self.state)
    }
}

/// 壁への沈み込みに対する反発力
fn repulsion(point: &Point2<f64>) -> Vector2<f64> {
    K * if point.y > omega(point.x) {
        perpendicular_foot::<Ceiling>(point) - point
    } else if point.y < -omega(point.x) {
        perpendicular_foot::<Floor>(point) - point
    } else {
        Vector2::zeros()
    }
}

/// 点から壁への垂線の足を求める関数
fn perpendicular_foot<W: Wall>(point: &Point2<f64>) -> Point2<f64> {
    std::iter::successors(Some(point.x), |&x| {
        let h = W::SIGN * omega(x) - point.y;
        let p = W::SIGN * omega_derivative(x);

        // 点 (x_0, y_0) から境界上の点 (x, f(x)) に降ろした垂線が満たす方程式: (x - x_0) + f'(x) * (f(x) - y_0) = 0
        // ニュートン法の更新式: x_next = x - (x - x_0 + f'(x) * (f(x) - y_0)) / (1 + f'(x)^2 - f''(x) * (f(x) - y_0))
        let d = (x - point.x + p * h) / (1.0 + p * p - W::SIGN * omega_derivative_second(x) * h);

        (d.abs() > 1e-10).then_some(x - d)
    })
    .take(32) // 132回の更新で収束しない場合は諦める
    .last()
    .map(|x| Point2::new(x, W::SIGN * omega(x)))
    .unwrap_or(*point)
}

#[inline]
/// omega(x) = sin(2πx) + 0.25sin(4πx) + 1.12 = sin(2πx) + 0.5sin(2πx)cos(2πx) + 1.12
pub fn omega(x: f64) -> f64 {
    let (s, c) = (TAU * x).sin_cos();
    s + 0.5 * s * c + 1.12
}

#[inline]
/// omega'(x) = 2πcos(2πx) + πcos(4πx) = 2πcos(2πx)(cos(2πx) + 1) - π
fn omega_derivative(x: f64) -> f64 {
    let c = (TAU * x).cos();
    TAU * c * (c + 1.0) - PI
}

#[inline]
/// omega"(x) = -4π^2sin(2πx) - 4π^2sin(4πx) = -(2π)^2sin(2πx)(1 + 2cos(2πx))
fn omega_derivative_second(x: f64) -> f64 {
    let (s, c) = (TAU * x).sin_cos();
    -TAU * TAU * s * (1.0 + 2.0 * c)
}
