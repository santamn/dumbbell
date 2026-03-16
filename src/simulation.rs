use nalgebra::{Point2, Vector2};
use rand::{Rng, RngExt};
use rand_distr::StandardNormal;
use std::f64::consts::{PI, TAU};
use std::ops::Add;

pub type Real = f64;

/// 定数文脈で平方根を計算するためのニュートン法による近似関数
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

const PI_SQUARED: Real = PI * PI;

pub const TIME: Real = 10.0; //                        総シミュレーションの時間       10.0
pub const DELTA_T: Real = 2e-7; //                     シミュレーションの時間刻み     2.0×10^-7
pub const NOISE_SCALE: Real = const_sqrt(DELTA_T); //  ブラウン運動のノイズのスケール
pub const STEPS: usize = (TIME / DELTA_T) as usize; // シミュレーションの総ステップ数  5.0×10^7

#[derive(Debug, Clone, Copy)]
/// 粒子の状態を表す構造体
pub struct State {
    pub position: Point2<Real>,
    pub angle: Real,
}

impl State {
    pub fn new<R: Rng>(rng: &mut R, length: Real) -> Self {
        let x = rng.random_range(-0.1..0.7);
        let limit = omega(x) - length * 0.5;
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
    const SIGN: Real; // 壁の上下を表す定数（上壁: 1, 下壁: -1）
}

struct Ceiling;
impl Wall for Ceiling {
    const SIGN: Real = 1.0;
}

struct Floor;
impl Wall for Floor {
    const SIGN: Real = -1.0;
}

/// 粒子の軌跡を逐次生成するイテレータ
#[derive(Debug, Clone)]
pub struct Trajectory<R: Rng> {
    rng: R,
    pub length: Real,
    pub force: Vector2<Real>,
    state: Option<State>,
}

impl<R: Rng> Trajectory<R> {
    pub fn new(rng: R, length: Real, force: Vector2<Real>) -> Self {
        Self {
            rng,
            length,
            force,
            state: None,
        }
    }
}

impl<R: Rng> Iterator for Trajectory<R> {
    type Item = State;

    fn next(&mut self) -> Option<Self::Item> {
        self.state = Some(match self.state {
            None => State::new(&mut self.rng, self.length),
            Some(state) => {
                let xi_x = self.rng.sample(StandardNormal);
                let xi_y = self.rng.sample(StandardNormal);
                let xi_phi = self.rng.sample::<Real, _>(StandardNormal);

                let (s, c) = state.angle.sin_cos();
                let e_phi = Vector2::new(-s, c);
                let h = 0.5 * self.length * Vector2::new(c, s);
                let (p1, p2) = (state.position + h, state.position - h);
                let (rep1, rep2) = (repulsion(&p1), repulsion(&p2));

                state
                    + (
                        (self.force + 0.5 * (rep1 + rep2)) * DELTA_T
                            + Vector2::new(xi_x, xi_y) * NOISE_SCALE,
                        ((rep1 - rep2).dot(&e_phi) * DELTA_T + 2.0 * xi_phi * NOISE_SCALE)
                            / self.length,
                    )
            }
        });

        self.state
    }
}

/// 壁への沈み込みに対する反発力
fn repulsion(point: &Point2<Real>) -> Vector2<Real> {
    const K: Real = 1.5e6;

    K * if point.y > omega(point.x) {
        perpendicular_foot::<Ceiling>(point) - point
    } else if point.y < -omega(point.x) {
        perpendicular_foot::<Floor>(point) - point
    } else {
        Vector2::zeros()
    }
}

/// 点から壁への垂線の足を求める関数
fn perpendicular_foot<W: Wall>(point: &Point2<Real>) -> Point2<Real> {
    std::iter::successors(Some(point.x), |&x| {
        let h = W::SIGN * omega(x) - point.y;
        let p = W::SIGN * omega_derivative(x);

        // 点 (x_0, y_0) から境界上の点 (x, f(x)) に降ろした垂線が満たす方程式: (x - x_0) + f'(x) * (f(x) - y_0) = 0
        // ニュートン法の更新式: x_next = x - (x - x_0 + f'(x) * (f(x) - y_0)) / (1 + f'(x)^2 - f''(x) * (f(x) - y_0))
        let d = (x - point.x + p * h) / (1.0 + p * p - W::SIGN * omega_derivative_second(x) * h);

        (d.abs() > Real::EPSILON).then_some(x - d)
    })
    .take(1000) // 1000回の更新で収束しない場合は諦める
    .last()
    .map(|x| Point2::new(x, W::SIGN * omega(x)))
    .unwrap_or(*point)
}

#[inline]
/// omega(x) = sin(2πx) + 0.25sin(4πx) + 1.12 = sin(2πx) + 0.5sin(2πx)cos(2πx) + 1.12
pub fn omega(x: Real) -> Real {
    let (s, c) = (TAU * x).sin_cos();
    s + 0.5 * s * c + 1.12
}

#[inline]
/// omega'(x) = 2πcos(2πx) + πcos(4πx) = 2πcos(2πx)(cos(2πx) + 1) - π
fn omega_derivative(x: Real) -> Real {
    let c = (TAU * x).cos();
    TAU * c * (c + 1.0) - PI
}

#[inline]
/// omega"(x) = -4π^2sin(2πx) - 4π^2sin(4πx) = -4π^2sin(2πx)(1 + 2cos(2πx))
fn omega_derivative_second(x: Real) -> Real {
    let (s, c) = (TAU * x).sin_cos();
    -4.0 * PI_SQUARED * s * (1.0 + 2.0 * c)
}
