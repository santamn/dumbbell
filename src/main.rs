use nalgebra::{Point2, Vector2};
use rand::{Rng, RngExt, SeedableRng, rngs::SmallRng};
use rand_distr::StandardNormal;
use std::f64::consts::{PI, TAU};

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
struct DumbbellParticle {
    length: Real,
    position: Point2<Real>,
    angle: Real,
}

impl DumbbellParticle {
    fn new<R>(rng: &mut R, length: Real) -> Self
    where
        R: Rng + ?Sized,
    {
        let x = rng.random_range(0.0..1.0);
        let limit = omega(&x) - length * 0.5;
        Self {
            position: Point2::new(x, rng.random_range(-limit..limit)),
            angle: rng.random_range(0.0..TAU),
            length,
        }
    }

    /// 粒子の向きが変化する方向への単位ベクトル
    fn e_phi(&self) -> Vector2<Real> {
        let (s, c) = self.angle.sin_cos();
        Vector2::new(-s, c)
    }

    /// 粒子の両端の座標を求める関数
    fn endpoints(&self) -> (Point2<Real>, Point2<Real>) {
        let (s, c) = self.angle.sin_cos();
        let half_length_vector = 0.5 * self.length * Vector2::new(c, s);
        (
            self.position + half_length_vector,
            self.position - half_length_vector,
        )
    }

    /// 粒子の位置と角度を更新する関数
    fn advance(&mut self, translation: Vector2<Real>, rotation: Real) {
        self.position += translation;
        self.angle += rotation;
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
    let particle_length = 0.02;
    let results = simulate_brownian_motion(1, Vector2::new(1.0, 0.0), particle_length);
    for (position, angle) in results {
        println!("{:.6} {:.6} {:.6}", position.x, position.y, angle);
    }
}

fn simulate_brownian_motion(
    seed: u64,
    force: Vector2<Real>,
    length: Real,
) -> Vec<(Point2<Real>, Real)> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut particle = DumbbellParticle::new(&mut rng, length);

    let mut states = Vec::with_capacity(STEPS);
    states.push((particle.position, particle.angle));

    for _ in 0..(STEPS - 1) {
        let xi_x = rng.sample(StandardNormal);
        let xi_y = rng.sample(StandardNormal);
        let xi_phi: Real = rng.sample(StandardNormal);

        let (p1, p2) = particle.endpoints();
        let f1 = force + repulsion(&p1);
        let f2 = force + repulsion(&p2);

        particle.advance(
            0.5 * (f1 + f2) * DELTA_T + Vector2::new(xi_x, xi_y) * NOISE_SCALE,
            ((f1 - f2).dot(&particle.e_phi()) * DELTA_T + 2.0 * xi_phi * NOISE_SCALE) / length,
        );
        states.push((particle.position, particle.angle));
    }

    states
}

/// 境界へのめり込みに対する反発力
fn repulsion(point: &Point2<Real>) -> Vector2<Real> {
    const K: Real = 5.0e6; // 反発力の強さの定数 5.0×10^6

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
