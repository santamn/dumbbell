// ============================================================================
// ダンベル型粒子のブラウン運動シミュレーション(GPUカーネル)
//
// docs/model.md の無次元化された確率微分方程式を予測子・修正子法で数値積分し、
// アンサンブル全体の x 方向変位の総和と二乗和を求める。
// CPU実装(simulation.rs)と同一の物理モデルであり、
// 物理モデルを変更する場合は必ず両者を同期させること。
//
// シミュレーションの定数はすべてカーネル引数として実行時に渡される。
// ============================================================================
#include <curand_kernel.h>
#include <math.h>
#include <stdint.h>

// 壁の垂線の足を求めるニュートン法の反復回数(model.md で規定)
constexpr int NEWTON_ITERATIONS = 5;
constexpr double TAU = 2.0 * M_PI;

// チャネル上壁の形状 ω(x) = sin(2πx) + 0.25sin(4πx) + 1.12 = sin(2πx)(0.5cos(2πx) + 1) + 1.12
__device__ double omega(double x)
{
  double s, c;
  sincospi(2.0 * x, &s, &c);
  return s + 0.5 * s * c + 1.12;
}

// ω(x), ω'(x), ω''(x) を1回の三角関数計算でまとめて求める
//   ω'(x)  = 2πcos(2πx) + πcos(4πx)   = 2πcos(2πx)(cos(2πx) + 1) - π
//   ω''(x) = -4π²sin(2πx) - 4π²sin(4πx) = -(2π)²sin(2πx)(1 + 2cos(2πx))
__device__ void omega_with_derivatives(double x, double *w, double *w_p, double *w_pp)
{
  double s, c;
  sincospi(2.0 * x, &s, &c);
  *w = s + 0.5 * s * c + 1.12;
  *w_p = TAU * c * (c + 1.0) - M_PI;
  *w_pp = -TAU * TAU * s * (1.0 + 2.0 * c);
}

// 点 (px, py) から壁 y = sign·ω(x) へ下ろした垂線の足の x 座標をニュートン法で求める。
// g(x) = (x - px) + φ'(x)(φ(x) - py) = 0 を初期値 x0 = px から固定回数反復して解く
__device__ double perpendicular_foot_x(double px, double py, double sign)
{
  double x = px;
  for (int i = 0; i < NEWTON_ITERATIONS; ++i)
  {
    double w, w_p, w_pp;
    omega_with_derivatives(x, &w, &w_p, &w_pp);

    double diff = sign * w - py; // φ(x) - py
    double phi_p = sign * w_p;   // φ'(x)
    double phi_pp = sign * w_pp; // φ''(x)

    // ニュートン法の更新式: x ← x - g(x) / g'(x)
    x -= (x - px + phi_p * diff) / (1.0 + phi_p * phi_p + phi_pp * diff);
  }
  return x;
}

// 壁への沈み込みに対する反発力(ペナルティ法): f = K((x*, y*) - (px, py))
__device__ void repulsion(double spring_k, double px, double py, double *fx, double *fy)
{
  *fx = 0.0;
  *fy = 0.0;

  double w = omega(px);
  if (-w <= py && py <= w)
    return; // チャネル内部なら力は働かない

  double sign = (py > w) ? 1.0 : -1.0;
  double x = perpendicular_foot_x(px, py, sign);
  *fx = spring_k * (x - px);
  *fy = spring_k * (sign * omega(x) - py);
}

// 指定した状態におけるドリフト項(重心・角度それぞれの力・トルクによる決定論的な変化率)を求める。
// 予測子・修正子法では1段階目と2段階目の両方でこの関数を呼び、ドリフトを評価し直す
__device__ void drift(
    double spring_k, double length, double force_x,        // 外力・壁のばね定数・棒の長さ
    double c_1, double c_2,                                // 電場パラメータ C1 = βEp/l̃, C2 = ΔαE/p
    double x, double y, double angle,                      // 現在の重心位置と角度
    double *drift_x, double *drift_y, double *drift_angle) // [出力] ドリフト項の各成分
{
  double s, c;
  sincos(angle, &s, &c);

  // 棒の両端に働く壁からの反発力
  double half_x = 0.5 * length * c;
  double half_y = 0.5 * length * s;
  double f_plus_x, f_plus_y, f_minus_x, f_minus_y;
  repulsion(spring_k, x + half_x, y + half_y, &f_plus_x, &f_plus_y);
  repulsion(spring_k, x - half_x, y - half_y, &f_minus_x, &f_minus_y);

  // 電場が電気双極子に及ぼすトルク: 2 C1 cosΦ (1 + C2 sinΦ)
  double torque = 2.0 * c_1 * c * (1.0 + c_2 * s);
  // n × (f+ − f−)。n = (cosΦ, sinΦ) との2次元外積
  double cross = c * (f_plus_y - f_minus_y) - s * (f_plus_x - f_minus_x);

  *drift_x = force_x + 0.5 * (f_plus_x + f_minus_x);
  *drift_y = 0.5 * (f_plus_y + f_minus_y);
  *drift_angle = (torque + cross) / length;
}

// 1スレッドが1粒子(1サンプル)を担当し、開始時と終了時のx座標を書き出すカーネル。
// 集計は行わず生データを返す。ホスト側でCSVへ書き出す。
extern "C" __global__ void simulate(
    uint64_t seed,         // このケースの乱数シード
    uint64_t steps,        // 総ステップ数 T/Δt
    uint32_t start_index,  // 今回計算する最初のサンプルID(= current_sample_size)
    uint32_t sample_count, // 今回計算するサンプル数
    double delta_t,        // 時間刻み幅 Δt
    double spring_k,       // 壁のばね定数 K
    double length,         // 棒の長さ l
    double force_x,        // 一定外力 f(x方向)
    double c_1,            // 電場パラメータ C1 = βEp/l̃
    double c_2,            // 電場パラメータ C2 = ΔαE/p
    double *samples)       // [出力] サンプルごとに [開始x, 終了x] の2要素
{
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= sample_count)
    return;

  // 乱数列はサンプルIDで分離する。スレッド番号ではなくIDを使うことで、
  // 後から追加したサンプルが既存のサンプルと同じ乱数列にならない
  uint32_t id = start_index + idx;

  double noise_scale = sqrt(delta_t); // Wiener過程の増分の標準偏差 √Δt
  double inv_length = 1.0 / length;   // ループ内の除算を避けるための逆数

  // サンプルごとに独立な乱数列を初期化(同じシードでも列番号 id で分離される)。
  // XORWOW を採用(A100 での実測で Philox4x32-10 より約5%高スループット)
  curandState rng;
  curand_init(seed, id, 0, &rng);

  // 初期状態: x は1周期弱の範囲、y はチャネル内部、角度は一様分布
  double x = curand_uniform_double(&rng) * 0.8 - 0.1;
  double limit = omega(x) - 0.5 * length;
  double y = (curand_uniform_double(&rng) * 2.0 - 1.0) * limit;
  double angle = curand_uniform_double(&rng) * TAU;

  double start_x = x;

  for (uint64_t t = 0; t < steps; ++t)
  {
    // Wiener過程の増分(標準正規乱数 × √Δt)。予測子・修正子で共通して使う
    double2 xi = curand_normal2_double(&rng);
    double xi_phi = curand_normal_double(&rng);
    double noise_x = xi.x * noise_scale;
    double noise_y = xi.y * noise_scale;
    double noise_angle = 2.0 * xi_phi * noise_scale * inv_length;

    // 予測子: 始状態でのドリフトによる Euler–Maruyama 法の1ステップ
    double drift_x, drift_y, drift_angle;
    drift(spring_k, length, force_x, c_1, c_2, x, y, angle, &drift_x, &drift_y, &drift_angle);
    double predicted_x = x + drift_x * delta_t + noise_x;
    double predicted_y = y + drift_y * delta_t + noise_y;
    double predicted_angle = angle + drift_angle * delta_t + noise_angle;

    // 修正子: 予測状態でドリフトを評価し直し、始状態に適用する(model.md の無次元化SDE)
    drift(spring_k, length, force_x, c_1, c_2, predicted_x, predicted_y, predicted_angle,
          &drift_x, &drift_y, &drift_angle);
    x += drift_x * delta_t + noise_x;
    y += drift_y * delta_t + noise_y;
    angle += drift_angle * delta_t + noise_angle;
  }

  samples[2 * idx] = start_x;
  samples[2 * idx + 1] = x;
}
