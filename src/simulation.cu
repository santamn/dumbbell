#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include <iostream>
#include <stdint.h>
#include <float.h>

#include "constants.h"

constexpr int THREADS_PER_BLOCK = 256;
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;
constexpr double TAU = 2.0 * M_PI;

// __device__ = GPU上で実行され、GPUからのみ呼び出せる関数
// __global__ = CPUから呼び出せて、GPUで実行される関数（カーネル）

// omega(x) = sin(2πx) + 0.25sin(4πx) + 1.12 = sin(2πx) + 0.5sin(2πx)cos(2πx) + 1.12
__device__ double omega(double x)
{
  double s, c;
  sincospi(2.0 * x, &s, &c);
  return fma(s, fma(0.5, c, 1.0), 1.12);
}

// Δ = (x - px + ω'(x) * (ω(x) - py)) / (1 + ω'(x)^2 + ω''(x) * (ω(x) - py))
__device__ double newton_delta(double px, double offset, double x)
{
  constexpr double MINUS_PI = -M_PI;
  constexpr double MINUS_FOUR_PI_SQ = -4.0 * M_PI * M_PI;
  constexpr double MINUS_EIGHT_PI_SQ = -8.0 * M_PI * M_PI;

  double s, c;
  sincospi(2.0 * x, &s, &c);

  double w_sub = fma(s, fma(0.5, c, 1.0), offset);               // s * (0.5 * c + 1.0) + 1.12 - ±py
  double w_p = fma(c, fma(TAU, c, TAU), MINUS_PI);               // c * (2πc + 2π) - π
  double w_pp = s * fma(MINUS_EIGHT_PI_SQ, c, MINUS_FOUR_PI_SQ); // s * (-8π^2 * c - 4π^2)

  // w_p * w_sub + (x - px) / w_pp * w_sub + (w_p^2 + 1.0)
  return fma(w_p, w_sub, x - px) / fma(w_pp, w_sub, fma(w_p, w_p, 1.0));
}

// 点から壁へ降ろした垂線の足のx座標を求める関数
__device__ double perpendicular_foot_x(double px, double py, double sign)
{
  constexpr double EPSILON = 1e-10;

  double x = px;
  double offset = fma(-sign, py, 1.12); // 1.12 - ±py

  for (int i = 0; i < 32; ++i)
  {
    double d = newton_delta(px, offset, x);
    if (fabs(d) > EPSILON)
    {
      x -= d;
    }
    else
    {
      break;
    }
  }
  return x;
}

// 壁への沈み込みに対する反発力を計算する関数
__device__ void repulsion(double px, double py, double *fx, double *fy)
{
  *fx = 0.0;
  *fy = 0.0;

  double w = omega(px);
  if (-w <= py && py <= w)
    return;

  double sign = (py > w) ? 1.0 : -1.0;
  double x = perpendicular_foot_x(px, py, sign);
  double y = sign * omega(x);

  *fx = K * (x - px);
  *fy = K * (y - py);
}

// 1粒子のシミュレーションを行い、最終的なx方向の変位を返す関数
__device__ double simulate_particle(
    uint64_t seed,
    int idx,
    double length,
    double inv_length,
    double force_x)
{
  // 1. 乱数生成器の初期化
  curandState state;
  // 同じシードでも、スレッドID(idx)を渡すことで、全スレッドが異なる乱数列を生成する
  curand_init(seed, idx, 0, &state);

  // 2. 粒子の初期状態の決定
  double x = (curand_uniform_double(&state) * 0.8) - 0.1;
  double limit = omega(x) - length * 0.5;
  double y = (curand_uniform_double(&state) * 2.0 * limit) - limit;
  double angle = curand_uniform_double(&state) * TAU;

  double start_x = x;

  // 3. シミュレーションのメインループ
  for (uint64_t t = 0; t < STEPS; ++t)
  {
    // ブラウン運動用の正規分布ノイズを生成
    double xi_x = curand_normal_double(&state);
    double xi_y = curand_normal_double(&state);
    double xi_phi = curand_normal_double(&state);

    double s, c;
    sincos(angle, &s, &c);

    double h_x = 0.5 * length * c;
    double h_y = 0.5 * length * s;

    // 棒の両端の座標
    double p1_x = x + h_x;
    double p1_y = y + h_y;
    double p2_x = x - h_x;
    double p2_y = y - h_y;

    // 壁からの反発力を計算
    double f1_x, f1_y;
    repulsion(p1_x, p1_y, &f1_x, &f1_y);

    double f2_x, f2_y;
    repulsion(p2_x, p2_y, &f2_x, &f2_y);

    // オイラー・丸山法による位置と角度の更新
    // x += (force_x + 0.5 * (f1_x + f2_x)) * delta_t + xi_x * noise_scale;
    // y += (0.5 * (f1_y + f2_y)) * delta_t + xi_y * noise_scale;
    // angle += (ext_prod * delta_t + 2.0 * xi_phi * noise_scale) / length;
    double force_sum_x = fma(0.5, f1_x + f2_x, force_x);
    double force_sum_y = 0.5 * (f1_y + f2_y);
    double ext_prod = fma(-s, f1_x - f2_x, c * (f1_y - f2_y));

    x = fma(force_sum_x, DELTA_T, fma(xi_x, NOISE_SCALE, x));
    y = fma(force_sum_y, DELTA_T, fma(xi_y, NOISE_SCALE, y));
    angle = fma(ext_prod * inv_length, DELTA_T, fma(2.0 * xi_phi * inv_length, NOISE_SCALE, angle));
  }

  // 4. 最終的な変位を計算
  return x - start_x;
}

// =====================================================
// ==============  GPUでの実行単位について  ================
// =====================================================
// CUDAでは、多数のスレッドを管理するために
//  グリッド (Grid) > ブロック (Block) > スレッド (Thread: 実行の最小単位)
// という階層構造を持つ。
// スレッドは仮想的な実行単位であり、物理的なコア以上に存在できるため、GPUは数万スレッドを同時に実行する
//
// ======================================================
// ==============  GPUでの並列実行のイメージ  ================
// ======================================================
// 1. SMへのブロックの割り当て
//  GPUの内部には SM (Streaming Multiprocessor) と呼ばれる演算ユニットの塊が複数搭載されていて
//  プログラム（カーネル）が起動すると、指定した数のブロックが手の空いているSMに次々と割り振られる。
// 2. Warp（ワープ）単位での命令実行
//  SMに割り当てられたブロック内のスレッドは、32個ずつのグループに分割される。これらは Warp（ワープ） と呼ばれる。
// 3. SIMTアーキテクチャ (Single Instruction, Multiple Threads)
//  Warp内の32個のCUDAコアは、「まったく同じ命令」を「同時に」実行する = SIMT
//  全員が同じ__global__関数のコードを読み込み、1行目から同時に進んでいく。

// 32スレッド（1Warp）内でレジスタを直接やり取りして総和を求める関数
__inline__ __device__ double warp_reduce_sum(double val)
{
  // 16->8->4->2->1と半分ずつズレたスレッドから値をもらって足す
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
  {
    // __shfl_down_sync(mask, val, offset)
    //  - mask: 参加するスレッドを指定するビットマスク; 0xffffffff = 全スレッドが参加
    //  - val: 各スレッドが持っている値
    //  - offset: スレッドIDが offset だけ大きいスレッドから値をもらう
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

// 粒子の変位の総和と、変位の二乗の総和を計算するカーネル関数
extern "C" __global__ void displacements_sum(
    uint64_t seed,
    double length,
    double inv_length,
    double force_x,
    double *out_displacement,
    double *out_square_displacement)
{
  // 1. グローバルなスレッドIDを計算
  // blockIdx.x = ブロックID
  // blockDim.x = ブロック内のスレッド数
  // threadIdx.x = ブロック内のスレッドID
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  double delta_x = 0.0;
  double delta_x_sq = 0.0;

  // 2. 有効なスレッドでのみシミュレーションを実行
  if (idx < ENSEMBLE_SIZE)
  {
    delta_x = simulate_particle(seed, idx, length, inv_length, force_x);
    delta_x_sq = delta_x * delta_x;
  }

  // 3. Warp(32スレッド)内で同期なしに一気に総和を計算する
  delta_x = warp_reduce_sum(delta_x);
  delta_x_sq = warp_reduce_sum(delta_x_sq);

  // 4. 各Warpの代表値（先頭スレッド）だけを共有メモリに書き込む
  // 1ブロック=256スレッド=8Warpなので、サイズは8で十分
  __shared__ double shared_disp[WARPS_PER_BLOCK];
  __shared__ double shared_sq_disp[WARPS_PER_BLOCK];

  int lane = threadIdx.x % WARP_SIZE;    // Warp内のID (0~31)
  int warp_id = threadIdx.x / WARP_SIZE; // WarpのID (0~7)

  if (lane == 0)
  {
    shared_disp[warp_id] = delta_x;
    shared_sq_disp[warp_id] = delta_x_sq;
  }
  // 代表スレッドが書き込んだ後、同じブロックの全スレッドで共有メモリの値を読み取るために同期する
  __syncthreads();

  // 5. 最初のWarpが、書き込まれた各Warpの合計値をさらにリダクション
  if (warp_id == 0)
  {
    // 存在するWarpの数だけ値を読み込み、それ以外は0にする
    delta_x = (lane < WARPS_PER_BLOCK) ? shared_disp[lane] : 0.0;
    delta_x_sq = (lane < WARPS_PER_BLOCK) ? shared_sq_disp[lane] : 0.0;

    delta_x = warp_reduce_sum(delta_x);
    delta_x_sq = warp_reduce_sum(delta_x_sq);

    // 6. 全体の代表スレッド(threadIdx.x == 0)がグローバルメモリに出力
    if (lane == 0)
    {
      atomicAdd(out_displacement, delta_x);
      atomicAdd(out_square_displacement, delta_x_sq);
    }
  }
}
