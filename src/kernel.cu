#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include <iostream>
#include <stdint.h>
#include <float.h>

#define PI M_PI
#define TAU (2.0 * M_PI)

// __device__ = GPU上で実行され、GPUからのみ呼び出せる関数
// omega(x) = sin(2πx) + 0.25sin(4πx) + 1.12 = sin(2πx) + 0.5sin(2πx)cos(2πx) + 1.12
__device__ double omega(double x)
{
  double s, c;
  sincos(TAU * x, &s, &c);
  return s + 0.5 * s * c + 1.12;
}

// omega'(x) = 2πcos(2πx) + πcos(4πx) = 2πcos(2πx)(cos(2πx) + 1) - π
__device__ double omega_derivative(double x)
{
  double c = cos(TAU * x);
  return TAU * c * (c + 1.0) - PI;
}

// omega"(x) = -4π^2sin(2πx) - 4π^2sin(4πx) = -(2π)^2sin(2πx)(1 + 2cos(2πx))
__device__ double omega_derivative_second(double x)
{
  double s, c;
  sincos(TAU * x, &s, &c);
  return -TAU * TAU * s * (1.0 + 2.0 * c);
}

// 点から壁へ降ろした垂線の足のx座標を求める関数
__device__ double perpendicular_foot_x(double px, double py, double sign)
{
  double x = px;
  for (int i = 0; i < 1000; ++i)
  {
    double h = sign * omega(x) - py;
    double p = sign * omega_derivative(x);
    double d = (x - px + p * h) / (1.0 + p * p - sign * omega_derivative_second(x) * h);
    if (fabs(d) > DBL_EPSILON)
    {
      x = x - d;
    }
    else
    {
      break;
    }
  }
  return x;
}

// 壁への沈み込みに対する反発力を計算する関数
__device__ void repulsion(double k, double px, double py, double *fx, double *fy)
{
  *fx = 0.0;
  *fy = 0.0;

  double w = omega(px);
  if (-w <= py && py <= w)
    return;

  double sign = (py > w) ? 1.0 : -1.0;
  double x = perpendicular_foot_x(px, py, sign);
  double y = sign * omega(x);

  *fx = k * (x - px);
  *fy = k * (y - py);
}

// =====================================================
// ==============  GPUでの実行単位について  ================
// =====================================================
// CUDAでは、多数のスレッドを管理するために
//  グリッド (Grid) > ブロック (Block) > スレッド (Thread: 実行の最小単位)
// という階層構造を持つ。
// スレッドは仮想的な実行単位であり、物理的なコア以上に存在できるため、GPUは数万スレッドを同時に実行する

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
//  全員が同じ simulate_particles という関数のコードを読み込み、1行目から同時に進んでいく。

// __global__ = CPUから呼び出せて、GPUで実行される関数（カーネル）
// シミュレーションの本体。オイラー・丸山法で粒子の位置と角度を更新し、アトミック演算で結果を集計する
__global__ void simulate_particles(
    double k,
    double delta_t,
    double noise_scale,
    uint64_t steps,
    uint64_t ensemble_size,
    unsigned long long seed,
    double length,
    double force_x,
    double *out_displacement,
    double *out_square_displacement)
{
  // 1. グローバルなスレッド番号（=担当する粒子のID）を求める
  // blockIdx.x: グリッド内でのブロック番号
  // blockDim.x: 1つのブロックに含まれるスレッドの数
  // threadIdx.x: ブロック内でのスレッド番号
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // スレッド数がこのGPUに割り当てられたアンサンブル数を超えた場合は何もしない（端数処理）
  if (idx >= ensemble_size)
    return;

  // 2. 乱数生成器の初期化
  curandState state;
  // 同じシードでも、スレッドID(idx)を渡すことで、全スレッドが異なる乱数列を生成する
  curand_init(seed, idx, 0, &state);

  // 3. 粒子の初期状態の決定
  double x = (curand_uniform_double(&state) * 0.8) - 0.1;
  double limit = omega(x) - length * 0.5;
  double y = (curand_uniform_double(&state) * 2.0 * limit) - limit;
  double angle = curand_uniform_double(&state) * TAU;

  double start_x = x;

  // 4. シミュレーションのメインループ（これが数千万回実行される）
  for (uint64_t t = 0; t < steps; ++t)
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
    repulsion(k, p1_x, p1_y, &f1_x, &f1_y);

    double f2_x, f2_y;
    repulsion(k, p2_x, p2_y, &f2_x, &f2_y);

    double exterior_product = -s * (f1_x - f2_x) + c * (f1_y - f2_y);

    // オイラー・丸山法による位置と角度の更新
    x += (force_x + 0.5 * (f1_x + f2_x)) * delta_t + xi_x * noise_scale;
    y += (0.5 * (f1_y + f2_y)) * delta_t + xi_y * noise_scale;
    angle += (exterior_product * delta_t + 2.0 * xi_phi * noise_scale) / length;
  }

  // 5. 最終的な変位を計算
  double delta_x = x - start_x;

  // 6. 結果の集計（アトミック演算）
  // 数万スレッドが同時に out_displacement に足し算をするとデータが壊れるため、
  // atomicAdd を使って1スレッドずつ順番に足すようにする
  atomicAdd(out_displacement, delta_x);
  atomicAdd(out_square_displacement, delta_x * delta_x);
}

// extern "C" とすることでC++特有の名前修飾（マングリング）を防ぎ、Rustから呼び出せるようにする
extern "C"
{
  void run_simulation_cuda(
      uint64_t device_id,
      double k,
      double delta_t,
      double noise_scale,
      uint64_t steps,
      uint64_t ensemble_size,
      unsigned long long seed,
      double length,
      double force_x,
      double *total_displacement,
      double *total_square_displacement)
  {
    // 対象のGPUデバイスを選択
    cudaSetDevice(device_id);

    // GPU上のメモリ領域を確保し、0で初期化
    double *d_out_disp;
    double *d_out_sq_disp;
    cudaMalloc(&d_out_disp, sizeof(double));
    cudaMalloc(&d_out_sq_disp, sizeof(double));
    cudaMemset(d_out_disp, 0, sizeof(double));
    cudaMemset(d_out_sq_disp, 0, sizeof(double));

    // CUDAの実行構成（ブロックとスレッド）の決定
    // 256スレッドを1ブロックとし、必要なブロック数を算出する
    int threads = 256;
    int blocks = (ensemble_size + threads - 1) / threads; // ⌈ensemble_size / threads⌉

    // カーネル（GPU関数）を起動
    simulate_particles<<<blocks, threads>>>(
        k,
        delta_t,
        noise_scale,
        steps,
        ensemble_size,
        seed,
        length,
        force_x,
        d_out_disp,
        d_out_sq_disp);

    // GPUの計算がすべて終わるまでCPUを待機させる
    cudaDeviceSynchronize();

    // GPU上で計算した結果を、CPU側（ホスト側）の変数にコピーして持ってくる
    double h_out_disp = 0;
    double h_out_sq_disp = 0;
    cudaMemcpy(&h_out_disp, d_out_disp, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_out_sq_disp, d_out_sq_disp, sizeof(double), cudaMemcpyDeviceToHost);

    // Rust側から渡されたポインタに結果を代入する
    *total_displacement = h_out_disp;
    *total_square_displacement = h_out_sq_disp;

    // GPUのメモリを解放
    cudaFree(d_out_disp);
    cudaFree(d_out_sq_disp);
  }
}
