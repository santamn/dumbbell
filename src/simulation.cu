#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include <iostream>
#include <stdint.h>
#include <float.h>

#define PI M_PI
#define TAU (2.0 * M_PI)

// __device__ = GPU上で実行され、GPUからのみ呼び出せる関数
// __global__ = CPUから呼び出せて、GPUで実行される関数（カーネル）

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
  for (int i = 0; i < 32; ++i)
  {
    double h = sign * omega(x) - py;
    double p = sign * omega_derivative(x);
    double d = (x - px + p * h) / (1.0 + p * p - sign * omega_derivative_second(x) * h);
    if (fabs(d) > 1e-10)
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

// 1粒子のシミュレーションを行い、最終的なx方向の変位を返す関数
__device__ double simulate_particle(
    double k,
    double delta_t,
    double noise_scale,
    uint64_t steps,
    unsigned long long seed,
    int idx,
    double length,
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

#define THREADS_PER_BLOCK 256

// 粒子の変位の総和と、変位の二乗の総和を計算するカーネル関数
__global__ void displacement_sum(
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
  // 1. グローバルなスレッドIDを計算
  // blockIdx.x = ブロックID
  // blockDim.x = ブロック内のスレッド数
  // threadIdx.x = ブロック内のスレッドID
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // 2. 各ブロックのスレッド間で共有されるメモリを静的に確保
  __shared__ double sum_disp[THREADS_PER_BLOCK];
  __shared__ double sum_sq_disp[THREADS_PER_BLOCK];

  double delta_x = 0.0;
  double delta_x_sq = 0.0;

  // 3. 有効なスレッドでのみシミュレーションを実行
  if (idx < ensemble_size)
  {
    delta_x = simulate_particle(k, delta_t, noise_scale, steps, seed, idx, length, force_x);
    delta_x_sq = delta_x * delta_x;
  }

  // 4. 各スレッドの計算結果を共有メモリ（Shared Memory）に書き込む
  sum_disp[threadIdx.x] = delta_x;
  sum_sq_disp[threadIdx.x] = delta_x_sq;
  // ブロック内の全スレッドがここへ到達するまで待機（同期）
  __syncthreads();

  // 5. ブロック内リダクション（並列和計算）
  // 256→128→64→32→16→8→4→2→1 のように半分ずつ足し合わせていく
  // blockDim.xはブロック内のスレッド数（この場合256）で、sは半分ずつ減っていく
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
  {
    if (threadIdx.x < s)
    {
      sum_disp[threadIdx.x] += sum_disp[threadIdx.x + s];
      sum_sq_disp[threadIdx.x] += sum_sq_disp[threadIdx.x + s];
    }
    // 各ステップの足し合わせが終わるのを待つ
    __syncthreads();
  }

  // 6. ブロック内の総和を代表してスレッド 0がグローバルメモリに足す
  if (threadIdx.x == 0)
  {
    atomicAdd(out_displacement, sum_disp[0]);
    atomicAdd(out_square_displacement, sum_sq_disp[0]);
  }
}

// extern "C" とすることでC++特有の名前修飾（マングリング）を防ぎ、Rustから呼び出せるようにする
extern "C"
{
  void calculate_displacement_sum_on_gpu(
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
    double *out_disp;
    double *out_sq_disp;
    cudaMalloc(&out_disp, sizeof(double));
    cudaMalloc(&out_sq_disp, sizeof(double));
    cudaMemset(out_disp, 0, sizeof(double));
    cudaMemset(out_sq_disp, 0, sizeof(double));

    // CUDAの実行構成（ブロックとスレッド）の決定
    // 256スレッドを1ブロックとし、必要なブロック数を算出する
    int blocks = (ensemble_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK; // ⌈ensemble_size / threads⌉

    // カーネル（GPU関数）を起動
    displacement_sum<<<blocks, THREADS_PER_BLOCK>>>(
        k,
        delta_t,
        noise_scale,
        steps,
        ensemble_size,
        seed,
        length,
        force_x,
        out_disp,
        out_sq_disp);

    // GPUの計算がすべて終わるまでCPUを待機させる
    cudaDeviceSynchronize();

    // GPU上で計算した結果を、直接Rust側から渡されたCPU側のポインタにコピーする
    cudaMemcpy(total_displacement, out_disp, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(total_square_displacement, out_sq_disp, sizeof(double), cudaMemcpyDeviceToHost);

    // GPUのメモリを解放
    cudaFree(out_disp);
    cudaFree(out_sq_disp);
  }
}
