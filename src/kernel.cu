#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include <iostream>

#define DELTA_T 2e-7
#define K 1.5e6
#define TIME 10.0
#define PI 3.14159265358979323846
#define TAU (2.0 * PI)
#define NOISE_SCALE sqrt(DELTA_T)

__device__ double omega(double x)
{
  double s = sin(TAU * x);
  double c = cos(TAU * x);
  return s + 0.5 * s * c + 1.12;
}

__device__ double omega_derivative(double x)
{
  double c = cos(TAU * x);
  return TAU * c * (c + 1.0) - PI;
}

__device__ double omega_derivative_second(double x)
{
  double s = sin(TAU * x);
  double c = cos(TAU * x);
  return -TAU * TAU * s * (1.0 + 2.0 * c);
}

__device__ double perpendicular_foot_x(double px, double py, double sign)
{
  double x = px;
  for (int i = 0; i < 1000; ++i)
  {
    double current_omega = omega(x);
    double h = sign * current_omega - py;
    double p = sign * omega_derivative(x);
    double d = (x - px + p * h) / (1.0 + p * p - sign * omega_derivative_second(x) * h);
    if (fabs(d) > 2.2204460492503131e-16)
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

__device__ void repulsion(double px, double py, double *fx, double *fy)
{
  double current_omega = omega(px);
  if (py > current_omega)
  {
    double f_x = perpendicular_foot_x(px, py, 1.0);
    double f_y = 1.0 * omega(f_x);
    *fx = K * (f_x - px);
    *fy = K * (f_y - py);
  }
  else if (py < -current_omega)
  {
    double f_x = perpendicular_foot_x(px, py, -1.0);
    double f_y = -1.0 * omega(f_x);
    *fx = K * (f_x - px);
    *fy = K * (f_y - py);
  }
  else
  {
    *fx = 0.0;
    *fy = 0.0;
  }
}

__global__ void simulate_particles(
    unsigned long long seed,
    double length,
    double force_x,
    uint64_t steps,
    uint64_t global_size,
    double *out_displacement,
    double *out_square_displacement)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= global_size)
    return;

  curandState state;
  curand_init(seed, idx, 0, &state);

  double x = (curand_uniform_double(&state) * 0.8) - 0.1;
  double limit = omega(x) - length * 0.5;
  double y = (curand_uniform_double(&state) * 2.0 * limit) - limit;
  double angle = curand_uniform_double(&state) * TAU;

  double start_x = x;

  for (uint64_t t = 0; t < steps; ++t)
  {
    double xi_x = curand_normal_double(&state);
    double xi_y = curand_normal_double(&state);
    double xi_phi = curand_normal_double(&state);

    double s, c;
    sincos(angle, &s, &c);

    double h_x = 0.5 * length * c;
    double h_y = 0.5 * length * s;
    double e_phi_x = -s;
    double e_phi_y = c;

    double p1_x = x + h_x;
    double p1_y = y + h_y;
    double p2_x = x - h_x;
    double p2_y = y - h_y;

    double f1_x, f1_y;
    repulsion(p1_x, p1_y, &f1_x, &f1_y);

    double f2_x, f2_y;
    repulsion(p2_x, p2_y, &f2_x, &f2_y);

    x += (force_x + 0.5 * (f1_x + f2_x)) * DELTA_T + xi_x * NOISE_SCALE;
    y += (0.5 * (f1_y + f2_y)) * DELTA_T + xi_y * NOISE_SCALE;

    double f_diff_x = f1_x - f2_x;
    double f_diff_y = f1_y - f2_y;
    double dot_prod = f_diff_x * e_phi_x + f_diff_y * e_phi_y;

    angle += (dot_prod * DELTA_T + 2.0 * xi_phi * NOISE_SCALE) / length;
  }

  double delta_x = x - start_x;

  atomicAdd(out_displacement, delta_x);
  atomicAdd(out_square_displacement, delta_x * delta_x);
}

extern "C"
{
  void run_simulation_cuda(
      uint64_t device_id,
      unsigned long long seed,
      double length,
      double force_x,
      uint64_t steps,
      uint64_t ensemble_size,
      double *total_displacement,
      double *total_square_displacement)
  {
    cudaSetDevice(device_id);

    double *d_out_disp;
    double *d_out_sq_disp;
    cudaMalloc(&d_out_disp, sizeof(double));
    cudaMalloc(&d_out_sq_disp, sizeof(double));
    cudaMemset(d_out_disp, 0, sizeof(double));
    cudaMemset(d_out_sq_disp, 0, sizeof(double));

    int threads = 256;
    int blocks = (ensemble_size + threads - 1) / threads;

    simulate_particles<<<blocks, threads>>>(seed, length, force_x, steps, ensemble_size, d_out_disp, d_out_sq_disp);
    cudaDeviceSynchronize();

    double h_out_disp = 0;
    double h_out_sq_disp = 0;
    cudaMemcpy(&h_out_disp, d_out_disp, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_out_sq_disp, d_out_sq_disp, sizeof(double), cudaMemcpyDeviceToHost);

    *total_displacement += h_out_disp;
    *total_square_displacement += h_out_sq_disp;

    cudaFree(d_out_disp);
    cudaFree(d_out_sq_disp);
  }
}
