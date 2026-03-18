#include <stdio.h>

int main()
{
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);

  if (deviceCount == 0)
  {
    printf("GPUが見つかりません。\n");
    return 1;
  }

  for (int i = 0; i < deviceCount; ++i)
  {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("GPU %d: %s -> Compute Capability: sm_%d%d\n", i, prop.name, prop.major, prop.minor);
  }
  return 0;
}