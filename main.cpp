#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdlib.h>
#include <random>
#include <cmath>
#include <math.h>
#include <omp.h>

#include "src/amx_config.hpp"
#include "src/amx_init.hpp"
#include "src/test_gemm.h"


int main(int argc, char* argv[]) {
  amx_init::amx_init();
  intel_mlperf::Tilecfg().set_config();
  intel_mlperf::test_conv_gemm(2097152, 64, 32);
  return 0;
}
