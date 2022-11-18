#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <torch/torch.h>

#include "i_gemm_tpp.hpp"
#include "test_gemm.h"

namespace intel_mlperf {

void save_input(at::Tensor input, std::string file_name)
{
  auto row = input.sizes()[0];
  auto col = input.sizes()[1];
  auto a_ = reinterpret_cast<int8_t (*)>(input.data_ptr());

  std::fstream input_file;
  input_file.open(file_name, std::ios::out);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
      input_file << +a_[(i * col) + j] << " ";
    }
    input_file << "\n";
	}
  input_file.close();
  printf("input saved\n");
}

void save_output(at::Tensor output, std::string file_name)
{
  auto row = output.sizes()[0];
  auto col = output.sizes()[1];
  auto a_ = reinterpret_cast<int (*)>(output.data_ptr());

  std::fstream output_file;
  output_file.open(file_name, std::ios::out);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
      output_file << a_[(i * col) + j] << " ";
    }
    output_file << "\n";
	}
  output_file.close();
  printf("output saved\n");
}

void save_weight(at::Tensor weight)
{
  auto row = weight.sizes()[0];
  auto col = weight.sizes()[1];
  auto w_ = reinterpret_cast<int8_t (*)>(weight.data_ptr());

  std::fstream wf;
  wf.open("weight_block.txt", std::ios::out);
	for (int i = 0; i < col; i++) {
		for (int j = 0; j < row; j++) {
      wf << +w_[(i * row) + j] << " ";
    }
    wf << "\n";
	}
  wf.close();
  printf("weight saved\n");
}


void test_conv_gemm(const int sl, const int ic, const int oc) {  // 32 64 32 int8

  torch::manual_seed(123456789);

  auto input = torch::randint(-50, 50, {sl, ic}).to(at::kChar);
  auto weight = torch::randint(-50, 50, {ic, oc}).to(at::kChar);
  auto output = at::zeros({sl, oc}).to(at::kInt);

  int row_blocks = sl / 32;

  auto input_ = reinterpret_cast<int8_t (*)[32][ic]>(input.data_ptr());
  auto weight_ = reinterpret_cast<int8_t (*)[16][ic]>(weight.data_ptr());
  auto output_ = reinterpret_cast<int (*)[32][oc]>(output.data_ptr());

  auto start = Time::now();
  # pragma omp parallel for
  for (int c = 0; c < row_blocks; ++c) {
    // int acc_pad[1024];
    // memset(acc_pad, 0, 1024 * 4);
    _tile_gemm_32x64::compute(input_[c], ic, weight_, output_[c]);
    // _tile_gemm_32x64::test_out(output_[c], oc, output_[c]);
  }
  auto during = std::chrono::duration_cast<std::chrono::nanoseconds>(Time::now() - start).count();
  printf("%d x %d gemm time : %f ms\n", ic, oc, (float)during * 1e-6);

//   save_input(input, "input.txt");
//   save_output(output, "output.txt");
//   save_weight(weight);
}

}
