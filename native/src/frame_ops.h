#pragma once
#include "dlss5.h"
#include <cuda.h>
namespace dlss5::frame_ops {
void alpha(const D5Tensor& original,D5Tensor encoded,D5Tensor current_alpha,CUstream);
void fsr_input(D5Tensor linear,D5Tensor alpha,D5Tensor previous_alpha,bool reset,D5Tensor invalid,D5Tensor reactive,D5Tensor target,D5Tensor combined,float white,CUstream);
void fsr_restore(D5Tensor source_linear,D5Tensor temporal,float white,CUstream);
}
