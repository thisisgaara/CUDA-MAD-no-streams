// header.h
//-------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
//-------------------------------------------------------------------
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cufft.h>
//-------------------------------------------------------------------
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
//-------------------------------------------------------------------
#include <windows.h>
#include <iostream> 
#include <fstream>
#include <stdio.h>
#include <cmath>
//-------------------------------------------------------------------
#define DEVICE_NUM    0 // 0 => Tesla, 1 => Quadro
#define N             512
#define IMG_SIZE      N*N
#define LMSE_CONST    7 + 7 * N
#define PI            3.1415927
#define thetaSigma    0.5235988
#define BLOCK_SIZE    16
#define REAL_SIZE     sizeof(float)* IMG_SIZE
#define COMPLEX_SIZE  sizeof(cufftComplex)* IMG_SIZE
#define NUMBER_OF_ITERATIONS	30
#define NUMBER_OF_IMAGES 60
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		getchar();
		if (abort) exit(code);
	}
}
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
   getchar(); \
   exit(0); \
  }                                                                 \
}
//-------------------------------------------------------------------
// User defined function
typedef struct out_data
{
	float hi_index;
	float lo_index;
	float mad_value;
}out_data;
//-------------------------------------------------------------------
// Declare Wrapper Function
struct out_data kernel_wrapper(const cv::Mat& mat_ref, const cv::Mat& mat_dst);

