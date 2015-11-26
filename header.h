// header.h
//=============================================================================
// Include OpenCV to read in image
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
//=============================================================================
// Include CUDA runtime & NSIGHT annotations
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "nvToolsExt.h" // Core NVTX API
#include "nvToolsExtCuda.h" // Only the nvToolsExt.h should be enough
#include <cuda_profiler_api.h>
//=============================================================================
#include <cufft.h>
//=============================================================================
// CPU Timer
#include <windows.h>
//=============================================================================
// Basic Stuff:
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <cstring>
#include <math.h>
//=============================================================================
#define CSF_FILTER
#define HI_STATS
#define LOG_GABOR
#define LOW_STATS
#define COMBINE_MAD

//----------------------------------------------------------------------------
//image array
typedef struct image
{
	double csf_filter_Gpu;
	double appearence_statistic_Gpu;
	double detection_gabor_filterbank_GPU;
	double detection_statistic_GPU;
}image;
extern image *image_array;
//-----------------------------------------------------------------------------
extern int NUM_IMAGES;
extern double timing_sum_global;
extern int image_index; //This variable indexes into the image_array
extern double timing_sum_combine_mad;
//=============================================================================
// Declare Functions:
cudaError_t kernel_wrapper(const cv::Mat &mat_ref, const cv::Mat &mat_dst);
void linearize_and_cast_from_Mat_to_float(const cv::Mat& mat_in, float* h_float);
void de_linearize_and_cast_float_to_Mat(float *float_array, cv::Mat &mat_out, const int SIZE);
void write_to_file_DEBUG(float* w, const int SIZE);
float reduce_sum_of_squares_2D_CPU(float* in, const int INSIDE_BOUND, const int OUTSIDE_BOUND);
//=============================================================================
