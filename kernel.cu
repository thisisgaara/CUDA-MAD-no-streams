//=============================================================================
#include "header.h"
//=============================================================================
#define N							512
#define IMG_SIZE			N*N
#define LMSE_CONST		7 + 7 * N
#define PI						3.1415927
#define thetaSigma		0.5235988
#define BLOCK_SIZE		16
#define REAL_SIZE			sizeof(float)* IMG_SIZE
#define COMPLEX_SIZE	sizeof(cufftComplex)* IMG_SIZE
//=============================================================================
__global__ void buildGabor(float* logGabor, const int orientIdx, const int scaleIdx)
{
	const float nOrient = 4.0f;
	const float nScale = 5.0f;
	const float sigmaOnf = 0.55f;
	float wavelength[5] = { 3.0f, 9.0f, 27.0f, 81.0f, 243.0f };
	float angl = orientIdx * PI / nOrient; // Calculate filter angle

	// Construct the filter - first calculate the radial filter component.
	float fo = 1.0f / wavelength[scaleIdx];	// Centre frequency of filter.
	float rfo = fo / 0.5f;										// Normalised radius from centre of frequency plane
	// corresponding to fo.
	//for (int i = 0; i < N; i++)
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	int lin_idx = i * N + j;

	// REDUCE GLOBAL MEMORY TRAFIC BY COMPUTING EACH INDIVIDUAL MATRIX VALUE IN EACH THREAD
	float Y = -1 + i*0.003906f;
	float X = -1 + j*0.003906f;
	float sin_theta_temp = sin(atan2f(-Y, X));
	float cos_theta_temp = cos(atan2f(-Y, X));
	float radius_temp = sqrt(X * X + Y * Y);

	float ds = sin_theta_temp * cos(angl) - cos_theta_temp * sin(angl); // Difference in sin
	float dc = cos_theta_temp * cos(angl) + sin_theta_temp * sin(angl); // Difference in cos
	float diffTheta = abs(atan2(ds, dc));																			// Absolute angular distance
	float spread = exp((-diffTheta * diffTheta) / (2 * thetaSigma * thetaSigma)); // Calculate the angular filter component.
	float gabor = exp((-(log(radius_temp / rfo)) * (log(radius_temp / rfo)) / (2 * log(0.55f) * log(0.55f))));
	logGabor[lin_idx] = spread * gabor;

	// WHY TEST ALL VALUES?
	// EXECUTE KERNEL THEN ONLY MODIFY THIS SINGLE VALUE IN ANOTHER KERNEL
	if (lin_idx == 131328)// lin_idx = (N^2 + N)/2 = 131328 (N=512)
		logGabor[lin_idx] = 0.0f;	//Get rid of the 0 radius value
}
//=============================================================================
__global__ void pointWise_complex_matrix_mult_kernel_2d(cufftComplex* img_spectrum, float* real_filter, cufftComplex* out)
{
	// Grab indices
	int index_x = threadIdx.x + blockIdx.x * blockDim.x;
	int index_y = threadIdx.y + blockIdx.y * blockDim.y;

	// map the two 2D indices to a single linear, 1D index
	int grid_width = gridDim.x * blockDim.x;
	int grid_index = index_y * grid_width + index_x;
	//     A*B = (a + jb)(c + jd) = ac + ajd + cjb - bd
	//				 = ac + (ad + cb)j - bd
	// Re[A*B] = ac - bd
	// Im[A*B] = ad + cb
	// d = 0
	// => Re(out) = ac and Im(out) = cb
	out[grid_index].x = img_spectrum[grid_index].x*real_filter[grid_index];	//Re(out)
	out[grid_index].y = real_filter[grid_index] * img_spectrum[grid_index].y;	//Im(out)
}
//=============================================================================
__global__ void magnitude_kernel(cufftComplex* d_inverse_complex, float* d_inverse_mag)
{
	// Grab indices
	int index_x = threadIdx.x + blockIdx.x * blockDim.x;
	int index_y = threadIdx.y + blockIdx.y * blockDim.y;

	// map the two 2D indices to a single linear, 1D index
	int grid_width = gridDim.x * blockDim.x;
	int grid_index = index_y * grid_width + index_x;

	// Grab Real and Imaginary parts of d_inverse_complex
	float a = d_inverse_complex[grid_index].x / float(IMG_SIZE);
	float b = d_inverse_complex[grid_index].y / float(IMG_SIZE);

	// Apply pythagorean formula (Euclidean L2-Norm)
	d_inverse_mag[grid_index] = sqrt(a*a + b*b);
}
//=============================================================================
__global__ void real_kernel(cufftComplex* complex_in, float* real_out)
{
	// Grab indices
	int index_x = threadIdx.x + blockIdx.x * blockDim.x;
	int index_y = threadIdx.y + blockIdx.y * blockDim.y;

	// map the two 2D indices to a single linear, 1D index
	int grid_width = gridDim.x * blockDim.x;
	int grid_index = index_y * grid_width + index_x;

	// Grab Real part of complex_in
	real_out[grid_index] = complex_in[grid_index].x / float(IMG_SIZE);
}
//=============================================================================
__global__ void R2C_kernel(float* float_in, cufftComplex* complex_out)
{
	// Grab indices
	int index_x = threadIdx.x + blockIdx.x * blockDim.x;
	int index_y = threadIdx.y + blockIdx.y * blockDim.y;

	// map the two 2D indices to a single linear, 1D index
	int grid_width = gridDim.x * blockDim.x;
	int grid_index = index_y * grid_width + index_x;

	complex_out[grid_index].x = float_in[grid_index];
	complex_out[grid_index].y = 0;
}
//=============================================================================
__global__ void yPlane_CSF_kernel(float* yPlane)
{
	float temp = (2 * 32 / 512.0f);
	float tempvar = -(256.0f) - 1 + 0.5f;

	tempvar = tempvar * temp;
	for (int j = 0; j < 512; j++)
	{
		tempvar += temp;
		yPlane[j] = tempvar;
	}
}
//=============================================================================
__global__ void xPlane_CSF_kernel(float* xPlane)
{
	float temp = (2 * 32 / 512.0f);
	float tempvar = -(256.0f) - 1 + 0.5f;
	tempvar = tempvar * temp;
	for (int i = 0; i < 512; i++)
	{
		tempvar += temp;
		xPlane[i] = tempvar;
	}
}
//=============================================================================
__global__ void map_to_luminance_domain_kernel1(float* float_img_in, float* L_hat)
{
	// Grab indices
	int index_x = threadIdx.x + blockIdx.x * blockDim.x;
	int index_y = threadIdx.y + blockIdx.y * blockDim.y;

	// map the two 2D indices to a single linear, 1D index
	int grid_width = gridDim.x * blockDim.x;
	int grid_index = index_y * grid_width + index_x;

	// Map from Pixel Domain [unitless] to Luminance Domain [cd/m^2] - (MAD eq. 1 and eq. 2)
	L_hat[grid_index] = pow((0.02874f*float_img_in[grid_index]), (2.2f / 3.0f));
}
//=============================================================================
__global__ void map_to_luminance_domain_kernel2(float* float_img_in1, float* L_hat1, float* float_img_in2, float* L_hat2)
{
	// Grab indices
	int index_x = threadIdx.x + blockIdx.x * blockDim.x;
	int index_y = threadIdx.y + blockIdx.y * blockDim.y;

	// map the two 2D indices to a single linear, 1D index
	int grid_width = gridDim.x * blockDim.x;
	int grid_index = index_y * grid_width + index_x;

	// Map from Pixel Domain [unitless] to Luminance Domain [cd/m^2] - (MAD eq. 1 and eq. 2)
	L_hat1[grid_index] = pow((0.02874f*float_img_in1[grid_index]), (2.2f / 3.0f));
	L_hat2[grid_index] = pow((0.02874f*float_img_in2[grid_index]), (2.2f / 3.0f));
}
//=============================================================================
__global__ void error_img_kernel(const float* ref, const float* dst, float* err)
{
	// Grab indices
	int index_x = threadIdx.x + blockIdx.x * blockDim.x;
	int index_y = threadIdx.y + blockIdx.y * blockDim.y;

	// map the two 2D indices to a single linear, 1D index
	int grid_width = gridDim.x * blockDim.x;
	int grid_index = index_y * grid_width + index_x;

	err[grid_index] = ref[grid_index] - dst[grid_index];
}
//=============================================================================
__global__ void build_CSF_kernel(float* csf, const float* yPlane, const float* xPlane)
{
	// Masking / luminance parameters
	float k = 0.02874;
	float G = 0.5;			// luminance threshold
	float C_slope = 1;		// slope of detection threshold
	float Ci_thrsh = -5;   // contrast to start slope, rather than const threshold
	float Cd_thrsh = -5;   // saturated threshold
	float ms_scale = 1;    // scaling constant

	float s_dbl, radfreq_dbl;
	float temp = (2 * 32 / 512.0);
	float tempvar = -(256.0) - 1 + 0.5f;

	//int idx = 0;
	//for (register int i = 0; i < width; i++)
	int i = blockIdx.x * blockDim.x + threadIdx.x; //Create unique Grid Index in x-dimension
	{
		float xVar = xPlane[i];
		//for (register int j = 0; j < height; j++)
		int j = blockIdx.y * blockDim.y + threadIdx.y; //Create unique Grid Index in y-dimension
		{
			float yVar = yPlane[j];
			s_dbl = ((1 - 0.7f) / 2 * cos(4 * atan2(yVar, xVar))) + (1 + 0.7f) / 2;

			radfreq_dbl = sqrt(xVar*xVar + yVar*yVar) / s_dbl;

			// map the two 2D indices to a single linear, 1D index
			int grid_width = gridDim.x * blockDim.x;
			int grid_index = j * grid_width + i;

			// (MAD eq. 3)
			if (radfreq_dbl < 7.8909)
			{
				csf[grid_index] = 0.9809f;
			}
			else
			{
				float tmp_real = 2.6f*(0.0192f + 0.114f* radfreq_dbl)*exp(-pow((0.114f*radfreq_dbl), 1.1f));
				csf[grid_index] = tmp_real;
			}
		}
	}
}
//=============================================================================
__global__ void fftShift_kernel(float* img)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < 256)
	{
		int j = threadIdx.y + blockIdx.y * blockDim.y;
		if (j < 256)
		{
			// Preserve spatial locality by storing data in registers
			float temp = img[i * 512 + j];
			img[i * 512 + j] = img[(i + 256) * 512 + (j + 256)];
			img[(i + 256) * 512 + (j + 256)] = temp;

			temp = img[(i + 256) * 512 + j];
			img[(i + 256) * 512 + j] = img[i * 512 + (j + 256)];
			img[i * 512 + (j + 256)] = temp;
		}
	}
}
//=============================================================================
__global__ void delta_stats_kernel(float *ref_outStd, float *ref_outSkw, float* ref_outKrt,
	float* dst_outStd, float* dst_outSkw, float* dst_outKrt, float scale, float* eta)
{
	// Grab indices
	int index_x = threadIdx.x + blockIdx.x * blockDim.x;
	int index_y = threadIdx.y + blockIdx.y * blockDim.y;

	// map the two 2D indices to a single linear, 1D index
	int grid_width = gridDim.x * blockDim.x;
	int grid_index = index_y * grid_width + index_x;

	float delta_stat1 = abs(ref_outStd[grid_index] - dst_outStd[grid_index]);
	float delta_stat2 = abs(ref_outSkw[grid_index] - dst_outSkw[grid_index]);
	float delta_stat3 = abs(ref_outKrt[grid_index] - dst_outKrt[grid_index]);
	eta[grid_index] += scale*(delta_stat1 + 2 * delta_stat2 + delta_stat3);
}
//=============================================================================
__global__ void fast_lo_stats_kernel(float* xVal, float* outStd, float* outSkw, float* outKrt)
{
	//Declarations
	//__shared__ float xVal_Shm[256];
	float xVal_local[256];

	float mean, stdev, skw, krt, stmp;
	int iB, jB;

	//for (i = 0; i<512 - 15; i += 4)
	int i = 4 * (threadIdx.x + blockIdx.x * blockDim.x);
	if (i < 497) //512-15=497
	{
		//for (j = 0; j<512 - 15; j += 4)
		int j = 4 * (threadIdx.y + blockIdx.y * blockDim.y);
		if (j < 497)
		{
			// THE FOLLOWING SET OF RUNNING SUMS CAN BE A set of PARALLEL REDUCTIONs (in shared memory?)
			// 256 itteratios -> log2(256)=8 itterations

			// Store block into registers (256 x 4Bytes = 1kB)
			int idx = 0;
			for (iB = i; iB < i + 16; iB++)
			{
				for (jB = j; jB < j + 16; jB++)
				{
					xVal_local[idx] = xVal[iB * 512 + jB];
					idx++;
				}
			}

			//Traverse through and get mean
			float mean = 0;
			for (idx = 0; idx < 256; idx++)
				mean += xVal_local[idx];				//this can be a simple reduction in shared memory
			mean = mean / 256.0f;

			//Traverse through and get stdev, skew and kurtosis
			stdev = 0;
			skw = 0;
			krt = 0;
			float xV_mean = 0;
			for (idx = 0; idx < 256; idx++)
			{
				// Place this commonly re-used value into a register to preserve temporal localitiy
				xV_mean = xVal_local[idx] - mean;
				stdev += xV_mean*xV_mean;
				skw += xV_mean*xV_mean*xV_mean;
				krt += xV_mean*xV_mean*xV_mean*xV_mean;
			}
			stmp = sqrt(stdev / 256.0f);
			stdev = sqrt(stdev / 255.0f);//MATLAB's std is a bit different

			if (stmp != 0){
				skw = (skw / 256.0f) / ((stmp)*(stmp)*(stmp));
				krt = (krt / 256.0f) / ((stmp)*(stmp)*(stmp)*(stmp));
			}
			else{
				skw = 0;
				krt = 0;
			}

			//---------------------------------------------------------------------------
			// This is the nearest neighbor interpolation - ACTUALLY NOT NEEDED!!!!!!!!
			// To remove the nested for loop here we need to modifie the algorithm to 
			// adjust for the pointwise muliplication done far later that uses a
			// 512x512 dimension matrix derived from the matrices this kernel produces
			// The modified output would be PxP (as described mathematically in the paper).
			//---------------------------------------------------------------------------
			// Only this final output should be written to global memory:
			for (iB = i; iB < i + 4; iB++)
			{
				for (jB = j; jB < j + 4; jB++)
				{
					outStd[(iB * 512) + jB] = stdev;
					outSkw[(iB * 512) + jB] = skw;
					outKrt[(iB * 512) + jB] = krt;
				}
			}

		}
	}
}
//=============================================================================
__global__ void zeta_map_kernel(float* outMean, float* outStd, float* outStdMod, float* zeta)
{
	// Grab indices
	int index_x = threadIdx.x + blockIdx.x * blockDim.x;
	int index_y = threadIdx.y + blockIdx.y * blockDim.y;

	// map the two 2D indices to a single linear, 1D index
	int grid_width = gridDim.x * blockDim.x;
	int p = index_y * grid_width + index_x;

	// Compute pth element of RMS contrast map C_err (MAD eq. 4 and eq. 5)
	float C_org = log(outStdMod[p] / outMean[p]);
	float C_err = log(outStd[p] / outMean[p]);
	if (outMean[p] < 0.5)
		C_err = -999999999999999; // log(0) = -infinity

	// Compute local visibility distortion map (MAD eq. 6)
	float delta = -5.0;
	if ((C_err > C_org) && (C_org > delta))
		zeta[p] = C_err - C_org;
	else if ((C_err > delta) && (delta >= C_org))
		zeta[p] = C_err - delta;
	else
		zeta[p] = 0;
}
//=============================================================================
__global__ void square_of_difference_kernel(float* ref, float* dst, float* out)
{
	// Grab indices
	int index_x = threadIdx.x + blockIdx.x * blockDim.x;
	int index_y = threadIdx.y + blockIdx.y * blockDim.y;

	// map the two 2D indices to a single linear, 1D index
	int grid_width = gridDim.x * blockDim.x;
	int i = index_y * grid_width + index_x;
	out[i] = (ref[i] - dst[i])*(ref[i] - dst[i]);
}
//=============================================================================
__global__ void LMSE_map_kernel(float* reflut, float* D)
{
	// This is kernel #D14
	//for (int j = 0; j < boundaray; j++)
	int j = threadIdx.x + blockIdx.x * blockDim.x;
	if (j < N - 15)
	{
		int idx = LMSE_CONST + j*N;
		//for (int i = 0; i < N - 15; i++)
		int i = threadIdx.y + blockIdx.y * blockDim.y;
		if (i < N - 15)
		{
			int idx = (LMSE_CONST + j*N) + i - 1;
			float temp_dbl = 0;
			int sub_idx = idx - (7 + 7 * N);
			for (int jB = j; jB < j + 16; ++jB)
			{
				for (int iB = i; iB < i + 16; ++iB)
				{
					temp_dbl += reflut[sub_idx];
					++sub_idx;
				}
				sub_idx += N - 16;
			}
			D[idx] = temp_dbl / 256.0;
		}
	}
}
//=============================================================================
//=============================================================================
__global__ void fast_hi_stats_kernel1(float* absRefs, float* absDsts, float* outStd, float* outStdMod, float* outMean, float* ref_img, float* dst_img, float* TMP)
{
	float mean, mean2, stdev;
	int i, j, iB, jB;

	i = 4 * (threadIdx.x + blockIdx.x * blockDim.x);
	if (i < 497) //512-15=497
	{
		j = 4 * (threadIdx.y + blockIdx.y * blockDim.y);
		if (j < 497)
		{
			//Traverse through and get mean
			mean = 0;
			mean2 = 0;
			for (iB = i; iB < i + 16; iB++)
			{
				for (jB = j; jB < j + 16; jB++)
				{
					mean += absRefs[(iB * 512) + jB];
					mean2 += absDsts[(iB * 512) + jB];
				}
			}
			mean = mean / 256.0f;
			mean2 = mean2 / 256.0f;

			//Traverse through and get stdev
			stdev = 0;
			for (iB = i; iB < i + 16; iB++)
			{
				for (jB = j; jB < j + 16; jB++)
				{
					float temp = absRefs[(iB * 512) + jB] - mean;
					stdev += temp*temp;
				}
			}
			stdev = sqrt(stdev / 255.0);//MATLAB's std is a bit different                       

			for (iB = i; iB < i + 4; iB++)
			{
				for (jB = j; jB < j + 4; jB++)
				{
					outMean[(iB * 512) + jB] = mean2;// mean of reference
					outStd[(iB * 512) + jB] = stdev;// stdev of dst
				}
			}
		} // end for over j
	} // end for over i

	//====================================================================
	//Modified STD
	//for (i = 0; i < 512 - 15; i += 4)
	i = 4 * (threadIdx.x + blockIdx.x * blockDim.x);
	if (i < 497) //512-15=497
	{
		//for (j = 0; j < 512 - 15; j += 4)
		j = 4 * (threadIdx.y + blockIdx.y * blockDim.y);
		if (j < 497) //512-15=497
		{
			//Traverse through and get mean
			mean = 0;

			for (iB = i; iB < i + 8; iB++)
			{
				for (jB = j; jB < j + 8; jB++)
				{
					mean += absDsts[(iB * 512) + jB];
				}
			}
			mean = mean / 64.0f;

			//Traverse through and get stdev
			stdev = 0;
			for (iB = i; iB < i + 8; iB++)
			{
				for (jB = j; jB < j + 8; jB++)
				{
					float temp = absDsts[(iB * 512) + jB] - mean;
					stdev += (temp)*(temp);
				}
			}
			stdev = sqrt(stdev / 63.0);//MATLAB's std is a bit different                       

			for (iB = i; iB < i + 4; iB++)
			{
				for (jB = j; jB < j + 4; jB++)
				{
					TMP[(iB * 512) + jB] = stdev;// stdev of ref 
					outStdMod[(iB * 512) + jB] = stdev;
				}
			}
		}
	}
}

__global__ void fast_hi_stats_kernel2(float* absRefs, float* absDsts, float* outStd, float* outStdMod, float* outMean, float* ref_img, float* dst_img, float* TMP)
{
	//Declarations
	float mean, mean2, stdev;
	//float* TMP = (float *)malloc(N * N*sizeof(float));
	int i, j, iB, jB;
	//for (i = 0; i < 512 - 15; i += 4)
	i = 4 * (threadIdx.x + blockIdx.x * blockDim.x);
	if (i < 497) //512-15=497
	{
		//for (j = 0; j < 512 - 15; j += 4)
		j = 4 * (threadIdx.y + blockIdx.y * blockDim.y);
		if (j < 497) //512-15=497
		{
			mean = TMP[(i * 512) + j];
			for (iB = i; iB < i + 8; iB += 5)
			{
				for (jB = j; jB < j + 8; jB += 5)
				{
					if (iB < 512 - 15 && jB < 512 - 15 && mean > TMP[(iB * 512) + jB])
						mean = TMP[(iB * 512) + jB];
				}
			}

			for (iB = i; iB < i + 4; iB++)
			{
				for (jB = j; jB < j + 4; jB++)
				{
					outStdMod[(iB * 512) + jB] = mean;
				}
			}
		}
	}
}
//=============================================================================
__global__ void product_array_kernel(float* out, float* in1, float* in2)
{
	//for (int i = BLOCK_SIZE; i < N - BLOCK_SIZE - 1; i++)
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= BLOCK_SIZE && i < N - BLOCK_SIZE - 1)
	{
		//for (int j = BLOCK_SIZE; j < N - BLOCK_SIZE - 1; j++)
		int j = threadIdx.y + blockIdx.y * blockDim.y;
		if (j >= BLOCK_SIZE && j < N - BLOCK_SIZE - 1)
			out[i*N + j] = in1[i*N + j] * in2[i*N + j];
	}
}
//=============================================================================
cudaError_t kernel_wrapper(const cv::Mat &mat_ref, const cv::Mat &mat_dst)
{
	int  GPU_N, device_num_used;
	cudaGetDeviceCount(&GPU_N);

	//OSU Workstation : 0 = Tesla, 1 = Titan1, 2 = Titan2
	//ASU Workstation: 0 = Tesla, 1 = Quadro (don't use)
	device_num_used = 1;
	cudaError_t cudaStatus = cudaSetDevice(device_num_used);	// OSU Workstation: 0=Tesla, 1=Titan1, 2=Titan2
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaSetDevice failed!");

	// Allocate Page-locked (Pinned) HOST-memory
	float* h_img_ref_float;				cudaMallocHost(&h_img_ref_float, REAL_SIZE);
	float* h_img_dst_float;				cudaMallocHost(&h_img_dst_float, REAL_SIZE);
	cufftComplex* h_ref_cufft;		cudaMallocHost(&h_ref_cufft, COMPLEX_SIZE);
	cufftComplex* h_dst_cufft;		cudaMallocHost(&h_dst_cufft, COMPLEX_SIZE);
	float* h_eta;									cudaMallocHost(&h_eta, REAL_SIZE);
	float* h_product;							cudaMallocHost(&h_product, REAL_SIZE);


	// Allocate DEVICE memory -- Appearance (Hi-Index)
	float* d_img_ref_float;		cudaMalloc((void **)&d_img_ref_float, REAL_SIZE);
	float* d_img_dst_float;		cudaMalloc((void **)&d_img_dst_float, REAL_SIZE);
	float* d_L_hat_ref;				cudaMalloc((void **)&d_L_hat_ref, REAL_SIZE);
	float* d_L_hat_dst;				cudaMalloc((void **)&d_L_hat_dst, REAL_SIZE);
	cufftComplex* d_L_hat_ref_complex;  cudaMalloc((void **)&d_L_hat_ref_complex, COMPLEX_SIZE);
	cufftComplex* d_L_hat_dst_complex;  cudaMalloc((void **)&d_L_hat_dst_complex, COMPLEX_SIZE);
	float* d_CSF;							cudaMalloc((void **)&d_CSF, REAL_SIZE);
	float* d_xPlane;				cudaMalloc((void **)&d_xPlane, 512 * sizeof(float));
	float* d_yPlane;				cudaMalloc((void **)&d_yPlane, 512 * sizeof(float));
	float* d_I_prime_org;	cudaMalloc((void **)&d_I_prime_org, REAL_SIZE);
	float* d_I_prime_dst;	cudaMalloc((void **)&d_I_prime_dst, REAL_SIZE);
	float* d_I_prime_err;	cudaMalloc((void **)&d_I_prime_err, REAL_SIZE);
	float* d_outStd;			cudaMalloc((void **)&d_outStd, REAL_SIZE);
	float* d_outStdMod;		cudaMalloc((void **)&d_outStdMod, REAL_SIZE);
	float* d_outMean;			cudaMalloc((void **)&d_outMean, REAL_SIZE);
	float* d_reflut;			cudaMalloc((void **)&d_reflut, REAL_SIZE);
	float* d_TEMP;				cudaMalloc((void **)&d_TEMP, REAL_SIZE);
	float* d_zeta;				cudaMalloc((void **)&d_zeta, REAL_SIZE);
	float* d_lmse;			cudaMalloc((void **)&d_lmse, REAL_SIZE);
	float* d_product;			cudaMalloc((void **)&d_product, REAL_SIZE);

	// Allocate DEVICE memory -- Appearance (Lo-Index)
	cufftComplex* d_ref_cufft; cudaMalloc((void **)&d_ref_cufft, COMPLEX_SIZE);
	cufftComplex* d_dst_cufft; cudaMalloc((void **)&d_dst_cufft, COMPLEX_SIZE);
	float* d_logGabor;			cudaMalloc((void **)&d_logGabor, REAL_SIZE);
	cufftComplex* d_ref_c;	cudaMalloc((void **)&d_ref_c, COMPLEX_SIZE);
	cufftComplex* d_dst_c;	cudaMalloc((void **)&d_dst_c, COMPLEX_SIZE);
	float* d_ref_c_mag;		cudaMalloc((void **)&d_ref_c_mag, REAL_SIZE);
	float* d_dst_c_mag;		cudaMalloc((void **)&d_dst_c_mag, REAL_SIZE);
	float* d_ref_Std;		cudaMalloc((void **)&d_ref_Std, REAL_SIZE);
	float* d_ref_Skw;		cudaMalloc((void **)&d_ref_Skw, REAL_SIZE);
	float* d_ref_Krt;		cudaMalloc((void **)&d_ref_Krt, REAL_SIZE);
	float* d_dst_Std;		cudaMalloc((void **)&d_dst_Std, REAL_SIZE);
	float* d_dst_Skw;		cudaMalloc((void **)&d_dst_Skw, REAL_SIZE);
	float* d_dst_Krt;		cudaMalloc((void **)&d_dst_Krt, REAL_SIZE);
	float* d_eta;				cudaMalloc((void **)&d_eta, REAL_SIZE);


	// Creates stream and cuFFT plans and set them in different streams
	const int NUM_STREAMS = 10;
	cudaStream_t stream[NUM_STREAMS];
	cufftHandle* fftPlan = (cufftHandle*)malloc(sizeof(cufftHandle)*NUM_STREAMS);
	for (int i = 0; i < NUM_STREAMS; i++)
	{
		cudaStreamCreate(&stream[i]);
		cufftPlan2d(&fftPlan[i], N, N, CUFFT_C2C);
		cufftSetStream(fftPlan[i], stream[i]);
	}


	// Configuration Parameters - EXPERIMENT WITH THESE TO DETERMINE OPTIMAL VALUES!!!!!!!
	//(Launch most kernels as 4-dimensional functions - with overall 512x512 threads in grid):
	dim3 gridSize(32, 32, 1);
	dim3 blockSize(16, 16, 1);
	dim3 fftShift_grid_size(16, 8, 1);
	dim3 fftShift_block_size(32, 32, 1);

	// The lo-stats kernels only need to be launced as (512^2)/4 threads due to 
	//	the 4 pixel sliding window (i.e. only 12 pixel overlap in neighboring 16x16 blocks)
	dim3 loStats_Grid_size(8, 8);
	dim3 loStats_Block_size(16, 16);

	//----------------------------------------------------------------------------
	// Program initialization complete - Begin main program body:
	//----------------------------------------------------------------------------

	std::cout << "Beginning Detection Stage" << std::endl;
	// Start CPU Timing
	int itteration_num = 1;
	double timing_sum = 0.0;
	LARGE_INTEGER start_CPU, end_CPU, frequency_CPU;
	float milliseconds_CPU;
	QueryPerformanceFrequency(&frequency_CPU);
	QueryPerformanceCounter(&start_CPU);

	for (int timing_idx = 0; timing_idx < itteration_num; ++timing_idx)
	{

		// Begin NVTX Marker:
		nvtxRangePushA("CUDA-MAD");


		// Build CSF on Device
		yPlane_CSF_kernel << < 1, 1, 0, stream[1] >> >(d_yPlane);
		xPlane_CSF_kernel << < 1, 1, 0, stream[1] >> >(d_xPlane);
		build_CSF_kernel << < gridSize, blockSize, 0, stream[1] >> >(d_CSF, d_yPlane, d_xPlane);
		fftShift_kernel << < fftShift_grid_size, fftShift_block_size, 0, stream[1] >> >(d_CSF);


		
		

		// Linearize REAL image data and copy data from HOST -> DEVICE
		nvtxRangePushA("Linearize ref");// Begin NVTX Marker for Linearize ref
		linearize_and_cast_from_Mat_to_float(mat_ref, h_img_ref_float);
		nvtxRangePop();		// End NVTX Marker for Linearize ref
		
		cudaMemcpyAsync(d_img_ref_float, h_img_ref_float, REAL_SIZE, cudaMemcpyDeviceToHost, stream[1]); //DEVICE -> HOST
		//cudaMemcpy(d_img_ref_float, h_img_ref_float, REAL_SIZE, cudaMemcpyDeviceToHost); //DEVICE -> HOST
		map_to_luminance_domain_kernel1 << < gridSize, blockSize, 0, stream[1] >> >(d_img_ref_float, d_L_hat_ref);
		R2C_kernel << < gridSize, blockSize, 0, stream[1] >> >(d_L_hat_ref, d_L_hat_ref_complex);
		cufftExecC2C(fftPlan[1], (cufftComplex *)d_L_hat_ref_complex, (cufftComplex *)d_L_hat_ref_complex, CUFFT_FORWARD);
		pointWise_complex_matrix_mult_kernel_2d << < gridSize, blockSize, 0, stream[1] >> >(d_L_hat_ref_complex, d_CSF, d_L_hat_ref_complex);
		cufftExecC2C(fftPlan[1], (cufftComplex *)d_L_hat_ref_complex, (cufftComplex *)d_L_hat_ref_complex, CUFFT_INVERSE);
		real_kernel << <gridSize, blockSize, 0, stream[1] >> >(d_L_hat_ref_complex, d_I_prime_org);

		nvtxRangePushA("Linearize dst");// Begin NVTX Marker for Linearize ref
		linearize_and_cast_from_Mat_to_float(mat_dst, h_img_dst_float);
		nvtxRangePop();		// End NVTX Marker for Linearize ref

		cudaMemcpyAsync(d_img_dst_float, h_img_dst_float, REAL_SIZE, cudaMemcpyDeviceToHost, stream[1]); //DEVICE -> HOST
		//cudaMemcpy(d_img_dst_float, h_img_dst_float, REAL_SIZE, cudaMemcpyDeviceToHost); //DEVICE -> HOST
		map_to_luminance_domain_kernel1 << < gridSize, blockSize, 0, stream[1] >> >(d_img_dst_float, d_L_hat_dst);
		R2C_kernel << < gridSize, blockSize, 0, stream[1] >> >(d_L_hat_dst, d_L_hat_dst_complex);
		cufftExecC2C(fftPlan[1], (cufftComplex *)d_L_hat_dst_complex, (cufftComplex *)d_L_hat_dst_complex, CUFFT_FORWARD);
		pointWise_complex_matrix_mult_kernel_2d << < gridSize, blockSize, 0, stream[1] >> >(d_L_hat_dst_complex, d_CSF, d_L_hat_dst_complex);
		cufftExecC2C(fftPlan[1], (cufftComplex *)d_L_hat_dst_complex, (cufftComplex *)d_L_hat_dst_complex, CUFFT_INVERSE);
		real_kernel << <gridSize, blockSize, 0, stream[1] >> >(d_L_hat_dst_complex, d_I_prime_dst);


		// Detection Statistics
		square_of_difference_kernel << <gridSize, blockSize, 0, stream[1] >> >(d_img_ref_float, d_img_dst_float, d_reflut);
		LMSE_map_kernel << <gridSize, blockSize, 0, stream[1] >> >(d_reflut, d_lmse);
		error_img_kernel << <gridSize, blockSize, 0, stream[1] >> >(d_I_prime_org, d_I_prime_dst, d_I_prime_err);
		fast_hi_stats_kernel1 << <loStats_Grid_size, loStats_Block_size, 0, stream[1] >> >(d_I_prime_err, d_I_prime_org, d_outStd, d_outStdMod, d_outMean, d_img_ref_float, d_img_dst_float, d_TEMP);
		fast_hi_stats_kernel2 << <loStats_Grid_size, loStats_Block_size, 0, stream[1] >> >(d_I_prime_err, d_I_prime_org, d_outStd, d_outStdMod, d_outMean, d_img_ref_float, d_img_dst_float, d_TEMP);
		zeta_map_kernel << <gridSize, blockSize, 0, stream[1] >> >(d_outMean, d_outStd, d_outStdMod, d_zeta);
		product_array_kernel << <gridSize, blockSize, 0, stream[1] >> >(d_product, d_zeta, d_lmse); // Product inside summation in MAD eq. 7
		//cudaMemcpyAsync(h_product, d_product, REAL_SIZE, cudaMemcpyDeviceToHost, stream[1]); //DEVICE -> HOST

		//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

		// Begin Gabor Filterbank:

		//- - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - 

		// Exectute "in-place" C2C 2D-DFT of REF (used in the LEFT side of the Gabor Filterbank)
		R2C_kernel << <gridSize, blockSize, 0, stream[1] >> >(d_img_ref_float, d_ref_cufft);
		R2C_kernel << <gridSize, blockSize, 0, stream[1] >> >(d_img_dst_float, d_dst_cufft);
		cufftExecC2C(fftPlan[1], (cufftComplex *)d_ref_cufft, (cufftComplex *)d_ref_cufft, CUFFT_FORWARD);
		cufftExecC2C(fftPlan[1], (cufftComplex *)d_dst_cufft, (cufftComplex *)d_dst_cufft, CUFFT_FORWARD);

		float scale[5] = { 0.5, 0.75, 1, 5, 6 };
		for (int o = 0; o < 4; o++)
		{
			for (int s = 0; s < 5; s++)
			{
				buildGabor << < gridSize, blockSize, 0, stream[1] >> >(d_logGabor, o, s);
				fftShift_kernel << < fftShift_grid_size, fftShift_block_size, 0, stream[1] >> >(d_logGabor);

				pointWise_complex_matrix_mult_kernel_2d << < gridSize, blockSize, 0, stream[1] >> >(d_ref_cufft, d_logGabor, d_ref_c);
				pointWise_complex_matrix_mult_kernel_2d << < gridSize, blockSize, 0, stream[1] >> >(d_dst_cufft, d_logGabor, d_dst_c);

				cufftExecC2C(fftPlan[1], (cufftComplex *)d_ref_c, (cufftComplex *)d_ref_c, CUFFT_INVERSE);
				cufftExecC2C(fftPlan[1], (cufftComplex *)d_dst_c, (cufftComplex *)d_dst_c, CUFFT_INVERSE);


				magnitude_kernel << <gridSize, blockSize, 0, stream[1] >> >(d_ref_c, d_ref_c_mag);
				magnitude_kernel << <gridSize, blockSize, 0, stream[1] >> >(d_dst_c, d_dst_c_mag);


				fast_lo_stats_kernel << <loStats_Grid_size, loStats_Block_size, 0, stream[1] >> >(d_ref_c_mag, d_ref_Std, d_ref_Skw, d_ref_Krt);
				fast_lo_stats_kernel << <loStats_Grid_size, loStats_Block_size, 0, stream[1] >> >(d_dst_c_mag, d_dst_Std, d_dst_Skw, d_dst_Krt);

				delta_stats_kernel << <gridSize, blockSize, 0, stream[1] >> >(d_ref_Std, d_ref_Skw, d_ref_Krt,
					d_dst_Std, d_dst_Skw, d_dst_Krt, scale[s] / 13.25f, d_eta);
			}
		}
		// Copy final eta map back to HOST for collapse (NEEDS TO BE DONE VIA REDUCTION)!
		//cudaMemcpyAsync(h_eta, d_eta, REAL_SIZE, cudaMemcpyDeviceToHost, stream[1]); //DEVICE -> HOST


		// Host Code waits here on memcpy
		cudaMemcpy(h_product, d_product, REAL_SIZE, cudaMemcpyDeviceToHost); //DEVICE -> HOST

		// Collapse the visibility-weighted local MSE via L2-norm (MAD eq. 7)
		nvtxRangePushA("CPU Detection Map Collapse");
		float d_detect = reduce_sum_of_squares_2D_CPU(h_product, BLOCK_SIZE, N - BLOCK_SIZE - 1);
		d_detect = sqrt(d_detect) / sqrt(229441.0f);   // Number of itterations in loop: counter = 229441
		d_detect = d_detect * 200;
		nvtxRangePop();

		cudaMemcpy(h_eta, d_eta, REAL_SIZE, cudaMemcpyDeviceToHost); //DEVICE -> HOST
		nvtxRangePushA("CPU Appearance Map Collapse");
		float d_appear = reduce_sum_of_squares_2D_CPU(h_eta, BLOCK_SIZE, N - BLOCK_SIZE);
		d_appear = sqrt(d_appear) / 479.0f;
		nvtxRangePop();

		float beta1 = 0.467;
		float beta2 = 0.130;
		float alpha = 1 / (1 + beta1*pow(d_detect, beta2));
		float MAD = pow(d_detect, alpha)*pow(d_appear, 1 - alpha);

		// End NVTX Marker for CUDA-MAD:
		nvtxRangePop();

		// End CPU Timing
		QueryPerformanceCounter(&end_CPU);
		milliseconds_CPU = (end_CPU.QuadPart - start_CPU.QuadPart) *
			1000.0 / frequency_CPU.QuadPart;
		timing_sum += milliseconds_CPU;

		std::cout << "Hi-Index d_detect = " << d_detect << std::endl;
		std::cout << "Lo-Index d_appear = " << d_appear << std::endl;
		std::cout << "\nMAD = " << MAD << std::endl;
	} // End timing loop


	fprintf(stderr, "\nTime  = %.3f ms\n", timing_sum / double(itteration_num));
	//getchar();

	//----------------------------------------------------------------------------
	// Main program body complete - Perform closing operations:
	//----------------------------------------------------------------------------

	//Error:
	// De-allocate memory here...

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaDeviceReset();

	return cudaStatus;
}
//=============================================================================
void linearize_and_cast_from_Mat_to_float(const cv::Mat& mat_in, float* h_float)
{
	for (int row = 0; row < 512; row++)
	for (int col = 0; col < 512; col++)
		h_float[row * 512 + col] = static_cast<float>(mat_in.at<unsigned char>(row, col));
}
//=============================================================================
void product_array_CPU(float* out, float* in1, float* in2)
{
	for (int i = BLOCK_SIZE; i < N - BLOCK_SIZE - 1; i++)
	{
		for (int j = BLOCK_SIZE; j < N - BLOCK_SIZE - 1; j++)
			out[i*N + j] = in1[i*N + j] * in2[i*N + j];
	}
}
//=============================================================================
float reduce_sum_of_squares_2D_CPU(float* in, const int INSIDE_BOUND, const int OUTSIDE_BOUND)
{
	float sum = 0.0f;
	for (int i = INSIDE_BOUND; i < OUTSIDE_BOUND; i++)
	{
		for (int j = INSIDE_BOUND; j < OUTSIDE_BOUND; j++)
			sum += in[i*N + j] * in[i*N + j];
	}
	return sum;
}
//=============================================================================
void write_to_file_DEBUG(float* w, const int SIZE)
{
	std::ofstream outFile;
	outFile.open("TEST.txt");
	for (int i = 0; i < SIZE; i++)  // Itterate over rows
	{
		for (int j = 0; j < SIZE; j++) // Itterate over cols
			outFile << w[i * SIZE + j] << " ";
		if (i != SIZE - 1)
			outFile << ";\n";
	}
	outFile.close();
}
//=============================================================================
