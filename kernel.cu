// kernel.cu
//-------------------------------------------------------------------
#include "header.h"
//-------------------------------------------------------------------

// CUDA Kernel
__global__ void colorKernel_GPU(	const unsigned char* d_in,
																	unsigned char* d_out,
																	const int num_cols,
																	const int num_rows,
																	const int colorWidthStep,
																	const int grayWidthStep)
{
	//2D Index of current thread
	const int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
	const int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;

	//Only valid threads perform memory I/O
	if ((colIdx < num_cols) && (rowIdx < num_rows))
	{
		//Location of colored pixel in input
		const int color_idx = rowIdx * colorWidthStep + (3 * colIdx);

		//Location of gray pixel in output
		const int gray_idx = rowIdx * grayWidthStep + (1 * colIdx);

		const unsigned char b = d_in[color_idx + 0];
		const unsigned char g = d_in[color_idx + 1];
		const unsigned char r = d_in[color_idx + 2];

		const float B_processed = b * 0.11f;//b * 1.0f; // Blue Channel is unchanged
		const float G_processed = g * 0.59f;//g * 1.0f; // Green Channel is unchanged
		const float R_processed = r * 0.3f; //r * 0.0f; // Turn off Red Channel


		d_out[gray_idx] = static_cast<unsigned char>(B_processed + G_processed + R_processed);
		//d_out[color_idx + 0] = static_cast<unsigned char>(B_processed);
		//d_out[color_idx + 1] = static_cast<unsigned char>(G_processed);
		//d_out[color_idx + 2] = static_cast<unsigned char>(R_processed);

	}
}
//-------------------------------------------------------------------

//=============================================================================
void linearize_and_cast_from_Mat_to_float(const cv::Mat& mat_in, float* h_float)
{
	for (int row = 0; row < 512; row++)
	{
		for (int col = 0; col < 512; col++)
		{
			h_float[row * 512 + col] = static_cast<float>(mat_in.at<unsigned char>(row, col));
		}
	}
}
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
	float a = d_inverse_complex[grid_index].x / (IMG_SIZE);
	float b = d_inverse_complex[grid_index].y / (IMG_SIZE);

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
	float xVal_local[256] = { 0 };

	float mean=0,  stdev = 0,skw = 0, krt=0,  stmp = 0;
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
				stdev = stdev + (xV_mean * xV_mean);
				skw = skw + (xV_mean * xV_mean * xV_mean);
				krt = krt + (xV_mean * xV_mean * xV_mean * xV_mean);
			}
			stmp = sqrt(stdev / 256.0f);
			stdev = sqrt(stdev / 255.0f);//MATLAB's std is a bit different
			/*
			if (i + j <5)
			{
				printf("%f %f %f %f %f \n", stdev,stmp,stdev, skw, krt);
			}
			*/
			if (stmp != 0){
				skw = (skw / 256.0f) / ((stmp)*(stmp)*(stmp));
				krt = (krt / 256.0f) / ((stmp)*(stmp)*(stmp)*(stmp));			
			}
			else{
				skw = 0;
				krt = 0;
			}
			/*
			if (i + j <5)
			{
				printf("%f %f \n", skw, krt);
			}*/
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
					// ADDED IF-ELSE STATEMENT HERE:
					if (i > 500 || j > 500)
					{ 
						outStd[(iB * 512) + jB] = 0;
						outSkw[(iB * 512) + jB] = 0;
						outKrt[(iB * 512) + jB] = 0;
					}
					else
					{
						outStd[(iB * 512) + jB] = stdev;
						outSkw[(iB * 512) + jB] = skw;
						outKrt[(iB * 512) + jB] = krt;
					}					 
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
//=============================================================================
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
__inline float reduce_sum_of_squares_2D_CPU(float in[], const int INSIDE_BOUND, const int OUTSIDE_BOUND)
{
	float sum = 0.0f;
	int i = 0, j = 0;
	for ( i = INSIDE_BOUND; i < OUTSIDE_BOUND; i++)
	{
		for ( j = INSIDE_BOUND; j < OUTSIDE_BOUND; j++)
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

// CPU Function
void colorFunction_CPU(const cv::Mat &h_in, cv::Mat &h_out, const int num_rows, const int num_cols)
{
	// Copy contents from input image into output image for processing
	//h_out = h_in.clone();

	// [B1 G1 R1 B2 G2 R2 ... BN GN RN] => Memory Stride of 3
	for (int r = 0; r < num_rows; ++r)
	{
		for (int c = 0; c < 3 * num_cols; c += 3)
		{
			//h_out.at<unsigned char>(r, c + 0) = 0; // Blue Channel is unchanged
			//h_out.at<unsigned char>(r, c + 1) = 0; // Green Channel is unchanged
			//h_out.at<unsigned char>(r, c + 2) = 0; // Turn off Red Channel
			float b_processed = h_in.at<unsigned char>(r, c + 0)*0.11f;
			float g_processed =	h_in.at<unsigned char>(r, c + 1)*0.59f;
			float r_processed = h_in.at<unsigned char>(r, c + 2)*0.3f;
			h_out.at<unsigned char>(r, c/3) = 
				static_cast<unsigned char>(b_processed + g_processed + r_processed);
		}
	}
}
//-------------------------------------------------------------------
//-------------------------------------------------------------------

// Kernel Wrapper
struct out_data kernel_wrapper(const cv::Mat& mat_ref, const cv::Mat& mat_dst)
{
	float h_img_ref_float[IMG_SIZE];
	linearize_and_cast_from_Mat_to_float(mat_ref, h_img_ref_float);

	float h_img_dst_float[IMG_SIZE];
	linearize_and_cast_from_Mat_to_float(mat_dst, h_img_dst_float);

	// Allocate Page-locked (Pinned) HOST-memory
	//float h_I_prime_org[REAL_SIZE];
	//float h_I_prime_err[REAL_SIZE];
	//cufftComplex h_ref_cufft[COMPLEX_SIZE];

	//cufftComplex h_dst_cufft[COMPLEX_SIZE];

	//float h_reflut[REAL_SIZE];
	//float h_lmse[REAL_SIZE];
	//float h_zeta[REAL_SIZE];
	float *h_eta = (float*)malloc(REAL_SIZE);
	float *h_product = (float*)malloc(REAL_SIZE);

		// Allocate DEVICE memory -- Appearance (Hi-Index)
		float* d_img_ref_float;
	gpuErrchk(cudaMalloc((void **)&d_img_ref_float, REAL_SIZE))

		float* d_img_dst_float;
	gpuErrchk(cudaMalloc((void **)&d_img_dst_float, REAL_SIZE))

		float* d_L_hat_ref;
	gpuErrchk(cudaMalloc((void **)&d_L_hat_ref, REAL_SIZE))

		float* d_L_hat_dst;
	gpuErrchk(cudaMalloc((void **)&d_L_hat_dst, REAL_SIZE))

		cufftComplex* d_L_hat_ref_complex;
	gpuErrchk(cudaMalloc((void **)&d_L_hat_ref_complex, COMPLEX_SIZE))

		cufftComplex* d_L_hat_dst_complex;
	gpuErrchk(cudaMalloc((void **)&d_L_hat_dst_complex, COMPLEX_SIZE))

		float* d_CSF;
	gpuErrchk(cudaMalloc((void **)&d_CSF, REAL_SIZE))

		float* d_xPlane;
	gpuErrchk(cudaMalloc((void **)&d_xPlane, 512 * sizeof(float)))
		float* d_yPlane;
	gpuErrchk(cudaMalloc((void **)&d_yPlane, 512 * sizeof(float)))

		float* d_I_prime_org;
	gpuErrchk(cudaMalloc((void **)&d_I_prime_org, REAL_SIZE))
		float* d_I_prime_dst;
	gpuErrchk(cudaMalloc((void **)&d_I_prime_dst, REAL_SIZE))

		float* d_I_prime_err;
	gpuErrchk(cudaMalloc((void **)&d_I_prime_err, REAL_SIZE))
		float* d_outStd;
	gpuErrchk(cudaMalloc((void **)&d_outStd, REAL_SIZE))

		float* d_outStdMod;
	gpuErrchk(cudaMalloc((void **)&d_outStdMod, REAL_SIZE))
		float* d_outMean;
	gpuErrchk(cudaMalloc((void **)&d_outMean, REAL_SIZE))
		float* d_reflut;
	gpuErrchk(cudaMalloc((void **)&d_reflut, REAL_SIZE))
		float* d_TEMP;
	gpuErrchk(cudaMalloc((void **)&d_TEMP, REAL_SIZE))
		float* d_zeta;
	gpuErrchk(cudaMalloc((void **)&d_zeta, REAL_SIZE))
		float* d_lmse;
	gpuErrchk(cudaMalloc((void **)&d_lmse, REAL_SIZE))

		float* d_product;
	gpuErrchk(cudaMalloc((void **)&d_product, REAL_SIZE))

		// Allocate DEVICE memory -- Appearance (Lo-Index)
		cufftComplex* d_ref_cufft;
	gpuErrchk(cudaMalloc((void **)&d_ref_cufft, COMPLEX_SIZE))
		cufftComplex* d_dst_cufft;
	gpuErrchk(cudaMalloc((void **)&d_dst_cufft, COMPLEX_SIZE))
		float* d_logGabor;
	gpuErrchk(cudaMalloc((void **)&d_logGabor, REAL_SIZE))
		cufftComplex* d_ref_c;
	gpuErrchk(cudaMalloc((void **)&d_ref_c, COMPLEX_SIZE))
		cufftComplex* d_dst_c;
		float dummy_array[IMG_SIZE] = { 0 };
	gpuErrchk(cudaMalloc((void **)&d_dst_c, COMPLEX_SIZE))
	cudaMemcpy(d_dst_c, dummy_array, REAL_SIZE, cudaMemcpyHostToDevice);
		float* d_ref_c_mag;
	gpuErrchk(cudaMalloc((void **)&d_ref_c_mag, REAL_SIZE))
	cudaMemcpy(d_ref_c_mag, dummy_array, REAL_SIZE, cudaMemcpyHostToDevice);
		float* d_dst_c_mag;
	gpuErrchk(cudaMalloc((void **)&d_dst_c_mag, REAL_SIZE))
	cudaMemcpy(d_dst_c_mag, dummy_array, REAL_SIZE, cudaMemcpyHostToDevice);
		float* d_ref_Std;
	gpuErrchk(cudaMalloc((void **)&d_ref_Std, REAL_SIZE))		
	cudaMemcpy(d_ref_Std, dummy_array, REAL_SIZE, cudaMemcpyHostToDevice);
		float* d_ref_Skw;
	gpuErrchk(cudaMalloc((void **)&d_ref_Skw, REAL_SIZE))
	cudaMemcpy(d_ref_Skw, dummy_array, REAL_SIZE, cudaMemcpyHostToDevice);
		float* d_ref_Krt;
	gpuErrchk(cudaMalloc((void **)&d_ref_Krt, REAL_SIZE))
	cudaMemcpy(d_ref_Krt, dummy_array, REAL_SIZE, cudaMemcpyHostToDevice);
		float* d_dst_Std;
	gpuErrchk(cudaMalloc((void **)&d_dst_Std, REAL_SIZE))
	cudaMemcpy(d_dst_Std, dummy_array, REAL_SIZE, cudaMemcpyHostToDevice);
		float* d_dst_Skw;
	gpuErrchk(cudaMalloc((void **)&d_dst_Skw, REAL_SIZE))
	cudaMemcpy(d_dst_Skw, dummy_array, REAL_SIZE, cudaMemcpyHostToDevice);
		float* d_dst_Krt;
	gpuErrchk(cudaMalloc((void **)&d_dst_Krt, REAL_SIZE))
	cudaMemcpy(d_dst_Krt, dummy_array, REAL_SIZE, cudaMemcpyHostToDevice);
		float* d_eta;
	gpuErrchk(cudaMalloc((void **)&d_eta, REAL_SIZE))
	cudaMemcpy(d_eta, dummy_array, REAL_SIZE, cudaMemcpyHostToDevice);
	
	// Creates stream and cuFFT plans and set them in different streams
	const int NUM_STREAMS = 10;
	cudaStream_t stream[NUM_STREAMS];
	cufftHandle* fftPlan = NULL;
	fftPlan = (cufftHandle*)malloc(sizeof(cufftHandle)*NUM_STREAMS);
	if (NULL == fftPlan)
	{
		printf("Error malloc\n");
	}
	//for (int i = 0; i < NUM_STREAMS; i++)
	//{
		gpuErrchk(cudaStreamCreate(&stream[1]))
		cufftPlan2d(&fftPlan[1], N, N, CUFFT_C2C);
		cufftSetStream(fftPlan[1], stream[1]);
	//}
	// Configuration Parameters - EXPERIMENT WITH THESE TO DETERMINE OPTIMAL VALUES!!!!!!!
	//(Launch most kernels as 4-dimensional functions - with overall 512x512 threads in grid):
	dim3 gridSize(32, 32, 1);
	dim3 blockSize(16, 16, 1);
	dim3 fftShift_grid_size(16, 8, 1);
	dim3 fftShift_block_size(32, 32, 1);

	// The lo-stats kernels only need to be launced as (512^2)/4 threads due to 
	//	the 4 pixel sliding window (i.e. only 12 pixel overlap in neighboring 16x16 blocks)
	
	// I'm only barely covering 125^2
	dim3 loStats_Grid_size(8, 8);
	dim3 loStats_Block_size(16, 16);
	
	gpuErrchk(cudaMemcpy(d_img_ref_float, h_img_ref_float, REAL_SIZE, cudaMemcpyHostToDevice)) // HOST -> DEVICE
	gpuErrchk(cudaMemcpy(d_img_dst_float, h_img_dst_float, REAL_SIZE, cudaMemcpyHostToDevice)) // HOST -> DEVICE

		// Map to luminance domain
	map_to_luminance_domain_kernel1 << < gridSize, blockSize, 0, stream[1] >> >(d_img_ref_float, d_L_hat_ref);
	cudaCheckError()
	map_to_luminance_domain_kernel1 << < gridSize, blockSize, 0, stream[1] >> >(d_img_dst_float, d_L_hat_dst);
	cudaCheckError()

	// CAN MOVE THIS ANYWHERE!!!!!
	// Launch Kernel to take the square of the differnce of the original image 
	square_of_difference_kernel << <gridSize, blockSize, 0, stream[1] >> >(d_img_ref_float, d_img_dst_float, d_reflut);
	cudaCheckError()
	// Filter L_hat_dst
	R2C_kernel << < gridSize, blockSize, 0, stream[1] >> >(d_L_hat_ref, d_L_hat_ref_complex);
	cudaCheckError()
	R2C_kernel << < gridSize, blockSize, 0, stream[1] >> >(d_L_hat_dst, d_L_hat_dst_complex);
	cudaCheckError()

	// Build CSF on Device
	yPlane_CSF_kernel << < 1, 1, 0, stream[1] >> >(d_yPlane);
	cudaCheckError()
	xPlane_CSF_kernel << < 1, 1, 0, stream[1] >> >(d_xPlane);
	cudaCheckError()
	build_CSF_kernel << < gridSize, blockSize, 0, stream[1] >> >(d_CSF, d_yPlane, d_xPlane);
	cudaCheckError()
	fftShift_kernel << < fftShift_grid_size, fftShift_block_size, 0, stream[1] >> >(d_CSF);
	cudaCheckError()
	// Exectute "in-place" Forward C2C 2D-FFT
	cufftExecC2C(fftPlan[1], (cufftComplex *)d_L_hat_ref_complex, (cufftComplex *)d_L_hat_ref_complex, CUFFT_FORWARD);
	cudaCheckError()
	cufftExecC2C(fftPlan[1], (cufftComplex *)d_L_hat_dst_complex, (cufftComplex *)d_L_hat_dst_complex, CUFFT_FORWARD);
	cudaCheckError()
	// Filter images
	pointWise_complex_matrix_mult_kernel_2d << < gridSize, blockSize, 0, stream[1] >> >(d_L_hat_ref_complex, d_CSF, d_L_hat_ref_complex);
	cudaCheckError()
	pointWise_complex_matrix_mult_kernel_2d << < gridSize, blockSize, 0, stream[1] >> >(d_L_hat_dst_complex, d_CSF, d_L_hat_dst_complex);
	cudaCheckError()
	// LMSE - CAN MOVE ALMOST ANYWHERE
	LMSE_map_kernel << <gridSize, blockSize, 0, stream[1] >> >(d_reflut, d_lmse);
	cudaCheckError()
#ifdef DEBUG_MODE
		float h_reflut1[IMG_SIZE];
	gpuErrchk(cudaMemcpy(h_reflut1, d_reflut, REAL_SIZE, cudaMemcpyDeviceToHost))
		float h_lmse11[IMG_SIZE];
	gpuErrchk(cudaMemcpy(h_lmse11, d_lmse, REAL_SIZE, cudaMemcpyDeviceToHost))//Issue with LMSE_map_kernel
#endif
	// Exectute "In-Place" C2C 2D-FFT^-1 (i.e. inverse)
	cufftExecC2C(fftPlan[1], (cufftComplex *)d_L_hat_ref_complex, (cufftComplex *)d_L_hat_ref_complex, CUFFT_INVERSE);
	cudaCheckError()
	cufftExecC2C(fftPlan[1], (cufftComplex *)d_L_hat_dst_complex, (cufftComplex *)d_L_hat_dst_complex, CUFFT_INVERSE);
	cudaCheckError()
	// Take Real Part
	real_kernel << <gridSize, blockSize, 0, stream[1] >> >(d_L_hat_ref_complex, d_I_prime_org);
	cudaCheckError()
	real_kernel << <gridSize, blockSize, 0, stream[1] >> >(d_L_hat_dst_complex, d_I_prime_dst);
	cudaCheckError()
	// Compute error image:
	error_img_kernel << <gridSize, blockSize, 0, stream[1] >> >(d_I_prime_org, d_I_prime_dst, d_I_prime_err);
	cudaCheckError()
	//cudaDeviceSynchronize();
	fast_hi_stats_kernel1 << <loStats_Grid_size, loStats_Block_size, 0, stream[1] >> >(d_I_prime_err, d_I_prime_org, d_outStd, d_outStdMod, d_outMean, d_img_ref_float, d_img_dst_float, d_TEMP);
	cudaCheckError()
	fast_hi_stats_kernel2 << <loStats_Grid_size, loStats_Block_size, 0, stream[1] >> >(d_I_prime_err, d_I_prime_org, d_outStd, d_outStdMod, d_outMean, d_img_ref_float, d_img_dst_float, d_TEMP);
	cudaCheckError()
	zeta_map_kernel << <gridSize, blockSize, 0, stream[1] >> >(d_outMean, d_outStd, d_outStdMod, d_zeta);
	cudaCheckError()
	// Product inside summation in MAD eq. 7
	product_array_kernel << <gridSize, blockSize, 0, stream[1] >> >(d_product, d_zeta, d_lmse);
#ifdef DEBUG_MODE
	gpuErrchk(cudaMemcpy(h_product, d_product, REAL_SIZE, cudaMemcpyDeviceToHost)) //DEVICE -> HOST  : TEST
	float h_zeta1[IMG_SIZE];
	gpuErrchk(cudaMemcpy(h_zeta1, d_zeta, REAL_SIZE, cudaMemcpyDeviceToHost))
	float h_lmse1[IMG_SIZE];
	gpuErrchk(cudaMemcpy(h_lmse1, d_lmse, REAL_SIZE, cudaMemcpyDeviceToHost))//Issue with lmse: Tracking back
#endif
	cudaCheckError()
	// Exectute "in-place" C2C 2D-DFT of REF (used in the LEFT side of the Gabor Filterbank)
	R2C_kernel << <gridSize, blockSize, 0, stream[1] >> >(d_img_ref_float, d_ref_cufft);
	cudaCheckError()										//in			out
	R2C_kernel << <gridSize, blockSize, 0, stream[1] >> >(d_img_dst_float, d_dst_cufft);
	cudaCheckError()

	// Forward FFT
	cufftExecC2C(fftPlan[1], (cufftComplex *)d_ref_cufft, (cufftComplex *)d_ref_cufft, CUFFT_FORWARD);
	cudaCheckError()
	cufftExecC2C(fftPlan[1], (cufftComplex *)d_dst_cufft, (cufftComplex *)d_dst_cufft, CUFFT_FORWARD);
	cudaCheckError()
#ifdef DEBUG_MODE
	cufftComplex h_ref_cufft_var[COMPLEX_SIZE];
	cudaMemcpy(h_ref_cufft_var, d_ref_cufft, COMPLEX_SIZE, cudaMemcpyDeviceToHost);
#endif
	float scale[5] = { 0.5, 0.75, 1, 5, 6 };

	for (int o = 0; o < 4; o++)
	{
		for (int s = 0; s < 5; s++)
		{
			buildGabor << < gridSize, blockSize, 0, stream[1] >> >(d_logGabor, o, s);
#ifdef DEBUG_MODE
			float gabor[IMG_SIZE];
			cudaMemcpy(gabor, d_logGabor, REAL_SIZE, cudaMemcpyDeviceToHost);
#endif
			cudaCheckError()
			fftShift_kernel << < fftShift_grid_size, fftShift_block_size, 0, stream[1] >> >(d_logGabor);
#ifdef DEBUG_MODE
			cudaMemcpy(gabor, d_logGabor, REAL_SIZE, cudaMemcpyDeviceToHost);
#endif
			cudaCheckError()

			pointWise_complex_matrix_mult_kernel_2d << < gridSize, blockSize, 0, stream[1] >> >(d_ref_cufft, d_logGabor, d_ref_c);
#ifdef DEBUG_MODE
			cufftComplex h_ref_c1[COMPLEX_SIZE];
			cudaMemcpy(h_ref_c1, d_ref_c, COMPLEX_SIZE, cudaMemcpyDeviceToHost);
#endif
			cudaCheckError()
			pointWise_complex_matrix_mult_kernel_2d << < gridSize, blockSize, 0, stream[1] >> >(d_dst_cufft, d_logGabor, d_dst_c);
#ifdef DEBUG_MODE
			cufftComplex h_dst_c1[COMPLEX_SIZE];
			cudaMemcpy(h_dst_c1, d_dst_c, COMPLEX_SIZE, cudaMemcpyDeviceToHost);
#endif
			cudaCheckError()

			// Inverse FFT
			cufftExecC2C(fftPlan[1], (cufftComplex *)d_ref_c, (cufftComplex *)d_ref_c, CUFFT_INVERSE);
#ifdef DEBUG_MODE
			cudaMemcpy(h_dst_c1, d_ref_c, COMPLEX_SIZE, cudaMemcpyDeviceToHost);
#endif
			cufftExecC2C(fftPlan[1], (cufftComplex *)d_dst_c, (cufftComplex *)d_dst_c, CUFFT_INVERSE);
#ifdef DEBUG_MODE
			cudaMemcpy(h_dst_c1, d_dst_c, COMPLEX_SIZE, cudaMemcpyDeviceToHost);
#endif
			cudaCheckError()

			magnitude_kernel << < gridSize, blockSize, 0, stream[1] >> >(	d_ref_c, d_ref_c_mag);
#ifdef DEBUG_MODE
			float h_ref_c_mag[IMG_SIZE];
			cudaMemcpy(h_ref_c_mag, d_ref_c_mag, REAL_SIZE, cudaMemcpyDeviceToHost);
#endif
			cudaCheckError()
			magnitude_kernel << < gridSize, blockSize, 0, stream[1] >> >(d_dst_c, d_dst_c_mag);
#ifdef DEBUG_MODE
			float h_dst_c_mag[IMG_SIZE];
			cudaMemcpy(h_dst_c_mag, d_dst_c_mag, REAL_SIZE, cudaMemcpyDeviceToHost);
#endif
			cudaCheckError()
				//write_to_file_DEBUG(h_dst_c_mag, N);
			//write_to_file_DEBUG(h_ref_c_mag, N);
			fast_lo_stats_kernel << < gridSize, blockSize, 0, stream[1] >> >(d_ref_c_mag, d_ref_Std, d_ref_Skw, d_ref_Krt);
#ifdef DEBUG_MODE
			float h_ref_Std[IMG_SIZE], h_ref_Skw[IMG_SIZE], h_ref_Krt[IMG_SIZE];
			cudaMemcpy(h_ref_Std, d_ref_Std, REAL_SIZE, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_ref_Skw, d_ref_Skw, REAL_SIZE, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_ref_Krt, d_ref_Krt, REAL_SIZE, cudaMemcpyDeviceToHost);
#endif
			cudaCheckError()
				fast_lo_stats_kernel << < gridSize, blockSize, 0, stream[1] >> >(d_dst_c_mag, d_dst_Std, d_dst_Skw, d_dst_Krt);
#ifdef DEBUG_MODE
			cudaMemcpy(h_ref_Std, d_dst_Std, REAL_SIZE, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_ref_Skw, d_dst_Skw, REAL_SIZE, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_ref_Krt, d_dst_Krt, REAL_SIZE, cudaMemcpyDeviceToHost);
#endif
			cudaCheckError()
			delta_stats_kernel << < gridSize, blockSize, 0, stream[1] >> >(d_ref_Std, d_ref_Skw, d_ref_Krt, d_dst_Std, d_dst_Skw, d_dst_Krt, scale[s] / 13.25f, d_eta);

			// NOTE: This should only write back after all 20 loop itterations:
#ifdef DEBUG_MODE
			float h_eta[IMG_SIZE];
			cudaMemcpy(h_eta, d_eta, REAL_SIZE, cudaMemcpyDeviceToHost);
#endif
			cudaCheckError()
		}
	}

	//// DEBUG:
	////printf("Loop number %d\n", o * 5 + s + 1);
	//float* test_HOST_data = (float*)malloc(REAL_SIZE);
	//cudaMemcpy(test_HOST_data, d_eta, REAL_SIZE, cudaMemcpyDeviceToHost); // HOST - >DEVICE
	//write_to_file_DEBUG(test_HOST_data, N);
	//printf("DEBUG DONE\n\n");
	//getchar();

	// Collapse the visibility-weighted local MSE via L2-norm (MAD eq. 7)
	gpuErrchk(cudaMemcpy(h_product, d_product, REAL_SIZE, cudaMemcpyDeviceToHost)) //DEVICE -> HOST
	float d_detect = reduce_sum_of_squares_2D_CPU(h_product, BLOCK_SIZE, N - BLOCK_SIZE - 1);
	d_detect = sqrt(d_detect) / sqrt(229441.0f);   //  Number of itterations in loop: counter = 229441
	d_detect = d_detect * 200;

	gpuErrchk(cudaMemcpy(h_eta, d_eta, REAL_SIZE, cudaMemcpyDeviceToHost)) //DEVICE -> HOST
	float d_appear;
	d_appear = reduce_sum_of_squares_2D_CPU(h_eta, BLOCK_SIZE, N - BLOCK_SIZE);
	d_appear = sqrt(d_appear) / 479.0f;

	float beta1 = 0.467;
	float beta2 = 0.130;
	float alpha = 1 / (1 + beta1*pow(d_detect, beta2));
	float MAD = pow(d_detect, alpha)*pow(d_appear, 1 - alpha);
	struct out_data d;
	d.hi_index = d_detect;
	d.lo_index = d_appear;
	d.mad_value = MAD;
	//std::cout << "Hi-Index d_detect = " << d_detect << std::endl;
	//std::cout << "Lo-Index d_appear = " << d_appear << std::endl;
	//std::cout << "\nMAD = " << MAD << std::endl;
	//getchar();
	free(h_eta);
	free(h_product);
	free(fftPlan);


free_routine:
	gpuErrchk(cudaFree(d_img_ref_float))
	gpuErrchk(cudaFree(d_img_dst_float));
	gpuErrchk(cudaFree(d_L_hat_ref));
	gpuErrchk(cudaFree(d_L_hat_dst));
	gpuErrchk(cudaFree(d_L_hat_ref_complex));
	gpuErrchk(cudaFree(d_L_hat_dst_complex));

	gpuErrchk(cudaFree(d_CSF));
	gpuErrchk(cudaFree(d_xPlane));
	gpuErrchk(cudaFree(d_yPlane));
	gpuErrchk(cudaFree(d_I_prime_org));
	gpuErrchk(cudaFree(d_I_prime_dst));
	gpuErrchk(cudaFree(d_I_prime_err));
	gpuErrchk(cudaFree(d_outStd));

	gpuErrchk(cudaFree(d_outStdMod));
	gpuErrchk(cudaFree(d_outMean));
	gpuErrchk(cudaFree(d_reflut));
	gpuErrchk(cudaFree(d_TEMP));
	gpuErrchk(cudaFree(d_zeta));
	gpuErrchk(cudaFree(d_lmse));

	gpuErrchk(cudaFree(d_product));
	gpuErrchk(cudaFree(d_ref_cufft));
	gpuErrchk(cudaFree(d_dst_cufft));
	gpuErrchk(cudaFree(d_logGabor));
	gpuErrchk(cudaFree(d_ref_c));
	gpuErrchk(cudaFree(d_dst_c));

	gpuErrchk(cudaFree(d_ref_c_mag));
	gpuErrchk(cudaFree(d_dst_c_mag));
	gpuErrchk(cudaFree(d_ref_Std));
	gpuErrchk(cudaFree(d_ref_Skw));
	gpuErrchk(cudaFree(d_ref_Krt));

	gpuErrchk(cudaFree(d_dst_Std));
	gpuErrchk(cudaFree(d_dst_Skw));
	gpuErrchk(cudaFree(d_dst_Krt));
	gpuErrchk(cudaFree(d_eta));
	// For Profiling (nvvp, NSIGHT, etc.)
	return d;
}
//-----------------------------------------------------------------------------
