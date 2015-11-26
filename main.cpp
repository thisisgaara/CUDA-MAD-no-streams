//=============================================================================
#include "header.h"
//=============================================================================
int NUM_IMAGES = 1;
int image_index = 0; //This variable indexes into the image_array
image *image_array; //Array of images
double timing_sum_global = 0;
double timing_sum_combine_mad = 0;


int do_main()
{
	image_array = (image *) malloc (NUM_IMAGES * 2 * sizeof(image));
	std::vector<std::string> reference_image_name =
	{		"ref_1.png", "ref_2.png"	};
	std::vector<std::string> distorted_image_name =
	{
		"child_swimming.AWGN.2.png", "child_swimming.AWGN.4.png", "child_swimming.JPEG.2.png", "child_swimming.JPEG.4.png",
		"child_swimming.jpeg2000.2.png", "child_swimming.jpeg2000.4.png",	"child_swimming.contrast.2.png", "child_swimming.contrast.4.png",
		"child_swimming.fnoise.2.png", "child_swimming.fnoise.4.png",		"child_swimming.BLUR.2.png", "child_swimming.BLUR.4.png", 
		"swarm.BLUR.2.png", "swarm.BLUR.4.png", "swarm.AWGN.2.png", "swarm.AWGN.4.png", "swarm.JPEG.2.png", "swarm.JPEG.4.png", 
		"swarm.contrast.2.png", "swarm.contrast.4.png", "swarm.fnoise.2.png", "swarm.fnoise.4.png",	"swarm.jpeg200.2.png", "swarm.jpeg200.4.png"
	};


	double csf_filter_Gpu_aevrage = 0;
	double appearence_statistic_Gpu_average = 0;
	double detection_gabor_filterbank_GPU_average = 0;
	double detection_statistic_GPU_average = 0;
	

	//// Read in images
	for (int i = 0; i < 1; i++)
	{
		for (int j = 0; j < NUM_IMAGES; j++)
		{
			// Read in images
			cv::Mat mat_ref = cv::imread("horse.bmp", CV_8UC1); // CV_LOAD_IMAGE_UNCHANGED);
			//cv::Mat mat_ref = cv::imread(reference_image_name[i], CV_8UC1); // CV_LOAD_IMAGE_UNCHANGED);
			cv::Mat mat_dst = cv::imread("horse.JP2.bmp", CV_8UC1); // CV_LOAD_IMAGE_UNCHANGED);
			//cv::Mat mat_dst = cv::imread(distorted_image_name[j], CV_8UC1); // CV_LOAD_IMAGE_UNCHANGED);

			// Call function in .cu file
			cudaError_t cudaStatus = kernel_wrapper(mat_ref, mat_dst);
			image_index = image_index + 1;
			if (cudaStatus != cudaSuccess)
			{
				fprintf(stderr, "kernel_wrapper function failed!");
				return 1;
			}
			else
				//printf("Exit with zero errors...\n");
			{
				//printf("Execution number %d\n", image_index);
#ifdef CSF_FILTER
				csf_filter_Gpu_aevrage += image_array[image_index - 1].csf_filter_Gpu;
				//printf("image_array[%d].csf_filter_Gpu = %lf\n", image_index, image_array[image_index - 1].csf_filter_Gpu);
#endif
#ifdef LOW_STATS
				appearence_statistic_Gpu_average += image_array[image_index - 1].appearence_statistic_Gpu;
				//printf("image_array[%d].appearence_statistic_Gpu = %lf\n", image_index, image_array[image_index - 1].appearence_statistic_Gpu);
#endif
#ifdef LOG_GABOR
				detection_gabor_filterbank_GPU_average += image_array[image_index - 1].detection_gabor_filterbank_GPU;
				//printf("image_array[%d].detection_gabor_filterbank_GPU = %lf\n", image_index, image_array[image_index - 1].detection_gabor_filterbank_GPU);
#endif
#ifdef HI_STATS
				detection_statistic_GPU_average += image_array[image_index - 1].detection_statistic_GPU;
				//printf("image_array[%d].detection_statistic_GPU = %lf\n", image_index, image_array[image_index - 1].detection_statistic_GPU);
#endif
			}
				
		}
	}
	//printf("*****************Average values as below ************************\n");

	//printf("timing_sum_global = %lf\n", timing_sum_global / (2 * NUM_IMAGES));
	//printf("csf_filter_Gpu_aevrage = %lf\n",  csf_filter_Gpu_aevrage / (2 * NUM_IMAGES));
	//printf("hi_statistic_Gpu_average = %lf\n", appearence_statistic_Gpu_average / (2 * NUM_IMAGES));
	//printf("gabor_filterbank_GPU_average = %lf\n", detection_gabor_filterbank_GPU_average / (2 * NUM_IMAGES));
	//printf("lo_statistic_GPU_average =   %lf\n", detection_statistic_GPU_average / (2 * NUM_IMAGES));
	//printf("timing_sum_combine_mad = %lf\n", timing_sum_combine_mad / (2 * NUM_IMAGES));
	return 0;
}
//=============================================================================
int main(int argc, char* argv[])
{
	int failure_code = 0;
	try
	{
		failure_code = do_main();
	}
	catch (std::exception const& err)
	{
		std::printf("%s\n", err.what());
		failure_code = 1;
		getchar();
	}

	system("pause"); // Remove this for profiling
	//return failure_code;
	return 0; 
}
