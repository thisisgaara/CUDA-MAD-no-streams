// main.cpp
//-------------------------------------------------------------------
#include "header.h"
//-------------------------------------------------------------------
#define PROFILE
void do_main()
{
	cudaSetDevice(DEVICE_NUM);
	out_data d;
#ifdef NPROFILE
	char *ref_images[] = { //"./imgs/ref/horse.bmp",
							"./imgs/ref/aerial_city.png",
							"./imgs/ref/child_swimming.png",
							"./imgs/ref/fisher.png",
							"./imgs/ref/geckos.png",
							"./imgs/ref/snow_leaves.png",
							"./imgs/ref/sunsetcolor.png",
							"./imgs/ref/swarm.png", };
	char *dst_images[] = {  //"./imgs/dst/horse.JP2.bmp",
							"./imgs/dst/aerial_city.AWGN.1.png",
							"./imgs/dst/aerial_city.AWGN.5.png",
							"./imgs/dst/aerial_city.BLUR.1.png",
							"./imgs/dst/aerial_city.BLUR.5.png",
							"./imgs/dst/aerial_city.jpeg2000.1.png",
							"./imgs/dst/aerial_city.jpeg2000.5.png",

							"./imgs/dst/child_swimming.AWGN.1.png",
							"./imgs/dst/child_swimming.AWGN.5.png",
							"./imgs/dst/child_swimming.BLUR.1.png",
							"./imgs/dst/child_swimming.BLUR.5.png",
							"./imgs/dst/child_swimming.jpeg2000.1.png",
							"./imgs/dst/child_swimming.jpeg2000.5.png",

							"./imgs/dst/fisher.AWGN.1.png",
							"./imgs/dst/fisher.AWGN.5.png",
							"./imgs/dst/fisher.BLUR.1.png",
							"./imgs/dst/fisher.BLUR.5.png",
							"./imgs/dst/fisher.jpeg2000.1.png",
							"./imgs/dst/fisher.jpeg2000.5.png",

							"./imgs/dst/geckos.AWGN.1.png",
							"./imgs/dst/geckos.AWGN.5.png",
							"./imgs/dst/geckos.BLUR.1.png",
							"./imgs/dst/geckos.BLUR.5.png",
							"./imgs/dst/geckos.jpeg2000.1.png",
							"./imgs/dst/geckos.jpeg2000.5.png",

							"./imgs/dst/snow_leaves.AWGN.1.png",
							"./imgs/dst/snow_leaves.AWGN.5.png",
							"./imgs/dst/snow_leaves.BLUR.1.png",
							"./imgs/dst/snow_leaves.BLUR.5.png",
							"./imgs/dst/snow_leaves.jpeg2000.1.png",
							"./imgs/dst/snow_leaves.jpeg2000.5.png",

							"./imgs/dst/sunsetcolor.AWGN.1.png",
							"./imgs/dst/sunsetcolor.AWGN.5.png",
							"./imgs/dst/sunsetcolor.BLUR.1.png",
							"./imgs/dst/sunsetcolor.BLUR.5.png",
							"./imgs/dst/sunsetcolor.jpeg2000.1.png",
							"./imgs/dst/sunsetcolor.jpeg2000.5.png",

							"./imgs/dst/swarm.AWGN.1.png",
							"./imgs/dst/swarm.AWGN.5.png",
							"./imgs/dst/swarm.BLUR.1.png",
							"./imgs/dst/swarm.BLUR.5.png",
							"./imgs/dst/swarm.jpeg2000.1.png",
							"./imgs/dst/swarm.jpeg2000.5.png"
	};
#else
	char *ref_images[] = { //"./imgs/ref/horse.bmp",
		"aerial_city.png",
		"child_swimming.png",
		"fisher.png",
		"geckos.png",
		"snow_leaves.png",
		"sunsetcolor.png",
		"swarm.png", };
	char *dst_images[] = {  //"./imgs/dst/horse.JP2.bmp",
		"aerial_city.AWGN.1.png",
		"aerial_city.AWGN.5.png",
		"aerial_city.BLUR.1.png",
		"aerial_city.BLUR.5.png",
		"aerial_city.jpeg2000.1.png",
		"aerial_city.jpeg2000.5.png",

		"child_swimming.AWGN.1.png",
		"child_swimming.AWGN.5.png",
		"child_swimming.BLUR.1.png",
		"child_swimming.BLUR.5.png",
		"child_swimming.jpeg2000.1.png",
		"child_swimming.jpeg2000.5.png",

		"fisher.AWGN.1.png",
		"fisher.AWGN.5.png",
		"fisher.BLUR.1.png",
		"fisher.BLUR.5.png",
		"fisher.jpeg2000.1.png",
		"fisher.jpeg2000.5.png",

		"geckos.AWGN.1.png",
		"geckos.AWGN.5.png",
		"geckos.BLUR.1.png",
		"geckos.BLUR.5.png",
		"geckos.jpeg2000.1.png",
		"geckos.jpeg2000.5.png",

		"snow_leaves.AWGN.1.png",
		"snow_leaves.AWGN.5.png",
		"snow_leaves.BLUR.1.png",
		"snow_leaves.BLUR.5.png",
		"snow_leaves.jpeg2000.1.png",
		"snow_leaves.jpeg2000.5.png",

		"sunsetcolor.AWGN.1.png",
		"sunsetcolor.AWGN.5.png",
		"sunsetcolor.BLUR.1.png",
		"sunsetcolor.BLUR.5.png",
		"sunsetcolor.jpeg2000.1.png",
		"sunsetcolor.jpeg2000.5.png",

		"swarm.AWGN.1.png",
		"swarm.AWGN.5.png",
		"swarm.BLUR.1.png",
		"swarm.BLUR.5.png",
		"swarm.jpeg2000.1.png",
		"swarm.jpeg2000.5.png"
	};
#endif
	double index_array[NUMBER_OF_IMAGES] = { 0xff };
	float mad_value = 0;
	int i, j, k;
	
	for ( i = 0; i < sizeof(ref_images)/sizeof(char*); i++) //For every image in ref_images
	{
		cv::Mat mat_ref = cv::imread(ref_images[i], CV_8UC1);
		for ( j = 0; j < 6; j++) //For every image in dst_images
		{
			cv::Mat mat_dst = cv::imread(dst_images[i*6 + j], CV_8UC1); //6 because there are 6 distorted image for every ref image
			mad_value = 0;
			for ( k = 0; k < NUMBER_OF_ITERATIONS; k++)
			{
				// Call the wrapper function
				d = kernel_wrapper(mat_ref, mat_dst);
				mad_value += d.mad_value;
			}
			printf("%d %f %f\n", i, mad_value, mad_value / (float)NUMBER_OF_ITERATIONS);
			index_array[i * 6 + j] = mad_value / (float)NUMBER_OF_ITERATIONS;
			printf("The MAD index for %s and %s is: %f\n", ref_images[i], dst_images[i * 6 + j], d.mad_value);
		}			
	}
	cudaDeviceReset();
}
//---------------------------------------------------------------------

int main(int argc, char* argv[])
{
	try
	{
		do_main();
	}
	catch (std::exception const& err)
	{
		std::printf("%s\n", err.what());
		getchar();
	}

	return 0;
}
//---------------------------------------------------------------------
