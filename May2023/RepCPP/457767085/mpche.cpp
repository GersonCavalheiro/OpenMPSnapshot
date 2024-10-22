#include <iostream>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include "headers/SerialLHE.h"



#include "headers/ParallelLHE.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>



#include "headers/ParallelFastLHE.h"
#include <chrono>



#include "headers/VideoCreator.h"

using namespace cv;


#define T_NUM_THREADS 12



using namespace std::chrono;





int main(int argc, char **argv)
{
#pragma omp parallel
{
omp_get_num_procs();
}

int color = 0;
char *input_path = NULL;
char *output_path = NULL;
int thread_num = 1;
double ratio = 1.0;
int mode = 1;
int is_stream = -1;
int index;
int w = 151;
int c;

opterr = 0;

while ((c = getopt(argc, argv, "hcs:i:o:t:r:m:w:")) != -1)
switch (c)
{
case 'h':
std::cout << "Usage: ./main -s <is_stream> -i <input_path> -o <output_path> [-t <thread_num> -r <ratio> -m <mode> -w <window> -c]" << std::endl;
std::cout << "Thread num: Number of threads to be used" << std::endl;
std::cout << "Ratio: Resize ratio for the image (Only in image mode)" << std::endl;
std::cout << "Mode: 1 for PLHE, 2 for FastPLHE, 3 for SLHE and 4 for FastSLHE" << std::endl;
std::cout << "Color: 1 for color, 0 for grayscale" << std::endl;
std::cout << "Stream: 1 for video, 0 for single image" << std::endl;
std::cout << "Window: Size of the window" << std::endl;
return 0;
break;
case 'c':
color = 1;
break;
case 'w':
w = atoi(optarg);
if (w < 1 || w % 2 == 0)
{
std::cout << "Window size must be greater than 1 and an odd number" << std::endl;
return 0;
}
break;
case 's':
is_stream = atoi(optarg);
if (is_stream != 0 && is_stream != 1)
{
std::cout << "Error: Invalid stream option, 0 for image and 1 for video" << std::endl;
exit(1);
}
break;
case 'i':
input_path = optarg;
break;
case 'o':
output_path = optarg;
break;
case 't':
thread_num = atoi(optarg);
if (thread_num == 0 || thread_num > omp_get_max_threads() || thread_num < 0)
{
fprintf(stderr, "Error: Thread number must be a positive integer and less than %d\n", omp_get_max_threads());
exit(1);
}
break;
case 'm':
mode = atoi(optarg);
if (mode != 1 && mode != 2 && mode != 3 && mode != 4)
{
fprintf(stderr, "Error: Mode must be 1 for PLHE, 2 for FastPLHE, 3 for SLHE or 4 for FastSLHE\n");
exit(1);
}
break;
case 'r':
ratio = strtod(optarg, NULL);
if (ratio == 0)
{
fprintf(stderr, "ratio must be floating point number\n");
exit(1);
}
if (ratio > 1)
{
fprintf(stderr, "ratio must be less than 1\n");
exit(1);
}
break;
case '?':
if (optopt == 'i')
fprintf(stderr, "Option -%c requires an argument.\n", optopt);
else if (optopt == 'o')
fprintf(stderr, "Option -%c requires an argument.\n", optopt);
else if (optopt == 't')
fprintf(stderr, "Option -%c requires an argument.\n", optopt);
else if (optopt == 'r')
fprintf(stderr, "Option -%c requires an argument.\n", optopt);
else if (isprint(optopt))
fprintf(stderr, "Unknown option `-%c'.\n", optopt);
else
fprintf(stderr,
"Unknown option character `\\x%x'.\n",
optopt);
return 1;
default:
abort();
}

if (input_path == NULL)
{
fprintf(stderr, "Error: Input path is required\n");
exit(1);
}
if (output_path == NULL)
{
fprintf(stderr, "Error: Output path is required\n");
exit(1);
}
if (is_stream == -1)
{
fprintf(stderr, "Error: Stream option is required\n");
exit(1);
}
for (index = optind; index < argc; index++)
printf("Invalid argument %s\n", argv[index]);

std::cout << "Running with these configurations: " << std::endl;
std::cout << "Input path: " << input_path << std::endl;
std::cout << "Output path: " << output_path << std::endl;
std::cout << "Thread number: " << thread_num << std::endl;
std::cout << "Ratio: " << ratio << std::endl;
std::cout << "Window size: " << w << std::endl;
switch (mode)
{
case 1:
std::cout << "Mode: ParallelLHE" << std::endl;
break;
case 2:
std::cout << "Mode: ParallelFastLHE" << std::endl;
break;
case 3:
std::cout << "Mode: SerialLHE" << std::endl;
break;
case 4:
std::cout << "Mode: SerialFastLHE" << std::endl;
break;
default:
std::cout << "Mode: ParallelLHE" << std::endl;
break;
}
char t_res[6] = {'\0'};
strncpy(t_res, "False\0", 6);
if (color != 0)
{
strncpy(t_res, "True\0", 6);
}
std::cout << "Has multiple channels: " << t_res << std::endl;
if (is_stream == 1)
{
std::cout << "Stream: Video" << std::endl;
}
else
{
std::cout << "Stream: Image" << std::endl;
}

if (is_stream == 1)
{
VideoCreator vc;
vc.videoHandlerPipeline(input_path, output_path, thread_num, mode, color, w);
}
else
{
SerialLHE slhe;
ParallelFastLHE pflhe;
ParallelLHE plhe;
Mat img;
Mat resized_img;
Mat base;
if (color == 1)
{
img = imread(input_path);
}
else
{
img = imread(input_path, 0);
}
if (img.empty())
{
fprintf(stderr, "Error: Cannot open image file\n");
exit(1);
}
if (ratio != 1)
{
resize(img, resized_img, Size(), ratio, ratio);
}
else
{
resized_img = img.clone();
}
if (w > resized_img.cols || w > resized_img.rows)
{
fprintf(stderr, "Error: Window size should not be bigger than the image\n");
exit(1);
}
base = Mat::zeros(resized_img.size(), CV_MAKETYPE(CV_8U, img.channels()));
switch (mode)
{
case 1:
omp_set_num_threads(thread_num);
plhe.ApplyLHE(base, resized_img, w);
break;
case 2:
omp_set_num_threads(thread_num);
pflhe.ApplyLHEWithInterpolation(base, resized_img, w);
break;
case 3:
slhe.ApplyLHE(base, resized_img, w);
break;
case 4:
slhe.ApplyLHEWithInterpol(base, resized_img, w);
break;
default:
break;
}
try
{
imwrite(output_path, base);
}
catch (std::runtime_error &ex)
{
fprintf(stderr, "Error: Could not write image file at %s\n", output_path);
exit(1);
}
}













return 0;
}
