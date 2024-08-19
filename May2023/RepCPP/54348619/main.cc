
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include "camera.h"
#include "renderer.h"
#include "mandelbox.h"
#include "vector3d.h"
#include "color.h"
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef _OPENACC
#include <openacc.h>
#endif
#ifdef USE_MPI
#include <mpi.h>
#endif

extern void getParameters(char *filename, CameraParams *camera_params, RenderParams *renderer_params, MandelBoxParams *mandelBox_paramsP);
extern void init3D       (CameraParams *camera_params, const RenderParams *renderer_params);
extern void renderFractal(const CameraParams &camera_params, const RenderParams &renderer_params, unsigned char* image);
extern void saveBMP(const char* filename, const unsigned char* image, int width, int height);
extern void fillDefaultParams(CameraParams *camP, RenderParams *renP, MandelBoxParams *boxP);
extern void generateCameraPath(CameraParams &camera_params, RenderParams &renderer_params, CameraParams *camera_params_array, int frames, double camera_speed);


extern void   printProgress( double perc, double time );
extern double getTime();

MandelBoxParams mandelBox_params;
struct stat st = {0};

int main(int argc, char** argv)
{
#ifdef USE_MPI
int rank, size,namelen,provided;
MPI_Init_thread(0, 0, MPI_THREAD_MULTIPLE, &provided );
MPI_Comm_size(MPI_COMM_WORLD, &size);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
char processor_name[MPI_MAX_PROCESSOR_NAME];
MPI_Get_processor_name(processor_name, &namelen);
printf("Nodes: %d, Name: %s, Rank: %d\n",size,processor_name,rank);
#else
int rank = 0;
int size = 1;
#endif



#if defined(_OPENMP)
int nProcessors=omp_get_num_procs();
omp_set_num_threads(nProcessors);
#endif


if (stat("_images", &st) == -1)
{
mkdir("_images", 0700);
}
if (stat("_videos", &st) == -1)
{
mkdir("_videos", 0700);
}

CameraParams    camera_params;
RenderParams    renderer_params;

if (argc > 1){ 
getParameters(argv[1], &camera_params, &renderer_params, &mandelBox_params);
printf("Reading File: %s\n",argv[1]);
int image_size = renderer_params.width * renderer_params.height;
unsigned char *image = (unsigned char*)malloc(3*image_size*sizeof(unsigned char));
init3D(&camera_params, &renderer_params);
renderFractal(camera_params, renderer_params, image);
printf("Saving %s\n",renderer_params.file_name);
saveBMP(renderer_params.file_name, image, renderer_params.width, renderer_params.height);
free(image);
}
else{ 

fillDefaultParams(&camera_params, &renderer_params, &mandelBox_params); 

camera_params.camPos[0] = 20;
camera_params.camPos[1] = 20;
camera_params.camPos[2] = 7;

camera_params.camTarget[0] = -4;
camera_params.camTarget[1] = -4;
camera_params.camTarget[2] = -1;

double fov = 1; 


int frames = 10000;
int out_width = 1920; 
int out_height = 1080; 
double camera_speed = 0.01; 


CameraParams *camera_params_array = (CameraParams *)malloc(sizeof(CameraParams)*frames);
if (rank ==0) generateCameraPath(camera_params, renderer_params, camera_params_array, frames,camera_speed);

#ifdef USE_MPI
if (size >1){
MPI_Bcast(camera_params_array, sizeof(CameraParams)*frames,MPI_CHAR, 0, MPI_COMM_WORLD);
if (rank==0)printf("Broadcasting Path\n");
}
#endif

if (rank==0 and size == 1) printf("Rendering HD Images\n");
if (rank==0 and size > 1) printf("Rendering HD Images Across %d Nodes\n",size);
renderer_params.width = out_width;
renderer_params.height = out_height;

int image_size = renderer_params.width * renderer_params.height;
unsigned char *image = (unsigned char*)malloc(3*image_size*sizeof(unsigned char));
init3D(&camera_params, &renderer_params);


camera_params.fov = fov;
double time;
if (rank == 0) time = getTime(); 

int i;
for (i=rank; i<frames;i+=size){
snprintf(renderer_params.file_name,80,"_images/image%010d.bmp",i+1);
camera_params= camera_params_array[i];
camera_params.fov = fov;

init3D(&camera_params, &renderer_params);
renderFractal(camera_params, renderer_params, image);


if (rank==0) printProgress((i+1)/(double)frames,getTime()-time);
saveBMP(renderer_params.file_name, image, renderer_params.width, renderer_params.height);
}
if (rank==0) printProgress((double)1,getTime()-time); 

free(image);
}



if (rank==0) printf("\n");
#ifdef USE_MPI
MPI_Finalize();
#endif
return 0;
}





