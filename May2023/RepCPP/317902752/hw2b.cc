#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
int *complete_image;
int width, height, iters, num_thread;
double dx, dy, real_min, imag_min;
int world_size, job_width, data_size, rank_num, *result, *results;
struct ComplexNum{
double real, imag;
};
void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
FILE* fp = fopen(filename, "wb");
assert(fp);
png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
assert(png_ptr);
png_infop info_ptr = png_create_info_struct(png_ptr);
assert(info_ptr);
png_init_io(png_ptr, fp);
png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
png_write_info(png_ptr, info_ptr);
png_set_compression_level(png_ptr, 1);
size_t row_size = 3 * width * sizeof(png_byte);
png_bytep row = (png_bytep)malloc(row_size);
for (int y = 0; y < height; ++y) {
memset(row, 0, row_size);
for (int x = 0; x < width; ++x) {
int p = buffer[(height - 1 - y) * width + x];
png_bytep color = row + x * 3;
if (p != iters) {
if (p & 16) {
color[0] = 240;
color[1] = color[2] = p % 16 * 16;
} else {
color[0] = p % 16 * 16;
}
}
}
png_write_row(png_ptr, row);
}
free(row);
png_write_end(png_ptr, NULL);
png_destroy_write_struct(&png_ptr, &info_ptr);
fclose(fp);
}
int calc_pixel(ComplexNum& c, int iters){
int repeats = 0;
double lengthsq = 0.0;
ComplexNum z = {0, 0};
while(repeats<iters && lengthsq<4.0){
double temp = z.real * z.real - z.imag * z.imag + c.real;
z.imag = 2 * z.real * z.imag + c.imag;
z.real = temp;
lengthsq = z.real * z.real + z.imag * z.imag;
repeats++;
}
return repeats;
}
void start(int sz){
ComplexNum c;
#pragma omp parallel for schedule(static) private(c) collapse(2)
for(int i=0;i<sz;++i){
for(int j=0;j<height; j++){
c.real = (i+rank_num*sz)*dx + real_min;
c.imag = j*dy+imag_min;
result[j*sz+i] = calc_pixel(c, iters);
}
}

}
void collect_results(){
if(rank_num == 0) results = (int*)malloc(world_size * data_size * sizeof(int));
MPI_Gather(result, data_size, MPI_INT, results, data_size, MPI_INT, 0, MPI_COMM_WORLD);
}
int main(int argc, char** argv) {
cpu_set_t cpu_set;
sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
num_thread = CPU_COUNT(&cpu_set);



const char* filename = argv[1];
iters = strtol(argv[2], 0, 10);
real_min = strtod(argv[3], 0);
double real_max = strtod(argv[4], 0);
imag_min = strtod(argv[5], 0);
double imag_max = strtod(argv[6], 0);
width = strtod(argv[7], 0);
height = strtod(argv[8], 0);



dx = (real_max - real_min) / width;
dy = (imag_max - imag_min) / height;

MPI_Init(&argc, &argv);
MPI_Comm_size(MPI_COMM_WORLD, &world_size); 
MPI_Comm_rank(MPI_COMM_WORLD, &rank_num); 
job_width = width / world_size;
if(width % world_size) job_width += 1;
data_size = job_width * height;
result = (int*)malloc(data_size * sizeof(int));


omp_set_num_threads(num_thread);
start(job_width);


collect_results(); 
if(rank_num==0){
assert(results);
write_png(filename, iters, width, height, results);
free(results);
}   
MPI_Finalize();
return 0;



}
