
#include <fstream>
#include <limits>
#include <math.h>
#include <chrono>
#include "kmeans.h"



#include <cassert>
#include <cstring>
#include <omp.h>




int
main (int argc, char *argv[])
{
if (argc != 3) {
std::cout << "Usage: " << argv[0] << " <input.txt> <output.txt>"
<< std::endl;
return -1;
}
if (!(bool)std::ifstream(argv[1])) {
std::cerr << "ERROR: Data file " << argv[1] << " does not exist!"
<< std::endl;
return -1;
}
if ((bool)std::ifstream(argv[2])) {
std::cerr << "ERROR: Destination " << argv[2] << " already exists!"
<< std::endl;
return -1;
}
FILE *fi = fopen(argv[1], "r"), *fo = fopen(argv[2], "w");


int pn, cn;

assert(fscanf(fi, "%d / %d\n", &pn, &cn) == 2);

point_t * const data = new point_t[pn];
color_t * const coloring = new color_t[pn];

for (int i = 0; i < pn; ++i)
coloring[i] = 0;

int i = 0, c;
double x, y;

while (fscanf(fi, "%lf, %lf, %d\n", &x, &y, &c) == 3) {
data[i++].setXY(x, y);
if (c < 0 || c >= cn) {
std::cerr << "ERROR: Invalid color code encoutered!"
<< std::endl;
return -1;
}
}
if (i != pn) {
std::cerr << "ERROR: Number of data points inconsistent!"
<< std::endl;
return -1;
}


point_t * const mean = new point_t[cn];

srand(5201314);
for (int i = 0; i < cn; ++i) {
int idx = rand() % pn;
mean[i].setXY(data[idx].getX(), data[idx].getY());
}


std::cout << "Doing K-Means clustering on " << pn
<< " points with K = " << cn << "..." << std::flush;
auto ts = std::chrono::high_resolution_clock::now();
kmeans(data, mean, coloring, pn, cn);
auto te = std::chrono::high_resolution_clock::now();
std::cout << "done." << std::endl;
std::cout << " Total time elapsed: "
<< std::chrono::duration_cast<std::chrono::milliseconds> \
(te - ts).count()
<< " milliseconds." << std::endl; 


fprintf(fo, "%d / %d\n", pn, cn);
for (i = 0; i < pn; ++i)
fprintf(fo, "%.8lf, %.8lf, %d\n", data[i].getX(), data[i].getY(),
coloring[i]);


delete[](data);
delete[](coloring);
delete[](mean);
fclose(fi);
fclose(fo);
return 0;
}



#define __ID__ 1
#define UNUSED(X) (void)X
#define __PACK_2_BITS(a, b, c, d) ((d << 6) | (c << 4) | (b << 2) | a )


void
kmeans (point_t * const data, point_t * const mean, color_t * const coloring,
const int pn, const int cn)
{
bool converge = true;


do {
converge = true;

double agg_sums_x[20] = {0.0f};
double agg_sums_y[20] = {0.0f};
int agg_counts[20] = {0};


#pragma omp parallel
{

double sums_x[20] = {0.0f};
double sums_y[20] = {0.0f};
int counts[20] = {0};
bool local_converge = true;

int thread_num = omp_get_num_threads();
int id = omp_get_thread_num();
int block_size = pn / thread_num;
int tail_size = pn % thread_num;

int start;
if(id + 1 <= tail_size){
block_size++;
start = id * block_size;
}else{
start = block_size * id + tail_size;
}
int end = start + block_size;

for (int i = start; i < start+(block_size/4)*4; i+=4)
{
double all_one_double;
::memset(&all_one_double, 0xFF, sizeof(double));

__m256d min_dist = _mm256_set1_pd(std::numeric_limits<double>::infinity());
__m128i new_color = _mm_set1_epi32((int)cn);

for (color_t c = 0; c < cn; ++c)
{
__m256d packed_pt1 = _mm256_loadu_pd((double*)(data+i));
__m256d packed_pt2 = _mm256_loadu_pd((double*)(data+i+2));

__m256d data_xs = _mm256_unpacklo_pd(packed_pt1, packed_pt2);
__m256d data_ys = _mm256_unpackhi_pd(packed_pt1, packed_pt2);

data_xs = _mm256_permute4x64_pd(data_xs, __PACK_2_BITS(0, 2, 1, 3));
data_ys = _mm256_permute4x64_pd(data_ys, __PACK_2_BITS(0, 2, 1, 3));

__m256d mean_xs = _mm256_set1_pd(mean[c].x);
__m256d mean_ys = _mm256_set1_pd(mean[c].y);

__m256d diff_x = _mm256_sub_pd(data_xs, mean_xs);
__m256d diff_y = _mm256_sub_pd(data_ys, mean_ys);

diff_x = _mm256_mul_pd(diff_x, diff_x);
diff_y = _mm256_mul_pd(diff_y, diff_y);

__m256d dist = _mm256_add_pd(diff_x, diff_y);

__m256d cmp = _mm256_cmp_pd(dist, min_dist, _CMP_LT_OQ);

min_dist = _mm256_or_pd(
_mm256_and_pd(
min_dist
, _mm256_xor_pd(
cmp
, _mm256_set1_pd(all_one_double)
)
)
, _mm256_and_pd(
dist
, cmp
)
);

new_color = _mm256_cvtpd_epi32(_mm256_or_pd(
_mm256_and_pd(
_mm256_cvtepi32_pd(new_color)
, _mm256_xor_pd(
cmp
, _mm256_set1_pd(all_one_double)
)
)
, _mm256_and_pd(
_mm256_set1_pd((double)c)
, cmp
)
));
}

unsigned int v_new_colors[4];
_mm_storeu_si128((__m128i*)v_new_colors, new_color);
for(int c = 0; c < 4; c++){
if (coloring[i+c] != v_new_colors[c])
{
coloring[i+c] = v_new_colors[c];
local_converge = false;
}

sums_x[v_new_colors[c]] += data[i+c].x;
sums_y[v_new_colors[c]] += data[i+c].y;
counts[v_new_colors[c]]++;
}

}

for (int i = start + (block_size/4) * 4; i < end; ++i)
{
color_t new_color = cn;
double min_dist = std::numeric_limits<double>::infinity();

for (color_t c = 0; c < cn; ++c)
{
double dist = pow(data[i].x - mean[c].x, 2)
+ pow(data[i].y - mean[c].y, 2);
if (dist < min_dist)
{
min_dist = dist;
new_color = c;
}
}

if (coloring[i] != new_color)
{
coloring[i] = new_color;
local_converge = false;
}
sums_x[new_color] += data[i].x;
sums_y[new_color] += data[i].y;
counts[new_color]++;
}

#pragma omp critical
for (color_t c = 0; c < cn; ++c)
{
agg_sums_y[c] += sums_y[c];
agg_sums_x[c] += sums_x[c];
}
#pragma omp critical
for (color_t c = 0; c < cn; ++c)
{
agg_counts[c] += counts[c];
converge &= local_converge;
}

}


for (color_t c = 0; c < cn; ++c)
{
mean[c].x = agg_sums_x[c] / agg_counts[c];
mean[c].y = agg_sums_y[c] / agg_counts[c];
}

} while (!converge);
}


