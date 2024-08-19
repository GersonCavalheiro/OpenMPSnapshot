#include "morphology.h"

enum class MorphOpType {
ERODE,
DILATE,
};


#pragma omp declare target
template <MorphOpType opType>
inline unsigned char elementOp(unsigned char lhs, unsigned char rhs)
{
}

template <>
inline unsigned char elementOp<MorphOpType::ERODE>(unsigned char lhs, unsigned char rhs)
{
return lhs < rhs ? lhs : rhs;
}

template <>
inline unsigned char elementOp<MorphOpType::DILATE>(unsigned char lhs, unsigned char rhs)
{
return lhs > rhs ? lhs : rhs;
}

template <MorphOpType opType>
inline unsigned char borderValue()
{
}

template <>
inline unsigned char borderValue<MorphOpType::ERODE>()
{
return BLACK;
}

template <>
inline unsigned char borderValue<MorphOpType::DILATE>()
{
return WHITE;
}

template <MorphOpType opType>
void twoWayScan(unsigned char* __restrict buffer,
unsigned char* __restrict opArray,
const int selSize,
const int tid)
{
opArray[tid] = buffer[tid];
opArray[tid + selSize] = buffer[tid + selSize];
#pragma omp barrier

for (int offset = 1; offset < selSize; offset *= 2) {
if (tid >= offset) {
opArray[tid + selSize - 1] = 
elementOp<opType>(opArray[tid + selSize - 1], opArray[tid + selSize - 1 - offset]);
}
if (tid <= selSize - 1 - offset) {
opArray[tid] = elementOp<opType>(opArray[tid], opArray[tid + offset]);
}
#pragma omp barrier
}
}
#pragma omp end declare target


template <MorphOpType opType>
double morphology(unsigned char* img_d,
unsigned char* tmp_d,
const int width,
const int height,
const int hsize,
const int vsize)
{
int blockSize_x_h = hsize;
int blockSize_y_h = 1;
int gridSize_x_h = roundUp(width, blockSize_x_h);
int gridSize_y_h = roundUp(height, blockSize_y_h);

int blockSize_x_v = 1;
int blockSize_y_v = vsize;
int gridSize_x_v = roundUp(width, blockSize_x_v);
int gridSize_y_v = roundUp(height, blockSize_y_v);

unsigned int memSize = width * height * sizeof(unsigned char);
memset(tmp_d, 0, memSize);
#pragma omp target update to(tmp_d[0:memSize])

auto start = std::chrono::steady_clock::now();

#pragma omp target teams num_teams(gridSize_x_h*gridSize_y_h) thread_limit(blockSize_x_h*blockSize_y_h)
{
unsigned char sMem[128];
#pragma omp parallel 
{
unsigned char* buffer = sMem;
unsigned char* opArray = buffer + 2 * hsize;

int bx = omp_get_team_num() % gridSize_x_h;
int by = omp_get_team_num() / gridSize_x_h;
int tx = omp_get_thread_num();   

const int tidx = tx + bx * blockSize_x_h;
const int tidy =      by * blockSize_y_h;
if (tidx < width && tidy < height) {

buffer[tx] = img_d[tidy * width + tidx];
if (tidx + hsize < width) {
buffer[tx + hsize] = img_d[tidy * width + tidx + hsize];
}
#pragma omp barrier

twoWayScan<opType>(buffer, opArray, hsize, tx);

if (tidx + hsize/2 < width - hsize/2) {
tmp_d[tidy * width + tidx + hsize/2] = 
elementOp<opType>(opArray[tx], opArray[tx + hsize - 1]);
}
}
}
}

#pragma omp target teams num_teams(gridSize_x_v*gridSize_y_v) thread_limit(blockSize_x_v*blockSize_y_v)
{
unsigned char sMem[128];
#pragma omp parallel 
{
unsigned char* buffer = sMem;
unsigned char* opArray = buffer + 2 * vsize;

int bx = omp_get_team_num() % gridSize_x_v;
int by = omp_get_team_num() / gridSize_x_v;
int ty = omp_get_thread_num();   

const int tidx =      bx * blockSize_x_v;
const int tidy = ty + by * blockSize_y_v;
if (tidx < width && tidy < height) {

buffer[ty] = tmp_d[tidy * width + tidx];
if (tidy + vsize < height) {
buffer[ty + vsize] = tmp_d[(tidy + vsize) * width + tidx];
}
#pragma omp barrier

twoWayScan<opType>(buffer, opArray, vsize, ty);

if (tidy + vsize/2 < height - vsize/2) {
img_d[(tidy + vsize/2) * width + tidx] = 
elementOp<opType>(opArray[ty], opArray[ty + vsize - 1]);
}

if (tidy < vsize/2 || tidy >= height - vsize/2) {
img_d[tidy * width + tidx] = borderValue<opType>();
}
}
}
}

auto end = std::chrono::steady_clock::now();
auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
return time;
}

extern "C"
double erode(unsigned char* img_d,
unsigned char* tmp_d,
const int width,
const int height,
const int hsize,
const int vsize)
{
return morphology<MorphOpType::ERODE>(img_d, tmp_d, width, height, hsize, vsize);
}

extern "C"
double dilate(unsigned char* img_d,
unsigned char* tmp_d,
const int width,
const int height,
const int hsize,
const int vsize)
{
return morphology<MorphOpType::DILATE>(img_d, tmp_d, width, height, hsize, vsize);
}
