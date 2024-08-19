#include <cstdlib>
#include <iomanip>
#include <omp.h>
#include <math.h>
#include <mutex>

#include "Mesh.hpp"
#include "kernel.hpp"

void cuda();

float CPU_SHARE = 0.f;

void init(int argc, const char* argv[]) {
if (argc < 6) {
std::cout << "Invalid number of arguments\n";
exit(-1);
}


omp_set_num_threads(std::atoi(argv[3]));


STEPS = std::atoi(argv[1]);
MESH_SIZE = std::atoi(argv[2]);
MESH_SIZE_EXTENDED = MESH_SIZE + 2;
BLOCK_DIM_X = std::atoi(argv[4]);
BLOCK_DIM_Y = std::atoi(argv[5]);
CPU_SHARE = std::atof(argv[6]);
}

void basic() {
Mesh m;
Mesh::ContainerType nodes;

try {
m.resize(MESH_SIZE);
nodes = Mesh::temperature;
}
catch (std::exception ex) {
std::cout << ex.what() << std::endl;
exit(-1);
}

SimpleTimer t{ "Basic implementation" };
for (size_t step = 0; step < STEPS; ++step) {
#pragma omp parallel for shared(nodes)
for (long long int t = 0; t < MESH_SIZE * MESH_SIZE; ++t) {
size_t x = t / MESH_SIZE;
size_t y = t % MESH_SIZE;
float t_l = Mesh::getTemperature(x - 1, y); 
float t_r = Mesh::getTemperature(x + 1, y); 
float t_t = Mesh::getTemperature(x, y - 1); 
float t_b = Mesh::getTemperature(x, y + 1); 

const float k = 1.f; 
const float d_x = 1.f; 
float q = 0.f; 
float newTemperature = (t_l + t_r + t_b + t_t + q * d_x * d_x / k) / 4;

nodes[y][x] = newTemperature;
}

Mesh::temperature.swap(nodes);
}
}

void optimized() {
float** input = allocMesh();
float** output = allocMesh();

{
SimpleTimer t{ "Optimized implementation" };
# pragma omp parallel shared(input, output)
for (int step = 0; step < STEPS; ++step) {
#pragma omp for
for (int i = 0; i < MESH_SIZE; ++i) {
#pragma ivdep
for (int j = 0; j < MESH_SIZE; ++j) {
const int y = i + 1;
const int x = j + 1;
const float t_l = input[y][x - 1]; 
const float t_r = input[y][x + 1]; 
const float t_t = input[y - 1][x]; 
const float t_b = input[y + 1][x]; 

const float k = 1.f; 
const float d_x = 1.f; 
const float q = 0.f; 
const float newTemperature = (t_l + t_r + t_b + t_t + q * d_x * d_x / k) / 4;
output[y][x] = newTemperature;
}
}

#pragma omp barrier

#pragma omp master
std::swap(input, output);

#pragma omp barrier
}
}

validateResults(input);

freeMesh(input);
freeMesh(output);
}

size_t runBenchmark() {
size_t divisionPoint = MESH_SIZE_EXTENDED * CPU_SHARE;
auto GPUShare = static_cast<float>(MESH_SIZE_EXTENDED - divisionPoint) / MESH_SIZE_EXTENDED; 
auto CPUShare = 1.f - GPUShare;


return divisionPoint;
}


void hybrid_nested_parallelism() {
static const size_t BATCH_SIZE = 100;
size_t pitch;
float* linearMesh_in = allocMeshLinear(pitch);
float* linearMesh_out = allocMeshLinear(pitch);

omp_set_nested(1);

size_t DIVISION_POINT = runBenchmark();

HybridCuda cuda(DIVISION_POINT, pitch, 0);

{
SimpleTimer t{ "Hybrid implementation nested parallelism" };
cuda.copyInitial(linearMesh_in);

for (int step = 0; step < STEPS; ++step) {
#pragma omp parallel sections num_threads(7)
{
#pragma omp section
{
cuda.launchCompute(linearMesh_in);
cuda.finalizeCompute(linearMesh_out);
}

#pragma omp section
{	
#pragma omp parallel for num_threads(6)
for (int i = 0; i < DIVISION_POINT - 1; ++i) {
#pragma ivdep
for (int j = 0; j < MESH_SIZE; ++j) {
const int y = i + 1;
const int x = j + 1;
const float t_l = *getElem(linearMesh_in, pitch, y, x - 1);
const float t_r = *getElem(linearMesh_in, pitch, y, x + 1);
const float t_t = *getElem(linearMesh_in, pitch, y - 1, x);
const float t_b = *getElem(linearMesh_in, pitch, y + 1, x);

const float newTemperature = (t_l + t_r + t_b + t_t) / 4;
*getElem(linearMesh_out, pitch, y, x) = newTemperature;
}
}
}
}

std::swap(linearMesh_in, linearMesh_out);
}

cuda.copyFinal(linearMesh_in);
}


validateResults(linearMesh_in, pitch);

delete[] linearMesh_in, linearMesh_out;
}

void hybrid() {
static const size_t BATCH_SIZE = 100;
size_t pitch;
float* linearMesh_in = allocMeshLinear(pitch);
float* linearMesh_out = allocMeshLinear(pitch);

omp_set_nested(1);

size_t DIVISION_POINT = runBenchmark();

HybridCuda cuda(DIVISION_POINT, pitch, 0);

try
{
SimpleTimer t{ "Hybrid implementation" };
cuda.copyInitial(linearMesh_in);

#pragma omp parallel
for (int step = 0; step < STEPS; ++step) {
#pragma omp master
cuda.launchCompute(linearMesh_in);

{
#pragma omp for
for (int i = 0; i < DIVISION_POINT - 1; ++i) {
#pragma ivdep
for (int j = 0; j < MESH_SIZE; ++j) {
const int y = i + 1;
const int x = j + 1;
const float t_l = *getElem(linearMesh_in, pitch, y, x - 1);
const float t_r = *getElem(linearMesh_in, pitch, y, x + 1);
const float t_t = *getElem(linearMesh_in, pitch, y - 1, x);
const float t_b = *getElem(linearMesh_in, pitch, y + 1, x);

const float newTemperature = (t_l + t_r + t_b + t_t) / 4;
*getElem(linearMesh_out, pitch, y, x) = newTemperature;
}
}
}

#pragma omp master
cuda.finalizeCompute(linearMesh_out);

#pragma omp barrier

#pragma omp single
std::swap(linearMesh_in, linearMesh_out);

#pragma omp barrier
}
cuda.copyFinal(linearMesh_in);
}
catch (std::exception& ex) {
std::cout << ex.what();
}


validateResults(linearMesh_in, pitch);

delete[] linearMesh_in, linearMesh_out;
}

void hybrid_cuda() {
static const size_t BATCH_SIZE = 100;
size_t pitch;
float* linearMesh_in = allocMeshLinear(pitch);
float* linearMesh_out = allocMeshLinear(pitch);

size_t DIVISION_POINT = runBenchmark();

HybridCuda cuda0(DIVISION_POINT, pitch, 0);
HybridCuda cuda1(DIVISION_POINT, pitch, 1);

{
SimpleTimer t{ "Hybrid implementation" };
cuda0.copyInitial(linearMesh_in);
cuda1.copyInitial(linearMesh_in);

#pragma omp parallel
for (int step = 0; step < STEPS; ++step) {
#pragma omp sections
{
#pragma omp section
{
cuda0.launchCompute(linearMesh_in);
cuda0.finalizeCompute(linearMesh_out);
}

#pragma omp section
{
cuda1.launchCompute(linearMesh_in);
cuda1.finalizeCompute(linearMesh_out);
}
}

#pragma omp barrier

#pragma omp single
std::swap(linearMesh_in, linearMesh_out);

#pragma omp barrier
}

cuda0.copyFinal(linearMesh_in);
cuda1.copyFinal(linearMesh_in);
}


validateResults(linearMesh_in, pitch);

delete[] linearMesh_in, linearMesh_out;
}

int main(int argc, const char* argv[]) {
init(argc, argv);

setValidateResults(false);

optimized();

return 0;
}
