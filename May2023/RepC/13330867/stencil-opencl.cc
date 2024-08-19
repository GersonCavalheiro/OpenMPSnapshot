#include "prk_util.h"
#include "prk_opencl.h"
template <typename T>
void run(cl::Context context, int iterations, int n, int radius, bool star)
{
auto precision = (sizeof(T)==8) ? 64 : 32;
std::string funcname1, filename1;
funcname1.reserve(255);
funcname1 += ( star ? "star" : "grid" );
funcname1 += std::to_string(radius);
funcname1 += "_" + std::to_string(precision);
filename1 = funcname1 + ( ".cl" );
auto funcname2 = (precision==64) ? "add64" : "add32";
auto filename2 = "add"+std::to_string(precision)+".cl";
std::string source = prk::opencl::loadProgram(filename1);
if ( source==std::string("FAIL") ) {
std::cerr << "OpenCL kernel source file (" << filename1 << ") not found. "
<< "Generating using Python script" << std::endl;
std::string command("./generate-opencl-stencil.py ");
command += ( star ? "star " : "grid " );
command += std::to_string(radius);
int rc = std::system( command.c_str() );
if (rc != 0) {
std::cerr << command.c_str() << " returned " << rc << std::endl;
}
}
source = prk::opencl::loadProgram(filename1);
cl::Program program1(context, source, true);
cl::Program program2(context, prk::opencl::loadProgram(filename2), true);
cl_int err;
auto kernel1 = cl::KernelFunctor<int, cl::Buffer, cl::Buffer>(program1, funcname1, &err);
if(err != CL_SUCCESS){
std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
std::cout << program1.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
}
auto kernel2 = cl::KernelFunctor<int, cl::Buffer>(program2, funcname2, &err);
if(err != CL_SUCCESS){
std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
std::cout << program2.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
}
cl::CommandQueue queue(context);
std::vector<T> h_in(n*n,  T(0));
std::vector<T> h_out(n*n, T(0));
double stencil_time{0};
for (int i=0; i<n; i++) {
for (int j=0; j<n; j++) {
h_in[i*n+j] = static_cast<T>(i+j);
}
}
cl::Buffer d_in = cl::Buffer(context, begin(h_in), end(h_in), true);
cl::Buffer d_out = cl::Buffer(context, begin(h_out), end(h_out), false);
for (int iter = 0; iter<=iterations; iter++) {
if (iter==1) stencil_time = prk::wtime();
kernel1(cl::EnqueueArgs(queue, cl::NDRange(n,n)), n, d_in, d_out);
kernel2(cl::EnqueueArgs(queue, cl::NDRange(n,n)), n, d_in);
queue.finish();
}
stencil_time = prk::wtime() - stencil_time;
cl::copy(queue, d_out, begin(h_out), end(h_out));
#ifdef VERBOSE
cl::copy(queue, d_in, begin(h_in), end(h_in));
#endif
size_t active_points = static_cast<size_t>(n-2*radius)*static_cast<size_t>(n-2*radius);
double norm = 0.0;
for (int i=radius; i<n-radius; i++) {
for (int j=radius; j<n-radius; j++) {
norm += prk::abs(static_cast<double>(h_out[i*n+j]));
}
}
norm /= active_points;
const double epsilon = (sizeof(T)==8) ? 1.0e-8 : 1.0e-4;
double reference_norm = 2*(iterations+1);
if (prk::abs(norm-reference_norm) > epsilon) {
std::cout << "ERROR: L1 norm = " << norm
<< " Reference L1 norm = " << reference_norm << std::endl;
} else {
std::cout << "Solution validates" << std::endl;
#ifdef VERBOSE
std::cout << "L1 norm = " << norm
<< " Reference L1 norm = " << reference_norm << std::endl;
#endif
const int stencil_size = star ? 4*radius+1 : (2*radius+1)*(2*radius+1);
size_t flops = (2L*(size_t)stencil_size+1L) * active_points;
auto avgtime = stencil_time/iterations;
std::cout << "Rate (MFlops/s): " << 1.0e-6 * static_cast<double>(flops)/avgtime
<< " Avg time (s): " << avgtime << std::endl;
}
}
int main(int argc, char* argv[])
{
std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
std::cout << "C++11/OpenCL stencil execution on 2D grid" << std::endl;
int iterations, n, radius, tile_size;
bool star = true;
try {
if (argc < 3) {
throw "Usage: <# iterations> <array dimension> [<tile_size> <star/grid> <radius>]";
}
iterations  = std::atoi(argv[1]);
if (iterations < 1) {
throw "ERROR: iterations must be >= 1";
}
n  = std::atoi(argv[2]);
if (n < 1) {
throw "ERROR: grid dimension must be positive";
} else if (n > prk::get_max_matrix_size()) {
throw "ERROR: grid dimension too large - overflow risk";
}
tile_size = 32;
if (argc > 3) {
tile_size = std::atoi(argv[3]);
if (tile_size <= 0) tile_size = n;
if (tile_size > n) tile_size = n;
}
if (argc > 4) {
auto stencil = std::string(argv[4]);
auto grid = std::string("grid");
star = (stencil == grid) ? false : true;
}
radius = 2;
if (argc > 5) {
radius = std::atoi(argv[5]);
}
if ( (radius < 1) || (2*radius+1 > n) ) {
throw "ERROR: Stencil radius negative or too large";
}
}
catch (const char * e) {
std::cout << e << std::endl;
return 1;
}
std::cout << "Number of iterations = " << iterations << std::endl;
std::cout << "Grid size            = " << n << std::endl;
std::cout << "Tile size            = " << tile_size << std::endl;
std::cout << "Type of stencil      = " << (star ? "star" : "grid") << std::endl;
std::cout << "Radius of stencil    = " << radius << std::endl;
std::vector<cl::Platform> platforms;
cl::Platform::get(&platforms);
for (auto i : platforms) {
std::vector<cl::Device> devices;
i.getDevices(CL_DEVICE_TYPE_ALL, &devices);
for (auto j : devices) {
auto t = j.getInfo<CL_DEVICE_TYPE>();
if (t == CL_DEVICE_TYPE_CPU || t == CL_DEVICE_TYPE_GPU) {
std::cout << "\n" << "CL_DEVICE_NAME=" << j.getInfo<CL_DEVICE_NAME>() << "\n";
auto e = j.getInfo<CL_DEVICE_EXTENSIONS>();
auto has64 = prk::stringContains(e,"cl_khr_fp64");
cl::Context ctx(j);
run<float>(ctx, iterations, n, radius, star);
if (has64) {
run<double>(ctx, iterations, n, radius, star);
}
}
}
}
return 0;
}
