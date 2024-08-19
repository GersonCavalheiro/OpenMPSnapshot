#pragma once

#include <cstdio> 
#include <cassert> 
#include <fstream> 
#include <sstream> 
#include <string> 

#include <complex> 
#include "complex_tools.hxx" 
#include "recorded_warnings.hxx" 
#include "status.hxx" 

namespace debug_tools {

template <typename real_t>
int read_from_file(
real_t y_data[] 
, char const *filename
, size_t const N 
, size_t const Stride=1 
, size_t const M=1 
, char const *title=nullptr 
, int const echo=0 
) {
assert(Stride >= M);
std::ifstream infile(filename, std::ifstream::in);
if (infile.fail()) {
if (echo > 1) std::printf("# %s Error opening file %s!\n", __func__, filename);
return 1;
} 

char const rc_name[][8] = {"real", "complex"};
bool const read_complex = is_complex<real_t>();

std::string line; 
unsigned linenumber = 1;
int ix = 0;
while (std::getline(infile, line)) {
char const *const line_ptr = line.c_str();
char const c0 = line_ptr ? line_ptr[0] : '\0';
switch (c0) {
case '#': {
if (echo > 3) std::printf("# %s:%d reads \'%s\'.\n", filename, linenumber, line.c_str());
char const c1 = line_ptr[1]; 
bool const contains_complex = ('c' == c1);
if (contains_complex != read_complex) {
warn("file %s contains %s numbers but trying to read %s numbers",
filename, rc_name[contains_complex], rc_name[read_complex]);
} 
} break;
case ' ': case '\n': case '\t': case '\0': {
if (echo > 9) std::printf("# %s:%d reads \'%s\'.\n", filename, linenumber, line.c_str());
} break;
default: {
std::istringstream iss(line);
double x{0};
iss >> x;
for (int iy = 0; iy < M; ++iy) {
double y{0};
iss >> y;
if (ix < N) {
if (read_complex) {
double y_imag{0};
iss >> y_imag; 
using real_real_t = decltype(std::real(real_t(1)));
auto const yc = std::complex<real_real_t>(y, y_imag);
y_data[ix*Stride + iy] = to_complex_t<real_t, real_real_t>(yc);
} else {
y_data[ix*Stride + iy] = y;
} 
} 
} 
++ix;
} 
} 
++linenumber;
} 
if (echo > 3) std::printf("# %d (of %ld) x %ld (of %ld) data entries%s%s read from file %s\n", 
ix, N, M, Stride, title?" for ":"", title, filename);
return ix - N; 
} 


template <char const write_read_delete='w'>
status_t manage_stop_file(int & number, int const echo=0, char const *filename="running.a43") {
if ('w' == (write_read_delete | 32)) {
auto *const f = std::fopen(filename, "w");
if (nullptr == f) {
if (echo > 1) std::printf("# %s<%c> unable to open file \"%s\"for writing!\n", __func__, write_read_delete, filename);
return 1;
} 
std::fprintf(f, "%d   is the max. number of self-consistency iterations, user may modify", number);
return std::fclose(f);
} else
if ('r' == (write_read_delete | 32)) {
std::ifstream infile(filename, std::ifstream::in);
if (infile.fail()) {
if (echo > 0) std::printf("# %s<%c> unable to find file \"%s\" for reading\n", __func__, write_read_delete, filename);
return 1; 
}
infile >> number;
if (echo > 1) std::printf("# %s<%c> found number=%d in file \"%s\" \n", __func__, write_read_delete, number, filename);
return 0;
} else
if ('d' == (write_read_delete | 32)) {
if (echo > 0) std::printf("# %s<%c> removes file \"%s\"\n", __func__, write_read_delete, filename);
return std::remove(filename);
} else {
if (echo > 0) std::printf("# %s<%c> only 'w'/'W'/write, 'r'/'R'/read or 'd'/'D'/delete implemented!\n", __func__, write_read_delete);
return -1; 
}
} 

} 
