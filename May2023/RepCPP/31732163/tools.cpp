

#include "trinity/tools.h"

namespace trinity { namespace tools {

uint32_t hash(const uint32_t id) {
return uint32_t(((uint64_t) id * 279470273UL) % 4294967291UL);
}


int format(int num) {
return (num > 0 ? ((int) std::floor(std::log10(num))) + 1 : 0);
}


void showElapsed(Time& tic, const char* msg, int step) {
#pragma omp single
{
std::printf("%d. %s : \e[32m(%d ms)\e[0m\n",
step, msg, timer::elapsed_ms(tic));
std::fflush(stdout);
tic = timer::now();
}
}


void ltrim(std::string& line) {
size_t off = line.find_first_not_of(" \t\r\n");
if (off != std::string::npos) {
line.erase(0, off);
}
}


std::string basename(const std::string& s) {
std::string b = s;
size_t i = b.find_last_not_of('/');
if (i == std::string::npos) {
if (b[0] == '/') {
b.erase(1);
}
return b;
}

b.erase(i + 1, b.length() - i - 1);
i = b.find_last_of('/');
if (i != std::string::npos) {
b.erase(0, i + 1);
}
return b;
}


bool exists(const std::string& path) {

std::ifstream file(path, std::ios::in);
bool ok = file.good();
file.close();
return ok;
}


void abort(char option, const char* msg, const optparse::Parser& parser) {
std::printf("\nError: \e[41moption -%c: %s\e[0m\n", option, msg);
parser.print_help();
std::exit(EXIT_FAILURE);
}


bool equals(const char* s1, const char* s2) {
return not std::strcmp(s1, s2);
}


std::string getExt(const char* path) {
std::string file(path);
size_t last_dot = file.find_last_of('.');
return (last_dot != std::string::npos ? file.substr(last_dot + 1) : "");
}


std::ifstream& seekToLine(int nb, std::ifstream& file) {
assert(nb);
file.seekg(std::ios::beg);
for (int i = 0; i < nb - 1; ++i)
file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
return file;
}


bool isDigit(const char* arg) {
std::string s(arg);
return std::all_of(s.begin(), s.end(), ::isdigit); 
}


std::string rootOf(const std::string& path) {

std::string s = basename(path);
size_t const last_dot = s.find_last_of('.');
return s.substr(0, last_dot);
}

std::string testcase(const std::string& path) {

std::string s = basename(path);
size_t const last_dot = s.find_last_of('.');
return s.substr(0, last_dot-1);
}


std::string replaceExt(const std::string& fname, const std::string& ext) {

size_t last_dot = fname.find_last_of('.');
assert(last_dot != std::string::npos);
std::string root_ = fname.substr(0, last_dot);
std::stringstream nuw;
nuw << root_ << ext;
return nuw.str();
}


void separator() {
for (int i = 0; i < 64; ++i)
std::printf("-");
std::printf("\n");
}


}} 
