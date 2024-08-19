#pragma once
#include <chrono>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <string>
#include <vector>

#define monolish_func __FUNCTION__

#if defined USE_MPI
#include <mpi.h>
#endif

namespace monolish {

class Logger {
private:
Logger() = default;

~Logger() {
if (pStream != &std::cout) {
delete pStream;
}
};

std::vector<std::string> calls;
std::vector<std::chrono::system_clock::time_point> times;
std::string filename;
std::ostream *pStream;

public:
size_t LogLevel = 0;

Logger(const Logger &) = delete;
Logger &operator=(const Logger &) = delete;
Logger(Logger &&) = delete;
Logger &operator=(Logger &&) = delete;

[[nodiscard]] static Logger &get_instance() {
static Logger instance;
return instance;
}


void set_log_level(size_t L) {
if (3 < L) { 
throw std::runtime_error("error bad LogLevel");
}
LogLevel = L;
}


void set_log_filename(const std::string file) {
filename = file;

pStream = new std::ofstream(filename);
if (pStream->fail()) {
delete pStream;
pStream = &std::cout;
}
}

void solver_in(const std::string func_name);
void solver_out();

void func_in(const std::string func_name);
void func_out();

void util_in(const std::string func_name);
void util_out();
};
} 
