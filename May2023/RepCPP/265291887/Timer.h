#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif
#include <chrono>
#include <thread>

using sec_t   = std::chrono::seconds;
using milli_t = std::chrono::milliseconds;
using micro_t = std::chrono::microseconds;
using nano_t  = std::chrono::nanoseconds;

template<typename duration_t>
class Timer{
using clock_t = std::chrono::system_clock;

public:
Timer() : _start{ getNow() }{ }

uint64_t getNow() {
return std::chrono::duration_cast<duration_t>(clock_t::now().time_since_epoch()).count();
}

void reset(){ _start = getNow(); }

uint64_t elapsed() { return getNow() - _start; }

template<typename time_T>
void sleep(uint64_t t){
#ifdef _OPENMP
int tid = omp_get_thread_num();
int team_id = omp_get_team_num();
int max_tid = omp_get_num_threads() - 1;
#endif
#ifdef _OPENMP
printf("start: (%d|%d)\n", team_id, tid);
#endif 
std::this_thread::sleep_for(time_T(t));
#ifdef _OPENMP
printf("end: (%d|%d)\n", team_id, tid);
#endif    
}

private:
uint64_t _start;
};