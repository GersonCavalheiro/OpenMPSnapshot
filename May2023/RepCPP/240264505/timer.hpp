

#ifndef LBT_TIMER
#define LBT_TIMER
#pragma once

#include <chrono>


namespace lbt {


class Timer {
public:

Timer() noexcept;


void start() noexcept;


double stop() noexcept;


double getRuntime() const noexcept;

private:
std::chrono::high_resolution_clock::time_point start_time; 
std::chrono::high_resolution_clock::time_point stop_time;
std::chrono::duration<double> start_to_stop;               
double runtime;                                            
};

}

#endif 
