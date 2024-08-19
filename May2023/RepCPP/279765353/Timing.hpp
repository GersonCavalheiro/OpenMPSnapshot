
#pragma once
#ifndef TIMING_HPP
#define TIMING_HPP
#include <chrono>

class Timing {
public:
Timing();

double getRunTime(bool cont=false);

private:
std::chrono::steady_clock::time_point startTime;
};

#endif
