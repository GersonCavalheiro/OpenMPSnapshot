#pragma once

#include <chrono> 

namespace pass
{

class stopwatch
{
public:

void start() noexcept;


std::chrono::nanoseconds get_elapsed() const noexcept;

private:
std::chrono::steady_clock::time_point start_time;
};
} 
