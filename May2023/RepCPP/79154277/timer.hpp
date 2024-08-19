#pragma once


#include <chrono>


namespace advscicomp {


class Timer
{
public:
Timer() : start_time(std::chrono::system_clock::now())
{ }

~Timer()
{
std::cout << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start_time).count() << std::endl;
}
private:
std::chrono::time_point<std::chrono::system_clock> start_time;
};


} 


