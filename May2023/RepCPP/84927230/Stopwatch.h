#pragma once

#include <chrono>

using namespace std::chrono;

class Stopwatch {
system_clock::time_point m_start;
system_clock::duration m_elapsed;
bool m_isRunning;

public:
Stopwatch() : m_isRunning{ false }, m_elapsed{ 0 } {}

void Start() {
m_elapsed = system_clock::duration::zero(); m_isRunning = true; m_start = system_clock::now();
}
void Restart() {
if (!m_isRunning) {
m_isRunning = true; m_start = system_clock::now();
}
}
void Stop() {
if (m_isRunning) {
system_clock::time_point end = system_clock::now(); m_elapsed += end - m_start; m_isRunning = false;
}
}
void Reset() {
m_elapsed = system_clock::duration::zero(); m_isRunning = false;
}
system_clock::duration GetSplitTime() const {
system_clock::duration result(0);
if (m_isRunning) {
system_clock::time_point end = system_clock::now(); result = end - m_start;
}
return result;
}
double GetSplitTimeSeconds() const {
nanoseconds ns(GetSplitTime());
return (double)(ns.count()*nanoseconds::period::den)/nanoseconds::period::den;
}
double GetSplitTimeMilliseconds() const {
nanoseconds ns(GetSplitTime());
return ns.count()/1000000.0;
}
long long GetSplitTimeNanoseconds() const {
nanoseconds ns(GetSplitTime());
return ns.count();
}
system_clock::duration GetElapsedTime() const {
system_clock::duration result = m_elapsed;
if (m_isRunning) {
system_clock::time_point end = system_clock::now(); result += end - m_start;
}
return result;
}
double GetElapsedTimeSeconds() const {
nanoseconds ns(GetElapsedTime());
return (double)(ns.count()*nanoseconds::period::den)/nanoseconds::period::den;
}
double GetElapsedTimeMilliseconds() const {
nanoseconds ns(GetElapsedTime());
return ns.count()/1000000.0;
}
long long GetElapsedTimeNanoseconds() const {
nanoseconds ns(GetElapsedTime());
return ns.count();
}
};
