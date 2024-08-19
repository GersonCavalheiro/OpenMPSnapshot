#pragma once
#include <vector>
#include <chrono>
#include <algorithm>
#include <mutex>
class Timer {
std::chrono::time_point<std::chrono::system_clock> start, end;
std::vector<long> times;
std::mutex values_mutex;
public:
enum class Mode {
Single,
Median
};
Mode mode;
Timer(Mode mode) : mode(mode) {
};
Timer& operator=(const Timer& other) {
if(&other == this)
return *this;
this->mode = other.mode;
this->times = other.times;
this->start = other.start;
this->end = other.end;
return *this;
}
Timer(const Timer& other) {
this->mode = other.mode;
this->times = other.times;
this->start = other.start;
this->end = other.end;
};
void Start() {
start = std::chrono::system_clock::now();
};
void Stop() {
end = std::chrono::system_clock::now();
if(mode != Mode::Single) {
values_mutex.lock();
times.push_back((std::chrono::duration_cast<std::chrono::nanoseconds>(end-start)).count());
values_mutex.unlock();
}
};
long Get() {
if(mode == Mode::Single) {
return (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start)).count();
} else {
std::sort(times.begin(), times.end());
return times[times.size()/2];
}
};
long GetAvg() {
if(mode == Mode::Single) {
return (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start)).count();
} else {
unsigned long long sum = 0;
for (int i = 0; i < times.size(); ++i) {
sum += times[i];
}
return sum/times.size();
}
};
void PushTime(long time) {
values_mutex.lock();
times.push_back(time);
values_mutex.unlock();
};
};
