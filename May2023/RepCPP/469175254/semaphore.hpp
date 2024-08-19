#pragma once

#include <condition_variable>
#include <mutex>


class Semaphore {
const size_t num_permissions;
size_t avail;
std::mutex m;
std::condition_variable cv;
public:

explicit Semaphore(const size_t& num_permissions = 1) : num_permissions(num_permissions), avail(0) { }


Semaphore(const Semaphore& s) : num_permissions(s.num_permissions), avail(s.avail) { }

void acquire() {
std::unique_lock<std::mutex> lk(m);
cv.wait(lk, [this] { return avail > 0; });
avail--;
lk.unlock();
}

void release() {
m.lock();
avail++;
m.unlock();
cv.notify_one();
}

size_t available() const {
return avail;
}
};