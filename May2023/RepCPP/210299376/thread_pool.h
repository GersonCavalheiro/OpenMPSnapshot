#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace ctranslate2 {

class Job {
public:
virtual ~Job();
virtual void run() = 0;

void set_job_counter(std::atomic<size_t>& counter);

private:
std::atomic<size_t>* _counter = nullptr;
};

class JobQueue {
public:
JobQueue(size_t maximum_size);
~JobQueue();

size_t size() const;

void put(std::unique_ptr<Job> job);

std::unique_ptr<Job> get(const std::function<void()>& before_wait = nullptr);

void close();

private:
bool can_get_job() const;

mutable std::mutex _mutex;
std::queue<std::unique_ptr<Job>> _queue;
std::condition_variable _can_put_job;
std::condition_variable _can_get_job;
size_t _maximum_size;
bool _request_end;
};

class Worker {
public:
virtual ~Worker() = default;

void start(JobQueue& job_queue, int thread_affinity = -1);
void join();

protected:
virtual void initialize() {}

virtual void finalize() {}

virtual void idle() {}

private:
void run(JobQueue& job_queue);

std::thread _thread;
};

class ThreadPool {
public:
ThreadPool(size_t num_threads,
size_t maximum_queue_size = std::numeric_limits<size_t>::max(),
int core_offset = -1);

ThreadPool(std::vector<std::unique_ptr<Worker>> workers,
size_t maximum_queue_size = std::numeric_limits<size_t>::max(),
int core_offset = -1);

~ThreadPool();

void post(std::unique_ptr<Job> job);

size_t num_threads() const;

size_t num_queued_jobs() const;

size_t num_active_jobs() const;

Worker& get_worker(size_t index);
static Worker& get_local_worker();

private:
void start_workers(int core_offset);

JobQueue _queue;
std::vector<std::unique_ptr<Worker>> _workers;
std::atomic<size_t> _num_active_jobs;
};

}
