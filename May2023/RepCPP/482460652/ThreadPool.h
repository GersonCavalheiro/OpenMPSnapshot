#include <thread>
#include "minirt/minirt.h"
#include <queue>
#include <mutex>
#include <condition_variable>
#pragma once

using namespace minirt;
template <typename T>
class ThreadSafeQueue
{
private:
std::queue<std::unique_ptr<T>> queue;
std::mutex lock;
std::condition_variable cond;
bool stop = false;
size_t addedJobs = 0;
size_t completedJobs = 0;

public:
void Push(std::unique_ptr<T> job);

std::unique_ptr<T> Pop();

void Stop();

void CompleteJob();
};

struct Point
{
int x;
int y;

Point(int x, int y) : x(x), y(y) {}
};

class RetraceJob
{
private:
Scene &scene;
Image &image;
ViewPlane &viewPlane;
Point cur_point;
size_t numOfSamples;

public:
RetraceJob(Scene &scene, Image &image, ViewPlane &viewPlane, Point &curPoint, size_t numOfSamples);

void Execute();
};

class ThreadPool
{
private:
size_t PoolSize = 1;
ThreadSafeQueue<RetraceJob> queue;
std::vector<std::thread *> workers;

public:
ThreadPool() = default;

explicit ThreadPool(size_t poolSize);

void ThreadFunc();

void AddJob(std::unique_ptr<RetraceJob> job);

void Join();
};
