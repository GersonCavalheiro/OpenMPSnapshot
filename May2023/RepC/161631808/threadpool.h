#pragma once
#include <mutex>
#include <vector>
#include <functional>
#include <condition_variable>
#include <atomic>
#include <memory>
typedef std::function<void()> Task;
class StateData
{
public:
void Wait();
private:
friend class ThreadPool;
private:
StateData();
void Complete();
private:
std::mutex Mutex;
std::condition_variable CompleteCondition;
std::atomic_bool IsCompleted;
};
typedef std::shared_ptr<StateData> State;
struct StatedTask
{
Task  Execute;
State State;
};
typedef std::vector<StatedTask> TaskQueue;
class ThreadPool
{
public:
ThreadPool();
explicit ThreadPool(const int threadsCount);
~ThreadPool();
State AddTask(const Task& task);
private:
void ThreadContext();
inline void InitThreads(); 
private:
unsigned int                            ThreadsCount;
std::mutex                              TaskMutex;
std::condition_variable                 GotTasksCondition;
std::vector<std::thread>                Threads;
TaskQueue                               Tasks;
std::atomic_bool                        StopExecutionFlag;
};
void ThreadPool::InitThreads()
{
StopExecutionFlag = false;
for (auto& t : Threads)
t = std::thread(&ThreadPool::ThreadContext, this);
}
