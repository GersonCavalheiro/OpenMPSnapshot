#pragma once

#include <pthread.h>
#include <unistd.h>
#include <deque>
#include <iostream>
#include <fstream>
#include <vector>
#include <errno.h>
#include <string.h>
#include <string>
#include <map>
#include <set>

#include "mutex.h"
#include "task.h"
#include "log/log.h"



const int DEFAULT_POOL_SIZE = 10;
const int STARTED = 0;
const int STOPPED = 1;

using namespace std;

class ThreadPool {
public:
static ThreadPool* getSingleInstance();

static void revokeSingleInstance();

private:
ThreadPool();
~ThreadPool();

private:
static ThreadPool* p_ThreadPool;

public:
int initialize_threadpool();
int destroy_threadpool();

int runningNumbers();

int getPoolCapacity();
int add_task(Task* task, const string& task_id);
int fetchResultByTaskID(const string task_id, TaskPackStruct& res);

void* execute_task(pthread_t thread_id);

private:
volatile int m_pool_state;

Mutex m_task_mutex;
Mutex m_finishMap_mutex;
Mutex m_taskMap_mutex;
CondVar m_task_cond_var;

std::deque<Task*> m_tasks;

std::vector<pthread_t> m_threads;
std::set<pthread_t> m_run_threads;

map<string, Task*> m_taskMap; 
map<string, TaskPackStruct> m_finishMap; 



};

