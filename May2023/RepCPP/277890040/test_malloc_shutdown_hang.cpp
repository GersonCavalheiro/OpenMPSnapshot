

#define HARNESS_CUSTOM_MAIN 1
#include "harness.h"

#include <tbb/task.h>
#include <tbb/scalable_allocator.h>
#include <tbb/task_scheduler_init.h>

const int MAX_DELAY = 5;
struct GlobalObject {
~GlobalObject() {
Harness::Sleep(rand( ) % MAX_DELAY);
}
} go;

void allocatorRandomThrashing() {
const int ARRAY_SIZE = 1000;
const int MAX_ITER = 10000;
const int MAX_ALLOC = 10 * 1024 * 1024;

void *arr[ARRAY_SIZE] = {0};
for (int i = 0; i < rand() % MAX_ITER; ++i) {
for (int j = 0; j < rand() % ARRAY_SIZE; ++j) {
arr[j] = scalable_malloc(rand() % MAX_ALLOC);
}
for (int j = 0; j < ARRAY_SIZE; ++j) {
scalable_free(arr[j]);
arr[j] = NULL;
}
}
}

struct AllocatorThrashTask : tbb::task {
tbb::task* execute() __TBB_override {
allocatorRandomThrashing();
return NULL;
}
};

void hangOnExitReproducer() {
const int P = tbb::task_scheduler_init::default_num_threads();
for (int i = 0; i < P-1; i++) {
tbb::task::enqueue(*new (tbb::task::allocate_root()) AllocatorThrashTask());
}
}

#if (_WIN32 || _WIN64) && !__TBB_WIN8UI_SUPPORT
#include <process.h> 
void processSpawn(const char* self) {
_spawnl(_P_WAIT, self, self, "1", NULL);
}
#elif __linux__ || __APPLE__
#include <unistd.h> 
#include <sys/wait.h> 
void processSpawn(const char* self) {
pid_t pid = fork();
if (pid == -1) {
REPORT("ERROR: fork failed.\n");
} else if (pid == 0) { 
execl(self, self, "1", NULL);
REPORT("ERROR: exec never returns\n");
exit(1);
} else { 
int status;
waitpid(pid, &status, 0);
}
}
#else
void processSpawn(const char* ) {
REPORT("Known issue: no support for process spawn on this platform.\n");
REPORT("done\n");
exit(0);
}
#endif

#if _MSC_VER && !__INTEL_COMPILER
#pragma warning (push)
#pragma warning (disable: 4702)  
#endif

HARNESS_EXPORT
int main(int argc, char* argv[]) {
ParseCommandLine( argc, argv );

if (argc == 2 && strcmp(argv[1],"1") == 0) {
hangOnExitReproducer();
return 0;
}

const int EXEC_TIMES = 100;
const char* self = argv[0];
for (int i = 0; i < EXEC_TIMES; i++) {
processSpawn(self);
}

#if _MSC_VER && !__INTEL_COMPILER
#pragma warning (pop)
#endif

REPORT("done\n");
return 0;
}

