#include <cstdio>
#include <omp.h>
#include <string>

using std::string;

void log(string message, int a, int b, int thread_id) {
printf("%s: a = %d, b = %d. from thread#%d\n",
message.c_str(), a, b, thread_id);
}

int main() {
int a = 1, b = 2;

log("main thread before all", a, b, 0);

printf("FIRST REGION\n");
#pragma omp parallel num_threads(2) private(a) firstprivate(b)
{

int id = omp_get_thread_num();
log("before inc", a, b, id);
a += id;
b += id;
log(" after inc", a, b, id);
}

log("after first region", a, b, 0);

printf("SECOND REGION a shared, b private\n");
#pragma omp parallel num_threads(4) shared(a) private(b)
{

int id = omp_get_thread_num();
log("before dec", a, b, id);
a -= id;
b -= id;
log(" after dec", a, b, id);
}
}
