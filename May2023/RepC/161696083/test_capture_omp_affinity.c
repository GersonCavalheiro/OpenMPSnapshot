#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include <omp.h>
#include "ompvv.h"
#define BUFFER_STORE 80
int main() {
int i, n, errors, thrd_num;
size_t nchars, max_req_store;
char **buffer;
n = omp_get_num_procs();
buffer = (char **)malloc( sizeof(char *) * n );
for (i = 0; i < n; i++) { 
buffer[i]=(char *)malloc( sizeof(char) * BUFFER_STORE); 
}
errors = 0;
max_req_store = 0;
#pragma omp parallel private(thrd_num,nchars) reduction(max:max_req_store)
{
if(omp_get_num_threads()>n) exit(1); 
thrd_num=omp_get_thread_num();
nchars=omp_capture_affinity(buffer[thrd_num],(size_t)BUFFER_STORE,NULL);
if(nchars > max_req_store) {
max_req_store=nchars;
}
}
OMPVV_TEST_AND_SET_VERBOSE(errors, max_req_store >= BUFFER_STORE);
OMPVV_ERROR_IF(max_req_store >= BUFFER_STORE, "Caution: Affinity string truncated, increase buffer size");
for(i = 0; i < n; i++){
free(buffer[i]);
}
free (buffer);
OMPVV_REPORT_AND_RETURN(errors);
}
