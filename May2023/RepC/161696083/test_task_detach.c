#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 1024
int test_callback(omp_event_handle_t event) {
omp_fulfill_event(event);
return 1;
}
int test_task_detach() {
OMPVV_INFOMSG("test_task_detach");
int errors = 0, x = 0, y = 0;
int num_threads = -1, record_x = -1, record_y = -1;
omp_event_handle_t flag_event;
#pragma omp parallel
#pragma omp single
{
#pragma omp task depend(out: y) detach(flag_event)
{
y++;
}
#pragma omp task
{
x = 1;
#pragma omp flush
test_callback(flag_event);
}
#pragma omp task depend(inout: y)
{
#pragma omp flush
record_x = x;
record_y = y;
num_threads = omp_get_num_threads();
}
}
OMPVV_ERROR_IF(num_threads < 0, "Test ran with invalid number of teams (less than zero)");
OMPVV_WARNING_IF(num_threads == 1, "Test ran with one thread, so the results are not conclusive");
OMPVV_TEST_AND_SET_VERBOSE(errors, record_x != 1);
OMPVV_ERROR_IF(record_x == 0, "Dependent task preceded event-fulfilling task, so detach did not work correctly.");
OMPVV_ERROR_IF(record_y == 0, "Dependent task preceded detached task body, so depend did not work correctly.");
OMPVV_ERROR_IF(record_x == -1, "Event-fulfilling task's recording variable was not set in the final task.");
OMPVV_ERROR_IF(record_y == -1, "Detached task's recording variable was not set in the final task.");
return errors;
}
int main() {
int errors = 0;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_task_detach());
OMPVV_REPORT_AND_RETURN(errors);
}
