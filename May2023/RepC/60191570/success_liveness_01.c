int z = 10;
void foo(int y)
{
int task_res = 0;
int res2 = 0;
int x = 5;
int l = 5;
#pragma analysis_check assert live_in(task_res, x, y, z) live_out(task_res, z) dead(l)
#pragma omp task shared(task_res) private(l)
{
l = 10;
task_res += x + y + z + l;
}
res2 = task_res + z;
#pragma omp taskwait
res2 += task_res + y + l;
}
