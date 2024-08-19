#include "bnb.h"
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#endif
template <typename T1, typename T2>
void BnB_solver<T1, T2>::solve(T1 p)
{
queue<T2> q;
q.push(p.empty_sol());
T2 best = p.worst_sol();
while (true)
{
int threads_num;
if (q.size() >= 4)
threads_num = 4;
else
threads_num = q.size();
T2 data[threads_num];
for (int i = 0; i < threads_num; i++)
{
data[i] = q.front();
q.pop();
}
if (threads_num == 0)
break;
#pragma omp parallel for
for (int i = 0; i < threads_num; i++)
{
T2 s;
s = data[i];
if (s.is_feasible())
{
if (p.get_goal())
{ 
#pragma omp critical
if (s.get_cost() > best.get_cost())
best = s;
}
else
{ 
#pragma omp critical
if (s.get_cost() < best.get_cost())
best = s;
}
}
else
{
if (p.get_goal())
{ 
if (s.get_bound() > best.get_cost())
{
vector<T2> ret = p.expand(s);
#pragma omp critical
{
for (int l = 0; l < ret.size(); l++)
q.push(ret[l]);
}
}
}
else
{
if (s.get_bound() < best.get_cost())
{
vector<T2> ret = p.expand(s);
#pragma omp critical
{
for (int l = 0; l < ret.size(); l++)
q.push(ret[l]);
}
}
}
}
}
}
best.print_sol();
}
