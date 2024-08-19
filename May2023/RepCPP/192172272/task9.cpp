#include "task9.h"

ITask::ResType Task9::Process()
{
const int N = 1000;
result = 0;

#pragma omp parallel for
for(int i = 1; i<N; ++i)
{
for(int j = 1; ((j<N) && (result==0)); ++j)
{
for(int k = 1; k<N; ++k)
{
if  ( ( (i*i) == (k*k + j*j) ) &&
((i + k + j) == N ))
{
result = i * j * k;
break;
}
}
}
}

return result;
}

std::string Task9::getDescriptionEng()
{
return std::string("Task 9\n"
"A Pythagorean triplet is a set of three natural numbers, a < b < c, for which,\n"
"a2+b2=c2\n"
"There exists exactly one Pythagorean triplet for which a + b + c = 1000.\n"
"Find the product abc.");
}

std::string Task9::getDescriptionRus()
{
return std::string("Задача 9\n"
"Тройка Пифагора - три натуральных числа a < b < c, для которых выполняется равенство\n"
"a2 + b2 = c2\n"
"Существует только одна тройка Пифагора, для которой a + b + c = 1000.\n"
"Найдите произведение abc.");

}
