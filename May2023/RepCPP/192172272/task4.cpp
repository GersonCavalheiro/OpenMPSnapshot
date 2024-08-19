#include "task4.h"
#include "omp.h"

ITask::ResType Task4::Process()
{
result = 0;

#pragma omp parallel for
for(size_t iii = 100; iii < 1000; ++iii)
{
for(size_t jjj = 100; jjj < 1000; ++jjj)
{
auto _mul = iii*jjj;
if( isPalendrom(_mul) && (_mul > result) )
{
result = _mul;
}
}
}

return result;
}

std::string Task4::getDescriptionEng()
{
return std::string("Task 4\n"
"A palindromic number reads the same both ways.\n"
"The largest palindrome made from the product of two 2-digit numbers is 9009 = 91x99\n"
"Find the largest palindrome made from the product of two 3-digit numbers.");
}

std::string Task4::getDescriptionRus()
{
return std::string("Задача 4\n"
"Число-палиндром с обеих сторон (справа налево и слева направо) читается одинаково."
"Самое большое число-палиндром, полученное умножением двух двузначных чисел – 9009 = 91 × 99."
"Найдите самый большой палиндром, полученный умножением двух трехзначных чисел.");
}
