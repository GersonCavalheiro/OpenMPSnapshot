#include <iostream>
#include <vector>
void foo1 ()
{
std::vector<int> v;
for (int i=1; i<=5; i++) v.push_back(i);
std::vector<int>::const_iterator it;
#pragma omp parallel for ordered(1)
for (it = v.begin(); it < v.end(); ++it)
{
#pragma omp ordered depend(sink:it-1)
std::cout << *it << '\n';
#pragma omp ordered depend(source)
}
}
template <int N>
void foo2 ()
{
std::vector<int> v;
for (int i=1; i<=5; i++) v.push_back(i);
std::vector<int>::const_iterator it;
#pragma omp parallel for ordered(1)
for (it = v.begin(); it < v.end(); ++it)
{
#pragma omp ordered depend(sink:it-1)
std::cout << *it << '\n';
#pragma omp ordered depend(source)
}
}
template <typename T>
void foo3 ()
{
std::vector<T> v;
for (int i=1; i<=5; i++) v.push_back(i);
typename std::vector<T>::const_iterator it;
#pragma omp parallel for ordered(1)
for (it = v.begin(); it < v.end(); ++it)
{
#pragma omp ordered depend(sink:it-1)
std::cout << *it << '\n';
#pragma omp ordered depend(source)
}
}  
int main ()
{
foo2 <0> ();
foo3 <int> ();
}
