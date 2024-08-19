#include <iostream>
#include <locale>
#include <omp.h>

using namespace std;

int main()
{
int inp_threads;
setlocale(LC_ALL, "RUS");
cout << "    - " << omp_get_max_threads() << "\n   : " << endl;
cin >> inp_threads;
omp_set_num_threads(inp_threads);
#pragma omp parallel
{
cout << "(cout) Hello World!   " << omp_get_thread_num() << endl;
printf("(printf) Hello World!  %d \n", omp_get_thread_num());
}
return 0;
}