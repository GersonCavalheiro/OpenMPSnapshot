#include <iostream>
#include <cmath>
#include <sstream>
#include <omp.h>
using namespace std;
void sum_2 (char* output, const long unsigned int d, const long unsigned int n, const int thread_count)
{
char fmt_str[200];
sprintf(fmt_str,"%%.%ulf\n",d);
double sum = 1.0;
#pragma omp parallel for reduction(+:sum) num_threads(thread_count)
for (long unsigned int i = 2; i <= n; i++)
{
sum += (1.0/(double)i);
}
sprintf(output,fmt_str,sum);
long unsigned int index = 0;
for (long unsigned int i = 0; i < d; i++)
{
if (output[i] == '.')
{
index = i;
break;
}
}
output[index] = ',';
}
int main(int argc, char *argv[]) {
if (argc-1 < 1)
{
cerr << "Usage:> ./solution [num_threads] < [input_file] > [output_file]" << endl;
exit(EXIT_FAILURE);
}
long unsigned int d, n;
int thread_count;
thread_count = atoi(argv[1]);
cin >> d >> n;
char output[d + 10]; 
sum_2(output, d, n, thread_count);
cout << output;
return 0;
}