#include <iostream>
#include <stdlib.h>
#include <omp.h>
#include <cstdlib>
#include <ctime>
#include <math.h>
#include <fstream>
#include <string>
#include <direct.h>
#include <time.h>
#define NumThreads 4        
#define N 5                 
#define Q 12                
#define S 10 + 1            

using namespace std;


void arithmetic_series(float* series, float a, float d, bool sign) 
{
series[0] = a;
if (!sign)
#pragma omp parallel for
for(int i = 1; i < S; i++)
{
series[i] = a + i * d;
printf("\na = %.4f, d = %.4f, series[i] = %.4f, series = arithmetic, tid = %d", a, d, series[i], omp_get_thread_num());
}
else
#pragma omp parallel for
for(int i = 1; i < S; i++)
{
series[i] = a - i * d;
printf("\na = %.4f, d = %.4f, series[i] = %.4f, series = arithmetic, tid = %d", a, d, series[i], omp_get_thread_num());
}
series[S - 1] = 0;
}

void geometric_series(float* series, float a, float r, bool op) 
{
series[0] = a;

if (!op)
#pragma omp parallel for
for (int i = 1; i < S; i++)
{
series[i] = a * pow(r, i);
printf("\na = %.4f, d = %.4f, series[i] = %.4f, series = geometric, tid = %d", a, r, series[i], omp_get_thread_num());
}
else
#pragma omp parallel for
for (int i = 1; i < S; i++)
{
series[i] = a / pow(r, i);
printf("\na = %.4f, d = %.4f, series[i] = %.4f, series = geometric, tid = %d", a, r, series[i], omp_get_thread_num());
}
series[S - 1] = 1;
}

void harmonic_series(float* series, float a, float d, bool sign) 
{
series[0] = a;

if (!sign)
#pragma omp parallel for
for (int i = 1; i < S; i++)
{
series[i] = 1 / (a + i * d);
printf("\na = %.4f, d = %.4f, series[i] = %.4f, series = harmonic, tid = %d", a, d, series[i], omp_get_thread_num());
}
else
#pragma omp parallel for
for (int i = 1; i < S; i++)
{
series[i] = 1 / (a - i * d);
printf("\na = %.4f, d = %.4f, series[i] = %.4f, series = harmonic, tid = %d", a, d, series[i], omp_get_thread_num());
}
series[S - 1] = 2;
}

void create_questions(float papers[][Q][S], int n, int ee) 
{
omp_set_num_threads(NumThreads);
#pragma omp parallel
{
float series[S];
#pragma omp for schedule(dynamic, 4)
for (int i = 0; i < Q; i++)
{
srand((n + 5) * (i + 33) * (i + 1));
int tid = omp_get_thread_num();

if ((tid % 3) == 0)
{
geometric_series(series, 1 + (rand() % 100), 2 + (rand() % 10), (rand() % 2)); 
for (int j = 0; j < S; j++)
{
papers[n][i][j] = series[j];
}
}
else if ((tid % 3) == 1)
{
harmonic_series(series, 1 + (rand() % 100), 1 + (rand() % 7), (rand() % 2)); 
for (int j = 0; j < S; j++)
{
papers[n][i][j] = series[j];
}
}
else
{
arithmetic_series(series, (rand() % 100), 1 + (rand() % 30), (rand() % 2)); 
for (int j = 0; j < S; j++)
{
papers[n][i][j] = series[j];
}
}
}
}
}

void create_papers(float papers[][Q][S]) 
{
omp_set_num_threads(NumThreads);
#pragma omp parallel
{
#pragma omp for
for (int i = 0; i < N; i++)
{
int tid = omp_get_thread_num();
create_questions(papers, i, tid);
}
}
}

void print_papers(float papers[][Q][S])
{
ofstream MyFile;
ifstream MyFile2("./files/iteration.txt");
ofstream MyFile3("./files/new.txt");
int s_number;
MyFile2 >> s_number;
if(mkdir(("./files/Batch " + to_string(s_number)).c_str()) == -1)
cout << " Error : " << strerror(errno) << endl;
else
cout << "Folder Created";
for (int i = 0; i < N; i++)
{
MyFile.open("./files/Batch " + to_string(s_number) + "/Paper " + to_string(i + 1) + ".txt");
cout << "\nPaper Number: " << i + 1 << "\n" << endl;
MyFile << "\nPaper Number: " << i + 1 << "\n" << endl;
for (int j = 0; j < Q; j++)
{
cout << "Question Number " << j + 1 << ":\t";
MyFile << "Question Number " << j + 1 << ":\t";
if(j < 9)
MyFile << "\t";
srand((S + 3321) * (j + 53) * (i + 44));
int b = 1 + (rand() % (S - 2));
for (int k = 0; k < S - 1; k++)
{
if(k == b)
{
cout << "__________ ";
MyFile << "__________ ";
}
else
{
cout << papers[i][j][k] << " ";
MyFile << papers[i][j][k] << " ";
}
if (k != S - 2)
{
cout << "+ ";
MyFile << "+ ";
}
}
cout << "\tAnswer: " << papers[i][j][b];
MyFile << "\tAnswer: " << papers[i][j][b];
if(!papers[i][j][S - 1])
{
MyFile << "\t\t\tArithmetic Series";
cout << "\t\t\tArithmetic Series";
}
else if (papers[i][j][S - 1] == 1)
{
MyFile << "\t\t\tGeometric Series";
cout << "\t\t\tGeometric Series";
}
else 
{
MyFile << "\t\t\tHarmonic Series";
cout << "\t\t\tHarmonic Series";
}
cout << endl;
MyFile << endl;
}
MyFile.close();
}
int number2 = s_number + 1;
MyFile3<<number2;
MyFile2.close();
MyFile3.close();
remove("./files/iteration.txt");
rename("./files/new.txt", "./files/iteration.txt");
}

int main()
{
omp_set_num_threads(NumThreads * N);
omp_set_nested(1);
clock_t before = clock();
float papers[N][Q][S];
create_papers(papers); 
print_papers(papers);
clock_t after = clock();
clock_t run_time = after - before;
cout << "Running Time: " << run_time << ", Number Of Threads: " << NumThreads << endl;
}