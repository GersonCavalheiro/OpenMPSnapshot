#include <iostream>
#include <locale>
#include <omp.h>

#define number_of_experiments 100

typedef double (*FuncPtr)(double*&, double*&, int&);

typedef double (*FuncPtr2)(double*&, double*&, double*&, int&);

typedef double (*FuncPtr3)(double*&, int&);

using namespace std;

double fill_omp_for(double*& m_arr, double*& m_2arr, int& n)
{
double t_start = omp_get_wtime();
#pragma omp parallel for
for (int i = 0; i < n; i++)
{
m_arr[i] = rand() % (i + 1);
m_2arr[i] = rand() % (i + 1);
}

return omp_get_wtime() - t_start;
}

double add_omp_for(double*& m_arr1, double*& m_arr2, double*& f_arr, int& n)
{
double t_start = omp_get_wtime();
#pragma omp parallel for
for (int i = 0; i < n; i++)
{
f_arr[i] = m_arr1[i] + m_arr2[i];
}

return omp_get_wtime() - t_start;
}

double add_omp_sect(double*& m_arr1, double*& m_arr2, double*& f_arr, int& n)
{
double t_start = omp_get_wtime();
int n_t = omp_get_num_threads() + 1;

int st = 0;
int s1 = n / n_t;
int s2 = n * 2 / n_t;
int s3 = n * 3 / n_t;
int se = n;

int i = 0;
#pragma omp parallel sections private(i)
{
#pragma omp section 
{
for (i = st; i < s1; i++) {
f_arr[i] = m_arr1[i] + m_arr2[i];
}
}
#pragma omp section 
{
if(n_t > 1)
for (i = s1; i < s2; i++)
f_arr[i] = m_arr1[i] + m_arr2[i];
}
#pragma omp section 
{
if (n_t > 2)
for (i = s2; i < s3; i++)
f_arr[i] = m_arr1[i] + m_arr2[i];
}
#pragma omp section 
{
if (n_t > 3) 
for (i = s3; i < se; i++)
f_arr[i] = m_arr1[i] + m_arr2[i];
}
}
return omp_get_wtime() - t_start;
}

double sum_omp_red(double*& m_arr, int& n)
{
double t_start = omp_get_wtime();
double s = 0.;
#pragma omp parallel for reduction(+:s)
for (int i = 0; i < n; i++) {
s += m_arr[i];
}
return omp_get_wtime() - t_start;
}

double sum_omp_crit(double*& m_arr, int& n)
{
double t_start = omp_get_wtime();
double s = 0;
#pragma omp parallel for 
for (int i = 0; i < n; i++) {
#pragma omp critical
s += m_arr[i];
}

return omp_get_wtime() - t_start;
}

double AvgTrustedInterval(double& avg, double*& times, int cnt)
{
double sd = 0., newAVg = 0.;
int newCnt = 0;
for (int i = 0; i < cnt; i++)
sd += (times[i] - avg) * (times[i] - avg);
sd /= (cnt - 1.0);
sd = sqrt(sd);
for (int i = 0; i < cnt; i++)
{
if (avg - sd <= times[i] && times[i] <= avg + sd)
{
newAVg += times[i];
newCnt++;
}
}
if (newCnt == 0) newCnt = 1;
return newAVg / newCnt;
}

void test_functions(void** Functions, std::string(Function_names)[7])
{
double* times = new double[number_of_experiments] {0};

for (int size = 100000; size <= 250000; size += 50000)
{
std::cout << "\n\t\t\t Число элементов в векторах : " << size << std::endl;

double* arr1 = new double[size],* arr2 = new double[size];

double* arr_sum = new double[size];

for (int threads = 1; threads <= 4; threads++)
{
omp_set_num_threads(threads);

for (int j = 0; j < 5; j++)
{

for (int i = 0; i < number_of_experiments; i++)
{

if (threads == 1)
{
if (j == 0)
times[i] = ((*(FuncPtr)Functions[j])(arr1, arr2, size));
else if (j == 1)
times[i] = ((*(FuncPtr2)Functions[j])(arr1, arr2, arr_sum, size));
else if (j == 3)
times[i] = ((*(FuncPtr3)Functions[j])(arr_sum, size));
}
else
{
if (j == 0)
times[i] = ((*(FuncPtr)Functions[j])(arr1, arr2, size));
else if (j > 1 && j < 3)
times[i] = ((*(FuncPtr2)Functions[j])(arr1, arr2, arr_sum, size));
else if (j > 2)
times[i] = ((*(FuncPtr3)Functions[j])(arr_sum, size));
}
}
int t = 0;
for (auto i = 0; i < number_of_experiments; i++)
t += times[i];
double avg = t / number_of_experiments, avg_t = AvgTrustedInterval(avg, times, number_of_experiments);

cout << "Потоков - " << threads << " " << Function_names[j] << " заняло " << avg_t * 1000 << std::endl;

}
}
}
delete[] times;
}

int main()
{
setlocale(LC_ALL, "RUS");

void** Functions = new void* [5] {fill_omp_for, add_omp_for, add_omp_sect, sum_omp_red, sum_omp_crit};

std::string Function_names[5]{ "Заполнение одномерного массива(Параллельный FOR)",
"Сложение двух одномерных массивов(Параллельный FOR)", "Сложение двух одномерных массивов(Параллельный Sections)",
"Подсчет суммы всех элементов итогового массива (Параллельный с использованием редукторов)",
"Подсчет суммы всех элементов итогового массива (Параллельный с использованием критических секций)" };

test_functions(Functions, Function_names);
}