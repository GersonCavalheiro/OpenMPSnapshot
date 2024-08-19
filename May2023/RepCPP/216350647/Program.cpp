#include <iostream>
#include <algorithm>
#include <functional>
#include <iomanip>
#include <omp.h>
#include <sstream>

namespace MyFunctions {
constexpr double PI = 3.14159265358979;
auto f = [](double x) {return sin(x) - x; };
auto g = [](double x) {return 2 * x - 2; };
auto h = [](double x) {return sin(x - 1); };

auto iG = [](double a, double b) {return b * b - a * a - 2 * (b - a); };
auto iH = [](double a, double b) {return cos(a - 1) - cos(b - 1); };
}

class Program {
public:
using Function = std::function<double(double)>;
static void Main(int numberOfThreads, double a, int N) {
PreSet(numberOfThreads);

std::cout << "First integrated: " << Integrate(a, 1, N, MyFunctions::g) << std::endl;
std::cout << "First integrated (4 | N): " << QuadIntegrate(a, 1, N, MyFunctions::g) << std::endl;
std::cout << "Real: " << MyFunctions::iG(a, 1) << std::endl;
std::cout << "Second integrated: " << Integrate(1, 1 + MyFunctions::PI / 2, N, MyFunctions::h) << std::endl;
std::cout << "Real: " << MyFunctions::iH(1, 1 + MyFunctions::PI / 2) << std::endl;

}
private:
static double Integrate(double a, double b, int n, Function f) {
double h = (b - a) / n;
double result = f(a) + f(b);

#pragma omp parallel for reduction(+ : result)
for (int i = 1; i < n; i++) {
result += 2 * f(i * h + a) * (i % 2  + 1);
}
return result * (h / 3);
}

static double QuadIntegrate(double a, double b, int n, Function f) {
double result = 0;
int difference = (b - a) / 4;
n /= 4;
#pragma omp parallel sections
{
#pragma omp section 
{
#pragma omp atomic
result += Integrate(a, a + difference, n, f);
} 

#pragma omp section 
{
#pragma omp atomic
result += Integrate(a + difference, a + 2 * difference, n, f);
}

#pragma omp section 
{
#pragma omp atomic
result += Integrate(a + 2 * difference, a + 3 * difference, n, f);
}

#pragma omp section 
{
#pragma omp atomic
result += Integrate(a + 3 * difference, b, n, f);
}
}
return result;
}
static void PreSet(int n) {
std::cout << std::setprecision(12);
omp_set_num_threads(n);
omp_set_nested(true);
}
};

int main(int argc, char** argv) {
if (argc != 4) {
std::cout << "Wrong number of parametrs" << std::endl;
}
else {
Program::Main(strtol(argv[1], nullptr, 10), strtod(argv[2], nullptr), strtol(argv[3], nullptr, 10));
system("pause");
}
return 0;
}