#include "../include/test_functions.hpp"


double calculateTestFunc1CosinusPart(const Point& point) {
if (point.empty())
return 0;    

int i = 1;
double res = std::cos(point[0] / (double)1);
int N = (int) point.size();
for (i = 2; i < N + 1; ++i) {
res *= std::cos(point[i-1] / i); 
}
return res;
}

double calculateTestFunc2CosinusPart(const Point& point) {
double res = 0;
for (double i : point) {
res += std::cos(2 * M_PI * i);
}
return res;
}


double testFunc1(Point point) {
double sum_of_x_squared = std::accumulate(point.begin(), point.end(), 0, square<double>());
double cosinus_part = calculateTestFunc1CosinusPart(point);
return 1/40.f * sum_of_x_squared + 1 - cosinus_part;
}


double testFunc2(Point point) {
int n = (int) point.size();
double sum_of_x_squared = std::accumulate(point.begin(), point.end(), 0, square<double>());
double first_exponent_part = -20 * std::exp(-0.2f * std::sqrt(sum_of_x_squared / n));
double second_exponent_part = std::exp(calculateTestFunc2CosinusPart(point) / n);
return first_exponent_part - second_exponent_part + 20 + std::exp(1.0);
}