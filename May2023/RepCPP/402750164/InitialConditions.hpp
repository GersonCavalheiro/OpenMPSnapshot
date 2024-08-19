#include <vector>
#include <cmath>
#pragma once


class InitialConditions
{
public:
int n = 128;

double leftBorder = 0.0;
double rightBorder = 1.0;

double h = (rightBorder - leftBorder) / (n - 1.0);

double epsilon = 1e-8;

double k = 1.0 / h;

const double PI = 3.14159265358979323846;
const double PI2 = PI * PI;

bool isDebugMode = false;


double f(double x, double y)
{
return std::sin(PI * y) * (2.0 + (k * k + PI2) * (1.0 - x) * x);
}


double analyticSolution(double x, double y)
{
return (1.0 - x) * x * std::sin(PI * y);
}

void computeSolution(std::vector<double> &solution)
{
for (int i = 0; i < n; i++)
for (int j = 0; j < n; j++)
solution[i * n + j] = analyticSolution(i * h, j * h);
}

void setParameters()
{
h = (rightBorder - leftBorder) / (n - 1.0);
k = 1.0 / h;
}
};
