#pragma once

namespace zero_finder
{

class Function1D
{
public:
virtual double operator()(double x) = 0;
double p1, p2;
void SetParameters(double a, double b);
};

class Function1D_7param
{
public:
virtual double operator()(double x) = 0;
double p1, p2, p3, p4, p5, p6, p7;
void SetParameters(double a, double b, double c, double d, double e, double f,
double g);
};

class FZero
{
private:
double a, c; 

public:
FZero(double a, double b); 
double FindZero(Function1D &f);
double FindZero7(Function1D_7param &f);
void SetBounds(double a, double b);
};
} 
