#pragma once
#include <math.h>
typedef struct complexx
{
double real;
double imag;
} complexx;
inline complexx complex_add(complexx num1, complexx num2)
{
complexx temp;
temp.real = num1.real + num2.real;
temp.imag = num1.imag + num2.imag;
return(temp);
}
inline complexx complex_subtract(complexx num1, complexx num2)
{
complexx temp;
temp.real = num1.real - num2.real;
temp.imag = num1.imag - num2.imag;
return(temp);
}
inline complexx complex_div(complexx num1, double divisor)
{
complexx temp;
temp.real = num1.real / divisor;
temp.imag = num1.imag / divisor;
return(temp);
}
inline complexx complex_mul(complexx num1, complexx num2)
{
complexx temp;
temp.real = num1.real * num2.real + num1.imag * num2.imag * -1;
temp.imag = num1.real * num2.imag + num1.imag * num2.real * -1;;
return(temp);
}
inline double get_complex_real(complexx complex_num)
{
return complex_num.real;
}
inline double get_complex_imag(complexx complex_num)
{
return complex_num.imag;
}
inline double complex_abs(complexx complex_num)
{
return sqrt(pow(complex_num.real, 2) + pow(complex_num.imag, 2));
}
