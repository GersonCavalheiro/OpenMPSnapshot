#include "constants.hpp"

#pragma omp declare target
int kernel(int xi, int yi) {
double x0 = xmin + xi * dx;
double y0 = ymin + yi * dy;
double x = 0.0;
double y = 0.0;
int i = 0;
while (x*x + y*y < 4.0 && i < max_iters) {
double xtemp = x*x - y*y + x0;
y = 2 * x * y + y0;
x = xtemp;
i++;
}
return i;
}
#pragma omp end declare target
