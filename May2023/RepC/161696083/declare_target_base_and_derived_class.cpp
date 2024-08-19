#include <new>
#include <vector>
#include <iostream>
#include "ompvv.h"
#include <omp.h>
#pragma omp declare target
class Surface {
public:
virtual double sag(double x, double y) = 0;
};
class S1 : public Surface {
public:
S1() : _devPtr(nullptr) {}
virtual double sag(double x, double y) override {
return x + y;
}
S1* cloneToDevice() {
S1* ptr;
#pragma omp target map(ptr)
{
ptr = new S1();
}
_devPtr = ptr;
return ptr;
}
private:
S1* _devPtr;
};
#pragma omp end declare target
int main() {
int errors = 0;
OMPVV_TEST_OFFLOADING;
S1 s;
Surface* devPtr = s.cloneToDevice();  
std::vector<double> in(10, 0.0);
for(int i=0; i<10; i++) {
in[i] = i;
}
std::vector<double> out(10, 0.0);
double* inptr = in.data();
double* outptr = out.data();
#pragma omp target teams distribute parallel for map(inptr[:10], outptr[:10]) is_device_ptr(devPtr)
for(int i=0; i<10; i++) {
outptr[i] = devPtr->sag(inptr[i], inptr[i]);
}
for(int i=0; i<10; i++) {
OMPVV_TEST_AND_SET(errors, out[i] != i * 2);
}
OMPVV_REPORT_AND_RETURN(errors);
}
