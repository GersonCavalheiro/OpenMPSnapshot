#include "mex.h"
#if !defined(MATLAB_MEX_FILE) && defined(printf)
#undef printf
#endif
#include "ompGetThreadNum.h"
#include "ompGetThreadNum_types.h"
#include "m2c.c"
#include "omp.h"
#include "lib2mex_helper.c"
static void __ompGetThreadNum_api(mxArray **plhs, const mxArray ** prhs) {
int32_T              _n;
#pragma omp parallel
{
_n = ompGetThreadNum();
}
plhs[0] = copy_scalar_to_mxArray(&_n, mxINT32_CLASS);
}
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
if (nrhs == 0) {
if (nlhs > 1)
mexErrMsgIdAndTxt("ompGetThreadNum:TooManyOutputArguments",
"Too many output arguments for entry-point ompGetThreadNum.\n");
__ompGetThreadNum_api(plhs, prhs);
}
else
mexErrMsgIdAndTxt("ompGetThreadNum:WrongNumberOfInputs",
"Incorrect number of input variables for entry-point ompGetThreadNum.");
}
