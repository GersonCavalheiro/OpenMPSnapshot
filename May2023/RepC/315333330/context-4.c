#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <openacc.h>
void
saxpy (int n, float a, float *x, float *y)
{
int i;
for (i = 0; i < n; i++)
{
y[i] = a * x[i] + y[i];
}
}
void
context_check (CUcontext ctx1)
{
CUcontext ctx2, ctx3;
CUresult r;
r = cuCtxGetCurrent (&ctx2);
if (r != CUDA_SUCCESS)
{
fprintf (stderr, "cuCtxGetCurrent failed: %d\n", r);
exit (EXIT_FAILURE);
}
if (ctx1 != ctx2)
{
fprintf (stderr, "new context established\n");
exit (EXIT_FAILURE);
}
ctx3 = (CUcontext) acc_get_current_cuda_context ();
if (ctx1 != ctx3)
{
fprintf (stderr, "acc_get_current_cuda_context returned wrong value\n");
exit (EXIT_FAILURE);
}
return;
}
int
main (int argc, char **argv)
{
cublasStatus_t s;
cublasHandle_t h;
CUcontext pctx;
CUresult r;
int i;
const int N = 256;
float *h_X, *h_Y1, *h_Y2;
float *d_X,*d_Y;
float alpha = 2.0f;
float error_norm;
float ref_norm;
acc_set_device_num (0, acc_device_nvidia);
r = cuCtxGetCurrent (&pctx);
if (r != CUDA_SUCCESS)
{
fprintf (stderr, "cuCtxGetCurrent failed: %d\n", r);
exit (EXIT_FAILURE);
}
h_X = (float *) malloc (N * sizeof (float));
if (h_X == 0)
{
fprintf (stderr, "malloc failed: for h_X\n");
exit (EXIT_FAILURE);
}
h_Y1 = (float *) malloc (N * sizeof (float));
if (h_Y1 == 0)
{
fprintf (stderr, "malloc failed: for h_Y1\n");
exit (EXIT_FAILURE);
}
h_Y2 = (float *) malloc (N * sizeof (float));
if (h_Y2 == 0)
{
fprintf (stderr, "malloc failed: for h_Y2\n");
exit (EXIT_FAILURE);
}
for (i = 0; i < N; i++)
{
h_X[i] = rand () / (float) RAND_MAX;
h_Y2[i] = h_Y1[i] = rand () / (float) RAND_MAX;
}
#pragma acc parallel copyin (h_X[0:N]), copy (h_Y2[0:N]) copy (alpha)
{
int i;
for (i = 0; i < N; i++)
{
h_Y2[i] = alpha * h_X[i] + h_Y2[i];
}
}
r = cuCtxGetCurrent (&pctx);
if (r != CUDA_SUCCESS)
{
fprintf (stderr, "cuCtxGetCurrent failed: %d\n", r);
exit (EXIT_FAILURE);
}
d_X = (float *) acc_copyin (&h_X[0], N * sizeof (float));
if (d_X == NULL)
{
fprintf (stderr, "copyin error h_Y1\n");
exit (EXIT_FAILURE);
}
d_Y = (float *) acc_copyin (&h_Y1[0], N * sizeof (float));
if (d_Y == NULL)
{
fprintf (stderr, "copyin error h_Y1\n");
exit (EXIT_FAILURE);
}
s = cublasCreate (&h);
if (s != CUBLAS_STATUS_SUCCESS)
{
fprintf (stderr, "cublasCreate failed: %d\n", s);
exit (EXIT_FAILURE);
}
context_check (pctx);
s = cublasSaxpy (h, N, &alpha, d_X, 1, d_Y, 1);
if (s != CUBLAS_STATUS_SUCCESS)
{
fprintf (stderr, "cublasSaxpy failed: %d\n", s);
exit (EXIT_FAILURE);
}
context_check (pctx);
acc_memcpy_from_device (&h_Y1[0], d_Y, N * sizeof (float));
context_check (pctx);
error_norm = 0;
ref_norm = 0;
for (i = 0; i < N; ++i)
{
float diff;
diff = h_Y1[i] - h_Y2[i];
error_norm += diff * diff;
ref_norm += h_Y2[i] * h_Y2[i];
}
error_norm = (float) sqrt ((double) error_norm);
ref_norm = (float) sqrt ((double) ref_norm);
if ((fabs (ref_norm) < 1e-7) || ((error_norm / ref_norm) >= 1e-6f))
{
fprintf (stderr, "math error\n");
exit (EXIT_FAILURE);
}
free (h_X);
free (h_Y1);
free (h_Y2);
acc_free (d_X);
acc_free (d_Y);
context_check (pctx);
s = cublasDestroy (h);
if (s != CUBLAS_STATUS_SUCCESS)
{
fprintf (stderr, "cublasDestroy failed: %d\n", s);
exit (EXIT_FAILURE);
}
context_check (pctx);
acc_shutdown (acc_device_nvidia);
r = cuCtxGetCurrent (&pctx);
if (r != CUDA_SUCCESS)
{
fprintf (stderr, "cuCtxGetCurrent failed: %d\n", r);
exit (EXIT_FAILURE);
}
if (pctx)
{
fprintf (stderr, "Unexpected context\n");
exit (EXIT_FAILURE);
}
return EXIT_SUCCESS;
}
