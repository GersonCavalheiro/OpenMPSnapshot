#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <polybench.h>
#include "reg_detect.h"
static
void init_array(int maxgrid,
DATA_TYPE POLYBENCH_2D(sum_tang,MAXGRID,MAXGRID,maxgrid,maxgrid),
DATA_TYPE POLYBENCH_2D(mean,MAXGRID,MAXGRID,maxgrid,maxgrid),
DATA_TYPE POLYBENCH_2D(path,MAXGRID,MAXGRID,maxgrid,maxgrid))
{
int i, j;
for (i = 0; i < maxgrid; i++)
for (j = 0; j < maxgrid; j++) {
sum_tang[i][j] = (DATA_TYPE)((i+1)*(j+1));
mean[i][j] = ((DATA_TYPE) i-j) / maxgrid;
path[i][j] = ((DATA_TYPE) i*(j-1)) / maxgrid;
}
}
static
void print_array(int maxgrid,
DATA_TYPE POLYBENCH_2D(path,MAXGRID,MAXGRID,maxgrid,maxgrid))
{
int i, j;
for (i = 0; i < maxgrid; i++)
for (j = 0; j < maxgrid; j++) {
fprintf (stderr, DATA_PRINTF_MODIFIER, path[i][j]);
if ((i * maxgrid + j) % 20 == 0) fprintf (stderr, "\n");
}
fprintf (stderr, "\n");
}
static
void kernel_reg_detect(int niter, int maxgrid, int length,
DATA_TYPE POLYBENCH_2D(sum_tang,MAXGRID,MAXGRID,maxgrid,maxgrid),
DATA_TYPE POLYBENCH_2D(mean,MAXGRID,MAXGRID,maxgrid,maxgrid),
DATA_TYPE POLYBENCH_2D(path,MAXGRID,MAXGRID,maxgrid,maxgrid),
DATA_TYPE POLYBENCH_3D(diff,MAXGRID,MAXGRID,LENGTH,maxgrid,maxgrid,length),
DATA_TYPE POLYBENCH_3D(sum_diff,MAXGRID,MAXGRID,LENGTH,maxgrid,maxgrid,length))
{
int t, i, j, cnt;
#pragma scop
for (t = 0; t < _PB_NITER; t++)
{
for (j = 0; j <= _PB_MAXGRID - 1; j++)
for (i = j; i <= _PB_MAXGRID - 1; i++)
for (cnt = 0; cnt <= _PB_LENGTH - 1; cnt++)
diff[j][i][cnt] = sum_tang[j][i];
for (j = 0; j <= _PB_MAXGRID - 1; j++)
{
for (i = j; i <= _PB_MAXGRID - 1; i++)
{
sum_diff[j][i][0] = diff[j][i][0];
for (cnt = 1; cnt <= _PB_LENGTH - 1; cnt++)
sum_diff[j][i][cnt] = sum_diff[j][i][cnt - 1] + diff[j][i][cnt];
mean[j][i] = sum_diff[j][i][_PB_LENGTH - 1];
}
}
for (i = 0; i <= _PB_MAXGRID - 1; i++)
path[0][i] = mean[0][i];
for (j = 1; j <= _PB_MAXGRID - 1; j++)
for (i = j; i <= _PB_MAXGRID - 1; i++)
path[j][i] = path[j - 1][i - 1] + mean[j][i];
}
#pragma endscop
}
int main(int argc, char** argv)
{
int niter = NITER;
int maxgrid = MAXGRID;
int length = LENGTH;
POLYBENCH_2D_ARRAY_DECL(sum_tang, DATA_TYPE, MAXGRID, MAXGRID, maxgrid, maxgrid);
POLYBENCH_2D_ARRAY_DECL(mean, DATA_TYPE, MAXGRID, MAXGRID, maxgrid, maxgrid);
POLYBENCH_2D_ARRAY_DECL(path, DATA_TYPE, MAXGRID, MAXGRID, maxgrid, maxgrid);
POLYBENCH_3D_ARRAY_DECL(diff, DATA_TYPE, MAXGRID, MAXGRID, LENGTH, maxgrid, maxgrid, length);
POLYBENCH_3D_ARRAY_DECL(sum_diff, DATA_TYPE, MAXGRID, MAXGRID, LENGTH, maxgrid, maxgrid, length);
init_array (maxgrid,
POLYBENCH_ARRAY(sum_tang),
POLYBENCH_ARRAY(mean),
POLYBENCH_ARRAY(path));
polybench_start_instruments;
kernel_reg_detect (niter, maxgrid, length,
POLYBENCH_ARRAY(sum_tang),
POLYBENCH_ARRAY(mean),
POLYBENCH_ARRAY(path),
POLYBENCH_ARRAY(diff),
POLYBENCH_ARRAY(sum_diff));
polybench_stop_instruments;
polybench_print_instruments;
polybench_prevent_dce(print_array(maxgrid, POLYBENCH_ARRAY(path)));
POLYBENCH_FREE_ARRAY(sum_tang);
POLYBENCH_FREE_ARRAY(mean);
POLYBENCH_FREE_ARRAY(path);
POLYBENCH_FREE_ARRAY(diff);
POLYBENCH_FREE_ARRAY(sum_diff);
return 0;
}
