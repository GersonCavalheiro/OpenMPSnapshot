void
test ()
{
int a[100], i, j, z;
#pragma acc parallel loop collapse (2)
for (i = 0; i < 100; i++)
for (j = 0; j < 10; j++)
;
#pragma acc parallel loop gang
for (i = 0; i < 100; i++)
;
#pragma acc parallel loop worker
for (i = 0; i < 100; i++)
for (j = 0; j < 10; j++)
;
#pragma acc parallel loop vector
for (i = 0; i < 100; i++)
for (j = 0; j < 10; j++)
;
#pragma acc parallel loop seq
for (i = 0; i < 100; i++)
for (j = 0; j < 10; j++)
;
#pragma acc parallel loop auto
for (i = 0; i < 100; i++)
for (j = 0; j < 10; j++)
;
#pragma acc parallel loop tile (2, 3)
for (i = 0; i < 100; i++)
for (j = 0; j < 10; j++)
;
#pragma acc parallel loop independent
for (i = 0; i < 100; i++)
;
#pragma acc parallel loop private (z)
for (i = 0; i < 100; i++)
z = 0;
#pragma acc kernels loop collapse (2)
for (i = 0; i < 100; i++)
for (j = 0; j < 10; j++)
;
#pragma acc kernels loop gang
for (i = 0; i < 100; i++)
;
#pragma acc kernels loop worker
for (i = 0; i < 100; i++)
for (j = 0; j < 10; j++)
;
#pragma acc kernels loop vector
for (i = 0; i < 100; i++)
for (j = 0; j < 10; j++)
;
#pragma acc kernels loop seq
for (i = 0; i < 100; i++)
for (j = 0; j < 10; j++)
;
#pragma acc kernels loop auto
for (i = 0; i < 100; i++)
for (j = 0; j < 10; j++)
;
#pragma acc kernels loop tile (2, 3)
for (i = 0; i < 100; i++)
for (j = 0; j < 10; j++)
;
#pragma acc kernels loop independent
for (i = 0; i < 100; i++)
;
#pragma acc kernels loop private (z)
for (i = 0; i < 100; i++)
z = 0;
}
