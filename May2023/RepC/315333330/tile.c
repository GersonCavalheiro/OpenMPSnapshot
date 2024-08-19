#include <stdbool.h>
int
main ()
{
int i, j, k, *a, b;
#pragma acc parallel loop tile (10)
for (i = 0; i < 100; i++)
;
#pragma acc parallel loop tile (*)
for (i = 0; i < 100; i++)
;
#pragma acc parallel loop tile (10, *)
for (i = 0; i < 100; i++)
for (j = 0; j < 100; j++)
;
#pragma acc parallel loop tile (10, *, i) 
for (i = 0; i < 100; i++)
for (j = 0; j < 100; j++)
for (k = 0; k < 100; k++)
;
#pragma acc parallel loop tile 
for (i = 0; i < 100; i++)
;  
#pragma acc parallel loop tile () 
for (i = 0; i < 100; i++)
;
#pragma acc parallel loop tile (,1) 
for (i = 0; i < 100; i++)
;
#pragma acc parallel loop tile (,,) 
for (i = 0; i < 100; i++)
;
#pragma acc parallel loop tile (1.1) 
for (i = 0; i < 100; i++)
;
#pragma acc parallel loop tile (-3) 
for (i = 0; i < 100; i++)
;
#pragma acc parallel loop tile (10,-3) 
for (i = 0; i < 100; i++)
for (j = 0; j < 100; j++)
;
#pragma acc parallel loop tile (-100,10,5) 
for (i = 0; i < 100; i++)
for (j = 0; j < 100; j++)
for (k = 0; k < 100; k++)
;
#pragma acc parallel loop tile (1,true)
for (i = 0; i < 100; i++)
for (j = 0; j < 100; j++)
;
#pragma acc parallel loop tile (*a, 1) 
for (i = 0; i < 100; i++)
for (j = 0; j < 100; j++)
;
#pragma acc parallel loop tile (1, b) 
for (i = 0; i < 100; i++)
for (j = 0; j < 100; j++)
;
#pragma acc parallel loop tile (b, 1) 
for (i = 0; i < 100; i++)
for (j = 0; j < 100; j++)
;
return 0;
}
void par (void)
{
int i, j, k;
#pragma acc parallel
{
#pragma acc loop tile 
for (i = 0; i < 10; i++)
{ }
#pragma acc loop tile() 
for (i = 0; i < 10; i++)
{ }
#pragma acc loop tile(1) 
for (i = 0; i < 10; i++)
{ }
#pragma acc loop tile(*) 
for (i = 0; i < 10; i++)
{ }
#pragma acc loop tile(2) 
for (i = 0; i < 10; i++)
{
for (j = 1; j < 10; j++)
{ }
}
#pragma acc loop tile(-2)  
for (i = 1; i < 10; i++)
{ }
#pragma acc loop tile(i)  
for (i = 1; i < 10; i++)
{ }
#pragma acc loop tile(2, 2, 1)
for (i = 1; i < 3; i++)
{
for (j = 4; j < 6; j++)
for (k = 0; k< 100; k++);
} 
#pragma acc loop tile(2, 2)
for (i = 1; i < 5; i+=2)
{
for (j = i + 1; j < 7; j+=i) 
{ }
}
#pragma acc loop vector tile(*) 
for (i = 0; i < 10; i++)
{ }
#pragma acc loop worker tile(*) 
for (i = 0; i < 10; i++)
{ }
#pragma acc loop gang tile(*) 
for (i = 0; i < 10; i++)
{ }
#pragma acc loop vector gang tile(*) 
for (i = 0; i < 10; i++)
{ }
#pragma acc loop vector worker tile(*) 
for (i = 0; i < 10; i++)
{ }
#pragma acc loop gang worker tile(*) 
for (i = 0; i < 10; i++)
{ }
}
}
void p3 (void)
{
int i, j;
#pragma acc parallel loop tile 
for (i = 0; i < 10; i++)
{ }
#pragma acc parallel loop tile() 
for (i = 0; i < 10; i++)
{ }
#pragma acc parallel loop tile(1) 
for (i = 0; i < 10; i++)
{ }
#pragma acc parallel loop tile(*) 
for (i = 0; i < 10; i++)
{ }
#pragma acc parallel loop tile(*, 1) 
for (i = 0; i < 10; i++)
{
for (j = 1; j < 10; j++)
{ }
}
#pragma acc parallel loop tile(-2)   
for (i = 1; i < 10; i++)
{ }
#pragma acc parallel loop tile(i)   
for (i = 1; i < 10; i++)
{ }
#pragma acc parallel loop tile(2, 2, 1)
for (i = 1; i < 3; i++)
for (j = 4; j < 6; j++)
for (int k = 1 ; k < 2; k++)
;
#pragma acc parallel loop tile(2, 2)
for (i = 1; i < 5; i+=2)
for (j = i + 1; j < 7; j++) 
{ }
#pragma acc parallel loop vector tile(*) 
for (i = 0; i < 10; i++)
{ }
#pragma acc parallel loop worker tile(*) 
for (i = 0; i < 10; i++)
{ }
#pragma acc parallel loop gang tile(*) 
for (i = 0; i < 10; i++)
{ }
#pragma acc parallel loop vector gang tile(*) 
for (i = 0; i < 10; i++)
{ }
#pragma acc parallel loop vector worker tile(*) 
for (i = 0; i < 10; i++)
{ }
#pragma acc parallel loop gang worker tile(*) 
for (i = 0; i < 10; i++)
{ }
}
void
kern (void)
{
int i, j;
#pragma acc kernels
{
#pragma acc loop tile 
for (i = 0; i < 10; i++)
{ }
#pragma acc loop tile() 
for (i = 0; i < 10; i++)
{ }
#pragma acc loop tile(1)
for (i = 0; i < 10; i++)
{ }
#pragma acc loop tile(2)
for (i = 0; i < 10; i++)
{ }
#pragma acc loop tile(6-2) 
for (i = 0; i < 10; i++)
{ }
#pragma acc loop tile(6+2) 
for (i = 0; i < 10; i++)
{ }
#pragma acc loop tile(*) 
for (i = 0; i < 10; i++)
{ }
#pragma acc loop tile(*, 1) 
for (i = 0; i < 10; i++)
{
for (j = 0; j < 10; i++) 
{ }
}
#pragma acc loop tile(-2) 
for (i = 0; i < 10; i++)
{ }
#pragma acc loop tile(i) 
for (i = 0; i < 10; i++)
{ }
#pragma acc loop tile(2, 2, 1)
for (i = 2; i < 4; i++)
for (j = 4; j < 6; j++)
for (int k = 4; k < 6; k++)
{ }
#pragma acc loop tile(2, 2)
for (i = 1; i < 5; i+=2)
for (j = i+1; j < 7; j++) 
{ }
#pragma acc loop vector tile(*) 
for (i = 0; i < 10; i++)
{ }
#pragma acc loop worker tile(*) 
for (i = 0; i < 10; i++)
{ }
#pragma acc loop gang tile(*) 
for (i = 0; i < 10; i++)
{ }
#pragma acc loop vector gang tile(*) 
for (i = 0; i < 10; i++)
{ }
#pragma acc loop vector worker tile(*) 
for (i = 0; i < 10; i++)
{ }
#pragma acc loop gang worker tile(*) 
for (i = 0; i < 10; i++)
{ }
}
}
void k3 (void)
{
int i, j;
#pragma acc kernels loop tile 
for (i = 0; i < 10; i++)
{ }
#pragma acc kernels loop tile() 
for (i = 0; i < 10; i++)
{ }
#pragma acc kernels loop tile(1) 
for (i = 0; i < 10; i++)
{ }
#pragma acc kernels loop tile(*) 
for (i = 0; i < 10; i++)
{ }
#pragma acc kernels loop tile(*, 1) 
for (i = 0; i < 10; i++)
{
for (j = 1; j < 10; j++)
{ }
}
#pragma acc kernels loop tile(-2) 
for (i = 1; i < 10; i++)
{ }
#pragma acc kernels loop tile(i) 
for (i = 1; i < 10; i++)
{ }
#pragma acc kernels loop tile(2, 2, 1)
for (i = 1; i < 3; i++)
for (j = 4; j < 6; j++)
for (int k = 1; k < 7; k++)
;
#pragma acc kernels loop tile(2, 2)
for (i = 1; i < 5; i++)
{
for (j = i + 1; j < 7; j += i) 
{ }
}
#pragma acc kernels loop vector tile(*) 
for (i = 0; i < 10; i++)
{ }
#pragma acc kernels loop worker tile(*) 
for (i = 0; i < 10; i++)
{ }
#pragma acc kernels loop gang tile(*) 
for (i = 0; i < 10; i++)
{ }
#pragma acc kernels loop vector gang tile(*) 
for (i = 0; i < 10; i++)
{ }
#pragma acc kernels loop vector worker tile(*) 
for (i = 0; i < 10; i++)
{ }
#pragma acc kernels loop gang worker tile(*) 
for (i = 0; i < 10; i++)
{ }
}
