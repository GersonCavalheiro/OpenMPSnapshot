void
foo (int x)
{
bad1:				
#pragma omp target teams
goto bad1;			
goto bad2;			
#pragma omp target teams
{
bad2: ;			
}
#pragma omp target teams
{
int i;
goto ok1;
for (i = 0; i < 10; ++i)
{ ok1: break; }
}
switch (x)
{
#pragma omp target teams
{ case 0:; }		
}
}
void
bar (int x)
{
bad1:				
#pragma omp target
#pragma omp teams
goto bad1;			
goto bad2;			
#pragma omp target
#pragma omp teams
{
bad2: ;			
}
#pragma omp target
#pragma omp teams
{
int i;
goto ok1;
for (i = 0; i < 10; ++i)
{ ok1: break; }
}
switch (x)
{
#pragma omp target		
#pragma omp teams
{ case 0:; }		
}
}
