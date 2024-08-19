int main ()
{
int dummy[10];
#pragma acc parallel num_workers (2<<20) 
{
#pragma acc loop worker
for (int  i = 0; i < 10; i++)
dummy[i] = i;
}
#pragma acc parallel vector_length (2<<20) 
{
#pragma acc loop vector
for (int  i = 0; i < 10; i++)
dummy[i] = i;
}
return 0;
}
