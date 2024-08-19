int main(int argc, char** argv)
{
volatile unsigned char gate = 0;
#pragma omp parallel shared(gate)
{
#pragma omp master
{
int i, j;
for (i = 0; i < 100; i++)
{
for (j = 0; j < 100; j++)
{
}
}
gate = 1;
#pragma omp flush
}
while (gate == 0);
}
return 0;
}
