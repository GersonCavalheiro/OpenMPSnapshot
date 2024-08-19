int main(int argc, char* argv[])
{
unsigned char flag = 0;
#pragma omp parallel shared(flag)
{
int i;
for (i = 0; i < 100; i++)
{
int j;
for (j = 0; j < 100; j++)
{
#pragma omp critical
{
unsigned char val_of_flag = j & 0x1;
if (flag != val_of_flag)
__builtin_abort();
flag = (~val_of_flag) & 0x1;
}
}
}
}
return 0;
}
