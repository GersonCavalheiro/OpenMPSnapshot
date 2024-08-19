#pragma omp task final(x1 > 4)
void f(int x1);
void f(int x)
{
if (x > 1)
f(x-1);
}
void g(int y)
{
f(y + 2);
}
int main(int argc, char *argv[])
{
g(10);
return 0;
}
