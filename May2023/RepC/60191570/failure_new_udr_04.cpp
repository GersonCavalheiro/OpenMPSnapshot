class A {
private:
int x;
};
#pragma omp declare reduction ( + : A : omp_out.x += omp_in.x )
int main (int argc, char* argv[])
{
A a;
#pragma omp parallel reduction( + : a )
{
a;
}
return 0;
}
