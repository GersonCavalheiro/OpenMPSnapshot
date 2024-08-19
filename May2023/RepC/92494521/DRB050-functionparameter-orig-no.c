void foo1(double o1[], double c[], int len)
{ 
int i ;
#pragma omp parallel for
for (i = 0; i < len; ++i) {
double volnew_o8 = 0.5 * c[i];
o1[i] = volnew_o8;
} 
}
double o1[100];
double c[100];
int main()
{
foo1 (o1, c, 100);
return 0;
}
