const int a = 10;
int b;
const int e[2] = {0, 1};
int f[2];
const int* g = &a;          
int* const h = &b;          
const int* const i = &b;    
int *j;
void rec(int p1, int &p2, int *p3, int *&p4, int *p5)
{
p1 = b + *p4;
p2 = a + *p5;
p3++;
p4 = j;
int x;
j = &x;
const int *v = g;
int* const w = h;
int *z;
#pragma analysis_check assert upper_exposed(h, *h, j, *j, p3, z, *z, g, a, b, x) defined(*j, j, z)
rec(*h, *j, p3+1, z, &x);
}