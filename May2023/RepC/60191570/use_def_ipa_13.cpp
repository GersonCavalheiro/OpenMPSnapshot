#include <cstdlib>
struct R {
int b[2];
R() {}
};
struct S {
int a;
struct R r;
S() {}
};
struct Q {
int* a;
int b[2];
Q() {}
};
struct T {
int* a;
Q* s;
int* b;
int* c;
T() {}
};
void foo1(int a, int &b, ...);
void foo2(int* a, int *&b, int **d, ...);
void foo3(int a, int &b, int &c);
void foo4(int a[], int (*b)[2], int* c, ...);
const int N = 4;
int main(int argc, char** argv)
{
struct S s[10];
s[5].a;
s[2].r.b[1];
#pragma analysis_check assert upper_exposed(s[2].r.b[1]) undefined(s[5].a)
foo1(s[2].r.b[1], s[5].a);
T t1;
t1.a = (int*) malloc(sizeof(int)*2);
t1.s = (Q*) malloc(sizeof(Q)*2);
#pragma analysis_check assert undefined(t1.a[0:1], (*t1.s).a, *(*t1.s).a, *t1.b, *t1.c)
foo2(t1.a, t1.s->a, &t1.b, t1.c);
#pragma analysis_check assert undefined((*t1.s).b[0:1], *(*t1.s).a) upper_exposed(*t1.a) 
foo3(*t1.a, *t1.s->a, *t1.s->b);
int c1[N];
int c2[2];
int c3[2];
int c4[2];
#pragma analysis_check assert undefined(c1[0:3], c2[0:1], c3[0:1], c4[0:1])
foo4(c1, &c2, c3, &c4[0]);
return 0;
}
