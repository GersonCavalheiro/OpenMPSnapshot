#pragma weak a
int a;
int b;
#pragma weak b
#pragma weak c
extern int c;
int c;
extern int d;
#pragma weak d
int d;
#pragma weak e
void e(void) { }
#if 0
void f(void) { }
#pragma weak f
#endif
#pragma weak g
int g = 1;
#if 0
int h = 1;
#pragma weak h
#endif
#pragma weak i
extern int i;
#pragma weak j
extern int j;
int use_j() { return j; }
