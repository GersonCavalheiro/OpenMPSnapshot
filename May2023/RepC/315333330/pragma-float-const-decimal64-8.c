#pragma STDC FLOAT_CONST_DECIMAL64 OFF
#define MAX 200
#pragma STDC FLOAT_CONST_DECIMAL64 ON
double a;
#pragma STDC FLOAT_CONST_DECIMAL64 OFF
struct S1 {
#pragma STDC FLOAT_CONST_DECIMAL64 ON	
int i;
int j;
};
struct S2 {
int i;
#pragma STDC FLOAT_CONST_DECIMAL64 ON	
int j;
};
struct S3 {
int i;
int j;
#pragma STDC FLOAT_CONST_DECIMAL64 ON	
};
enum E1 {
#pragma STDC FLOAT_CONST_DECIMAL64 ON	
one,
two
};
enum E2 {
red,
#pragma STDC FLOAT_CONST_DECIMAL64 ON	
blue
};
enum E3 {
cat,
dog
#pragma STDC FLOAT_CONST_DECIMAL64 ON	
};
double
#pragma STDC FLOAT_CONST_DECIMAL64 OFF	
b;
double
f1 (void)
{
#pragma STDC FLOAT_CONST_DECIMAL64 ON
return a;
}
double
f2 (void)
{
double b;
#pragma STDC FLOAT_CONST_DECIMAL64 ON	
b = 0.5;
return a + b;
}
#pragma STDC FLOAT_CONST_DECIMAL64 OFF
double
f3 (void)
{
typedef double b32;
#pragma STDC FLOAT_CONST_DECIMAL64 ON	
b32 b = 0.5;
return b;
}
double
f4 (int i)
{
top:
#pragma STDC FLOAT_CONST_DECIMAL64 OFF	
if (i == 0)
return a;  
a *= 2.;
i = 0;
goto top;
}
double
f5 (int i)
{
a = a * i;
#pragma STDC FLOAT_CONST_DECIMAL64 OFF	
return a * 2.;
}
double
#pragma STDC FLOAT_CONST_DECIMAL64 ON	
f6 (void)
{
return a;
}
double
f7
#pragma STDC FLOAT_CONST_DECIMAL64 ON	
(void)					
{
return a;
}
double
f8 (void)
{
{
#pragma STDC FLOAT_CONST_DECIMAL64 OFF
}
#pragma STDC FLOAT_CONST_DECIMAL64 ON   
return a;
}
extern void foo9 (void *);
double
f9 (void)
{
__label__ here;
#pragma STDC FLOAT_CONST_DECIMAL64 ON	
foo9 (&&here);
here:
return a;
}
double
f10 (void)
{
void foo10 (void) 
{
a = 1.0;
}
#pragma STDC FLOAT_CONST_DECIMAL64 ON	
return a;
}
double
f11 (void)
{
__extension__
struct A {
struct { char a; };
char b;
};
#pragma STDC FLOAT_CONST_DECIMAL64 ON	
return a;
}
double
f12 (void)
{
__extension__ ({ a = 0.5; });
#pragma STDC FLOAT_CONST_DECIMAL64 ON	
return a;
}
