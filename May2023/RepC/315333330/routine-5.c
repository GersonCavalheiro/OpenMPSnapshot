struct PC
{
#pragma acc routine 
};
void PC1( 
#pragma acc routine
) 
{
}
void PC2()
{
if (0)
#pragma acc routine 
;
}
void PC3()
{
#pragma acc routine 
}
#pragma acc routine ( 
#pragma acc routine () 
#pragma acc routine (+) 
#pragma acc routine (?) 
#pragma acc routine (:) 
#pragma acc routine (4) 
#pragma acc routine ('4') 
#pragma acc routine ("4") 
extern void R1(void);
extern void R2(void);
#pragma acc routine (R1, R2, R3) worker 
#pragma acc routine (R1 R2 R3) worker 
#pragma acc routine (R1) worker
#pragma acc routine (R2) worker
#pragma acc routine 
int a;
#pragma acc routine 
void fn1 (void), fn1b (void);
#pragma acc routine 
int b, fn2 (void);
#pragma acc routine 
int b_, fn2_ (void), B_;
#pragma acc routine 
int fn3 (void), b2;
#pragma acc routine 
typedef struct c c;
#pragma acc routine 
struct d {} d;
#pragma acc routine 
#pragma acc routine 
void fn1_2 (void), fn1b_2 (void);
#pragma acc routine 
#pragma acc routine 
int b_2, fn2_2 (void);
#pragma acc routine 
#pragma acc routine 
int b_2_, fn2_2_ (void), B_2_;
#pragma acc routine 
#pragma acc routine 
int fn3_2 (void), b2_2;
#pragma acc routine 
#pragma acc routine 
typedef struct c_2 c_2;
#pragma acc routine 
#pragma acc routine 
struct d_2 {} d_2;
#pragma acc routine 
#pragma acc routine
int fn4 (void);
int fn5a (void);
int fn5b (void);
#pragma acc routine 
#pragma acc routine (fn5a)
#pragma acc routine (fn5b)
int fn5 (void);
#pragma acc routine 
#pragma acc routine (fn6a) 
#pragma acc routine (fn6b) 
int fn6 (void);
#ifdef __cplusplus
#pragma acc routine 
namespace f {}
namespace g {}
#pragma acc routine 
using namespace g;
#pragma acc routine (g) 
#endif 
#pragma acc routine (a) 
#pragma acc routine (c) 
#pragma acc routine 
#ifndef __cplusplus 
_Static_assert(0, ""); 
#elif __cplusplus < 201103L 
int dummy_instead_of_static_assert;
#else 
static_assert(0, ""); 
#endif
void f_static_assert();
#pragma acc routine (f_static_assert) 
#pragma acc routine
__extension__ extern void ex1();
#pragma acc routine (ex1) 
#pragma acc routine
__extension__ __extension__ __extension__ __extension__ __extension__ void ex2()
{
}
#pragma acc routine (ex2) 
#pragma acc routine 
__extension__ int ex3;
#pragma acc routine (ex3) 
extern void fungsi_1();
#pragma acc routine(fungsi_1) gang
#pragma acc routine(fungsi_1) gang 
#pragma acc routine(fungsi_1) worker 
#pragma acc routine(fungsi_1) vector 
#pragma acc routine seq
extern void fungsi_2();
#pragma acc routine(fungsi_2) seq 
#pragma acc routine(fungsi_2) worker 
#pragma acc routine(fungsi_2) 
#pragma acc routine vector
extern void fungsi_3();
#pragma acc routine vector 
void fungsi_3()
{
}
extern void fungsi_4();
#pragma acc routine (fungsi_4) worker
#pragma acc routine gang 
void fungsi_4()
{
}
#pragma acc routine gang
void fungsi_5()
{
}
#pragma acc routine (fungsi_5) worker 
#pragma acc routine seq
void fungsi_6()
{
}
#pragma acc routine seq 
extern void fungsi_6();
void Bar ();
void Foo ()
{
Bar ();
}
#pragma acc routine (Bar) 
#pragma acc routine (Foo) gang 
#pragma acc routine (Baz) 
int vb1;		
extern int vb2;		
static int vb3;		
#pragma acc routine
int
func1 (int a)
{
vb1 = a + 1;
vb2 = vb1 + 1;
vb3 = vb2 + 1;
return vb3;
}
#pragma acc routine
int
func2 (int a)
{
extern int vb4;	
static int vb5;	
vb4 = a + 1;
vb5 = vb4 + 1;
return vb5;
}
extern int vb6;			
#pragma acc declare link (vb6)
static int vb7;			
#pragma acc declare link (vb7)
#pragma acc routine
int
func3 (int a)
{
vb6 = a + 1;
vb7 = vb6 + 1;
return vb7;
}
int vb8;
#pragma acc declare create (vb8)
extern int vb9;
#pragma acc declare create (vb9)
static int vb10;
#pragma acc declare create (vb10)
#pragma acc routine
int
func4 (int a)
{
vb8 = a + 1;
vb9 = vb8 + 1;
vb10 = vb9 + 1;
return vb10;
}
int vb11;
#pragma acc declare device_resident (vb11)
extern int vb12;
#pragma acc declare device_resident (vb12)
extern int vb13;
#pragma acc declare device_resident (vb13)
#pragma acc routine
int
func5 (int a)
{
vb11 = a + 1;
vb12 = vb11 + 1;
vb13 = vb12 + 1;
return vb13;
}
#pragma acc routine
int
func6 (int a)
{
extern int vb14;
#pragma acc declare create (vb14)
static int vb15;
#pragma acc declare create (vb15)
vb14 = a + 1;
vb15 = vb14 + 1;
return vb15;
}
