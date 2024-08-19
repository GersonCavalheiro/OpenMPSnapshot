namespace A {
extern int i;
constexpr int *p0 = &i;
constexpr bool b0  = p0;        
constexpr bool b1  = p0 == 0;   
constexpr bool b2  = p0 != 0;   
constexpr bool b3  = p0 <  0;   
constexpr bool b4  = p0 <= 0;   
constexpr bool b5  = p0 >  0;   
constexpr bool b6  = p0 >= 0;   
constexpr bool b7  = !p0;       
constexpr bool b8  = 0 == p0;   
constexpr bool b9  = 0 != p0;   
constexpr bool b10 = 0 <  p0;   
constexpr bool b11 = 0 <= p0;   
constexpr bool b12 = 0 >  p0;   
constexpr bool b13 = 0 >= p0;   
}
namespace B {
struct S { int a, b[1]; } s;
constexpr S *p0 = &s;
constexpr int *q0 = p0->b;      
}
namespace WeakRefTest1 {
extern __attribute__ ((weak)) int i;
constexpr int *p0 = &i;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wextra"
constexpr bool b0  = p0;        
constexpr bool b1  = p0 == 0;   
constexpr bool b2  = p0 != 0;   
constexpr bool b4  = p0 <= 0;   
constexpr bool b5  = p0 >  0;   
constexpr bool b7  = !p0;       
constexpr bool b8  = 0 == p0;   
constexpr bool b9  = 0 != p0;   
constexpr bool b10 = 0 <  p0;   
constexpr bool b13 = 0 >= p0;   
constexpr bool b3  = p0 <  0;
constexpr bool b6  = p0 >= 0;
constexpr bool b11 = 0 <= p0;
constexpr bool b12 = 0 >  p0;
#pragma GCC diagnostic pop
}
namespace WeakRefTest2 {
extern __attribute__ ((weak)) int i;
constexpr int *p1 = &i + 1;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wextra"
constexpr bool b0  = p1;        
constexpr bool b1  = p1 == 0;   
constexpr bool b2  = p1 != 0;   
constexpr bool b4  = p1 <= 0;   
constexpr bool b5  = p1 >  0;   
constexpr bool b7  = !p1;       
constexpr bool b8  = 0 == p1;   
constexpr bool b9  = 0 != p1;   
constexpr bool b10 = 0 <  p1;   
constexpr bool b13 = 0 >= p1;   
#pragma GCC diagnostic pop
}
