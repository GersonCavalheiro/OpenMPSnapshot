typedef void (F)(void);
typedef __UINTPTR_TYPE__ uintptr_t;
static void func_arg (F *p, ...) { (void)p; }
static F* test_taking_address_of_gcc_builtin (void)
{
F *p;
void *q;
uintptr_t a;
__builtin_trap ();
(void)__builtin_trap;
__builtin_trap;
{
typedef __typeof__ (__builtin_trap) F;     
}
p = &__builtin_trap;               
p = *__builtin_trap;               
a = !__builtin_trap;               
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpointer-arith"
a = sizeof __builtin_trap;
#pragma GCC diagnostic pop
#ifndef __STDC_VERSION__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
a = _Alignof __builtin_trap;
#ifndef __STDC_VERSION__
#pragma GCC diagnostic pop
#endif
p = (F*)__builtin_trap;            
a = (uintptr_t)__builtin_trap;     
p = __builtin_trap + 0;            
p = __builtin_trap - 0;            
a = __builtin_trap - p;            
a = p - __builtin_trap;            
a = __builtin_trap < p;            
a = p < __builtin_trap;            
a = __builtin_trap <= p;           
a = p <= __builtin_trap;           
a = __builtin_trap > p;            
a = p > __builtin_trap;            
a = __builtin_trap > p;            
a = p > __builtin_trap;            
a = __builtin_trap <= p;           
a = p <= __builtin_trap;           
a = __builtin_trap <= p;           
a = p <= __builtin_trap;           
a = __builtin_trap == p;           
a = p == __builtin_trap;           
a = __builtin_trap != p;           
a = p != __builtin_trap;           
a = __builtin_trap && p;           
a = p && __builtin_trap;           
a = __builtin_trap || p;           
a = p || __builtin_trap;           
a = __builtin_trap ? 1 : 0;        
p = a ? __builtin_trap : 0;        
p = a ? 0 : __builtin_trap;        
p = __builtin_trap;                
q = __builtin_trap;                
a = __builtin_trap;                
func_arg (__builtin_trap);         
func_arg (0, __builtin_trap);      
return __builtin_trap;             
(void)a;
(void)p;
(void)q;
}
extern int abs (int);
extern __SIZE_TYPE__ strlen (const char*);
void test_taking_address_of_library_builtin (int i)
{
{
typedef int F (int);
F *p = __builtin_abs;
p = &__builtin_abs;
p = *__builtin_abs;
p = abs;
p = &abs;
p = *abs;
(void)p;
}
{
typedef __SIZE_TYPE__ size_t;
typedef size_t F (const char*);
F *p = __builtin_strlen;
p = &__builtin_strlen;
p = *__builtin_strlen;
p = strlen;
p = &strlen;
p = *strlen;
(void)p;
}
{
typedef int F (int);
F *p = __builtin_isalnum;
p = &__builtin_isalpha;
p = *__builtin_iscntrl;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wimplicit-function-declaration"
i = isalnum (i) || isalpha (i) || iscntrl (i);
#pragma GCC diagnostic pop
p = isalnum;
p = &isalpha;
p = *iscntrl;
(void)p;
}
}
