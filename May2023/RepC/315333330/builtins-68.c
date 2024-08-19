#define CHAR_BIT  __CHAR_BIT__
#define SIZE_MAX  __SIZE_MAX__
#define UINT_MAX  (__INT_MAX__ + 1U)
#define ALIGN_MAX ((UINT_MAX >> 1) + 1)
#if UINT_MAX < SIZE_MAX
#  define MAX_X_2   (ALIGN_MAX << 1)
#else
#  define MAX_X_2   (ALIGN_MAX + 1)
#endif
static void* p;
void test_valid (int n)
{
enum {
A1   = CHAR_BIT *   1,
A2   = CHAR_BIT *   2,
A4   = CHAR_BIT *   4,
A8   = CHAR_BIT *   8,
A16  = CHAR_BIT *  16,
A32  = CHAR_BIT *  32
};
p =  __builtin_alloca_with_align (n, CHAR_BIT *  1);
p =  __builtin_alloca_with_align (n, CHAR_BIT *  2);
p =  __builtin_alloca_with_align (n, CHAR_BIT *  4);
p =  __builtin_alloca_with_align (n, CHAR_BIT *  8);
p =  __builtin_alloca_with_align (n, CHAR_BIT * 16);
p =  __builtin_alloca_with_align (n, CHAR_BIT * 32);
p =  __builtin_alloca_with_align (n, A1);
p =  __builtin_alloca_with_align (n, A2);
p =  __builtin_alloca_with_align (n, A4);
p =  __builtin_alloca_with_align (n, A8);
p =  __builtin_alloca_with_align (n, A16);
p =  __builtin_alloca_with_align (n, A32);
}
void test_arg2_non_int (int n)
{
p =  __builtin_alloca_with_align (n, 0.0);         
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-conversion"
p =  __builtin_alloca_with_align (n, (void*)0);    
p =  __builtin_alloca_with_align (n, "");          
p =  __builtin_alloca_with_align (n, L"");         
#pragma GCC diagnostic pop
}
void test_arg2_non_const (int n, int a1)
{
extern const int a2;
static const int a3 = CHAR_BIT;
static volatile const int a4 = CHAR_BIT;
p =  __builtin_alloca_with_align (n, a1);       
p =  __builtin_alloca_with_align (n, a2);       
p =  __builtin_alloca_with_align (n, a3);       
p =  __builtin_alloca_with_align (n, a4);       
}
void test_arg2_non_pow2 (int n)
{
p =  __builtin_alloca_with_align (n,  0);          
p =  __builtin_alloca_with_align (n,  1);          
p =  __builtin_alloca_with_align (n,  2);          
p =  __builtin_alloca_with_align (n,  3);          
p =  __builtin_alloca_with_align (n,  4);          
p =  __builtin_alloca_with_align (n,  5);          
p =  __builtin_alloca_with_align (n,  6);          
p =  __builtin_alloca_with_align (n,  7);          
p =  __builtin_alloca_with_align (n,  9);          
p =  __builtin_alloca_with_align (n, 10);          
p =  __builtin_alloca_with_align (n, 11);          
p =  __builtin_alloca_with_align (n, 12);          
p =  __builtin_alloca_with_align (n, 13);          
p =  __builtin_alloca_with_align (n, 14);          
p =  __builtin_alloca_with_align (n, 15);          
p =  __builtin_alloca_with_align (n, 17);          
p =  __builtin_alloca_with_align (n, 31);          
p =  __builtin_alloca_with_align (n, 33);          
p =  __builtin_alloca_with_align (n, 63);          
p =  __builtin_alloca_with_align (n, 65);          
p =  __builtin_alloca_with_align (n, SIZE_MAX);    
p =  __builtin_alloca_with_align (n, MAX_X_2);     
}
