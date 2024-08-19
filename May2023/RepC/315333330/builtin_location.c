#if __cplusplus >= 201103L
#  define Assert(expr) static_assert ((expr), #expr)
#elif __STDC_VERSION__ >= 201112L
#  define Assert(expr) _Static_assert ((expr), #expr)
#else
#  define CONCAT(a, b)  a ## b
#  define CAT(a, b)     CONCAT (a, b)
#  define Assert(expr)  typedef int CAT (Assert_, __LINE__) [1 - 2 * !(expr)]
#endif
const char* const file = __builtin_FILE ();
const char* const function = __builtin_FUNCTION ();
Assert (__builtin_constant_p (__builtin_FILE ()));
Assert (__builtin_constant_p (__builtin_FUNCTION ()));
#pragma GCC diagnostic push  
#pragma GCC diagnostic ignored "-Waddress"
enum E0 {
e0 = __FILE__ - __FILE__,
e1 = __builtin_FILE () - __builtin_FILE (),
#if !__cplusplus || __cplusplus >= 201103L
e2 = __FUNCTION__ - __FUNCTION__,
e3 = __builtin_FUNCTION () - __builtin_FUNCTION ()
#endif
};
#pragma GCC diagnostic pop
#line 13
int a [__builtin_LINE ()][__builtin_LINE ()];
enum F { f0 = __builtin_LINE () };
struct S { unsigned bitfield: __builtin_LINE (); } s;
Assert (__builtin_constant_p (__builtin_LINE ()));
