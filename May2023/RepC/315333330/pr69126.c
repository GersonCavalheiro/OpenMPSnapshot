#pragma GCC diagnostic push
#define MACRO_1 \
_Pragma("GCC diagnostic ignored \"-Wunused-variable\"")
int test_1()
{
_Pragma("GCC diagnostic ignored \"-Wunused-variable\"")
int x;
return 0;
}
#pragma GCC diagnostic pop
#pragma GCC diagnostic push
int test_2()
{
_Pragma("GCC diagnostic ignored \"-Wunused-variable\"")
int x;
return 0;
}
#pragma GCC diagnostic pop
#define MACRO_3 \
int x;
#pragma GCC diagnostic push
int test_3()
{
_Pragma("GCC diagnostic ignored \"-Wunused-variable\"")
MACRO_3
return 0;
}
#pragma GCC diagnostic pop
#pragma GCC diagnostic push
#define MACRO_4A \
_Pragma("GCC diagnostic ignored \"-Wunused-variable\"")
#define MACRO_4B \
int x;
int test_4()
{
MACRO_4A;
MACRO_4B
return 0;
}
#pragma GCC diagnostic pop
#pragma GCC diagnostic push
#define MACRO_5 \
_Pragma("GCC diagnostic ignored \"-Wunused-variable\"") \
int x;
int test_5()
{
MACRO_5;
return 0;
}
#pragma GCC diagnostic pop
