#pragma GCC target ("arch=armv8.2-a+nofp16")
_Float16
foo_v8 (_Float16 x, _Float16 y, unsigned int *eval)
{
*eval = __FLT_EVAL_METHOD__;
return x * x + y;
}
__fp16
bar_v8 (__fp16 x, __fp16 y, unsigned int *eval)
{
*eval = __FLT_EVAL_METHOD__;
return x * x + y;
}
#pragma GCC target ("arch=armv8.2-a+fp16")
_Float16
foo_v82 (_Float16 x, _Float16 y, unsigned int *eval)
{
*eval = __FLT_EVAL_METHOD__;
return x * x + y;
}
__fp16
bar_v82 (__fp16 x, __fp16 y, unsigned int *eval)
{
*eval = __FLT_EVAL_METHOD__;
return x * x + y;
}
