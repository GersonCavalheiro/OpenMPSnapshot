#pragma GCC target ("arch=armv8.2-a+nofp16")
__fp16
foo (int x)
{
return x;
}
__fp16
bar (unsigned int x)
{
return x;
}
__fp16
fool (long long x)
{
return x;
}
__fp16
barl (unsigned long long x)
{
return x;
}
