#pragma GCC diagnostic error "-Wstrict-overflow"
int
foo (int i)
{
return __builtin_abs (i) >= 0; 
}
