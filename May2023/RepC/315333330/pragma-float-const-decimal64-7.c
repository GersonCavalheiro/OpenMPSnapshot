double
f1 (void)
{
#pragma STDC FLOAT_CONST_DECIMAL64 OFF
double a = 0x1.0p1;
double b = 1.0i;
return a + b;
}
double
f2 (void)
{
#pragma STDC FLOAT_CONST_DECIMAL64 OFF
double a = 0x1.0p1dd;		
double b = 1.0idd;		
return a + b;
}
double
f3 (void)
{
#pragma STDC FLOAT_CONST_DECIMAL64 ON
double a = 0x1.0p1;	
double b = 1.0i;	
return a + b;
}
