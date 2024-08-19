double a;
void f1 (void)
{
#pragma STDC FLOAT_CONST_DECIMAL64 ON		
a = 1.0;
}
void f2 (void)
{
#pragma STDC FLOAT_CONST_DECIMAL64 OFF		
a = 2.0;
}
void f3 (void)
{
#pragma STDC FLOAT_CONST_DECIMAL64 DEFAULT	
a = 3.0;
}
void f4 (void)
{
_Pragma ("STDC FLOAT_CONST_DECIMAL64 ON")	
a = 1.0;
}
void f5 (void)
{
_Pragma ("STDC FLOAT_CONST_DECIMAL64 OFF")	
a = 2.0;
}
void f6 (void)
{
_Pragma ("STDC FLOAT_CONST_DECIMAL64 DEFAULT") 
a = 3.0;
}
