int some_func (int c);
#pragma GCCPLUGIN sayhello "here" 
int some_func (const char* s)
{
#pragma GCCPLUGIN sayhello "at start" 
#define DO_PRAGMA(x) _Pragma(#x)
if (!s)
{
DO_PRAGMA(GCCPLUGIN sayhello "in block"); 
return 0;
}
return 1;
}
