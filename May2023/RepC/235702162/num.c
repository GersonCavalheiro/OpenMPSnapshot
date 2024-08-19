int isInteger(char * str)
{
int _ret_val_0;
if (( * str)=='\0')
{
_ret_val_0=0;
return _ret_val_0;
}
#pragma loop name isInteger#0 
for (; ( * str)!='\0'; str ++ )
{
if ((( * str)<48)||(( * str)>57))
{
_ret_val_0=0;
return _ret_val_0;
}
}
_ret_val_0=1;
return _ret_val_0;
}
