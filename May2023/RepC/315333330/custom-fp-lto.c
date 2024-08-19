#include <stdio.h> 
#include <math.h>
#pragma GCC target ("custom-fabss=224")
float
custom_fp (float operand_a)
{
return fabsf (operand_a);
}
int
main (int argc, char *argv[])
{
return custom_fp ((float)argc) > 1.0;
}
