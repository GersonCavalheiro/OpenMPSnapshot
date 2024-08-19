#include <stdio.h> 
#include <math.h>
#pragma GCC target ("custom-fmaxs=246")
extern void
custom_fp (float operand_a, float operand_b, float *result)
__attribute__ ((target ("custom-fmins=247")));
void
custom_fp (float operand_a, float operand_b, float *result)
{   
result[0] = fmaxf (operand_a, operand_b);
result[1] = fminf (operand_a, operand_b);
}
