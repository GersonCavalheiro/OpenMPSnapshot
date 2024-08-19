#include <stdio.h> 
#include <math.h>
#pragma GCC target ("custom-fmaxs=246,custom-fmins=247")
extern void
custom_fp (float operand_a, float operand_b, float *result);
void
custom_fp (float operand_a, float operand_b, float *result)
{
result[0] = fmaxf (operand_a, operand_b);
result[1] = fminf (operand_a, operand_b);
}
