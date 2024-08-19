#include <stdbool.h>
int
main (void)
{
int l, l2, l3, l4;
bool b, b2, b3, b4;
int i, i2;
#pragma acc parallel if(l) 
;
#pragma acc parallel if(b) 
;
#pragma acc kernels if(l2) 
;
#pragma acc kernels if(b2) 
;
#pragma acc data if(l3) 
;
#pragma acc data if(b3) 
;
#pragma acc update if(l4) self(i) 
;
#pragma acc update if(b4) self(i2) 
;
}
