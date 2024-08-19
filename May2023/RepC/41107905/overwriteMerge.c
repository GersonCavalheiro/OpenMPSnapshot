#include<stdint.h>
#include<ndlib.h>
void overwriteMerge ( uint32_t * data1, uint32_t * data2, int dim )
{
int i;
for ( i=0; i<dim; i++ )
{
if ( data2[i] != 0 )
data1[i] = data2[i];
}
}
