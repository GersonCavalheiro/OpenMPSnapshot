#include <stdlib.h>  
int main(int argc, char* argv[])  
{
int i;
int len=100;
if (argc>1)
len = atoi(argv[1]);
int numNodes=len, numNodes2=0; 
int x[len]; 
for (i=0; i< len; i++)
{
if (i%2==0)
x[i]=5;
else
x[i]= -5;
}
#pragma omp parallel for
for (i=numNodes-1 ; i>-1 ; --i) {
if (x[i]<=0) {
numNodes2-- ;
}
}         
return 0;
} 
