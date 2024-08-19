#include "classify.h"
#include <omp.h>

Data classify(Data &D, const Ranges &R, unsigned int numt)
{ 
assert(numt < MAXTHREADS);
Counter counts[(R.num())]; 
#pragma omp parallel num_threads(numt)
{
int tid = omp_get_thread_num(); 

for(int i=(D.ndata/numt)*tid; i<std::min(D.ndata,(tid+1)*(D.ndata)/numt); i+=1) { 
int v = D.data[i].value = R.range(D.data[i].key);
counts[v].increase(tid); 
}
}

unsigned int *rangecount = new unsigned int[R.num()];
for(int r=0; r<R.num(); r++) { 
rangecount[r] = 0;
for(int t=0; t<numt; t++) 
rangecount[r] += counts[r].get(t);
}

for(int i=1; i<R.num(); i++) {
rangecount[i] += rangecount[i-1];
}


Data D2 = Data(D.ndata); 

#pragma omp parallel num_threads(numt)
{
int tid = omp_get_thread_num();
for(int r=(D.ndata/numt)*tid; r<R.num(); r+=1) { 
int rcount = 0;
int l =0;
int h = D.ndata-1;
while(l<=h){
int mid1 = l + (h-l)/3;
int mid2 = h - (h-l)/3;
if(D.data[mid1].value == r) {
D2.data[rangecount[r-1]+rcount++] = D.data[mid1]; 
break;}
else if(D.data[mid2].value == r) {
D2.data[rangecount[r-1]+rcount++] = D.data[mid2]; 
break;}
else if (D.data[mid1].value <r && D.data[mid2].value>r){
l = mid1+1;
h = mid2-1;
}
else if (D.data[mid1].value >r){
h = mid1-1;
}

else{
l= mid2+1;
}
}
}
}
return D2;
}


