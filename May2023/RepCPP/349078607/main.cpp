#include <iostream>
#include <omp.h>

using namespace std;
int main(){
int i;
#pragma omp parallel shared(i)
#pragma omp for
for(i = 0; i < 10; i++){
#pragma omp critical
{
cout << i << endl;
}
}
return 0;
}