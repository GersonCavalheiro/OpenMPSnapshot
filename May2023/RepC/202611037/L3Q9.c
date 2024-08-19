#include<stdio.h>
#include<omp.h>
int main(int argc, char *argv[]){
int myid, i, j, a[10][10], n=10;
omp_set_num_threads(4);
#pragma omp parallel default(none) private(i,myid,j)shared(a,n)
{
myid = omp_get_thread_num(); 
for(i = 0; i < n; i++){
for(j = myid; j < n; j+=omp_get_num_threads()){
printf("eu %d posição (%d,%d)\n", myid, i, j);
a[i][j] = 1.0;
}
}
}
for(i = 0; i < n; i++){
for(j = 0; j < n; j++){
printf("%d - ", a[i][j]);
}
printf("\n");
}
return 0;
}
