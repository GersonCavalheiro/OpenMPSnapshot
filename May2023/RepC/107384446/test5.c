#include <stdio.h>
#include <omp.h>
int main() {
FILE *fp ;
int count=0;
char str[10];
fp= fopen("files/negative.txt","r");
#pragma omp parallel
{
while(fscanf(fp,"%s",str)!=EOF){
#pragma omp task shared(count)
count=count+1;
}
}
#pragma omp taskwait
printf("%d\n",count );
fclose(fp);
return 0;
}
