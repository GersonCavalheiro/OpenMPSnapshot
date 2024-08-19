#include <stdio.h>
#include <omp.h>
int main() {
char str[10];
char temp[20] = "files/";
int i=0,length=0,count=0;
length=strlen(temp);
#pragma omp parallel for shared(temp,i)
for(i=0;i<5;i++){
temp[length+i]='c';
}
temp[length+5]='\0';
printf("%s %d\n",temp,i );
return 0;
}
