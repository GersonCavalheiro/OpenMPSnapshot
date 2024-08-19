#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#define LEN_SIZE 15
#define _DEBUG
int main(){
char str_0[100];
char **str_1;
int size_0 = 0;
int thread_num;
int i;
int count = 0;
double start_time, end_time;
FILE *fp = fopen("words.txt","r");
printf("Input the number of thread\n");
scanf("%d",&thread_num);
omp_set_num_threads(thread_num);
while(fscanf(fp,"%s",str_0) != EOF)size_0 += 1;
fclose(fp);
str_1 = (char **)malloc(sizeof(char *) * (size_0 + 1));
for(i = 0; i < size_0; i++){
str_1[i] = (char *)malloc(sizeof(char) * LEN_SIZE);
}
printf("The number of word is %d\n",size_0);
#ifdef DEBUG
printf("State 1\n");
#endif 
fp = fopen("words.txt","r");
for(i = 0; i < size_0; i++){
fscanf(fp,"%s",str_1[i]);
}
fclose(fp);
#ifdef DEBUG
printf("State 2\n");
printf("size_0 is %d\n",size_0);
#endif
start_time = omp_get_wtime();
fp = fopen("result.txt","w");
#pragma omp parallel for schedule(static) reduction(+:count)
for(i = 0; i < size_0; i++){
int j, k;
char l_0, l_1;
int len_0, len_1;
int flag;
len_0 = strlen(str_1[i]);
#ifdef DEBUG
printf("len_0 is %d i value is %d\n",len_0,i);
#endif
for(j = 0; j < size_0; j++){
len_1 = strlen(str_1[j]);
#ifdef DEBUG
if(i == j)printf("i, j, len, len %d %d %d %d\n",i,j,len_0,len_1);
#endif
if(len_0 == len_1){
flag = 0;
#ifdef DEBUG
printf("i j value is %d %d len size is %d %d\n",i,j,len_0,len_1);
#endif
for(k = 0; k < len_0; k++){
l_0 = str_1[i][k];
l_1 = str_1[j][len_0 - k - 1];
if(l_0 != l_1){
flag = 1;
}
}
if(flag == 0){
#ifdef DEBUG
printf("i and j is %d %d\n",i,j);
#endif
count = count + 1;
#pragma omp critical(write)
fprintf(fp,"%s\n",str_1[i]);
break;
}
}
}
}
fclose(fp);
end_time = omp_get_wtime();
printf("Number of Palindromic is %d\n",count);
printf("Elapsed time %f sec.\n",end_time - start_time);
return 0;
}
