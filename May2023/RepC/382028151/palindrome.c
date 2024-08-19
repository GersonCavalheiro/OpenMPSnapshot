#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<omp.h>
int main(int argc, char* argv[]){
int numthread;
FILE *fp1, *fp2;
char origin[50], reverse[50];
char dict[30000][30];
int cnt = 0, flag = 0;
double start, end;
if (argc != 4){
fputs("ERROR: # of arguments must be 4\n\t[exec filename] [# of threads] [input filename] [output filename]\n", stderr);
exit(1);
}
fp1 = fopen(argv[2], "r");
if (fp1 == NULL){
fputs("ERROR: input file open error\n", stderr);
exit(1);
}
fp2 = fopen(argv[3], "w");
if (fp2 == NULL){
fputs("ERROR: output file open error\n", stderr);
exit(1);
}
start = omp_get_wtime();
numthread = atoi(argv[1]);
omp_set_num_threads(numthread);
while (fgets(origin, sizeof(origin), fp1) != NULL){
origin[strlen(origin) - 2] = '\0';
for (int i = 0; i < strlen(origin); i++)
reverse[i] = origin[strlen(origin) - i - 1];
reverse[strlen(origin)] = '\0';
strcpy(dict[cnt], origin);
cnt++;
flag = 0;
#pragma omp parallel for
for(int i = 0; i < cnt; i++){
if (!strcmp(dict[i], reverse)){
#pragma omp critical
flag = 1;
}
}
if (flag == 1)
fprintf(fp2, "%s %s\n", origin, reverse);
}
end = omp_get_wtime();
printf("%d threads > %lf sec\n", numthread, end-start);
fclose(fp1);
fclose(fp2);
return 0;
}
