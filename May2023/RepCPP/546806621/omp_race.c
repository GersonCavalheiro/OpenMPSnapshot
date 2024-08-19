#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <omp.h>

#define MAX_LINE_LENGTH 80

int read_file(char *path,int *buff,int *size){

char line[MAX_LINE_LENGTH] = {0};
unsigned int line_count = 0,i=0;



FILE *file = fopen(path, "r");

if (!file)
{
perror(path);
return EXIT_FAILURE;
}


while (fgets(line, MAX_LINE_LENGTH, file))
{

buff[i] = atoi(line);
i++;

}

*size = i;

if (fclose(file))
{
return EXIT_FAILURE;
perror(path);
}
return 0;
}






int main(int argc, char *argv[]){

int num_size, true_n0=646016;
int numbers[2000000];

read_file("num.txt",numbers,&num_size);
printf("Size of integer array/file: %d\n",num_size);

int maxval = 0;
#pragma omp parallel for
for (int i=0;i<num_size;i++) if (numbers[i] > maxval) maxval = numbers[i];
printf("max number in file: %d\n",maxval);  

int num_n0 = 0;
#pragma omp parallel for
for (int i=0;i<num_size;i++) if (numbers[i] == 0) num_n0++;
printf("number of 0s in file: %d\n",num_n0);  
printf("true number of 0s in file: %d\n",true_n0);  

return 0;
}
