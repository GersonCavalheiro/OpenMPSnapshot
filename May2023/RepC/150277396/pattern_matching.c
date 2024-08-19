#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <ctype.h>
#include <string.h>
#include <time.h>
#include <omp.h>
bool match(char *first, char * second) { 
if (*first == '\0' && *second == '\0') 
return true; 
if (*first == '*' && *(first+1) != '\0' && *second == '\0') 
return false; 
if (*first == '?' || *first == *second) 
return match(first+1, second+1); 
if (*first == '*') 
return match(first+1, second) || match(first, second+1); 
return false; 
} 
void remove_punct_and_make_lower_case(char *p) {
char *src = p, *dst = p;
while (*src) {
if (ispunct((unsigned char)*src)) {
src++;
} else if (isupper((unsigned char)*src)) {
*dst++ = tolower((unsigned char)*src);
src++;
} else if (src == dst) {
src++;
dst++;
} else {
*dst++ = *src++;
}
}
*dst = 0;
}
int main(void) {
FILE *infile;
char *buffer;
long numbytes;
char pattern[100];
char *array[400000];
infile = fopen("texto.txt", "r");
if(infile == NULL)
return 1;
fseek(infile, 0L, SEEK_END);
numbytes = ftell(infile);
fseek(infile, 0L, SEEK_SET);
buffer = (char*)calloc(numbytes, sizeof(char));	
if(buffer == NULL)
return 1;
fread(buffer, sizeof(char), numbytes, infile);
fclose(infile);
printf("WILDCARD PATTERN MATCHING");
printf("\n\n\nDigite o padrão desejado: ");
scanf("%s", pattern);
printf("\n\n");
char * line = strtok(strdup(buffer), "\n");
int numLines = 0;
while(line != NULL) {
remove_punct_and_make_lower_case(line); 
array[numLines] = line;
line  = strtok(NULL, "\n");
numLines++;
}
clock_t timeStart = clock(); 
printf("LINHAS QUE OCORREU O PADRÃO\n");
int numPatternFound = 0;
char linhaChar[100];
int i = 0;
#pragma omp parallel
{
#pragma omp for reduction(+:numPatternFound)
for(int i = 0; i < numLines; i++) {
if(match(pattern, array[i])) {
printf("%d: %s\n", i, array[i]);
numPatternFound++;
}
}
}
clock_t timeFinish = clock();
double executionTime = (double)(timeFinish - timeStart) / CLOCKS_PER_SEC;
printf("\n\n");
printf("TEMPO DE EXECUÇÃO: %lf\n", executionTime);
printf("O PADRÃO OCORREU %d VEZES\n", numPatternFound);
printf("\nFIM DA EXECUÇÃO\n");
free(buffer);
return 0;
}