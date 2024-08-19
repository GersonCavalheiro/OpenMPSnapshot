#include <ctype.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[]) { 
int num_words = 0;
int i=0;
#pragma omp parallel for
for (i = 0; i <= strlen(argv[1]); i++) { 
#pragma omp atomic update
if ((isspace(argv[1][i])) && !(isspace(argv[1][i+1])))
num_words ++;
}
printf("Num words = %d\n", num_words+1);
}