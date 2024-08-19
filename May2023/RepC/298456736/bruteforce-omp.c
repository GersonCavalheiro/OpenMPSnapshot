#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "bruteforce-util.c"
#include "../hash/hash.h"
#include "../globals.h"
int bruteforce_crack(char *password_hash, char *characters, int password_max_length, int verbose)
{
int number_of_characters = strlen(characters);
if (verbose)  print_stats(password_hash, characters, number_of_characters, password_max_length);
int result = NOT_FOUND;
int i, j;
for (i = 1; i <= password_max_length && result > 0; i++)
{
long possibilities = calculate_possibilities(number_of_characters, i, verbose, 0);
for (j = 0; j < possibilities && result > 0;)
{
int nextStep = calculate_next_step(j, possibilities, CHUNK_SIZE);
int k;
#pragma omp parallel
{
#pragma omp for schedule(auto)
for (k = j; k < nextStep; k++)
{
unsigned char buffer[65];
char passwordToTest[i + 1];
generate_password(i, characters, number_of_characters, k, passwordToTest);
hash(passwordToTest, buffer);
if (!strcmp(password_hash, buffer))
{
#pragma omp critical
{
printf("Password found: %s\n", passwordToTest);
result = FOUND;
}
}
}
}
j=nextStep;
}
}
if (result == NOT_FOUND)
{
printf("Password not found.\n");
}
return result;
}