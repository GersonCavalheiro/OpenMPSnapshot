#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sys/time.h>
#include <algorithm>
#include <fstream>
#include <random>
#include <time.h>
#include <fcntl.h>
#include <ctime>
#include <omp.h>
#include <sys/time.h>

int main(int argc, char *argv[])
{
struct timeval _ttime;
struct timezone _tzone;

gettimeofday(&_ttime, &_tzone);
double time_start = (double)_ttime.tv_sec + (double)_ttime.tv_usec / 1000000.;

if (argc > 0)
{
for (int i = 0; i < argc; ++i)
{
printf("%s\n", argv[i]);
}
}

printf("argc: %d\n", argc);

int user_numbers[7] = {4, 7, 17, 20, 32, 42, 14};
int best_attempt[7] = {0, 0, 0, 0, 0, 0, 0};
int best_attempt_correct_rows[7] = {0, 0, 0, 0, 0, 0, 0};

if (argc == 8)
{
for (int i = 0; i < 7; ++i)
{
user_numbers[i] = atoi(argv[i + 1]);
}
}
else
{
printf("No input arguments\n");
}

bool has_won = false;
unsigned long long int attempts = 0ULL;
int tid;
int maximum_correct_numbers = 0;

while (!has_won)
#pragma omp parallel shared(user_numbers, has_won, maximum_correct_numbers, best_attempt) private(tid)
{
#pragma omp atomic
attempts++;

tid = omp_get_thread_num();

srand(time(NULL));

bool all_zeroes = true;
for (int i = 0; i < 7; ++i)
{
if (user_numbers[i] != 0)
{
all_zeroes = false;
}
}

if (all_zeroes)
{
for (int i = 0; i < 7; ++i)
{
printf("please choose %d number in the range of 1-49: ", i + 1);
scanf("%d", &user_numbers[i]);
while (user_numbers[i] < 1 || user_numbers[i] > 49)
{
printf("please choose %d number in the range of 1-49: ", i + 1);
scanf("%d", &user_numbers[i]);
}
for (int j = 0; j < i; ++j)
{
if (user_numbers[i] == user_numbers[j])
{
printf("please choose %d number in the range of 1-49: ", i + 1);
scanf("%d", &user_numbers[i]);
}
}
}
}

int random_numbers[7];
for (int i = 0; i < 7; ++i)
{
random_numbers[i] = rand() % 49 + 1;
for (int j = 0; j < i; ++j)
{
if (random_numbers[i] == random_numbers[j])
{
random_numbers[i] = rand() % 49 + 1;
j = -1;
}
}
}

int correct = 0;
for (int i = 0; i < 7; ++i)
{
if (std::find(user_numbers, user_numbers + 7, random_numbers[i]) != user_numbers + 7)
{
correct++;
}
else
{
}
}

if (correct > maximum_correct_numbers)
{
maximum_correct_numbers = correct;

for (int i = 0; i < 7; ++i)
{
best_attempt[i] = user_numbers[i];

best_attempt_correct_rows[i] = random_numbers[i];
}
}

if (std::equal(user_numbers, user_numbers + 7, random_numbers))
{
printf("\n\n\033[1;32myou have won!\033[0m\n");
has_won = true;
}

if (attempts % 100000ULL == 0)
{
if (tid == 0)
{
printf("\n\n\033[1;34mmaximum correct numbers: \033[0m%d\n", maximum_correct_numbers);

printf("\n\n\033[1;34mbest attempt: \033[0m");
for (int i = 0; i < 7; ++i)
{
if (std::find(best_attempt_correct_rows, best_attempt_correct_rows + 7, best_attempt[i]) != best_attempt_correct_rows + 7)
{
printf("\033[1;32m%d\033[0m ", best_attempt[i]);
}
else
{
printf("\033[1;31m%d\033[0m ", best_attempt[i]);
}
}

printf("\n\n\033[1;33mwinning row:\033[0m \n");
for (int i = 0; i < 7; ++i)
{
printf("\033[1;33m%d\033[0m ", best_attempt_correct_rows[i]);
}

printf("\n\n\nattempts: %llu\n", attempts);

gettimeofday(&_ttime, &_tzone);
double time_end = (double)_ttime.tv_sec + (double)_ttime.tv_usec / 1000000.;

printf("   Wall clock run time    = %.1lf secs\n", time_end - time_start);
}
}
} 

return (0);
}
