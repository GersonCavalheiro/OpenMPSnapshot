#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include "randomizer.h"
#include "probabilities.h"
int startingLevel = 141;
int counts[59] = {};
int main(int argc, char* argv[]){
if(argc > 1){
startingLevel = atoi(argv[1]);
}
TimeOfDaySeed();
omp_set_num_threads(8);
#pragma omp parallel for default(none) shared(startingLevel,probabilities) reduction(+:counts)
for(int i = 0; i < 10000000; i++){
int currentLevel = startingLevel;
int bonkCount = 0;
while(currentLevel < 200){
int bonk = probabilities[currentLevel-141][rand() % 20];
currentLevel += bonk;
bonkCount++;
}
counts[bonkCount-1]++;
}
for(int j = 1; j <= 59; j++){
printf("%d,%d\n", j, counts[j-1]);
}
}