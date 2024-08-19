#include<stdio.h>
#include<omp.h>
#define THREADS 1
int main() {
const int REPS = 1000000;
int i;
double balance = 0.0;

printf("\nYour starting bank account balance is %0.2f\n", balance);
#pragma omp parallel for  num_threads(THREADS)    
for (i = 0; i < REPS; i++) {
# pragma omp reduction ( + : balance )       
balance += 10.0;
}

printf("\nAfter %d $10 deposits, your balance is %0.2f\n",
REPS, balance);

#pragma omp parallel for  num_threads(THREADS) 
for (i = 0; i < REPS; i++) {
# pragma omp reduction ( - : balance )  
balance -= 10.0;
}
printf("\nAfter %d $10 withdrawals, your balance is %0.2f\n\n",
REPS, balance);
return 0;
}
