# Parallel Crout Decomposition
In this project, I have made a parallelized and distributed version of a sequential Crout Decomposition of a matrix using various strategies.
Crout decomposition factors a matrix as the product of a lower and upper triangular matrix. More about LU decomposition can be found in this [Wikipedia Article]
## Installation and Usage Instructions
- Clone the repository to your local computer.
- Run compile.sh to compile the c files.
- To execute the compiled files do this
```
bash run.sh <Dimension of input square matrix> <Input File> <Number of Threads> <Strategy (0/1/2/3/4)>
```
- N denotes the dimension of the square matrix for which is given as input.
- The Input file contains the matrix that needs to be decomposed. Each line represents a row with space separated numbers for different columns.
- I have implemented multiple strategies to make the code parallel. The number specifies which one to run. I have talked about the different stategies in the subsequent section.
- The code output 2 files, one containing te L matrix and the other U matrix.
## Strategies Employed
### Strategy 0
- This corresponds to the below sequential code 
```
void crout_0(double **A, double **L, double **U, int n) {
int i, j, k;
double sum = 0;
for (i = 0; i < n; i++) {
U[i][i] = 1;
}
for (j = 0; j < n; j++) {
for (i = j; i < n; i++) {
sum = 0;
for (k = 0; k < j; k++) {
sum = sum + L[i][k] * U[k][j];
}
L[i][j] = A[i][j] - sum;
}
for (i = j; i < n; i++) {
sum = 0;
for(k = 0; k < j; k++) {
sum = sum + L[j][k] * U[k][i];
}
if (L[j][j] == 0) {
exit(0);
}
U[j][i] = (A[j][i] - sum) / L[j][j];
}
}
}
```
### Strategy 1
- I used the <code>parallel for</code> construct of <code> OpenMP</code> to parallelize the 2 internal loops inside the big j loop.
- I removed any possibilty of data race by making variable sum,i,k private.
### Strategy 2
- I used the ```parallel sections``` construct of ```OpenMP```.
- In this I distribute the 2 internal loops to 2 sections and also break each loop into 2 chunks. Thus I created 4 sections in total.
- I did not increase the number of sections and preferred to hard code it since it was leading to great increase in the code base and also to the errors that I was facing. Thus I did a tradeoff and settled for 4 sections. For number of threads different that 4, openmp internally handles them.
- When trying to run the 2 loops in parallel, a new data race is created since the first loop tries to update L[j][j] when the other loops reads it. To remove this I started the loop from i=j+1 rather than j and computed the i=j iteration before the sections pragma.
### Strategy 3
- I used a combination of both ```parallel for``` and ```parallel sections``` construct of ```OpenMP```.
- In this I divide the 2 internal loops to two sections and then use the ```parallel for pragma```. I used ideas used in 1,2 to remove data races from our code.
### Strategy 4
- I wrote an ```MPI``` version in this that solves the problem in a distributed manner. I did so by exploring 2 strategies to do the same.
- Strategy-1
- In this I divide the iterations of the loop in a round robin fashion. for the ith iteration, give it to the process that has its rank such that i%(num of processes)=rank i.e. iteration 1 goes to process 1, 2 to process 2....
- After computing the necesarry matrix value, broadcast it to all the processes.
- Strategy-2
- In some cases as the number of processes increased, the above strategy was taking a lot of time. Especially with 16 processes, the strategy did not stop for quite a lot of time (I was running it on 8 core machine)
- So I tried another strategy, where we do not broadcast updates cell by cell. Instead we scatter the L, U array, do the computation in each process and gather it again.
- This strategy finished faster on p=16 case. But for all other cases it was taking more time than strategy1.
- The possible increase in time for p=16 case might be due to the less nuber of cores available in my executions. Thus I have stuck to the first strategy in the submission. The bash files compile and run the strategy1 which is in crout_4. I have included strategy2 just to show our exploration while coming up with an implementation. Strategy2 is in crout_4_strat2.c file.
---
[Wikipedia Article]: https:
