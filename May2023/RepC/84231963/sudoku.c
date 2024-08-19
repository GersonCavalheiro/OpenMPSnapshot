#include <stdio.h>
#include <stdlib.h>
#include "sudoku.h"
#include <string.h>
#include <stdbool.h>
#include <omp.h>
#define UNASSIGNED 0
bool possiblevalues[SIZE][SIZE][SIZE+1];
int **zerogrid;
int stack_sz;
bool solved;
struct stack* solvedgrid;
int** originalGrid ;
struct stack* top;
struct stack* bottom;
struct grid
{
int** my_grid;
bool** assigned; 
int** possiblevalues;     
};
struct stack
{
struct grid* Grid;
struct stack* next;
int row;
int col;
};
bool isok(int **grid, int row, int col, int num);
int eliminations(int **grid);
void twins(int **grid);
void loneranger(int **grid);
void stackinit(int** Grid, struct grid* curr);
void push(struct stack* newelement);
struct stack* pop();
struct stack* stackalloc(int index_i, int index_j);
void deletestack(struct stack* mystack);
void assign_possiblevalues(int value, struct grid* curr, int i, int j);
void Sudoku();
int elimination(struct grid* curr);
bool isok(int **grid, int row, int col, int num)
{
int myrow,mycol;
bool alreadyinrow=false;
for (mycol = 0; mycol < SIZE; mycol++)
if (grid[row][mycol] == num)
{
alreadyinrow=true;
mycol=SIZE;
}
bool alreadyincol=false;
for (myrow = 0; myrow < SIZE; myrow++)
if (grid[myrow][col] == num)
{
alreadyincol=true;
myrow=SIZE;
}
bool alreadyinbox=false;
int boxStartRow=row - row % MINIGRIDSIZE;
int boxStartCol= col - col % MINIGRIDSIZE;
for (myrow = 0; myrow < MINIGRIDSIZE; myrow++)
for (mycol = 0; mycol < MINIGRIDSIZE; mycol++)
if (grid[myrow+boxStartRow][mycol+boxStartCol] == num)
{
alreadyinbox=true;
myrow=MINIGRIDSIZE;
mycol=MINIGRIDSIZE;
}
return (!alreadyinrow && !alreadyincol && !alreadyinbox);
}
void Sudoku()
{
#pragma omp parallel
{
while ( top && !solved)
{
struct stack* curr = NULL;
#pragma omp critical
{
if (top!=NULL)
{
curr = pop();
}
}
bool cond=false;
int i,j;
for (i = curr->row ; i < SIZE; i++)
{
for (j = 0 ; j <SIZE; j++){
if ( curr->Grid->assigned[i][j] == 0)
{
curr->row = i;
curr->col = j;
cond=true;
i=SIZE;
j=SIZE;
}
}
}
if (curr && cond)
{
int num;
int test = curr->Grid->possiblevalues[curr->row][curr->col];
for ( num = 1; num <= SIZE ; num++)
{
if (!solved && (test & (1 << (num - 1))))
{
struct stack* mystack = stackalloc(curr->row, curr->col);
int i,j;
for ( i =0 ; i < SIZE ; i++)
{
for (j = 0 ; j < SIZE ; j++)
{
mystack->Grid->assigned[i][j] = curr->Grid->assigned[i][j] ;
mystack->Grid->my_grid[i][j] = curr->Grid->my_grid[i][j];
mystack->Grid->possiblevalues[i][j]= curr->Grid->possiblevalues[i][j];
}
}
assign_possiblevalues(num, mystack->Grid,mystack->row, mystack->col);
if (!solved && elimination(mystack->Grid) == 0){
if (!solved &&  isValid(zerogrid,mystack->Grid->my_grid))
{
#pragma omp critical
{
if (!solved)
{
solvedgrid = mystack;
solved = true;
}
}
}
#pragma omp critical
{
push(mystack);
}
}
}
}
}
if ( curr!=NULL)
deletestack(curr);
}
}
return;
}
void stackinit(int** Grid, struct grid* curr)
{
int i,j;
int* row = (int*) malloc (SIZE*sizeof (int));
int* column = (int*) malloc (SIZE*sizeof (int));
int* box = (int*) malloc (SIZE*sizeof (int));
for(i=0;i<SIZE;i++){
row[i]=0;
column[i]=0;
box[i]=0;
}
for( i = 0; i < SIZE ; i++)
{
for ( j = 0 ; j < SIZE ; j++)
{
curr->my_grid[i][j] = Grid[i][j];
if ( Grid[i][j] != 0){
int k = 1 << ( Grid[i][j] - 1);
curr->assigned[i][j] = 1;
row[i] |= k;
column[j] |= k;
box[(i/MINIGRIDSIZE)*MINIGRIDSIZE + j/MINIGRIDSIZE] |= k;
}
}
}
for( i = 0; i < SIZE ; i++)
{
for ( j = 0 ; j < SIZE ; j++)
{
if ( curr->assigned[i][j] == 0)
{
int temp = row[i] | column[j] | box[(i/MINIGRIDSIZE)*MINIGRIDSIZE + j/MINIGRIDSIZE] ;
temp = ~temp;
temp &= ((1 << SIZE) - 1);
curr->possiblevalues[i][j] = temp;
}
else
{
curr->possiblevalues[i][j] = 0;
}
}
}
}
void push(struct stack* newelement)
{
if ( !top)
{
top = newelement;
top->next = NULL;
bottom = newelement;
}
else
{
bottom->next = newelement;
bottom = newelement;
}
}
struct stack* pop()
{
if (top)
{
struct stack* ret = top;
top = top->next;
return ret;
}
else
{
printf("error:trying to pop off from an empty stack\n");
exit(0);
}
}
struct stack* stackalloc(int myrow, int mycol)
{
struct stack* mystack;
mystack = malloc(sizeof(struct stack));
mystack->Grid = malloc(sizeof(struct grid));
int row,col;
mystack->Grid->my_grid = (int**)malloc(SIZE*sizeof(int*));
for (row=0; row<SIZE; row++)
{
mystack->Grid->my_grid[row] = (int*) malloc (SIZE * sizeof (int));
}
mystack->Grid->possiblevalues = (int**)malloc(SIZE*sizeof(int*));
for (row=0; row<SIZE; row++)
{
mystack->Grid->possiblevalues[row] = (int*) malloc (SIZE*sizeof (int));
for (col=0;col<SIZE;col++){
mystack->Grid->possiblevalues[row][col] = 0;
}
}
mystack->Grid->assigned = (bool**)malloc(SIZE*sizeof(bool*));
for (row=0; row<SIZE; row++)
{
mystack->Grid->assigned[row] = (bool*) malloc (SIZE*sizeof (bool));
for (col=0;col<SIZE;col++)
{
mystack->Grid->assigned[row][col] = 0;
}
}
mystack->next = NULL;
mystack->row = myrow;
mystack->col = mycol;
return mystack;
}
void deletestack(struct stack* mystack){
int i;
for (i = 0; i < SIZE; i++) 
{
free(mystack->Grid->my_grid[i]);
free(mystack->Grid->possiblevalues[i]);
free(mystack->Grid->assigned[i]);
}
free(mystack->Grid->my_grid);
free(mystack->Grid->possiblevalues);
free(mystack->Grid->assigned);
free(mystack);
}
void twins(int **grid)
{
for (int row = 0; row < SIZE; row++)
{
for (int col=0; col<SIZE; col++)
{
int nonzero=0;
int t;
for(t=1;t<SIZE+1;t++)
{
if(possiblevalues[row][col][t])nonzero++;
}
if(nonzero>=2)
{
int nonzero_possible[SIZE];
int nonzero_index=0;
int j;
for(j=1;j<=SIZE;j++){
if(possiblevalues[row][col][j]){
nonzero_possible[nonzero_index]=j;
nonzero_index++;
}
}
int p,q;
for(p=0;p<nonzero_index-1;p++){
int twin1=nonzero_possible[p];
for(q=p+1;q<nonzero_index;q++){
int twin2=nonzero_possible[q];
int index=0;
int nxtcol;
int uniquecol;
for(nxtcol=0;nxtcol<SIZE;nxtcol++){
if(nxtcol!=col){
if(possiblevalues[row][nxtcol][twin1] && possiblevalues[row][nxtcol][twin2]){
index++;
uniquecol=nxtcol;
}
if(index>1)nxtcol=SIZE;
}
}
if(index==1){
bool occur=false;
for(nxtcol=0;nxtcol<SIZE;nxtcol++){
if(nxtcol!=col && nxtcol!=uniquecol){
if(possiblevalues[row][nxtcol][twin1] || possiblevalues[row][nxtcol][twin2]){
occur=true;  
nxtcol=SIZE;
}
}
}
if(!occur){
#pragma omp parallel for
for(int i=1;i<=SIZE;i++){
if(i!=twin1 && i!=twin2){
possiblevalues[row][col][i]=false;  
possiblevalues[row][uniquecol][i]=false;  
}
}
}
}
}
}
}
}
}
for (int col = 0; col < SIZE; col++){
for (int row=0; row<SIZE; row++){
int nonzero=0;
int t;
for(t=1;t<SIZE+1;t++){
if(possiblevalues[row][col][t])nonzero++;
}
if(nonzero>=2){
int nonzero_possible[SIZE];
int nonzero_index=0;
int j;
for(j=1;j<=SIZE;j++){
if(possiblevalues[row][col][j]){
nonzero_possible[nonzero_index]=j;
nonzero_index++;
}
}
int p,q;
for(p=0;p<nonzero_index-1;p++){
int twin1=nonzero_possible[p];
for(q=p+1;q<nonzero_index;q++){
int twin2=nonzero_possible[q];
int index=0;
int nxtcol;
int uniquecol;
for(nxtcol=0;nxtcol<SIZE;nxtcol++){
if(nxtcol!=col){
if(possiblevalues[row][nxtcol][twin1] && possiblevalues[row][nxtcol][twin2]){
index++;
uniquecol=nxtcol;
}
if(index>1)nxtcol=SIZE;
}
}
if(index==1){
bool occur=false;
for(nxtcol=0;nxtcol<SIZE;nxtcol++){
if(nxtcol!=col && nxtcol!=uniquecol){
if(possiblevalues[row][nxtcol][twin1] || possiblevalues[row][nxtcol][twin2]){
occur=true;  
nxtcol=SIZE;
}
}
}
if(!occur){
int i;
#pragma omp parallel for
for(i=1;i<=SIZE;i++){
if(i!=twin1 && i!=twin2){
possiblevalues[row][col][i]=false;  
possiblevalues[row][uniquecol][i]=false;  
}
}
}
}
}
}
}
}
}
return;
}
int eliminations(int **grid){ 
int row,col;
int eliminations=0;
for (row = 0; row < SIZE; row++){
for (col = 0; col < SIZE; col++){
if(grid[row][col]==UNASSIGNED)
{
int i;
int index;
int num_possible=0;
for (i = 0; i <= SIZE; i++)
{
if(possiblevalues[row][col][i])
{
index=i;
num_possible++;
}
if(num_possible>1)
i=SIZE;
}
if(num_possible==1)
{
eliminations++;
grid[row][col]=index;
possiblevalues[row][col][index]=true;
#pragma omp parallel
{
#pragma omp sections
{
#pragma omp section
{
#pragma omp parallel for
for(int i=0;i<=SIZE;i++){
if(i!=index)
possiblevalues[row][col][i]=false;
}
}
#pragma omp section
{
#pragma omp parallel for
for(int myrow=0;myrow<SIZE;myrow++){
if(myrow!=row)
possiblevalues[myrow][col][index]=false;
}
}
#pragma omp section
{
#pragma omp parallel for
for(int mycol=0;mycol<SIZE;mycol++)
{
if(mycol!=col)
possiblevalues[row][mycol][index]=false;
}
}    
#pragma omp section
{
int boxStartRow=row - row % MINIGRIDSIZE;
int boxStartCol= col - col % MINIGRIDSIZE;
#pragma omp parallel for
for (int myrow = 0; myrow < MINIGRIDSIZE; myrow++)
for (int mycol = 0; mycol < MINIGRIDSIZE; mycol++)
if (myrow!=row && mycol!=col)
{
possiblevalues[myrow+boxStartRow][mycol+boxStartCol][index]=false;
}
}
}             
}             
}
}
}
}
return eliminations;
}
void loneranger(int **grid){
int row,col;
for (row = 0; row < SIZE; row++)
{
int num;
for(num=1;num<=SIZE;num++)
{
int index_row=0;
int uniquecol;
for (col = 0; col < SIZE; col++)
{
if(possiblevalues[row][col][num])
{
uniquecol=col;
index_row++;
}
if(index_row>1)
col=SIZE;
}
if(index_row==1)
{
grid[row][uniquecol]=num;
possiblevalues[row][uniquecol][num]=true;
#pragma omp parallel
{
#pragma omp sections
{
#pragma omp section
{
#pragma omp parallel for
for(int i=0;i<=SIZE;i++)
{
if(i!=num)
possiblevalues[row][uniquecol][i]=false;
}
}
#pragma omp section
{
#pragma omp parallel for
for(int myrow=0;myrow<SIZE;myrow++)
{
if(myrow!=row)
possiblevalues[myrow][uniquecol][num]=false;
}
}
#pragma omp section
{
#pragma omp parallel for
for(int mycol=0;mycol<SIZE;mycol++)
{
if(mycol!=uniquecol)
possiblevalues[row][mycol][num]=false;
}
}
#pragma omp section
{    
int boxStartRow=row - row % MINIGRIDSIZE;
int boxStartCol= uniquecol - uniquecol % MINIGRIDSIZE;
#pragma omp parallel for
for (int myrow = 0; myrow < MINIGRIDSIZE; myrow++)
for (int mycol = 0; mycol < MINIGRIDSIZE; mycol++)
if (myrow!=row && mycol!=uniquecol)
{
possiblevalues[myrow+boxStartRow][mycol+boxStartCol][num]=false;
}
}            
}
}        
}
int index_col=0;
int uniquerow;
for (col = 0; col < SIZE; col++)
{
if(possiblevalues[col][row][num])
{
uniquerow=col;
index_col++;
}
if(index_col>1)
col=SIZE;
}
if(index_col==1)
{
grid[uniquerow][row]=num;
possiblevalues[uniquerow][row][num]=true;
#pragma omp parallel 
{
#pragma omp sections
{
#pragma omp section
{
#pragma omp parallel for
for(int i=0;i<=SIZE;i++)
{
if(i!=num)
possiblevalues[uniquerow][row][i]=false;
}
}
#pragma omp section
{
#pragma omp parallel for
for(int myrow=0;myrow<SIZE;myrow++)
{
if(myrow!=uniquerow)
possiblevalues[myrow][row][num]=false;
}
}
#pragma omp section
{
#pragma omp parallel for
for(int mycol=0;mycol<SIZE;mycol++)
{
if(mycol!=row)
possiblevalues[uniquerow][mycol][num]=false;
}
}
#pragma omp section
{
int boxStartRow=uniquerow - uniquerow % MINIGRIDSIZE;
int boxStartCol= row - row % MINIGRIDSIZE;
#pragma omp parallel for
for (int myrow = 0; myrow < MINIGRIDSIZE; myrow++)
for (int mycol = 0; mycol < MINIGRIDSIZE; mycol++)
if (myrow!=uniquerow && mycol!=row)
{
possiblevalues[myrow+boxStartRow][mycol+boxStartCol][num]=false;
}
}            
}            
}       
}
}
}
return;
}
int elimination(struct grid* curr)
{
int row,col;
for( row = 0; row < SIZE ; row++){
for ( col = 0 ; col < SIZE ; col++){
if ( !(curr->assigned[row][col])){
if ( curr->possiblevalues[row][col] == 0 )
return -1; 
else
{
if ( (curr->possiblevalues[row][col] & (curr->possiblevalues[row][col] - 1)) == 0 ){
int k = 0;
do{
k++;
curr->possiblevalues[row][col]>>=1;
}while (curr->possiblevalues[row][col]);
assign_possiblevalues(k,curr,row,col);
col = SIZE;
row=-1;
}
}
}
}
}
return 0;
}
void init_possiblevalues(int **grid){
#pragma omp parallel for
for (int row = 0; row < SIZE; row++)
{
for (int col = 0; col < SIZE; col++)
{
for(int i=0;i<SIZE+1;i++)
{
possiblevalues[row][col][i]=false;
}
if(grid[row][col] != UNASSIGNED)
{
possiblevalues[row][col][grid[row][col]]=true;
}
else
{   
for (int num = 1; num <= SIZE; num++)
{
if(isok(grid,row,col,num))
{
possiblevalues[row][col][num]=true;
}
}
}
}
}
}
void assign_possiblevalues(int value, struct grid* curr, int row, int col)
{
curr->my_grid[row][col] = value;
curr->assigned[row][col] = 1;
curr->possiblevalues[row][col] = 0;
int temp = ~(1 << ( value - 1));
int ind;
for (ind = 0 ; ind < SIZE ; ind++)
{
curr->possiblevalues[row][ind] &= temp;
curr->possiblevalues[ind][col] &= temp;
}
int boxStartRow = (row/MINIGRIDSIZE) * MINIGRIDSIZE;
int boxStartCol = (col/MINIGRIDSIZE) * MINIGRIDSIZE;
int i,j;
for (i = boxStartRow; i < boxStartRow + MINIGRIDSIZE; i++)
for (j = boxStartCol; j < boxStartCol+ MINIGRIDSIZE; j++)
curr->possiblevalues[i][j] &= temp;
}
int** solveSudoku(int **original_Grid)
{
int i,j,num_threads;
zerogrid= (int**)malloc(sizeof(int*)*SIZE);
for (i=0;i<SIZE;i++)
{
zerogrid[i] = (int*)malloc(sizeof(int)*SIZE);
for (j=0;j<SIZE;j++)
zerogrid[i][j] = 0;
}
if(SIZE==9||SIZE==16 || SIZE==25)
{
init_possiblevalues(original_Grid);
int x;
for (bool isupdate=true;isupdate;isupdate= (x>0) ?true:false )
{
#pragma omp parallel 
{
#pragma omp sections
{ 
#pragma omp section
{   
x=eliminations(original_Grid);
}
#pragma omp section
{
loneranger(original_Grid);
}
#pragma omp section
{
twins(original_Grid);
} 
}
}
}
}
solvedgrid = NULL;
solved = false;
top = NULL;
struct stack* curr;
curr = stackalloc(0,0);
stackinit(original_Grid, curr->Grid);
int update;
update = elimination(curr->Grid);
if ( update == -1)
{
return original_Grid;
}
else
{
if (isValid(zerogrid,curr->Grid->my_grid))
{
return curr->Grid->my_grid;
}
else
{
push(curr);
num_threads = 1;
#pragma omp parallel
{
num_threads = omp_get_num_threads();
}
update=0;
for(stack_sz=1;top!=NULL &&  stack_sz < num_threads;stack_sz--)
{
struct stack* curr = pop();
bool cond=false;
int i,j;
for (i = curr->row ; i < SIZE; i++)
{
for (j = 0 ; j <SIZE; j++)
{
if ( curr->Grid->assigned[i][j] == 0)
{
curr->row = i;
curr->col = j;
cond=true;
i=SIZE;
j=SIZE;
}
}
}
if (cond)
{
int num;
int test = curr->Grid->possiblevalues[curr->row][curr->col];
for ( num = 1; num <= SIZE ; num++)
{
if (test & (1 << (num - 1)))
{
struct stack* temp = stackalloc(curr->row, curr->col);
for ( i =0 ; i < SIZE ; i++)
{
for (j = 0 ; j < SIZE ; j++)
{
temp->Grid->assigned[i][j] = curr->Grid->assigned[i][j] ;
temp->Grid->my_grid[i][j] = curr->Grid->my_grid[i][j];
temp->Grid->possiblevalues[i][j]= curr->Grid->possiblevalues[i][j];
}
}
assign_possiblevalues(num, temp->Grid,temp->row, temp->col);
if (elimination(temp->Grid) == 0)
{
if (isValid(zerogrid,temp->Grid->my_grid))
{
solvedgrid = temp;
return solvedgrid->Grid->my_grid;
}
push(temp);
stack_sz++;
}
}
}
}
deletestack(curr);
}
if ( update == 0 && stack_sz == 0)
{
return original_Grid;
}
else
{ 
Sudoku();
if ( !solved)
{
return original_Grid;
}
else
{
return solvedgrid->Grid->my_grid;
}
}
}
}
}
