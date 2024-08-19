#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>

#define N 4


int ROWS_PLACED[N];	

bool VALID_ROWS[N][N] = {false};	


void board_init();

void calc_next_valid(int col);
int get_next_valid(int col);
bool isvalid(int row, int col);

void print_results(int tries);
void print_board(void);


int main(){
board_init();

int tries = 0;
int valid_row;

for(int col=0;col<N;){
calc_next_valid(col);
valid_row = get_next_valid(col);
printf("COLUMN: %d VALID ROW: %d\n", col, valid_row);
if(valid_row >=0){
ROWS_PLACED[col++] = valid_row;		
tries++;
}else{
ROWS_PLACED[col] = -1;	
if(col == 0){
printf("NO SOLUTION EXISTS\n");
return 0;
}

col--;	
}
}

print_results(tries);
return 0;
}


void board_init(){
#pragma omp parallel for
for(int i=0;i<N;i++){
ROWS_PLACED[i] = -1;	
VALID_ROWS[0][i] = true;	
}
}


void calc_next_valid(int col){
#pragma omp parallel for
for(int row=0;row<N;row++){
VALID_ROWS[col][row] = isvalid(row, col);
}
}


bool isvalid(int row, int col){
for(int prev_col=0;prev_col<col;prev_col++){
if((row == ROWS_PLACED[prev_col]) ||
(abs(row - ROWS_PLACED[prev_col]) == (col - prev_col)))
return false;
}
return true;
}


int get_next_valid(int col){
int old_position = ROWS_PLACED[col];

if(old_position == -1){		
for(int i=0;i<N;i++){
if(VALID_ROWS[col][i])
return i;
}
}else if(old_position == col-1){	
return -1;
}else{
for(int i=old_position+1;i<N;i++){
if(VALID_ROWS[col][i]){
return i;
}
}
}
return -1;
}

void print_results(int tries){
printf("TRIES: %d\n", tries);
printf("-----BOARD-----\n\n");
print_board();
}

void print_board(void){
for(int i=0;i<N;i++){
for(int j=0;j<N;j++){
if(ROWS_PLACED[j] == i){
printf("* ");
}else{
printf("- ");
}
}
printf("\n");
}
printf("\nQUEEN = *\n");
}
