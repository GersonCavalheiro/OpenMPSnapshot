#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>



typedef struct matrix {
short **data;
short **fixed;
} MATRIX;

struct list_el {
MATRIX mat;
short i, j;
struct list_el *next;
};

typedef struct list_el item;
int l;
int SIZE;
FILE *inputMatrix;
MATRIX solution;
item *head;
item *tail;






MATRIX read_matrix_with_spaces() {
int i,j;  
MATRIX matrix;  
int element_int;

inputMatrix = fopen("three.txt", "r");


fscanf(inputMatrix, "%d", &element_int);
l = element_int;
SIZE = l*l;

matrix.data = (short**)malloc(SIZE*sizeof(short*));  
for (i=0;i<SIZE;i++)
matrix.data[i] = (short*) malloc (SIZE * sizeof (short));

matrix.fixed = (short**) malloc(SIZE * sizeof(short*));
for (i=0;i<SIZE;i++)
matrix.fixed[i] = (short*) malloc (SIZE * sizeof (short));

for (i=0; i<SIZE; i++) {
for (j=0; j<SIZE; j++) {     
matrix.fixed[i][j] = 0;
}
}

for(i = 0; i < SIZE; i++) {
for(j = 0; j < SIZE; j++){
fscanf(inputMatrix, "%d", &element_int);
matrix.data[i][j] = element_int;
if (matrix.data[i][j] != 0)
matrix.fixed[i][j] = 1;
}  
}

fclose(inputMatrix);

return matrix;
}



void printMatrix(MATRIX *matrix) {
int i,j;
for (i = 0; i < SIZE; i++) {
for (j = 0; j < SIZE; j++) {
printf("%2d ", matrix->data[i][j]);
}
printf("\n");
}
}



short permissible(MATRIX matrix, short i_line, short j_col) {

short line, column;
short value = matrix.data[i_line][j_col];

for (line = 0; line < SIZE; line++) {
if (matrix.data[line][j_col] == 0)
continue;

if ((i_line != line) && 
(matrix.data[line][j_col] == value)) 
return 0;
}

for (column = 0; column < SIZE; column++) {
if (matrix.data[i_line][column] == 0)
continue;

if (j_col != column && matrix.data[i_line][column] == value)
return 0;
}

short igroup = (i_line / l) * l;
short jgroup = (j_col / l) * l;
for (line = igroup; line < igroup+l; line++) {
for (column = jgroup; column < jgroup+l; column++) {
if (matrix.data[line][column] == 0)
continue;

if ((i_line != line) &&
(j_col != column) &&
(matrix.data[line][column] == value)) {
return 0;
}
}
}

return 1;
}


void decreasePosition(MATRIX* matrix, short* iPointer, short* jPointer){
do {
if (*jPointer == 0 && *iPointer > 0) {
*jPointer = SIZE - 1;
(*iPointer)--;
} else
(*jPointer)--;
} while (*jPointer >= 0 && (*matrix).fixed[*iPointer][*jPointer] == 1);
}



void increasePosition(MATRIX* matrix, short* iPointer, short* jPointer){

do{
if(*jPointer < SIZE-1)
(*jPointer)++;
else {
*jPointer = 0;
(*iPointer)++;
}
} while (*iPointer < SIZE && (*matrix).fixed[*iPointer][*jPointer]);
}


void freeListElement(item *node) {
int i;
for (i = 0; i < SIZE; i++) {
free(node->mat.data[i]);
free(node->mat.fixed[i]);
}
free(node->mat.data);
free(node->mat.fixed);
free(node);
}



item* createItem(MATRIX matrix, short i, short j){
item * curr = (item *)malloc(sizeof(item));
int m;
short x, y;


curr->mat.data = (short**)malloc(SIZE*sizeof(short*));
for (m=0;m<SIZE;m++)
curr->mat.data[m] = (short*) malloc (SIZE * sizeof (short));

curr->mat.fixed = (short**) malloc(SIZE * sizeof(short*));
for (m=0;m<SIZE;m++)
curr->mat.fixed[m] = (short*) malloc (SIZE * sizeof (short));


for(x = 0; x < SIZE; x++){
for(y = 0; y < SIZE; y++){
curr->mat.data[x][y] = matrix.data[x][y];
curr->mat.fixed[x][y] = matrix.fixed[x][y]; 
}
}


curr->i = i;
curr->j = j;
curr->next = NULL;

return curr;
}


void attachItem(item* newItem){

if(head == NULL){
head = newItem;
tail = newItem;
} else {
tail->next = newItem;
tail = newItem;
}
}


item* removeItem(){
item* result = NULL;
if(head != NULL){
result = head;
head = result->next;
if(head == NULL){
tail = NULL;
}
}
return result;
}



void initializePool2(MATRIX* matrix){

short i = 0;
short j = 0;

if ((*matrix).fixed[i][j] == 1)
increasePosition(matrix, &i, &j);

short num;
for(num = 0; num < SIZE; num++){
((*matrix).data[i][j])++;    
if (permissible(*matrix, i, j) == 1) {
item* newPath = createItem(*matrix, i, j);
attachItem(newPath);
} 
}

}


void initializePool(MATRIX* matrix){

short i = 0;
short j = 0;

if ((*matrix).fixed[i][j] == 1)
increasePosition(matrix, &i, &j);

short num=0;

item* current = NULL;

while(1) {
((*matrix).data[i][j])++;    

if (matrix->data[i][j] <= SIZE && permissible(*matrix, i, j) == 1) {
item* newPath = createItem(*matrix, i, j);
attachItem(newPath);
num++;
} else if(matrix->data[i][j] > SIZE) {
if(current != NULL){
freeListElement(current);
}

if(num >= SIZE){
break;
}

item* current = removeItem();

if(current == NULL){
break;
}

matrix = &(current->mat);
i = current->i;
j = current->j;

if(i == SIZE-1 && j == SIZE-1){
attachItem(current);
break;
}

num--;

increasePosition(matrix, &i, &j);
}
}

if(current != NULL){
freeListElement(current);
}


}

short bf_pool(MATRIX matrix) {

head = NULL;
tail = NULL;

initializePool(&matrix);






short found = 0;
short i, j;
item* current;
int level;

#pragma omp parallel shared(found) private(i,j, current, level)
{

#pragma omp critical (pool)
current = removeItem();

while(current != NULL && found == 0){

MATRIX currMat = current->mat;

i = current->i;
j = current->j;

increasePosition(&currMat, &i, &j);

level = 1;

while (level > 0 && i < SIZE && found == 0) {
if (currMat.data[i][j] < SIZE) {    
currMat.data[i][j]++;

if (permissible(currMat, i, j) == 1) {
increasePosition(&currMat, &i, &j);
level++;
}
} else {

currMat.data[i][j] = 0;
decreasePosition(&currMat, &i, &j);
level--;
} 

} 

if(i == SIZE){
found = 1;
solution = currMat;

continue;
}

free(current);

#pragma omp critical (pool)
current = removeItem();

}

}   

return found;
}



int main(int argc, char* argv[]) {


MATRIX m = read_matrix_with_spaces();

short hasSolution = bf_pool(m);
if(hasSolution == 0){
printf("No result!\n");
return 1;
}

printMatrix(&solution);
printf("lel");
item* node = head;
while (node != NULL) {
item* next = node->next;
freeListElement(node);
node = next;
}

return 0;
}