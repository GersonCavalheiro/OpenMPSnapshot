#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <time.h>
#include <locale.h>
#include <windows.h>

#define M 4
#define N 4
#define MATRIX_SIZE 16
#define BUFFER_LEN 32
#define WORDS_LEN 50000

int meshCount = 0;

typedef struct puzzle {
char *matrix;
int m;
int n;
} PuzzleMatrix;

typedef struct {
char *holder;
int *path;
int pathLen;
int currentCell;
} WalkBundle;

typedef struct mesh {
struct mesh *a;
struct mesh *b;
struct mesh *c;
struct mesh *ch;
struct mesh *d;
struct mesh *e;
struct mesh *f;
struct mesh *g;
struct mesh *gh;
struct mesh *h;
struct mesh *i;
struct mesh *ih;
struct mesh *j;
struct mesh *k;
struct mesh *l;
struct mesh *m;
struct mesh *n;
struct mesh *o;
struct mesh *oh;
struct mesh *p;
struct mesh *r;
struct mesh *s;
struct mesh *sh;
struct mesh *t;
struct mesh *u;
struct mesh *uh;
struct mesh *v;
struct mesh *y;
struct mesh *z;
struct mesh *w;
struct mesh *q;
struct mesh *x;
int word;
} WordMesh;

PuzzleMatrix* CreatePuzzleMatrix(char* matrix, int m, int n) {
if(matrix != NULL && m > 0 && n > 0) {
char *myMatrix = (char*) malloc(sizeof(char) * (m *  n));
strcpy(myMatrix, matrix);
PuzzleMatrix *myPuzzleMatrix = (PuzzleMatrix*) malloc(sizeof(PuzzleMatrix));
myPuzzleMatrix->matrix = myMatrix;
myPuzzleMatrix->m = m;
myPuzzleMatrix->n = n;
return myPuzzleMatrix;
} else {
return NULL;
}
}

WordMesh* CreateWordMesh() {
WordMesh *mesh = (WordMesh*) malloc(sizeof(WordMesh));
mesh->word = 0;
mesh->a = NULL;
mesh->b = NULL;
mesh->c = NULL;
mesh->ch = NULL;
mesh->d = NULL;
mesh->e = NULL;
mesh->f = NULL;
mesh->g = NULL;
mesh->gh = NULL;
mesh->h = NULL;
mesh->i = NULL;
mesh->ih = NULL;
mesh->j = NULL;
mesh->k = NULL;
mesh->l = NULL;
mesh->m = NULL;
mesh->n = NULL;
mesh->o = NULL;
mesh->oh = NULL;
mesh->p = NULL;
mesh->r = NULL;
mesh->s = NULL;
mesh->sh = NULL;
mesh->t = NULL;
mesh->u = NULL;
mesh->uh = NULL;
mesh->v = NULL;
mesh->y = NULL;
mesh->z = NULL;
mesh->x = NULL;
mesh->w = NULL;
mesh->q = NULL;
meshCount++;
return mesh;
}

void PushWord(WordMesh *masterMesh, char* word) {
int i;
int len = strlen(word);
for(i = 0; i < len; i++) {
if(word[i] == 'a') {
if(masterMesh->a == NULL)
masterMesh->a = CreateWordMesh();
masterMesh = masterMesh->a;
if(i + 1 == len)
masterMesh->word = 1;
} else if(word[i] == 'e') {
if(masterMesh->e == NULL)
masterMesh->e = CreateWordMesh();
masterMesh = masterMesh->e;
if(i + 1 == len)
masterMesh->word = 1;
} else if(word[i] == 'i') {
if(masterMesh->i == NULL)
masterMesh->i = CreateWordMesh();
masterMesh = masterMesh->i;
if(i + 1 == len)
masterMesh->word = 1;
} else if(word[i] == 'o') {
if(masterMesh->o == NULL)
masterMesh->o = CreateWordMesh();
masterMesh = masterMesh->o;
if(i + 1 == len)
masterMesh->word = 1;
} else if(word[i] == 'u') {
if(masterMesh->u == NULL)
masterMesh->u = CreateWordMesh();
masterMesh = masterMesh->u;
if(i + 1 == len)
masterMesh->word = 1;
} else if(word[i] == 'f') {
if(masterMesh->f == NULL)
masterMesh->f = CreateWordMesh();
masterMesh = masterMesh->f;
if(i + 1 == len)
masterMesh->word = 1;
} else if(word[i] == 'g') {
if(masterMesh->g == NULL)
masterMesh->g = CreateWordMesh();
masterMesh = masterMesh->g;
if(i + 1 == len)
masterMesh->word = 1;
} else if(word[i] == 'h') {
if(masterMesh->h == NULL)
masterMesh->h = CreateWordMesh();
masterMesh = masterMesh->h;
if(i + 1 == len)
masterMesh->word = 1;
} else if(word[i] == 'c') {
if(masterMesh->c == NULL)
masterMesh->c = CreateWordMesh();
masterMesh = masterMesh->c;
if(i + 1 == len)
masterMesh->word = 1;
} else if(word[i] == 'j') {
if(masterMesh->j == NULL)
masterMesh->j = CreateWordMesh();
masterMesh = masterMesh->j;
if(i + 1 == len)
masterMesh->word = 1;
} else if(word[i] == 'k') {
if(masterMesh->k == NULL)
masterMesh->k = CreateWordMesh();
masterMesh = masterMesh->k;
if(i + 1 == len)
masterMesh->word = 1;
} else if(word[i] == 'l') {
if(masterMesh->l == NULL)
masterMesh->l = CreateWordMesh();
masterMesh = masterMesh->l;
if(i + 1 == len)
masterMesh->word = 1;
} else if(word[i] == 'm') {
if(masterMesh->m == NULL)
masterMesh->m = CreateWordMesh();
masterMesh = masterMesh->m;
if(i + 1 == len)
masterMesh->word = 1;
} else if(word[i] == 'n') {
if(masterMesh->n == NULL)
masterMesh->n = CreateWordMesh();
masterMesh = masterMesh->n;
if(i + 1 == len)
masterMesh->word = 1;
} else if(word[i] == 'd') {
if(masterMesh->d == NULL)
masterMesh->d = CreateWordMesh();
masterMesh = masterMesh->d;
if(i + 1 == len)
masterMesh->word = 1;
} else if(word[i] == 'p') {
if(masterMesh->p == NULL)
masterMesh->p = CreateWordMesh();
masterMesh = masterMesh->p;
if(i + 1 == len)
masterMesh->word = 1;
} else if(word[i] == 'r') {
if(masterMesh->r == NULL)
masterMesh->r = CreateWordMesh();
masterMesh = masterMesh->r;
if(i + 1 == len)
masterMesh->word = 1;
} else if(word[i] == 's') {
if(masterMesh->s == NULL)
masterMesh->s = CreateWordMesh();
masterMesh = masterMesh->s;
if(i + 1 == len)
masterMesh->word = 1;
} else if(word[i] == 't') {
if(masterMesh->t == NULL)
masterMesh->t = CreateWordMesh();
masterMesh = masterMesh->t;
if(i + 1 == len)
masterMesh->word = 1;
} else if(word[i] == 'b') {
if(masterMesh->b == NULL)
masterMesh->b = CreateWordMesh();
masterMesh = masterMesh->b;
if(i + 1 == len)
masterMesh->word = 1;
} else if(word[i] == 'v') {
if(masterMesh->v == NULL)
masterMesh->v = CreateWordMesh();
masterMesh = masterMesh->v;
if(i + 1 == len)
masterMesh->word = 1;
} else if(word[i] == 'y') {
if(masterMesh->y == NULL)
masterMesh->y = CreateWordMesh();
masterMesh = masterMesh->y;
if(i + 1 == len)
masterMesh->word = 1;
} else if(word[i] == 'z') {
if(masterMesh->z == NULL)
masterMesh->z = CreateWordMesh();
masterMesh = masterMesh->z;
if(i + 1 == len)
masterMesh->word = 1;
} else if(word[i] == 'x') {
if(masterMesh->x == NULL)
masterMesh->x = CreateWordMesh();
masterMesh = masterMesh->x;
if(i + 1 == len)
masterMesh->word = 1;
} else if(word[i] == 'w') {
if(masterMesh->w == NULL)
masterMesh->w = CreateWordMesh();
masterMesh = masterMesh->w;
if(i + 1 == len)
masterMesh->word = 1;
} else if(word[i] == 'q') {
if(masterMesh->q == NULL)
masterMesh->q = CreateWordMesh();
masterMesh = masterMesh->q;
if(i + 1 == len)
masterMesh->word = 1;
} else if(word[i] == '') {
if(masterMesh->ch == NULL)
masterMesh->ch = CreateWordMesh();
masterMesh = masterMesh->ch;
if(i + 1 == len)
masterMesh->word = 1;
} else if(word[i] == '') {
if(masterMesh->gh == NULL)
masterMesh->gh = CreateWordMesh();
masterMesh = masterMesh->gh;
if(i + 1 == len)
masterMesh->word = 1;
} else if(word[i] == '') {
if(masterMesh->ih == NULL)
masterMesh->ih = CreateWordMesh();
masterMesh = masterMesh->ih;
if(i + 1 == len)
masterMesh->word = 1;
} else if(word[i] == '') {
if(masterMesh->oh == NULL)
masterMesh->oh = CreateWordMesh();
masterMesh = masterMesh->oh;
if(i + 1 == len)
masterMesh->word = 1;
} else if(word[i] == '') {
if(masterMesh->sh == NULL)
masterMesh->sh = CreateWordMesh();
masterMesh = masterMesh->sh;
if(i + 1 == len)
masterMesh->word = 1;
} else if(word[i] == '') {
if(masterMesh->uh == NULL)
masterMesh->uh = CreateWordMesh();
masterMesh = masterMesh->uh;
if(i + 1 == len)
masterMesh->word = 1;
}
}
}

int SearchWord(WordMesh *master, char* word) {
int len = strlen(word);
int i;
for(i = 0; i < len; i++) {
if(word[i] == 'a') {
if(master->a != NULL)
master = master->a;
else return -1;
} else if(word[i] == 'e') {
if(master->e != NULL)
master = master->e;
else return -1;
} else if(word[i] == 'i') {
if(master->i != NULL)
master = master->i;
else return -1;
} else if(word[i] == 'o') {
if(master->o != NULL)
master = master->o;
else return -1;
} else if(word[i] == 'u') {
if(master->u != NULL)
master = master->u;
else return -1;
} else if(word[i] == 'f') {
if(master->f != NULL)
master = master->f;
else return -1;
} else if(word[i] == 'g') {
if(master->g != NULL)
master = master->g;
else return -1;
} else if(word[i] == 'h') {
if(master->h != NULL)
master = master->h;
else return -1;
} else if(word[i] == 'c') {
if(master->c != NULL)
master = master->c;
else return -1;
} else if(word[i] == 'j') {
if(master->j != NULL)
master = master->j;
else return -1;
} else if(word[i] == 'k') {
if(master->k != NULL)
master = master->k;
else return -1;
} else if(word[i] == 'l') {
if(master->l != NULL)
master = master->l;
else return -1;
} else if(word[i] == 'm') {
if(master->m != NULL)
master = master->m;
else return -1;
} else if(word[i] == 'n') {
if(master->n != NULL)
master = master->n;
else return -1;
} else if(word[i] == 'd') {
if(master->d != NULL)
master = master->d;
else return -1;
} else if(word[i] == 'p') {
if(master->p != NULL)
master = master->p;
else return -1;
} else if(word[i] == 'r') {
if(master->r != NULL)
master = master->r;
else return -1;
} else if(word[i] == 's') {
if(master->s != NULL)
master = master->s;
else return -1;
} else if(word[i] == 't') {
if(master->t != NULL)
master = master->t;
else return -1;
} else if(word[i] == 'b') {
if(master->b != NULL)
master = master->b;
else return -1;
} else if(word[i] == 'v') {
if(master->v != NULL)
master = master->v;
else return -1;
} else if(word[i] == 'y') {
if(master->y != NULL)
master = master->y;
else return -1;
} else if(word[i] == 'z') {
if(master->z != NULL)
master = master->z;
else return -1;
} else if(word[i] == 'x') {
if(master->x != NULL)
master = master->x;
else return -1;
} else if(word[i] == 'w') {
if(master->w != NULL)
master = master->w;
else return -1;
} else if(word[i] == 'q') {
if(master->q != NULL)
master = master->q;
else return -1;
} else if(word[i] == '') {
if(master->ch != NULL)
master = master->ch;
else return -1;
} else if(word[i] == '') {
if(master->gh != NULL)
master = master->gh;
else return -1;
} else if(word[i] == '') {
if(master->ih != NULL)
master = master->ih;
else return -1;
} else if(word[i] == '') {
if(master->oh != NULL)
master = master->oh;
else return -1;
} else if(word[i] == '') {
if(master->sh != NULL)
master = master->sh;
else return -1;
} else if(word[i] == '') {
if(master->uh != NULL)
master = master->uh;
else return -1;
} else {
return -1;
}
}
return master->word;
}

int IsPathContain(WalkBundle* bundle, int index) {
int i;
for(i = 0; i < bundle->pathLen; i++) {
if(*(bundle->path + i) == index) return i;
}
return -1;
}

char GetCell(PuzzleMatrix *puzzle, int x, int y) {
if(puzzle != NULL) {
if(puzzle->m >= x && puzzle->n >= y) {
return *(puzzle->matrix + (puzzle->n * x) + y);
} else {
return;
}
} else {
return;
}
}

int GetNextCell(PuzzleMatrix* puzzle, int direction, int x, int y) {
if(puzzle != NULL) {
if(puzzle->m >= x && puzzle->n >= y) {
switch(direction) {
case 1:
if(y + 1 == puzzle->n) {
return -1;
} else {
return (puzzle->n * x) + y + 1;
}
break;
case 2:
if(y + 1 == puzzle->n || x == 0) {
return -1;
} else {
return (puzzle->n * (x - 1)) + y + 1;
}
break;
case 3:
if(x == 0) {
return -1;
} else {
return (puzzle->n * (x - 1)) + y;
}
break;
case 4:
if(x == 0 || y == 0) {
return -1;
} else {
return (puzzle->n * (x - 1)) + y - 1;
}
break;
case 5:
if(y == 0) {
return -1;
} else {
return (puzzle->n * x) + y - 1;
}
break;
case 6:
if(x + 1 == puzzle->m || y == 0) {
return -1;
} else {
return (puzzle->n * (x + 1)) + y - 1;
}
break;
case 7:
if(x + 1 == puzzle->m) {
return -1;
} else {
return (puzzle->n * (x + 1)) + y;
}
break;
case 8:
if(x + 1 == puzzle->m || y + 1 == puzzle->n) {
return -1;
} else {
return (puzzle->n * (x + 1)) + y + 1;
}
break;
}

return *(puzzle->matrix + (puzzle->n * x) + y);
} else {
return;
}
} else {
return;
}
}

int GetX(int index) {
return (index - (index % N)) / M;
}

int GetY(int index) {
return index % N;
}

int GetIndex(int x, int y) {
return (x * N) + y;
}

int** CalculateAllNext(PuzzleMatrix* puzzle) {
int i, j;
int **allNext = (int**) malloc(sizeof(int*) * MATRIX_SIZE);
for(i = 0; i < MATRIX_SIZE; i++) {
int *next = (int*) malloc(sizeof(int) * 8);
for(j = 0; j < 8; j++) {
next[j] = GetNextCell(puzzle, j + 1, GetX(i), GetY(i));
}
allNext[i] = next;
}
return allNext;
}

int GetNextCellAuto(int** allNext, int current, int direction) {
int *dir = allNext[current];
return dir[direction - 1];
}

void AddStringList(char** list, int* iterator, char* holder, int len) {
char *s = (char*) malloc(sizeof(char) * (len + 1));
int i;
for(i = 0; i < len; i++) {
s[i] = holder[i];
}
list[*iterator] = s;
(*iterator)++;

}

char* CreateString(char* holder, int len) {
char *s = (char*) malloc(sizeof(char) * (len + 1));
int i;
for(i = 0; i < len; i++) {
s[i] = holder[i];
}
return s;
}

WalkBundle* CreateWalkBundle(int count) {
WalkBundle *bundle = (WalkBundle*) malloc(sizeof(WalkBundle) * count);
int i;
for(i = 0; i < count; i++) {
bundle[i].holder = (char*) malloc(sizeof(char) * MATRIX_SIZE);
bundle[i].path = (int*) malloc(sizeof(int) * MATRIX_SIZE);
bundle[i].currentCell = 0;
bundle[i].pathLen = 0;
}
return bundle;
}

void WalkMatrix(int direction, PuzzleMatrix* puzzle, char** list, int* i, WalkBundle* bundle, int** allNext, WordMesh* master) {
int nextPos = GetNextCellAuto(allNext, bundle->currentCell, direction);

if(nextPos != -1) {
if(IsPathContain(bundle, nextPos) == -1) {
int myCurrentCell = bundle->currentCell;
bundle->currentCell = nextPos;
bundle->path[bundle->pathLen] = nextPos;
bundle->holder[bundle->pathLen] = puzzle->matrix[nextPos];
bundle->pathLen++;

char *s = CreateString(bundle->holder, bundle->pathLen);
int res = SearchWord(master, s);
if(res == 0 || res == 1) {
if(res == 1)
AddStringList(list, i, bundle->holder, bundle->pathLen);

WalkMatrix(1, puzzle, list, i, bundle, allNext, master); 
WalkMatrix(2, puzzle, list, i, bundle, allNext, master); 
WalkMatrix(3, puzzle, list, i, bundle, allNext, master); 
WalkMatrix(4, puzzle, list, i, bundle, allNext, master); 
WalkMatrix(5, puzzle, list, i, bundle, allNext, master); 
WalkMatrix(6, puzzle, list, i, bundle, allNext, master); 
WalkMatrix(7, puzzle, list, i, bundle, allNext, master); 
WalkMatrix(8, puzzle, list, i, bundle, allNext, master); 
}

bundle->currentCell = myCurrentCell;
bundle->pathLen = bundle->pathLen - 1;
}
}

}

char** GetStrings(PuzzleMatrix* puzzle, int* count, WordMesh* master, int startX, int startY) {
char **stringList = (char**) malloc(sizeof(char*) * 100);
int **allNext = CalculateAllNext(puzzle);
int iterator = 0, i;
WalkBundle *bundle = CreateWalkBundle(8);

for(i = 0; i < 8; i++) {
bundle[i].currentCell = GetIndex(startX, startY);
bundle[i].path[0] = GetIndex(startX, startY);
bundle[i].holder[0] = puzzle->matrix[GetIndex(startX, startY)];
bundle[i].pathLen = 1;
}

AddStringList(stringList, &iterator, bundle->holder, bundle->pathLen);

WalkMatrix(1, puzzle, stringList, &iterator, &bundle[0], allNext, master); 
WalkMatrix(2, puzzle, stringList, &iterator, &bundle[1], allNext, master); 
WalkMatrix(3, puzzle, stringList, &iterator, &bundle[2], allNext, master); 
WalkMatrix(4, puzzle, stringList, &iterator, &bundle[3], allNext, master); 
WalkMatrix(5, puzzle, stringList, &iterator, &bundle[4], allNext, master); 
WalkMatrix(6, puzzle, stringList, &iterator, &bundle[5], allNext, master); 
WalkMatrix(7, puzzle, stringList, &iterator, &bundle[6], allNext, master); 
WalkMatrix(8, puzzle, stringList, &iterator, &bundle[7], allNext, master); 

for(i = 0; i < 8; i++) {
bundle[i].pathLen--;
}

*count = iterator;
return stringList;
}

PuzzleMatrix* ReadPuzzle() {
char *puzzleMatrix = (char*) malloc(sizeof(char) * 16);
int i;

for(i = 0; i < 4; i++) {
if(i == 3) {
scanf("%c %c %c %c", (puzzleMatrix + (i * 4) + 0), (puzzleMatrix + (i * 4) + 1), (puzzleMatrix + (i * 4) + 2), (puzzleMatrix + (i * 4) + 3));
} else {
scanf("%c %c %c %c\n", (puzzleMatrix + (i * 4) + 0), (puzzleMatrix + (i * 4) + 1), (puzzleMatrix + (i * 4) + 2), (puzzleMatrix + (i * 4) + 3));
}
}


PuzzleMatrix *puzzle = CreatePuzzleMatrix(puzzleMatrix, 4, 4);
return puzzle;
}

PuzzleMatrix** DuplicatePuzzle(PuzzleMatrix* puzzle, int count) {
int i;
PuzzleMatrix **matrix = (PuzzleMatrix**) malloc(sizeof(PuzzleMatrix*) * count);
PuzzleMatrix *pMatrix;
char *temp;
for(i = 0; i < count; i++) {
pMatrix = (PuzzleMatrix*) malloc(sizeof(PuzzleMatrix));
pMatrix->m = puzzle->m;
pMatrix->n = puzzle->n;
temp = (char*) malloc(sizeof(char) * strlen(puzzle->matrix));
strcpy(temp, puzzle->matrix);
pMatrix->matrix = temp;
matrix[i] = pMatrix;
}
return matrix;
}

void UniqueWords(char** words, int count) {
int i, j, k, l = 0;
char **unique = (char**) malloc(sizeof(char*) * count);
char *t = (char*) malloc(sizeof(char) * BUFFER_LEN);
for(i = 0; i < count; i++) {
k = 0;
for(j = 0; j < i; j++) {
if(strcmp(words[j], words[i]) == 0) {
k = 1;
}

}
if(k == 0) {
unique[l] = (char*) malloc(sizeof(char) * BUFFER_LEN);
strcpy(unique[l], words[i]);
l++;
}
}

l--;
for (i = 1; i < l; i++) {
for (j = 1; j < l; j++) {
if (strcmp(unique[j - 1], unique[j]) > 0) {
strcpy(t, unique[j - 1]);
strcpy(unique[j - 1], unique[j]);
strcpy(unique[j], t);
}
}
}

for(i = 0; i < l; i++) {
printf("> %s\n", unique[i]);
}
}

void SeekLine(FILE* file, int lineNum) {
if(file != NULL && lineNum >= 0) {
int i;
for(i = 0; i < lineNum;)
if(fgetc(file) == '\n')
i++;
}
}

void ReadLineBlock(char* sourcePath, int lineCount, char** words) {
int myRank = omp_get_thread_num();
int threadCount = omp_get_num_threads();
int myLineCount = lineCount / threadCount;
int myi = myRank * myLineCount;
char string[BUFFER_LEN];
char *word;
int j;

words += myi;

FILE* source = fopen(sourcePath, "r");

SeekLine(source, myi);

if(myRank == threadCount - 1)
myLineCount += lineCount - (threadCount * myLineCount);


for(j = 0; j < myLineCount; j++) {
fgets(string, BUFFER_LEN, source);
string[strlen(string) - 1] = '\0';
word = (char*) malloc(sizeof(char) * (strlen(string) + 1));
strcpy(word, string);
words[j] = word;
}

fclose(source);
}

int ReadWords(char* filePath, char** words, WordMesh* master) {
if(filePath != NULL && words != NULL) {
char c;
int lineCount = 0;

FILE *file = NULL;
file = fopen(filePath, "r");

if(file != NULL) {
while((c = fgetc(file)) != EOF)
if(c == '\n')
lineCount++;

fclose(file);

#pragma omp parallel num_threads(1)
ReadLineBlock(filePath, lineCount, words);

int i;
for(i = 0; i < lineCount; i++) {
PushWord(master, words[i]);
}
}

return lineCount;
} else {
return 0;
}
}

void FindWords(PuzzleMatrix* puzzle, WordMesh* master, char** allWords, int* iterator) {
int myRank = omp_get_thread_num();
int threadNum = omp_get_num_threads();
int i, stringsCount = 0;
if(myRank >= MATRIX_SIZE) return;
int myWork = MATRIX_SIZE / threadNum;

char **strings = GetStrings(puzzle, &stringsCount, master, GetX(myRank), GetY(myRank));

for(i = 0; i < stringsCount; i++) {
if(strlen(strings[i]) > 2) {
#pragma omp critical
{
allWords[*iterator] = strings[i];
(*iterator)++;
}

}	
}
}

int main(int argc, char* argv[]) {
char *words[WORDS_LEN];
clock_t begin, end;
double runtime;
int iterator = 0;
char **allWords = (char**) malloc(sizeof(char*) * 200);
char **unique;
int i, j, k;
setlocale(LC_ALL, "Turkish");
PuzzleMatrix *puzzle = ReadPuzzle();

begin = clock();

WordMesh *master = CreateWordMesh();
int count = ReadWords("cleanwords.txt", words, master);
printf("\n");

#pragma omp parallel num_threads(MATRIX_SIZE)
FindWords(puzzle, master, allWords, &iterator);

system("CLS");
UniqueWords(allWords, iterator);

end = clock();
runtime = (double) (end - begin) / CLOCKS_PER_SEC;
printf("\nRuntime duration = %.3lf\n\n", runtime);

for(i = 0; i < M; i++) {
for(j = 0; j < N; j++) {
printf("%c ", GetCell(puzzle, i, j));
}
printf("\n");
}

getch();
return 0;
}
