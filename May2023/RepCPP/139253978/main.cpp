#ifdef _WIN32
#include <windows.h>
#endif

#include <GL/gl.h>
#include <GL/glut.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <math.h>
#include <omp.h>

#include "Patterns.h"

#define WINDOW_TITLE "Game of Life - Parallel CPU implementation OpenMP 4.0 SIMD"

#define SIZE 1024

#define SCREEN_SIZE 1024

#define POINT_SIZE 1

#define ALIVE 1
#define DEAD 0

#define SIMDLEN 16

#define ITERATIONS 1000

#define INFO_SIZE 100
#define FILENAME_SIZE 30
#define STRING_SIZE 10

typedef bool grid_t;

grid_t grid[SIZE * SIZE];
grid_t tempGrid[SIZE * SIZE];
int aliveCells;
int generation;

const char *patternFileName = NULL;

bool benchmarkEnabled = false;
bool isFileClosed = false;
bool fileFirstOpened = false;
int benchmarkGeneration;
int numOfBenchmarkFiles;
char saveFileName[FILENAME_SIZE] = "benchmarkTestFile";
FILE *saveFile;

bool canRender = false;
bool canShowInfo = true;

clock_t before, after;
double diff;

void initGameVariables();
FILE * openBenchmarkFile(const char *filename);

void initCell(grid_t grid[SIZE * SIZE], int pos);
void assignCell(grid_t newGrid[SIZE * SIZE], int pos, bool state);
void updateCell(grid_t newGrid[SIZE * SIZE], grid_t oldGrid[SIZE * SIZE], int pos);
void linearInitGrid(grid_t grid[SIZE * SIZE]);
void linearFillGrid(grid_t grid[SIZE * SIZE], const char *patternFileName);
void linearRandomFillGrid(grid_t grid[SIZE * SIZE]);
void linearFillGridFromPatternFile(grid_t grid[SIZE * SIZE], const char *filename);
void linearUpdateGrid(grid_t grid[SIZE * SIZE]);
int linearGetAliveNeighbors(grid_t grid[SIZE * SIZE], int pos);

void displayText(GLint x, GLint y, char *text);
void displayInfo(GLint x, GLint y);
void display(void);
void idle(void);
void changeInfoView(unsigned char key, int x, int y);

int main(int argc, char** argv)
{
initGameVariables();

linearInitGrid(grid);
linearFillGrid(grid, NULL);

glutInit(&argc, argv);
glutInitDisplayMode(GLUT_SINGLE);
glutInitWindowSize(fmin(SIZE, SCREEN_SIZE),fmin(SIZE, SCREEN_SIZE));
glutCreateWindow(WINDOW_TITLE);
glClearColor(0,0,0,0);
glClear(GL_COLOR_BUFFER_BIT);
glutDisplayFunc(display);
glutIdleFunc(idle);
glutKeyboardFunc(changeInfoView);
glutMainLoop();

return 0;
}

void initGameVariables()
{
aliveCells = 0;
generation = 0;
benchmarkGeneration = 0;
numOfBenchmarkFiles = 1;
}

FILE * openBenchmarkFile(const char *filename)
{
FILE *file;

if(filename != NULL){
file = fopen(filename, "w");
if(file == NULL){
printf("Error: Could not open file %s.\n", filename);
exit(EXIT_FAILURE);
}
else printf("Time calculation file \"%s\" created.\n", filename);
}
else {
printf("Error: You need to define a name for your save file.\n");
exit(EXIT_FAILURE);
}
return file;
}

#pragma omp declare simd notinbranch
void inline initCell(grid_t grid[SIZE * SIZE], int pos)
{
grid[pos] = DEAD;
}
#pragma omp declare simd notinbranch
void inline assignCell(grid_t newGrid[SIZE * SIZE], int pos, bool state)
{
newGrid[pos] = state;
}

#pragma omp declare simd notinbranch uniform(oldGrid)
void inline updateCell(grid_t newGrid[SIZE * SIZE], grid_t oldGrid[SIZE * SIZE], int pos)
{
int aliveNeighbors = linearGetAliveNeighbors(oldGrid, pos);
bool state = 0;
if(oldGrid[pos]==ALIVE){
if(aliveNeighbors==2 || aliveNeighbors==3){
state = ALIVE;
}
else{
state = DEAD;
}
}
else{
if(aliveNeighbors==3){
state = ALIVE;
}
}
assignCell(newGrid, pos, state);
}

void linearInitGrid(grid_t grid[SIZE * SIZE])
{
int i;

#pragma omp parallel for simd collapse(1)
for(i=0; i<SIZE * SIZE; i++){
initCell(grid, i);
}
}

void linearFillGrid(grid_t grid[SIZE * SIZE], const char *patternFileName)
{
if(patternFileName == NULL){
linearRandomFillGrid(grid);
}
else linearFillGridFromPatternFile(grid, patternFileName);
}

void linearRandomFillGrid(grid_t grid[SIZE * SIZE])
{
int i, randomNumber;

srand(time(NULL));

for(i=0; i<SIZE * SIZE; i++){
randomNumber = rand() % 10 + 1;
if(randomNumber >= 9){
grid[i] = ALIVE;
aliveCells++;
}
}
}

void linearUpdateGrid(grid_t grid[SIZE * SIZE])
{
int i;

linearInitGrid(tempGrid);

before = clock();

#pragma omp parallel for simd collapse(1)
for(i=0; i<SIZE * SIZE; i++){
updateCell(tempGrid, grid, i);
}

memcpy(grid, tempGrid, SIZE * SIZE);

after = clock();

diff = (after - before) * 1000.0 / (CLOCKS_PER_SEC * 1.0);

if(benchmarkEnabled){
if(!fileFirstOpened){
char numOfBenchmarkFilesString[STRING_SIZE];
itoa(numOfBenchmarkFiles, numOfBenchmarkFilesString, 10);
strcat(saveFileName, numOfBenchmarkFilesString);
saveFile = openBenchmarkFile(saveFileName);
memset(saveFileName,0,sizeof(saveFileName));
strcpy(saveFileName, "benchmarkTestFile");
fileFirstOpened = true;
numOfBenchmarkFiles++;
}

if(benchmarkGeneration <= ITERATIONS-1){
fprintf(saveFile,"%f\n",diff);
benchmarkGeneration++;
}

else{
if(!isFileClosed){
fclose(saveFile);
isFileClosed = true;
}
fileFirstOpened = false;
benchmarkGeneration = 0;
benchmarkEnabled = false;
}
}
else{
fileFirstOpened = false;
benchmarkGeneration = 0;
}

generation++;
}

#pragma omp declare simd notinbranch uniform(grid)
int linearGetAliveNeighbors(grid_t grid[SIZE * SIZE], int pos)
{
int aliveNeighbors = 0;
int rowUp, rowDown;
rowUp = pos - SIZE;
rowDown = pos + SIZE;

bool outOfBounds = (pos < SIZE);
outOfBounds |= (pos > (SIZE * (SIZE-1)));
outOfBounds |= (pos % SIZE == 0);
outOfBounds |= (pos % SIZE == SIZE-1);

if(outOfBounds) return 0;

aliveNeighbors += grid[rowUp-1] + grid[rowUp] + grid[rowUp+1];
aliveNeighbors += grid[pos-1] + grid[pos+1];
aliveNeighbors += grid[rowDown-1] + grid[rowDown] + grid[rowDown+1];

return aliveNeighbors;
}

void linearFillGridFromPatternFile(grid_t grid[SIZE * SIZE], const char *filename) {

FILE *patternFile = fopen(filename, "r");
if(patternFile == NULL){
printf("Failed to load file to project file.");
return;
}

char cellChar;
int row = 400;
int column = 350;
while((cellChar = fgetc(patternFile)) != EOF){
if(cellChar == '\n'){
row++;
column = 350;
}
else{
column++;
if(cellChar=='*'){
grid[row * SIZE + column] = ALIVE;
aliveCells++;
}
else grid[row * SIZE + column] = DEAD;
}
}
}


void changeInfoView(unsigned char key, int x, int y)
{
if(key == 'v' || key =='V'){
canShowInfo = !canShowInfo;
}

if(key == 'b' || key == 'B'){
benchmarkEnabled = !benchmarkEnabled;
}
}

void idle(void){

linearUpdateGrid(grid);
canRender = true;

if(canRender){
glutPostRedisplay();
aliveCells = 0;
}
}

void display(void) {

GLint pos, row, column;

glMatrixMode(GL_PROJECTION);

glLoadIdentity();

GLfloat windowWidth = fmin(SIZE, SCREEN_SIZE) * 1.0;
GLfloat windowHeight = fmin(SIZE, SCREEN_SIZE) * 1.0;

gluOrtho2D(0.0, windowWidth, 0.0, windowHeight);
glPointSize(POINT_SIZE);

glBegin(GL_POINTS);
for(pos=0; pos<SIZE * SIZE; pos++){
row = pos % SIZE;
column = pos / SIZE;
if(grid[pos]==ALIVE){
glColor3f(0.0, 0.6, 0.0);
aliveCells++;
}
else glColor3f(0.0, 0.0, 0.0);
glVertex2i(row, column);
}
glEnd();

glColor3f(1.0,1.0,0.0);
displayInfo(16,windowHeight-20);

glFlush();
}

void displayText(GLint x, GLint y, char *text)
{
unsigned int i;

glRasterPos2d(x,y);
for(i=0; i<strlen(text); i++)
{
glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, text[i]);
}
}

void displayInfo(GLint x, GLint y)
{
char timeString[INFO_SIZE];
char generationString[INFO_SIZE];
char aliveCellsString[INFO_SIZE];
char benchmarkEnabledString[INFO_SIZE];
char completePercentageString[INFO_SIZE];
char keyboardInfoString[INFO_SIZE];
char benchmarkInfoString[INFO_SIZE];
char gameSizeInfoString[INFO_SIZE];

char time[INFO_SIZE];
char gen[INFO_SIZE];
char alive[INFO_SIZE];

sprintf(time, "%f", diff);
sprintf(gen, "%d", generation);
sprintf(alive, "%d", aliveCells);

sprintf(timeString, "%s%s ms", "Time: ", time);
sprintf(generationString, "%s%s", "Generation: ", gen);
sprintf(aliveCellsString, "%s%s", "Alive cells: ", alive);

if(canShowInfo) sprintf(keyboardInfoString, "%s", "Press <v> to toggle game's info OFF");
else sprintf(keyboardInfoString, "%s", "Press <v> to toggle game's info ON");

if(benchmarkEnabled){
sprintf(benchmarkInfoString, "Press <b> to turn benchmarking mode OFF");
sprintf(benchmarkEnabledString, "%s", "Benchmarking mode: [ on ]");
sprintf(completePercentageString, "%s%f%s", "Benchmarking completion: [", fmin(100.0,(benchmarkGeneration * 1.0 / ITERATIONS) * 100), "% ]");
}
else{
sprintf(benchmarkInfoString, "Press <b> to turn benchmarking mode ON");
sprintf(benchmarkEnabledString, "%s", "Benchmarking mode: [ off ]");
sprintf(completePercentageString, "%s", "Benchmarking completion: [ -- ]");
}
sprintf(gameSizeInfoString, "Game size: [ %d x %d ]", SIZE, SIZE);

glPushMatrix();

displayText(x,y,keyboardInfoString);

if(canShowInfo)
{
displayText(x,y-30, benchmarkInfoString);
displayText(x,y-60, timeString);
displayText(x,y-90, generationString);
displayText(x,y-120, aliveCellsString);
displayText(x,y-150,benchmarkEnabledString);
displayText(x,y-180,completePercentageString);
displayText(x,y-210, gameSizeInfoString);
}

glPopMatrix();
}
