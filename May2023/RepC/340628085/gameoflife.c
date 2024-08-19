#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>
#define calcIndex(width, x, y)  ((y)*(width) + (x))
void writeVTK2(long timestep, const double *data, char prefix[1024], int w, int tw, int th, int offsetX, int offsetY) {
char filename[2048];
int x, y;
float deltax = 1.0;
long nxy = tw * th * sizeof(float);
int threadnum = omp_get_thread_num();
snprintf(filename, sizeof(filename), "%s-%05ld-%03d%s", prefix, timestep, threadnum, ".vti");
FILE *fp = fopen(filename, "w");
fprintf(fp, "<?xml version=\"1.0\"?>\n");
fprintf(fp, "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n");
fprintf(fp, "<ImageData WholeExtent=\"%d %d %d %d %d %d\" Origin=\"0 0 0\" Spacing=\"%le %le %le\">\n", offsetX,
offsetX + tw, offsetY, offsetY + th, 0, 0, deltax, deltax, 0.0);
fprintf(fp, "<CellData Scalars=\"%s\">\n", prefix);
fprintf(fp, "<DataArray type=\"Float32\" Name=\"%s\" format=\"appended\" offset=\"0\"/>\n", prefix);
fprintf(fp, "</CellData>\n");
fprintf(fp, "</ImageData>\n");
fprintf(fp, "<AppendedData encoding=\"raw\">\n");
fprintf(fp, "_");
fwrite((unsigned char *) &nxy, sizeof(long), 1, fp);
for (y = 0; y < th; y++) {
for (x = 0; x < tw; x++) {
float value = data[calcIndex(w, x + offsetX, y + offsetY)];
fwrite((unsigned char *) &value, sizeof(float), 1, fp);
}
}
fprintf(fp, "\n</AppendedData>\n");
fprintf(fp, "</VTKFile>\n");
fclose(fp);
}
void writeVTK2_parallel(long timestep, char prefix[1024], char vti_prefix[1024], int w, int h, int px, int py){
char filename[2048];
snprintf(filename, sizeof(filename), "%s-%05ld%s", prefix, timestep, ".pvti");
FILE *fp = fopen(filename, "w");
fprintf(fp, "<?xml version=\"1.0\"?>\n");
fprintf(fp, "<VTKFile type=\"PImageData\" version=\"0.1\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n");
fprintf(fp, "<PImageData WholeExtent=\"%d %d %d %d %d %d\" Origin=\"0 0 0\" Spacing=\"%le %le %le\">\n", 0,
w, 0, h, 0, 0, 1.0, 1.0, 0.0);
fprintf(fp, "<PCellData Scalars=\"%s\">\n", vti_prefix);
fprintf(fp, "<PDataArray type=\"Float32\" Name=\"%s\" format=\"appended\" offset=\"0\"/>\n", vti_prefix);
fprintf(fp, "</PCellData>\n");
for(int x = 0; x < px; x++){
for(int y = 0; y < py; y++){
int start_x = x * (w/px);
int end_x = start_x + (w/px);
int start_y = y * (h/py);
int end_y = start_y + (w/py);
char file[2048];
snprintf(file, sizeof(file), "%s-%05ld-%03d%s", vti_prefix, timestep, px * y + x, ".vti");
fprintf(fp, "<Piece Extent=\"%d %d %d %d 0 0\" Source=\"%s\"/>\n", start_x, end_x, start_y, end_y, file);
}
}
fprintf(fp, "</PImageData>\n");
fprintf(fp, "</VTKFile>\n");
fclose(fp);
}
void show(double *currentfield, int w, int h) {
printf("\033[H");
int x, y;
for (y = 0; y < h; y++) {
for (x = 0; x < w; x++) printf(currentfield[calcIndex(w, x, y)] ? "\033[07m  \033[m" : "  ");
printf("\033[E");
printf("\n");
}
fflush(stdout);
}
int countLivingsPeriodic(double *currentfield, int x, int y, int w, int h) {
int n = 0;
for (int y1 = y - 1; y1 <= y + 1; y1++) {
for (int x1 = x - 1; x1 <= x + 1; x1++) {
if (currentfield[calcIndex(w, (x1 + w) % w, (y1 + h) % h)]) {
n++;
}
}
}
return n;
}
void evolve(int timestep, double *currentfield, double *newfield, int w, int h, int px, int py, int tw, int th) {
#pragma omp parallel num_threads(px*py) default(none) shared(currentfield, newfield) firstprivate(timestep, px, tw, th,  w, h)
{
int this_thread = omp_get_thread_num();
int x, y;
int tx = this_thread % px;
int ty = this_thread / px;
int offsetX = tx * tw;
int offsetY = ty * th;
for (y = 0; y < th; y++) {
for (x = 0; x < tw; x++) {
int n = countLivingsPeriodic(currentfield, x + offsetX, y + offsetY, w, h);
int index = calcIndex(w, x + offsetX, y + offsetY);
if (currentfield[index]) n--;
newfield[index] = (n == 3 || (n == 2 && currentfield[index]));
}
}
#ifndef performance
writeVTK2(timestep, currentfield, "gol", w, tw, th, offsetX, offsetY);
#endif
}
}
void fillRandom(double *currentField, int w, int h) {
for (int i = 0; i < h * w; i++) {
currentField[i] = (rand() < RAND_MAX / 10) ? 1 : 0; 
}
}
void readUntilNewLine(FILE *pFile) {
int readCharacter;
do {
readCharacter = fgetc(pFile);
} while (readCharacter != '\n' && readCharacter != EOF);
}
void setCellState(double *currentField, int alive, int *index, int *count) {
while((*count) > 0) {
currentField[(*index)] = (double)alive;
(*index)++;
(*count)--;
}
}
void fillFromFile(double *currentField, int w, char *fileName) {
FILE *file = fopen(fileName, "r");
readUntilNewLine(file);
readUntilNewLine(file);
int index = 0;
int count = 0;
int readCharacter = fgetc(file);
while(readCharacter != EOF) {
if(readCharacter >= '0' && readCharacter <= '9') {
count = count * 10 + (readCharacter - '0');
} else if(readCharacter == 'b' || readCharacter == 'o') {
int alive = readCharacter == 'o';
if(count == 0) {
count = 1;
}
setCellState(currentField, alive, &index, &count);
} else if(readCharacter == '$' || readCharacter == '!') {
if((index % w) != 0) {
count = (w - (index % w));
}
setCellState(currentField, 0, &index, &count);
} else if(readCharacter == '#') {
readUntilNewLine(file);
} else if(readCharacter == '\n') {
}
else {
printf("Invalid character read: %d", readCharacter);
exit(1);
}
readCharacter = fgetc(file);
}
}
void filling(double *currentField, int w, int h, char *fileName) {
if (access(fileName, R_OK) == 0) {
fillFromFile(currentField, w, fileName);
} else {
fillRandom(currentField, w, h);
}
}
void game(long timeSteps, int tw, int th, int px, int py) {
int w, h;
w = tw * px;
h = th * py;
double *currentfield = calloc(w * h, sizeof(double));
double *newfield = calloc(w * h, sizeof(double));
filling(currentfield, w, h, "file.rle");
long t;
for (t = 0; t < timeSteps; t++) {
evolve(t, currentfield, newfield, w, h, px, py, tw, th);
#ifndef performance
writeVTK2_parallel(t, "golp", "gol", w, h, px, py);    
printf("%ld timestep\n", t);
#endif
double *temp = currentfield;
currentfield = newfield;
newfield = temp;
}
free(currentfield);
free(newfield);
}
int main(int c, char **v) {
srand(42 * 0x815);
long n = 0;
int tw = 0, th = 0, px = 0, py = 0;
if(c > 1) n = atoi(v[1]);   
if (c > 2) tw = atoi(v[2]); 
if (c > 3) th = atoi(v[3]); 
if (c > 4) px = atoi(v[4]); 
if (c > 5) py = atoi(v[5]); 
if(n <= 0) n = 100;         
if (tw <= 0) tw = 18;       
if (th <= 0) th = 12;       
if (px <= 0) px = 1;        
if (py <= 0) py = 1;        
game(n, tw, th, px, py);
}
