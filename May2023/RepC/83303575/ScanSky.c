#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include "cputils.h"
#include <omp.h>
#define min(x,y)    ((x) < (y)? (x) : (y))
int main (int argc, char* argv[])
{
if (argc < 2) 	{
printf("Uso: %s <imagen_a_procesar>\n", argv[0]);
return(EXIT_SUCCESS);
}
char* image_filename = argv[1];
int rows=-1;
int columns =-1;
int *matrixData=NULL;
int *matrixResult=NULL;
int *matrixResultCopy=NULL;
int numBlocks=-1;
FILE *f = cp_abrir_fichero(image_filename);
if (f==NULL)
{
perror ("Error al abrir fichero.txt");
return -1;
}
int i,j,valor;
fscanf (f, "%d\n", &rows);
fscanf (f, "%d\n", &columns);
rows=rows+2;
columns = columns+2;
matrixData= (int *)malloc( rows*(columns) * sizeof(int) );
if ( (matrixData == NULL)   ) {
perror ("Error reservando memoria");
return -1;
}
for(i=0;i< rows; i++){
for(j=0;j< columns; j++){
matrixData[i*(columns)+j]=-1;
}
}
for(i=1;i<rows-1;i++){
matrixData[i*(columns)+0]=0;
matrixData[i*(columns)+columns-1]=0;
}
for(i=1;i<columns-1;i++){
matrixData[0*(columns)+i]=0;
matrixData[(rows-1)*(columns)+i]=0;
}
for(i=1;i<rows-1;i++){
for(j=1;j<columns-1;j++){
fscanf (f, "%d\n", &matrixData[i*(columns)+j]);
}
}
fclose(f);
#ifdef WRITE
printf("Inicializacion \n");
for(i=0;i<rows;i++){
for(j=0;j<columns;j++){
printf ("%d\t", matrixData[i*(columns)+j]);
}
printf("\n");
}
#endif
double t_ini = cp_Wtime();
int *matrixIndex, contIndex;
matrixResult= (int *)malloc( (rows)*(columns) * sizeof(int) );
matrixResultCopy= (int *)malloc( (rows)*(columns) * sizeof(int) );
matrixIndex= (int *)malloc( (rows)*(columns) * sizeof(int) );
contIndex=0;
if ( (matrixResult == NULL)  || (matrixResultCopy == NULL)  ) {
perror ("Error reservando memoria");
return -1;
}
#pragma omp nowait parallel for shared(matrixIndex,matrixResult) private(i,j) firstprivate(columns, rows,matrixData)
for(i=0;i< rows; i++){
for(j=0;j< columns; j++){
matrixResult[i*(columns)+j]=-1;
if(matrixData[i*(columns)+j]!=0){
matrixResult[i*(columns)+j]=i*(columns)+j;
}
if(matrixData[i*(columns)+j]>0){
#pragma omp atomic write
matrixIndex[contIndex++] = i*(columns)+j;
}
}
}
#ifdef DEBUG
for(i=0;i<contIndex; i++){
printf("%d\n",matrixIndex[i]);
}
#endif
int t=0;
int flagCambio=1;
for(t=0; flagCambio !=0; t++){
flagCambio=0;
#pragma omp parallel for private(i,j) firstprivate(matrixResult,matrixResultCopy)
for(i=0;i<contIndex;i++){
j=matrixIndex[i];
matrixResultCopy[j]=matrixResult[j];
}
#pragma omp parallel for reduction(+:flagCambio) private(i,j) firstprivate(matrixIndex,columns,rows,matrixData,matrixResult,matrixResultCopy)
for(i=0;i<contIndex;i++){
int result,sol;
j=matrixIndex[i];
result=matrixResultCopy[j];
sol=0;
if(matrixData[j-columns] == matrixData[j])
{
result = min (result, matrixResultCopy[j-columns]);
}
if(matrixData[j+columns] == matrixData[j])
{
result = min (result, matrixResultCopy[j+columns]);
}
if(matrixData[j-1] == matrixData[j])
{
result = min (result, matrixResultCopy[j-1]);
}
if(matrixData[j+1] == matrixData[j])
{
result = min (result, matrixResultCopy[j+1]);
}
if(matrixResult[j] == result){ sol=0; }
else { matrixResult[j]=result; sol=1;}
flagCambio= flagCambio+ sol;
}
#ifdef DEBUG
printf("\nResultados iter %d: \n", t);
for(i=0;i<rows;i++){
for(j=0;j<columns;j++){
printf ("%d\t", matrixResult[i*columns+j]);
}
printf("\n");
}
#endif
}
numBlocks=0;
#pragma omp parallel for reduction(+:numBlocks) private(i,j) firstprivate(matrixResult,columns,rows)
for(i=1;i<rows-1;i++){
for(j=1;j<columns-1;j++){
if(matrixResult[i*columns+j] == i*columns+j) numBlocks++;
}
}
double t_fin = cp_Wtime();
double t_total = (double)(t_fin - t_ini);
printf("Result: %d\n", numBlocks);
printf("Time: %lf\n", t_total);
#ifdef WRITE
printf("Resultado: \n");
for(i=0;i<rows;i++){
for(j=0;j<columns;j++){
printf ("%d\t", matrixResult[i*columns+j]);
}
printf("\n");
}
#endif
free(matrixData);
free(matrixResult);
free(matrixResultCopy);
free(matrixIndex);
}
