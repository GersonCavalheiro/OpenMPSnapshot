#include<stdio.h>
#include<stdlib.h>
#include "gdal.h"
#include<omp.h>
#include "fourier.h"
#include "fillin.h"
#include "movavg.h"
#define OBSERVATION_MAX 46
#define MAXFILES 5000
#define NODATA -32768
#define TBC 10000
#define BBC -10000
#define RC 538
void usage()
{
printf( "-----------------------------------------\n");
printf( "--Modis Processing chain--OpenMP code----\n");
printf( "-----------------------------------------\n");
printf( "./fourier in[in,in,in...]\n");
printf( "\tout[out,out,out...]\n");
printf( "-----------------------------------------\n");
return;
}
int main( int argc, char *argv[] )
{
if( argc < 4 ){
usage();
return 1;
}
if((argc-1)%2!=0){
printf("argv[0]=%s\n",argv[0]);
printf("argc=%i\n",argc);
printf("argcm2=%i\n",argc%2);
printf("input number != output number\n");
exit(1);
} 
char *in,*out;
int i,j, length = (argc-1)/2;
int vegetated_seasons=4;
int imagesperyear=23;
int harmonic_number = vegetated_seasons*length/imagesperyear;
printf("harmonic_number=%i\n",harmonic_number);
double t_obs[MAXFILES+1]; 
double t_sim[MAXFILES+1]; 
double t_fil[MAXFILES+1]; 
double t_avg[MAXFILES+1]; 
GDALAllRegister();
GDALDatasetH hD[MAXFILES+1];
GDALDatasetH hDOut[MAXFILES+1];
GDALRasterBandH hB[MAXFILES+1];
GDALRasterBandH hBOut[MAXFILES+1];
for(i=0;i<length;i++){
in=argv[i+1];
hD[i] = GDALOpen(in,GA_ReadOnly);
if(hD[i]==NULL){
printf("%s could not be loaded\n",in);
exit(1);
}
hB[i] = GDALGetRasterBand(hD[i],1);
}
GDALDriverH hDr = GDALGetDatasetDriver(hD[0]);
char **options = NULL;
options = CSLSetNameValue( options, "TILED", "YES" );
options = CSLSetNameValue( options, "COMPRESS", "DEFLATE" );
options = CSLSetNameValue( options, "PREDICTOR", "2" );
for(i=length+1;i<argc;i++){
j=i-length-1;
out=argv[i];
hDOut[j] = GDALCreateCopy(hDr,out,hD[0],FALSE,options,NULL,NULL);
hBOut[j] = GDALGetRasterBand(hDOut[j],1);
}
int nX = GDALGetRasterBandXSize(hB[1]);
int nY = GDALGetRasterBandYSize(hB[1]);
int N = nX*nY;
float *l[MAXFILES+1];
float *lOut[MAXFILES+1];
int rowcol=N;
for(i=0;i<length;i++){
lOut[i] = (float *) malloc(sizeof(float)*N);
for(rowcol=0;rowcol<N;rowcol++){
lOut[i][rowcol] = 0.0;
}
l[i] = (float *) malloc(sizeof(float)*N);
GDALRasterIO(hB[i],GF_Read,0,0,nX,nY,l[i],nX,nY,GDT_Float32,0,0);
GDALRasterIO(hBOut[i],GF_Read,0,0,nX,nY,lOut[i],nX,nY,GDT_Float32,0,0);
}
int countNODATA=0;
#pragma omp parallel for default(none) shared(l, lOut, length, harmonic_number, N) private (rowcol,t_obs,t_sim,t_fil,t_avg,i,countNODATA)
for(rowcol=0;rowcol<N;rowcol++){
countNODATA=0;
for(i=0;i<length;i++){
t_sim[i]=0.0;
t_obs[i]=l[i][rowcol];
if(t_obs[i]>TBC||t_obs[i]<BBC){
t_obs[i]=NODATA;
countNODATA++;
}
}
if(rowcol==RC){
printf("NOData,");
for(i=0;i<length;i++){
printf("%f,",t_obs[i]);
}
printf("\n");
}
if(countNODATA){
fillin(t_sim,t_obs,length,NODATA);
countNODATA=0;
}
for(i=0;i<length;i++){
if(t_obs[i]==NODATA){
t_fil[i]=t_sim[i];
} else {
t_fil[i]=t_obs[i];
}
}
if(rowcol==RC){
printf("FilledInData,");
for(i=0;i<length;i++){
printf("%f,",t_fil[i]);
}
printf("\n");
}
movavg(t_avg,t_fil,length);
if(rowcol==RC){
printf("MovAvgData,");
for(i=0;i<length;i++){
printf("%f,",t_avg[i]);
}
printf("\n");
}
for(i=0;i<length;i++){
lOut[i][rowcol]=t_avg[i];
}
}
#pragma omp barrier
#pragma omp parallel for default(none) shared(l, lOut, length, harmonic_number, N) private (rowcol,t_obs,t_sim,i)
for(rowcol=0;rowcol<N;rowcol++){
for(i=0;i<length;i++){
t_sim[i]=0.0;
t_obs[i]=lOut[i][rowcol];
}
fourier(t_sim,t_obs,length,harmonic_number);
for(i=0;i<length;i++){
lOut[i][rowcol]=t_sim[i];
}
if(rowcol==RC){
printf("FourierData,");
for(i=0;i<length;i++){
printf("%f,",t_sim[i]);
}
printf("\n");
}
}
#pragma omp barrier
for(i=0;i<length;i++){
GDALRasterIO(hBOut[i],GF_Write,0,0,nX,nY,lOut[i],nX,nY,GDT_Float32,0,0);
if(l[i]!= NULL) free( l[i] );
if(lOut[i]!= NULL) free( lOut[i] );
if(hD[i]!=NULL) GDALClose(hD[i]);
if(hDOut[i]!=NULL) GDALClose(hDOut[i]);
}
return(EXIT_SUCCESS);
}
