#include<stdio.h>
#include "gdal.h"
#include<omp.h>
unsigned int L8QA_cloud(unsigned int pixel) {
unsigned int qa;
pixel >>= 4; 	
qa = (unsigned int) (pixel & 0x01);
return qa;
}
unsigned int L8QA_cloud_confidence(unsigned int pixel) {
unsigned int qa;
pixel >>= 5; 	
qa = (unsigned int) (pixel & 0x03);
return qa;
}
unsigned int L8QA_cloud_shadow(unsigned int pixel) {
unsigned int qa;
pixel >>= 7; 	
qa = (unsigned int) (pixel & 0x03);
return qa;
}
unsigned int L8QA_cirrus_confidence(unsigned int pixel) {
unsigned int qa;
pixel >>= 11; 	
qa = (unsigned int) (pixel & 0x03);
return qa;
}
void usage()
{
printf( "-----------------------------------------\n");
printf( "--L8 Processing chain--OpenMP code----\n");
printf( "-----------------------------------------\n");
printf( "./L8 inL8b4 inL8b5 inL8b6 inL8B7 inL8_QA\n");
printf( "\toutL8vi outL8wi outL8LSWI outL8NBR2\n");
printf( "-----------------------------------------\n");
printf( "inL8b4\t\tL8 Band 4 UInt16 (Red)\n");
printf( "inL8b5\t\tL8 Band 5 UInt16 (NIR)\n");
printf( "inL8b6\t\tL8 Band 6 UInt16 (SWIR1)\n");
printf( "inL8b7\t\tL8 Band 7 UInt16 (SWIR2)\n");
printf( "inL8_QA\t\tL8_Qa UInt16\n");
printf( "outL8vi\tCloud removed L8 NDVI output [0-10000]\n");
printf( "outL8wi\tCloud removed L8 NDWI output [0-10000]\n");
printf( "outL8lswi\tCloud removed L8 LSWI output [0-10000]\n");
printf( "outL8nbr2\tCloud removed L8 NBR2 output [0-10000]\n");
return;
}
int main( int argc, char *argv[] )
{
if( argc < 9 ) {
usage();
return 1;
}
char	*inB4	= argv[1]; 
char	*inB5	= argv[2]; 
char	*inB6	= argv[3]; 
char	*inB7	= argv[4]; 
char	*inB8 	= argv[5]; 
char	*L8viF 	= argv[6]; 
char	*L8wiF 	= argv[7]; 
char	*L8lswiF= argv[8]; 
char	*L8nbr2F= argv[9]; 
GDALAllRegister();
GDALDatasetH hD4 = GDALOpen(inB4,GA_ReadOnly);
GDALDatasetH hD5 = GDALOpen(inB5,GA_ReadOnly);
GDALDatasetH hD6 = GDALOpen(inB6,GA_ReadOnly);
GDALDatasetH hD7 = GDALOpen(inB7,GA_ReadOnly);
GDALDatasetH hD8 = GDALOpen(inB8,GA_ReadOnly);
if(hD4==NULL||hD5==NULL||hD6==NULL||hD7==NULL||hD8==NULL){
printf("One or more input files ");
printf("could not be loaded\n");
exit(1);
}
GDALDriverH hDr4 = GDALGetDatasetDriver(hD4);
char **options = NULL;
options = CSLSetNameValue( options, "TILED", "YES" );
options = CSLSetNameValue( options, "COMPRESS", "DEFLATE" );
options = CSLSetNameValue( options, "PREDICTOR", "2" );
GDALDatasetH hDOutVI = GDALCreateCopy(hDr4,L8viF,hD4,FALSE,options,NULL,NULL);
GDALDatasetH hDOutWI = GDALCreateCopy(hDr4,L8wiF,hD4,FALSE,options,NULL,NULL);
GDALDatasetH hDOutLSWI = GDALCreateCopy(hDr4,L8lswiF,hD4,FALSE,options,NULL,NULL);
GDALDatasetH hDOutNBR2 = GDALCreateCopy(hDr4,L8nbr2F,hD4,FALSE,options,NULL,NULL);
GDALRasterBandH hBOutVI = GDALGetRasterBand(hDOutVI,1);
GDALRasterBandH hBOutWI = GDALGetRasterBand(hDOutWI,1);
GDALRasterBandH hBOutLSWI = GDALGetRasterBand(hDOutLSWI,1);
GDALRasterBandH hBOutNBR2 = GDALGetRasterBand(hDOutNBR2,1);
GDALRasterBandH hB4 = GDALGetRasterBand(hD4,1);
GDALRasterBandH hB5 = GDALGetRasterBand(hD5,1);
GDALRasterBandH hB6 = GDALGetRasterBand(hD6,1);
GDALRasterBandH hB7 = GDALGetRasterBand(hD7,1);
GDALRasterBandH hB8 = GDALGetRasterBand(hD8,1);
int nX = GDALGetRasterBandXSize(hB4);
int nY = GDALGetRasterBandYSize(hB4);
int N=nX*nY;
unsigned int *l4 = (unsigned int *) malloc(sizeof(unsigned int)*N);
unsigned int *l5 = (unsigned int *) malloc(sizeof(unsigned int)*N);
unsigned int *l6 = (unsigned int *) malloc(sizeof(unsigned int)*N);
unsigned int *l7 = (unsigned int *) malloc(sizeof(unsigned int)*N);
unsigned int *l8 = (unsigned int *) malloc(sizeof(unsigned int)*N);
unsigned int *lOutVI = (unsigned int *) malloc(sizeof(unsigned int)*N);
unsigned int *lOutWI = (unsigned int *) malloc(sizeof(unsigned int)*N);
unsigned int *lOutLSWI = (unsigned int *) malloc(sizeof(unsigned int)*N);
unsigned int *lOutNBR2 = (unsigned int *) malloc(sizeof(unsigned int)*N);
int rc, qac, qacc, qacs, qaci;
GDALRasterIO(hB4,GF_Read,0,0,nX,nY,l4,nX,nY,GDT_UInt32,0,0);
GDALRasterIO(hB5,GF_Read,0,0,nX,nY,l5,nX,nY,GDT_UInt32,0,0);
GDALRasterIO(hB6,GF_Read,0,0,nX,nY,l6,nX,nY,GDT_UInt32,0,0);
GDALRasterIO(hB7,GF_Read,0,0,nX,nY,l7,nX,nY,GDT_UInt32,0,0);
GDALRasterIO(hB8,GF_Read,0,0,nX,nY,l8,nX,nY,GDT_UInt32,0,0);
int n = omp_get_num_threads();
omp_set_num_threads(n-1);
#pragma omp parallel for default(none) private (rc, qac, qacc, qacs, qaci) shared (N, l4, l5, l6, l7, l8, lOutVI, lOutWI, lOutLSWI, lOutNBR2)
for(rc=0;rc<N;rc++){
qac=L8QA_cloud(l8[rc]);
qacc=L8QA_cloud_confidence(l8[rc]);
qacs=L8QA_cloud_shadow(l8[rc]);
qaci=L8QA_cirrus_confidence(l8[rc]);
if(l8[rc]==1){ 
lOutVI[rc] = 32768;
lOutWI[rc] = 32768;
lOutLSWI[rc] = 32768;
lOutNBR2[rc] = 32768;
}else if(qac == 1 || qacc > 2 || qacs > 2 || qaci > 2){
lOutVI[rc] = 32767; 
lOutWI[rc] = 32767; 
lOutLSWI[rc] = 32767; 
lOutNBR2[rc] = 32767; 
}else{
if((l5[rc]+l4[rc])==0){
lOutVI[rc]=32768;
}else{
lOutVI[rc]=(int)(10000.0*l5[rc]-l4[rc])/(1.0*l5[rc]+l4[rc]);
}
if((l6[rc]+l5[rc])==0){
lOutWI[rc]=32768;
}else{
lOutWI[rc]=(int)(10000.0*l6[rc]-l5[rc])/(1.0*l6[rc]+l5[rc]);
}
if((l6[rc]+l5[rc])==0){
lOutLSWI[rc]=32768;
}else{
lOutLSWI[rc]=(int)(10000.0*l5[rc]-l6[rc])/(1.0*l5[rc]+l6[rc]);
}
if((l6[rc]+l7[rc])==0){
lOutNBR2[rc]=32768;
}else{
lOutNBR2[rc]=(int)(10000.0*l6[rc]-l7[rc])/(1.0*l6[rc]+l7[rc]);
}
}
}
#pragma omp barrier
GDALRasterIO(hBOutVI,GF_Write,0,0,nX,nY,lOutVI,nX,nY,GDT_UInt32,0,0);
GDALRasterIO(hBOutWI,GF_Write,0,0,nX,nY,lOutWI,nX,nY,GDT_UInt32,0,0);
GDALRasterIO(hBOutLSWI,GF_Write,0,0,nX,nY,lOutLSWI,nX,nY,GDT_UInt32,0,0);
GDALRasterIO(hBOutNBR2,GF_Write,0,0,nX,nY,lOutNBR2,nX,nY,GDT_UInt32,0,0);
if( l4 != NULL ) free( l4 );
if( l5 != NULL ) free( l5 );
if( l6 != NULL ) free( l6 );
if( l7 != NULL ) free( l7 );
if( l8 != NULL ) free( l8 );
GDALClose(hD4);
GDALClose(hD5);
GDALClose(hD6);
GDALClose(hD7);
GDALClose(hD8);
GDALClose(hDOutVI);
GDALClose(hDOutWI);
GDALClose(hDOutLSWI);
GDALClose(hDOutNBR2);
return(EXIT_SUCCESS);
}
