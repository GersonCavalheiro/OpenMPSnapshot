#include<stdio.h>
#include "gdal.h"
#include "arrays.h"
#include<omp.h>
int mod11A1a(int pixel) {
return (pixel & 0x03);
}
void usage()
{
printf( "-----------------------------------------\n");
printf( "--Modis Processing chain--OpenMP code----\n");
printf( "-----------------------------------------\n");
printf( "./lst inLST inLST_QA\n");
printf( "\toutLST\n");
printf( "-----------------------------------------\n");
printf( "inLST\t\tModis MOD11A1 LST 1000m\n");
printf( "inLST_QA\t\tModis MOD11A1 LST Reliability\n");
printf( "outLST\tQA corrected LST output [-]\n");
return;
}
int main( int argc, char *argv[] )
{
if( argc < 4 ) {
usage();
return (EXIT_FAILURE);
}
char	*inB2 	= argv[1]; 
char	*inB3 	= argv[2]; 
char	*lstF	= argv[3];
GDALAllRegister();
GDALDatasetH hD2 = GDALOpen(inB2,GA_ReadOnly);
GDALDatasetH hD3 = GDALOpen(inB3,GA_ReadOnly);
if(hD2==NULL||hD3==NULL){
printf("One or more input files ");
printf("could not be loaded\n");
exit(EXIT_FAILURE);
}
GDALDriverH hDr2 = GDALGetDatasetDriver(hD2);
char **options = NULL;
options = CSLSetNameValue( options, "TILED", "YES" );
options = CSLSetNameValue( options, "COMPRESS", "DEFLATE" );
options = CSLSetNameValue( options, "PREDICTOR", "2" );
GDALDatasetH hDOut = GDALCreateCopy(hDr2,lstF,hD2,FALSE,options,NULL,NULL);
GDALRasterBandH hBOut = GDALGetRasterBand(hDOut,1);
GDALRasterBandH hB2 = GDALGetRasterBand(hD2,1);
GDALRasterBandH hB3 = GDALGetRasterBand(hD3,1);
int nX = GDALGetRasterBandXSize(hB2);
int nY = GDALGetRasterBandYSize(hB2);
int N=nX*nY;
float *l2 = af1d(N);
float *l3 = af1d(N);
float *lOut = af1d(N);
int rowcol, qa;
GDALRasterIO(hB2,GF_Read,0,0,nX,nY,l2,nX,nY,GDT_Float32,0,0);
GDALRasterIO(hB3,GF_Read,0,0,nX,nY,l3,nX,nY,GDT_Float32,0,0);
#pragma omp parallel for default(none) private (rowcol, qa) shared (N, l2, l3, lOut)
for(rowcol=0;rowcol<N;rowcol++){
qa=mod11A1a(l3[rowcol]);
if( qa == 0 || qa == 1 ) lOut[rowcol] = l2[rowcol];
else lOut[rowcol] = -28768;
}
#pragma omp barrier
GDALRasterIO(hBOut,GF_Write,0,0,nX,nY,lOut,nX,nY,GDT_Float32,0,0);
if( l2 != NULL ) free( l2 );
if( l3 != NULL ) free( l3 );
GDALClose(hD2);
GDALClose(hD3);
GDALClose(hDOut);
return(EXIT_SUCCESS);
}
