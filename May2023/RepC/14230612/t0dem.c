#include <stdio.h>
#include "gdal.h"
#include <omp.h>
int mod11A1a(int pixel) {
return (pixel & 0x03);
}
void usage()
{
printf( "-----------------------------------------\n");
printf( "--Modis Processing chain--Serial code----\n");
printf( "-----------------------------------------\n");
printf( "./t0dem inDEM inLST inLST_QA\n");
printf( "\toutT0DEM\n");
printf( "-----------------------------------------\n");
printf( "inDEM\t\tDigital Elevation Model 250m [m]\n");
printf( "inLST\t\tModis LST day 1Km\n");
printf( "inLST_QA\t\tModis LST day 1Km Quality Assessment\n");
printf( "outT0DEM\tAltitude corrected Temperature output [K]\n");
return;
}
int main( int argc, char *argv[] )
{
if( argc < 4 ) {
usage();
return 1;
}
char	*inB1	= argv[1]; 
char	*inB2	= argv[2]; 
char	*inB3	= argv[3]; 
char	*t0demF	= argv[4];
GDALAllRegister();
GDALDatasetH hD1 = GDALOpen(inB1,GA_ReadOnly);
GDALDatasetH hD2 = GDALOpen(inB2,GA_ReadOnly);
GDALDatasetH hD3 = GDALOpen(inB3,GA_ReadOnly);
if(hD1==NULL||hD2==NULL||hD3==NULL){
printf("One or more input files ");
printf("could not be loaded\n");
exit(1);
}
GDALDriverH hDr1 = GDALGetDatasetDriver(hD1);
char **options = NULL;
options = CSLSetNameValue( options, "TILED", "YES" );
options = CSLSetNameValue( options, "COMPRESS", "DEFLATE" );
options = CSLSetNameValue( options, "PREDICTOR", "2" );
GDALDatasetH hDOut = GDALCreateCopy(hDr1,t0demF,hD1,FALSE,options,NULL,NULL);
GDALRasterBandH hBOut = GDALGetRasterBand(hDOut,1);
GDALRasterBandH hB1 = GDALGetRasterBand(hD1,1);
GDALRasterBandH hB2 = GDALGetRasterBand(hD2,1);
GDALRasterBandH hB3 = GDALGetRasterBand(hD3,1);
int nX = GDALGetRasterBandXSize(hB1);
int nY = GDALGetRasterBandYSize(hB1);
int N=nX*nY;
float *	l1 = (float *) malloc(sizeof(float)*N);
float *	l2 = (float *) malloc(sizeof(float)*N);
int *	l3 = (int *) malloc(sizeof(int)*N);
float *	lOut = (float *) malloc(sizeof(float)*N);
int rowcol,qa;
GDALRasterIO(hB1,GF_Read,0,0,nX,nY,l1,nX,nY,GDT_Float32,0,0);
GDALRasterIO(hB2,GF_Read,0,0,nX,nY,l2,nX,nY,GDT_Float32,0,0);
GDALRasterIO(hB3,GF_Read,0,0,nX,nY,l3,nX,nY,GDT_Int32,0,0);
#pragma omp parallel for default (none) private (rowcol, qa) shared (N, l1, l2, l3, lOut)	
for(rowcol=0;rowcol<N;rowcol++){
qa = mod11A1a(l3[rowcol]);
if( qa == 0 || qa == 1 ) lOut[rowcol] = (l2[rowcol]*0.02)+0.00627*l1[rowcol];
else lOut[rowcol] = -28768;
}
#pragma omp barrier
GDALRasterIO(hBOut,GF_Write,0,0,nX,nY,lOut,nX,nY,GDT_Float32,0,0);
if( l1 != NULL ) free( l1 );
if( l2 != NULL ) free( l2 );
if( l3 != NULL ) free( l3 );
GDALClose(hD1);
GDALClose(hD2);
GDALClose(hD3);
GDALClose(hDOut);
return(EXIT_SUCCESS);
}
