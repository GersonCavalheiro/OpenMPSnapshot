#include<stdio.h>
#include<omp.h>
#include<math.h>
#include "gdal.h"
#include "pm_eto.h"
void usage()
{
printf( "-----------------------------------------\n");
printf( "--Modis Processing chain--OpenMP code----\n");
printf( "-----------------------------------------\n");
printf( "./pm_eto inLst inDem inRnet\n");
printf( "\toutPm_eto\n");
printf( "-----------------------------------------\n");
printf( "\trh u\n");
return;
}
int main( int argc, char *argv[] )
{
if( argc < 6 ) {
usage();
return 1;
}
char	*inB1 		= argv[1]; 
char	*inB2	 	= argv[2]; 
char	*inB3	 	= argv[3]; 
char	*pm_etoF	= argv[4];
double rh	= atof(argv[5]); 
double u	= atof(argv[6]); 
GDALAllRegister();
GDALDatasetH hD1 = GDALOpen(inB1,GA_ReadOnly);
GDALDatasetH hD2 = GDALOpen(inB2,GA_ReadOnly);
GDALDatasetH hD3 = GDALOpen(inB3,GA_ReadOnly);
if(hD1==NULL||hD2==NULL||hD3==NULL){
printf("One or more input files ");
printf("could not be loaded\n");
exit(1);
}
char **options = NULL;
options = CSLSetNameValue( options, "TILED", "YES" );
options = CSLSetNameValue( options, "COMPRESS", "DEFLATE" );
options = CSLSetNameValue( options, "PREDICTOR", "2" );
GDALDriverH hDr2 = GDALGetDatasetDriver(hD2);
GDALDatasetH hDOut = GDALCreateCopy( hDr2, pm_etoF,hD2,FALSE,options,NULL,NULL);
GDALRasterBandH hBOut = GDALGetRasterBand(hDOut,1);
GDALRasterBandH hB1 = GDALGetRasterBand(hD1,1);
GDALRasterBandH hB2 = GDALGetRasterBand(hD2,1);
GDALRasterBandH hB3 = GDALGetRasterBand(hD3,1);
int nX = GDALGetRasterBandXSize(hB1);
int nY = GDALGetRasterBandYSize(hB1);
int N=nX*nY;
float *mat1 = (float *) malloc(sizeof(float)*N);
float *mat2 = (float *) malloc(sizeof(float)*N);
float *mat3 = (float *) malloc(sizeof(float)*N);
float *matOut = (float *) malloc(sizeof(float)*N);
float pmeto;
int rowcol;
GDALRasterIO(hB1,GF_Read,0,0,nX,nY,mat1,nX,nY,GDT_Float32,0,0);
GDALRasterIO(hB2,GF_Read,0,0,nX,nY,mat2,nX,nY,GDT_Float32,0,0);
GDALRasterIO(hB3,GF_Read,0,0,nX,nY,mat3,nX,nY,GDT_Float32,0,0);
#pragma omp parallel for default(none) private(rowcol, pmeto)shared(N, rh, u, mat1, mat2, mat3, matOut )
for(rowcol=0;rowcol<N;rowcol++){
if(mat1[rowcol]==-28768||mat1[rowcol]*0.02<250.0||mat1[rowcol]*0.02>360.0) matOut[rowcol] = -28768;
else {
pmeto = EToPM( mat1[rowcol]*0.02, mat2[rowcol], u, mat3[rowcol]*0.0864, rh, 0.6);
matOut[rowcol] = pmeto;
}
}
#pragma omp barrier
GDALRasterIO(hBOut,GF_Write,0,0,nX,nY,matOut,nX,nY,GDT_Float32,0,0);
if(mat1 != NULL) free(mat1);
if(mat2 != NULL) free(mat2);
if(mat3 != NULL) free(mat3);
if(matOut != NULL) free(matOut);
GDALClose(hD1);
GDALClose(hD2);
GDALClose(hD3);
GDALClose(hDOut);
return(EXIT_SUCCESS);
}
