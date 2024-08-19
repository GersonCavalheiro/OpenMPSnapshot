#include<stdio.h>
#include<omp.h>
#include<math.h>
#include "gdal.h"
#include "r_netd.h"
void usage()
{
printf( "-----------------------------------------\n");
printf( "--Modis Processing chain--Serial code----\n");
printf( "-----------------------------------------\n");
printf( "./r_netd inAlbedo inDEM inE31 inE32 inLST\n");
printf( "\toutRNETD\n");
printf( "\tdoy\n");
printf( "-----------------------------------------\n");
printf( "outETPOT\tPotential ET output [mm/d]\n");
printf( "doy\t\tDay of Year [-]\n");
return;
}
int main( int argc, char *argv[] )
{
if( argc < 7 ) {
usage();
return 1;
}
int 	row, col;
double 	geomx[6]={0.0};
char *inB1 	= argv[1]; 
char *inB2 	= argv[2]; 
char *inB3	= argv[3]; 
char *inB4	= argv[4]; 
char *inB5	= argv[5]; 
char *rnetdF 	= argv[6];
float doy	= atof( argv[7] );
GDALAllRegister();
GDALDatasetH hD1 = GDALOpen(inB1,GA_ReadOnly);
GDALDatasetH hD2 = GDALOpen(inB2,GA_ReadOnly);
GDALDatasetH hD3 = GDALOpen(inB3,GA_ReadOnly);
GDALDatasetH hD4 = GDALOpen(inB4,GA_ReadOnly);
GDALDatasetH hD5 = GDALOpen(inB5,GA_ReadOnly);
if(hD1==NULL||hD2==NULL||hD3==NULL||hD4==NULL||hD5==NULL){
printf("One or more input files ");
printf("could not be loaded\n");
exit(1);
}
if(GDALGetGeoTransform(hD1,geomx)==CE_None){
} else {
printf("ERROR: Projection acquisition problem from Band1\n");
exit(1);
}
GDALDriverH hDr2 = GDALGetDatasetDriver(hD2);
char **options = NULL;
options = CSLSetNameValue( options, "TILED", "YES" );
options = CSLSetNameValue( options, "COMPRESS", "DEFLATE" );
options = CSLSetNameValue( options, "PREDICTOR", "2" );
GDALDatasetH hDOut = GDALCreateCopy( hDr2, rnetdF,hD2,FALSE,options,NULL,NULL);
GDALRasterBandH hBOut = GDALGetRasterBand(hDOut,1);
GDALRasterBandH hB1 = GDALGetRasterBand(hD1,1);
GDALRasterBandH hB2 = GDALGetRasterBand(hD2,1);
GDALRasterBandH hB3 = GDALGetRasterBand(hD3,1);
GDALRasterBandH hB4 = GDALGetRasterBand(hD4,1);
GDALRasterBandH hB5 = GDALGetRasterBand(hD5,1);
int nX = GDALGetRasterBandXSize(hB1);
int nY = GDALGetRasterBandYSize(hB1);
float *mat1 = (float *) malloc(sizeof(float)*nX);
float *mat2 = (float *) malloc(sizeof(float)*nX);
float *mat3 = (float *) malloc(sizeof(float)*nX);
float *mat4 = (float *) malloc(sizeof(float)*nX);
float *mat5 = (float *) malloc(sizeof(float)*nX);
float *matOut = (float *) malloc(sizeof(float)*nX);
float solar, rnetd, e0;
for(row=0;row<nY;row++){
GDALRasterIO(hB1,GF_Read,0,row,nX,1,mat1,nX,1,GDT_Float32,0,0);
GDALRasterIO(hB2,GF_Read,0,row,nX,1,mat2,nX,1,GDT_Float32,0,0);
GDALRasterIO(hB3,GF_Read,0,row,nX,1,mat3,nX,1,GDT_Float32,0,0);
GDALRasterIO(hB4,GF_Read,0,row,nX,1,mat4,nX,1,GDT_Float32,0,0);
GDALRasterIO(hB5,GF_Read,0,row,nX,1,mat5,nX,1,GDT_Float32,0,0);
#pragma omp parallel for default(none) private(col, solar, rnetd, e0)shared( row, doy, geomx,nX, mat1, mat2, mat3, mat4, mat5, matOut )
for(col=0;col<nX;col++){
if(mat1[col]==-28768||mat5[col]==-28768||mat5[col]==0){
matOut[col] = -28768;
}else {
e0 = 0.5*((mat3[col]*0.002+0.49)+(mat4[col]*0.002+0.49));
solar = solar_day(geomx[3]+geomx[4]*col+geomx[5]*row, doy, mat2[col] );
rnetd = r_net_day( mat1[col]*0.001, solar, mat2[col]);
matOut[col]=rnetd;
}
}
#pragma omp barrier
GDALRasterIO(hBOut,GF_Write,0,row,nX,1,matOut,nX,1,GDT_Float32,0,0);
}
GDALClose(hDOut);
if(mat1 != NULL) free(mat1);
if(mat2 != NULL) free(mat2);
if(mat3 != NULL) free(mat3);
if(mat4 != NULL) free(mat4);
if(mat5 != NULL) free(mat5);
if(matOut != NULL) free(matOut);
GDALClose(hD1);
GDALClose(hD2);
GDALClose(hD3);
GDALClose(hD4);
GDALClose(hD5);
return(EXIT_SUCCESS);
}
