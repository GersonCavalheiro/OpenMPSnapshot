#include<stdio.h>
#include<omp.h>
#include<math.h>
#include "gdal.h"
#include "sam_eta.h"
#include "libcrop.h"
void usage()
{
printf( "-----------------------------------------\n");
printf( "--Modis Processing chain--OpenMP code----\n");
printf( "-----------------------------------------\n");
printf( "./sam_eta inLULC inCropAge inRnet inG0\n");
printf( "\tinLST inAlbedo inNDVI inSUNZA inTime\n
printf( "\tinEmissivity inETPOT\n");
printf( "\toutSam_evapfr outSam_eta\n");
printf( "---Following inputs from meteorological station----\n");
printf( "\twind_speed wind_height\n");
printf( "\ttemperature temperature_height\n");
printf( "\tatmospheric_pressure\n");
printf( "-----------------------------------------\n");
return;
}
int main( int argc, char *argv[] )
{
if( argc < 19 ) {
usage();
return 1;
}
char	*inB1 		= argv[1]; 
char	*inB2	 	= argv[2]; 
char	*inB3 		= argv[3]; 
char	*inB4	 	= argv[4]; 
char	*inB5 		= argv[5]; 
char	*inB6	 	= argv[6]; 
char	*inB7 		= argv[7]; 
char	*inB8	 	= argv[8]; 
char	*inB9 		= argv[9]; 
char	*inB10	 	= argv[10]; 
char	*inB11	 	= argv[11]; 
char	*inB12	 	= argv[12]; 
char	*sam_evapfrF	= argv[13];
char	*sam_etaF	= argv[14];
float	wind_speed	= argv[15];
float	wind_height	= argv[16];
float	temperature	= argv[17];
float	temperature_height= argv[18];
float	atmo_pressure	= argv[19];
float	ndvimin=0.1;
float	ndvimax=0.9;
GDALAllRegister();
GDALDatasetH hD1 = GDALOpen(inB1,GA_ReadOnly);
GDALDatasetH hD2 = GDALOpen(inB2,GA_ReadOnly);
GDALDatasetH hD3 = GDALOpen(inB3,GA_ReadOnly);
GDALDatasetH hD4 = GDALOpen(inB4,GA_ReadOnly);
GDALDatasetH hD5 = GDALOpen(inB5,GA_ReadOnly);
GDALDatasetH hD6 = GDALOpen(inB6,GA_ReadOnly);
GDALDatasetH hD7 = GDALOpen(inB7,GA_ReadOnly);
GDALDatasetH hD8 = GDALOpen(inB8,GA_ReadOnly);
GDALDatasetH hD9 = GDALOpen(inB9,GA_ReadOnly);
GDALDatasetH hD10 = GDALOpen(inB10,GA_ReadOnly);
GDALDatasetH hD11 = GDALOpen(inB11,GA_ReadOnly);
GDALDatasetH hD12 = GDALOpen(inB12,GA_ReadOnly);
if(hD1==NULL||hD2==NULL||hD3==NULL||hD4==NULL
||hD5==NULL||hD6==NULL||hD7==NULL||hD8==NULL
||hD9==NULL||hD10==NULL||hD11==NULL||hD12==NULL){
printf("One or more input files ");
printf("could not be loaded\n");
exit(1);
}
GDALDriverH hDr3 = GDALGetDatasetDriver(hD3);
char **options = NULL;
options = CSLSetNameValue( options, "TILED", "YES" );
options = CSLSetNameValue( options, "COMPRESS", "DEFLATE" );
options = CSLSetNameValue( options, "PREDICTOR", "2" );
GDALDatasetH hDOut0 	= GDALCreateCopy( hDr3, sam_evapfrF,hD3,FALSE,options,NULL,NULL);
GDALRasterBandH hBOut0 	= GDALGetRasterBand(hDOut0,1);
GDALDatasetH hDOut 	= GDALCreateCopy( hDr3, sam_etaF,hD3,FALSE,options,NULL,NULL);
GDALRasterBandH hBOut 	= GDALGetRasterBand(hDOut,1);
GDALRasterBandH hB1 	= GDALGetRasterBand(hD1,1);
GDALRasterBandH hB2 	= GDALGetRasterBand(hD2,1);
GDALRasterBandH hB3 	= GDALGetRasterBand(hD3,1);
GDALRasterBandH hB4	= GDALGetRasterBand(hD4,1);
GDALRasterBandH hB5	= GDALGetRasterBand(hD5,1);
GDALRasterBandH hB6	= GDALGetRasterBand(hD6,1);
GDALRasterBandH hB7	= GDALGetRasterBand(hD7,1);
GDALRasterBandH hB8	= GDALGetRasterBand(hD8,1);
GDALRasterBandH hB9	= GDALGetRasterBand(hD9,1);
GDALRasterBandH hB10	= GDALGetRasterBand(hD10,1);
GDALRasterBandH hB11	= GDALGetRasterBand(hD11,1);
GDALRasterBandH hB12	= GDALGetRasterBand(hD12,1);
int nX = GDALGetRasterBandXSize(hB1);
int nY = GDALGetRasterBandYSize(hB1);
int N = nX*nY;
int rowcol;
float *mat1	= (float *) malloc(sizeof(float)*N);
float *mat2	= (float *) malloc(sizeof(float)*N);
float *mat3	= (float *) malloc(sizeof(float)*N);
float *mat4	= (float *) malloc(sizeof(float)*N);
float *mat5	= (float *) malloc(sizeof(float)*N);
float *mat6	= (float *) malloc(sizeof(float)*N);
float *mat7	= (float *) malloc(sizeof(float)*N);
float *mat8	= (float *) malloc(sizeof(float)*N);
float *mat9	= (float *) malloc(sizeof(float)*N);
float *mat10	= (float *) malloc(sizeof(float)*N);
float *mat11	= (float *) malloc(sizeof(float)*N);
float *mat12	= (float *) malloc(sizeof(float)*N);
float *matOut0	= (float *) malloc(sizeof(float)*N);
float *matOut	= (float *) malloc(sizeof(float)*N);
GDALRasterIO(hB1,GF_Read,0,0,nX,nY,mat1,nX,nY,GDT_Float32,0,0);
GDALRasterIO(hB2,GF_Read,0,0,nX,nY,mat2,nX,nY,GDT_Float32,0,0);
GDALRasterIO(hB3,GF_Read,0,0,nX,nY,mat3,nX,nY,GDT_Float32,0,0);
GDALRasterIO(hB4,GF_Read,0,0,nX,nY,mat4,nX,nY,GDT_Float32,0,0);
GDALRasterIO(hB5,GF_Read,0,0,nX,nY,mat5,nX,nY,GDT_Float32,0,0);
GDALRasterIO(hB6,GF_Read,0,0,nX,nY,mat6,nX,nY,GDT_Float32,0,0);
GDALRasterIO(hB7,GF_Read,0,0,nX,nY,mat7,nX,nY,GDT_Float32,0,0);
GDALRasterIO(hB8,GF_Read,0,0,nX,nY,mat8,nX,nY,GDT_Float32,0,0);
GDALRasterIO(hB9,GF_Read,0,0,nX,nY,mat9,nX,nY,GDT_Float32,0,0);
GDALRasterIO(hB10,GF_Read,0,0,nX,nY,mat10,nX,nY,GDT_Float32,0,0);
GDALRasterIO(hB11,GF_Read,0,0,nX,nY,mat11,nX,nY,GDT_Float32,0,0);
GDALRasterIO(hB12,GF_Read,0,0,nX,nY,mat12,nX,nY,GDT_Float32,0,0);
#pragma omp parallel for default(none) private(rowcol)shared(N,mat1,mat2,mat3,mat4,mat5,mat6,mat7,mat8,mat9,mat10,mat11,mat12,matOut0, matOut )
for(rowcol=0;rowcol<N;rowcol++){
if(mat1[rowcol]==-28768||mat1[rowcol]==0){
matOut[rowcol] = -28768;
matOut0[rowcol] = -28768;
} else {
float lulc_code	= mat1[rowcol];
float crop_age	= mat2[rowcol];
float lst	= mat5[rowcol]*0.02;
float lst_c	= lst-273.15;
float	tsoil 	= Tsoil(lst_c, crop_code);
float	tair	= Tair(lst_c, crop_code);
float 	z0_m 	= z0m(crop_age, crop_code);
float	la_i	= lai(ndvi, crop_code);
float	leaf_area	= leafarea(crop_code);
float	leaf_perimeter	= leafperimeter(crop_code);
float	canopy_height	= z0_m/0.136;
if(canopy_height<1.0) {
float disp = 0.65*canopy_height;
} else {
float disp = 0;
}
float	kg	= 1.0;
float	k 	= 0.4631;
float	c	= 0.5;
float	rn	= mat3[rowcol];
float	g0	= mat4[rowcol];
float	albedo	= mat6[rowcol]*0.001;
float	ndvi	= mat7[rowcol]*0.0001;
float	sunza	= mat8[rowcol]*0.1;
float	time	= mat9[rowcol]*0.1;
float	e0	= mat10[rowcol];
float	dem	= mat11[rowcol];
float	etpot	= mat12[rowcol];
float	h	= sens_h(rn,c,la_i,sunza,k,ndvi,ndvimin,ndvimax,lst_c,e0,wind_speed,wind_height,canopy_height,atmo_pressure,z0_m,disp,tair,temperature_height,leaf_area,leaf_perimeter,tsoil);
if(crop_code!=0){
g0	= g0_sam(rn,c,la_i,sunza,kg,ndvi,ndvimin,ndvimax,crop_code);
}
float	samevapfr 	= sam_evapfr(rn, g0, h, lst);
matOut0[rowcol]		= samevapfr;
float	sameta		= samevapfr * etpot;
matOut[rowcol]		= sameta;
}
}
GDALRasterIO(hBOut,GF_Write,0,0,nX,nY,matOut1,nX,nY,GDT_Float32,0,0);
GDALClose(hDOut);
GDALRasterIO(hBOut0,GF_Write,0,0,nX,nY,matOut0,nX,nY,GDT_Float32,0,0);
GDALClose(hDOut0);
if(mat1 != NULL) free(mat1);
if(mat2 != NULL) free(mat2);
if(matOut0 != NULL) free(matOut0);
if(matOut != NULL) free(matOut);
GDALClose(hD1);
GDALClose(hD2);
return(EXIT_SUCCESS);
}
