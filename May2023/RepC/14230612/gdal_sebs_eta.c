#include<stdio.h>
#include<omp.h>
#include<math.h>
#include "gdal.h"
#include "cpl_conv.h"
#include "sebs_eta.h"
void usage()
{
printf( "-----------------------------------------\n");
printf( "--Modis Processing chain--OpenMP code----\n");
printf( "-----------------------------------------\n");
printf( "./sebs_eta inLst inDem inE031 inE032 inNdvi inAlbedo inSunza\n");
printf( "\toutSebs_evapfr outSebs_eta\n");
printf( "\tz_pbl t_s p_s u_s hr_s alt_ms doy\n");
printf( "-----------------------------------------\n");
return;
}
int main( int argc, char *argv[] )
{
if( argc < 16 ) {
usage();
return 1;
}
char *inB1	= argv[1]; 
char *inB2	= argv[2]; 
char *inB3	= argv[3]; 
char *inB4	= argv[4]; 
char *inB5	= argv[5]; 
char *inB6	= argv[6]; 
char *inB7	= argv[7]; 
char *sebs_evapfrF = argv[8];
char *sebs_etaF	= argv[9];
double z_pbl	= atof(argv[10]); 
double t_s	= atof(argv[11]); 
double p_s	= atof(argv[12]); 
double u_s	= atof(argv[13]); 
double hr_s	= atof(argv[14]); 
double alt_ms	= atof(argv[15]); 
double doy	= atof(argv[16]); 
printf("inB5 = %s\n", inB5);
GDALDatasetH hD1;
GDALDatasetH hD2;
GDALDatasetH hD3;
GDALDatasetH hD4;
GDALDatasetH hD5;
GDALDatasetH hD6;
GDALDatasetH hD7;
GDALAllRegister();
hD1 = GDALOpen(inB1,GA_ReadOnly);
hD2 = GDALOpen(inB2,GA_ReadOnly);
hD3 = GDALOpen(inB3,GA_ReadOnly);
hD4 = GDALOpen(inB4,GA_ReadOnly);
hD5 = GDALOpen(inB5,GA_ReadOnly);
hD6 = GDALOpen(inB6,GA_ReadOnly);
hD7 = GDALOpen(inB7,GA_ReadOnly);
if(hD1==NULL||hD2==NULL||hD3==NULL||hD4==NULL
||hD5==NULL||hD6==NULL||hD7==NULL){
printf("One or more input files ");
printf("could not be loaded\n");
exit(1);
}
double geomx[6]={0.0};
if(GDALGetGeoTransform(hD1,geomx)==CE_None){
} else {
printf("ERROR: Projection acquisition problem from Band1\n");
exit(1);
}
GDALDriverH hDr1;
GDALDriverH hDr2;
GDALDriverH hDr3;
GDALDriverH hDr4;
GDALDriverH hDr5;
GDALDriverH hDr6;
GDALDriverH hDr7;
hDr1 = GDALGetDatasetDriver(hD1);
hDr2 = GDALGetDatasetDriver(hD2);
hDr3 = GDALGetDatasetDriver(hD3);
hDr4 = GDALGetDatasetDriver(hD4);
hDr5 = GDALGetDatasetDriver(hD5);
hDr6 = GDALGetDatasetDriver(hD6);
hDr7 = GDALGetDatasetDriver(hD7);
char **options = NULL;
options = CSLSetNameValue( options, "TILED", "YES" );
options = CSLSetNameValue( options, "COMPRESS", "DEFLATE" );
options = CSLSetNameValue( options, "PREDICTOR", "2" );
GDALDatasetH hDOut0;
hDOut0 = GDALCreateCopy( hDr2, sebs_evapfrF,hD2,FALSE,options,NULL,NULL);
GDALRasterBandH hBOut0;
hBOut0 = GDALGetRasterBand(hDOut0,1);
GDALDatasetH hDOut;
hDOut = GDALCreateCopy( hDr2, sebs_etaF,hD2,FALSE,options,NULL,NULL);
GDALRasterBandH hBOut;
hBOut = GDALGetRasterBand(hDOut,1);
GDALRasterBandH hB1;
GDALRasterBandH hB2;
GDALRasterBandH hB3;
GDALRasterBandH hB4;
GDALRasterBandH hB5;
GDALRasterBandH hB6;
GDALRasterBandH hB7;
hB1 = GDALGetRasterBand(hD1,1);
hB2 = GDALGetRasterBand(hD2,1);
hB3 = GDALGetRasterBand(hD3,1);
hB4 = GDALGetRasterBand(hD4,1);
hB5 = GDALGetRasterBand(hD5,1);
hB6 = GDALGetRasterBand(hD6,1);
hB7 = GDALGetRasterBand(hD7,1);
int nX = GDALGetRasterBandXSize(hB1);
int nY = GDALGetRasterBandYSize(hB1);
int N=nX*nY;
float *mat1 = (float *) CPLMalloc(sizeof(float)*N);
float *mat2 = (float *) CPLMalloc(sizeof(float)*N);
float *mat3 = (float *) CPLMalloc(sizeof(float)*N);
float *mat4 = (float *) CPLMalloc(sizeof(float)*N);
float *mat5 = (float *) CPLMalloc(sizeof(float)*N);
float *mat6 = (float *) CPLMalloc(sizeof(float)*N);
float *mat7 = (float *) CPLMalloc(sizeof(float)*N);
float *matOut0 = (float *) CPLMalloc(sizeof(float)*N);
float *matOut = (float *) CPLMalloc(sizeof(float)*N);
float *latitude = (float *) CPLMalloc(sizeof(float)*N);
float *ndvi_median = (float *) CPLMalloc(sizeof(float)*N);
float sebsevapfr, sebseta, kin, kin24, e0;
int row=nY, col=nX, rowcol, n=0;
double ndvi_min=10000.0;
double ndvi_max=-10000.0;
double ndvimedian=0.0;
int i,j,temp;
double esat_avg, e_sat_sum=0.0, e_act;
GDALRasterIO(hB1,GF_Read,0,0,nX,nY,mat1,nX,nY,GDT_Float32,0,0);
GDALRasterIO(hB2,GF_Read,0,0,nX,nY,mat2,nX,nY,GDT_Float32,0,0);
GDALRasterIO(hB3,GF_Read,0,0,nX,nY,mat3,nX,nY,GDT_Float32,0,0);
GDALRasterIO(hB4,GF_Read,0,0,nX,nY,mat4,nX,nY,GDT_Float32,0,0);
GDALRasterIO(hB5,GF_Read,0,0,nX,nY,mat5,nX,nY,GDT_Float32,0,0);
GDALRasterIO(hB6,GF_Read,0,0,nX,nY,mat6,nX,nY,GDT_Float32,0,0);
GDALRasterIO(hB7,GF_Read,0,0,nX,nY,mat7,nX,nY,GDT_Float32,0,0);
GDALRasterIO(hB5,GF_Read,0,0,nX,nY,ndvi_median,nX,nY,GDT_Float32,0,0);
printf("Latitude\t");
for(row=0;row<nY;row++){
#pragma omp parallel for default(none) private(col) shared( row, nX, nY, geomx, latitude)
for(col=0;col<nX;col++){
latitude[row*nY+col] = geomx[3]+geomx[4]*col+geomx[5]*row;
}
#pragma omp barrier
}
printf("\n");
#pragma omp parallel for default(none) private(i)shared(N, mat1, mat5, ndvi_median)
for(i=0;i<N;i++){
if(mat1[i]==-28768||mat1[i]*0.02<=250.0||mat1[i]*0.02>365.0
||mat5[i]<=-10000.0||mat5[i]>=10000.0||mat5[i]==0.0){ 
ndvi_median[i]=99999.99;
} else if(mat5[i]<0.0){
ndvi_median[i]=99999.99;
}
}
#pragma omp barrier
#pragma omp parallel for default(none) private(rowcol)shared(N, ndvi_min, ndvi_max, ndvi_median )
for(rowcol=0;rowcol<N;rowcol++){
if(ndvi_median[rowcol]<=99999.99){
if(ndvi_median[rowcol]*0.0001<ndvi_min) ndvi_min=ndvi_median[rowcol]*0.0001;
if(ndvi_median[rowcol]*0.0001>ndvi_max) ndvi_max=ndvi_median[rowcol]*0.0001;
}
}
#pragma omp barrier
printf("ndvimin = %f [-]\tndvimax = %f [-]\n",ndvi_min, ndvi_max);
n=0;
#pragma omp parallel for default(none) private(rowcol)shared(N, n, e_sat_sum, mat1, mat2, ndvi_median )
for(rowcol=0;rowcol<N;rowcol++){
if(ndvi_median[rowcol]<99999.99){
#pragma omp atomic
n+=1;
#pragma omp atomic
e_sat_sum+=e_sat(mat1[rowcol]*0.02,mat2[rowcol]);
}
}
#pragma omp barrier
esat_avg=e_sat_sum/n;
e_act=eact(esat_avg,hr_s);
printf("ndvimedian\t");
#pragma omp parallel for default(none) private(i, j, temp)shared(N, ndvi_median)
for(i=0;i<N;i++){
for(j=i+1;j<N-1;j++){
if(ndvi_median[i]>ndvi_median[j]){
temp=ndvi_median[j];
ndvi_median[j]=ndvi_median[i];
ndvi_median[i]=temp;
}
}
}
#pragma omp barrier
if(n%2==0) ndvimedian=((ndvi_median[n/2]+ndvi_median[(n/2)-1])/2)*0.0001;
else ndvimedian=ndvi_median[n/2]*0.0001;
printf("= %i <> %f [-]\t",n,ndvimedian);
n=0;
printf("sebs\n");
#pragma omp parallel for default(none) private(rowcol, sebseta, sebsevapfr, kin, kin24, e0)shared(N, e_act, doy, p_s, t_s, z_pbl, u_s, ndvimedian, ndvi_min, ndvi_max, alt_ms, hr_s,mat1, mat2, mat3, mat4, mat5, mat6, mat7, matOut0, matOut, latitude )
for(rowcol=0;rowcol<N;rowcol++){
if(mat1[rowcol]==-28768||mat1[rowcol]*0.02<=250.0||mat1[rowcol]*0.02>365.0
||mat5[rowcol]>=10000.0||mat5[rowcol]<=0.0) matOut[rowcol] = -28768;
else {
e0 = 0.5*((mat3[rowcol]*0.002+0.49)+(mat4[rowcol]*0.002+0.49));
kin = k_in(mat2[rowcol],doy,mat7[rowcol]*0.01);
kin24 = k_in24(latitude[rowcol], doy, mat2[rowcol]);
sebsevapfr = sebs_evapfr(z_pbl,t_s,p_s,u_s,hr_s,alt_ms,e_act,kin,mat6[rowcol]*0.001,mat5[rowcol]*0.0001,ndvi_min,ndvi_max,ndvimedian,e0,mat1[rowcol]*0.02,mat2[rowcol]);
matOut0[rowcol] = sebsevapfr;
sebseta = sebs_eta(doy,mat5[rowcol]*0.0001,ndvi_max,ndvi_min,ndvimedian,u_s,z_pbl,t_s,p_s,alt_ms,e_act,latitude[rowcol],kin24,mat6[rowcol]*0.001,mat2[rowcol],mat1[rowcol]*0.02,e0,sebsevapfr);
matOut[rowcol] = sebseta;
}
}
#pragma omp barrier
printf("write to file\n");
GDALRasterIO(hBOut0,GF_Write,0,0,nX,nY,matOut0,nX,nY,GDT_Float32,0,0);
GDALClose(hDOut0);
GDALRasterIO(hBOut,GF_Write,0,0,nX,nY,matOut,nX,nY,GDT_Float32,0,0);
GDALClose(hDOut);
if(mat1 != NULL) free(mat1);
if(mat2 != NULL) free(mat2);
if(mat3 != NULL) free(mat3);
if(mat4 != NULL) free(mat4);
if(mat5 != NULL) free(mat5);
if(mat6 != NULL) free(mat6);
if(matOut0 != NULL) free(matOut0);
if(matOut != NULL) free(matOut);
if(latitude != NULL) free(latitude);
if(ndvi_median != NULL) free(ndvi_median);
GDALClose(hD1);
GDALClose(hD2);
GDALClose(hD3);
GDALClose(hD4);
GDALClose(hD5);
GDALClose(hD6);
return(EXIT_SUCCESS);
}
