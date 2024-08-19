#include<stdio.h>
#include "gdal.h"
#include "arrays.h"
#include<omp.h>
unsigned int mod09GAa(unsigned int pixel)
{
return (pixel & 0x03);
}
unsigned int mod09GAc(unsigned int pixel, int bandno) 
{
unsigned int qctemp;
pixel >>= 2 + (4 * (bandno - 1));	
qctemp = pixel & 0x0F;    
return qctemp;
}
void usage()
{
printf( "-----------------------------------------\n");
printf( "--Modis Processing chain--OpenMP code----\n");
printf( "-----------------------------------------\n");
printf( "./mod09ga inQA inB3\n");
printf( "\tout\n");
printf( "-----------------------------------------\n");
printf( "inQA\t\tModis MOD09GA QC_500m_1\n");
printf( "inB3\t\tModis MOD09GA Band3\n");
printf( "out\tQA corrected B3 output [-]\n");
return;
}
int main( int argc, char *argv[] )
{
if( argc < 4 ) {
usage();
return (EXIT_FAILURE);
}
char	*inB 	= argv[1]; 
char	*inB3 	= argv[2]; 
char	*outF	= argv[3];
GDALAllRegister();
GDALDatasetH hD = GDALOpen(inB,GA_ReadOnly);
GDALDatasetH hD3 = GDALOpen(inB3,GA_ReadOnly);
if(hD==NULL||hD3==NULL){
printf("One or more input files ");
printf("could not be loaded\n");
exit(EXIT_FAILURE);
}
GDALDriverH hDr3 = GDALGetDatasetDriver(hD3);
char **options = NULL;
options = CSLSetNameValue( options, "TILED", "YES" );
options = CSLSetNameValue( options, "COMPRESS", "DEFLATE" );
options = CSLSetNameValue( options, "PREDICTOR", "2" );
GDALDatasetH hDOut = GDALCreateCopy(hDr3,outF,hD3,FALSE,options,NULL,NULL);
GDALRasterBandH hBOut = GDALGetRasterBand(hDOut,1);
GDALRasterBandH hB = GDALGetRasterBand(hD,1);
GDALRasterBandH hB3 = GDALGetRasterBand(hD3,1);
int nX = GDALGetRasterBandXSize(hB3);
int nY = GDALGetRasterBandYSize(hB3);
int N=nX*nY;
unsigned int *l = aui1d(N);
int *l3 = ai1d(N);
int *lOut = ai1d(N);
int rowcol, qa, qa1;
GDALRasterIO(hB,GF_Read,0,0,nX,nY,l,nX,nY,GDT_UInt32,0,0);
GDALRasterIO(hB3,GF_Read,0,0,nX,nY,l3,nX,nY,GDT_Int32,0,0);
#pragma omp parallel for default(none) private (rowcol, qa, qa1) shared (N, l, l3, lOut)
for(rowcol=0;rowcol<N;rowcol++){
qa=mod09GAa(l[rowcol]);
qa1=mod09GAc(l3[rowcol],3);
if( qa == 0 || qa1 == 0 ) lOut[rowcol] = l3[rowcol];
else lOut[rowcol] = -28768;
}
#pragma omp barrier
GDALRasterIO(hBOut,GF_Write,0,0,nX,nY,lOut,nX,nY,GDT_Int32,0,0);
if( l != NULL ) free( l );
if( l3 != NULL ) free( l3 );
GDALClose(hD);
GDALClose(hD3);
GDALClose(hDOut);
return(EXIT_SUCCESS);
}
