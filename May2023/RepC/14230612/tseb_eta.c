


#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <gdal.h>
#include <string.h>
#include "tseb_eta.h"

void usage()
{
printf( "-----------------------------------------\n");
printf( "--Modis Processing OpenMP Code------------\n");
printf( "-----------------------------------------\n");
printf( "./eta_tseb inLst\n");
printf( "\tout_ETA_TSEB\n");
printf( "-----------------------------------------\n");
return;
}

int main(int argc,char *argv[])
{
if( argc < 2) {
usage();
return 1;
}
char	*inB1	= argv[1]; 
char	*smF	= argv[2]; 


GDALAllRegister();

GDALDatasetH hD1 = GDALOpen(inB1,GA_ReadOnly);

if(hD1==NULL){
printf("One or more input files ");
printf("could not be loaded\n");
exit(1);
}

GDALDriverH hDr1 = GDALGetDatasetDriver(hD1);

GDALDatasetH hDOut = GDALCreateCopy( hDr1, smF,hD1,FALSE,NULL,NULL,NULL);
GDALRasterBandH hBOut = GDALGetRasterBand(hDOut,1);

GDALRasterBandH hB1 = GDALGetRasterBand(hD1,1);

int nX = GDALGetRasterBandXSize(hB1);
int nY = GDALGetRasterBandYSize(hB1);
int N=nX*nY; 


float *lst = (float*) malloc(N*sizeof(float));
float *sm = (float*) malloc(N*sizeof(float));


GDALRasterIO(hB1,GF_Read,0,0,nX,nY,lst,nX,nY,GDT_Float32,0,0);
float lst_h=0.0,lst_c=400.0;
int i;
#pragma omp parallel for default(none) private(i)shared(N, lst, lst_h, lst_c, sm )
for(i=0;i<N;i++){
sm[i]=0.0;
if(lst[i]*0.02>=250.0&&lst[i]*0.02<345.0){
if (lst[i]*0.02>lst_h) lst_h=lst[i]*0.02;
if (lst[i]*0.02<lst_c) lst_c=lst[i]*0.02;
}}
#pragma omp barrier
printf("%f %f\n",lst_h, lst_c);

meteo		met		={0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
tkstruct 	tk		={0.0,0.0,0.0,0.0};
soilabsemiss 	soilabs_ems	={0.0,0.0,0.0};
leafabsemiss 	leafabs_ems	={0.0,0.0,0.0,0.0};
radweights 	radwts		={0.0,0.0,0.8,0.2,0.9,0.1};
Choud 		choudparms	={0.0,0.0};
NDVIrng 	ndvi_rng	={0.0,0.0,0.0};
Cover 		vegcover	={0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
refhts 		Z		={0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};

fluxstruct 	Flux		={0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
resistor 	resist		={0.0,0.0};
CanopyLight 	cpylight	={0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
LngWave 	RL		={0.0,0.0,0.0};
Albeds 		albedvegcan;


char	dumrec[NUMCHAR];
char	infofil[NUMCHAR];



char	inputimagefil[NUMCHAR],outputimagefil[NUMCHAR];
int	numbands,numpixels,numlines;
double	tkscale,tkoffset;
double	ndviscale,ndvioffset;
int	numlandusetypes=200;

char **landusename = (char **) malloc(numlandusetypes*sizeof(char *));
if(landusename == NULL) {
printf("Could not allocate space for landusename, exiting!\n");
exit(EXIT_FAILURE);
}
double *canopy_height = (double *) malloc(numlandusetypes*sizeof(double));
if(canopy_height==NULL) {
printf("Could not allocate space for canopy heights, exiting!\n");
exit(EXIT_FAILURE);
}
int *landusecode = (int *) malloc(numlandusetypes*sizeof(int));
if(landusecode==NULL) {
printf("Could not allocate space for land use codes, exiting!\n");
exit(EXIT_FAILURE);
}
double cG;



char curlandusename[NUMCHAR];
int refinfo[5];
short int *invec,*outvec,*outvecz;
int imageluval,imageluindex;
int bct,pct,lct;

int outbands = 12;
int stabflag = 0;
int exitflag = 0;
int iternum;
int condenseflag;
int nodataval = 0;

double tempval,esat,eslope,rsat,rhodryair;
double lst,lndvi;  
double frac_cover,LAI;

double ndviflt;
double const Lmonobfinit = -1e9;
double Lmonobfprev;
double Re,kB;		

FILE *fimgp;
FILE *fomgp;

nodataval = NODATAVAL;


if(argc < 2) {
printf("you must enter a control file name!\n");
exit(EXIT_FAILURE);
}

strcpy(infofil,argv[1]);
printf("information file is:\n %s\n",infofil);




richesat_ptr(tk.air,&esat,&eslope);
printf("esat is %f and eslope is %f\n",esat,eslope);
met.esat = esat;
met.desat_dtk = eslope;
rsat = e2r(esat,met.presmbar);
met.mixratio = met.rh*rsat;
printf("mixing ratio is %f\n",met.mixratio);
tempval = r2e(met.mixratio,met.presmbar);

met.ea = tempval;
printf("vapor pressure ea is %f\n",met.ea);
tempval = e_mb2rhoden(met.ea,tk.air);
met.vapden = 0.001*tempval;
rhodryair = 0.001*rhodry(met.vapden,met.presmbar,tk.air);
met.rhoair = e2rhomoist(met.ea,met.presmbar,tk.air);
tempval = mixr2spechum(met.mixratio);
met.spechum = tempval;

tempval = cpd2cp(met.spechum);
met.cp = tempval;

met.lambdav = latenthtvap(k2c(tk.air));
met.gamma = gammafunc(tk.air,met.presmbar,met.cp,met.lambdav);


printf("air temperature(Kelvin): %f\n",tk.air);
printf("met structure:\n");
printf("rh %f ea %f mixratio %f spechum %f\n",met.rh,met.ea,met.mixratio,met.spechum);
printf("vapden %f rhoair %f cp %f esat %f\n",met.vapden,met.rhoair,met.cp,met.esat);

invec = (short int *) malloc(numbands*sizeof(short int));

outvec = (short int *) malloc(outbands*sizeof(short int));

outvecz = (short int *) malloc(outbands*sizeof(short int));
for(bct=0;bct<outbands;bct++) {
outvecz[bct] = (short int) nodataval;
}

if((fimgp = fopen(inputimagefil,"rb"))==NULL) {
printf("Cannot open input image file!!\n");
exit(EXIT_FAILURE);
};
Z.d0bare = 0.01;

if((fomgp = fopen(outputimagefil,"wb"))==NULL) {
printf("Cannot open output image file!!\n");
exit(EXIT_FAILURE);
}


for(lct=0;lct<numlines;lct++) {
printf("=====\n");
printf("now on line: %d\n",lct+1);
refinfo[0] = lct;
for(pct=0;pct<numpixels;pct++) {
if(pct==22 && lct == 4) {
printf("stopping\n");
} 
refinfo[1] = pct;

fread(invec,sizeof(short int)*numbands,1,fimgp);
refinfo[2] = invec[0];
refinfo[3] = invec[1];
refinfo[4] = invec[2];




lst = (((double) invec[0])*tkscale+tkoffset)-273.15;


lndvi = ((double) invec[1])*ndviscale+ndvioffset;

if((lst < 0.00) || (lndvi < -0.5) || (lndvi > 1.0)) {

for(bct=0;bct<outbands;bct++) {
fwrite(&nodataval,sizeof(short int),1,fomgp);
} 
} else {


if(invec[2]==0) imageluval = 14;
else imageluval = invec[2];

imageluindex = -1;
do {
imageluindex++;
} while(imageluval != landusecode[imageluindex]);
strcpy(dumrec,landusename[imageluindex]);
sscanf(dumrec,"%s",curlandusename);



vegcover.canopyheight = canopy_height[imageluindex];

vegcover.clumpfactor = clump_factor(vegcover.clumpfactornadir,vegcover.canopyheight,vegcover.canopywidth,vegcover.viewangrad);

Z.d0 = 0.67*canopy_height[imageluindex];
Z.z0 = 0.125*canopy_height[imageluindex];

ndviflt = lndvi;

frac_cover = frac_cover_choud(ndvi_rng.min,ndvi_rng.max,ndviflt,choudparms.p); 
LAI = LAI_choudfunc(frac_cover,choudparms.Beta);

vegcover.frac = frac_cover;
vegcover.LAI = LAI;

if(vegcover.LAI < 0.5) radwts.Kd = 0.9;
else if(vegcover.LAI > 2.0) radwts.Kd = 0.7;
else radwts.Kd = 0.6;

tk.composite = lst+273.15;


if(vegcover.frac < 0.5) tk.canopy = 0.5*(tk.air+tk.composite);
else tk.canopy = tk.air;

if(vegcover.frac < 0.8) component_tempk_soil(vegcover.frac,&tk);


else tk.soil = tk.composite;



Z.L = Lmonobfinit; 
if(tk.composite > tk.air) stabflag = 1;
else stabflag = 0;




if(stabflag == 1) xandpsi(&Z,&vegcover);
else stabphipsi(&Z);



if(vegcover.frac > 0.1) {



canopyrho(&met,&vegcover,&radwts,&leafabs_ems,&soilabs_ems,&cpylight);


rhosoil2albedo(&soilabs_ems,&albedvegcan);
rhocpy2albedo(&cpylight,&albedvegcan,&radwts);
} else {



cpylight.rhovisdir = 0.0;
cpylight.rhonirdir = 0.0;
cpylight.rhovisdif = 0.0;
cpylight.rhonirdif = 0.0;
cpylight.tauvisdir = 1.0;
cpylight.taunirdir = 1.0;
cpylight.tauvisdif = 1.0;
cpylight.taunirdif = 1.0;
cpylight.tautir = 1.0;
rhosoil2albedo(&soilabs_ems,&albedvegcan);
} 

iternum = 0;
exitflag = 0;

while(exitflag == 0) {

iternum++;


if((strcmp(curlandusename,"bare")==0) ||
(strcmp(curlandusename,"urban")==0) ||
(strcmp(curlandusename,"water")==0) || 
(invec[1] < -1000) || 
(ndviflt <= ndvi_rng.baresoil))
{




getwindbare(&met,&Z,&resist,&tk);


onelayer(&Flux,&tk,cG,&met,&albedvegcan,&soilabs_ems,
&leafabs_ems,&resist, &RL);

} else {



getwind(&met,&Z,&vegcover,&resist);



getrls(&met,&tk,&soilabs_ems,&leafabs_ems,&RL);

getRnG(&cpylight,&albedvegcan,&RL,&met,&Flux,cG,&vegcover,refinfo);
printf("RL.soil %f RL.air %f RL.canopy %f\n",RL.soil,RL.air, RL.canopy);
printf("Rn: %f Rncanopy: %f Rnsoil: %f\n",Flux.Rntotal,Flux.Rncanopy,Flux.Rnsoil);

twolayer(&tk,&Flux,&met,&resist,&vegcover);



condenseflag = nocondense(&Flux,&tk,&resist,&met,&vegcover);
} 

Flux.Htotal  = (Flux.Hsoil) + (Flux.Hcanopy);
Flux.LEtotal = (Flux.LEsoil) + (Flux.LEcanopy);

Lmonobfprev = Z.L;
monobfs(&met,&tk,&Flux,&Z);




if((((Z.L)-Lmonobfprev ) < 0.001) && ((Z.L)-Lmonobfprev) > -0.001) exitflag = 1;

else {
if((Z.L) > 0.0 && (Lmonobfprev > 0.0)) {
exitflag = 1;

} else {
if(iternum > MAXITER) {
exitflag = 1;

} else {
exitflag = 0;
xandpsi(&Z,&vegcover);
}

} 
} 

} 

if(stabflag == 0) { 


stabphipsi(&Z);

if((strcmp(curlandusename,"bare")==0) ||
(strcmp(curlandusename,"urban")==0)) 
{



getwindbare(&met,&Z,&resist,&tk);

onelayer(&Flux,&tk,cG,&met,&albedvegcan,&soilabs_ems,
&leafabs_ems,&resist, &RL);
} else {
if(strcmp(curlandusename,"water")==0) 
{


getwindbare(&met,&Z,&resist,&tk);


radiatewaters(&tk,&Flux,&met,&albedvegcan);

Re =  reynolds(&tk,&met,&Z);
Z.z0h = (Z.z0)*7.48*exp(-2.46*pow(Re,0.25));
kB = log((Z.z0)/(Z.z0h));
resist.air = ((log(((Z.t)-(Z.d0))/(Z.z0)))+kB)/((met.ustar)*VONKARMAN);
hfluxresist_soil(&tk,&resist,&Flux,&met);
Flux.LEsoil = (Flux.Rnsoil)-(Flux.G)-(Flux.Hsoil);
Flux.Hcanopy = 0.0;
Flux.LEcanopy = 0.0;
Flux.Htotal = Flux.Hsoil;
Flux.LEtotal = Flux.LEsoil;


} else {



getwind(&met,&Z,&vegcover,&resist);

getrls(&met,&tk,&soilabs_ems,&leafabs_ems,&RL);
getRnG(&cpylight,&albedvegcan,&RL,&met,&Flux,cG,&vegcover,refinfo);
twolayer(&tk,&Flux,&met,&resist,&vegcover);
condenseflag = nocondense(&Flux,&tk,&resist,&met,&vegcover);

}
}
} 
outvec[0] = (short int) Flux.Hsoil;
outvec[1] = (short int) Flux.Hcanopy;
outvec[2] = (short int) Flux.LEsoil;
outvec[3] = (short int) Flux.LEcanopy;
outvec[4] = (short int) Flux.G;
outvec[5] = (short int) Flux.Rntotal;
if(lct==22 && pct==4) {
printf("Flux.Rntotal: %f lct: %d pct: %d\n",Flux.Rntotal,lct,pct);
exit(1);
}
outvec[6] = (short int) invec[0];
outvec[7] = (short int) invec[1];
outvec[8] = (short int) imageluval;
outvec[9] = (short int) (vegcover.LAI*1000);
outvec[10] = (short int) resist.air;
outvec[11] = (short int) resist.soil;
fwrite(outvec,sizeof(short int)*outbands,1,fomgp);
Z.L = Lmonobfinit ;
for(bct=0;bct<outbands;bct++) {
outvec[bct] = outvecz[bct];
} 
} 
} 
}
fclose(fimgp);
fclose(fomgp);
printf("wrote:\n%s\n",outputimagefil);
printf("Bands: %d Pixels: %d Lines: %d\n",outbands,numpixels,numlines);
return(EXIT_SUCCESS);
} 