#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <dirent.h>
#include "hdf5.h"
#include <math.h>
#include <time.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_sf_bessel.h>
#include "mclib_3d.h"
#include <omp.h>
#define R_DIM 1260
#define THETA_DIM 280
#define PHI_DIM 280
void read_hydro(char hydro_prefix[200], int frame, double r_inj, double **x, double **y,  double **z, double **szx, double **szy, double **r, double **theta, double **phi,\
double **velx, double **vely, double **velz, double **dens, double **pres, double **gamma, double **dens_lab, double **temp, int *number,  int ph_inj_switch, double min_r, double max_r, double fps, FILE *fPtr)
{
FILE *hydroPtr=NULL;
char hydrofile[200]="", file_num[200]="", full_file[200]="",file_end[200]=""  ;
char buf[10]="";
int i=0, j=0, k=0, elem=0, elem_factor=0;
int phi_min_index=0, phi_max_index=0, r_min_index=0, r_max_index=0, theta_min_index=0, theta_max_index=0; 
int r_index=0, theta_index=0, phi_index=0, hydro_index=0, all_index_buffer=0, adjusted_remapping_index=0, dr_index=0;
int *remapping_indexes=NULL;
float buffer=0;
float *dens_unprc=NULL;
float *vel_r_unprc=NULL;
float *vel_theta_unprc=NULL;
float *vel_phi_unprc=NULL;
float *pres_unprc=NULL;
double ph_rmin=0, ph_rmax=0;
double r_in=1e10, r_ref=2e13;
double *r_edge=NULL;
double *dr=NULL;
double *r_unprc=malloc(sizeof(double)*R_DIM);
double *theta_unprc=malloc(sizeof(double)*THETA_DIM);
double *phi_unprc=malloc(sizeof(double)*PHI_DIM);
if (ph_inj_switch==0)
{
ph_rmin=min_r;
ph_rmax=max_r;
}
snprintf(file_end,sizeof(file_end),"%s","small.data" );
snprintf(hydrofile,sizeof(hydrofile),"%s%s%d%s",hydro_prefix,"u0", 1,"-" );
modifyFlashName(file_num, hydrofile, frame,1);
fprintf(fPtr,">> Opening file %s\n", file_num);
fflush(fPtr);
snprintf(full_file, sizeof(full_file), "%s%s", file_num, file_end);
hydroPtr=fopen(full_file, "rb");
fread(&buffer, sizeof(float), 1,hydroPtr); 
fread(&phi_min_index, sizeof(int)*1, 1,hydroPtr); 
fread(&phi_max_index, sizeof(int)*1, 1,hydroPtr);
fread(&theta_min_index, sizeof(int)*1, 1,hydroPtr);
fread(&theta_max_index, sizeof(int)*1, 1,hydroPtr);
fread(&r_min_index, sizeof(int)*1, 1,hydroPtr);
fread(&r_max_index, sizeof(int)*1, 1,hydroPtr);
fclose(hydroPtr);
r_min_index--;
r_max_index--;
theta_min_index--;
theta_max_index--;
phi_min_index--;
phi_max_index--;
elem=(r_max_index+1-r_min_index)*(theta_max_index+1-theta_min_index)*(phi_max_index+1-phi_min_index); 
dens_unprc=malloc(elem*sizeof(float));
vel_r_unprc=malloc(elem*sizeof(float));
vel_theta_unprc=malloc(elem*sizeof(float));
pres_unprc=malloc(elem*sizeof(float));
vel_phi_unprc=malloc(elem*sizeof(float));
hydroPtr=fopen(full_file, "rb");
fread(&buffer, sizeof(float), 1,hydroPtr); 
fread(&all_index_buffer, sizeof(int)*1, 1,hydroPtr); 
fread(&all_index_buffer, sizeof(int)*1, 1,hydroPtr);
fread(&all_index_buffer, sizeof(int)*1, 1,hydroPtr);
fread(&all_index_buffer, sizeof(int)*1, 1,hydroPtr);
fread(&all_index_buffer, sizeof(int)*1, 1,hydroPtr);
fread(&all_index_buffer, sizeof(int)*1, 1,hydroPtr);
fread(&buffer, sizeof(float), 1,hydroPtr); 
fread(&buffer, sizeof(float), 1,hydroPtr); 
fread(dens_unprc, sizeof(float),elem, hydroPtr); 
fclose(hydroPtr);
snprintf(hydrofile,sizeof(hydrofile),"%s%s%d%s",hydro_prefix,"u0", 2,"-" );
modifyFlashName(file_num, hydrofile, frame,1);
snprintf(full_file, sizeof(full_file), "%s%s", file_num, file_end);
hydroPtr=fopen(full_file, "rb");
fread(&buffer, sizeof(float), 1,hydroPtr); 
fread(&all_index_buffer, sizeof(int)*1, 1,hydroPtr); 
fread(&all_index_buffer, sizeof(int)*1, 1,hydroPtr);
fread(&all_index_buffer, sizeof(int)*1, 1,hydroPtr);
fread(&all_index_buffer, sizeof(int)*1, 1,hydroPtr);
fread(&all_index_buffer, sizeof(int)*1, 1,hydroPtr);
fread(&all_index_buffer, sizeof(int)*1, 1,hydroPtr);
fread(&buffer, sizeof(float), 1,hydroPtr); 
fread(&buffer, sizeof(float), 1,hydroPtr); 
fread(vel_r_unprc, sizeof(float),elem, hydroPtr);
fclose(hydroPtr);
snprintf(hydrofile,sizeof(hydrofile),"%s%s%d%s",hydro_prefix,"u0", 3,"-" );
modifyFlashName(file_num, hydrofile, frame,1);
snprintf(full_file, sizeof(full_file), "%s%s", file_num, file_end);
hydroPtr=fopen(full_file, "rb");
fread(&buffer, sizeof(float), 1,hydroPtr); 
fread(&all_index_buffer, sizeof(int)*1, 1,hydroPtr); 
fread(&all_index_buffer, sizeof(int)*1, 1,hydroPtr);
fread(&all_index_buffer, sizeof(int)*1, 1,hydroPtr);
fread(&all_index_buffer, sizeof(int)*1, 1,hydroPtr);
fread(&all_index_buffer, sizeof(int)*1, 1,hydroPtr);
fread(&all_index_buffer, sizeof(int)*1, 1,hydroPtr);
fread(&buffer, sizeof(float), 1,hydroPtr); 
fread(&buffer, sizeof(float), 1,hydroPtr); 
fread(vel_theta_unprc, sizeof(float),elem, hydroPtr);
fclose(hydroPtr);
snprintf(hydrofile,sizeof(hydrofile),"%s%s%d%s",hydro_prefix,"u0", 4,"-" );
modifyFlashName(file_num, hydrofile, frame,1);
snprintf(full_file, sizeof(full_file), "%s%s", file_num, file_end);
hydroPtr=fopen(full_file, "rb");
fread(&buffer, sizeof(float), 1,hydroPtr); 
fread(&all_index_buffer, sizeof(int)*1, 1,hydroPtr); 
fread(&all_index_buffer, sizeof(int)*1, 1,hydroPtr);
fread(&all_index_buffer, sizeof(int)*1, 1,hydroPtr);
fread(&all_index_buffer, sizeof(int)*1, 1,hydroPtr);
fread(&all_index_buffer, sizeof(int)*1, 1,hydroPtr);
fread(&all_index_buffer, sizeof(int)*1, 1,hydroPtr);
fread(&buffer, sizeof(float), 1,hydroPtr); 
fread(&buffer, sizeof(float), 1,hydroPtr); 
fread(vel_phi_unprc, sizeof(float),elem, hydroPtr);
fclose(hydroPtr);
snprintf(hydrofile,sizeof(hydrofile),"%s%s%d%s",hydro_prefix,"u0", 8,"-" );
modifyFlashName(file_num, hydrofile, frame,1);
snprintf(full_file, sizeof(full_file), "%s%s", file_num, file_end);
hydroPtr=fopen(full_file, "rb");
fread(&buffer, sizeof(float), 1,hydroPtr); 
fread(&all_index_buffer, sizeof(int)*1, 1,hydroPtr); 
fread(&all_index_buffer, sizeof(int)*1, 1,hydroPtr);
fread(&all_index_buffer, sizeof(int)*1, 1,hydroPtr);
fread(&all_index_buffer, sizeof(int)*1, 1,hydroPtr);
fread(&all_index_buffer, sizeof(int)*1, 1,hydroPtr);
fread(&all_index_buffer, sizeof(int)*1, 1,hydroPtr);
fread(&buffer, sizeof(float), 1,hydroPtr); 
fread(&buffer, sizeof(float), 1,hydroPtr); 
fread(pres_unprc, sizeof(float),elem, hydroPtr);
fclose(hydroPtr);
if (frame<=1300)
{
snprintf(hydrofile,sizeof(hydrofile),"%s%s%d%s",hydro_prefix,"grid0", 0,"-x1.data" );
adjusted_remapping_index=(0*420)+r_min_index;
}
else if (frame<=2000)
{
snprintf(hydrofile,sizeof(hydrofile),"%s%s%d%s",hydro_prefix,"grid0", 1,"-x1.data" );
adjusted_remapping_index=(1*420)+r_min_index;
}
else if (frame<=10000)
{
snprintf(hydrofile,sizeof(hydrofile),"%s%s%d%s",hydro_prefix,"grid0", 2,"-x1.data" );
adjusted_remapping_index=(2*420)+r_min_index;
}
else if (frame<=20000)
{
snprintf(hydrofile,sizeof(hydrofile),"%s%s%d%s",hydro_prefix,"grid0", 3,"-x1.data" );
adjusted_remapping_index=(3*420)+r_min_index;
}
else if (frame<=35000)
{
snprintf(hydrofile,sizeof(hydrofile),"%s%s%d%s",hydro_prefix,"grid0", 4,"-x1.data" );
adjusted_remapping_index=(4*420)+r_min_index;
}
else if (frame<=50000)
{
snprintf(hydrofile,sizeof(hydrofile),"%s%s%d%s",hydro_prefix,"grid0", 5,"-x1.data" );
adjusted_remapping_index=(5*420)+r_min_index;
}
else if (frame<=60000)
{
snprintf(hydrofile,sizeof(hydrofile),"%s%s%d%s",hydro_prefix,"grid0", 6,"-x1.data" );
adjusted_remapping_index=(6*420)+r_min_index;
}
fprintf(fPtr,"Reading Radius: %s\n", hydrofile);
fflush(fPtr);
hydroPtr=fopen(hydrofile, "r");
i=0;
while (i<R_DIM)
{
fscanf(hydroPtr, "%lf", (r_unprc+i));  
fgets(buf, 3,hydroPtr); 
i++;
}
fclose(hydroPtr);
r_edge=malloc(sizeof(double)*(3780+1));
dr=malloc(sizeof(double)*(3780));
*(r_edge+0)=r_in;
i=0;
for (i=1;i<3780;i++)
{
*(r_edge+i)=(*(r_edge+i-1))+((*(r_edge+i-1))*(M_PI/560)/(1+((*(r_edge+i-1))/r_ref))); 
*(dr+i-1)=(*(r_edge+i))-(*(r_edge+i-1));
}
free(r_edge);
snprintf(hydrofile,sizeof(hydrofile),"%s%s",hydro_prefix,"grid-x2.data" );
fprintf(fPtr,"Reading Theta: %s\n", hydrofile);
fflush(fPtr);
hydroPtr=fopen(hydrofile, "r");
i=0;
while (i<THETA_DIM)
{
fscanf(hydroPtr, "%lf", (theta_unprc+i));  
fgets(buf, 3,hydroPtr); 
i++;
}
fclose(hydroPtr);
snprintf(hydrofile,sizeof(hydrofile),"%s%s",hydro_prefix,"grid-x3.data" );
hydroPtr=fopen(hydrofile, "r");
i=0;
while (i<PHI_DIM)
{
fscanf(hydroPtr, "%lf", (phi_unprc+i));  
fgets(buf, 3,hydroPtr); 
i++;
}
fclose(hydroPtr);
elem_factor=0;
elem=0;
while (elem==0)
{
elem=0;
elem_factor++;
for (i=0;i<(phi_max_index+1-phi_min_index);i++)
{
for (j=0;j<(theta_max_index+1-theta_min_index);j++)
{
for (k=0;k<(r_max_index+1-r_min_index);k++)
{
r_index=r_min_index+k;
if (ph_inj_switch==0)
{
if (((ph_rmin - elem_factor*C_LIGHT/fps)<(*(r_unprc+r_index))) && (*(r_unprc+r_index)  < (ph_rmax + elem_factor*C_LIGHT/fps) ))
{
elem++;
}
}
else
{
if (((r_inj - elem_factor*C_LIGHT/fps)<(*(r_unprc+r_index))) && (*(r_unprc+r_index)  < (r_inj + elem_factor*C_LIGHT/fps) ))
{
elem++;
}
}
}
}
}
}
fprintf(fPtr,"Number of post restricted Elems: %d %e\n", elem, r_inj);
fflush(fPtr);
(*pres)=malloc (elem * sizeof (double ));
(*velx)=malloc (elem * sizeof (double ));
(*vely)=malloc (elem * sizeof (double ));
(*velz)=malloc (elem * sizeof (double ));
(*dens)=malloc (elem * sizeof (double ));
(*x)=malloc (elem * sizeof (double ));
(*y)=malloc (elem * sizeof (double ));
(*z)=malloc (elem * sizeof (double ));
(*r)=malloc (elem * sizeof (double ));
(*theta)=malloc (elem * sizeof (double ));
(*phi)=malloc (elem * sizeof (double ));
(*gamma)=malloc (elem * sizeof (double ));
(*dens_lab)=malloc (elem * sizeof (double ));
(*szx)=malloc (elem * sizeof (double )); 
(*szy)=malloc (elem * sizeof (double )); 
(*temp)=malloc (elem * sizeof (double ));
elem=0;
for (i=0;i<(phi_max_index+1-phi_min_index);i++)
{
for (j=0;j<(theta_max_index+1-theta_min_index);j++)
{
for (k=0;k<(r_max_index+1-r_min_index);k++)
{
r_index=r_min_index+k; 
theta_index=theta_min_index+j;
phi_index=phi_min_index+i;
dr_index=adjusted_remapping_index+k;
hydro_index=(i*(r_max_index+1-r_min_index)*(theta_max_index+1-theta_min_index) + j*(r_max_index+1-r_min_index) + k  );
if (ph_inj_switch==0)
{
if (((ph_rmin - elem_factor*C_LIGHT/fps)<(*(r_unprc+r_index))) && (*(r_unprc+r_index)  < (ph_rmax + elem_factor*C_LIGHT/fps) ))
{
(*pres)[elem] = *(pres_unprc+hydro_index);
(*dens)[elem] = *(dens_unprc+hydro_index);
(*temp)[elem] =  pow(3*(*(pres_unprc+hydro_index))*pow(C_LIGHT,2.0)/(A_RAD) ,1.0/4.0);
(*gamma)[elem] = pow(pow(1.0-(pow(*(vel_r_unprc+hydro_index),2)+ pow(*(vel_theta_unprc+hydro_index),2)+pow(*(vel_phi_unprc+hydro_index),2)),0.5),-1);
(*dens_lab)[elem] = (*(dens_unprc+hydro_index))*pow(pow(1.0-(pow(*(vel_r_unprc+hydro_index),2)+ pow(*(vel_theta_unprc+hydro_index),2)+pow(*(vel_phi_unprc+hydro_index),2)),0.5),-1);
(*r)[elem] = *(r_unprc+r_index);
(*theta)[elem] = *(theta_unprc+theta_index);
(*phi)[elem] = *(phi_unprc+phi_index);
(*x)[elem] = (*(r_unprc+r_index))*sin(*(theta_unprc+theta_index))*cos(*(phi_unprc+phi_index));
(*y)[elem] = (*(r_unprc+r_index))*sin(*(theta_unprc+theta_index))*sin(*(phi_unprc+phi_index));
(*z)[elem] = (*(r_unprc+r_index))*cos(*(theta_unprc+theta_index));
(*szx)[elem] =  *(dr+dr_index);
(*szy)[elem] =  M_PI/560;
(*velx)[elem]=((*(vel_r_unprc+hydro_index))*sin(*(theta_unprc+theta_index))*cos(*(phi_unprc+phi_index))) + ((*(vel_theta_unprc+hydro_index))*cos(*(theta_unprc+theta_index))*cos(*(phi_unprc+phi_index))) - ((*(vel_phi_unprc+hydro_index))*sin(*(phi_unprc+phi_index)));
(*vely)[elem]=((*(vel_r_unprc+hydro_index))*sin(*(theta_unprc+theta_index))*sin(*(phi_unprc+phi_index))) + ((*(vel_theta_unprc+hydro_index))*cos(*(theta_unprc+theta_index))*sin(*(phi_unprc+phi_index))) + ((*(vel_phi_unprc+hydro_index))*cos(*(phi_unprc+phi_index)));
(*velz)[elem]=((*(vel_r_unprc+hydro_index))*cos(*(theta_unprc+theta_index))) - ((*(vel_theta_unprc+hydro_index))*sin(*(theta_unprc+theta_index)));
elem++;
}
}
else
{
if (((r_inj - elem_factor*C_LIGHT/fps)<(*(r_unprc+r_index))) && (*(r_unprc+r_index)  < (r_inj + elem_factor*C_LIGHT/fps) ))
{
(*pres)[elem] = *(pres_unprc+hydro_index);
(*dens)[elem] = *(dens_unprc+hydro_index);
(*temp)[elem] =  pow(3*(*(pres_unprc+hydro_index))*pow(C_LIGHT,2.0)/(A_RAD) ,1.0/4.0);
(*gamma)[elem] = pow(pow(1.0-(pow(*(vel_r_unprc+hydro_index),2)+ pow(*(vel_theta_unprc+hydro_index),2)+pow(*(vel_phi_unprc+hydro_index),2)),0.5),-1);
(*dens_lab)[elem] = (*(dens_unprc+hydro_index))*pow(pow(1.0-(pow(*(vel_r_unprc+hydro_index),2)+ pow(*(vel_theta_unprc+hydro_index),2)+pow(*(vel_phi_unprc+hydro_index),2)),0.5),-1);
(*r)[elem] = *(r_unprc+r_index);
(*theta)[elem] = *(theta_unprc+theta_index);
(*phi)[elem] = *(phi_unprc+phi_index);
(*x)[elem] = (*(r_unprc+r_index))*sin(*(theta_unprc+theta_index))*cos(*(phi_unprc+phi_index));
(*y)[elem] = (*(r_unprc+r_index))*sin(*(theta_unprc+theta_index))*sin(*(phi_unprc+phi_index));
(*z)[elem] = (*(r_unprc+r_index))*cos(*(theta_unprc+theta_index));
(*szx)[elem] =  *(dr+dr_index);
(*szy)[elem] =  M_PI/560;
(*velx)[elem]=((*(vel_r_unprc+hydro_index))*sin(*(theta_unprc+theta_index))*cos(*(phi_unprc+phi_index))) + ((*(vel_theta_unprc+hydro_index))*cos(*(theta_unprc+theta_index))*cos(*(phi_unprc+phi_index))) - ((*(vel_phi_unprc+hydro_index))*sin(*(phi_unprc+phi_index)));
(*vely)[elem]=((*(vel_r_unprc+hydro_index))*sin(*(theta_unprc+theta_index))*sin(*(phi_unprc+phi_index))) + ((*(vel_theta_unprc+hydro_index))*cos(*(theta_unprc+theta_index))*sin(*(phi_unprc+phi_index))) + ((*(vel_phi_unprc+hydro_index))*cos(*(phi_unprc+phi_index)));
(*velz)[elem]=((*(vel_r_unprc+hydro_index))*cos(*(theta_unprc+theta_index))) - ((*(vel_theta_unprc+hydro_index))*sin(*(theta_unprc+theta_index)));
elem++;
}
}
}
}
}
*number=elem;
free(pres_unprc); free(dens_unprc); free(r_unprc); free(theta_unprc); free(phi_unprc);free(dr);free(vel_r_unprc); free(vel_theta_unprc); free(vel_phi_unprc); 
}
void photonInjection3D( struct photon **ph, int *ph_num, double r_inj, double ph_weight, int min_photons, int max_photons, char spect, int array_length, double fps, double theta_min, double theta_max,\
double *x, double *y, double *z, double *szx, double *szy, double *r, double *theta, double *phi, double *temps, double *vx, double *vy, double *vz, gsl_rng * rand, FILE *fPtr)
{
int i=0, block_cnt=0, *ph_dens=NULL, ph_tot=0, j=0,k=0;
double ph_dens_calc=0.0, fr_dum=0.0, y_dum=0.0, yfr_dum=0.0, fr_max=0, bb_norm=0, position_phi, ph_weight_adjusted, theta_prime=0;
double com_v_phi, com_v_theta, *p_comv=NULL, *boost=NULL; 
double *l_boost=NULL; 
float num_dens_coeff;
if (spect=='w') 
{
num_dens_coeff=8.44;
}
else
{
num_dens_coeff=20.29; 
}
printf("%e, %e\n",*(phi+i), theta_max);
for(i=0;i<array_length;i++)
{
theta_prime=acos(*(y+i)/(*(r+i))); 
if ( (theta_prime< theta_max) && (theta_prime >= theta_min) ) 
{
block_cnt++;
}
}
printf("Blocks: %d\n", block_cnt);
ph_dens=malloc(block_cnt * sizeof(int));
j=0;
ph_tot=0;
ph_weight_adjusted=ph_weight;
while ((ph_tot>max_photons) || (ph_tot<min_photons) )
{
j=0;
ph_tot=0;
for (i=0;i<array_length;i++)
{
theta_prime=acos(*(y+i)/(*(r+i)));
if ( (theta_prime< theta_max) && (theta_prime >= theta_min) )
{
ph_dens_calc=(num_dens_coeff*pow(*(temps+i),3.0)*pow(*(r+i),2)*sin(*(theta+i))* pow(*(szy+i),2.0)*(*(szx+i)) /(ph_weight_adjusted))*pow(pow(1.0-(pow(*(vx+i),2)+pow(*(vy+i),2)+pow(*(vz+i),2)),0.5),-1) ; 
(*(ph_dens+j))=gsl_ran_poisson(rand,ph_dens_calc) ; 
ph_tot+=(*(ph_dens+j));
j++;
}
}
if (ph_tot>max_photons)
{
ph_weight_adjusted*=10;
}
else if (ph_tot<min_photons)
{
ph_weight_adjusted*=0.5;
}
}
printf("%d\n", ph_tot);
(*ph)=malloc (ph_tot * sizeof (struct photon ));
p_comv=malloc(4*sizeof(double));
boost=malloc(3*sizeof(double));
l_boost=malloc(4*sizeof(double));
ph_tot=0;
k=0;
for (i=0;i<array_length;i++)
{
theta_prime=acos(*(y+i)/(*(r+i)));
if ( (theta_prime< theta_max) && (theta_prime >= theta_min) )  
{
/
j++;
}
fclose(hydroPtr);
}
j=0; 
i=0; 
rPtr=r_unprc_0; 
*(remapping_start_index+1)=j; 
while (i<R_DIM)
{
if (*(rPtr+i)== *(r_unprc_1+0))
{
rPtr=r_unprc_1;
i=0;
*(remapping_start_index+1)=j;
}
else if (*(rPtr+i)== *(r_unprc_2+0))
{
rPtr=r_unprc_2;
i=0;
*(remapping_start_index+2)=j;
}
else if (*(rPtr+i)== *(r_unprc_3+0))
{
rPtr=r_unprc_3;
i=0;
*(remapping_start_index+3)=j;
}
else if (*(rPtr+i)== *(r_unprc_4+0))
{
rPtr=r_unprc_4;
i=0;
*(remapping_start_index+4)=j;
}
else if (*(rPtr+i)== *(r_unprc_5+0))
{
rPtr=r_unprc_5;
i=0;
*(remapping_start_index+5)=j;
}
else if (*(rPtr+i)== *(r_unprc_6+0))
{
rPtr=r_unprc_6;
i=0;
*(remapping_start_index+6)=j;
}
j++;
i++;
}
printf("Indexes %d, %d, %d, %d, %d, %d, %d\n Elems: %d\n", *(remapping_start_index+0), *(remapping_start_index+1), *(remapping_start_index+2), *(remapping_start_index+3), *(remapping_start_index+4), *(remapping_start_index+5), *(remapping_start_index+6), j);
r_edge=malloc(sizeof(double)*(j+1));
dr=malloc(sizeof(double)*j);
*(r_edge+0)=r_in;
i=0;
for (i=1;i<j;i++)
{
*(r_edge+i)=(*(r_edge+i-1))+((*(r_edge+i-1))*(M_PI/560)/(1+((*(r_edge+i-1))/r_ref))); 
*(dr+i-1)=(*(r_edge+i))-(*(r_edge+i-1));
}
free(r_edge);
free(r_unprc_0);
free(r_unprc_1);
free(r_unprc_2);
free(r_unprc_3);
free(r_unprc_4);
free(r_unprc_5);
free(r_unprc_6);
return remapping_start_index;
}
