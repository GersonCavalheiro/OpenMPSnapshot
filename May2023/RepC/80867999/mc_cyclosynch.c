#include "mcrat.h"
double calcCyclotronFreq(double magnetic_field)
{
return CHARGE_EL*magnetic_field/(2*M_PI*M_EL*C_LIGHT);
}
double calcEB(double magnetic_field)
{
return PL_CONST*calcCyclotronFreq(magnetic_field);
}
double calcBoundaryE(double magnetic_field, double temp)
{
return 14*pow(M_EL*C_LIGHT*C_LIGHT, 1.0/10.0)*pow(calcEB(magnetic_field), 9.0/10.0)*pow(calcDimlessTheta(temp), 3.0/10.0);
}
double calcDimlessTheta(double temp)
{
return K_B*temp/(M_EL*C_LIGHT*C_LIGHT);
}
double calcB(double el_dens, double temp)
{
#if B_FIELD_CALC == INTERNAL_E
return sqrt(EPSILON_B*8*M_PI*3*el_dens*K_B*temp/2);
#elif B_FIELD_CALC == TOTAL_E
return sqrt(8*M_PI*EPSILON_B*(el_dens*M_P*C_LIGHT*C_LIGHT+4*A_RAD*temp*temp*temp*temp/3));
#endif
}
double n_el_MJ(double el_dens, double dimlesstheta, double gamma)
{
return el_dens*gamma*sqrt(gamma*gamma-1)*exp(-gamma/dimlesstheta)/(dimlesstheta*gsl_sf_bessel_Kn(2, 1.0/dimlesstheta));
}
double n_el_MB(double el_dens, double dimlesstheta, double gamma)
{
double temp=dimlesstheta*(M_EL*C_LIGHT*C_LIGHT)/K_B;
double v=C_LIGHT*sqrt(1-(1/pow(gamma, 2)));
return el_dens*4*M_PI*pow(M_EL/(2*M_PI*K_B*temp) , 3/2)*(v*C_LIGHT*C_LIGHT/(pow(gamma, 3)))*exp((-M_EL*pow(v, 2))/(2*K_B*temp));
}
double Z(double nu, double nu_c, double gamma )
{
return pow(sqrt(pow(gamma,2)-1)*exp(1/gamma)/(1+gamma) ,2*nu*gamma/nu_c);
}
double Z_sec_der(double nu, double nu_c, double gamma)
{
return nu*(-2*pow(gamma,3)*(1+gamma) + 4*pow(gamma,4)*(1+gamma-pow(gamma,2)-pow(gamma,3))*log(sqrt(pow(gamma,2)-1)*exp(1/gamma)/(1+gamma) ))/(nu_c*pow(gamma,5)*(1+gamma));
}
double chi(double dimlesstheta, double gamma)
{
double val=0;
if (dimlesstheta<=0.08)
{
val=sqrt(2*dimlesstheta*(pow(gamma,2)-1)/(gamma*(3*pow(gamma,2)-1)));
}
else
{
val=sqrt(2*dimlesstheta/(3*gamma));
}
return val;
}
double gamma0(double nu, double nu_c, double dimlesstheta)
{
double val=0;
if (dimlesstheta<=0.08)
{
val=sqrt(pow(1+(2*nu*dimlesstheta/nu_c)*(1+(9*nu*dimlesstheta/(2*nu_c))), (-1.0/3.0) ));
}
else
{
val=sqrt(pow((1+(4*nu*dimlesstheta/(3*nu_c))), (2.0/3.0)) );
}
return val;
}
double jnu(double nu, double nu_c, double dimlesstheta, double el_dens)
{
double dimlesstheta_ref=calcDimlessTheta(1e7);
double gamma=gamma0(nu, nu_c, dimlesstheta);
double val=0;
if (dimlesstheta<dimlesstheta_ref)
{
val=(pow(M_PI,(3.0/2.0))*pow(CHARGE_EL, 2)/(pow(2,(3.0/2.0))*C_LIGHT))*sqrt(nu*nu_c)*n_el_MB(el_dens, dimlesstheta, gamma)* Z(nu, nu_c, gamma)*chi( dimlesstheta, gamma)* pow(fabs(Z_sec_der(nu, nu_c, gamma)),(-1.0/2.0));
}
else
{
val=(pow(M_PI,(3.0/2.0))*pow(CHARGE_EL, 2)/(pow(2,(3.0/2.0))*C_LIGHT))*sqrt(nu*nu_c)*n_el_MJ(el_dens, dimlesstheta, gamma)* Z(nu, nu_c, gamma)*chi( dimlesstheta, gamma)* pow(fabs(Z_sec_der(nu, nu_c, gamma)),(-1.0/2.0));
}
return val;
}
double jnu_ph_spect(double nu, void *p)
{
double *params;
params=(double *)p;
double nu_c = params[0];
double dimlesstheta = params[1];
double el_dens = params[2];
return jnu(nu, nu_c, dimlesstheta, el_dens)/(PL_CONST*nu);
}
double blackbody_ph_spect(double nu, void *p)
{
double *params;
params=(double *)p;
double temp = params[0];
return (8*M_PI*nu*nu)/(exp(PL_CONST*nu/(K_B*temp))-1)/(C_LIGHT*C_LIGHT*C_LIGHT);
}
double C(double nu_ph, double nu_c, double gamma_el, double p_el)
{
return ((2.0*pow(gamma_el,2)-1)/(gamma_el*pow(p_el,2)))+2*nu_ph*((gamma_el/pow(p_el,2))-gamma_el*log((gamma_el+1)/p_el))/nu_c;
}
double G(double gamma_el, double p_el)
{
return sqrt(1-2*pow(p_el,2)*(gamma_el*log((gamma_el+1)/p_el)-1));
}
double G_prime(double gamma_el, double p_el)
{
return (3*gamma_el-(3*pow(gamma_el,2)-1)*log((gamma_el+1)/p_el))/G(gamma_el, p_el);
}
double synCrossSection(double el_dens, double T, double nu_ph, double p_el)
{
double b_cr=FINE_STRUCT*sqrt(M_EL*C_LIGHT*C_LIGHT/pow(R_EL,3.0));
double B=calcB(el_dens, T); 
double nu_c=calcCyclotronFreq(B);
double gamma_el=sqrt(p_el*p_el+1);
return (3.0*M_PI*M_PI/8.0)*(THOM_X_SECT/FINE_STRUCT)*(b_cr/B)*pow(nu_c/nu_ph, 2.0) * exp(-2*nu_ph*(gamma_el*log((gamma_el+1)/p_el)-1)/nu_c)* ((C(nu_ph, nu_c, gamma_el, p_el)/G(gamma_el, p_el))-(G_prime(gamma_el, p_el)/pow(G(gamma_el, p_el),2.0)));
}
double calcCyclosynchRLimits(int frame_scatt, int frame_inj, double fps,  double r_inj, char *min_or_max)
{
double val=r_inj;
if (strcmp(min_or_max, "min")==0)
{
val+=(C_LIGHT*(frame_scatt-frame_inj)/fps - 0.5*C_LIGHT/fps);
}
else
{
val+=(C_LIGHT*(frame_scatt-frame_inj)/fps + 0.5*C_LIGHT/fps);
}
return val;
}
int rebinCyclosynchCompPhotons(struct photon **ph_orig, int *num_ph,  int *num_null_ph, int *num_cyclosynch_ph_emit, int *scatt_cyclosynch_num_ph, double **all_time_steps, int **sorted_indexes, int max_photons, double thread_theta_min, double thread_theta_max , gsl_rng * rand, FILE *fPtr)
{
int i=0, j=0, k=0, count=0, count_x=0, count_y=0, count_z=0, count_c_ph=0, end_count=(*scatt_cyclosynch_num_ph), idx=0, num_thread=1;
#if defined(_OPENMP)
num_thread=omp_get_num_threads();
#endif
int synch_comp_photon_count=0, synch_photon_count=0, num_avg=12, num_bins=(CYCLOSYNCHROTRON_REBIN_E_PERC)*max_photons; 
double dtheta_bin=CYCLOSYNCHROTRON_REBIN_ANG*M_PI/180; 
int num_bins_theta=(thread_theta_max-thread_theta_min)/dtheta_bin;
int num_bins_phi=1;
#if DIMENSIONS == THREE
double dphi_bin=CYCLOSYNCHROTRON_REBIN_ANG_PHI; 
double ph_phi=0, temp_phi_max=0, temp_phi_min=DBL_MAX, min_range_phi=0, max_range_phi=0;
#endif
double avg_values[12]={0}; 
double p0_min=DBL_MAX, p0_max=0, log_p0_min=0, log_p0_max=0;
double rand1=0, rand2=0, phi=0, theta=0;
double min_range=0, max_range=0, min_range_theta=0, max_range_theta=0, energy=0;
double ph_r=0, ph_theta=0, temp_theta_max=0, temp_theta_min=DBL_MAX;
int synch_comp_photon_idx[*scatt_cyclosynch_num_ph], total_bins=0;
int num_null_rebin_ph=0, num_in_bin=0;
struct photon *tmp=NULL;
double *tmp_double=NULL;
int *tmp_int=NULL;
double count_weight=0;
fprintf(fPtr, "In the rebin func; num_threads %d scatt_cyclosynch_num_ph %d, num_ph %d\n", num_thread, (*scatt_cyclosynch_num_ph), *num_ph);
fflush(fPtr);
int min_idx=0, max_idx=0;
count=0;
for (i=0;i<*num_ph;i++)
{
if (((*ph_orig)[i].weight != 0) && (((*ph_orig)[i].type == COMPTONIZED_PHOTON) || ((*ph_orig)[i].type == UNABSORBED_CS_PHOTON)) && ((*ph_orig)[i].p0 > 0))
{
if (((*ph_orig)[i].p0< p0_min))
{
p0_min= (*ph_orig)[i].p0;
min_idx=i;
}
if ((*ph_orig)[i].p0> p0_max)
{
p0_max= (*ph_orig)[i].p0;
max_idx=i;
}
ph_r=sqrt(((*ph_orig)[i].r0)*((*ph_orig)[i].r0) + ((*ph_orig)[i].r1)*((*ph_orig)[i].r1) + ((*ph_orig)[i].r2)*((*ph_orig)[i].r2));
ph_theta=acos(((*ph_orig)[i].r2) /ph_r); 
if (ph_theta > temp_theta_max )
{
temp_theta_max=ph_theta;
}
if (ph_theta<temp_theta_min)
{
temp_theta_min=ph_theta;
}
#if DIMENSIONS == THREE
ph_phi=fmod(atan(((*ph_orig)[i].r1)/ ((*ph_orig)[i].r0))*180/M_PI + 360.0,360.0);
if (ph_phi > temp_phi_max )
{
temp_phi_max=ph_phi;
}
if (ph_phi<temp_phi_min)
{
temp_phi_min=ph_phi;
}
#endif
synch_comp_photon_idx[count]=i;
count++;
if ((*ph_orig)[i].type == COMPTONIZED_PHOTON)
{
count_c_ph+=1;
}
}
else if (((*ph_orig)[i].type == CS_POOL_PHOTON) && ((*ph_orig)[i].weight != 0))
{
synch_photon_count++;
}
}
num_bins_theta=ceil((temp_theta_max-temp_theta_min)/dtheta_bin);
#if DIMENSIONS == THREE
num_bins_phi=ceil((temp_phi_max-temp_phi_min)/dphi_bin);
num_avg=13;
#endif
fprintf(fPtr, "Rebin: min, max (keV): %e %e log p0 min, max: %e %e idx: %d %d\n", p0_min*C_LIGHT/1.6e-9,p0_max*C_LIGHT/1.6e-9 , log10(p0_min), log10(p0_max), min_idx, max_idx );
fprintf(fPtr, "Rebin: min, max (theta in deg): %e %e number of bins %d count: %d\n", temp_theta_min*180/M_PI, temp_theta_max*180/M_PI, num_bins_theta, count );
#if DIMENSIONS == THREE
fprintf(fPtr, "Rebin: min, max (phi in deg): %e %e number of bins %d count: %d\n", temp_phi_min, temp_phi_max, num_bins_phi, count );
#endif
fflush(fPtr);
if (count != end_count)
{
end_count=count; 
fprintf(fPtr, "Rebin: not equal to end_count therefore resetting count to be: %d\n", count );
fflush(fPtr);
}
#if DIMENSIONS == THREE
total_bins=num_bins_phi*num_bins_theta*num_bins;
#else
total_bins=num_bins_theta*num_bins;
#endif
if (total_bins>=max_photons)
{
fprintf(fPtr, "The number of rebinned photons, %d, is larger than max_photons %d and will not rebin efficiently. Adjust the parameters such that the number of bins in theta and energy are less than the number of photons that will lead to rebinning.\n",  total_bins, max_photons);
fflush(fPtr);
printf("Rebin: min, max (theta in deg): %e %e number of bins %d count: %d\n", temp_theta_min*180/M_PI, temp_theta_max*180/M_PI, num_bins_theta, count );
#if DIMENSIONS == THREE
printf("Rebin: min, max (phi in deg): %e %e number of bins %d count: %d\n", temp_phi_min, temp_phi_max, num_bins_phi, count );
#endif
printf( "In angle range: %e-%e: The number of rebinned photons, %d, is larger than max_photons %d and will not rebin efficiently. Adjust the parameters such that the number of bins in theta and energy are less than the number of photons that will lead to rebinning.\n",  thread_theta_min*180/M_PI, thread_theta_max*180/M_PI, total_bins, max_photons);
exit(1);
}
struct photon *rebin_ph=malloc(total_bins* sizeof (struct photon ));
struct photon *synch_ph=malloc(synch_photon_count* sizeof (struct photon ));
int synch_photon_idx[synch_photon_count];
gsl_histogram2d * h_energy_theta = gsl_histogram2d_alloc (num_bins, num_bins_theta); 
gsl_histogram2d_set_ranges_uniform (h_energy_theta, log10(p0_min), log10(p0_max*(1+1e-6)), temp_theta_min, temp_theta_max*(1+1e-6));
#if DIMENSIONS == THREE
gsl_histogram2d * h_energy_phi = gsl_histogram2d_alloc (num_bins, num_bins_phi); 
gsl_histogram2d_set_ranges_uniform (h_energy_phi, log10(p0_min), log10(p0_max*(1+1e-6)), temp_phi_min, temp_phi_max*(1+1e-6));
gsl_histogram2d * h_theta_phi = gsl_histogram2d_alloc (num_bins_theta, num_bins_phi); 
gsl_histogram2d_set_ranges_uniform (h_theta_phi, temp_theta_min, temp_theta_max*(1+1e-6), temp_phi_min, temp_phi_max*(1+1e-6));
#endif
count=0;
for (i=0;i<*num_ph;i++)
{
if (((*ph_orig)[i].weight != 0) && (((*ph_orig)[i].type == COMPTONIZED_PHOTON) || ((*ph_orig)[i].type == UNABSORBED_CS_PHOTON)) && ((*ph_orig)[i].p0 > 0))
{
ph_r=sqrt(((*ph_orig)[i].r0)*((*ph_orig)[i].r0) + ((*ph_orig)[i].r1)*((*ph_orig)[i].r1) + ((*ph_orig)[i].r2)*((*ph_orig)[i].r2));
ph_theta=acos(((*ph_orig)[i].r2) /ph_r); 
gsl_histogram2d_increment(h_energy_theta, log10((*ph_orig)[i].p0), ph_theta);
#if DIMENSIONS == THREE
ph_phi=fmod(atan(((*ph_orig)[i].r1)/ ((*ph_orig)[i].r0))*180/M_PI + 360.0,360.0);
gsl_histogram2d_increment(h_energy_phi, log10((*ph_orig)[i].p0), ph_phi);
gsl_histogram2d_increment(h_theta_phi, ph_theta, ph_phi);
#endif
count_weight+=(*ph_orig)[i].weight;
}
if (((*ph_orig)[i].type == CS_POOL_PHOTON) && ((*ph_orig)[i].weight != 0))
{
(synch_ph+count)->p0=(*ph_orig)[i].p0;
(synch_ph+count)->p1=(*ph_orig)[i].p1;
(synch_ph+count)->p2=(*ph_orig)[i].p2;
(synch_ph+count)->p3=(*ph_orig)[i].p3;
(synch_ph+count)->comv_p0=(*ph_orig)[i].comv_p0;
(synch_ph+count)->comv_p1=(*ph_orig)[i].comv_p1;
(synch_ph+count)->comv_p2=(*ph_orig)[i].comv_p2;
(synch_ph+count)->comv_p3=(*ph_orig)[i].comv_p3;
(synch_ph+count)->r0=(*ph_orig)[i].r0;
(synch_ph+count)->r1= (*ph_orig)[i].r1;
(synch_ph+count)->r2=(*ph_orig)[i].r2; 
(synch_ph+count)->s0=(*ph_orig)[i].s0; 
(synch_ph+count)->s1=(*ph_orig)[i].s1;
(synch_ph+count)->s2=(*ph_orig)[i].s2;
(synch_ph+count)->s3=(*ph_orig)[i].s3;
(synch_ph+count)->num_scatt=(*ph_orig)[i].num_scatt;
(synch_ph+count)->weight=(*ph_orig)[i].weight;
(synch_ph+count)->nearest_block_index=(*ph_orig)[i].nearest_block_index; 
synch_photon_idx[count]=i;
count++;
}
}
count_weight=0;
double** avg_values_2d = (double**)malloc(total_bins * sizeof(double*));
for (i = 0; i < total_bins; i++)
avg_values_2d[i] = (double*)malloc((num_avg) * sizeof(double));
for (i = 0; i < total_bins; i++)
for (count=0;count<num_avg;count++)
{
avg_values_2d[i][count]=0;
}
count=0; 
for (i=0;i<*num_ph;i++)
{
if (((*ph_orig)[i].weight != 0) && (((*ph_orig)[i].type == COMPTONIZED_PHOTON) || ((*ph_orig)[i].type == UNABSORBED_CS_PHOTON)) && ((*ph_orig)[i].p0 > 0))
{
ph_r=sqrt(((*ph_orig)[i].r0)*((*ph_orig)[i].r0) + ((*ph_orig)[i].r1)*((*ph_orig)[i].r1) + ((*ph_orig)[i].r2)*((*ph_orig)[i].r2));
ph_theta=acos(((*ph_orig)[i].r2) /ph_r); 
gsl_histogram2d_find(h_energy_theta, log10((*ph_orig)[i].p0), ph_theta, &count_x, &count_y);
#if DIMENSIONS == THREE
ph_phi=fmod(atan(((*ph_orig)[i].r1)/ ((*ph_orig)[i].r0))*180/M_PI + 360.0,360.0);
gsl_histogram2d_find(h_energy_phi, log10((*ph_orig)[i].p0), ph_phi, &count_x, &count_z);
gsl_histogram2d_find(h_theta_phi, ph_theta, ph_phi, &count_y, &count_z);
#endif
count=count_z*num_bins*num_bins_theta+count_x*num_bins_theta+count_y; 
avg_values_2d[count][0] += ph_r*(*ph_orig)[i].weight; 
avg_values_2d[count][1] += ph_theta*(*ph_orig)[i].weight;
avg_values_2d[count][2] += ((atan((*ph_orig)[i].p2/((*ph_orig)[i].p1))*180/M_PI)-(atan(((*ph_orig)[i].r1)/ ((*ph_orig)[i].r0))*180/M_PI))*(*ph_orig)[i].weight;
avg_values_2d[count][3] += (*ph_orig)[i].s0*(*ph_orig)[i].weight;
avg_values_2d[count][4] += (*ph_orig)[i].s1*(*ph_orig)[i].weight;
avg_values_2d[count][5] += (*ph_orig)[i].s2*(*ph_orig)[i].weight;
avg_values_2d[count][6] += (*ph_orig)[i].s3*(*ph_orig)[i].weight;
avg_values_2d[count][7] += (*ph_orig)[i].num_scatt*(*ph_orig)[i].weight;
avg_values_2d[count][8] += (*ph_orig)[i].weight;
{
avg_values_2d[count][9] += fmod(atan2((*ph_orig)[i].p2,((*ph_orig)[i].p1))*180/M_PI + 360.0,360.0) *(*ph_orig)[i].weight;
avg_values_2d[count][10] += (180/M_PI)*acos(((*ph_orig)[i].p3)/((*ph_orig)[i].p0))*(*ph_orig)[i].weight;
}
avg_values_2d[count][11] +=(*ph_orig)[i].p0*(*ph_orig)[i].weight;
#if DIMENSIONS == THREE
avg_values_2d[count][12] += ph_phi*(*ph_orig)[i].weight;
#endif
}
}
for (i=0;i<total_bins;i++)
{
if (avg_values_2d[i][8]==0)
{
(rebin_ph+i)->type = COMPTONIZED_PHOTON;
(rebin_ph+i)->p0=1;
(rebin_ph+i)->p1=0;
(rebin_ph+i)->p2=0;
(rebin_ph+i)->p3=0;
(rebin_ph+i)->comv_p0=0;
(rebin_ph+i)->comv_p1=0;
(rebin_ph+i)->comv_p2=0;
(rebin_ph+i)->comv_p3=0;
(rebin_ph+i)->r0=0;
(rebin_ph+i)->r1= 0;
(rebin_ph+i)->r2=0;
(rebin_ph+i)->s0=1; 
(rebin_ph+i)->s1=0;
(rebin_ph+i)->s2=0;
(rebin_ph+i)->s3=0;
(rebin_ph+i)->num_scatt=0;
(rebin_ph+i)->weight=0;
(rebin_ph+i)->nearest_block_index=-1; 
count_weight+=(rebin_ph+i)->weight;
}
else
{
energy=avg_values_2d[i][11]/avg_values_2d[i][8];
phi=avg_values_2d[i][9]/avg_values_2d[i][8];
theta=avg_values_2d[i][10]/avg_values_2d[i][8];
(rebin_ph+i)->type = COMPTONIZED_PHOTON;
(rebin_ph+i)->p0=energy;
(rebin_ph+i)->p1=energy*sin(theta*M_PI/180)*cos(phi*M_PI/180);
(rebin_ph+i)->p2=energy*sin(theta*M_PI/180)*sin(phi*M_PI/180);
(rebin_ph+i)->p3=energy*cos(theta*M_PI/180);
(rebin_ph+i)->comv_p0=0;
(rebin_ph+i)->comv_p1=0;
(rebin_ph+i)->comv_p2=0;
(rebin_ph+i)->comv_p3=0;
#if DIMENSIONS == THREE
rand1=(M_PI/180)*(avg_values_2d[i][12]/avg_values_2d[i][8]);
#else
rand1=(M_PI/180)*(phi-avg_values_2d[i][2]/avg_values_2d[i][8]);
#endif
(rebin_ph+i)->r0= (avg_values_2d[i][0]/avg_values_2d[i][8])*sin(avg_values_2d[i][1]/avg_values_2d[i][8])*cos(rand1); 
(rebin_ph+i)->r1= (avg_values_2d[i][0]/avg_values_2d[i][8])*sin(avg_values_2d[i][1]/avg_values_2d[i][8])*sin(rand1); 
(rebin_ph+i)->r2= (avg_values_2d[i][0]/avg_values_2d[i][8])*cos(avg_values_2d[i][1]/avg_values_2d[i][8]); 
(rebin_ph+i)->s0=avg_values_2d[i][3]/avg_values_2d[i][8]; 
(rebin_ph+i)->s1=avg_values_2d[i][4]/avg_values_2d[i][8];
(rebin_ph+i)->s2=avg_values_2d[i][5]/avg_values_2d[i][8];
(rebin_ph+i)->s3=avg_values_2d[i][6]/avg_values_2d[i][8];
(rebin_ph+i)->num_scatt=avg_values_2d[i][7]/avg_values_2d[i][8];
(rebin_ph+i)->weight=avg_values_2d[i][8];
(rebin_ph+i)->nearest_block_index=0; 
count_weight+=(rebin_ph+i)->weight;
}
}
for (i = 0; i < total_bins; i++)
free(avg_values_2d[i]);
free(avg_values_2d);
if ((count_c_ph+(*num_null_ph))<total_bins)
{
tmp=realloc(*ph_orig, ((*num_ph)+total_bins-count_c_ph+(*num_null_ph))* sizeof (struct photon )); 
if (tmp != NULL)
{
*ph_orig = tmp;
}
else
{
printf("Error with reserving space to hold old and new photons\n");
exit(1);
}
tmp_double=realloc(*all_time_steps, ((*num_ph)+total_bins-count_c_ph+(*num_null_ph))*sizeof(double));
if (tmp_double!=NULL)
{
*all_time_steps=tmp_double;
}
else
{
printf("Error with reallocating space to hold data about each photon's time step until an interaction occurs\n");
exit(1);
}
tmp_int=realloc(*sorted_indexes, ((*num_ph)+total_bins-count_c_ph+(*num_null_ph))*sizeof(int));
if (tmp_int!=NULL)
{
*sorted_indexes=tmp_int;
}
else
{
printf("Error with reallocating space to hold data about the order in which each photon would have an interaction\n");
exit(1);
}
*num_ph=( *num_ph)+(total_bins-count_c_ph+(*num_null_ph));
end_count=(*scatt_cyclosynch_num_ph)+total_bins-count_c_ph+(*num_null_ph);
count=0;
for (i=0;i<synch_photon_count;i++)
{
{
idx=synch_photon_idx[i];
(*ph_orig)[idx].p0=(synch_ph+count)->p0;
(*ph_orig)[idx].p1=(synch_ph+count)->p1;
(*ph_orig)[idx].p2=(synch_ph+count)->p2;
(*ph_orig)[idx].p3=(synch_ph+count)->p3;
(*ph_orig)[idx].comv_p0=(synch_ph+count)->comv_p0;
(*ph_orig)[idx].comv_p1=(synch_ph+count)->comv_p1;
(*ph_orig)[idx].comv_p2=(synch_ph+count)->comv_p2;
(*ph_orig)[idx].comv_p3=(synch_ph+count)->comv_p3;
(*ph_orig)[idx].r0=(synch_ph+count)->r0;
(*ph_orig)[idx].r1=(synch_ph+count)->r1;
(*ph_orig)[idx].r2=(synch_ph+count)->r2; 
(*ph_orig)[idx].s0=(synch_ph+count)->s0; 
(*ph_orig)[idx].s1=(synch_ph+count)->s1;
(*ph_orig)[idx].s2=(synch_ph+count)->s2;
(*ph_orig)[idx].s3=(synch_ph+count)->s3;
(*ph_orig)[idx].num_scatt=(synch_ph+count)->num_scatt;
(*ph_orig)[idx].weight=(synch_ph+count)->weight;
(*ph_orig)[idx].nearest_block_index=(synch_ph+count)->nearest_block_index;
count++;
}
}
for (i=( *num_ph)-(total_bins-count_c_ph+(*num_null_ph));i<(*num_ph);i++)
{
if ((*ph_orig)[i].type == CS_POOL_PHOTON)
{
(*ph_orig)[i].type = COMPTONIZED_PHOTON;
}
}
}
j=0;
count=0;
i=0;
for (i=0;i<end_count;i++)
{
if (i<(*scatt_cyclosynch_num_ph))
{
idx=synch_comp_photon_idx[i];
}
else
{
idx=i-(*scatt_cyclosynch_num_ph)+( *num_ph)-(total_bins-count_c_ph+(*num_null_ph));
}
if (((*ph_orig)[idx].type == UNABSORBED_CS_PHOTON) || ((*ph_orig)[idx].type == COMPTONIZED_PHOTON))
{
if (count<total_bins)
{
(*ph_orig)[idx].p0=(rebin_ph+count)->p0;
(*ph_orig)[idx].p1=(rebin_ph+count)->p1;
(*ph_orig)[idx].p2=(rebin_ph+count)->p2;
(*ph_orig)[idx].p3=(rebin_ph+count)->p3;
(*ph_orig)[idx].comv_p0=(rebin_ph+count)->comv_p0;
(*ph_orig)[idx].comv_p1=(rebin_ph+count)->comv_p1;
(*ph_orig)[idx].comv_p2=(rebin_ph+count)->comv_p2;
(*ph_orig)[idx].comv_p3=(rebin_ph+count)->comv_p3;
(*ph_orig)[idx].r0=(rebin_ph+count)->r0;
(*ph_orig)[idx].r1=(rebin_ph+count)->r1;
(*ph_orig)[idx].r2=(rebin_ph+count)->r2; 
(*ph_orig)[idx].s0=(rebin_ph+count)->s0; 
(*ph_orig)[idx].s1=(rebin_ph+count)->s1;
(*ph_orig)[idx].s2=(rebin_ph+count)->s2;
(*ph_orig)[idx].s3=(rebin_ph+count)->s3;
(*ph_orig)[idx].num_scatt=(rebin_ph+count)->num_scatt;
(*ph_orig)[idx].weight=(rebin_ph+count)->weight;
(*ph_orig)[idx].nearest_block_index=(rebin_ph+count)->nearest_block_index;
(*ph_orig)[idx].type = COMPTONIZED_PHOTON;
if ((rebin_ph+count)->weight==0)
{
num_null_rebin_ph++;
}
count++;
}
else
{
if ((*ph_orig)[idx].type == UNABSORBED_CS_PHOTON)
{
(*ph_orig)[idx].p0=-1; 
(*ph_orig)[idx].nearest_block_index=-1;
(*ph_orig)[idx].weight=0; 
}
else
{
(*ph_orig)[idx].weight=0;
(*ph_orig)[idx].nearest_block_index=-1;
}
}
}
else if ((*ph_orig)[idx].type != CS_POOL_PHOTON)
{
(*ph_orig)[idx].type = COMPTONIZED_PHOTON;
(*ph_orig)[idx].weight=0;
(*ph_orig)[idx].nearest_block_index=-1;
}
if ((*ph_orig)[idx].weight==0)
{
j++;
}
}
if (count<total_bins)
{
fprintf(fPtr, "There was an issue where MCRaT was not able to save all of the rebinned photons\n");
printf("TThere was an issue where MCRaT was not able to save all of the rebinned photons\n");
fflush(fPtr);
exit(1);
}
*scatt_cyclosynch_num_ph=total_bins-num_null_rebin_ph;
*num_cyclosynch_ph_emit=total_bins+synch_photon_count-num_null_rebin_ph; 
*num_null_ph=j; 
gsl_histogram2d_free (h_energy_theta);
free(rebin_ph);
free(synch_ph);
return num_null_rebin_ph;
}
int photonEmitCyclosynch(struct photon **ph_orig, int *num_ph, int *num_null_ph, double **all_time_steps, int **sorted_indexes, double r_inj, double ph_weight, int maximum_photons, double theta_min, double theta_max, struct hydro_dataframe *hydro_data, gsl_rng *rand, int inject_single_switch, int scatt_ph_index, FILE *fPtr)
{
double rmin=0, rmax=0, max_photons=CYCLOSYNCHROTRON_REBIN_E_PERC*maximum_photons; 
double ph_weight_adjusted=0, position_phi=0;
double dimlesstheta=0, nu_c=0, el_dens=0, error=0, ph_dens_calc=0, max_jnu=0, b_field=0;
double r_grid_innercorner=0, r_grid_outercorner=0, theta_grid_innercorner=0, theta_grid_outercorner=0;
double el_p[4], ph_p_comv[4];
double params[3];
double fr_dum=0.0, y_dum=0.0, yfr_dum=0.0, com_v_phi=0, com_v_theta=0, position_rand=0, position2_rand=0, position3_rand=0, cartesian_position_rand_array[3];
double *p_comv=NULL, *boost=NULL, *l_boost=NULL; 
int status;
int block_cnt=0, i=0, j=0, k=0, null_ph_count=0, *ph_dens=NULL, ph_tot=0, net_ph=0, min_photons=1;
int *null_ph_indexes=NULL;
#if defined(_OPENMP)
int num_thread=omp_get_num_threads();
#endif
int count_null_indexes=0, idx=0;
struct photon *ph_emit=NULL; 
struct photon *tmp=NULL;
double *tmp_double=NULL;
int *tmp_int=NULL, n_pool=0;
gsl_integration_workspace *w = gsl_integration_workspace_alloc (10000);
gsl_function F;
F.function = &blackbody_ph_spect; 
if (inject_single_switch == 0)
{
rmin=calcCyclosynchRLimits(hydro_data->scatt_frame_number,  hydro_data->inj_frame_number, hydro_data->fps,  r_inj, "min");
rmax=calcCyclosynchRLimits(hydro_data->scatt_frame_number,  hydro_data->inj_frame_number, hydro_data->fps,  r_inj, "max");
fprintf(fPtr, "rmin %e rmax %e, theta min/max: %e %e\n", rmin, rmax, theta_min, theta_max);
#pragma omp parallel for num_threads(num_thread) reduction(+:block_cnt)
for(i=0;i<hydro_data->num_elements;i++)
{
#if DIMENSIONS == THREE
hydroCoordinateToSpherical(&r_grid_innercorner, &theta_grid_innercorner, fabs((hydro_data->r0)[i])-0.5*(hydro_data->r0_size)[i], fabs((hydro_data->r1)[i])-0.5*(hydro_data->r1_size)[i], fabs((hydro_data->r2)[i])-0.5*(hydro_data->r2_size)[i]);
hydroCoordinateToSpherical(&r_grid_outercorner, &theta_grid_outercorner, fabs((hydro_data->r0)[i])+0.5*(hydro_data->r0_size)[i], fabs((hydro_data->r1)[i])+0.5*(hydro_data->r1_size)[i], fabs((hydro_data->r2)[i])+0.5*(hydro_data->r2_size)[i]);
#else
hydroCoordinateToSpherical(&r_grid_innercorner, &theta_grid_innercorner, (hydro_data->r0)[i]-0.5*(hydro_data->r0_size)[i], (hydro_data->r1)[i]-0.5*(hydro_data->r1_size)[i], 0);
hydroCoordinateToSpherical(&r_grid_outercorner, &theta_grid_outercorner, (hydro_data->r0)[i]+0.5*(hydro_data->r0_size)[i], (hydro_data->r1)[i]+0.5*(hydro_data->r1_size)[i], 0);
#endif
if ((rmin <= r_grid_outercorner) && (r_grid_innercorner  < rmax ) && (theta_grid_outercorner >= theta_min) && (theta_grid_innercorner < theta_max))
{
block_cnt+=1;
}
}
fprintf(fPtr, "MCRaT has chosen %d hydro elements that it will emit cyclosynchrotron photons into.\n", block_cnt);
fflush(fPtr);
if (block_cnt==0)
{
min_photons=block_cnt; 
}
ph_dens=malloc(block_cnt * sizeof(int));
j=0;
ph_tot=-1;
ph_weight_adjusted=ph_weight;
while ((ph_tot>max_photons) || (ph_tot<min_photons) ) 
{
j=0;
ph_tot=0;
for (i=0;i< hydro_data->num_elements;i++)
{
#if DIMENSIONS == THREE
hydroCoordinateToSpherical(&r_grid_innercorner, &theta_grid_innercorner, fabs((hydro_data->r0)[i])-0.5*(hydro_data->r0_size)[i], fabs((hydro_data->r1)[i])-0.5*(hydro_data->r1_size)[i], fabs((hydro_data->r2)[i])-0.5*(hydro_data->r2_size)[i]);
hydroCoordinateToSpherical(&r_grid_outercorner, &theta_grid_outercorner, fabs((hydro_data->r0)[i])+0.5*(hydro_data->r0_size)[i], fabs((hydro_data->r1)[i])+0.5*(hydro_data->r1_size)[i], fabs((hydro_data->r2)[i])+0.5*(hydro_data->r2_size)[i]);
#else
hydroCoordinateToSpherical(&r_grid_innercorner, &theta_grid_innercorner, (hydro_data->r0)[i]-0.5*(hydro_data->r0_size)[i], (hydro_data->r1)[i]-0.5*(hydro_data->r1_size)[i], 0);
hydroCoordinateToSpherical(&r_grid_outercorner, &theta_grid_outercorner, (hydro_data->r0)[i]+0.5*(hydro_data->r0_size)[i], (hydro_data->r1)[i]+0.5*(hydro_data->r1_size)[i], 0);
#endif
if ((rmin <= r_grid_outercorner) && (r_grid_innercorner  < rmax ) && (theta_grid_outercorner >= theta_min) && (theta_grid_innercorner < theta_max))
{
el_dens= ((hydro_data->dens)[i])/M_P;
#if B_FIELD_CALC == TOTAL_E || B_FIELD_CALC == INTERNAL_E
b_field=calcB(el_dens,(hydro_data->temp)[i]);
#else
#if DIMENSIONS == TWO
b_field=vectorMagnitude((hydro_data->B0)[i], (hydro_data->B1)[i], 0);
#else
b_field=vectorMagnitude((hydro_data->B0)[i], (hydro_data->B1)[i], (hydro_data->B2)[i]);
#endif
#endif
nu_c=calcCyclotronFreq(b_field);
dimlesstheta=calcDimlessTheta( (hydro_data->temp)[i]);
params[0] = (hydro_data->temp)[i]; 
params[1] = dimlesstheta;
params[2] = el_dens;
F.params = &params;
status=gsl_integration_qags(&F, 10, nu_c, 0, 1e-2, 10000, w, &ph_dens_calc, &error); 
ph_dens_calc*=hydroElementVolume(hydro_data, i)/(ph_weight_adjusted);
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
else
{
fprintf(fPtr, "dens: %d, photons: %d, adjusted weight: %e\n", *(ph_dens+(j-1)), ph_tot, ph_weight_adjusted);
fflush(fPtr);
}
}
if (block_cnt!=0)
{
fprintf(fPtr, "Emitting %d cyclosynchrotron photon(s) with weight %e\n", ph_tot,ph_weight_adjusted );
fflush(fPtr);
}
else
{
fprintf(fPtr, "Emitting 0 cyclosynchrotron photons\n" );
fflush(fPtr);
}
}
else
{
ph_tot=1;
}
#pragma omp parallel for num_threads(num_thread) reduction(+:null_ph_count)
for (i=0;i<*num_ph;i++)
{
if (((*ph_orig)[i].weight == 0)) 
{
null_ph_count+=1;
}
}
p_comv=malloc(4*sizeof(double));
boost=malloc(4*sizeof(double));
l_boost=malloc(4*sizeof(double));
if (null_ph_count < ph_tot)
{
tmp=realloc(*ph_orig, ((*num_ph)+ph_tot-null_ph_count)* sizeof (struct photon )); 
if (tmp != NULL)
{
*ph_orig = tmp;
}
else
{
printf("Error with reserving space to hold old and new photons\n");
exit(0);
}
tmp_double=realloc(*all_time_steps, ((*num_ph)+ph_tot-null_ph_count)*sizeof(double));
if (tmp_double!=NULL)
{
*all_time_steps=tmp_double;
}
else
{
printf("Error with reallocating space to hold data about each photon's time step until an interaction occurs\n");
}
tmp_int=realloc(*sorted_indexes, ((*num_ph)+ph_tot-null_ph_count)*sizeof(int));
if (tmp_int!=NULL)
{
*sorted_indexes=tmp_int;
}
else
{
printf("Error with reallocating space to hold data about the order in which each photon would have an interaction\n");
}
net_ph=(ph_tot-null_ph_count);
null_ph_count=ph_tot; 
null_ph_indexes=malloc((ph_tot+null_ph_count)*sizeof(int));
j=0;
for (i=((*num_ph)+net_ph)-1;i >=0 ;i--)
{
if (((*ph_orig)[i].weight == 0)   || (i >= *num_ph))
{
(*ph_orig)[i].weight=0;
(*ph_orig)[i].nearest_block_index=-1;
*(null_ph_indexes+j)=i; 
j++;
}
}
count_null_indexes=ph_tot; 
*num_ph+=net_ph; 
*num_null_ph=ph_tot-null_ph_count; 
}
else
{
null_ph_indexes=malloc(null_ph_count*sizeof(int));
j=0;
for (i=(*num_ph)-1;i>=0;i--)
{
if ((*ph_orig)[i].weight == 0)  
{
*(null_ph_indexes+j)=i;
j++;
if (j == null_ph_count)
{
i=-1; 
}
}
}
count_null_indexes=null_ph_count;
*num_null_ph=null_ph_count-ph_tot;
}
if (inject_single_switch == 0)
{
ph_tot=0;
for (i=0;i< hydro_data->num_elements;i++)
{
#if DIMENSIONS == THREE
hydroCoordinateToSpherical(&r_grid_innercorner, &theta_grid_innercorner, fabs((hydro_data->r0)[i])-0.5*(hydro_data->r0_size)[i], fabs((hydro_data->r1)[i])-0.5*(hydro_data->r1_size)[i], fabs((hydro_data->r2)[i])-0.5*(hydro_data->r2_size)[i]);
hydroCoordinateToSpherical(&r_grid_outercorner, &theta_grid_outercorner, fabs((hydro_data->r0)[i])+0.5*(hydro_data->r0_size)[i], fabs((hydro_data->r1)[i])+0.5*(hydro_data->r1_size)[i], fabs((hydro_data->r2)[i])+0.5*(hydro_data->r2_size)[i]);
#else
hydroCoordinateToSpherical(&r_grid_innercorner, &theta_grid_innercorner, (hydro_data->r0)[i]-0.5*(hydro_data->r0_size)[i], (hydro_data->r1)[i]-0.5*(hydro_data->r1_size)[i], 0);
hydroCoordinateToSpherical(&r_grid_outercorner, &theta_grid_outercorner, (hydro_data->r0)[i]+0.5*(hydro_data->r0_size)[i], (hydro_data->r1)[i]+0.5*(hydro_data->r1_size)[i], 0);
#endif
if ((rmin <= r_grid_outercorner) && (r_grid_innercorner  < rmax ) && (theta_grid_outercorner >= theta_min) && (theta_grid_innercorner < theta_max))
{
el_dens= ((hydro_data->dens)[i])/M_P;
#if B_FIELD_CALC == TOTAL_E || B_FIELD_CALC == INTERNAL_E
b_field=calcB(el_dens,(hydro_data->temp)[i]);
#else
#if DIMENSIONS == TWO
b_field=vectorMagnitude((hydro_data->B0)[i], (hydro_data->B1)[i], 0);
#else
b_field=vectorMagnitude((hydro_data->B0)[i], (hydro_data->B1)[i], (hydro_data->B2)[i]);
#endif
#endif
nu_c=calcCyclotronFreq(b_field);
dimlesstheta=calcDimlessTheta( (hydro_data->temp)[i]);
for(j=0;j<( *(ph_dens+k) ); j++ )
{
fr_dum=nu_c; 
#if DIMENSIONS == TWO || DIMENSIONS == TWO_POINT_FIVE
position_phi=gsl_rng_uniform(rand)*2*M_PI;
#else
position_phi=0;
#endif
com_v_phi=gsl_rng_uniform(rand)*2*M_PI;
com_v_theta=gsl_rng_uniform(rand)*M_PI; 
*(p_comv+0)=PL_CONST*fr_dum/C_LIGHT;
*(p_comv+1)=(PL_CONST*fr_dum/C_LIGHT)*sin(com_v_theta)*cos(com_v_phi);
*(p_comv+2)=(PL_CONST*fr_dum/C_LIGHT)*sin(com_v_theta)*sin(com_v_phi);
*(p_comv+3)=(PL_CONST*fr_dum/C_LIGHT)*cos(com_v_theta);
#if DIMENSIONS == THREE
hydroVectorToCartesian(boost, (hydro_data->v0)[i], (hydro_data->v1)[i], (hydro_data->v2)[i], (hydro_data->r0)[i], (hydro_data->r1)[i], (hydro_data->r2)[i]);
#elif DIMENSIONS == TWO_POINT_FIVE
hydroVectorToCartesian(boost, (hydro_data->v0)[i], (hydro_data->v1)[i], (hydro_data->v2)[i], (hydro_data->r0)[i], (hydro_data->r1)[i], position_phi);
#else
hydroVectorToCartesian(boost, (hydro_data->v0)[i], (hydro_data->v1)[i], 0, (hydro_data->r0)[i], (hydro_data->r1)[i], position_phi);
#endif
(*(boost+0))*=-1;
(*(boost+1))*=-1;
(*(boost+2))*=-1;
lorentzBoost(boost, p_comv, l_boost, 'p', fPtr);
idx=(*(null_ph_indexes+count_null_indexes-1));
(*ph_orig)[idx].p0=(*(l_boost+0));
(*ph_orig)[idx].p1=(*(l_boost+1));
(*ph_orig)[idx].p2=(*(l_boost+2));
(*ph_orig)[idx].p3=(*(l_boost+3));
(*ph_orig)[idx].comv_p0=(*(p_comv+0));
(*ph_orig)[idx].comv_p1=(*(p_comv+1));
(*ph_orig)[idx].comv_p2=(*(p_comv+2));
(*ph_orig)[idx].comv_p3=(*(p_comv+3));
#if DIMENSIONS == THREE
hydroCoordinateToMcratCoordinate(&cartesian_position_rand_array, (hydro_data->r0)[i], (hydro_data->r1)[i], (hydro_data->r2)[i]);
#else
hydroCoordinateToMcratCoordinate(&cartesian_position_rand_array, (hydro_data->r0)[i], (hydro_data->r1)[i], position_phi);
#endif
(*ph_orig)[idx].r0= cartesian_position_rand_array[0]; 
(*ph_orig)[idx].r1= cartesian_position_rand_array[1] ;
(*ph_orig)[idx].r2= cartesian_position_rand_array[2]; 
(*ph_orig)[idx].s0=1; 
(*ph_orig)[idx].s1=0;
(*ph_orig)[idx].s2=0;
(*ph_orig)[idx].s3=0;
(*ph_orig)[idx].num_scatt=0;
(*ph_orig)[idx].weight=ph_weight_adjusted;
(*ph_orig)[idx].nearest_block_index=0; 
(*ph_orig)[idx].type=CS_POOL_PHOTON;
ph_tot++; 
count_null_indexes--; 
if ((count_null_indexes == 0) || (ph_tot == null_ph_count))
{
i= hydro_data->num_elements;
}
}
k++;
}
}
}
else
{
idx=(*(null_ph_indexes+count_null_indexes-1));
i=(*ph_orig)[scatt_ph_index].nearest_block_index;
el_dens= ((hydro_data->dens)[i])/M_P;
#if B_FIELD_CALC == TOTAL_E || B_FIELD_CALC == INTERNAL_E
b_field=calcB(el_dens,(hydro_data->temp)[i]);
#else
#if DIMENSIONS == TWO
b_field=vectorMagnitude((hydro_data->B0)[i], (hydro_data->B1)[i], 0);
#else
b_field=vectorMagnitude((hydro_data->B0)[i], (hydro_data->B1)[i], (hydro_data->B2)[i]);
#endif
#endif
nu_c=calcCyclotronFreq(b_field);
fr_dum=nu_c; 
#if DIMENSIONS == TWO || DIMENSIONS == TWO_POINT_FIVE
position_phi=gsl_rng_uniform(rand)*2*M_PI;
#else
position_phi=0;
#endif
com_v_phi=gsl_rng_uniform(rand)*2*M_PI;
com_v_theta=gsl_rng_uniform(rand)*M_PI; 
*(p_comv+0)=PL_CONST*fr_dum/C_LIGHT;
*(p_comv+1)=(PL_CONST*fr_dum/C_LIGHT)*sin(com_v_theta)*cos(com_v_phi);
*(p_comv+2)=(PL_CONST*fr_dum/C_LIGHT)*sin(com_v_theta)*sin(com_v_phi);
*(p_comv+3)=(PL_CONST*fr_dum/C_LIGHT)*cos(com_v_theta);
#if DIMENSIONS == THREE
hydroVectorToCartesian(boost, (hydro_data->v0)[i], (hydro_data->v1)[i], (hydro_data->v2)[i], (hydro_data->r0)[i], (hydro_data->r1)[i], (hydro_data->r2)[i]);
#elif DIMENSIONS == TWO_POINT_FIVE
hydroVectorToCartesian(boost, (hydro_data->v0)[i], (hydro_data->v1)[i], (hydro_data->v2)[i], (hydro_data->r0)[i], (hydro_data->r1)[i], position_phi);
#else
hydroVectorToCartesian(boost, (hydro_data->v0)[i], (hydro_data->v1)[i], 0, (hydro_data->r0)[i], (hydro_data->r1)[i], position_phi);
#endif
(*(boost+0))*=-1;
(*(boost+1))*=-1;
(*(boost+2))*=-1;
lorentzBoost(boost, p_comv, l_boost, 'p', fPtr);
(*ph_orig)[idx].p0=(*(l_boost+0));
(*ph_orig)[idx].p1=(*(l_boost+1));
(*ph_orig)[idx].p2=(*(l_boost+2));
(*ph_orig)[idx].p3=(*(l_boost+3));
(*ph_orig)[idx].comv_p0=(*(p_comv+0));
(*ph_orig)[idx].comv_p1=(*(p_comv+1));
(*ph_orig)[idx].comv_p2=(*(p_comv+2));
(*ph_orig)[idx].comv_p3=(*(p_comv+3));
#if DIMENSIONS == THREE
hydroCoordinateToMcratCoordinate(&cartesian_position_rand_array, (hydro_data->r0)[i], (hydro_data->r1)[i], (hydro_data->r2)[i]);
#else
hydroCoordinateToMcratCoordinate(&cartesian_position_rand_array, (hydro_data->r0)[i], (hydro_data->r1)[i], position_phi);
#endif
(*ph_orig)[idx].r0= cartesian_position_rand_array[0]; 
(*ph_orig)[idx].r1= cartesian_position_rand_array[1] ;
(*ph_orig)[idx].r2= cartesian_position_rand_array[2]; 
(*ph_orig)[idx].s0=1; 
(*ph_orig)[idx].s1=0;
(*ph_orig)[idx].s2=0;
(*ph_orig)[idx].s3=0;
(*ph_orig)[idx].num_scatt=0;
(*ph_orig)[idx].weight=(*ph_orig)[scatt_ph_index].weight;
(*ph_orig)[idx].nearest_block_index=i; 
(*ph_orig)[idx].type=CS_POOL_PHOTON;
position_rand=gsl_rng_uniform_pos(rand)*((hydro_data->r0_size)[i])-((hydro_data->r0_size)[i])/2.0; 
position2_rand=gsl_rng_uniform_pos(rand)*((hydro_data->r1_size)[i])-((hydro_data->r1_size)[i])/2.0;
#if DIMENSIONS == THREE
position3_rand=gsl_rng_uniform_pos(rand)*((hydro_data->r2_size)[i])-((hydro_data->r2_size)[i])/2.0;
hydroCoordinateToMcratCoordinate(&cartesian_position_rand_array, (hydro_data->r0)[i]+position_rand, (hydro_data->r1)[i]+position2_rand, (hydro_data->r2)[i]+position3_rand);
#else
hydroCoordinateToMcratCoordinate(&cartesian_position_rand_array, (hydro_data->r0)[i]+position_rand, (hydro_data->r1)[i]+position2_rand, position_phi);
#endif
(*ph_orig)[scatt_ph_index].r0=cartesian_position_rand_array[0];
(*ph_orig)[scatt_ph_index].r1=cartesian_position_rand_array[1];
(*ph_orig)[scatt_ph_index].r2=cartesian_position_rand_array[2];
}
free(null_ph_indexes);
free(ph_dens); free(p_comv); free(boost); free(l_boost);
gsl_integration_workspace_free (w);
return ph_tot;
}
double phAbsCyclosynch(struct photon **ph_orig, int *num_ph, int *num_abs_ph, int *scatt_cyclosynch_num_ph, struct hydro_dataframe *hydro_data, FILE *fPtr) 
{
int i=0, count=0, abs_ph_count=0, synch_ph_count=0, num_thread=1;
int other_count=0;
#if defined(_OPENMP)
num_thread=omp_get_num_threads();
#endif
double el_dens=0, nu_c=0, abs_count=0, b_field=0;
fprintf(fPtr, "In phAbsCyclosynch func begin: abs_ph_count: %d synch_ph_count: %d scatt_cyclosynch_num_ph: %d num_threads: %d\n", abs_ph_count, synch_ph_count, *scatt_cyclosynch_num_ph, num_thread);
*scatt_cyclosynch_num_ph=0;
#pragma omp parallel for num_threads(num_thread) firstprivate(b_field, el_dens, nu_c) reduction(+:abs_ph_count)
for (i=0;i<*num_ph;i++)
{
if (((*ph_orig)[i].weight != 0) && ((*ph_orig)[i].nearest_block_index != -1))
{
el_dens= (hydro_data->dens)[(*ph_orig)[i].nearest_block_index]/M_P;
#if B_FIELD_CALC == TOTAL_E || B_FIELD_CALC == INTERNAL_E
b_field=calcB(el_dens,(hydro_data->temp)[(*ph_orig)[i].nearest_block_index]);
#else
#if DIMENSIONS == TWO
b_field=vectorMagnitude((hydro_data->B0)[(*ph_orig)[i].nearest_block_index], (hydro_data->B1)[(*ph_orig)[i].nearest_block_index], 0);
#else
b_field=vectorMagnitude((hydro_data->B0)[(*ph_orig)[i].nearest_block_index], (hydro_data->B1)[(*ph_orig)[i].nearest_block_index], (hydro_data->B2)[(*ph_orig)[i].nearest_block_index]);
#endif
#endif
nu_c=calcCyclotronFreq(b_field);
if (((*ph_orig)[i].comv_p0*C_LIGHT/PL_CONST <= nu_c) || ((*ph_orig)[i].type == CS_POOL_PHOTON))
{
if (((*ph_orig)[i].type != INJECTED_PHOTON) && ((*ph_orig)[i].type != UNABSORBED_CS_PHOTON) )
{
(*ph_orig)[i].weight=0;
(*ph_orig)[i].nearest_block_index=-1;
abs_ph_count++;
if ((*ph_orig)[i].type == CS_POOL_PHOTON)
{
synch_ph_count++;
}
}
else
{
abs_count+=(*ph_orig)[i].weight;
(*ph_orig)[i].p0=-1; 
(*ph_orig)[i].nearest_block_index=-1;
(*ph_orig)[i].weight=0;
abs_ph_count++;
}
}
else
{
(*ph_orig)[count].p0=(*ph_orig)[i].p0;
(*ph_orig)[count].p1=(*ph_orig)[i].p1;
(*ph_orig)[count].p2=(*ph_orig)[i].p2;
(*ph_orig)[count].p3=(*ph_orig)[i].p3;
(*ph_orig)[count].comv_p0=(*ph_orig)[i].comv_p0;
(*ph_orig)[count].comv_p1=(*ph_orig)[i].comv_p1;
(*ph_orig)[count].comv_p2=(*ph_orig)[i].comv_p2;
(*ph_orig)[count].comv_p3=(*ph_orig)[i].comv_p3;
(*ph_orig)[count].r0= (*ph_orig)[i].r0;
(*ph_orig)[count].r1=(*ph_orig)[i].r1 ;
(*ph_orig)[count].r2=(*ph_orig)[i].r2;
(*ph_orig)[count].s0=(*ph_orig)[i].s0;
(*ph_orig)[count].s1=(*ph_orig)[i].s1;
(*ph_orig)[count].s2=(*ph_orig)[i].s2;
(*ph_orig)[count].s3=(*ph_orig)[i].s3;
(*ph_orig)[count].num_scatt=(*ph_orig)[i].num_scatt;
(*ph_orig)[count].weight=(*ph_orig)[i].weight;
(*ph_orig)[count].nearest_block_index=(*ph_orig)[i].nearest_block_index;
(*ph_orig)[count].type=(*ph_orig)[i].type;
count+=1;
if (((*ph_orig)[i].type == COMPTONIZED_PHOTON) || ((*ph_orig)[i].type == UNABSORBED_CS_PHOTON) )
{
*scatt_cyclosynch_num_ph+=1;
}
}
}
else
{
if (((*ph_orig)[i].p0 < 0) )
{
(*ph_orig)[count].p0=(*ph_orig)[i].p0;
(*ph_orig)[count].p1=(*ph_orig)[i].p1;
(*ph_orig)[count].p2=(*ph_orig)[i].p2;
(*ph_orig)[count].p3=(*ph_orig)[i].p3;
(*ph_orig)[count].comv_p0=(*ph_orig)[i].comv_p0;
(*ph_orig)[count].comv_p1=(*ph_orig)[i].comv_p1;
(*ph_orig)[count].comv_p2=(*ph_orig)[i].comv_p2;
(*ph_orig)[count].comv_p3=(*ph_orig)[i].comv_p3;
(*ph_orig)[count].r0= (*ph_orig)[i].r0;
(*ph_orig)[count].r1=(*ph_orig)[i].r1 ;
(*ph_orig)[count].r2=(*ph_orig)[i].r2;
(*ph_orig)[count].s0=(*ph_orig)[i].s0;
(*ph_orig)[count].s1=(*ph_orig)[i].s1;
(*ph_orig)[count].s2=(*ph_orig)[i].s2;
(*ph_orig)[count].s3=(*ph_orig)[i].s3;
(*ph_orig)[count].num_scatt=(*ph_orig)[i].num_scatt;
(*ph_orig)[count].weight=(*ph_orig)[i].weight;
(*ph_orig)[count].nearest_block_index=(*ph_orig)[i].nearest_block_index;
(*ph_orig)[count].type=(*ph_orig)[i].type;
count+=1;
}
}
}
*num_abs_ph=abs_ph_count; 
while (count<*num_ph)
{
(*ph_orig)[count].weight=0;
(*ph_orig)[count].nearest_block_index=-1;
count+=1;
}
return abs_count;
}
