#include "mcrat.h"
const double A_RAD=7.56e-15, C_LIGHT=2.99792458e10, PL_CONST=6.6260755e-27, FINE_STRUCT=7.29735308e-3, CHARGE_EL= 4.8032068e-10;
const double K_B=1.380658e-16, M_P=1.6726231e-24, THOM_X_SECT=6.65246e-25, M_EL=9.1093879e-28 , R_EL=2.817941499892705e-13;
void photonInjection(struct photon **ph, int *ph_num, double r_inj, double ph_weight, int min_photons, int max_photons, char spect, double theta_min, double theta_max, struct hydro_dataframe *hydro_data, gsl_rng * rand, FILE *fPtr)
{
int i=0, block_cnt=0, *ph_dens=NULL, ph_tot=0, j=0,k=0;
double ph_dens_calc=0.0, fr_dum=0.0, y_dum=0.0, yfr_dum=0.0, fr_max=0, bb_norm=0, position_phi, ph_weight_adjusted, rmin, rmax;
double com_v_phi, com_v_theta, *p_comv=NULL, *boost=NULL; 
double *l_boost=NULL; 
float num_dens_coeff;
double r_grid_innercorner=0, r_grid_outercorner=0, theta_grid_innercorner=0, theta_grid_outercorner=0;
double position_rand=0, position2_rand=0, position3_rand=0, cartesian_position_rand_array[3];
if (spect=='w') 
{
num_dens_coeff=8.44;
}
else
{
num_dens_coeff=20.29; 
}
rmin=r_inj - 0.5*C_LIGHT/hydro_data->fps;
rmax=r_inj + 0.5*C_LIGHT/hydro_data->fps;
for(i=0; i<hydro_data->num_elements; i++)
{
#if DIMENSIONS == THREE
hydroCoordinateToSpherical(&r_grid_innercorner, &theta_grid_innercorner, fabs((hydro_data->r0)[i])-0.5*(hydro_data->r0_size)[i], fabs((hydro_data->r1)[i])-0.5*(hydro_data->r1_size)[i], fabs((hydro_data->r2)[i])-0.5*(hydro_data->r2_size)[i]);
hydroCoordinateToSpherical(&r_grid_outercorner, &theta_grid_outercorner, fabs((hydro_data->r0)[i])+0.5*(hydro_data->r0_size)[i], fabs((hydro_data->r1)[i])+0.5*(hydro_data->r1_size)[i], fabs((hydro_data->r2)[i])+0.5*(hydro_data->r2_size)[i]);
#else
hydroCoordinateToSpherical(&r_grid_innercorner, &theta_grid_innercorner, (hydro_data->r0)[i]-0.5*(hydro_data->r0_size)[i], (hydro_data->r1)[i]-0.5*(hydro_data->r1_size)[i], 0);
hydroCoordinateToSpherical(&r_grid_outercorner, &theta_grid_outercorner, (hydro_data->r0)[i]+0.5*(hydro_data->r0_size)[i], (hydro_data->r1)[i]+0.5*(hydro_data->r1_size)[i], 0);
#endif
if ((rmin <= r_grid_outercorner) && (r_grid_innercorner  <= rmax ) && (theta_grid_outercorner >= theta_min) && (theta_grid_innercorner <= theta_max))
{
block_cnt++;
}
}
ph_dens=malloc(block_cnt * sizeof(int));
j=0;
ph_tot=0;
ph_weight_adjusted=ph_weight;
while ((ph_tot>max_photons) || (ph_tot<min_photons) )
{
j=0;
ph_tot=0;
for (i=0;i<hydro_data->num_elements;i++)
{
#if DIMENSIONS == THREE
hydroCoordinateToSpherical(&r_grid_innercorner, &theta_grid_innercorner, fabs((hydro_data->r0)[i])-0.5*(hydro_data->r0_size)[i], fabs((hydro_data->r1)[i])-0.5*(hydro_data->r1_size)[i], fabs((hydro_data->r2)[i])-0.5*(hydro_data->r2_size)[i]);
hydroCoordinateToSpherical(&r_grid_outercorner, &theta_grid_outercorner, fabs((hydro_data->r0)[i])+0.5*(hydro_data->r0_size)[i], fabs((hydro_data->r1)[i])+0.5*(hydro_data->r1_size)[i], fabs((hydro_data->r2)[i])+0.5*(hydro_data->r2_size)[i]);
#else
hydroCoordinateToSpherical(&r_grid_innercorner, &theta_grid_innercorner, (hydro_data->r0)[i]-0.5*(hydro_data->r0_size)[i], (hydro_data->r1)[i]-0.5*(hydro_data->r1_size)[i], 0);
hydroCoordinateToSpherical(&r_grid_outercorner, &theta_grid_outercorner, (hydro_data->r0)[i]+0.5*(hydro_data->r0_size)[i], (hydro_data->r1)[i]+0.5*(hydro_data->r1_size)[i], 0);
#endif
if ((rmin <= r_grid_outercorner) && (r_grid_innercorner  <= rmax ) && (theta_grid_outercorner >= theta_min) && (theta_grid_innercorner <= theta_max))
{
ph_dens_calc=(4.0/3.0)*hydroElementVolume(hydro_data, i) *(((hydro_data->gamma)[i]*num_dens_coeff*(hydro_data->temp)[i]*(hydro_data->temp)[i]*(hydro_data->temp)[i])/ph_weight_adjusted); 
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
(*ph)=malloc (ph_tot * sizeof (struct photon ));
p_comv=malloc(4*sizeof(double));
boost=malloc(3*sizeof(double));
l_boost=malloc(4*sizeof(double));
ph_tot=0;
k=0;
double test=0, test_rand1=gsl_rng_uniform_pos(rand), test_rand2=gsl_rng_uniform_pos(rand), test_rand3=gsl_rng_uniform_pos(rand), test_rand4=gsl_rng_uniform_pos(rand), test_rand5=gsl_rng_uniform_pos(rand);
double test_cnt=0;
for (i=0;i<hydro_data->num_elements;i++)
{
#if DIMENSIONS == THREE
hydroCoordinateToSpherical(&r_grid_innercorner, &theta_grid_innercorner, fabs((hydro_data->r0)[i])-0.5*(hydro_data->r0_size)[i], fabs((hydro_data->r1)[i])-0.5*(hydro_data->r1_size)[i], fabs((hydro_data->r2)[i])-0.5*(hydro_data->r2_size)[i]);
hydroCoordinateToSpherical(&r_grid_outercorner, &theta_grid_outercorner, fabs((hydro_data->r0)[i])+0.5*(hydro_data->r0_size)[i], fabs((hydro_data->r1)[i])+0.5*(hydro_data->r1_size)[i], fabs((hydro_data->r2)[i])+0.5*(hydro_data->r2_size)[i]);
#else
hydroCoordinateToSpherical(&r_grid_innercorner, &theta_grid_innercorner, (hydro_data->r0)[i]-0.5*(hydro_data->r0_size)[i], (hydro_data->r1)[i]-0.5*(hydro_data->r1_size)[i], 0);
hydroCoordinateToSpherical(&r_grid_outercorner, &theta_grid_outercorner, (hydro_data->r0)[i]+0.5*(hydro_data->r0_size)[i], (hydro_data->r1)[i]+0.5*(hydro_data->r1_size)[i], 0);
#endif
if ((rmin <= r_grid_outercorner) && (r_grid_innercorner  <= rmax ) && (theta_grid_outercorner >= theta_min) && (theta_grid_innercorner <= theta_max))
{
for(j=0;j<( *(ph_dens+k) ); j++ )
{
if (spect=='w')
{
y_dum=1; 
yfr_dum=0;
while (y_dum>yfr_dum)
{
fr_dum=gsl_rng_uniform_pos(rand)*6.3e11*((hydro_data->temp)[i]); 
y_dum=gsl_rng_uniform_pos(rand);
yfr_dum=(1.0/(1.29e31))*pow((fr_dum/((hydro_data->temp)[i])),3.0)/(exp((PL_CONST*fr_dum)/(K_B*((hydro_data->temp)[i]) ))-1); 
}
}
else
{
test=0;
test_rand1=gsl_rng_uniform_pos(rand);
test_rand2=gsl_rng_uniform_pos(rand);
test_rand3=gsl_rng_uniform_pos(rand);
test_rand4=gsl_rng_uniform_pos(rand);
test_rand5=gsl_rng_uniform_pos(rand);
test_cnt=0;
while (test<M_PI*M_PI*M_PI*M_PI*test_rand1/90.0)
{
test_cnt+=1;
test+=1/(test_cnt*test_cnt*test_cnt*test_cnt);
}
fr_dum=-log(test_rand2*test_rand3*test_rand4*test_rand5)/test_cnt;
fr_dum*=K_B*((hydro_data->temp)[i])/PL_CONST;
y_dum=0; yfr_dum=1;
}
#if DIMENSIONS == TWO || DIMENSIONS == TWO_POINT_FIVE
position_phi=gsl_rng_uniform(rand)*2*M_PI;
#else
position_phi=0;
#endif
com_v_phi=gsl_rng_uniform(rand)*2*M_PI;
com_v_theta=acos((gsl_rng_uniform(rand)*2)-1);
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
(*ph)[ph_tot].p0=(*(l_boost+0));
(*ph)[ph_tot].p1=(*(l_boost+1));
(*ph)[ph_tot].p2=(*(l_boost+2));
(*ph)[ph_tot].p3=(*(l_boost+3));
(*ph)[ph_tot].comv_p0=(*(p_comv+0));
(*ph)[ph_tot].comv_p1=(*(p_comv+1));
(*ph)[ph_tot].comv_p2=(*(p_comv+2));
(*ph)[ph_tot].comv_p3=(*(p_comv+3));
position_rand=gsl_rng_uniform_pos(rand)*((hydro_data->r0_size)[i])-0.5*((hydro_data->r0_size)[i]); 
position2_rand=gsl_rng_uniform_pos(rand)*((hydro_data->r1_size)[i])-0.5*((hydro_data->r1_size)[i]);
#if DIMENSIONS == THREE
position3_rand=gsl_rng_uniform_pos(rand)*((hydro_data->r2_size)[i])-0.5*((hydro_data->r2_size)[i]);
hydroCoordinateToMcratCoordinate(&cartesian_position_rand_array, (hydro_data->r0)[i]+position_rand, (hydro_data->r1)[i]+position2_rand, (hydro_data->r2)[i]+position3_rand);
#else
hydroCoordinateToMcratCoordinate(&cartesian_position_rand_array, (hydro_data->r0)[i]+position_rand, (hydro_data->r1)[i]+position2_rand, position_phi);
#endif
(*ph)[ph_tot].r0=cartesian_position_rand_array[0];
(*ph)[ph_tot].r1=cartesian_position_rand_array[1];
(*ph)[ph_tot].r2=cartesian_position_rand_array[2];
(*ph)[ph_tot].s0=1; 
(*ph)[ph_tot].s1=0;
(*ph)[ph_tot].s2=0;
(*ph)[ph_tot].s3=0;
(*ph)[ph_tot].num_scatt=0;
(*ph)[ph_tot].weight=ph_weight_adjusted;
(*ph)[ph_tot].nearest_block_index=0;
(*ph)[ph_tot].type=INJECTED_PHOTON; 
ph_tot++;
}
k++;
}
}
*ph_num=ph_tot; 
free(ph_dens); free(p_comv);free(boost); free(l_boost);
}
void lorentzBoost(double *boost, double *p_ph, double *result, char object,  FILE *fPtr)
{
double beta=0, gamma=0, *boosted_p=NULL;
gsl_vector_view b=gsl_vector_view_array(boost, 3); 
gsl_vector_view p=gsl_vector_view_array(p_ph, 4); 
gsl_matrix *lambda1= gsl_matrix_calloc (4, 4); 
gsl_vector *p_ph_prime =gsl_vector_calloc(4); 
if (gsl_blas_dnrm2(&b.vector) > 0)
{
beta=gsl_blas_dnrm2(&b.vector);
gamma=1.0/sqrt(1-beta*beta);
gsl_matrix_set(lambda1, 0,0, gamma);
gsl_matrix_set(lambda1, 0,1,  -1*gsl_vector_get(&b.vector,0)*gamma);
gsl_matrix_set(lambda1, 0,2,  -1*gsl_vector_get(&b.vector,1)*gamma);
gsl_matrix_set(lambda1, 0,3,  -1*gsl_vector_get(&b.vector,2)*gamma);
gsl_matrix_set(lambda1, 1,1,  1+((gamma-1)*(gsl_vector_get(&b.vector,0)*gsl_vector_get(&b.vector,0))/(beta*beta) ) );
gsl_matrix_set(lambda1, 1,2,  ((gamma-1)*(gsl_vector_get(&b.vector,0)*  gsl_vector_get(&b.vector,1)/(beta*beta) ) ));
gsl_matrix_set(lambda1, 1,3,  ((gamma-1)*(gsl_vector_get(&b.vector,0)*  gsl_vector_get(&b.vector,2)/(beta*beta) ) ));
gsl_matrix_set(lambda1, 2,2,  1+((gamma-1)*(gsl_vector_get(&b.vector,1)*gsl_vector_get(&b.vector,1))/(beta*beta) ) );
gsl_matrix_set(lambda1, 2,3,  ((gamma-1)*(gsl_vector_get(&b.vector,1)*  gsl_vector_get(&b.vector,2))/(beta*beta) ) );
gsl_matrix_set(lambda1, 3,3,  1+((gamma-1)*(gsl_vector_get(&b.vector,2)*gsl_vector_get(&b.vector,2))/(beta*beta) ) );
gsl_matrix_set(lambda1, 1,0, gsl_matrix_get(lambda1,0,1));
gsl_matrix_set(lambda1, 2,0, gsl_matrix_get(lambda1,0,2));
gsl_matrix_set(lambda1, 3,0, gsl_matrix_get(lambda1,0,3));
gsl_matrix_set(lambda1, 2,1, gsl_matrix_get(lambda1,1,2));
gsl_matrix_set(lambda1, 3,1, gsl_matrix_get(lambda1,1,3));
gsl_matrix_set(lambda1, 3,2, gsl_matrix_get(lambda1,2,3));
gsl_blas_dgemv(CblasNoTrans, 1, lambda1, &p.vector, 0, p_ph_prime );
if (object == 'p')
{
boosted_p=zeroNorm(gsl_vector_ptr(p_ph_prime, 0));
}
else
{
boosted_p=gsl_vector_ptr(p_ph_prime, 0);
}
}
else
{
if (object=='p')
{
boosted_p=zeroNorm(p_ph);
}
else
{
boosted_p=gsl_vector_ptr(&p.vector, 0);
}
}
*(result+0)=*(boosted_p+0);
*(result+1)=*(boosted_p+1);
*(result+2)=*(boosted_p+2);
*(result+3)=*(boosted_p+3);
gsl_matrix_free (lambda1); gsl_vector_free(p_ph_prime);
}
double *zeroNorm(double *p_ph)
{
int i=0;
double normalizing_factor=0;
gsl_vector_view p=gsl_vector_view_array((p_ph+1), 3); 
if (*(p_ph+0) != gsl_blas_dnrm2(&p.vector ) )
{
normalizing_factor=(gsl_blas_dnrm2(&p.vector ));
*(p_ph+1)= ((*(p_ph+1))/(normalizing_factor))*(*(p_ph+0));
*(p_ph+2)= ((*(p_ph+2))/(normalizing_factor))*(*(p_ph+0));
*(p_ph+3)= ((*(p_ph+3))/(normalizing_factor))*(*(p_ph+0));
}
return p_ph;
}
int findNearestPropertiesAndMinMFP( struct photon *ph, int num_ph, double *all_time_steps, int *sorted_indexes, struct hydro_dataframe *hydro_data, gsl_rng * rand, int find_nearest_block_switch, FILE *fPtr)
{
int i=0, min_index=0, ph_block_index=0, num_thread=1, thread_id=0;
double ph_x=0, ph_y=0, ph_phi=0, ph_z=0, ph_r=0, ph_theta=0;
double fl_v_x=0, fl_v_y=0, fl_v_z=0; 
double ph_v_norm=0, fl_v_norm=0, synch_x_sect=0;
double n_cosangle=0, n_dens_lab_tmp=0,n_vx_tmp=0, n_vy_tmp=0, n_vz_tmp=0, n_temp_tmp=0 ;
double rnd_tracker=0, n_dens_min=0, n_vx_min=0, n_vy_min=0, n_vz_min=0, n_temp_min=0;
#if defined(_OPENMP)
num_thread=omp_get_num_threads(); 
#endif
bool is_in_block=0; 
int index=0, num_photons_find_new_element=0;
double mfp=0,min_mfp=0, beta=0;
double el_p[4];
double ph_p_comv[4], ph_p[4], fluid_beta[3], photon_hydro_coord[3];
const gsl_rng_type *rng_t;
gsl_rng **rng;
gsl_rng_env_setup();
rng_t = gsl_rng_ranlxs0;
rng = (gsl_rng **) malloc((num_thread ) * sizeof(gsl_rng *));
rng[0]=rand;
for(i=1;i<num_thread;i++)
{
rng[i] = gsl_rng_alloc (rng_t);
gsl_rng_set(rng[i],gsl_rng_get(rand));
}
min_mfp=1e12;
#pragma omp parallel for num_threads(num_thread) firstprivate( is_in_block, ph_block_index, ph_x, ph_y, ph_z, ph_phi, ph_r, min_index, n_dens_lab_tmp,n_vx_tmp, n_vy_tmp, n_vz_tmp, n_temp_tmp, fl_v_x, fl_v_y, fl_v_z, fl_v_norm, ph_v_norm, n_cosangle, mfp, beta, rnd_tracker, ph_p_comv, el_p, ph_p, fluid_beta) private(i) shared(min_mfp ) reduction(+:num_photons_find_new_element)
for (i=0;i<num_ph; i++)
{
if (find_nearest_block_switch==0)
{
ph_block_index=(ph+i)->nearest_block_index; 
}
else
{
ph_block_index=0; 
}
mcratCoordinateToHydroCoordinate(&photon_hydro_coord, (ph+i)->r0, (ph+i)->r1, (ph+i)->r2);
#if DIMENSIONS == TWO || DIMENSIONS == TWO_POINT_FIVE
if (((photon_hydro_coord[1]<(hydro_data->r1_domain)[1]) &&
(photon_hydro_coord[1]>(hydro_data->r1_domain)[0]) &&
(photon_hydro_coord[0]<(hydro_data->r0_domain)[1]) &&
(photon_hydro_coord[0]>(hydro_data->r0_domain)[0])) && ((ph+i)->nearest_block_index != -1) ) 
#else
if (((photon_hydro_coord[2]<(hydro_data->r2_domain)[1]) &&
(photon_hydro_coord[2]>(hydro_data->r2_domain)[0]) &&
(photon_hydro_coord[1]<(hydro_data->r1_domain)[1]) &&
(photon_hydro_coord[1]>(hydro_data->r1_domain)[0]) &&
(photon_hydro_coord[0]<(hydro_data->r0_domain)[1]) &&
(photon_hydro_coord[0]>(hydro_data->r0_domain)[0])) && ((ph+i)->nearest_block_index != -1) )
#endif
{
is_in_block=checkInBlock(photon_hydro_coord[0], photon_hydro_coord[1], photon_hydro_coord[2], hydro_data, ph_block_index);
#if CYCLOSYNCHROTRON_SWITCH == ON
if ((ph_block_index==0) && ( ((ph+i)->comv_p0)+((ph+i)->comv_p1)+((ph+i)->comv_p2)+((ph+i)->comv_p3) == 0 ) )
{
is_in_block=0; 
}
#endif
if (find_nearest_block_switch==0 && is_in_block)
{
min_index=ph_block_index;
}
else
{
min_index=findContainingBlock(photon_hydro_coord[0], photon_hydro_coord[1], photon_hydro_coord[2], hydro_data, fPtr); 
if (min_index != -1)
{
(ph+i)->nearest_block_index=min_index; 
ph_p[0]=((ph+i)->p0);
ph_p[1]=((ph+i)->p1);
ph_p[2]=((ph+i)->p2);
ph_p[3]=((ph+i)->p3);
#if DIMENSIONS == THREE
hydroVectorToCartesian(&fluid_beta, (hydro_data->v0)[min_index], (hydro_data->v1)[min_index], (hydro_data->v2)[min_index], (hydro_data->r0)[min_index], (hydro_data->r1)[min_index], (hydro_data->r2)[min_index]);
#elif DIMENSIONS == TWO_POINT_FIVE
ph_phi=atan2(((ph+i)->r1), ((ph+i)->r0));
hydroVectorToCartesian(&fluid_beta, (hydro_data->v0)[min_index], (hydro_data->v1)[min_index], (hydro_data->v2)[min_index], (hydro_data->r0)[min_index], (hydro_data->r1)[min_index], ph_phi);
#else
ph_phi=atan2(((ph+i)->r1), ((ph+i)->r0));
hydroVectorToCartesian(&fluid_beta, (hydro_data->v0)[min_index], (hydro_data->v1)[min_index], 0, (hydro_data->r0)[min_index], (hydro_data->r1)[min_index], ph_phi);
#endif
lorentzBoost(&fluid_beta, &ph_p, &ph_p_comv, 'p', fPtr);
((ph+i)->comv_p0)=ph_p_comv[0];
((ph+i)->comv_p1)=ph_p_comv[1];
((ph+i)->comv_p2)=ph_p_comv[2];
((ph+i)->comv_p3)=ph_p_comv[3];
num_photons_find_new_element+=1;
}
else
{
fprintf(fPtr, "Photon number %d FLASH index not found, making sure it doesnt scatter.\n", i);
}
}
if (min_index != -1)
{
(n_dens_lab_tmp)= (hydro_data->dens_lab)[min_index];
(n_temp_tmp)= (hydro_data->temp)[min_index];
#if DIMENSIONS == THREE
hydroVectorToCartesian(&fluid_beta, (hydro_data->v0)[min_index], (hydro_data->v1)[min_index], (hydro_data->v2)[min_index], (hydro_data->r0)[min_index], (hydro_data->r1)[min_index], (hydro_data->r2)[min_index]);
#elif DIMENSIONS == TWO_POINT_FIVE
ph_phi=atan2(((ph+i)->r1), ((ph+i)->r0));
hydroVectorToCartesian(&fluid_beta, (hydro_data->v0)[min_index], (hydro_data->v1)[min_index], (hydro_data->v2)[min_index], (hydro_data->r0)[min_index], (hydro_data->r1)[min_index], ph_phi);
#else
ph_phi=atan2(((ph+i)->r1), ((ph+i)->r0));
hydroVectorToCartesian(&fluid_beta, (hydro_data->v0)[min_index], (hydro_data->v1)[min_index], 0, (hydro_data->r0)[min_index], (hydro_data->r1)[min_index], ph_phi);
#endif
fl_v_x=fluid_beta[0];
fl_v_y=fluid_beta[1];
fl_v_z=fluid_beta[2];
fl_v_norm=sqrt(fl_v_x*fl_v_x+fl_v_y*fl_v_y+fl_v_z*fl_v_z);
ph_v_norm=sqrt(((ph+i)->p1)*((ph+i)->p1)+((ph+i)->p2)*((ph+i)->p2)+((ph+i)->p3)*((ph+i)->p3));
n_cosangle=((fl_v_x* ((ph+i)->p1))+(fl_v_y* ((ph+i)->p2))+(fl_v_z* ((ph+i)->p3)))/(fl_v_norm*ph_v_norm ); 
beta=sqrt(1.0-1.0/((hydro_data->gamma)[min_index]*(hydro_data->gamma)[min_index]));
rnd_tracker=0;
#if defined(_OPENMP)
thread_id=omp_get_thread_num();
#endif
rnd_tracker=gsl_rng_uniform_pos(rng[thread_id]);
mfp=(-1)*(M_P/((n_dens_lab_tmp))/THOM_X_SECT/(1.0-beta*n_cosangle))*log(rnd_tracker) ;
}
else
{
mfp=min_mfp;
}
}
else
{
mfp=min_mfp;
}
*(all_time_steps+i)=mfp/C_LIGHT;
}
for (i=1;i<num_thread;i++)
{
gsl_rng_free(rng[i]);
}
free(rng);
for (i=0;i<num_ph;i++)
{
*(sorted_indexes+i)= i; 
}
#if (defined _GNU_SOURCE || defined __GNU__ || defined __linux__)
qsort_r(sorted_indexes, num_ph, sizeof (int),  compare2, all_time_steps);
#elif (defined __APPLE__ || defined __MACH__ || defined __DARWIN__ || defined __FREEBSD__ || defined __BSD__ || defined OpenBSD3_1 || defined OpenBSD3_9)
qsort_r(sorted_indexes, num_ph, sizeof (int), all_time_steps, compare);
#else
#error Cannot detect operating system
#endif
if (find_nearest_block_switch!=0)
{
num_photons_find_new_element=0; 
}
return num_photons_find_new_element;
}
int compare (void *ar, const void *a, const void *b)
{
int aa = *(int *) a;
int bb = *(int *) b;
double *arr=NULL;
arr=ar;
return ((arr[aa] > arr[bb]) - (arr[aa] < arr[bb]));
}
int compare2 ( const void *a, const void *b, void *ar)
{
int aa = *(int *) a;
int bb = *(int *) b;
double *arr=NULL;
arr=ar;
return ((arr[aa] > arr[bb]) - (arr[aa] < arr[bb]));
}
int interpolatePropertiesAndMinMFP( struct photon *ph, int num_ph, int array_num, double *time_step, double *x, double  *y, double *z, double *szx, double *szy, double *velx,  double *vely, double *velz, double *dens_lab,\
double *temp, double *n_dens_lab, double *n_vx, double *n_vy, double *n_vz, double *n_temp, gsl_rng * rand, int find_nearest_block_switch, FILE *fPtr)
{
return 0;
}
void updatePhotonPosition(struct photon *ph, int num_ph, double t, FILE *fPtr)
{
int i=0;
#if defined(_OPENMP)
int num_thread=omp_get_num_threads();
#endif
double old_position=0, new_position=0, divide_p0=0;
#pragma omp parallel for num_threads(num_thread) firstprivate(old_position, new_position, divide_p0)
for (i=0;i<num_ph;i++)
{
if (((ph+i)->type != CS_POOL_PHOTON) && ((ph+i)->weight != 0))
{
old_position= sqrt(((ph+i)->r0)*((ph+i)->r0)+((ph+i)->r1)*((ph+i)->r1)+((ph+i)->r2)*((ph+i)->r2)); 
divide_p0=1.0/((ph+i)->p0);
((ph+i)->r0)+=((ph+i)->p1)*divide_p0*C_LIGHT*t; 
((ph+i)->r1)+=((ph+i)->p2)*divide_p0*C_LIGHT*t;
((ph+i)->r2)+=((ph+i)->p3)*divide_p0*C_LIGHT*t;
new_position= sqrt(((ph+i)->r0)*((ph+i)->r0)+((ph+i)->r1)*((ph+i)->r1)+((ph+i)->r2)*((ph+i)->r2));
{
}
}
}
}
double photonEvent(struct photon *ph, int num_ph, double dt_max, double *all_time_steps, int *sorted_indexes, struct hydro_dataframe *hydro_data, int *scattered_ph_index, int *frame_scatt_cnt, int *frame_abs_cnt,  gsl_rng * rand, FILE *fPtr)
{
int  i=0, index=0, ph_index=0, event_did_occur=0; 
double scatt_time=0, old_scatt_time=0; 
double phi=0, theta=0; 
double ph_phi=0, flash_vx=0, flash_vy=0, flash_vz=0, fluid_temp=0;    
double *ph_p=malloc(4*sizeof(double)); 
double *el_p_comov=malloc(4*sizeof(double));
double *ph_p_comov=malloc(4*sizeof(double));
double *fluid_beta=malloc(3*sizeof(double));
double *negative_fluid_beta=malloc(3*sizeof(double));
double *s=malloc(4*sizeof(double)); 
i=0;
old_scatt_time=0;
event_did_occur=0;
while (i<num_ph && event_did_occur==0 )
{
ph_index=(*(sorted_indexes+i));
scatt_time= *(all_time_steps+ph_index); 
if (scatt_time<dt_max)
{
updatePhotonPosition(ph, num_ph, scatt_time-old_scatt_time, fPtr);
index=(ph+ph_index)->nearest_block_index; 
fluid_temp=(hydro_data->temp)[index];
ph_phi=atan2(((ph+ph_index)->r1), (((ph+ph_index)->r0)));
#if DIMENSIONS == THREE
hydroVectorToCartesian(fluid_beta, (hydro_data->v0)[index], (hydro_data->v1)[index], (hydro_data->v2)[index], (hydro_data->r0)[index], (hydro_data->r1)[index], (hydro_data->r2)[index]);
#elif DIMENSIONS == TWO_POINT_FIVE
hydroVectorToCartesian(fluid_beta, (hydro_data->v0)[index], (hydro_data->v1)[index], (hydro_data->v2)[index], (hydro_data->r0)[index], (hydro_data->r1)[index], ph_phi);
#else
hydroVectorToCartesian(fluid_beta, (hydro_data->v0)[index], (hydro_data->v1)[index], 0, (hydro_data->r0)[index], (hydro_data->r1)[index], ph_phi);
#endif
*(ph_p+0)=((ph+ph_index)->p0);
*(ph_p+1)=((ph+ph_index)->p1);
*(ph_p+2)=((ph+ph_index)->p2);
*(ph_p+3)=((ph+ph_index)->p3);
*(ph_p_comov+0)=((ph+ph_index)->comv_p0);
*(ph_p_comov+1)=((ph+ph_index)->comv_p1);
*(ph_p_comov+2)=((ph+ph_index)->comv_p2);
*(ph_p_comov+3)=((ph+ph_index)->comv_p3);
*(s+0)=((ph+ph_index)->s0); 
*(s+1)=((ph+ph_index)->s1); 
*(s+2)=((ph+ph_index)->s2); 
*(s+3)=((ph+ph_index)->s3); 
#if STOKES_SWITCH == ON
{
stokesRotation(fluid_beta, (ph_p+1), (ph_p_comov+1), s, fPtr);
}
#endif
singleElectron(el_p_comov, fluid_temp, ph_p_comov, rand, fPtr);
event_did_occur=singleScatter(el_p_comov, ph_p_comov, s, rand, fPtr);
if (event_did_occur==1)
{
*(negative_fluid_beta+0)=-1*( *(fluid_beta+0));
*(negative_fluid_beta+1)=-1*( *(fluid_beta+1));
*(negative_fluid_beta+2)=-1*( *(fluid_beta+2));
lorentzBoost(negative_fluid_beta, ph_p_comov, ph_p, 'p',  fPtr);
#if STOKES_SWITCH == ON
{
stokesRotation(negative_fluid_beta, (ph_p_comov+1), (ph_p+1), s, fPtr); 
((ph+ph_index)->s0)= *(s+0); 
((ph+ph_index)->s1)= *(s+1);
((ph+ph_index)->s2)= *(s+2);
((ph+ph_index)->s3)= *(s+3);
}
#endif
if (((*(ph_p+0))*C_LIGHT/1.6e-9) > 1e4)
{
fprintf(fPtr,"Extremely High Photon Energy!!!!!!!!\n");
fflush(fPtr);
}
((ph+ph_index)->p0)=(*(ph_p+0));
((ph+ph_index)->p1)=(*(ph_p+1));
((ph+ph_index)->p2)=(*(ph_p+2));
((ph+ph_index)->p3)=(*(ph_p+3));
((ph+ph_index)->comv_p0)=(*(ph_p_comov+0));
((ph+ph_index)->comv_p1)=(*(ph_p_comov+1));
((ph+ph_index)->comv_p2)=(*(ph_p_comov+2));
((ph+ph_index)->comv_p3)=(*(ph_p_comov+3));
((ph+ph_index)->num_scatt)+=1;
*frame_scatt_cnt+=1; 
}
}
else
{
scatt_time=dt_max;
updatePhotonPosition(ph, num_ph, scatt_time-old_scatt_time, fPtr); 
event_did_occur=1; 
}
old_scatt_time=scatt_time;
i++;
}
*scattered_ph_index=ph_index; 
free(el_p_comov); 
free(ph_p_comov);
free(fluid_beta); 
free(negative_fluid_beta);
free(ph_p);
free(s);
ph_p=NULL;negative_fluid_beta=NULL;ph_p_comov=NULL; el_p_comov=NULL;
return scatt_time;
}
void singleElectron(double *el_p, double temp, double *ph_p, gsl_rng * rand, FILE *fPtr)
{
double factor=0, gamma=0;
double y_dum=0, f_x_dum=0, x_dum=0, beta_x_dum=0, beta=0, phi=0, theta=0, ph_theta=0, ph_phi=0;
gsl_matrix *rot= gsl_matrix_calloc (3, 3); 
gsl_vector_view el_p_prime ; 
gsl_vector *result=gsl_vector_alloc (3);
if (temp>= 1e7)
{
factor=K_B*temp/(M_EL*C_LIGHT*C_LIGHT);
y_dum=1; 
f_x_dum=0;
while ((isnan(f_x_dum) !=0) || (y_dum>f_x_dum) )
{
x_dum=gsl_rng_uniform_pos(rand)*(1+100*factor);
beta_x_dum=sqrt(1-(1/(x_dum*x_dum)));
y_dum=gsl_rng_uniform(rand)/2.0;
f_x_dum=x_dum*x_dum*(beta_x_dum/gsl_sf_bessel_Kn (2, 1.0/factor))*exp(-1*x_dum/factor); 
}
gamma=x_dum;
}
else
{
factor=sqrt(K_B*temp/M_EL);
gamma=1.0/sqrt( 1- (pow(gsl_ran_gaussian(rand, factor)/C_LIGHT, 2)+ pow(gsl_ran_gaussian(rand, factor)/C_LIGHT, 2)+pow(gsl_ran_gaussian(rand, factor)/C_LIGHT, 2)  )); 
}
beta=sqrt( 1- (1/(gamma*gamma)) );
phi=gsl_rng_uniform(rand)*2*M_PI;
y_dum=1; 
f_x_dum=0;
while (y_dum>f_x_dum)
{
y_dum=gsl_rng_uniform(rand)*1.3;
x_dum=gsl_rng_uniform(rand)*M_PI;
f_x_dum=sin(x_dum)*(1-(beta*cos(x_dum)));
}
theta=x_dum;
*(el_p+0)=gamma*(M_EL)*(C_LIGHT);
*(el_p+1)=gamma*(M_EL)*(C_LIGHT)*beta*cos(theta);
*(el_p+2)=gamma*(M_EL)*(C_LIGHT)*beta*sin(theta)*sin(phi);
*(el_p+3)=gamma*(M_EL)*(C_LIGHT)*beta*sin(theta)*cos(phi);
el_p_prime=gsl_vector_view_array((el_p+1), 3);
ph_phi=atan2(*(ph_p+2), *(ph_p+3)); 
ph_theta=atan2(sqrt( pow(*(ph_p+2),2)+  pow(*(ph_p+3),2)) , (*(ph_p+1)) );
gsl_matrix_set(rot, 1,1,1);
gsl_matrix_set(rot, 2,2,cos(ph_theta));
gsl_matrix_set(rot, 0,0,cos(ph_theta));
gsl_matrix_set(rot, 0,2,-sin(ph_theta));
gsl_matrix_set(rot, 2,0,sin(ph_theta));
gsl_blas_dgemv(CblasNoTrans, 1, rot, &el_p_prime.vector, 0, result);
gsl_matrix_set_all(rot,0);
gsl_matrix_set(rot, 0,0,1);
gsl_matrix_set(rot, 1,1,cos(-ph_phi));
gsl_matrix_set(rot, 2,2,cos(-ph_phi));
gsl_matrix_set(rot, 1,2,-sin(-ph_phi));
gsl_matrix_set(rot, 2,1,sin(-ph_phi));
gsl_blas_dgemv(CblasNoTrans, 1, rot, result, 0, &el_p_prime.vector);
gsl_matrix_free (rot);gsl_vector_free(result);
}
double averagePhotonEnergy(struct photon *ph, int num_ph)
{
int i=0;
#if defined(_OPENMP)
int num_thread=omp_get_num_threads();
#endif
double e_sum=0, w_sum=0;
#pragma omp parallel for reduction(+:e_sum) reduction(+:w_sum)
for (i=0;i<num_ph;i++)
{
#if CYCLOSYNCHROTRON_SWITCH == ON
if (((ph+i)->weight != 0)) 
#endif
{
e_sum+=(((ph+i)->p0)*((ph+i)->weight));
w_sum+=((ph+i)->weight);
}
}
return (e_sum*C_LIGHT)/w_sum;
}
void phScattStats(struct photon *ph, int ph_num, int *max, int *min, double *avg, double *r_avg, FILE *fPtr  )
{
int temp_max=0, temp_min=INT_MAX,  i=0, count=0, count_synch=0, count_comp=0, count_i=0;
#if defined(_OPENMP)
int num_thread=omp_get_num_threads();
#endif
double sum=0, avg_r_sum=0, avg_r_sum_synch=0, avg_r_sum_comp=0, avg_r_sum_inject=0;
#pragma omp parallel for num_threads(num_thread) reduction(min:temp_min) reduction(max:temp_max) reduction(+:sum) reduction(+:avg_r_sum) reduction(+:count)
for (i=0;i<ph_num;i++)
{
#if CYCLOSYNCHROTRON_SWITCH == ON
if (((ph+i)->weight != 0)) 
#endif
{
sum+=((ph+i)->num_scatt);
avg_r_sum+=sqrt(((ph+i)->r0)*((ph+i)->r0) + ((ph+i)->r1)*((ph+i)->r1) + ((ph+i)->r2)*((ph+i)->r2));
if (((ph+i)->num_scatt) > temp_max )
{
temp_max=((ph+i)->num_scatt);
}
if (((ph+i)->num_scatt)<temp_min)
{
temp_min=((ph+i)->num_scatt);
}
if (((ph+i)->type) == INJECTED_PHOTON )
{
avg_r_sum_inject+=sqrt(((ph+i)->r0)*((ph+i)->r0) + ((ph+i)->r1)*((ph+i)->r1) + ((ph+i)->r2)*((ph+i)->r2));
count_i++;
}
#if CYCLOSYNCHROTRON_SWITCH == ON
if ((((ph+i)->type) == COMPTONIZED_PHOTON) || (((ph+i)->type) == UNABSORBED_CS_PHOTON))
{
avg_r_sum_comp+=sqrt(((ph+i)->r0)*((ph+i)->r0) + ((ph+i)->r1)*((ph+i)->r1) + ((ph+i)->r2)*((ph+i)->r2));
count_comp++;
}
#endif
count++;
}
#if CYCLOSYNCHROTRON_SWITCH == ON
if (((ph+i)->type) == CS_POOL_PHOTON )
{
avg_r_sum_synch+=sqrt(((ph+i)->r0)*((ph+i)->r0) + ((ph+i)->r1)*((ph+i)->r1) + ((ph+i)->r2)*((ph+i)->r2));
count_synch++;
}
#endif
}
#if CYCLOSYNCHROTRON_SWITCH == ON
fprintf(fPtr, "In this frame Avg r for i type: %e c and o type: %e and s type: %e\n", avg_r_sum_inject/count_i, avg_r_sum_comp/count_comp, avg_r_sum_synch/count_synch);
#else
fprintf(fPtr, "In this frame Avg r for i type: %e \n", avg_r_sum_inject/count_i);
#endif
fflush(fPtr);
*avg=sum/count;
*r_avg=avg_r_sum/count;
*max=temp_max;
*min=temp_min;
}
void cylindricalPrep(struct hydro_dataframe *hydro_data, FILE *fPtr)
{
double  gamma_infinity=100, t_comov=1e5, ddensity=3e-7;
int i=0;
double vel=sqrt(1-pow(gamma_infinity, -2.0)), lab_dens=gamma_infinity*ddensity;
fprintf(fPtr, "The Cylindrical Outflow values are: Gamma_infinity=%e, T_comv=%e K, comv dens=%e g/cm^3 \n", gamma_infinity, t_comov, ddensity);
fflush(fPtr);
for (i=0; i<hydro_data->num_elements; i++)
{
((hydro_data->gamma))[i]=gamma_infinity;
((hydro_data->dens))[i]=ddensity;
((hydro_data->dens_lab))[i]=lab_dens;
((hydro_data->pres))[i]=(A_RAD*pow(t_comov, 4.0))/(3);
((hydro_data->temp))[i]=t_comov; 
#if DIMENSIONS == TWO || DIMENSIONS == TWO_POINT_FIVE
#if GEOMETRY == CARTESIAN || GEOMETRY == CYLINDRICAL
((hydro_data->v0))[i]=0;
((hydro_data->v1))[i]=vel; 
#endif
#if GEOMETRY == SPHERICAL
((hydro_data->v0))[i]=vel*cos(((hydro_data->r1))[i]);
((hydro_data->v1))[i]=-vel*sin(((hydro_data->r1))[i]);
#endif
#if DIMENSIONS == TWO_POINT_FIVE
((hydro_data->v2))[i]=0;
#endif
#else
#if GEOMETRY == CARTESIAN
((hydro_data->v0))[i]=0;
((hydro_data->v1))[i]=0;
((hydro_data->v2))[i]=vel;
#endif
#if GEOMETRY == SPHERICAL
((hydro_data->v0))[i]=vel*cos(((hydro_data->r1))[i]);
((hydro_data->v1))[i]=-vel*sin(((hydro_data->r1))[i]);
((hydro_data->v2))[i]=0;
#endif
#if GEOMETRY == POLAR
((hydro_data->v0))[i]=0;
((hydro_data->v1))[i]=0;
((hydro_data->v2))[i]=vel;
#endif
#endif
}
}
void sphericalPrep(struct hydro_dataframe *hydro_data, FILE *fPtr)
{
double  gamma_infinity=100, lumi=1e54, r00=1e8; 
double vel=0, r=0;
int i=0;
fprintf(fPtr, "The Spherical Outflow values are: Gamma_infinity=%e, Luminosity=%e erg/s, r_0=%e cm \n", gamma_infinity, lumi, r00);
fflush(fPtr);
for (i=0; i<hydro_data->num_elements; i++)
{
if (((hydro_data->r))[i] >= (r00*gamma_infinity))
{
((hydro_data->gamma))[i]=gamma_infinity;
((hydro_data->pres))[i]=(lumi*pow(r00, 2.0/3.0)*pow(((hydro_data->r))[i], -8.0/3.0) )/(12.0*M_PI*C_LIGHT*pow(gamma_infinity, 4.0/3.0));
}
else
{
((hydro_data->gamma))[i]=((hydro_data->r))[i]/r00;
((hydro_data->pres))[i]=(lumi*pow(r00, 2.0))/(12.0*M_PI*C_LIGHT*pow(((hydro_data->r))[i], 4.0) );
}
((hydro_data->dens))[i]=lumi/(4*M_PI*pow(((hydro_data->r))[i], 2.0)*pow(C_LIGHT, 3.0)*gamma_infinity*(((hydro_data->gamma))[i]));
((hydro_data->dens_lab))[i]=(((hydro_data->dens))[i])*(((hydro_data->gamma))[i]);
((hydro_data->temp))[i]=pow(3*(((hydro_data->pres))[i])/(A_RAD) ,1.0/4.0);
vel=sqrt(1-(pow(((hydro_data->gamma))[i], -2.0)));
#if DIMENSIONS == TWO || DIMENSIONS == TWO_POINT_FIVE
#if GEOMETRY == CARTESIAN || GEOMETRY == CYLINDRICAL
r=sqrt(pow(((hydro_data->r0))[i], 2)+ pow(((hydro_data->r1))[i], 2));
((hydro_data->v0))[i]=(vel*(((hydro_data->r0))[i]))/r;
((hydro_data->v1))[i]=(vel*(((hydro_data->r1))[i]))/r; 
#endif
#if GEOMETRY == SPHERICAL
((hydro_data->v0))[i]=vel;
((hydro_data->v1))[i]=0;
#endif
#if DIMENSIONS == TWO_POINT_FIVE
((hydro_data->v2))[i]=0;
#endif
#else
#if GEOMETRY == CARTESIAN
r=sqrt(pow(((hydro_data->r0))[i], 2)+ pow(((hydro_data->r1))[i], 2)+pow(((hydro_data->r2))[i], 2));
((hydro_data->v0))[i]=(vel*(((hydro_data->r0))[i]))/r;
((hydro_data->v1))[i]=(vel*(((hydro_data->r1))[i]))/r; 
((hydro_data->v2))[i]=(vel*(((hydro_data->r2))[i]))/r;
#endif
#if GEOMETRY == SPHERICAL
((hydro_data->v0))[i]=vel;
((hydro_data->v1))[i]=0;
((hydro_data->v2))[i]=0;
#endif
#if GEOMETRY == POLAR
r=sqrt(pow(((hydro_data->r0))[i], 2)+ pow(((hydro_data->r2))[i], 2));
((hydro_data->v0))[i]=(vel*(((hydro_data->r0))[i]))/r; 
((hydro_data->v1))[i]=0;
((hydro_data->v2))[i]=(vel*(((hydro_data->r2))[i]))/r;
#endif
#endif
}
}
void structuredFireballPrep(struct hydro_dataframe *hydro_data, FILE *fPtr)
{
double  gamma_0=100, lumi=1e52, r00=1e8, theta_j=1e-2, p=4; 
double T_0=pow(lumi/(4*M_PI*r00*r00*A_RAD*C_LIGHT), 1.0/4.0);
double eta=0, r_sat=0, r;
double vel=0, theta_ratio=0;
int i=0;
fprintf(fPtr, "The Structured Spherical Outflow values are: Gamma_0=%e, Luminosity=%e erg/s, r_0=%e cm, theta_j=%e rad, p=%e \n", gamma_0, lumi, r00, theta_j, p);
fflush(fPtr);
for (i=0; i<hydro_data->num_elements; i++)
{
theta_ratio=((hydro_data->theta)[i])/theta_j;
eta=gamma_0/sqrt(1+pow(theta_ratio, 2*p));
if ((hydro_data->theta)[i] >= theta_j*pow(gamma_0/2, 1.0/p))
{
eta=2.0;
}
r_sat=eta*r00;
if (((hydro_data->r)[i]) >= r_sat)
{
(hydro_data->gamma)[i]=eta;
(hydro_data->temp)[i]=T_0*pow(r_sat/((hydro_data->r)[i]), 2.0/3.0)/eta;
}
else
{
(hydro_data->gamma)[i]=((hydro_data->r)[i])/r_sat; 
(hydro_data->temp)[i]=T_0;
}
vel=sqrt(1-(pow((hydro_data->gamma)[i], -2.0)));
(hydro_data->dens)[i] = M_P*lumi/(4*M_PI*M_P*C_LIGHT*C_LIGHT*C_LIGHT*eta*vel*((hydro_data->gamma)[i])*((hydro_data->r)[i])*((hydro_data->r)[i])); 
(hydro_data->dens_lab)[i]=((hydro_data->dens)[i])*((hydro_data->gamma)[i]);
(hydro_data->pres)[i]=(A_RAD*pow((hydro_data->temp)[i], 4.0))/(3);
#if DIMENSIONS == TWO || DIMENSIONS == TWO_POINT_FIVE
#if GEOMETRY == CARTESIAN || GEOMETRY == CYLINDRICAL
r=sqrt(pow(((hydro_data->r0))[i], 2)+ pow(((hydro_data->r1))[i], 2));
((hydro_data->v0))[i]=(vel*(((hydro_data->r0))[i]))/r;
((hydro_data->v1))[i]=(vel*(((hydro_data->r1))[i]))/r; 
#endif
#if GEOMETRY == SPHERICAL
((hydro_data->v0))[i]=vel;
((hydro_data->v1))[i]=0;
#endif
#if DIMENSIONS == TWO_POINT_FIVE
((hydro_data->v2))[i]=0;
#endif
#else
#if GEOMETRY == CARTESIAN
r=sqrt(pow(((hydro_data->r0))[i], 2)+ pow(((hydro_data->r1))[i], 2)+pow(((hydro_data->r2))[i], 2));
((hydro_data->v0))[i]=(vel*(((hydro_data->r0))[i]))/r;
((hydro_data->v1))[i]=(vel*(((hydro_data->r1))[i]))/r; 
((hydro_data->v2))[i]=(vel*(((hydro_data->r2))[i]))/r;
#endif
#if GEOMETRY == SPHERICAL
((hydro_data->v0))[i]=vel;
((hydro_data->v1))[i]=0;
((hydro_data->v2))[i]=0;
#endif
#if GEOMETRY == POLAR
r=sqrt(pow(((hydro_data->r0))[i], 2)+ pow(((hydro_data->r2))[i], 2));
((hydro_data->v0))[i]=(vel*(((hydro_data->r0))[i]))/r;
((hydro_data->v1))[i]=0;
((hydro_data->v2))[i]=(vel*(((hydro_data->r2))[i]))/r;
#endif
#endif
}
}
void phMinMax(struct photon *ph, int ph_num, double *min, double *max, double *min_theta, double *max_theta, FILE *fPtr)
{
double temp_r_max=0, temp_r_min=DBL_MAX, temp_theta_max=0, temp_theta_min=DBL_MAX;
int i=0;
#if defined(_OPENMP)
int num_thread=omp_get_num_threads();
#endif
double ph_r=0, ph_theta=0;
#pragma omp parallel for num_threads(num_thread) firstprivate(ph_r, ph_theta) reduction(min:temp_r_min) reduction(max:temp_r_max) reduction(min:temp_theta_min) reduction(max:temp_theta_max)
for (i=0;i<ph_num;i++)
{
if ((ph+i)->weight != 0)
{
ph_r=sqrt(((ph+i)->r0)*((ph+i)->r0) + ((ph+i)->r1)*((ph+i)->r1) + ((ph+i)->r2)*((ph+i)->r2));
ph_theta=acos(((ph+i)->r2) /ph_r); 
if (ph_r > temp_r_max )
{
temp_r_max=ph_r;
}
if (ph_r<temp_r_min)
{
temp_r_min=ph_r;
}
if (ph_theta > temp_theta_max )
{
temp_theta_max=ph_theta;
}
if (ph_theta<temp_theta_min)
{
temp_theta_min=ph_theta;
}
}
}
*max=temp_r_max;
*min=temp_r_min;
*max_theta=temp_theta_max;
*min_theta=temp_theta_min;
}
