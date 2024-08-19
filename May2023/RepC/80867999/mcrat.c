#include "mcrat.h"
int main(int argc, char **argv)
{
char hydro_prefix[STR_BUFFER]="";
char mc_file[STR_BUFFER]="" ;
char spect;
char restrt;
double fps, fps_modified, theta_jmin, theta_jmax,hydro_domain_y, hydro_domain_x ;
double inj_radius_small, inj_radius_large,  ph_weight_suggest=1e50, ph_weight_small, ph_weight_large, *inj_radius_input=NULL, ph_weight_default=1e50;
int frm0,last_frm, frm2_small, frm2_large, j=0, min_photons, max_photons, frm0_small, frm0_large, *frm2_input=NULL, *frm0_input=NULL ;
int dim_switch=0;
int find_nearest_grid_switch=0;
int increment_inj=1, increment_scatt=1; 
double inj_radius;
int frm2,save_chkpt_success=0;
char mc_filename[STR_BUFFER]="";
char mc_filename_2[STR_BUFFER]="";
char mc_operation[STR_BUFFER]="";
char mc_dir[STR_BUFFER]="" ;
int file_count = 0;
DIR * dirp;
struct dirent * entry;
struct stat st = {0};
double theta_jmin_thread=0, theta_jmax_thread=0;
char hydro_file[STR_BUFFER]="";
char log_file[STR_BUFFER]="";
FILE *fPtr=NULL; 
double *xPtr=NULL,  *yPtr=NULL,  *rPtr=NULL,  *thetaPtr=NULL,  *velxPtr=NULL,  *velyPtr=NULL,  *densPtr=NULL,  *presPtr=NULL,  *gammaPtr=NULL,  *dens_labPtr=NULL;
double *szxPtr=NULL,*szyPtr=NULL, *tempPtr=NULL; 
double *phiPtr=NULL, *velzPtr=NULL, *zPtr=NULL, *all_time_steps=NULL ;
int num_ph=0, scatt_cyclosynch_num_ph=0, num_null_ph=0, array_num=0, ph_scatt_index=0, num_photons_find_new_element=0, max_scatt=0, min_scatt=0,i=0; 
double dt_max=0, thescatt=0, accum_time=0; 
double  gamma_infinity=0, time_now=0, time_step=0, avg_scatt=0,avg_r=0; 
double ph_dens_labPtr=0, ph_vxPtr=0, ph_vyPtr=0, ph_tempPtr=0, ph_vzPtr=0;
double min_r=0, max_r=0, min_theta=0, max_theta=0, nu_c_scatt=0, n_comptonized=0;
int frame=0, scatt_frame=0, frame_scatt_cnt=0, frame_abs_cnt=0, scatt_framestart=0, framestart=0;
struct photon *phPtr=NULL; 
struct hydro_dataframe hydrodata; 
int angle_count=0, num_cyclosynch_ph_emit=0;
int num_angles=0, old_num_angle_procs=0; 
int *frame_array=NULL, *proc_frame_array=NULL, *element_num=NULL, *sorted_indexes=NULL,  proc_frame_size=0;
double *thread_theta=NULL; 
double delta_theta=1, num_theta_bins=0;
double test_cyclosynch_inj_radius=0;
int myid, numprocs, angle_procs, angle_id, procs_per_angle;
int temporary[3]={0}, tempo=0;
MPI_Init(NULL,NULL);
MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
MPI_Comm_rank(MPI_COMM_WORLD, &myid);
const gsl_rng_type *rng_t;
gsl_rng *rng;
gsl_rng_env_setup();
rng_t = gsl_rng_ranlxs0;
rng = gsl_rng_alloc (rng_t); 
hydroDataFrameInitialize(&hydrodata);
readMcPar(&hydrodata, &theta_jmin, &theta_jmax, &num_theta_bins, &inj_radius_input, &frm0_input , &frm2_input, &min_photons, &max_photons, &spect, &restrt); 
fps=hydrodata.fps;
last_frm=hydrodata.last_frame;
num_angles= (int) num_theta_bins;
delta_theta=((theta_jmax-theta_jmin)/num_theta_bins);
thread_theta=malloc( num_angles *sizeof(double) );
*(thread_theta+0)=theta_jmin;/
if (restrt==CONTINUE)
{
printf(">> Rank %d: Starting from photons injected at frame: %d out of %d\n", angle_id,framestart, frm2);
printf(">> Rank %d with angles %0.1lf-%0.1lf: Continuing scattering %d photons from frame: %d\n", angle_id, theta_jmin_thread*180/M_PI, theta_jmax_thread*180/M_PI,num_ph, scatt_framestart);
printf(">> Rank %d with angles %0.1lf-%0.1lf: The time now is: %e\n", angle_id, theta_jmin_thread*180/M_PI, theta_jmax_thread*180/M_PI,time_now);
}
else
{
printf(">> Rank %d with angles %0.1lf-%0.1lf: Continuing simulation by injecting photons at frame: %d out of %d\n", angle_id, theta_jmin_thread*180/M_PI, theta_jmax_thread*180/M_PI,framestart, frm2); 
}
}
else if ((stat(mc_dir, &st) == -1) && (restrt==INITALIZE))
{
mkdir(mc_dir, 0777); 
}
else
{
if (angle_id==0)
{
printf(">> proc %d with angles %0.1lf-%0.1lf:  Cleaning directory \n",angle_id, theta_jmin_thread*180/M_PI, theta_jmax_thread*180/M_PI);
dirp = opendir(mc_dir);
while ((entry = readdir(dirp)) != NULL)
{
if (entry->d_type == DT_REG) { 
file_count++; 
}
}
printf("File count %d\n", file_count);
if (file_count>0)
{
snprintf(mc_operation,sizeof(mc_operation),"%s%s%s","exec rm ", mc_dir,"mc_proc_*"); 
system(mc_operation);
snprintf(mc_operation,sizeof(mc_operation),"%s%s%s","exec rm ", mc_dir,"mcdata_PW_*"); 
system(mc_operation);
snprintf(mc_operation,sizeof(mc_operation),"%s%s%s","exec rm ", mc_dir,"mcdata_PW*"); 
system(mc_operation);
snprintf(mc_operation,sizeof(mc_operation),"%s%s%s","exec rm ", mc_dir,"mc_chkpt_*.dat"); 
system(mc_operation);
snprintf(mc_operation,sizeof(mc_operation),"%s%s%s","exec rm ", mc_dir,"mc_output_*.log"); 
system(mc_operation);
}
}
}
#if SIM_SWITCH == RIKEN && DIMENSIONS == THREE
if (framestart>=3000)
{
hydrodata.increment_inj_frame=10;
hydrodata.fps=1;
}
#else
{
hydrodata.increment_inj_frame=1;
hydrodata.fps=fps; 
}
#endif
dt_max=1.0/hydrodata.fps;
MPI_Barrier(angle_comm);
snprintf(log_file,sizeof(log_file),"%s%s%d%s",mc_dir,"mc_output_", angle_id,".log" );
printf("%s\n",log_file);
fPtr=fopen(log_file, "a");
printf( "Im Proc %d with angles %0.1lf-%0.1lf proc_frame_size is %d Starting on Frame: %d Injecting until %d scatt_framestart: %d\n", angle_id, theta_jmin_thread*180/M_PI, theta_jmax_thread*180/M_PI, proc_frame_size, framestart, frm2, scatt_framestart);
fprintf(fPtr, "Im Proc %d with angles %0.1lf-%0.1lf  Starting on Frame: %d scatt_framestart: %d\n", angle_id, theta_jmin_thread*180/M_PI, theta_jmax_thread*180/M_PI, framestart, scatt_framestart);
fflush(fPtr);
free(frame_array);
frame_array=NULL;
for (frame=framestart;frame<=frm2;frame=frame+hydrodata.increment_inj_frame)
{
hydrodata.inj_frame_number=frame;
#if SIM_SWITCH == RIKEN && DIMENSIONS == THREE
if (frame>=3000)
{
hydrodata.increment_inj_frame=10; 
hydrodata.fps=1; 
}
#else
{
hydrodata.increment_inj_frame=1;
hydrodata.fps=fps;
}
#endif
if (restrt==INITALIZE)
{
time_now=frame/hydrodata.fps; 
}
printHydroGeometry(fPtr);
fprintf(fPtr,">> Im Proc: %d with angles %0.1lf - %0.1lf Working on Frame: %d\n", angle_id, theta_jmin_thread*180/M_PI, theta_jmax_thread*180/M_PI, frame);
fflush(fPtr);
if (restrt==INITALIZE)
{
getHydroData(&hydrodata, frame, inj_radius, 1, min_r, max_r, min_theta, max_theta, fPtr);
fprintf(fPtr,">>  Proc: %d with angles %0.1lf-%0.1lf: Injecting photons\n",angle_id, theta_jmin_thread*180/M_PI, theta_jmax_thread*180/M_PI);
fflush(fPtr);
photonInjection(&phPtr, &num_ph, inj_radius, ph_weight_suggest, min_photons, max_photons,spect, theta_jmin_thread, theta_jmax_thread, &hydrodata,rng, fPtr );
}
freeHydroDataFrame(&hydrodata);
if (restrt==INITALIZE)
{
scatt_framestart=frame; 
}
num_null_ph=0;
hydrodata.increment_scatt_frame=1;
for (scatt_frame=scatt_framestart;scatt_frame<=last_frm;scatt_frame=scatt_frame+hydrodata.increment_scatt_frame)
{
hydrodata.scatt_frame_number=scatt_frame;
#if SIM_SWITCH == RIKEN && DIMENSIONS == THREE
if (scatt_frame>=3000)
{
hydrodata.increment_scatt_frame=10; 
hydrodata.fps=1; 
}
#else
{
hydrodata.increment_scatt_frame=1;
hydrodata.fps=fps;
}
#endif
dt_max=1.0/hydrodata.fps; 
fprintf(fPtr,">>\n");
printHydroGeometry(fPtr);
fprintf(fPtr,">> Proc %d with angles %0.1lf-%0.1lf: Working on photons injected at frame: %d out of %d\n", angle_id, theta_jmin_thread*180/M_PI, theta_jmax_thread*180/M_PI,frame, frm2);
#if SIMULATION_TYPE == SCIENCE
fprintf(fPtr,">> Proc %d with angles %0.1lf-%0.1lf: Simulation type Science - Working on scattering photons in frame %d\n",angle_id, theta_jmin_thread*180/M_PI, theta_jmax_thread*180/M_PI, scatt_frame);
#elif SIMULATION_TYPE == SPHERICAL_OUTFLOW
fprintf(fPtr,">> Proc %d with angles %0.1lf-%0.1lf: Simulation type Spherical Outflow - Working on scattering photons in frame %d\n",angle_id, theta_jmin_thread*180/M_PI, theta_jmax_thread*180/M_PI, scatt_frame);
#elif SIMULATION_TYPE == CYLINDRICAL_OUTFLOW
fprintf(fPtr,">> Proc %d with angles %0.1lf-%0.1lf: Simulation type Cylindrical Outflow - Working on scattering photons in frame %d\n",angle_id, theta_jmin_thread*180/M_PI, theta_jmax_thread*180/M_PI, scatt_frame);
#elif SIMULATION_TYPE == STRUCTURED_SPHERICAL_OUTFLOW
fprintf(fPtr,">> Proc %d with angles %0.1lf-%0.1lf: Simulation type Structured Spherical Outflow - Working on scattering photons in frame %d\n",angle_id, theta_jmin_thread*180/M_PI, theta_jmax_thread*180/M_PI, scatt_frame);
#endif
gsl_rng_set(rng, gsl_rng_get(rng));
phMinMax(phPtr, num_ph, &min_r, &max_r, &min_theta, &max_theta, fPtr);
#if CYCLOSYNCHROTRON_SWITCH == ON
if ((scatt_frame != scatt_framestart) || (restrt==CONTINUE))
{
test_cyclosynch_inj_radius=calcCyclosynchRLimits( scatt_frame, frame, hydrodata.fps,  inj_radius, "min");
min_r=(min_r < test_cyclosynch_inj_radius) ? min_r : test_cyclosynch_inj_radius ;
test_cyclosynch_inj_radius=calcCyclosynchRLimits( scatt_frame, frame, hydrodata.fps,  inj_radius, "max");
max_r=(max_r > test_cyclosynch_inj_radius ) ? max_r : test_cyclosynch_inj_radius ;
}
#endif
getHydroData(&hydrodata, scatt_frame, inj_radius, 0, min_r, max_r, min_theta, max_theta, fPtr);
num_cyclosynch_ph_emit=0;
all_time_steps=malloc(num_ph*sizeof(double));
sorted_indexes=malloc(num_ph*sizeof(int));
#if CYCLOSYNCHROTRON_SWITCH == ON
if ((scatt_frame != scatt_framestart) || (restrt==CONTINUE)) 
{
fprintf(fPtr, "Emitting Cyclosynchrotron Photons in frame %d\n", scatt_frame);
#if B_FIELD_CALC == INTERNAL_E
fprintf(fPtr, "Calculating the magnetic field using internal energy and epsilon_B is set to %lf.\n", EPSILON_B);
#elif B_FIELD_CALC == TOTAL_E
fprintf(fPtr, "Calculating the magnetic field using the total energy and epsilon_B is set to %lf.\n", EPSILON_B);
#else
fprintf(fPtr, "Using the magnetic field from the hydro simulation.\n");
#endif
phScattStats(phPtr, num_ph, &max_scatt, &min_scatt, &avg_scatt, &avg_r, fPtr); 
num_cyclosynch_ph_emit=photonEmitCyclosynch(&phPtr, &num_ph, &num_null_ph, &all_time_steps, &sorted_indexes, inj_radius, ph_weight_suggest, max_photons, theta_jmin_thread, theta_jmax_thread, &hydrodata, rng, 0, 0, fPtr);
}
#endif
fprintf(fPtr,">> Proc %d with angles %0.1lf-%0.1lf: propagating and scattering %d photons\n",angle_id, theta_jmin_thread*180/M_PI, theta_jmax_thread*180/M_PI,num_ph-num_null_ph);
fflush(fPtr);
frame_scatt_cnt=0;
frame_abs_cnt=0;
find_nearest_grid_switch=1; 
num_photons_find_new_element=0;
n_comptonized=0;
while (time_now<((scatt_frame+hydrodata.increment_scatt_frame)/hydrodata.fps))
{
num_photons_find_new_element+=findNearestPropertiesAndMinMFP(phPtr, num_ph, all_time_steps, sorted_indexes, &hydrodata, rng, find_nearest_grid_switch, fPtr);
find_nearest_grid_switch=0; 
if (*(all_time_steps+(*(sorted_indexes+0)))<dt_max)
{
time_step=photonEvent( phPtr, num_ph, dt_max, all_time_steps, sorted_indexes, &hydrodata, &ph_scatt_index, &frame_scatt_cnt, &frame_abs_cnt, rng, fPtr );
time_now+=time_step;
#if CYCLOSYNCHROTRON_SWITCH == ON
if ((phPtr+ph_scatt_index)->type == CS_POOL_PHOTON)
{
n_comptonized+=(phPtr+ph_scatt_index)->weight;
(phPtr+ph_scatt_index)->type = COMPTONIZED_PHOTON; 
num_cyclosynch_ph_emit+=photonEmitCyclosynch(&phPtr, &num_ph, &num_null_ph, &all_time_steps, &sorted_indexes, inj_radius, ph_weight_suggest, max_photons, theta_jmin_thread, theta_jmax_thread, &hydrodata, rng, 1, ph_scatt_index, fPtr);
scatt_cyclosynch_num_ph++;
}
#endif
if ((frame_scatt_cnt%1000 == 0) && (frame_scatt_cnt != 0)) 
{
fprintf(fPtr,"Scattering Number: %d\n", frame_scatt_cnt);
fprintf(fPtr,"The local temp is: %e K\n", *(hydrodata.temp + (phPtr+ph_scatt_index)->nearest_block_index) );
fprintf(fPtr,"Average photon energy is: %e ergs\n", averagePhotonEnergy(phPtr, num_ph)); 
fprintf(fPtr,"The last time step was: %e.\nThe time now is: %e\n", time_step,time_now);
fflush(fPtr);
#if CYCLOSYNCHROTRON_SWITCH == ON
if (scatt_cyclosynch_num_ph>max_photons)
{
rebinCyclosynchCompPhotons(&phPtr, &num_ph, &num_null_ph, &num_cyclosynch_ph_emit, &scatt_cyclosynch_num_ph, &all_time_steps, &sorted_indexes, max_photons, theta_jmin_thread, theta_jmax_thread, rng, fPtr);
}
#endif
}
}
else
{
time_now+=dt_max;
updatePhotonPosition(phPtr, num_ph, dt_max, fPtr);
}
}
#if CYCLOSYNCHROTRON_SWITCH == ON
if ((scatt_frame != scatt_framestart) || (restrt==CONTINUE)) 
{
if (scatt_cyclosynch_num_ph>max_photons)
{
fprintf(fPtr, "Num_ph: %d\n", num_ph);
rebinCyclosynchCompPhotons(&phPtr, &num_ph, &num_null_ph, &num_cyclosynch_ph_emit, &scatt_cyclosynch_num_ph, &all_time_steps, &sorted_indexes, max_photons, theta_jmin_thread, theta_jmax_thread, rng, fPtr);
}
if (num_cyclosynch_ph_emit>0)
{
n_comptonized-=phAbsCyclosynch(&phPtr, &num_ph, &frame_abs_cnt, &scatt_cyclosynch_num_ph, &hydrodata, fPtr);
}
}
#endif
phScattStats(phPtr, num_ph, &max_scatt, &min_scatt, &avg_scatt, &avg_r, fPtr);
fprintf(fPtr,"The number of scatterings in this frame is: %d\n", frame_scatt_cnt);
#if CYCLOSYNCHROTRON_SWITCH == ON
fprintf(fPtr,"The number of cyclosynchrotron photons absorbed in this frame is: %d\n", frame_abs_cnt);
#endif
fprintf(fPtr,"The last time step was: %e.\nThe time now is: %e\n", time_step,time_now);
fprintf(fPtr,"MCRaT had to refind the position of photons %d times in this frame.\n", num_photons_find_new_element);
fprintf(fPtr,"The maximum number of scatterings for a photon is: %d\nThe minimum number of scatterings for a photon is: %d\n", max_scatt, min_scatt);
fprintf(fPtr,"The average number of scatterings thus far is: %lf\nThe average position of photons is %e\n", avg_scatt, avg_r);
fflush(fPtr);
fprintf(fPtr, ">> Proc %d with angles %0.1lf-%0.1lf: Making checkpoint file\n", angle_id, theta_jmin_thread*180/M_PI, theta_jmax_thread*180/M_PI);
fflush(fPtr);
save_chkpt_success=saveCheckpoint(mc_dir, frame, frm2, scatt_frame, num_ph, time_now, phPtr, last_frm, angle_id, old_num_angle_procs);
if (save_chkpt_success==0)
{
printPhotons(phPtr, num_ph, frame_abs_cnt, num_cyclosynch_ph_emit, num_null_ph, scatt_cyclosynch_num_ph, scatt_frame , frame, last_frm, mc_dir, angle_id, fPtr);
}
else
{
fprintf(fPtr, "There is an issue with opening and saving the chkpt file therefore MCRaT is not saving data to the checkpoint or mc_proc files to prevent corruption of those data.\n");
printf("There is an issue with opening and saving the chkpt file therefore MCRaT is not saving data to the checkpoint or mc_proc files to prevent corruption of those data.\n");
fflush(fPtr);
exit(1);
}
#if SIM_SWITCH == RIKEN && DIMENSIONS == THREE
{
{
free(zPtr);free(phiPtr);free(velzPtr);
zPtr=NULL; phiPtr=NULL; velzPtr=NULL;
}
}
#endif
free(xPtr);free(yPtr);free(szxPtr);free(szyPtr);free(rPtr);free(thetaPtr);free(velxPtr);free(velyPtr);free(densPtr);free(presPtr);
free(gammaPtr);free(dens_labPtr);free(tempPtr);
xPtr=NULL; yPtr=NULL;  rPtr=NULL;thetaPtr=NULL;velxPtr=NULL;velyPtr=NULL;densPtr=NULL;presPtr=NULL;gammaPtr=NULL;dens_labPtr=NULL;
szxPtr=NULL; szyPtr=NULL; tempPtr=NULL;
free(all_time_steps); 
all_time_steps=NULL;
free(sorted_indexes);
sorted_indexes=NULL;
freeHydroDataFrame(&hydrodata);
}
restrt=INITALIZE;
scatt_cyclosynch_num_ph=0; 
num_null_ph=0; 
free(phPtr);
phPtr=NULL;
free(all_time_steps);
all_time_steps=NULL;
free(sorted_indexes);
sorted_indexes=NULL;
}
save_chkpt_success=saveCheckpoint(mc_dir, frame, frm2, scatt_frame, 0, time_now, phPtr, last_frm, angle_id, old_num_angle_procs); 
fprintf(fPtr, "Process %d has completed the MC calculation.\n", angle_id);
fflush(fPtr);
MPI_Barrier(angle_comm);
hydrodata.increment_scatt_frame=1;
file_count=0;
for (i=frm0;i<=last_frm;i=i+hydrodata.increment_scatt_frame)
{
#if SIM_SWITCH == RIKEN && DIMENSIONS == THREE
if (i>=3000)
{
hydrodata.increment_scatt_frame=10; 
}
#endif
file_count++;
}
MPI_Comm_size(angle_comm, &angle_procs); 
MPI_Comm_rank(angle_comm, &angle_id); 
proc_frame_size=floor(file_count/ (float) angle_procs);
frame_array=malloc(file_count*sizeof(int));
proc_frame_array=malloc(angle_procs*sizeof(int)); 
element_num=malloc(angle_procs*sizeof(int));
for (i=0;i<angle_procs;i++)
{
*(proc_frame_array+i)=i*proc_frame_size;
*(element_num+i)=1;
}
hydrodata.increment_scatt_frame=1;
file_count=0;
for (i=frm0;i<=last_frm;i=i+hydrodata.increment_scatt_frame)
{
#if SIM_SWITCH == RIKEN && DIMENSIONS == THREE
if (i>=3000)
{
hydrodata.increment_scatt_frame=10; 
}
#endif
*(frame_array+file_count)=i ;
file_count++;
}
MPI_Scatterv(frame_array, element_num, proc_frame_array, MPI_INT, &frm0, 1, MPI_INT, 0, angle_comm);
if (angle_id==angle_procs-1)
{
proc_frame_size=file_count-proc_frame_size*(angle_procs-1); 
}
i=0;
last_frm=frm0;
while(i<proc_frame_size)
{
#if SIM_SWITCH == RIKEN && DIMENSIONS == THREE
if (last_frm>=3000)
{
hydrodata.increment_scatt_frame=10; 
}
#else
{
hydrodata.increment_scatt_frame=1;
}
#endif
last_frm+=hydrodata.increment_scatt_frame;
i++;
}
fprintf(fPtr, ">> Proc %d with angles %0.1lf-%0.1lf: Merging Files from %d to %d\n", angle_id, theta_jmin_thread*180/M_PI, theta_jmax_thread*180/M_PI, frm0, last_frm);
fflush(fPtr);
dirFileMerge(mc_dir, frm0, last_frm, old_num_angle_procs, angle_id, fPtr);
fprintf(fPtr, "Process %d has completed merging files.\n", angle_id);
fflush(fPtr);
fclose(fPtr);
gsl_rng_free (rng);
free(frame_array);
free(proc_frame_array);
MPI_Finalize();
return 0;    
}
