#include "mcrat.h"
int getOrigNumProcesses(int *counted_cont_procs,  int **proc_array, char dir[STR_BUFFER], int angle_rank,  int angle_procs, int last_frame)
{
int i=0, j=0, val=0, original_num_procs=-1, rand_num=0;
int frame2=0, framestart=0, scatt_framestart=0, ph_num=0;
double time=0;
char mc_chkpt_files[STR_BUFFER]="", restrt=""; 
struct photon *phPtr=NULL; 
glob_t  files;
{
snprintf(mc_chkpt_files, sizeof(mc_chkpt_files), "%s%s", dir,"mc_chkpt_*" );
val=glob(mc_chkpt_files, 0, NULL,&files );
srand(angle_rank);
rand_num=rand() % files.gl_pathc;
snprintf(mc_chkpt_files, sizeof(mc_chkpt_files), "%s%s%d%s", dir,"mc_chkpt_",  rand_num,".dat" );
if ( access( mc_chkpt_files, F_OK ) == -1 )
{
while(( access( mc_chkpt_files, F_OK ) == -1 ) )
{
rand_num=rand() % files.gl_pathc;
snprintf(mc_chkpt_files, sizeof(mc_chkpt_files), "%s%s%d%s", dir,"mc_chkpt_",  rand_num,".dat" );
}
}
readCheckpoint(dir, &phPtr, &frame2, &framestart, &scatt_framestart, &ph_num, &restrt, &time, rand_num, &original_num_procs);
}
int count_procs[original_num_procs], count=0;
int cont_procs[original_num_procs];
for (j=0;j<original_num_procs;j++)
{
count_procs[j]=j;
cont_procs[j]=-1; 
}
int limit= (angle_rank != angle_procs-1) ? (angle_rank+1)*original_num_procs/angle_procs : original_num_procs;
printf("Angle ID: %d, start_num: %d, limit: %d\n", angle_rank, (angle_rank*original_num_procs/angle_procs),  limit);
count=0;
for (j=floor(angle_rank*original_num_procs/angle_procs);j<limit;j++)
{
snprintf(mc_chkpt_files, sizeof(mc_chkpt_files), "%s%s%d%s", dir,"mc_chkpt_",  j,".dat" );
if ( access( mc_chkpt_files, F_OK ) != -1 )
{
readCheckpoint(dir, &phPtr, &frame2, &framestart, &scatt_framestart, &ph_num, &restrt, &time, count_procs[j], &i);
free(phPtr);
phPtr=NULL;
if ((framestart<=frame2) && (scatt_framestart<=last_frame)) 
{
cont_procs[count]=j;
count++;
}
}
else
{
cont_procs[count]=j;
count++;
}
}
(*proc_array)=malloc (count * sizeof (int )); 
count=0;
for (i=0;i<original_num_procs;i++)
{
if (cont_procs[i]!=-1)
{
(*proc_array)[count]=cont_procs[i];
count++;
}
}
*counted_cont_procs=count;
globfree(& files);
return original_num_procs;
}
void printPhotons(struct photon *ph, int num_ph, int num_ph_abs, int num_cyclosynch_ph_emit, int num_null_ph, int scatt_cyclosynch_num_ph, int frame,int frame_inj, int frame_last, char dir[STR_BUFFER], int angle_rank, FILE *fPtr )
{
int i=0, count=0, rank=1, net_num_ph=num_ph-num_ph_abs-num_null_ph; 
#if defined(_OPENMP)
int num_thread=omp_get_num_threads();
#endif
char mc_file[STR_BUFFER]="", group[200]="", group_weight[200]="", *ph_type=NULL;
double p0[net_num_ph], p1[net_num_ph], p2[net_num_ph], p3[net_num_ph] , r0[net_num_ph], r1[net_num_ph], r2[net_num_ph], num_scatt[net_num_ph], weight[net_num_ph], global_weight[net_num_ph];
double s0[net_num_ph], s1[net_num_ph], s2[net_num_ph], s3[net_num_ph], comv_p0[net_num_ph], comv_p1[net_num_ph], comv_p2[net_num_ph], comv_p3[net_num_ph];
hid_t  file, file_init, dspace, dspace_weight, dspace_global_weight, fspace, mspace, prop, prop_weight, prop_global_weight, group_id;
hid_t dset_p0, dset_p1, dset_p2, dset_p3, dset_r0, dset_r1, dset_r2, dset_s0, dset_s1, dset_s2, dset_s3, dset_num_scatt, dset_weight, dset_weight_2, dset_comv_p0, dset_comv_p1, dset_comv_p2, dset_comv_p3, dset_ph_type;
herr_t status, status_group, status_weight, status_weight_2;
hsize_t dims[1]={net_num_ph}, dims_weight[1]={net_num_ph}, dims_old[1]={0}; 
hsize_t maxdims[1]={H5S_UNLIMITED};
hsize_t      size[1];
hsize_t      offset[1];
fprintf(fPtr, "num_ph %d num_ph_abs %d num_null_ph %d num_cyclosynch_ph_emit %d\nAllocated weight to be %d values large and other arrays to be %d\n",num_ph,num_ph_abs,num_null_ph,num_cyclosynch_ph_emit, net_num_ph, net_num_ph);
ph_type=malloc((net_num_ph)*sizeof(char));
count=0;
for (i=0;i<num_ph;i++)
{
if ((ph+i)->weight != 0)
{
p0[count]= ((ph+i)->p0);
p1[count]= ((ph+i)->p1);
p2[count]= ((ph+i)->p2);
p3[count]= ((ph+i)->p3);
r0[count]= ((ph+i)->r0);
r1[count]= ((ph+i)->r1);
r2[count]= ((ph+i)->r2);
#if COMV_SWITCH == ON
{
comv_p0[count]= ((ph+i)->comv_p0);
comv_p1[count]= ((ph+i)->comv_p1);
comv_p2[count]= ((ph+i)->comv_p2);
comv_p3[count]= ((ph+i)->comv_p3);
}
#endif
#if STOKES_SWITCH == ON
{
s0[count]= ((ph+i)->s0);
s1[count]= ((ph+i)->s1);
s2[count]= ((ph+i)->s2);
s3[count]= ((ph+i)->s3);
}
#endif
num_scatt[count]= ((ph+i)->num_scatt);
weight[count]= ((ph+i)->weight);
if ((frame==frame_last))
{
global_weight[count]=((ph+i)->weight);
}
*(ph_type+count)=(ph+i)->type;
count++;
}
}
snprintf(mc_file,sizeof(mc_file),"%s%s%d%s",dir,"mc_proc_", angle_rank, ".h5" );
snprintf(group,sizeof(group),"%d",frame );
status = H5Eset_auto(NULL, NULL, NULL); 
file_init=H5Fcreate(mc_file, H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT); 
file=file_init;
status = H5Eset_auto(H5E_DEFAULT, H5Eprint2, stderr); 
if (file_init<0)
{
file=H5Fopen(mc_file, H5F_ACC_RDWR, H5P_DEFAULT);
status = H5Eset_auto(NULL, NULL, NULL);
status_group = H5Gget_objinfo (file, group, 0, NULL);
status = H5Eset_auto(H5E_DEFAULT, H5Eprint2, stderr);
}
if ((file_init>=0) || (status_group != 0) )
{
status = H5Eset_auto(NULL, NULL, NULL);
status_weight = H5Gget_objinfo (file, "/PW", 0, NULL);
status = H5Eset_auto(H5E_DEFAULT, H5Eprint2, stderr);
fprintf(fPtr,"Status of /PW %d\n", status_weight);
group_id = H5Gcreate2(file, group, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
prop = H5Pcreate (H5P_DATASET_CREATE);
status = H5Pset_chunk (prop, rank, dims);
prop_weight= H5Pcreate (H5P_DATASET_CREATE);
status = H5Pset_chunk (prop_weight, rank, dims_weight);
dspace = H5Screate_simple (rank, dims, maxdims);
dspace_weight=H5Screate_simple (rank, dims_weight, maxdims);
dset_p0 = H5Dcreate2 (group_id, "P0", H5T_NATIVE_DOUBLE, dspace,
H5P_DEFAULT, prop, H5P_DEFAULT);
dset_p1 = H5Dcreate2 (group_id, "P1", H5T_NATIVE_DOUBLE, dspace,
H5P_DEFAULT, prop, H5P_DEFAULT);
dset_p2 = H5Dcreate2 (group_id, "P2", H5T_NATIVE_DOUBLE, dspace,
H5P_DEFAULT, prop, H5P_DEFAULT);
dset_p3 = H5Dcreate2 (group_id, "P3", H5T_NATIVE_DOUBLE, dspace,
H5P_DEFAULT, prop, H5P_DEFAULT);
#if COMV_SWITCH == ON
{
dset_comv_p0 = H5Dcreate2 (group_id, "COMV_P0", H5T_NATIVE_DOUBLE, dspace,
H5P_DEFAULT, prop, H5P_DEFAULT);
dset_comv_p1 = H5Dcreate2 (group_id, "COMV_P1", H5T_NATIVE_DOUBLE, dspace,
H5P_DEFAULT, prop, H5P_DEFAULT);
dset_comv_p2 = H5Dcreate2 (group_id, "COMV_P2", H5T_NATIVE_DOUBLE, dspace,
H5P_DEFAULT, prop, H5P_DEFAULT);
dset_comv_p3 = H5Dcreate2 (group_id, "COMV_P3", H5T_NATIVE_DOUBLE, dspace,
H5P_DEFAULT, prop, H5P_DEFAULT);
}
#endif
dset_r0 = H5Dcreate2 (group_id, "R0", H5T_NATIVE_DOUBLE, dspace,
H5P_DEFAULT, prop, H5P_DEFAULT);
dset_r1 = H5Dcreate2 (group_id, "R1", H5T_NATIVE_DOUBLE, dspace,
H5P_DEFAULT, prop, H5P_DEFAULT);
dset_r2 = H5Dcreate2 (group_id, "R2", H5T_NATIVE_DOUBLE, dspace,
H5P_DEFAULT, prop, H5P_DEFAULT);
#if STOKES_SWITCH == ON
{
dset_s0 = H5Dcreate2 (group_id, "S0", H5T_NATIVE_DOUBLE, dspace,
H5P_DEFAULT, prop, H5P_DEFAULT);
dset_s1 = H5Dcreate2 (group_id, "S1", H5T_NATIVE_DOUBLE, dspace,
H5P_DEFAULT, prop, H5P_DEFAULT);
dset_s2 = H5Dcreate2 (group_id, "S2", H5T_NATIVE_DOUBLE, dspace,
H5P_DEFAULT, prop, H5P_DEFAULT);
dset_s3 = H5Dcreate2 (group_id, "S3", H5T_NATIVE_DOUBLE, dspace,
H5P_DEFAULT, prop, H5P_DEFAULT);
}
#endif
#if SAVE_TYPE == ON
{
dset_ph_type = H5Dcreate2 (group_id, "PT", H5T_NATIVE_CHAR, dspace,
H5P_DEFAULT, prop, H5P_DEFAULT);
}
#endif
dset_num_scatt = H5Dcreate2 (group_id, "NS", H5T_NATIVE_DOUBLE, dspace,
H5P_DEFAULT, prop, H5P_DEFAULT);
dset_weight_2 = H5Dcreate2 (group_id, "PW", H5T_NATIVE_DOUBLE, dspace_weight,
H5P_DEFAULT, prop_weight, H5P_DEFAULT); 
status = H5Dwrite (dset_p0, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
H5P_DEFAULT, p0);
status = H5Dwrite (dset_p1, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
H5P_DEFAULT, p1);
status = H5Dwrite (dset_p2, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
H5P_DEFAULT, p2);
status = H5Dwrite (dset_p3, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
H5P_DEFAULT, p3);
#if COMV_SWITCH == ON
{
status = H5Dwrite (dset_comv_p0, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
H5P_DEFAULT, comv_p0);
status = H5Dwrite (dset_comv_p1, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
H5P_DEFAULT, comv_p1);
status = H5Dwrite (dset_comv_p2, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
H5P_DEFAULT, comv_p2);
status = H5Dwrite (dset_comv_p3, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
H5P_DEFAULT, comv_p3);
}
#endif
status = H5Dwrite (dset_r0, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
H5P_DEFAULT, r0);
status = H5Dwrite (dset_r1, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
H5P_DEFAULT, r1);
status = H5Dwrite (dset_r2, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
H5P_DEFAULT, r2);
#if STOKES_SWITCH == ON
{
status = H5Dwrite (dset_s0, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
H5P_DEFAULT, s0);
status = H5Dwrite (dset_s1, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
H5P_DEFAULT, s1);
status = H5Dwrite (dset_s2, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
H5P_DEFAULT, s2);
status = H5Dwrite (dset_s3, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
H5P_DEFAULT, s3);
}
#endif
#if SAVE_TYPE == ON
{
status = H5Dwrite (dset_ph_type, H5T_NATIVE_CHAR, H5S_ALL, H5S_ALL,
H5P_DEFAULT, ph_type);
}
#endif
status = H5Dwrite (dset_num_scatt, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
H5P_DEFAULT, num_scatt);
status = H5Dwrite (dset_weight_2, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
H5P_DEFAULT, weight);
status = H5Pclose (prop_weight);
status = H5Dclose (dset_weight_2);
status = H5Pclose (prop);
}
else
{
group_id = H5Gopen2(file, group, H5P_DEFAULT);
dset_p0 = H5Dopen (group_id, "P0", H5P_DEFAULT); 
dspace = H5Dget_space (dset_p0);
status=H5Sget_simple_extent_dims(dspace, dims_old, NULL); 
size[0] = dims[0]+ dims_old[0];
status = H5Dset_extent (dset_p0, size);
fspace = H5Dget_space (dset_p0);
offset[0] = dims_old[0];
status = H5Sselect_hyperslab (fspace, H5S_SELECT_SET, offset, NULL,
dims, NULL);
mspace = H5Screate_simple (rank, dims, NULL);
status = H5Dwrite (dset_p0, H5T_NATIVE_DOUBLE, mspace, fspace,
H5P_DEFAULT, p0);
status = H5Sclose (dspace);
status = H5Sclose (mspace);
status = H5Sclose (fspace);
dset_p1 = H5Dopen (group_id, "P1", H5P_DEFAULT); 
dspace = H5Dget_space (dset_p1);
status=H5Sget_simple_extent_dims(dspace, dims_old, NULL); 
size[0] = dims[0]+ dims_old[0];
status = H5Dset_extent (dset_p1, size);
fspace = H5Dget_space (dset_p1);
offset[0] = dims_old[0];
status = H5Sselect_hyperslab (fspace, H5S_SELECT_SET, offset, NULL,
dims, NULL);
mspace = H5Screate_simple (rank, dims, NULL);
status = H5Dwrite (dset_p1, H5T_NATIVE_DOUBLE, mspace, fspace,
H5P_DEFAULT, p1);
status = H5Sclose (dspace);
status = H5Sclose (mspace);
status = H5Sclose (fspace);
dset_p2 = H5Dopen (group_id, "P2", H5P_DEFAULT); 
dspace = H5Dget_space (dset_p2);
status=H5Sget_simple_extent_dims(dspace, dims_old, NULL); 
size[0] = dims[0]+ dims_old[0];
status = H5Dset_extent (dset_p2, size);
fspace = H5Dget_space (dset_p2);
offset[0] = dims_old[0];
status = H5Sselect_hyperslab (fspace, H5S_SELECT_SET, offset, NULL,
dims, NULL);
mspace = H5Screate_simple (rank, dims, NULL);
status = H5Dwrite (dset_p2, H5T_NATIVE_DOUBLE, mspace, fspace,
H5P_DEFAULT, p2);
status = H5Sclose (dspace);
status = H5Sclose (mspace);
status = H5Sclose (fspace);
dset_p3 = H5Dopen (group_id, "P3", H5P_DEFAULT); 
dspace = H5Dget_space (dset_p3);
status=H5Sget_simple_extent_dims(dspace, dims_old, NULL); 
size[0] = dims[0]+ dims_old[0];
status = H5Dset_extent (dset_p3, size);
fspace = H5Dget_space (dset_p3);
offset[0] = dims_old[0];
status = H5Sselect_hyperslab (fspace, H5S_SELECT_SET, offset, NULL,
dims, NULL);
mspace = H5Screate_simple (rank, dims, NULL);
status = H5Dwrite (dset_p3, H5T_NATIVE_DOUBLE, mspace, fspace,
H5P_DEFAULT, p3);
status = H5Sclose (dspace);
status = H5Sclose (mspace);
status = H5Sclose (fspace);
#if COMV_SWITCH == ON
{
dset_comv_p0 = H5Dopen (group_id, "COMV_P0", H5P_DEFAULT); 
dspace = H5Dget_space (dset_comv_p0);
status=H5Sget_simple_extent_dims(dspace, dims_old, NULL); 
size[0] = dims[0]+ dims_old[0];
status = H5Dset_extent (dset_comv_p0, size);
fspace = H5Dget_space (dset_comv_p0);
offset[0] = dims_old[0];
status = H5Sselect_hyperslab (fspace, H5S_SELECT_SET, offset, NULL,
dims, NULL);
mspace = H5Screate_simple (rank, dims, NULL);
status = H5Dwrite (dset_comv_p0, H5T_NATIVE_DOUBLE, mspace, fspace,
H5P_DEFAULT, comv_p0);
status = H5Sclose (dspace);
status = H5Sclose (mspace);
status = H5Sclose (fspace);
dset_comv_p1 = H5Dopen (group_id, "COMV_P1", H5P_DEFAULT); 
dspace = H5Dget_space (dset_comv_p1);
status=H5Sget_simple_extent_dims(dspace, dims_old, NULL); 
size[0] = dims[0]+ dims_old[0];
status = H5Dset_extent (dset_comv_p1, size);
fspace = H5Dget_space (dset_comv_p1);
offset[0] = dims_old[0];
status = H5Sselect_hyperslab (fspace, H5S_SELECT_SET, offset, NULL,
dims, NULL);
mspace = H5Screate_simple (rank, dims, NULL);
status = H5Dwrite (dset_comv_p1, H5T_NATIVE_DOUBLE, mspace, fspace,
H5P_DEFAULT, comv_p1);
status = H5Sclose (dspace);
status = H5Sclose (mspace);
status = H5Sclose (fspace);
dset_comv_p2 = H5Dopen (group_id, "COMV_P2", H5P_DEFAULT); 
dspace = H5Dget_space (dset_comv_p2);
status=H5Sget_simple_extent_dims(dspace, dims_old, NULL); 
size[0] = dims[0]+ dims_old[0];
status = H5Dset_extent (dset_comv_p2, size);
fspace = H5Dget_space (dset_comv_p2);
offset[0] = dims_old[0];
status = H5Sselect_hyperslab (fspace, H5S_SELECT_SET, offset, NULL,
dims, NULL);
mspace = H5Screate_simple (rank, dims, NULL);
status = H5Dwrite (dset_comv_p2, H5T_NATIVE_DOUBLE, mspace, fspace,
H5P_DEFAULT, comv_p2);
status = H5Sclose (dspace);
status = H5Sclose (mspace);
status = H5Sclose (fspace);
dset_comv_p3 = H5Dopen (group_id, "COMV_P3", H5P_DEFAULT); 
dspace = H5Dget_space (dset_comv_p3);
status=H5Sget_simple_extent_dims(dspace, dims_old, NULL); 
size[0] = dims[0]+ dims_old[0];
status = H5Dset_extent (dset_comv_p3, size);
fspace = H5Dget_space (dset_comv_p3);
offset[0] = dims_old[0];
status = H5Sselect_hyperslab (fspace, H5S_SELECT_SET, offset, NULL,
dims, NULL);
mspace = H5Screate_simple (rank, dims, NULL);
status = H5Dwrite (dset_comv_p3, H5T_NATIVE_DOUBLE, mspace, fspace,
H5P_DEFAULT, comv_p3);
status = H5Sclose (dspace);
status = H5Sclose (mspace);
status = H5Sclose (fspace);
}
#endif
dset_r0 = H5Dopen (group_id, "R0", H5P_DEFAULT); 
dspace = H5Dget_space (dset_r0);
status=H5Sget_simple_extent_dims(dspace, dims_old, NULL); 
size[0] = dims[0]+ dims_old[0];
status = H5Dset_extent (dset_r0, size);
fspace = H5Dget_space (dset_r0);
offset[0] = dims_old[0];
status = H5Sselect_hyperslab (fspace, H5S_SELECT_SET, offset, NULL,
dims, NULL);
mspace = H5Screate_simple (rank, dims, NULL);
status = H5Dwrite (dset_r0, H5T_NATIVE_DOUBLE, mspace, fspace,
H5P_DEFAULT, r0);
status = H5Sclose (dspace);
status = H5Sclose (mspace);
status = H5Sclose (fspace);
dset_r1 = H5Dopen (group_id, "R1", H5P_DEFAULT); 
dspace = H5Dget_space (dset_r1);
status=H5Sget_simple_extent_dims(dspace, dims_old, NULL); 
size[0] = dims[0]+ dims_old[0];
status = H5Dset_extent (dset_r1, size);
fspace = H5Dget_space (dset_r1);
offset[0] = dims_old[0];
status = H5Sselect_hyperslab (fspace, H5S_SELECT_SET, offset, NULL,
dims, NULL);
mspace = H5Screate_simple (rank, dims, NULL);
status = H5Dwrite (dset_r1, H5T_NATIVE_DOUBLE, mspace, fspace,
H5P_DEFAULT, r1);
status = H5Sclose (dspace);
status = H5Sclose (mspace);
status = H5Sclose (fspace);
dset_r2 = H5Dopen (group_id, "R2", H5P_DEFAULT); 
dspace = H5Dget_space (dset_r2);
status=H5Sget_simple_extent_dims(dspace, dims_old, NULL); 
size[0] = dims[0]+ dims_old[0];
status = H5Dset_extent (dset_r2, size);
fspace = H5Dget_space (dset_r2);
offset[0] = dims_old[0];
status = H5Sselect_hyperslab (fspace, H5S_SELECT_SET, offset, NULL,
dims, NULL);
mspace = H5Screate_simple (rank, dims, NULL);
status = H5Dwrite (dset_r2, H5T_NATIVE_DOUBLE, mspace, fspace,
H5P_DEFAULT, r2);
status = H5Sclose (dspace);
status = H5Sclose (mspace);
status = H5Sclose (fspace);
#if STOKES_SWITCH == ON
{
dset_s0 = H5Dopen (group_id, "S0", H5P_DEFAULT); 
dspace = H5Dget_space (dset_s0);
status=H5Sget_simple_extent_dims(dspace, dims_old, NULL); 
size[0] = dims[0]+ dims_old[0];
status = H5Dset_extent (dset_s0, size);
fspace = H5Dget_space (dset_s0);
offset[0] = dims_old[0];
status = H5Sselect_hyperslab (fspace, H5S_SELECT_SET, offset, NULL,
dims, NULL);
mspace = H5Screate_simple (rank, dims, NULL);
status = H5Dwrite (dset_s0, H5T_NATIVE_DOUBLE, mspace, fspace,
H5P_DEFAULT, s0);
status = H5Sclose (dspace);
status = H5Sclose (mspace);
status = H5Sclose (fspace);
dset_s1 = H5Dopen (group_id, "S1", H5P_DEFAULT); 
dspace = H5Dget_space (dset_s1);
status=H5Sget_simple_extent_dims(dspace, dims_old, NULL); 
size[0] = dims[0]+ dims_old[0];
status = H5Dset_extent (dset_s1, size);
fspace = H5Dget_space (dset_s1);
offset[0] = dims_old[0];
status = H5Sselect_hyperslab (fspace, H5S_SELECT_SET, offset, NULL,
dims, NULL);
mspace = H5Screate_simple (rank, dims, NULL);
status = H5Dwrite (dset_s1, H5T_NATIVE_DOUBLE, mspace, fspace,
H5P_DEFAULT, s1);
status = H5Sclose (dspace);
status = H5Sclose (mspace);
status = H5Sclose (fspace);
dset_s2 = H5Dopen (group_id, "S2", H5P_DEFAULT); 
dspace = H5Dget_space (dset_s2);
status=H5Sget_simple_extent_dims(dspace, dims_old, NULL); 
size[0] = dims[0]+ dims_old[0];
status = H5Dset_extent (dset_s2, size);
fspace = H5Dget_space (dset_s2);
offset[0] = dims_old[0];
status = H5Sselect_hyperslab (fspace, H5S_SELECT_SET, offset, NULL,
dims, NULL);
mspace = H5Screate_simple (rank, dims, NULL);
status = H5Dwrite (dset_s2, H5T_NATIVE_DOUBLE, mspace, fspace,
H5P_DEFAULT, s2);
status = H5Sclose (dspace);
status = H5Sclose (mspace);
status = H5Sclose (fspace);
dset_s3 = H5Dopen (group_id, "S3", H5P_DEFAULT); 
dspace = H5Dget_space (dset_s3);
status=H5Sget_simple_extent_dims(dspace, dims_old, NULL); 
size[0] = dims[0]+ dims_old[0];
status = H5Dset_extent (dset_s3, size);
fspace = H5Dget_space (dset_s3);
offset[0] = dims_old[0];
status = H5Sselect_hyperslab (fspace, H5S_SELECT_SET, offset, NULL,
dims, NULL);
mspace = H5Screate_simple (rank, dims, NULL);
status = H5Dwrite (dset_s3, H5T_NATIVE_DOUBLE, mspace, fspace,
H5P_DEFAULT, s3);
status = H5Sclose (dspace);
status = H5Sclose (mspace);
status = H5Sclose (fspace);
}
#endif
#if SAVE_TYPE == ON
{
dset_ph_type = H5Dopen (group_id, "PT", H5P_DEFAULT); 
dspace = H5Dget_space (dset_ph_type);
status=H5Sget_simple_extent_dims(dspace, dims_old, NULL); 
size[0] = dims[0]+ dims_old[0];
status = H5Dset_extent (dset_ph_type, size);
fspace = H5Dget_space (dset_ph_type);
offset[0] = dims_old[0];
status = H5Sselect_hyperslab (fspace, H5S_SELECT_SET, offset, NULL,
dims, NULL);
mspace = H5Screate_simple (rank, dims, NULL);
status = H5Dwrite (dset_ph_type, H5T_NATIVE_CHAR, mspace, fspace,
H5P_DEFAULT, ph_type);
status = H5Sclose (dspace);
status = H5Sclose (mspace);
status = H5Sclose (fspace);
}
#endif
dset_num_scatt = H5Dopen (group_id, "NS", H5P_DEFAULT); 
dspace = H5Dget_space (dset_num_scatt);
status=H5Sget_simple_extent_dims(dspace, dims_old, NULL); 
size[0] = dims[0]+ dims_old[0];
status = H5Dset_extent (dset_num_scatt, size);
fspace = H5Dget_space (dset_num_scatt);
offset[0] = dims_old[0];
status = H5Sselect_hyperslab (fspace, H5S_SELECT_SET, offset, NULL,
dims, NULL);
mspace = H5Screate_simple (rank, dims, NULL);
status = H5Dwrite (dset_num_scatt, H5T_NATIVE_DOUBLE, mspace, fspace,
H5P_DEFAULT, num_scatt);
snprintf(group_weight,sizeof(group_weight),"PW",i );
status = H5Eset_auto(NULL, NULL, NULL);
status_weight = H5Gget_objinfo (group_id, "PW", 0, NULL);
status = H5Eset_auto(H5E_DEFAULT, H5Eprint2, stderr);
fprintf(fPtr,"Status of /frame/PW %d\n", status_weight);
status = H5Sclose (dspace);
status = H5Sclose (mspace);
status = H5Sclose (fspace);
if (status_weight >= 0)
{
status = H5Eset_auto(NULL, NULL, NULL);
status_weight_2 = H5Gget_objinfo (group_id, "PW", 0, NULL);
status = H5Eset_auto(H5E_DEFAULT, H5Eprint2, stderr);
if (status_weight_2 < 0)
{
prop = H5Pcreate (H5P_DATASET_CREATE);
status = H5Pset_chunk (prop, rank, dims);
dspace = H5Screate_simple (rank, dims, maxdims);
dset_weight_2 = H5Dcreate2 (group_id, "PW", H5T_NATIVE_DOUBLE, dspace,
H5P_DEFAULT, prop, H5P_DEFAULT); 
status = H5Dwrite (dset_weight_2, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
H5P_DEFAULT, weight);
status = H5Pclose (prop);
}
else
{
dset_weight_2 = H5Dopen (group_id, "PW", H5P_DEFAULT); 
dspace = H5Dget_space (dset_weight_2);
status=H5Sget_simple_extent_dims(dspace, dims_old, NULL); 
size[0] = dims_weight[0]+ dims_old[0];
status = H5Dset_extent (dset_weight_2, size);
fspace = H5Dget_space (dset_weight_2);
offset[0] = dims_old[0];
status = H5Sselect_hyperslab (fspace, H5S_SELECT_SET, offset, NULL,
dims_weight, NULL);
mspace = H5Screate_simple (rank, dims_weight, NULL);
status = H5Dwrite (dset_weight_2, H5T_NATIVE_DOUBLE, mspace, fspace,
H5P_DEFAULT, weight);
}
}
else
{
fprintf(fPtr, "The frame exists in the hdf5 file but the weight dataset for the frame doesnt exist, therefore creating it.\n");
fflush(fPtr);
prop_weight= H5Pcreate (H5P_DATASET_CREATE);
status = H5Pset_chunk (prop_weight, rank, dims_weight);
dspace_weight=H5Screate_simple (rank, dims_weight, maxdims);
dset_weight_2 = H5Dcreate2 (group_id, "PW", H5T_NATIVE_DOUBLE, dspace_weight,
H5P_DEFAULT, prop_weight, H5P_DEFAULT);
status = H5Dwrite (dset_weight_2, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
H5P_DEFAULT, weight);
status = H5Pclose (prop_weight);
}
status = H5Dclose (dset_weight_2);
status = H5Sclose (dspace);
status = H5Sclose (mspace);
status = H5Sclose (fspace);
}
free(ph_type);
status = H5Dclose (dset_p0); status = H5Dclose (dset_p1); status = H5Dclose (dset_p2); status = H5Dclose (dset_p3);
#if COMV_SWITCH == ON
{
status = H5Dclose (dset_comv_p0); status = H5Dclose (dset_comv_p1); status = H5Dclose (dset_comv_p2); status = H5Dclose (dset_comv_p3);
}
#endif
status = H5Dclose (dset_r0); status = H5Dclose (dset_r1); status = H5Dclose (dset_r2);
#if STOKES_SWITCH == ON
{
status = H5Dclose (dset_s0); status = H5Dclose (dset_s1); status = H5Dclose (dset_s2); status = H5Dclose (dset_s3);
}
#endif
#if SAVE_TYPE == ON
{
status = H5Dclose (dset_ph_type);
}
#endif
status = H5Dclose (dset_num_scatt);
status = H5Gclose(group_id);
status = H5Fclose(file);
}
int saveCheckpoint(char dir[STR_BUFFER], int frame, int frame2, int scatt_frame, int ph_num,double time_now, struct photon *ph, int last_frame, int angle_rank,int angle_size )
{
FILE *fPtr=NULL;
char checkptfile[2000]="";
char command[2000]="";
char restart;
int i=0, success=0;
snprintf(checkptfile,sizeof(checkptfile),"%s%s%d%s",dir,"mc_chkpt_", angle_rank,".dat" );
if ((scatt_frame!=last_frame) && (scatt_frame != frame))
{
snprintf(command, sizeof(command), "%s%s %s_old","exec cp ",checkptfile, checkptfile);
system(command);
fPtr=fopen(checkptfile, "wb");
if (fPtr==NULL)
{
printf("Cannot open %s to save checkpoint\n", checkptfile);
success=1;
}
else
{
fwrite(&angle_size, sizeof(int), 1, fPtr);
restart=CONTINUE;
fwrite(&restart, sizeof(char), 1, fPtr);
fflush(stdout);
fwrite(&frame, sizeof(int), 1, fPtr);
fflush(stdout);
fwrite(&frame2, sizeof(int), 1, fPtr);
fflush(stdout);
fwrite(&scatt_frame, sizeof(int), 1, fPtr);
fflush(stdout);
fwrite(&time_now, sizeof(double), 1, fPtr);
fflush(stdout);
fwrite(&ph_num, sizeof(int), 1, fPtr);
fflush(stdout);
for(i=0;i<ph_num;i++)
{
#if CYCLOSYNCHROTRON_SWITCH == ON
if (((ph+i)->type == COMPTONIZED_PHOTON) && ((ph+i)->weight != 0))
{
(ph+i)->type = UNABSORBED_CS_PHOTON; 
}
#endif
fwrite((ph+i), sizeof(struct photon ), 1, fPtr);
}
success=0;
}
fflush(stdout);
}
else if  (scatt_frame == frame)
{
snprintf(command, sizeof(command), "%s%s","exec rm ",checkptfile);
system(command);
fPtr=fopen(checkptfile, "wb");
fflush(stdout);
if (fPtr==NULL)
{
printf("Cannot open %s to save checkpoint\n", checkptfile);
success=1;
}
else
{
fwrite(&angle_size, sizeof(int), 1, fPtr);
restart=CONTINUE;
fwrite(&restart, sizeof(char), 1, fPtr);
fflush(stdout);
fwrite(&frame, sizeof(int), 1, fPtr);
fflush(stdout);
fwrite(&frame2, sizeof(int), 1, fPtr);
fflush(stdout);
fwrite(&scatt_frame, sizeof(int), 1, fPtr);
fflush(stdout);
fwrite(&time_now, sizeof(double), 1, fPtr);
fflush(stdout);
fwrite(&ph_num, sizeof(int), 1, fPtr);
fflush(stdout);
for(i=0;i<ph_num;i++)
{
#if CYCLOSYNCHROTRON_SWITCH == ON
if (((ph+i)->type == COMPTONIZED_PHOTON) && ((ph+i)->weight != 0))
{
(ph+i)->type = UNABSORBED_CS_PHOTON; 
}
#endif
fwrite((ph+i), sizeof(struct photon ), 1, fPtr);
}
success=0;
}
fflush(stdout);
}
else
{
snprintf(command, sizeof(command), "%s%s %s_old","exec cp ",checkptfile, checkptfile);
system(command);
fPtr=fopen(checkptfile, "wb");
if (fPtr==NULL)
{
printf("Cannot open %s to save checkpoint\n", checkptfile);
success=1;
}
else
{
fwrite(&angle_size, sizeof(int), 1, fPtr);
restart=INITALIZE;
fwrite(&restart, sizeof(char), 1, fPtr);
fwrite(&frame, sizeof(int), 1, fPtr);
fwrite(&frame2, sizeof(int), 1, fPtr);
for(i=0;i<ph_num;i++)
{
#if CYCLOSYNCHROTRON_SWITCH == ON
if (((ph+i)->type == COMPTONIZED_PHOTON) && ((ph+i)->weight != 0))
{
(ph+i)->type = UNABSORBED_CS_PHOTON; 
}
#endif
fwrite((ph+i), sizeof(struct photon ), 1, fPtr);
}
success=0;
}
}
if (success==0)
{
fclose(fPtr);
}
return success;
}
int readCheckpoint(char dir[STR_BUFFER], struct photon **ph, int *frame2, int *framestart, int *scatt_framestart, int *ph_num, char *restart, double *time, int angle_rank, int *angle_size )
{
FILE *fPtr=NULL;
char checkptfile[STR_BUFFER]="";
int i=0;
int scatt_cyclosynch_num_ph=0;
struct photon *phHolder=NULL; 
snprintf(checkptfile,sizeof(checkptfile),"%s%s%d%s",dir,"mc_chkpt_", angle_rank,".dat" );
printf("Checkpoint file: %s\n", checkptfile);
if (access( checkptfile, F_OK ) != -1) 
{
fPtr=fopen(checkptfile, "rb");
{
fread(angle_size, sizeof(int), 1, fPtr); 
}
fread(restart, sizeof(char), 1, fPtr);
fread(framestart, sizeof(int), 1, fPtr);
fread(frame2, sizeof(int), 1, fPtr);
if((*restart)==CONTINUE)
{
fread(scatt_framestart, sizeof(int), 1, fPtr);
#if SIM_SWITCH == RIKEN && DIMENSIONS == THREE
if ((*scatt_framestart)>=3000)
{
*scatt_framestart+=10; 
}
#else
{
*scatt_framestart+=1; 
}
#endif
fread(time, sizeof(double), 1, fPtr);
fread(ph_num, sizeof(int), 1, fPtr);
phHolder=malloc(sizeof(struct photon));
(*ph)=malloc(sizeof(struct photon)*(*ph_num)); 
for (i=0;i<(*ph_num);i++)
{
fread(phHolder, sizeof(struct photon), 1, fPtr);
(*ph)[i].p0=phHolder->p0;
(*ph)[i].p1=phHolder->p1;
(*ph)[i].p2=phHolder->p2;
(*ph)[i].p3=phHolder->p3;
(*ph)[i].comv_p0=phHolder->comv_p0;
(*ph)[i].comv_p1=phHolder->comv_p1;
(*ph)[i].comv_p2=phHolder->comv_p2;
(*ph)[i].comv_p3=phHolder->comv_p3;
(*ph)[i].r0= phHolder->r0;
(*ph)[i].r1=phHolder->r1 ;
(*ph)[i].r2=phHolder->r2;
(*ph)[i].s0=phHolder->s0;
(*ph)[i].s1=phHolder->s1;
(*ph)[i].s2=phHolder->s2;
(*ph)[i].s3=phHolder->s3;
(*ph)[i].num_scatt=phHolder->num_scatt;
(*ph)[i].weight=phHolder->weight;
(*ph)[i].nearest_block_index= phHolder->nearest_block_index;
(*ph)[i].type= phHolder->type;
#if CYCLOSYNCHROTRON_SWITCH == ON
if (((*ph)[i].weight != 0) && (((*ph)[i].type == COMPTONIZED_PHOTON) || ((*ph)[i].type == UNABSORBED_CS_PHOTON)) && ((*ph)[i].p0 > 0))
{
scatt_cyclosynch_num_ph++;
}
#endif
}
free(phHolder);
}
else
{
#if SIM_SWITCH == RIKEN && DIMENSIONS == THREE
if ((*framestart)>=3000)
{
*framestart+=10; 
}
#else
{
*framestart+=1; 
}
#endif
*scatt_framestart=(*framestart);
}
fclose(fPtr);
}
else 
{
*scatt_framestart=(*framestart);
*restart=INITALIZE;
}
return scatt_cyclosynch_num_ph;
}
void readMcPar(struct hydro_dataframe *hydro_data, double *theta_jmin, double *theta_j, double *n_theta_j, double **inj_radius, int **frm0, int **frm2, int *min_photons, int *max_photons, char *spect, char *restart)
{
char mc_file[STR_BUFFER]="" ;
FILE *fptr=NULL;
char buf[100]="", buf2[100]="", *value, *context = NULL, copied_str[100]="";
double theta_deg;
int i, val;
snprintf(mc_file,sizeof(mc_file),"%s%s%s",FILEPATH, MC_PATH,MCPAR);
printf(">> MCRaT:  Reading parameter file %s\n", mc_file);
fptr=fopen(mc_file,"r");
fgets(buf, sizeof(buf), fptr); 
fgets(buf, sizeof(buf),fptr); 
fscanf(fptr, "%lf", &(hydro_data->fps));
fgets(buf, sizeof(buf),fptr); 
fscanf(fptr, "%d",&(hydro_data->last_frame));
fgets(buf, sizeof(buf),fptr); 
fscanf(fptr, "%lf", &((hydro_data->r0_domain)[0]) );
fscanf(fptr, "%lf", &((hydro_data->r0_domain)[1]) );
fgets(buf, sizeof(buf),fptr); 
fscanf(fptr, "%lf", &((hydro_data->r1_domain)[0]));
fscanf(fptr, "%lf", &((hydro_data->r1_domain)[1]));
fgets(buf, sizeof(buf),fptr); 
fscanf(fptr, "%lf", &((hydro_data->r2_domain)[0]));
fscanf(fptr, "%lf", &((hydro_data->r2_domain)[1]));
fgets(buf, sizeof(buf),fptr); 
fgets(buf, sizeof(buf),fptr); 
fgets(buf, sizeof(buf),fptr); 
fscanf(fptr, "%lf",&theta_deg);
*theta_jmin=theta_deg;
fgets(buf, sizeof(buf),fptr);
fscanf(fptr, "%lf",&theta_deg);
*theta_j=theta_deg;
fgets(buf, sizeof(buf),fptr);
fscanf(fptr, "%lf",&theta_deg);
*n_theta_j=theta_deg;
fgets(buf, sizeof(buf),fptr); 
(*inj_radius)=malloc( ((int) *n_theta_j)*sizeof(double) );
(*frm0)=malloc(((int) *n_theta_j)*sizeof(int));
(*frm2)=malloc(((int) *n_theta_j)*sizeof(int));
fgets(buf, sizeof(buf),fptr); 
value = strtok_r(buf, " ", &context);
for (i=0;i< (int) *n_theta_j;i++)
{
strcpy(copied_str, value);
(*frm0)[i]=strtol(copied_str, buf2, 10);
value = strtok_r(NULL, " ", &context);
}
fgets(buf, sizeof(buf),fptr); 
value = strtok_r(buf, " ", &context);
for (i=0;i< (int) *n_theta_j;i++)
{
strcpy(copied_str, value);
(*frm2)[i]=strtol(copied_str, buf2, 10)+(*frm0)[i];
value = strtok_r(NULL, " ", &context);
}
fgets(buf, sizeof(buf),fptr); 
value = strtok_r(buf, " ", &context);
for (i=0;i< (int) *n_theta_j;i++)
{
strcpy(copied_str, value);
(*inj_radius)[i]=strtof(copied_str, NULL);
value = strtok_r(NULL, " ", &context);
}
fgets(buf, sizeof(buf),fptr); 
fgets(buf, sizeof(buf),fptr); 
fgets(buf, sizeof(buf),fptr); 
*spect=getc(fptr);
fgets(buf, sizeof(buf),fptr); 
fscanf(fptr, "%d",min_photons);
fgets(buf, 100,fptr);
fscanf(fptr, "%d",max_photons);
fgets(buf, 100,fptr);
fgets(buf, 100,fptr);
fgets(buf, sizeof(buf),fptr); 
fgets(buf, sizeof(buf),fptr); 
*restart=getc(fptr);
fgets(buf, 100,fptr);
fclose(fptr);
}
void dirFileMerge(char dir[STR_BUFFER], int start_frame, int last_frame, int numprocs, int angle_id, FILE *fPtr )
{
double *p0=NULL, *p1=NULL, *p2=NULL, *p3=NULL, *comv_p0=NULL, *comv_p1=NULL, *comv_p2=NULL, *comv_p3=NULL, *r0=NULL, *r1=NULL, *r2=NULL, *s0=NULL, *s1=NULL, *s2=NULL, *s3=NULL, *num_scatt=NULL, *weight=NULL;
int i=0, j=0, k=0, isNotCorrupted=0, num_types=9; 
int increment=1;
char filename_k[STR_BUFFER]="", file_no_thread_num[STR_BUFFER]="", cmd[STR_BUFFER]="", mcdata_type[20]="";
char group[200]="", *ph_type=NULL;
hid_t  file, file_new, group_id, dspace;
hsize_t dims[1]={0};
herr_t status, status_group;
hid_t dset_p0, dset_p1, dset_p2, dset_p3, dset_comv_p0, dset_comv_p1, dset_comv_p2, dset_comv_p3, dset_r0, dset_r1, dset_r2, dset_s0, dset_s1, dset_s2, dset_s3, dset_num_scatt, dset_weight, dset_weight_frame, dset_ph_type;
#if COMV_SWITCH == ON && STOKES_SWITCH == ON
{
num_types=17;
}
#elif COMV_SWITCH == ON || STOKES_SWITCH == ON
{
num_types=13;
}
#else
{
num_types=9;
}
#endif
#if SAVE_TYPE == ON
{
num_types+=1;
}
#endif
for (i=start_frame;i<last_frame;i=i+increment)
{
fprintf(fPtr, "Merging files for frame: %d\n", i);
fflush(fPtr);
#if SIM_SWITCH == RIKEN && DIMENSIONS == THREE
if (i>=3000)
{
increment=10; 
}
#endif
j=0;
for (k=0;k<numprocs;k++)
{
snprintf(filename_k,sizeof(filename_k),"%s%s%d%s",dir,"mc_proc_", k, ".h5" );
file=H5Fopen(filename_k, H5F_ACC_RDONLY, H5P_DEFAULT);
snprintf(group,sizeof(group),"%d",i );
status = H5Eset_auto(NULL, NULL, NULL);
status_group = H5Gget_objinfo (file, group, 0, NULL);
status = H5Eset_auto(H5E_DEFAULT, H5Eprint2, stderr);
if (status_group == 0)
{
group_id = H5Gopen2(file, group, H5P_DEFAULT);
dset_p0 = H5Dopen (group_id, "P0", H5P_DEFAULT); 
dspace = H5Dget_space (dset_p0);
status=H5Sget_simple_extent_dims(dspace, dims, NULL); 
j+=dims[0];
status = H5Sclose (dspace);
status = H5Dclose (dset_p0);
status = H5Gclose(group_id);
}
status = H5Fclose(file);
}
snprintf(file_no_thread_num,sizeof(file_no_thread_num),"%s%s%d%s",dir,"mcdata_", i, ".h5" );
status = H5Eset_auto(NULL, NULL, NULL); 
file_new=H5Fcreate(file_no_thread_num, H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT); 
status = H5Eset_auto(H5E_DEFAULT, H5Eprint2, stderr); 
if (file_new<0)
{
file_new=H5Fopen(file_no_thread_num, H5F_ACC_RDWR, H5P_DEFAULT);
for (k=0;k<num_types;k++)
{
#if COMV_SWITCH == ON && STOKES_SWITCH == ON
{
switch (k)
{
case 0: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "P0"); break;
case 1: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "P1");break;
case 2: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "P2"); break;
case 3: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "P3"); break;
case 4: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "COMV_P0"); break;
case 5: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "COMV_P1");break;
case 6: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "COMV_P2"); break;
case 7: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "COMV_P3"); break;
case 8: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "R0"); break;
case 9: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "R1"); break;
case 10: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "R2"); break;
case 11: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "S0"); break;
case 12: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "S1");break;
case 13: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "S2"); break;
case 14: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "S3"); break;
case 15: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "NS"); break;
case 16: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "PW"); break;
#if SAVE_TYPES == ON
{
case 17: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "PT"); break;
}
#endif
}
}
#elif STOKES_SWITCH == ON && COMV_SWITCH == OFF
{
switch (k)
{
case 0: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "P0"); break;
case 1: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "P1");break;
case 2: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "P2"); break;
case 3: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "P3"); break;
case 4: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "R0"); break;
case 5: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "R1"); break;
case 6: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "R2"); break;
case 7: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "S0"); break;
case 8: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "S1");break;
case 9: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "S2"); break;
case 10: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "S3"); break;
case 11: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "NS"); break;
case 12: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "PW"); break;
#if SAVE_TYPES == ON
{
case 13: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "PT"); break;
}
#endif
}
}
#elif STOKES_SWITCH == OFF && COMV_SWITCH == ON
{
switch (k)
{
case 0: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "P0"); break;
case 1: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "P1");break;
case 2: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "P2"); break;
case 3: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "P3"); break;
case 4: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "COMV_P0"); break;
case 5: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "COMV_P1");break;
case 6: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "COMV_P2"); break;
case 7: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "COMV_P3"); break;
case 8: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "R0"); break;
case 9: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "R1"); break;
case 10: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "R2"); break;
case 11: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "NS"); break;
case 12: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "PW"); break;
#if SAVE_TYPES == ON
{
case 13: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "PT"); break;
}
#endif
}
}
#else
{
switch (k)
{
case 0: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "P0"); break;
case 1: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "P1");break;
case 2: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "P2"); break;
case 3: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "P3"); break;
case 4: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "R0"); break;
case 5: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "R1"); break;
case 6: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "R2"); break;
case 7: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "NS"); break;
case 8: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "PW"); break;
#if SAVE_TYPES == ON
{
case 9: snprintf(mcdata_type,sizeof(mcdata_type), "%s", "PT"); break;
}
#endif
}
}
#endif
dset_p0 = H5Dopen (file_new, mcdata_type, H5P_DEFAULT); 
dspace = H5Dget_space (dset_p0);
status=H5Sget_simple_extent_dims(dspace, dims, NULL); 
isNotCorrupted += fmod(dims[0], j); 
status = H5Sclose (dspace);
status = H5Dclose (dset_p0);
}
status = H5Fclose(file_new);
file_new=-1; 
}
if ((file_new>=0) || (isNotCorrupted != 0 ))
{
if (isNotCorrupted != 0)
{
file_new = H5Fcreate (file_no_thread_num, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
}
p0=malloc(j*sizeof(double));  p1=malloc(j*sizeof(double));  p2=malloc(j*sizeof(double));  p3=malloc(j*sizeof(double));
comv_p0=malloc(j*sizeof(double));  comv_p1=malloc(j*sizeof(double));  comv_p2=malloc(j*sizeof(double));  comv_p3=malloc(j*sizeof(double));
r0=malloc(j*sizeof(double));  r1=malloc(j*sizeof(double));  r2=malloc(j*sizeof(double));
s0=malloc(j*sizeof(double));  s1=malloc(j*sizeof(double));  s2=malloc(j*sizeof(double));  s3=malloc(j*sizeof(double));
num_scatt=malloc(j*sizeof(double)); weight=malloc(j*sizeof(double));
ph_type=malloc((j)*sizeof(char));
j=0;
for (k=0;k<numprocs;k++)
{
snprintf(filename_k,sizeof(filename_k),"%s%s%d%s",dir,"mc_proc_", k, ".h5" );
file=H5Fopen(filename_k, H5F_ACC_RDONLY, H5P_DEFAULT);
snprintf(group,sizeof(group),"%d",i );
status = H5Eset_auto(NULL, NULL, NULL);
status_group = H5Gget_objinfo (file, group, 0, NULL);
status = H5Eset_auto(H5E_DEFAULT, H5Eprint2, stderr);
if (status_group == 0)
{
group_id = H5Gopen2(file, group, H5P_DEFAULT);
dset_p0 = H5Dopen (group_id, "P0", H5P_DEFAULT); 
dset_p1 = H5Dopen (group_id, "P1", H5P_DEFAULT);
dset_p2 = H5Dopen (group_id, "P2", H5P_DEFAULT);
dset_p3 = H5Dopen (group_id, "P3", H5P_DEFAULT);
#if COMV_SWITCH == ON
{
dset_comv_p0 = H5Dopen (group_id, "COMV_P0", H5P_DEFAULT); 
dset_comv_p1 = H5Dopen (group_id, "COMV_P1", H5P_DEFAULT);
dset_comv_p2 = H5Dopen (group_id, "COMV_P2", H5P_DEFAULT);
dset_comv_p3 = H5Dopen (group_id, "COMV_P3", H5P_DEFAULT);
}
#endif
dset_r0 = H5Dopen (group_id, "R0", H5P_DEFAULT);
dset_r1 = H5Dopen (group_id, "R1", H5P_DEFAULT);
dset_r2 = H5Dopen (group_id, "R2", H5P_DEFAULT);
#if STOKES_SWITCH == ON
{
dset_s0 = H5Dopen (group_id, "S0", H5P_DEFAULT);
dset_s1 = H5Dopen (group_id, "S1", H5P_DEFAULT);
dset_s2 = H5Dopen (group_id, "S2", H5P_DEFAULT);
dset_s3 = H5Dopen (group_id, "S3", H5P_DEFAULT);
}
#endif
dset_num_scatt = H5Dopen (group_id, "NS", H5P_DEFAULT);
dset_weight = H5Dopen (group_id, "PW", H5P_DEFAULT); 
#if SAVE_TYPE == ON
{
dset_ph_type = H5Dopen (group_id, "PT", H5P_DEFAULT);
}
#endif
status = H5Dread(dset_p0, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, (p0+j));
status = H5Dread(dset_p1, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, (p1+j));
status = H5Dread(dset_p2, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, (p2+j));
status = H5Dread(dset_p3, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, (p3+j));
#if COMV_SWITCH == ON
{
status = H5Dread(dset_comv_p0, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, (comv_p0+j));
status = H5Dread(dset_comv_p1, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, (comv_p1+j));
status = H5Dread(dset_comv_p2, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, (comv_p2+j));
status = H5Dread(dset_comv_p3, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, (comv_p3+j));
}
#endif
status = H5Dread(dset_r0, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, (r0+j));
status = H5Dread(dset_r1, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, (r1+j));
status = H5Dread(dset_r2, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, (r2+j));
#if STOKES_SWITCH == ON
{
status = H5Dread(dset_s0, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, (s0+j));
status = H5Dread(dset_s1, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, (s1+j));
status = H5Dread(dset_s2, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, (s2+j));
status = H5Dread(dset_s3, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, (s3+j));
}
#endif
status = H5Dread(dset_num_scatt, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, (num_scatt+j));
status = H5Dread(dset_weight, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, (weight+j));
#if SAVE_TYPE == ON
{
status = H5Dread(dset_ph_type, H5T_NATIVE_CHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, (ph_type+j));
}
#endif
dspace = H5Dget_space (dset_p0);
status=H5Sget_simple_extent_dims(dspace, dims, NULL); 
j+=dims[0];
status = H5Sclose (dspace);
status = H5Dclose (dset_p0); status = H5Dclose (dset_p1); status = H5Dclose (dset_p2); status = H5Dclose (dset_p3);
#if COMV_SWITCH == ON
{
status = H5Dclose (dset_comv_p0); status = H5Dclose (dset_comv_p1); status = H5Dclose (dset_comv_p2); status = H5Dclose (dset_comv_p3);
}
#endif
status = H5Dclose (dset_r0); status = H5Dclose (dset_r1); status = H5Dclose (dset_r2);
#if STOKES_SWITCH == ON
{
status = H5Dclose (dset_s0); status = H5Dclose (dset_s1); status = H5Dclose (dset_s2); status = H5Dclose (dset_s3);
}
#endif
#if SAVE_TYPE == ON
{
status = H5Dclose (dset_ph_type);
}
#endif
status = H5Dclose (dset_num_scatt);
status = H5Dclose (dset_weight);
status = H5Gclose(group_id);
}
status = H5Fclose(file);
}
dims[0]=j;
dspace = H5Screate_simple(1, dims, NULL);
dset_p0=H5Dcreate2(file_new, "P0", H5T_NATIVE_DOUBLE, dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
dset_p1=H5Dcreate2(file_new, "P1", H5T_NATIVE_DOUBLE, dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
dset_p2=H5Dcreate2(file_new, "P2", H5T_NATIVE_DOUBLE, dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
dset_p3=H5Dcreate2(file_new, "P3", H5T_NATIVE_DOUBLE, dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
#if COMV_SWITCH == ON
{
dset_comv_p0=H5Dcreate2(file_new, "COMV_P0", H5T_NATIVE_DOUBLE, dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
dset_comv_p1=H5Dcreate2(file_new, "COMV_P1", H5T_NATIVE_DOUBLE, dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
dset_comv_p2=H5Dcreate2(file_new, "COMV_P2", H5T_NATIVE_DOUBLE, dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
dset_comv_p3=H5Dcreate2(file_new, "COMV_P3", H5T_NATIVE_DOUBLE, dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
}
#endif
dset_r0=H5Dcreate2(file_new, "R0", H5T_NATIVE_DOUBLE, dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
dset_r1=H5Dcreate2(file_new, "R1", H5T_NATIVE_DOUBLE, dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
dset_r2=H5Dcreate2(file_new, "R2", H5T_NATIVE_DOUBLE, dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
#if STOKES_SWITCH == ON
{
dset_s0=H5Dcreate2(file_new, "S0", H5T_NATIVE_DOUBLE, dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
dset_s1=H5Dcreate2(file_new, "S1", H5T_NATIVE_DOUBLE, dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
dset_s2=H5Dcreate2(file_new, "S2", H5T_NATIVE_DOUBLE, dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
dset_s3=H5Dcreate2(file_new, "S3", H5T_NATIVE_DOUBLE, dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
}
#endif
dset_num_scatt=H5Dcreate2(file_new, "NS", H5T_NATIVE_DOUBLE, dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
dset_weight=H5Dcreate2(file_new, "PW", H5T_NATIVE_DOUBLE, dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
#if SAVE_TYPE == ON
{
dset_ph_type=H5Dcreate2(file_new, "PT", H5T_NATIVE_CHAR, dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
}
#endif
status = H5Dwrite (dset_p0, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
H5P_DEFAULT, p0);
status = H5Dwrite (dset_p1, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
H5P_DEFAULT, p1);
status = H5Dwrite (dset_p2, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
H5P_DEFAULT, p2);
status = H5Dwrite (dset_p3, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
H5P_DEFAULT, p3);
#if COMV_SWITCH == ON
{
status = H5Dwrite (dset_comv_p0, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
H5P_DEFAULT, comv_p0);
status = H5Dwrite (dset_comv_p1, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
H5P_DEFAULT, comv_p1);
status = H5Dwrite (dset_comv_p2, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
H5P_DEFAULT, comv_p2);
status = H5Dwrite (dset_comv_p3, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
H5P_DEFAULT, comv_p3);
}
#endif
status = H5Dwrite (dset_r0, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
H5P_DEFAULT, r0);
status = H5Dwrite (dset_r1, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
H5P_DEFAULT, r1);
status = H5Dwrite (dset_r2, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
H5P_DEFAULT, r2);
#if STOKES_SWITCH == ON
{
status = H5Dwrite (dset_s0, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
H5P_DEFAULT, s0);
status = H5Dwrite (dset_s1, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
H5P_DEFAULT, s1);
status = H5Dwrite (dset_s2, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
H5P_DEFAULT, s2);
status = H5Dwrite (dset_s3, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
H5P_DEFAULT, s3);
}
#endif
#if SAVE_TYPE == ON
{
status = H5Dwrite (dset_ph_type, H5T_NATIVE_CHAR, H5S_ALL, H5S_ALL,
H5P_DEFAULT, ph_type);
}
#endif
status = H5Dwrite (dset_num_scatt, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
H5P_DEFAULT, num_scatt);
status = H5Dwrite (dset_weight, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
H5P_DEFAULT, weight);
status = H5Sclose (dspace);
status = H5Dclose (dset_p0); status = H5Dclose (dset_p1); status = H5Dclose (dset_p2); status = H5Dclose (dset_p3);
#if COMV_SWITCH == ON
{
status = H5Dclose (dset_comv_p0); status = H5Dclose (dset_comv_p1); status = H5Dclose (dset_comv_p2); status = H5Dclose (dset_comv_p3);
}
#endif
status = H5Dclose (dset_r0); status = H5Dclose (dset_r1); status = H5Dclose (dset_r2);
#if STOKES_SWITCH == ON
{
status = H5Dclose (dset_s0); status = H5Dclose (dset_s1); status = H5Dclose (dset_s2); status = H5Dclose (dset_s3);
}
#endif
#if SAVE_TYPE == ON
{
status = H5Dclose (dset_ph_type);
}
#endif
status = H5Dclose (dset_num_scatt);
status = H5Dclose (dset_weight);
status = H5Fclose (file_new);
free(p0);free(p1); free(p2);free(p3);
free(comv_p0);free(comv_p1); free(comv_p2);free(comv_p3);
free(r0);free(r1); free(r2);
free(s0);free(s1); free(s2);free(s3);
free(num_scatt); free(weight);
free(ph_type);
isNotCorrupted=0;
}
}
}
void hydroDataFrameInitialize(struct hydro_dataframe *hydro_data)
{
hydro_data->r0=NULL;
hydro_data->r1=NULL;
hydro_data->r2=NULL;
hydro_data->r0_size=NULL;
hydro_data->r1_size=NULL;
hydro_data->r2_size=NULL;
hydro_data->r=NULL;
hydro_data->theta=NULL;
hydro_data->v0=NULL;
hydro_data->v1=NULL;
hydro_data->v2=NULL;
hydro_data->dens=NULL;
hydro_data->dens_lab=NULL;
hydro_data->pres=NULL;
hydro_data->temp=NULL;
hydro_data->gamma=NULL;
hydro_data->B0=NULL;
hydro_data->B1=NULL;
hydro_data->B2=NULL;
}
void freeHydroDataFrame(struct hydro_dataframe *hydro_data)
{
free(hydro_data->r0);
free(hydro_data->r1);
free(hydro_data->r2);
free(hydro_data->r0_size);
free(hydro_data->r1_size);
free(hydro_data->r2_size);
free(hydro_data->r);
free(hydro_data->theta);
free(hydro_data->v0);
free(hydro_data->v1);
free(hydro_data->v2);
free(hydro_data->dens);
free(hydro_data->dens_lab);
free(hydro_data->pres);
free(hydro_data->temp);
free(hydro_data->gamma);
free(hydro_data->B0);
free(hydro_data->B1);
free(hydro_data->B2);
hydro_data->r0=NULL;
hydro_data->r1=NULL;
hydro_data->r2=NULL;
hydro_data->r0_size=NULL;
hydro_data->r1_size=NULL;
hydro_data->r2_size=NULL;
hydro_data->r=NULL;
hydro_data->theta=NULL;
hydro_data->v0=NULL;
hydro_data->v1=NULL;
hydro_data->v2=NULL;
hydro_data->dens=NULL;
hydro_data->dens_lab=NULL;
hydro_data->pres=NULL;
hydro_data->temp=NULL;
hydro_data->gamma=NULL;
hydro_data->B0=NULL;
hydro_data->B1=NULL;
hydro_data->B2=NULL;
}
int getHydroData(struct hydro_dataframe *hydro_data, int frame, double inj_radius, int ph_inj_switch, double min_r, double max_r, double min_theta, double max_theta, FILE *fPtr)
{
char hydro_file[STR_BUFFER]="";
char hydro_prefix[STR_BUFFER]="";
snprintf(hydro_prefix,sizeof(hydro_prefix),"%s%s",FILEPATH,FILEROOT );
#if DIMENSIONS == TWO
#if SIM_SWITCH == FLASH
modifyFlashName(hydro_file, hydro_prefix, frame);
fprintf(fPtr,">> MCRaT is opening FLASH file %s\n", hydro_file);
fflush(fPtr);
readAndDecimate(hydro_file, hydro_data, inj_radius, ph_inj_switch, min_r, max_r, min_theta, max_theta, fPtr);
#elif SIM_SWITCH == PLUTO_CHOMBO
modifyPlutoName(hydro_file, hydro_prefix, frame);
fprintf(fPtr,">> MCRaT is opening PLUTO-Chombo file %s\n", hydro_file);
fflush(fPtr);
readPlutoChombo(hydro_file, hydro_data, inj_radius, ph_inj_switch, min_r, max_r, min_theta, max_theta, fPtr);
#elif SIM_SWITCH == PLUTO
modifyPlutoName(hydro_file, hydro_prefix, frame);
fprintf(fPtr,">> MCRaT is opening PLUTO file %s\n", hydro_file);
fflush(fPtr);
readPluto(hydro_file, hydro_data, inj_radius, ph_inj_switch, min_r, max_r, min_theta, max_theta, fPtr);
#else
readHydro2D(FILEPATH, frame, inj_radius, fps_modified, &xPtr,  &yPtr,  &szxPtr, &szyPtr, &rPtr,\
&thetaPtr, &velxPtr,  &velyPtr,  &densPtr,  &presPtr,  &gammaPtr,  &dens_labPtr, &tempPtr, &array_num, ph_inj_switch, min_r, max_r, fPtr);
#endif
#else
#if SIM_SWITCH == FLASH
#error 3D FLASH simulations are not supported in MCRaT yet.
#elif SIM_SWITCH == PLUTO_CHOMBO
modifyPlutoName(hydro_file, hydro_prefix, frame);
fprintf(fPtr,">> MCRaT is opening PLUTO-Chombo file %s\n", hydro_file);
fflush(fPtr);
readPlutoChombo(hydro_file, hydro_data, inj_radius, ph_inj_switch, min_r, max_r, min_theta, max_theta, fPtr);
#elif SIM_SWITCH == PLUTO
modifyPlutoName(hydro_file, hydro_prefix, frame);
fprintf(fPtr,">> MCRaT is opening PLUTO file %s\n", hydro_file);
fflush(fPtr);
readPluto(hydro_file, hydro_data, inj_radius, ph_inj_switch, min_r, max_r, min_theta, max_theta, fPtr);
#else
read_hydro(FILEPATH, frame, inj_radius, &xPtr,  &yPtr, &zPtr,  &szxPtr, &szyPtr, &rPtr,\
&thetaPtr, &phiPtr, &velxPtr,  &velyPtr, &velzPtr,  &densPtr,  &presPtr,  &gammaPtr,  &dens_labPtr, &tempPtr, &array_num, ph_inj_switch, min_r, max_r, fps_modified, fPtr);
#endif
#endif
fprintf(fPtr, "MCRaT: The chosen number of hydro elements is %d\n", hydro_data->num_elements);
fillHydroCoordinateToSpherical(hydro_data);
#if SIMULATION_TYPE == CYLINDRICAL_OUTFLOW
cylindricalPrep(hydro_data, fPtr);
#elif SIMULATION_TYPE == SPHERICAL_OUTFLOW
sphericalPrep(hydro_data, fPtr);
#elif SIMULATION_TYPE == STRUCTURED_SPHERICAL_OUTFLOW
structuredFireballPrep(hydro_data, fPtr);
#endif
return 0;
}
int printHydroGeometry(FILE *fPtr)
{
#if DIMENSIONS == TWO
char dim[]="2D";
#elif DIMENSIONS == TWO_POINT_FIVE
char dim[]="2.5D";
#else
char dim[]="3D";
#endif
#if SIM_SWITCH == FLASH
char sim[]="Flash";
#elif SIM_SWITCH == PLUTO_CHOMBO
char sim[]="PLUTO-Chombo";
#elif SIM_SWITCH == PLUTO
char sim[]="PLUTO";
#endif
#if GEOMETRY == CARTESIAN
char geo[]="Cartesian";
#elif GEOMETRY == CYLINDRICAL
char geo[]="Cylindrical";
#elif GEOMETRY == SPHERICAL
char geo[]="Spherical";
#elif GEOMETRY == POLAR
char geo[]="Polar";
#endif
fprintf(fPtr, "MCRaT is working on a %s %s %s simulation.\n", dim, geo, sim );
fflush(fPtr);
return 0;
}
