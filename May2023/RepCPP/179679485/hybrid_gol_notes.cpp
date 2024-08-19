

#pragma omp parallel num_threads(thread_count)
{

int thread_ID, total_threads;
thread_ID = omp_get_thread_num(); 
total_threads = omp_get_num_threads(); 

int mpi_chunk_sz = floor(org_gol_rows / world_size);
int omp_chunk_sz = floor(mpi_chunk_sz / th_cnt);

if(world_rank == world_size - 1)
{
int btm_mpi_chunk_sz = org_rows - ((world_size - 1) * mpi_chunk_sz);
omp_chunk_sz = btm_mpi_chunk_sz / thread_count;
}

omp_thread_row_start = (mpi_chunk_sz * world_rank) + (omp_chunk_sz * thread_ID);
omp_thread_row_end = omp_thread_row_start + (omp_chunk_sz - 1);

if(world_rank == 0 || world_rank == 1)
{
cout << thread_ID << endl;
}

for(int i = omp_thread_row_start; i < omp_thread_row_end; i++)
{
for(int j)
{

}
}
}