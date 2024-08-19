#pragma omp target teams num_teams(global/local) thread_limit(local)
{
unsigned int s_Hist[HISTOGRAM_BLOCK_MEMORY];
#pragma omp parallel
{

const int lid = omp_get_thread_num();
const int lsize = omp_get_num_threads();
const int tid = omp_get_team_num();
const int gid = tid * lsize + lid; 
const int gsize = omp_get_num_teams() * lsize;

int mulBase = (lid >> BUCKET_WARP_LOG_SIZE);
const int warpBase = IMUL(mulBase, HISTOGRAM_BIN_COUNT);

for(uint i = lid; i < HISTOGRAM_BLOCK_MEMORY; i+=lsize) {
s_Hist[i] = 0;
}

#pragma omp barrier

for(int pos = gid; pos < listsize; pos += gsize) {
uint data4 = ((d_input[pos] - minimum)/(maximum - minimum)) * HISTOGRAM_BIN_COUNT;
#pragma omp atomic update
s_Hist[warpBase+(data4 & 0x3FFU)]++;
}

#pragma omp barrier

for(int pos = lid; pos < HISTOGRAM_BIN_COUNT; pos += lsize){
uint sum = 0;
for(int i = 0; i < HISTOGRAM_BLOCK_MEMORY; i+= HISTOGRAM_BIN_COUNT){ 
sum += s_Hist[pos + i] & 0x07FFFFFFU;
}
#pragma omp atomic update
h_offsets[pos] += sum;
}
}
}
