#pragma omp target teams num_teams(done_size) thread_limit(LQSORT_LOCAL_WORKGROUP_SIZE)
{
workstack_record workstack[QUICKSORT_BLOCK_SIZE/SORT_THRESHOLD]; 
int workstack_pointer;

T mys[QUICKSORT_BLOCK_SIZE], mysn[QUICKSORT_BLOCK_SIZE], temp[SORT_THRESHOLD];
T *s, *sn;
uint ltsum, gtsum;
uint lt[LQSORT_LOCAL_WORKGROUP_SIZE+1], gt[LQSORT_LOCAL_WORKGROUP_SIZE+1];
#pragma omp parallel
{
const uint blockid    = omp_get_team_num();
const uint localid    = omp_get_thread_num();

uint i, tmp, ltp, gtp;

work_record<T> block = seqs[blockid];
const uint d_offset = block.start;
uint start = 0; 
uint end   = block.end - d_offset;

uint direction = 1; 
if (localid == 0) {
workstack_pointer = 0; 
workstack_record wr = { start, end, direction };
workstack[0] = wr;
}
if (block.direction == 1) {
for (i = localid; i < end; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
mys[i] = d[i+d_offset];
}
} else {
for (i = localid; i < end; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
mys[i] = dn[i+d_offset];
}
}
#pragma omp barrier

while (workstack_pointer >= 0) { 
workstack_record wr = workstack[workstack_pointer];
start = wr.start;
end = wr.end;
direction = wr.direction;
#pragma omp barrier
if (localid == 0) {
--workstack_pointer;

ltsum = gtsum = 0;	
}
if (direction == 1) {
s = mys;
sn = mysn;
} else {
s = mysn;
sn = mys;
}
lt[localid] = gt[localid] = 0;
ltp = gtp = 0;
#pragma omp barrier

uint pivot = s[start];
if (start < end) {
pivot = median(pivot, s[(start+end) >> 1], s[end-1]);
}
for(i = start + localid; i < end; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
tmp = s[i];
if (tmp < pivot)
ltp++;
if (tmp > pivot) 
gtp++;
}
lt[localid] = ltp;
gt[localid] = gtp;
#pragma omp barrier

uint n;
for(i = 1; i < LQSORT_LOCAL_WORKGROUP_SIZE; i <<= 1) {
n = 2*i - 1;
if ((localid & n) == n) {
lt[localid] += lt[localid-i];
gt[localid] += gt[localid-i];
}
#pragma omp barrier
}

if ((localid & n) == n) {
lt[LQSORT_LOCAL_WORKGROUP_SIZE] = ltsum = lt[localid];
gt[LQSORT_LOCAL_WORKGROUP_SIZE] = gtsum = gt[localid];
lt[localid] = 0;
gt[localid] = 0;
}

for(i = LQSORT_LOCAL_WORKGROUP_SIZE/2; i >= 1; i >>= 1) {
n = 2*i - 1;
if ((localid & n) == n) {
plus_prescan(&lt[localid - i], &lt[localid]);
plus_prescan(&gt[localid - i], &gt[localid]);
}
#pragma omp barrier
}

uint lfrom = start + lt[localid];
uint gfrom = end - gt[localid+1];

for (i = start + localid; i < end; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
tmp = s[i];
if (tmp < pivot) 
sn[lfrom++] = tmp;

if (tmp > pivot) 
sn[gfrom++] = tmp;
}
#pragma omp barrier

for (i = start + ltsum + localid;i < end - gtsum; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
d[i+d_offset] = pivot;
}
#pragma omp barrier

if (ltsum <= SORT_THRESHOLD) {
sort_threshold(sn, d+d_offset, start, start + ltsum, temp, localid);
} else {
PUSH(start, start + ltsum);
#pragma omp barrier
}

if (gtsum <= SORT_THRESHOLD) {
sort_threshold(sn, d+d_offset, end - gtsum, end, temp, localid);
} else {
PUSH(end - gtsum, end);
#pragma omp barrier
}
}
}
}


