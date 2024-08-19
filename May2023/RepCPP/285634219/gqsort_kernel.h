#pragma omp target teams num_teams(blocks_size) thread_limit(GQSORT_LOCAL_WORKGROUP_SIZE)
{
uint lt[GQSORT_LOCAL_WORKGROUP_SIZE+1];
uint gt[GQSORT_LOCAL_WORKGROUP_SIZE+1];
uint ltsum, gtsum, lbeg, gbeg;
#pragma omp parallel
{
const uint blockid = omp_get_team_num();
const uint localid = omp_get_thread_num();

uint i, lfrom, gfrom, ltp = 0, gtp = 0;
T lpivot, gpivot, tmp; 
T *s, *sn;

block_record<T> block = blocksb[blockid];
uint start = block.start, end = block.end, pivot = block.pivot, direction = block.direction;

parent_record* pparent = parentsb + block.parent; 
uint* psstart, *psend, *poldstart, *poldend, *pblockcount;

if (direction == 1) {
s = d;
sn = dn;
} else {
s = dn;
sn = d;
}

lt[localid] = gt[localid] = 0;
#pragma omp barrier

for(i = start + localid; i < end; i += GQSORT_LOCAL_WORKGROUP_SIZE) {
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
for(i = 1; i < GQSORT_LOCAL_WORKGROUP_SIZE; i <<= 1) {
n = 2*i - 1;
if ((localid & n) == n) {
lt[localid] += lt[localid-i];
gt[localid] += gt[localid-i];
}
#pragma omp barrier
}

if ((localid & n) == n) {
lt[GQSORT_LOCAL_WORKGROUP_SIZE] = ltsum = lt[localid];
gt[GQSORT_LOCAL_WORKGROUP_SIZE] = gtsum = gt[localid];
lt[localid] = 0;
gt[localid] = 0;
}

for(i = GQSORT_LOCAL_WORKGROUP_SIZE/2; i >= 1; i >>= 1) {
n = 2*i - 1;
if ((localid & n) == n) {
plus_prescan(&lt[localid - i], &lt[localid]);
plus_prescan(&gt[localid - i], &gt[localid]);
}
#pragma omp barrier
}

if (localid == 0) {
psstart = &pparent->sstart;
psend = &pparent->send;
poldstart = &pparent->oldstart;
poldend = &pparent->oldend;
pblockcount = &pparent->blockcount;
#pragma omp atomic capture
{
lbeg = *psstart;
*psstart += ltsum;
}

#pragma omp atomic capture
{
gbeg = *psend;
*psend -= gtsum;
}
gbeg -= gtsum;

}
#pragma omp barrier

lfrom = lbeg + lt[localid];
gfrom = gbeg + gt[localid];

for(i = start + localid; i < end; i += GQSORT_LOCAL_WORKGROUP_SIZE) {
tmp = s[i];
if (tmp < pivot) 
sn[lfrom++] = tmp;

if (tmp > pivot) 
sn[gfrom++] = tmp;
}
#pragma omp barrier

if (localid == 0) {
uint old_blockcount;
#pragma omp atomic capture
{
old_blockcount = *pblockcount;
(*pblockcount)--;
}

if (old_blockcount == 0) { 
uint sstart = *psstart;
uint send = *psend;
uint oldstart = *poldstart;
uint oldend = *poldend;

for(i = sstart; i < send; i ++) {
d[i] = pivot;
}

lpivot = sn[oldstart];
gpivot = sn[oldend-1];
if (oldstart < sstart) {
lpivot = median(lpivot,sn[(oldstart+sstart) >> 1], sn[sstart-1]);
} 
if (send < oldend) {
gpivot = median(sn[send],sn[(oldend+send) >> 1], gpivot);
}

work_record<T>* result1 = result + 2*blockid;
work_record<T>* result2 = result1 + 1;

direction ^= 1;

work_record<T> r1 = {oldstart, sstart, lpivot, direction};
*result1 = r1;

work_record<T> r2 = {send, oldend, gpivot, direction};
*result2 = r2;
}
}
}
}
