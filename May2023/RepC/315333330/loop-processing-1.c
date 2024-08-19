extern int place ();
void vector_1 (int *ary, int size)
{
#pragma acc parallel num_workers (32) vector_length(32) copy(ary[0:size]) firstprivate (size)
{
#pragma acc loop gang
for (int jx = 0; jx < 1; jx++)
#pragma acc loop auto
for (int ix = 0; ix < size; ix++)
ary[ix] = place ();
}
}
