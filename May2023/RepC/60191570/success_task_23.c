void dealloc_tiled_matrix(int MBS, int NBS, int M, int N, double (*a)[N/NBS][MBS][NBS]) {
for (int i=0; i<M/MBS; i ++) {
for (int j=0; j<N/NBS; j++) {
if (i != 0 || j != 0) {
#pragma omp task inout(a[i][j]) concurrent(a[0][0])
{
}
}
}
}
}
int main(int argc, char *argv[])
{
return 0;
}
