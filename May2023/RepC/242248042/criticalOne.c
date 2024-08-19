int main() {
int s1, s2;
int arr[10];
int * arr2 = (int *) malloc(15 * sizeof(int));
arr[0] = 1;
#pragma omp parallel
{
int i = 0;
#pragma omp for
for (i = 0; i < 10; i++) {
#pragma omp critical
{
s1 = s1 + 2;
s2 = s1;
}
#pragma omp critical
{
s2 = 5;
arr[i] = 16;
}
#pragma omp critical
{
s1 = 1;
s1 = s1 + 2;
s2 = s1;
arr[i] = 10;
}
}
}
}
