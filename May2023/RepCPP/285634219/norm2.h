double norm2(int i, int j, const double *c1, const double *c2)
{
double dist, diff; 
int k; 


dist = 0.0;

for (k = 0; k < n_d; k++)
{

diff = *(c1 + (i - 1)*n_d + k) - *(c2 + (j - 1)*n_d + k);

dist += diff*diff;
}


dist = sqrt(dist);

return dist;
}

double recursive_norm2(int i, int j, int n_2, double *ca,
const double *c1, const double *c2)
{

double *ca_ij = ca + (i - 1)*n_2 + (j - 1);


if (*ca_ij > -1.0) 
{
return *ca_ij;
}
else if ((i == 1) && (j == 1))
{
*ca_ij = norm2(1, 1, c1, c2);
}
else if ((i > 1) && (j == 1))
{
*ca_ij = fmax(recursive_norm2(i - 1, 1, n_2, ca, c1, c2), norm2(i, 1, c1, c2));
}
else if ((i == 1) && (j > 1))
{
*ca_ij = fmax(recursive_norm2(1, j - 1, n_2, ca, c1, c2), norm2(1, j, c1, c2));
}
else if ((i > 1) && (j > 1))
{
*ca_ij = fmax(
fmin(fmin(
recursive_norm2(i - 1, j    , n_2, ca, c1, c2),
recursive_norm2(i - 1, j - 1, n_2, ca, c1, c2)),
recursive_norm2(i,     j - 1, n_2, ca, c1, c2)),
norm2(i, j, c1, c2));
}
else
{
*ca_ij = INFINITY;
}

return *ca_ij;
}

void distance_norm2 (
int n_1, int n_2,
double *__restrict ca,
const double *__restrict c1,
const double *__restrict c2)
{
#pragma omp target teams distribute parallel for collapse(2) thread_limit(256)
for (int i = 1; i <= n_1; i++)
for (int j = 1; j <= n_2; j++)
recursive_norm2(i, j, n_2, ca, c1, c2);
}
