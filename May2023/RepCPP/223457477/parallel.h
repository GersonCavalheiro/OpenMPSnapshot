
#include <omp.h>
#include <vector>

using namespace std;

#define threads 4

vector<double> mean(vector<vector<double>> &mat)
{
vector<double> avg(mat[0].size(),0); 
int j;
omp_set_num_threads(threads);

#pragma omp parallel for schedule(dynamic,4) collapse(2)
for(int i=0; i<mat.size(); i++)
{
for(j=0; j<mat[0].size(); j++)
{
avg[j] += mat[i][j];
}
}

#pragma omp parallel for schedule(dynamic,4)
for(int i=0;i<avg.size();i++)
{
avg[i] /= mat.size();
}

return avg;
}

vector<vector<double>> add(vector<vector<double>> &a, vector<vector<double>> &b)
{
vector<vector<double>> res(a.size(), vector<double>(a[0].size(),0));
int j;

if((a.size()!=b.size())||(a[0].size()!=b[0].size()))
{
return res;
}

omp_set_num_threads(threads);

#pragma omp parallel for schedule(dynamic,4) collapse(2) 
for(int i=0; i<a.size(); i++)
{
for(j=0; j<a[0].size(); j++)
{
res[i][j] = a[i][j] + b[i][j];
}
}

return res;
}

vector<vector<double>> mult(vector<double> &p, vector<double> &q)
{
vector<vector<double>> res(p.size(),vector<double>(q.size(),0));
int j;

if(p.size() != q.size())
{
return res;
}

omp_set_num_threads(threads);

#pragma omp parallel for schedule(dynamic,4) collapse(2) 
for(int i=0; i<p.size(); i++)
{ 
for(j=0; j<p.size(); j++)
{
res[i][j] = p[i] * q[j];
}
}

return res;
}

vector<vector<double>> covariance(vector<vector<double>> &mat, vector<double> &avg)
{
vector<vector<double>> res(avg.size(), vector<double>(avg.size(),0));
vector<vector<double>> prod(avg.size(), vector<double>(avg.size(),0));
vector<double> diff(avg.size(),0);
int N = mat.size(),j;

omp_set_num_threads(threads);

for(int i=0; i<N; i++)
{ 
#pragma omp parallel for schedule(dynamic,4) 
for(j=0; j<avg.size(); j++)
{
diff[j] = mat[i][j] - avg[j];
}

prod = mult(diff,diff);
res = add(res,prod);
}

#pragma omp parallel for schedule(dynamic,4) collapse(2)
for(int i=0; i<res.size(); i++) 
{        
for(j=0; j<res[0].size(); j++)
{
res[i][j] = res[i][j] / N;
}
}

return res;
}