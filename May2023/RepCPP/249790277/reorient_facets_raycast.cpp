#include "reorient_facets_raycast.h"
#include "../per_face_normals.h"
#include "../doublearea.h"
#include "../random_dir.h"
#include "../bfs_orient.h"
#include "EmbreeIntersector.h"
#include <iostream>
#include <random>
#include <ctime>
#include <limits>

template <
typename DerivedV,
typename DerivedF,
typename DerivedI,
typename DerivedC>
IGL_INLINE void igl::embree::reorient_facets_raycast(
const Eigen::PlainObjectBase<DerivedV> & V,
const Eigen::PlainObjectBase<DerivedF> & F,
int rays_total,
int rays_minimum,
bool facet_wise,
bool use_parity,
bool is_verbose,
Eigen::PlainObjectBase<DerivedI> & I,
Eigen::PlainObjectBase<DerivedC> & C)
{
using namespace Eigen;
using namespace std;
assert(F.cols() == 3);
assert(V.cols() == 3);

const int m = F.rows();

MatrixXi FF = F;
if (facet_wise) {
C.resize(m);
for (int i = 0; i < m; ++i) C(i) = i;

} else {
if (is_verbose) cout << "extracting patches... ";
bfs_orient(F,FF,C);
}
if (is_verbose) cout << (C.maxCoeff() + 1)  << " components. ";

const int num_cc = C.maxCoeff()+1;

EmbreeIntersector ei;
ei.init(V.template cast<float>(),FF);

MatrixXd N;
per_face_normals(V,FF,N);

Matrix<typename DerivedV::Scalar,Dynamic,1> A;
doublearea(V,FF,A);
double area_total = A.sum();

VectorXd area_per_component;
area_per_component.setZero(num_cc);
for (int f = 0; f < m; ++f)
{
area_per_component(C(f)) += A(f);
}
VectorXi num_rays_per_component(num_cc);
for (int c = 0; c < num_cc; ++c)
{
num_rays_per_component(c) = max<int>(static_cast<int>(rays_total * area_per_component(c) / area_total), rays_minimum);
}
rays_total = num_rays_per_component.sum();

if (is_verbose) cout << "generating rays... ";
uniform_real_distribution<float> rdist;
mt19937 prng;
prng.seed(time(nullptr));
vector<int     > ray_face;
vector<Vector3f> ray_ori;
vector<Vector3f> ray_dir;
ray_face.reserve(rays_total);
ray_ori .reserve(rays_total);
ray_dir .reserve(rays_total);
for (int c = 0; c < num_cc; ++c)
{
if (area_per_component[c] == 0)
{
continue;
}
vector<int> CF;     
vector<double> CF_area;
for (int f = 0; f < m; ++f)
{
if (C(f)==c)
{
CF.push_back(f);
CF_area.push_back(A(f));
}
}
discrete_distribution<int> ddist(CF.size(), 0, CF.size(), [&](double i){ return CF_area[static_cast<int>(i)]; });       
for (int i = 0; i < num_rays_per_component[c]; ++i)
{
int f = CF[ddist(prng)];          
float s = rdist(prng);            
float t = rdist(prng);
float sqrt_t = sqrtf(t);
float a = 1 - sqrt_t;
float b = (1 - s) * sqrt_t;
float c = s * sqrt_t;
Vector3f p = a * V.row(FF(f,0)).template cast<float>().eval()       
+ b * V.row(FF(f,1)).template cast<float>().eval()
+ c * V.row(FF(f,2)).template cast<float>().eval();
Vector3f n = N.row(f).cast<float>();
if (n.isZero()) continue;
Vector3f d;
while (true) {
d = random_dir().cast<float>();
float ndotd = n.dot(d);
if (fabsf(ndotd) < 0.1f)
{
continue;
}
if (ndotd < 0)
{
d *= -1.0f;
}
break;
}
ray_face.push_back(f);
ray_ori .push_back(p);
ray_dir .push_back(d);

if (is_verbose && ray_face.size() % (rays_total / 10) == 0) cout << ".";
}
}
if (is_verbose) cout << ray_face.size()  << " rays. ";

vector<pair<float, float>> C_vote_distance(num_cc, make_pair(0, 0));      
vector<pair<int  , int  >> C_vote_infinity(num_cc, make_pair(0, 0));      
vector<pair<int  , int  >> C_vote_parity(num_cc, make_pair(0, 0));        

if (is_verbose) cout << "shooting rays... ";
#pragma omp parallel for
for (int i = 0; i < (int)ray_face.size(); ++i)
{
int      f = ray_face[i];
Vector3f o = ray_ori [i];
Vector3f d = ray_dir [i];
int c = C(f);

vector<Hit> hits_front;
vector<Hit> hits_back;
int num_rays_front;
int num_rays_back;
ei.intersectRay(o,  d, hits_front, num_rays_front);
ei.intersectRay(o, -d, hits_back , num_rays_back );
if (!hits_front.empty() && hits_front[0].id == f) hits_front.erase(hits_front.begin());
if (!hits_back .empty() && hits_back [0].id == f) hits_back .erase(hits_back .begin());

if (use_parity) {
#pragma omp atomic
C_vote_parity[c].first  += hits_front.size() % 2;
#pragma omp atomic
C_vote_parity[c].second += hits_back .size() % 2;

} else {
if (hits_front.empty())
{
#pragma omp atomic
C_vote_infinity[c].first++;
} else {
#pragma omp atomic
C_vote_distance[c].first += hits_front[0].t;
}

if (hits_back.empty())
{
#pragma omp atomic
C_vote_infinity[c].second++;
} else {
#pragma omp atomic
C_vote_distance[c].second += hits_back[0].t;
}
}
}

I.resize(m);
for(int f = 0; f < m; ++f)
{
int c = C(f);
if (use_parity) {
I(f) = C_vote_parity[c].first > C_vote_parity[c].second ? 1 : 0;      

} else {
I(f) = (C_vote_infinity[c].first == C_vote_infinity[c].second && C_vote_distance[c].first <  C_vote_distance[c].second) ||
C_vote_infinity[c].first <  C_vote_infinity[c].second
? 1 : 0;
}
if (F.row(f) != FF.row(f))
I(f) = 1 - I(f);
}
if (is_verbose) cout << "done!" << endl;
}

template <
typename DerivedV,
typename DerivedF,
typename DerivedFF,
typename DerivedI>
IGL_INLINE void igl::embree::reorient_facets_raycast(
const Eigen::PlainObjectBase<DerivedV> & V,
const Eigen::PlainObjectBase<DerivedF> & F,
Eigen::PlainObjectBase<DerivedFF> & FF,
Eigen::PlainObjectBase<DerivedI> & I)
{
const int rays_total = F.rows()*100;
const int rays_minimum = 10;
const bool facet_wise = false;
const bool use_parity = false;
const bool is_verbose = false;
Eigen::VectorXi C;
reorient_facets_raycast(
V,F,rays_total,rays_minimum,facet_wise,use_parity,is_verbose,I,C);
FF.conservativeResize(F.rows(),F.cols());
for(int i = 0;i<I.rows();i++)
{
if(I(i))
{
FF.row(i) = (F.row(i).reverse()).eval();
}else
{
FF.row(i) = F.row(i);
}
}
}

#ifdef IGL_STATIC_LIBRARY
template void igl::embree::reorient_facets_raycast<Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, 1, 0, -1, 1> >(Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> >&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> >&);
template void igl::embree::reorient_facets_raycast<Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<bool, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1> >(Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const&, int, int, bool, bool, bool, Eigen::PlainObjectBase<Eigen::Matrix<bool, -1, 1, 0, -1, 1> >&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> >&);
template void igl::embree::reorient_facets_raycast<Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1> >(Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const&, int, int, bool, bool, bool, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> >&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> >&);
#endif
