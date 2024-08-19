#ifndef cimg_plugin_chlpca
#define cimg_plugin_chlpca
#define cimg_for_step1(bound,i,step) for (int i = 0; i<(int)(bound); i+=step)
#define cimg_for_stepX(img,x,step) cimg_for_step1((img)._width,x,step)
#define cimg_for_stepY(img,y,step) cimg_for_step1((img)._height,y,step)
#define cimg_for_stepZ(img,z,step) cimg_for_step1((img)._depth,z,step)
#define cimg_for_stepXY(img,x,y,step) cimg_for_stepY(img,y,step) cimg_for_stepX(img,x,step)
#define cimg_for_stepXYZ(img,x,y,step) cimg_for_stepZ(img,z,step) cimg_for_stepY(img,y,step) cimg_for_stepX(img,x,step)
#define cimg_forXY_window(img,xi,yi,xj,yj,rx,ry)                        \
for (int yi0=cimg::max(0,yi-ry), yi1=cimg::min(yi + ry,(int)img.height() - 1), yj=yi0;yj<=yi1;++yj) \
for (int xi0=cimg::max(0,xi-rx), xi1=cimg::min(xi + rx,(int)img.width() - 1), xj=xi0;xj<=xi1;++xj)
#define cimg_forXYZ_window(img,xi,yi,zi,xj,yj,zj,rx,ry,rz)                                      \
for (int zi0=cimg::max(0,zi-rz), zi1=cimg::min(zi + rz,(int)img.depth() - 1) , zj=zi0;zj<=zi1;++zj) \
for (int yi0=cimg::max(0,yi-ry), yi1=cimg::min(yi + ry,(int)img.height() - 1), yj=yi0;yj<=yi1;++yj) \
for (int xi0=cimg::max(0,xi-rx), xi1=cimg::min(xi + rx,(int)img.width() - 1) , xj=xi0;xj<=xi1;++xj)
CImg<T> get_patch(int x, int y, int z,
int px, int py, int pz) const {
if (depth() == 1){
const int x0 = x - px, y0 = y - py, x1 = x + px, y1 = y + py;
return get_crop(x0, y0, x1, y1).unroll('y');
} else {
const int
x0 = x - px, y0 = y - py, z0 = z - pz,
x1 = x + px, y1 = y + py, z1 = z + pz;
return get_crop(x0, y0, z0, x1, y1, z1).unroll('y');
}
}
CImg<T> get_patch_dictionnary(const int xi, const int yi, const int zi,
const int px, const int py, const int pz,
const int wx, const int wy, const int wz,
int & idc) const {
const int
n = (2*wx + 1) * (2*wy + 1) * (2 * (depth()==1?0:wz) + 1),
d = (2*px + 1) * (2*py + 1) * (2 * (depth()==1?0:px) + 1) * spectrum();
CImg<> S(n, d);
int idx = 0;
if (depth() == 1) {
cimg_forXY_window((*this), xi, yi, xj, yj, wx, wy){
CImg<T> patch = get_patch(xj, yj, 0, px, py, 1);
cimg_forY(S,y) S(idx,y) = patch(y);
if (xj==xi && yj==yi) idc = idx;
idx++;
}
} else  {
cimg_forXYZ_window((*this), xi,yi,zi,xj,yj,zj,wx,wy,wz){
CImg<T> patch = get_patch(xj, yj, zj, px, py, pz);
cimg_forY(S,y) S(idx,y) = patch(y);
if (xj==xi && yj==yi && zj==zi) idc = idx;
idx++;
}
}
S.columns(0, idx - 1);
return S;
}
CImg<T> & add_patch(const int xi, const int yi, const int zi,
const CImg<T> & patch,
const int px, const int py, const int pz) {
const int
x0 = xi - px, y0 = yi - py, z0 = (depth() == 1 ? 0 : zi - pz),
sx = 2 * px + 1, sy = 2 * py + 1, sz = (depth() == 1 ? 1 : 2 * pz +1);
draw_image(x0, y0, z0, 0, patch.get_resize(sx, sy, sz, spectrum(), -1), -1);
return (*this);
}
CImg<T> & add_patch(const int xi, const int yi, const int zi, const T value,
const int px, const int py, const int pz) {
const int
x0 = xi - px, y0 = yi - py, z0 = (depth() == 1 ? 0 : zi - pz),
x1 = xi + px, y1 = yi + py, z1 = (depth() == 1 ? 0 : zi + pz);
draw_rectangle(x0, y0, z0, 0, x1, y1, z1, spectrum()-1, value, -1);
return (*this);
}
CImg<T> get_chlpca(const int px, const int py, const int pz,
const int wx, const int wy, const int wz,
const int nstep, const float nsim,
const float lambda_min, const float threshold,
const float noise_std,  const bool pca_use_svd) const {
const int
nd = (2*px + 1) * (2*py + 1) * (depth()==1?1:2*pz + 1) * spectrum(),
K = (int)(nsim * nd);
#ifdef DEBUG
fprintf(stderr,"chlpca: p:%dx%dx%d,w:%dx%dx%d,nd:%d,K:%d\n",
2*px + 1,2*py + 1,2*pz + 1,2*wx + 1,2*wy + 1,2*wz + 1,nd,K);
#endif
float sigma;
if (noise_std<0) sigma = (float)std::sqrt(variance_noise());
else sigma = noise_std;
CImg<T> dest(*this), count(*this);
dest.fill(0);
count.fill(0);
cimg_for_stepZ(*this,zi,(depth()==1||pz==0)?1:nstep){
#ifdef cimg_use_openmp
#pragma omp parallel for
#endif
cimg_for_stepXY((*this),xi,yi,nstep){
int idc = 0;
CImg<T> S = get_patch_dictionnary(xi,yi,zi,px,py,pz,wx,wy,wz,idc);
CImg<T> Sk(S);
CImg<unsigned int> index(S.width());
if (K < Sk.width() - 1){
CImg<T> mse(S.width());
CImg<unsigned int> perms;
cimg_forX(S,x) { mse(x) = (T)S.get_column(idc).MSE(S.get_column(x)); }
mse.sort(perms,true);
cimg_foroff(perms,i) {
cimg_forY(S,j) Sk(i,j) = S(perms(i),j);
index(perms(i)) = i;
}
Sk.columns(0, K);
perms.threshold(K);
} else {
cimg_foroff(index,i) index(i)=i;
}
CImg<T> M(1, Sk.height(), 1, 1, 0);
cimg_forXY(Sk,x,y) { M(y) += Sk(x,y); }
M /= (T)Sk.width();
cimg_forXY(Sk,x,y) { Sk(x,y) -= M(y); }
CImg<T> P, lambda;
if (pca_use_svd) {
CImg<T> V;
Sk.get_transpose().SVD(V,lambda,P,true,100);
} else {
(Sk * Sk.get_transpose()).symmetric_eigen(lambda, P);
lambda.sqrt();
}
int s = 0;
const T tx = (T)(std::sqrt((double)Sk.width()-1.0) * lambda_min * sigma);
while((lambda(s) > tx) && (s < ((int)lambda.size() - 1))) { s++; }
P.columns(0,s);
Sk = P.get_transpose() * Sk;
if (threshold > 0) { Sk.threshold(threshold, 1); }
Sk =  P * Sk;
cimg_forXY(Sk,x,y) { Sk(x,y) += M(y); }
int j = 0;
cimg_forXYZ_window((*this),xi,yi,zi,xj,yj,zj,wx,wy,wz){
const int id = index(j);
if (id < Sk.width()) {
dest.add_patch(xj, yj, zj, Sk.get_column(id), px, py, pz);
count.add_patch(xj, yj, zj, (T)1, px, py, pz);
}
j++;
}
}
}
cimg_foroff(dest, i) {
if(count(i) != 0) { dest(i) /= count(i); }
else { dest(i) = (*this)(i); }
}
return dest;
}
CImg<T> & chlpca(const int px, const int py, const int pz,
const int wx, const int wy, const int wz,
const int nstep, const float nsim,
const float lambda_min, const float threshold,
const float noise_std,  const bool pca_use_svd)  {
(*this) = get_chlpca(px, py, pz, wx, wy, wz, nstep, nsim, lambda_min,
threshold, noise_std, pca_use_svd);
return (*this);
}
CImg<T> get_chlpca(const int p=3, const int w=10,
const int nstep=5, const float nsim=10,
const float lambda_min=2, const float threshold = -1,
const float noise_std=-1, const bool pca_use_svd=true) const {
if (depth()==1) return get_chlpca(p, p, 0, w, w, 0, nstep, nsim, lambda_min,
threshold, noise_std, pca_use_svd);
else return get_chlpca(p, p, p, w, w, w, nstep, nsim, lambda_min,
threshold, noise_std, pca_use_svd);
}
CImg<T> chlpca(const int p=3, const int w=10,
const int nstep=5, const float nsim=10,
const float lambda_min=2, const float threshold = -1,
const float noise_std=-1, const bool pca_use_svd=true) {
(*this) = get_chlpca(p, w, nstep, nsim, lambda_min,
threshold, noise_std, pca_use_svd);
return (*this);
}
#endif 
