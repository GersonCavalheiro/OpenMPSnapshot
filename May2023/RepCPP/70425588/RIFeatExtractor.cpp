
#include <cmath>
#include <limits>
#include <cassert>
#include <boost/math/special_functions/bessel.hpp>
#include <RIFeatures/struve.hpp>
#include <RIFeatures/RIFeatExtractor.hpp>

namespace RIFeatures
{


RIFeatExtractor::RIFeatExtractor()
{

}


RIFeatExtractor::RIFeatExtractor(const cv::Size image_size, const float basis_diameter, const int num_radii, const int num_rot_orders, const int num_fourier_coefs, const calculationMethod_enum method, const bool use_spatial_memoisation, const couplingMethod_enum couple_method, const featureSet_enum feature_set, const int max_derived_rotation_order, const basisType_enum basis_form)
{
initialise(image_size,basis_diameter,num_radii,num_rot_orders,num_fourier_coefs,method,use_spatial_memoisation,couple_method, feature_set, max_derived_rotation_order, basis_form);
}


void RIFeatExtractor::initialise(const cv::Size image_size, const float basis_diameter, const int num_radii, const int num_rot_orders, const int num_fourier_coefs, const calculationMethod_enum method, const bool use_spatial_memoisation, const couplingMethod_enum couple_method, const featureSet_enum feature_set, const int max_derived_rotation_order, const basisType_enum basis_form)
{
xsize = image_size.width;
ysize = image_size.height;
basis_radius = basis_diameter/2.0;
nj = (num_radii < 0) ? 0 : num_radii;
nk = (num_rot_orders > 6) ? 6 : num_rot_orders;
nm = (num_fourier_coefs < 0) ? 0 : num_fourier_coefs;
max_r = max_derived_rotation_order;
basis_type = basis_form;

pad_xsize = cv::getOptimalDFTSize(xsize);
pad_ysize = cv::getOptimalDFTSize(ysize);
planes[1] = cv::Mat::zeros(pad_ysize,pad_xsize,CV_32F);

switch(basis_type)
{
default:
case btSoftHist:
num_bases = (nm == 0) ? nj*(nk+1) : nj*(2*nk+1);
break;
case btZernike:
if (nm == 0)
num_bases = (nj%2 == 0) ? nj*(2+nj)/4 : (nj+1)*(nj+1)/4;
else
num_bases = nj*(nj+1)/2;
break;
} 

use_frequency = ((method == cmAuto) || (method == cmFrequency));
always_use_frequency = (method == cmFrequency);
auto_method = (method == cmAuto);
use_vectorised_coupling = ((feature_set == fsSimpleCouple) || (feature_set == fsExtraCouple)) && use_frequency && ((couple_method == comAuto) || (couple_method == comVectorised));
always_use_vectorised_coupling = use_frequency && (couple_method == comVectorised);
use_memoiser = use_spatial_memoisation && (!always_use_frequency);
if(use_frequency)
U_freq.resize(num_bases);
U_spat.resize(num_bases);
basis_j_list.resize(num_bases);
basis_k_list.resize(num_bases);
r_space = basis_radius/nj;
area_normaliser = basis_radius*basis_radius;
findSpatSupport();

switch(basis_type)
{
case btSoftHist:
if(use_frequency)
freqBasisSoftHist();

for(int j = 0, u = 0; j < nj; ++j)
{
for( int k = (nm==0) ? 0 : -nk; k <= nk; ++k)
{
spatBasisSoftHist(u,j,k);
basis_j_list[u] = j;
basis_k_list[u] = k;
++u;
} 
} 
break;

case btZernike:
if(use_frequency)
freqBasisZernike();

for(int j = 0, u = 0; j < nj; ++j)
{
for( int k = (nm==0) ? j%2 : -j; k <= j; k+=2)
{
spatBasisZernike(u,j,k);
basis_j_list[u] = j;
basis_k_list[u] = k;
++u;
} 
} 
break;

}

FFT_im.resize(nm+1);

raw_feat_j_list.clear();
raw_feat_k_list.clear();
raw_feat_m_list.clear();
raw_feat_r_list.clear();
raw_feat_basis_list.clear();
index_lookup_table.resize((nm+1)*num_bases);
for(int m = 0, f = 0; m <= nm; ++m)
{
for(int u = 0; u < num_bases; ++u)
{
const int k = basis_k_list[u];
const int j = basis_j_list[u];
if((m == 0) && (k < 0)) continue;
if((max_r >= 0) && std::abs(m-k) > max_r) continue;
index_lookup_table[m*num_bases+u] = raw_feat_j_list.size();
raw_feat_j_list.emplace_back(j);
raw_feat_k_list.emplace_back(k);
raw_feat_m_list.emplace_back(m);
raw_feat_r_list.emplace_back(k-m);
raw_feat_basis_list.emplace_back(u);
++f;
}
}
num_raw_features = raw_feat_j_list.size();

if(use_frequency)
{
raw_feat_images.resize(num_raw_features);
raw_feats_unpadded.resize(num_raw_features);

for(unsigned f = 0; f < raw_feat_frequency_creation_thread_lock.size(); ++f)
omp_destroy_lock(&(raw_feat_frequency_creation_thread_lock[f]));
raw_feat_frequency_creation_thread_lock.resize(num_raw_features);

raw_features_valid.resize(num_raw_features);
for(int f = 0; f < num_raw_features; ++f)
{
raw_features_valid[f] = false;
omp_init_lock(&(raw_feat_frequency_creation_thread_lock[f]));
}
}

num_pixels = xsize*ysize;
spat_memoiser.clear();
spat_memoiser_valid.clear();
if(use_memoiser)
{
spat_memoiser.resize(num_raw_features);
spat_memoiser_valid.resize(num_raw_features);

for(int f = 0; f < num_raw_features; ++f)
{
spat_memoiser[f].resize(num_pixels);
spat_memoiser_valid[f].resize(num_pixels);
std::fill(spat_memoiser_valid[f].begin(),spat_memoiser_valid[f].end(),false);
}
}

createDerivedFeatureList(feature_set);

raw_feature_magnitude_images.clear();
for(unsigned f = 0; f < magnitude_images_creation_locks.size(); ++f)
omp_destroy_lock(&(magnitude_images_creation_locks[f]));
magnitude_image_valid.clear();
if(use_frequency)
{
raw_feature_magnitude_images.resize(num_magnitude_features);
magnitude_images_creation_locks.resize(num_magnitude_features);
for(int f = 0; f < num_magnitude_features; ++f)
omp_init_lock(&(magnitude_images_creation_locks[f]));
magnitude_image_valid.resize(num_magnitude_features,false);
}

coupled_image_valid.clear();
coupled_images.clear();
for(unsigned f = 0; f < coupled_images_creation_locks.size(); ++f)
omp_destroy_lock(&(coupled_images_creation_locks[f]));
if(use_vectorised_coupling)
{
coupled_images.resize(num_coupled_images);
coupled_images_creation_locks.resize(num_coupled_images);
for(int f = 0; f < num_coupled_images; ++f)
omp_init_lock(&(coupled_images_creation_locks[f]));
coupled_image_valid.resize(num_coupled_images,false);
}

raw_feat_usage_this_frame.clear();
raw_feat_usage_last_frame.clear();
if(auto_method)
{
raw_feat_usage_this_frame.resize(num_raw_features,0);
raw_feat_usage_last_frame.resize(num_raw_features,0);
for(int f = 0; f < num_raw_features; ++f)
raw_feat_usage_last_frame[f] = std::numeric_limits<int>::max() - 1;

use_spatial_threshold = speedTestForThresh(); 
}
else
{
use_spatial_threshold = 0;
use_pixelwise_coupling_threshold = 0;
}

if( use_vectorised_coupling && (couple_method == comAuto) )
use_pixelwise_coupling_threshold = speedTestForCoupling();
else
use_pixelwise_coupling_threshold = 0;

}

RIFeatExtractor::~RIFeatExtractor(void)
{
for(unsigned i = 0; i < raw_feat_frequency_creation_thread_lock.size(); ++i)
omp_destroy_lock(&(raw_feat_frequency_creation_thread_lock[i]));

for(unsigned i = 0; i < magnitude_images_creation_locks.size(); ++i)
omp_destroy_lock(&(magnitude_images_creation_locks[i]));

for(unsigned i = 0; i < coupled_images_creation_locks.size(); ++i)
omp_destroy_lock(&(coupled_images_creation_locks[i]));

}


void RIFeatExtractor::createDerivedFeatureList(const featureSet_enum feature_set)
{
derived_feature_primary_list.clear();
derived_feature_secondary_list.clear();
derived_feature_type_list.clear();

raw_feat_to_magnitude_index.resize(num_raw_features);
num_magnitude_features = 0;

for( int f1 = 0; f1 < num_raw_features; ++f1)
{
if((raw_feat_k_list[f1] == 0) && (raw_feat_m_list[f1] == 0))
{
derived_feature_primary_list.emplace_back(f1);
derived_feature_secondary_list.emplace_back(int(C_SECOND_FEATURE_NONE));
derived_feature_type_list.emplace_back(ftReal);
raw_feat_to_magnitude_index[f1] = -1;
}
else if(raw_feat_r_list[f1] == 0)
{
derived_feature_primary_list.emplace_back(f1);
derived_feature_secondary_list.emplace_back(int(C_SECOND_FEATURE_NONE));
derived_feature_type_list.emplace_back(ftReal);
derived_feature_primary_list.emplace_back(f1);
derived_feature_secondary_list.emplace_back(int(C_SECOND_FEATURE_NONE));
derived_feature_type_list.emplace_back(ftImaginary);
raw_feat_to_magnitude_index[f1] = -1;
}
else
{
derived_feature_primary_list.emplace_back(f1);
derived_feature_secondary_list.emplace_back(int(C_SECOND_FEATURE_NONE));
derived_feature_type_list.emplace_back(ftMagnitude);
raw_feat_to_magnitude_index[f1] = num_magnitude_features++;
}
} 

coupled_image_index_lookup.clear();
num_coupled_images = 0;
if(use_vectorised_coupling)
{
coupled_image_index_lookup.resize(num_raw_features);
for(int f1 = 0; f1 < num_raw_features; ++f1)
coupled_image_index_lookup[f1].resize(num_raw_features-f1-1,-1);
}

if(feature_set == fsSimpleCouple)
{
for( int f1 = 0; f1 < num_raw_features - 1; ++f1)
{
for( int f2 = f1 + 1; f2 < num_raw_features; ++f2)
{
if((raw_feat_r_list[f1] == raw_feat_r_list[f2]) && (raw_feat_r_list[f1] != 0))
{
derived_feature_primary_list.emplace_back(f1);
derived_feature_secondary_list.emplace_back(f2);
derived_feature_type_list.emplace_back(ftReal);
derived_feature_primary_list.emplace_back(f1);
derived_feature_secondary_list.emplace_back(f2);
derived_feature_type_list.emplace_back(ftImaginary);
if(use_vectorised_coupling)
coupled_image_index_lookup[f1][f2-f1-1] = num_coupled_images++;
}
} 
} 
}
else if(feature_set == fsExtraCouple)
{
for( int f1 = 0; f1 < num_raw_features - 1; ++f1)
{
for( int f2 = f1 + 1; f2 < num_raw_features; f2++)
{
if((raw_feat_r_list[f1] != 0) && (raw_feat_r_list[f2] != 0))
{
derived_feature_primary_list.emplace_back(f1);
derived_feature_secondary_list.emplace_back(f2);
derived_feature_type_list.emplace_back(ftReal);
derived_feature_primary_list.emplace_back(f1);
derived_feature_secondary_list.emplace_back(f2);
derived_feature_type_list.emplace_back(ftImaginary);
if(use_vectorised_coupling)
coupled_image_index_lookup[f1][f2-f1-1] = num_coupled_images++;
}
} 
} 
}

num_derived_features = derived_feature_primary_list.size();
}

void RIFeatExtractor::findSpatSupport()
{
int x,y;
cv::MatIterator_<float> it_R, it_theta, end_R;

spat_basis_size = std::ceil(2*basis_radius);
if (spat_basis_size%2 == 0)
++spat_basis_size;
spat_basis_half_size = (spat_basis_size-1)/2;

const float midpoint = (spat_basis_size-1.0)/2.0;

R = cv::Mat_<float>::zeros(spat_basis_size,spat_basis_size);
theta = cv::Mat_<float>::zeros(spat_basis_size,spat_basis_size);

it_theta = theta.begin();
end_R = R.end();
x = 0;
y = 0;

for(it_R = R.begin(); it_R != end_R; ++it_R,++it_theta)
{
const float x_c = x - midpoint;
const float y_c = midpoint - y; 

*it_R = std::sqrt(std::pow(x_c,2) + std::pow(y_c,2));
*it_theta = std::atan2(y_c,x_c);

++x;
if (x == spat_basis_size)
{
x = 0;
++y;
}
} 
}

void RIFeatExtractor::spatBasisSoftHist(const int u, const int j, const int k)
{
cv::Mat_<float> radial_part, parts[2];

radial_part = 1.0 - cv::abs(R - j*r_space)/r_space;

if(k == 0)
{
threshold(radial_part,parts[0],0.0,0.0,cv::THRESH_TOZERO);
parts[1] = cv::Mat_<float>::zeros(spat_basis_size,spat_basis_size);
}
else
{
cv::MatIterator_<float> it = radial_part.begin();
it += (spat_basis_size + 1)*spat_basis_half_size;
it[0] = 0.0;

threshold(radial_part,radial_part,0.0,0.0,cv::THRESH_TOZERO);
polarToCart(radial_part,k*theta,parts[0],parts[1]);
}

merge(parts,2,U_spat[u]);

U_spat[u] /= area_normaliser;

const int reqhalfsize = std::ceil((j+1)*r_space) > spat_basis_half_size ? spat_basis_half_size : std::ceil((j+1)*r_space);
U_spat[u] = U_spat[u](cv::Range(spat_basis_half_size-reqhalfsize,spat_basis_half_size+reqhalfsize+1),cv::Range(spat_basis_half_size-reqhalfsize,spat_basis_half_size+reqhalfsize+1));

}

void RIFeatExtractor::spatBasisZernike(const int u, const int j, const int k)
{
int s, absk, smax;
float coef;
cv::Mat radial, mask, parts[2], term;

absk = std::abs(k);
smax = (j - absk)/2;

if(j == 0)
{
threshold(R,parts[0],basis_radius,1.0/area_normaliser,cv::THRESH_BINARY_INV);
parts[1] = cv::Mat::zeros(spat_basis_size,spat_basis_size, CV_32F);
merge(parts,2,U_spat[u]);
}
else
{
threshold(R,mask,basis_radius,1.0,cv::THRESH_TOZERO_INV);

radial = cv::Mat::zeros(spat_basis_size,spat_basis_size, CV_32F);
for(s = 0; s <= smax; s++)
{
coef = factorial(j-s)/(factorial(s)*factorial((j+absk)/2-s)*factorial((j-absk)/2-s));

if(j-2*s == 0)
threshold(R,term,basis_radius,1.0,cv::THRESH_BINARY_INV);
else
cv::pow(mask/basis_radius,j-2*s,term);

if(s % 2 == 0)
radial = radial + coef*term;
else
radial = radial - coef*term;
}

polarToCart(radial,k*theta,parts[0],parts[1]);
merge(parts,2,U_spat[u]);

U_spat[u] /= area_normaliser;
}

}

int RIFeatExtractor::factorial(const int n)
{
return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
}


std::complex<float> RIFeatExtractor::singleWindowFeature(const int raw_feature_num, const cv::Point p)
{
const int point_id = p.y*xsize + p.x;
if(use_memoiser)
{
if(spat_memoiser_valid[raw_feature_num][point_id])
return spat_memoiser[raw_feature_num][point_id];
}

cv::Mat_<cv::Vec2f> win, temp;
cv::Scalar s;
const int halfsize = (U_spat[raw_feat_basis_list[raw_feature_num]].rows-1)/2;
const cv::Rect roi(p-cv::Point(halfsize,halfsize),p+cv::Point(halfsize+1,halfsize+1));

if(nm == 0)
{
win = I_2chan(roi);
}
else
{
win = I_fou[raw_feat_m_list[raw_feature_num]](roi);
}

mulSpectrums(win,U_spat[raw_feat_basis_list[raw_feature_num]],temp,0);
s = cv::sum(temp);
const std::complex<float> result = std::complex<float>(s(0),s(1));

if(use_memoiser)
{
spat_memoiser[raw_feature_num][point_id] = result;
spat_memoiser_valid[raw_feature_num][point_id] = true;
}

return result;
}

void RIFeatExtractor::freqBasisSoftHist()
{
const float xsizef = (float) pad_xsize;
const float ysizef = (float) pad_ysize;
cv::Mat_<float> psi;

const int xswitch = (pad_xsize % 2 == 0) ? pad_xsize/2 : (pad_xsize+1)/2;
const int yswitch = (pad_ysize % 2 == 0) ? pad_ysize/2 : (pad_ysize+1)/2;

psi = cv::Mat_<float>::zeros(pad_ysize,pad_xsize);
for (int x = 0 ; x < pad_xsize; ++x)
{
for (int y = 0 ; y < pad_ysize; ++y)
{
if((x==0) && (y==0))
psi(0,0) = 0.0;
else
{
const float w_x = (x < xswitch) ? float(-x)/xsizef : (xsizef-float(x))/xsizef;
const float w_y = (y < yswitch) ? float(y)/ysizef : (float(y)- ysizef)/ysizef;
psi(y,x) = std::atan2(w_y,w_x);
}
}
}

#pragma omp parallel for
for(int k = 0; k <= nk; ++k)
{
int pos_k_basis_index = (nm == 0) ? k : nk + k;
int neg_k_basis_index = (nm == 0) ? 0 : nk - k;

std::vector<cv::Mat_<float>> intermediate_array(nj);

for(int j = 0; j < nj; ++j)
{
intermediate_array[j] = cv::Mat_<float>::zeros(pad_ysize,pad_xsize);
}

for (int x = 0 ; x < xswitch; ++x)
{
for (int y = 0 ; y < yswitch; ++y)
{
const float w_x = float(-x)/xsizef ;
const float w_y = float(y)/ysizef ;
const float rho = 2.0*M_PI*std::sqrt(w_x*w_x + w_y*w_y);

for(int j = 0; j < nj; ++j)
{
const float f = ((x == 0) && (y == 0)) ? 0.0 : coneHankel(rho,(j+1)*r_space,k)/area_normaliser;

intermediate_array[j](y,x) = f; 
if(x > 0)
intermediate_array[j](y,pad_xsize-x) = f; 
if(y > 0)
intermediate_array[j](pad_ysize-y,x) = f; 
if((x > 0) && (y > 0))
intermediate_array[j](pad_ysize-y,pad_xsize-x) = f; 
}
} 
} 

for(int j = 0; j < nj; ++j)
{
cv::Mat_<float> complex_parts[2], radial_part;

switch(j)
{
case 0:
radial_part = 2*M_PI*intermediate_array[0];
break;
case 1:
radial_part = 2*M_PI*(2*intermediate_array[1] - 2*intermediate_array[0]);
break;
default:
radial_part = 2*M_PI*((j+1)*intermediate_array[j] - 2*j*intermediate_array[j-1] + (j-1)*intermediate_array[j-2]);
break;
}

if(k == 0)
{
switch(j)
{
case 0:
radial_part(0,0) = (1.0/3.0)*M_PI*std::pow(r_space,2.0);
break;
case 1:
radial_part(0,0) = 2.0*(1.0/3.0)*M_PI*std::pow(2*r_space,2.0) - 2.0*(1.0/3.0)*M_PI*std::pow(r_space,2.0);
break;
default:
radial_part(0,0)= (j+1)*(1.0/3.0)*M_PI*std::pow((j+1)*r_space,2.0) - 2.0*j*(1.0/3.0)*M_PI*std::pow(j*r_space,2.0) + (j-1)*(1.0/3.0)*M_PI*std::pow((j-1)*r_space,2.0);
break;
}
radial_part(0,0) = radial_part(0,0)/area_normaliser;

} 

polarToCart(radial_part,k*psi,complex_parts[0],complex_parts[1]);

switch(k%4)
{
case 0: 
break;

case 1: 
{
cv::Mat_<float> temp = -complex_parts[0].clone();
complex_parts[0] = complex_parts[1];
complex_parts[1] = temp;
}
break;

case 2: 
complex_parts[0] *= -1.0;
complex_parts[1] *= -1.0;
break;

case 3: 
{
cv::Mat_<float> temp = complex_parts[0].clone();
complex_parts[0] = -complex_parts[1];
complex_parts[1] = temp;
}
break;
}
merge(complex_parts,2,U_freq[pos_k_basis_index]);

if(nm > 0)
{

if((k)%2 == 0)
complex_parts[1] *= -1.0;
else
complex_parts[0] *= -1.0;
merge(complex_parts,2,U_freq[neg_k_basis_index]);
}

if(nm == 0)
pos_k_basis_index += nk+1;
else
{
pos_k_basis_index += 2*nk+1;
neg_k_basis_index += 2*nk+1;
}
}
} 
}

void RIFeatExtractor::freqBasisZernike()
{
const float xsizef = (float) pad_xsize;
const float ysizef = (float) pad_ysize;
cv::Mat_<float> psi;

const int xswitch = (pad_xsize % 2 == 0) ? pad_xsize/2 : (pad_xsize+1)/2;
const int yswitch = (pad_ysize % 2 == 0) ? pad_ysize/2 : (pad_ysize+1)/2;

psi = cv::Mat_<float>::zeros(pad_ysize,pad_xsize);
for (int x = 0 ; x < pad_xsize; x++)
{
for (int y = 0 ; y < pad_ysize; y++)
{
if((x==0) && (y==0))
psi(0,0) = 0.0;
else
{
float w_x = (x < xswitch) ? float(-x)/xsizef : (xsizef-float(x))/xsizef;
float w_y = (y < yswitch) ? float(y)/ysizef : (float(y)- ysizef)/ysizef;
psi(y,x) = std::atan2(w_y,w_x);
}
}
}

#pragma omp parallel for
for(int j = 0; j < nj; j++)
{
int basis_index;
cv::Mat_<float> radial_part;

if (nm == 0)
basis_index = (j%2 == 0) ? j*(2+j)/4 : (j+1)*(j+1)/4;
else
basis_index = j*(j+1)/2;


radial_part = cv::Mat_<float>::zeros(pad_ysize,pad_xsize);

for (int x = 0 ; x < xswitch; ++x)
{
for (int y = 0 ; y < yswitch; ++y)
{
const float w_x = float(-x)/xsizef ;
const float w_y = float(y)/ysizef ;
const float rho = 2.0*M_PI*std::sqrt(w_x*w_x + w_y*w_y);

const float f = ((x == 0) && (y == 0)) ? 0.0 : boost::math::cyl_bessel_j(j+1,basis_radius*rho)/(rho*area_normaliser);

radial_part(y,x) = f; 
if(x > 0)
radial_part(y,pad_xsize-x) = f; 
if(y > 0)
radial_part(pad_ysize-y,x) = f; 
if((x > 0) && (y > 0))
radial_part(pad_ysize-y,pad_xsize-x) = f; 

} 
} 

for(int k = (nm == 0) ? j%2 : -j ; k <= j; k += 2)
{
cv::Mat_<float> complex_parts[2];

if(((j-std::abs(k))/2)%2 == 1)
radial_part *= -1;

if((k == 0) && (j == 0))
{
radial_part(0,0) = 1.0/(2.0*basis_radius);
}

switch(std::abs(k)%4)
{
case 0: 
case 4:
polarToCart(2*M_PI*basis_radius*radial_part,k*psi,complex_parts[0],complex_parts[1]);
break;
case 1: 
polarToCart(2*M_PI*basis_radius*radial_part,k*psi,complex_parts[1],complex_parts[0]);
complex_parts[1] *= -1;
break;
case 2: 
polarToCart(-2*M_PI*basis_radius*radial_part,k*psi,complex_parts[0],complex_parts[1]);
break;
case 3: 
polarToCart(2*M_PI*basis_radius*radial_part,k*psi,complex_parts[1],complex_parts[0]);
complex_parts[0] *= -1;
break;
}
if((k < 0) && (std::abs(k)%2 == 1))
{
complex_parts[0] *= -1;
complex_parts[1] *= -1;
}
merge(complex_parts,2,U_freq[basis_index]);

basis_index += 1;

} 

} 

}

float RIFeatExtractor::coneHankel(const float rho, const float a, const int k)
{
using boost::math::cyl_bessel_j;
float h = 0.0;

switch(std::abs(k))
{
case 0:
h = 1.0/(a*std::pow(rho,3))*struvePhi(a*rho) ;
break;
case 1:
h = struvePhi(rho*a)/std::pow(rho,2) - (2.0/std::pow(rho,2))*cyl_bessel_j(1,a*rho) + a*cyl_bessel_j(0,a*rho)/rho;
break;
case 2:
h = (1.0/std::pow(rho,2))*cyl_bessel_j(0,a*rho) + (2.0/std::pow(rho,2)) - (3.0/(a*std::pow(rho,3)))*struveLambda0(a*rho);
break;
case 3:
h = (8.0/(a*std::pow(rho,3)))*(cyl_bessel_j(0,a*rho) - 1.0) - (2.0/std::pow(rho,2))*cyl_bessel_j(1,a*rho) + (3.0/std::pow(rho,2))*struveLambda0(a*rho);
break;
case 4:
h = -(1.0/std::pow(rho,2))*cyl_bessel_j(0,a*rho) + (24.0/(a*std::pow(rho,3)))*cyl_bessel_j(1,a*rho) + (4.0/std::pow(rho,2)) - 15.0/(a*std::pow(rho,3))*struveLambda0(a*rho);
break;
case 5:
h = -(8.0/(a*std::pow(rho,3)))*cyl_bessel_j(0,a*rho) + (64.0/(std::pow(a,2)*std::pow(rho,4)) - 6.0/(std::pow(rho,2)))*cyl_bessel_j(1,a*rho) - (24.0/(a*std::pow(rho,3))) + 5.0/(std::pow(rho,2))*struveLambda0(a*rho);
break;
case 6:
h = (1.0/std::pow(rho,2) - 160.0/(std::pow(a,2)*std::pow(rho,4)))*cyl_bessel_j(0,a*rho) + (16.0/(a*std::pow(rho,3)) + 320.0/(std::pow(a,3)*std::pow(rho,5)))*cyl_bessel_j(1,a*rho) + (6.0/std::pow(rho,2)) - 35.0/(a*std::pow(rho,3))*struveLambda0(a*rho);
break;
} 

return h;
}

float RIFeatExtractor::struvePhi(const float x)
{
return 0.5*M_PI*x*(boost::math::cyl_bessel_j(1,x)*struveh0(x) - boost::math::cyl_bessel_j(0,x)*struveh1(x));
}

float RIFeatExtractor::struveLambda0(const float x)
{
return x*boost::math::cyl_bessel_j(0,x) + struvePhi(x);
}

int RIFeatExtractor::speedTestForThresh()
{
cv::Mat_<float> test_image, test_image_ori;
double t, time_taken_spat, time_taken_freq;
std::complex<float> out;
cv::Vec2f vec_result;

if(nm == 0)
{
test_image = cv::Mat::ones(ysize,xsize,CV_32F);
setScalarInputImage(test_image);
}
else
{
test_image = cv::Mat::ones(ysize,xsize,CV_32F);
test_image_ori = cv::Mat::ones(ysize,xsize,CV_32F);
setVectorInputImage(test_image,test_image_ori);
}

t = (double)cv::getTickCount();
for(int i = 0 ; i < C_SPEED_TEST_RUNS; ++i)
{
out = singleWindowFeature(i%num_raw_features,cv::Point(xsize/2, ysize/2));
dummy_result_float = out.real(); 
}
time_taken_spat = ((double)cv::getTickCount() - t)/((double)cv::getTickFrequency());

t = (double)cv::getTickCount();
for(int i = 0 ; i < C_SPEED_TEST_RUNS; ++i)
{
rawFeatureFrequencyCalculation(i%(num_bases),nm);
vec_result = raw_feats_unpadded[index_lookup_table[nm*num_bases+i%num_bases]].at<cv::Vec2f>(cv::Point(0,0));
dummy_result_float = vec_result[0]; 
}
time_taken_freq = ((double)cv::getTickCount() - t)/((double)cv::getTickFrequency());

return std::round(time_taken_freq/time_taken_spat);
}

int RIFeatExtractor::speedTestForCoupling()
{
double t, time_taken_mat, time_taken_elem;

cv::Mat_<cv::Vec2f> test_image_1, test_image_2;
cv::Mat_<float> temp_planes[2];
test_image_1 = cv::Mat_<float>::zeros(pad_ysize,pad_xsize);
test_image_2 = cv::Mat_<float>::zeros(pad_ysize,pad_xsize);
cv::randu(test_image_1,cv::Vec2f(0.0,0.0),cv::Vec2f(100.0,100.0));
cv::randu(test_image_2,cv::Vec2f(0.0,0.0),cv::Vec2f(100.0,100.0));

t = (double) cv::getTickCount();
for(int i = 0; i < C_COUPLING_SPEED_TEST_NUM; ++i)
{
cv::Mat_<cv::Vec2f> result;
cv::Mat_<cv::Vec2f> coupled_unnormalised;
cv::mulSpectrums(test_image_1,test_image_2,coupled_unnormalised,0,true);
cv::split(coupled_unnormalised,temp_planes);
cv::Mat_<float> mag;
cv::magnitude(temp_planes[0], temp_planes[1], mag);
cv::divide(temp_planes[0],mag,temp_planes[0]);
cv::divide(temp_planes[1],mag,temp_planes[1]);
cv::merge(temp_planes,2,coupled_images[0]);
}
time_taken_mat = (((double)cv::getTickCount() - t)/((double)cv::getTickFrequency()))/C_COUPLING_SPEED_TEST_NUM;

t = (double) cv::getTickCount();
for(int i = 0; i < C_COUPLING_SPEED_TEST_NUM; ++i)
{
float* const f1_val = test_image_1.ptr<float>(i%pad_ysize,i%pad_xsize);
const std::complex<float> c1 = std::complex<float>(f1_val[0],f1_val[1]);
float* const f2_val = test_image_2.ptr<float>((i+100)%pad_ysize,(i+100)%pad_xsize);
const std::complex<float>c2 = std::complex<float>(f2_val[0],f2_val[1]);
const std::complex<float> complex_result = c1*std::conj(c2);
dummy_result_float = complex_result.real()/std::abs(complex_result);
}
time_taken_elem = (((double)cv::getTickCount() - t)/((double)cv::getTickFrequency()))/C_COUPLING_SPEED_TEST_NUM;

return std::round(time_taken_mat/time_taken_elem);
}


void RIFeatExtractor::setScalarInputImage(const cv::Mat &in)
{
cv::Mat_<float> temp[2];
cv::Mat_<cv::Vec2f> padded;

if(I.channels() > 1)
{
cv::Mat one_chan;
cvtColor(in,one_chan,cv::COLOR_BGR2GRAY);
one_chan.convertTo(I,CV_32F);
}
else
in.convertTo(I,CV_32F);

temp[0] = I;
temp[1] = cv::Mat::zeros(I.rows,I.cols, CV_32F);
merge(temp,2,I_2chan);

if(use_frequency)
{
copyMakeBorder(I, planes[0], 0, pad_ysize - ysize, 0, pad_xsize - xsize, cv::BORDER_CONSTANT, cv::Scalar::all(0));  
merge(planes, 2, padded);

dft(padded,FFT_im[0]);
}

refreshImage();

}


void RIFeatExtractor::setVectorInputImage(const cv::Mat &in_magnitude, const cv::Mat &in_orientation)
{
in_magnitude.convertTo(I,CV_32FC1);
in_orientation.convertTo(I_ori,CV_32FC1);

expandFourierImages();

refreshImage();
}

void RIFeatExtractor::refreshImage()
{
raw_feat_usage_last_frame.swap(raw_feat_usage_this_frame);

std::fill(raw_feat_usage_this_frame.begin(),raw_feat_usage_this_frame.end(),0);
std::fill(raw_features_valid.begin(),raw_features_valid.end(),false);
std::fill(magnitude_image_valid.begin(),magnitude_image_valid.end(),false);
std::fill(coupled_image_valid.begin(),coupled_image_valid.end(),false);

if(use_memoiser)
for(int f = 0; f < num_raw_features; ++f)
std::fill(spat_memoiser_valid[f].begin(),spat_memoiser_valid[f].end(),false);
}

void RIFeatExtractor::expandFourierImages()
{
cv::Mat_<cv::Vec2f> padded;
cv::Mat_<float> temp_planes[2]; 

if(I_fou.size() == 0)
I_fou.resize(nm+1); 


I.copyTo(temp_planes[0]);
temp_planes[1] = cv::Mat_<float>::zeros(I.rows,I.cols);
merge(temp_planes,2,I_fou[0]);

if(use_frequency)
{
cv::Mat_<cv::Vec2f> padded;

if((I_fou[0].rows != pad_ysize) || (I_fou[0].cols != pad_xsize))
{
copyMakeBorder(I_fou[0], padded, 0, pad_ysize - ysize, 0, pad_xsize - xsize, cv::BORDER_CONSTANT, cv::Scalar::all(0));
dft(padded,FFT_im[0]);
}
else
dft(I_fou[0],FFT_im[0]);
}

#pragma omp parallel for
for(int m = 1; m <= nm; ++m)
{
cv::Mat_<cv::Vec2f> padded;
cv::Mat_<float> temp_planes_thread[2]; 

polarToCart(I,-m*I_ori,temp_planes_thread[0],temp_planes_thread[1]);

merge(temp_planes_thread,2,I_fou[m]);

if(use_frequency)
{
if((I_fou[m].rows != pad_ysize) || (I_fou[m].cols != pad_xsize))
{
copyMakeBorder(I_fou[m], padded, 0, pad_ysize - ysize, 0, pad_xsize - xsize, cv::BORDER_CONSTANT, cv::Scalar::all(0));
dft(padded,FFT_im[m]);
}
else
dft(I_fou[m],FFT_im[m]);
}
}
}


void RIFeatExtractor::createRawFeats()
{
for(int m = 0; m <= nm; ++m)
{
#pragma omp parallel for
for(int u = 0; u < num_bases; ++u)
{
if((m == 0) && (basis_k_list[u] < 0)) continue;
if((max_r >= 0) && std::abs(basis_k_list[u] - m) > max_r) continue;

rawFeatureFrequencyCalculation(u,m);
}
}
}

void RIFeatExtractor::rawFeatureFrequencyCalculation(const int f, const int u, const int m)
{
cv::Mat_<cv::Vec2f> temp;

cv::mulSpectrums(FFT_im[m],U_freq[u],temp,0);
cv::idft(temp, raw_feat_images[f],cv::DFT_SCALE);

raw_feats_unpadded[f] = raw_feat_images[f](cv::Range(0,ysize),cv::Range(0,xsize));
raw_features_valid[f] = true;
}

void RIFeatExtractor::rawFeatureFrequencyCalculation(const int u, const int m)
{
rawFeatureFrequencyCalculation(index_lookup_table[m*num_bases+u],u,m);
}

void RIFeatExtractor::rawFeatureFrequencyCalculation(const int f)
{
rawFeatureFrequencyCalculation(f,raw_feat_basis_list[f],raw_feat_m_list[f]);
}

void RIFeatExtractor::raiseComplexImageToPower(const cv::Mat_<cv::Vec2f>& in, cv::Mat_<cv::Vec2f>& out, const int power)
{
cv::Mat_<float> mag, arg;
cv::Mat_<float> temp_planes[2];
cv::split(in,temp_planes);
cv::cartToPolar(temp_planes[0],temp_planes[1],mag,arg);
cv::pow(mag,power,mag);
cv::polarToCart(mag,power*arg,temp_planes[0],temp_planes[1]);
cv::merge(temp_planes,2,out);
}

void RIFeatExtractor::fullImageCouple(cv::Mat_<cv::Vec2f>& coupled_image, const int f1, const int f2)
{
cv::Mat_<cv::Vec2f> coupled_unnormalised;
cv::Mat_<float> temp_planes[2];

if(raw_feat_r_list[f1] != raw_feat_r_list[f2])
{
cv::Mat_<cv::Vec2f> f1_power_image, f2_power_image;

if(raw_feat_r_list[f2] == 1)
f1_power_image = raw_feats_unpadded[f1];
else
raiseComplexImageToPower(raw_feats_unpadded[f1],f1_power_image,raw_feat_r_list[f2]);

if(raw_feat_r_list[f1] == 1)
f2_power_image = raw_feats_unpadded[f2];
else
raiseComplexImageToPower(raw_feats_unpadded[f2],f2_power_image,raw_feat_r_list[f1]);

mulSpectrums(f1_power_image,f2_power_image,coupled_unnormalised,0,true);
}
else 
mulSpectrums(raw_feats_unpadded[f1],raw_feats_unpadded[f2],coupled_unnormalised,0,true);

split(coupled_unnormalised,temp_planes);

cv::Mat_<float> mag;
cv::magnitude(temp_planes[0], temp_planes[1], mag);
cv::divide(temp_planes[0],mag,temp_planes[0]);
cv::divide(temp_planes[1],mag,temp_planes[1]);
cv::merge(temp_planes,2,coupled_image);

}

float RIFeatExtractor::derivedFeatureFromComplex(const std::complex<float> complex_feat, const featureType_enum type)
{
float result = 0.0;
switch(type)
{
case ftMagnitude:
assert(!std::isnan(std::abs(complex_feat)));
result = std::abs(complex_feat);
break;

case ftReal:
assert(!std::isnan(complex_feat.real()));
result = complex_feat.real();
break;

case ftImaginary:
assert(!std::isnan(complex_feat.real()));
result = complex_feat.imag();
break;
}
return result;
}

std::complex<float> RIFeatExtractor::coupleFeatures(std::complex<float> f1_val, const int r1, std::complex<float> f2_val, const int r2)
{
if(std::abs(f1_val) < 1.0/std::numeric_limits<float>::max())
f1_val = std::complex<float>(1.0,0.0);
else
f1_val /= std::abs(f1_val);

if(std::abs(f2_val) < 1.0/std::numeric_limits<float>::max())
f2_val = std::complex<float>(1.0,0.0);
else
f2_val /= std::abs(f2_val);

if(r1 != r2)
{
f2_val = std::pow(f2_val,r1);
f1_val = std::pow(f1_val,r2);
}

const std::complex<float> coupledcmplx = f1_val*std::conj(f2_val);
assert(!std::isnan(coupledcmplx.real()) && !std::isnan(coupledcmplx.imag()));

return coupledcmplx;
}

bool RIFeatExtractor::checkRawFeatureValidity(const int f, const bool calculate_if_invalid)
{
if(!use_frequency)
return false;

bool valid;

omp_set_lock(&(raw_feat_frequency_creation_thread_lock[f]));
if(raw_features_valid[f])
{
valid = true;
}
else
{
if(calculate_if_invalid)
{
rawFeatureFrequencyCalculation(f);
valid = true;
}
else
valid = false;
}

omp_unset_lock(&(raw_feat_frequency_creation_thread_lock[f]));
return valid;
}

bool RIFeatExtractor::checkCoupledImageValidity(const int f1, const int f2, const bool calculate_if_invalid, int& index)
{
bool valid;
assert(f2 > f1);
index = coupled_image_index_lookup[f1][f2-f1-1];
assert(index >= 0);

omp_set_lock(&(coupled_images_creation_locks[index]));
if(coupled_image_valid[index])
{
valid = true;
}
else
{
if(calculate_if_invalid)
{
fullImageCouple(coupled_images[index],f1,f2);
coupled_image_valid[index] = true;
valid = true;
}
else
valid = false;
}

omp_unset_lock(&(coupled_images_creation_locks[index]));
return valid;
}

int RIFeatExtractor::ensureMagnitudeImageValidity(const int f)
{
const int i = raw_feat_to_magnitude_index[f];
assert(i >= 0);
omp_set_lock(&(magnitude_images_creation_locks[i]));

if(!magnitude_image_valid[i])
{
cv::Mat_<float> temp_planes[2];
cv::split(raw_feats_unpadded[f],temp_planes);
cv::magnitude(temp_planes[0],temp_planes[1],raw_feature_magnitude_images[i]);
magnitude_image_valid[i] = true;
}

omp_unset_lock(&(magnitude_images_creation_locks[i]));
return i;
}



int RIFeatExtractor::getNumDerivedFeats() const
{
return num_derived_features;
}


int RIFeatExtractor::getNumRawFeats() const
{
return num_raw_features;
}


void RIFeatExtractor::getFeatsWithGivenR(const int r, std::vector<int>& raw_feat_ind, const bool include_negatives, const int Jmax) const
{
raw_feat_ind.clear();

for(int f = 0; f < num_raw_features; ++f)
if( ( (raw_feat_r_list[f] == r) || (include_negatives && (raw_feat_r_list[f] == -r) ) ) && ( (Jmax < 0) || (raw_feat_j_list[f] <= Jmax) ) )
raw_feat_ind.emplace_back(f);
}



void RIFeatExtractor::getFeatsUsingLowJ(const int Jmax, std::vector<int>& output_feature_list) const
{
output_feature_list.clear();

for(int f = 0; f < num_derived_features; ++f)
{
const int rf1 = derived_feature_primary_list[f];
const int rf2 = derived_feature_secondary_list[f];

if(raw_feat_j_list[rf1] > Jmax)
continue;

if( (rf2 != C_SECOND_FEATURE_NONE) && (raw_feat_j_list[rf2] > Jmax ))
continue;

output_feature_list.emplace_back(f);
}
}


int RIFeatExtractor::getMaxSpatBasisHalfsize(const int Jmax) const
{
if(Jmax < 0)
return spat_basis_half_size;
else
{
int u = 0;
while(basis_j_list[u] != Jmax) ++u;

return (U_spat[u].cols-1)/2;
}
}


cv::Mat RIFeatExtractor::getSpatialBasisCopy(const int basis_index) const
{
if((basis_index >= 0) && (basis_index < num_bases))
return U_spat[basis_index].clone();

return cv::Mat();
}


cv::Mat RIFeatExtractor::getFrequencyBasisCopy(const int basis_index) const
{
if((basis_index >= 0) && (basis_index < num_bases))
return U_freq[basis_index].clone();

return cv::Mat();
}


int RIFeatExtractor::getNumBases() const
{
return num_bases;
}


bool RIFeatExtractor::getBasisInfo(const int basis_index, int& j, int& k) const
{
if((basis_index >= 0) && (basis_index < num_bases))
{
j = basis_j_list[basis_index];
k = basis_k_list[basis_index];
return true;
}

j = -1;
k= -1;
return false;
}


bool RIFeatExtractor::stringToCalcMethod(const std::string& method_string, calculationMethod_enum& method_enum)
{
if(method_string == "spatial" || method_string == "s")
{
method_enum = cmSpatial;
return true;
}
if(method_string == "frequency" || method_string == "f")
{
method_enum = cmFrequency;
return true;
}
if(method_string == "auto" || method_string == "a")
{
method_enum = cmAuto;
return true;
}
return false;
}


bool RIFeatExtractor::stringToCoupleMethod(const std::string& method_string, couplingMethod_enum& method_enum)
{
if(method_string == "element-wise" || method_string == "e")
{
method_enum = comElementwise;
return true;
}
if(method_string == "vectorised" || method_string == "v")
{
method_enum = comVectorised;
return true;
}
if(method_string == "auto" || method_string == "a")
{
method_enum = comAuto;
return true;
}
return false;
}


bool RIFeatExtractor::stringBasisType(const std::string& basis_type_string, basisType_enum& basis_type_enum)
{
if(basis_type_string == "softhist" || basis_type_string == "s" || basis_type_string == "soft_histograms")
{
basis_type_enum = btSoftHist;
return true;
}
if(basis_type_string == "zernike" || basis_type_string == "z")
{
basis_type_enum = btZernike;
return true;
}
return false;
}


bool RIFeatExtractor::stringToFeatureSet(const std::string& feature_set_string, featureSet_enum& feature_set_enum)
{
if(feature_set_string == "basic" || feature_set_string == "b")
{
feature_set_enum = fsBasic;
return true;
}
if(feature_set_string == "couple_simple" || feature_set_string == "simple" || feature_set_string == "c")
{
feature_set_enum = fsSimpleCouple;
return true;
}
if(feature_set_string == "couple_extra" || feature_set_string == "extra" || feature_set_string == "ce")
{
feature_set_enum = fsExtraCouple;
return true;
}
return false;

}

} 
