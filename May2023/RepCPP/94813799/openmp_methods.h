


#ifndef MAPPEL_OPENMP_METHODS
#define MAPPEL_OPENMP_METHODS

#include <omp.h>
#include "Mappel/OMPExceptionCatcher/OMPExceptionCatcher.h"
#include "Mappel/util.h"
#include "Mappel/mcmc.h"

using omp_exception_catcher::Strategy;

namespace mappel {
namespace methods {
inline namespace openmp {


template<class Model>
void sample_prior_stack(const Model &model, ParamVecT<Model> &theta_stack)
{
int nthetas = static_cast<int>(theta_stack.n_cols);
omp_exception_catcher::OMPExceptionCatcher catcher(omp_exception_catcher::Strategy::Continue);
#pragma omp parallel
{
auto &rng = model.get_rng_generator();
#pragma omp for
for(int n=0; n<nthetas; n++){
catcher.run([&]{theta_stack.col(n) = model.sample_prior(rng);});
}
}
catcher.rethrow();
}


template<class Model>
void model_image_stack(const Model &model,
const ParamVecT<Model> &theta_stack,
ImageStackT<Model> &image_stack)
{
int nthetas = static_cast<int>(theta_stack.n_cols);
omp_exception_catcher::OMPExceptionCatcher catcher;
#pragma omp parallel for
for(int n=0; n<nthetas; n++)
catcher.run([&]{
model.set_image_in_stack(image_stack,n, model_image(model, theta_stack.col(n)));
});
catcher.rethrow();
}


template<class Model>
void simulate_image_stack(const Model &model,
const ParamVecT<Model> &theta_stack,
ImageStackT<Model> &image_stack)
{
int nimages = model.get_size_image_stack(image_stack);
int nthetas = static_cast<int>(theta_stack.n_cols);
if (nimages==1 && nthetas==1) {
auto &rng = model.get_rng_generator();
model.set_image_in_stack(image_stack,0,simulate_image(model,theta_stack.col(0),rng));
} else {
omp_exception_catcher::OMPExceptionCatcher catcher;
if (nthetas==1) {
auto model_im=model_image(model, theta_stack.col(0));
#pragma omp parallel
{
auto &rng = model.get_rng_generator();
#pragma omp for
for(int n=0; n<nimages; n++)
catcher.run([&]{
model.set_image_in_stack(image_stack,n,simulate_image_from_model(model, model_im,rng));
});
}
} else {
#pragma omp parallel
{
auto &rng = model.get_rng_generator();
#pragma omp for
for(int n=0; n<nimages; n++)
catcher.run([&]{
model.set_image_in_stack(image_stack,n,simulate_image(model,theta_stack.col(n),rng));
});
}
}
catcher.rethrow();
}
}

template<class Model>
void cr_lower_bound_stack(const Model &model,
const ParamVecT<Model> &theta_stack,
ParamVecT<Model> &crlb_stack)
{
int nthetas = static_cast<int>(theta_stack.n_cols);
omp_exception_catcher::OMPExceptionCatcher catcher;
#pragma omp parallel for
for(int n=0; n<nthetas; n++)
catcher.run([&]{
crlb_stack.col(n) = methods::cr_lower_bound(model,theta_stack.col(n));
});
catcher.rethrow();
}

template<class Model>
void expected_information_stack(const Model &model,
const ParamVecT<Model> &theta_stack,
CubeT &fisherI_stack)
{
int nthetas = static_cast<int>(theta_stack.n_cols);
omp_exception_catcher::OMPExceptionCatcher catcher;
#pragma omp parallel for
for(int n=0; n<nthetas; n++)
catcher.run([&]{
fisherI_stack.slice(n) = methods::expected_information(model,theta_stack.col(n));
});
catcher.rethrow();
}

template<class Model>
void estimate_max_stack(const Model &model, const ModelDataStackT<Model> &data_stack, const std::string &method,
estimator::MLEDataStack &mle_data_stack)
{
auto estimator = make_estimator(model,method);
auto theta_init=model.make_param_stack(model.get_size_image_stack(data_stack));
theta_init.zeros();
estimator->estimate_max_stack(data_stack, theta_init, mle_data_stack);
}

template<class Model>
void estimate_max_stack(const Model &model, const ModelDataStackT<Model> &data_stack, const std::string &method,
estimator::MLEDataStack &mle_data_stack, StatsT &stats)
{
auto estimator = make_estimator(model,method);
auto theta_init=model.make_param_stack(model.get_size_image_stack(data_stack));
theta_init.zeros();
estimator->estimate_max_stack(data_stack, theta_init, mle_data_stack);
stats = estimator->get_stats();
}

template<class Model>
void estimate_max_stack(const Model &model, const ModelDataStackT<Model> &data_stack,  const std::string &method,
ParamVecT<Model> &theta_init_stack, estimator::MLEDataStack &mle_data_stack)
{
auto estimator = make_estimator(model,method);
estimator->estimate_max_stack(data_stack, theta_init_stack, mle_data_stack);
}


template<class Model>
void estimate_max_stack(const Model &model, const ModelDataStackT<Model> &data_stack, const std::string &method,
ParamVecT<Model> &theta_init_stack, estimator::MLEDataStack &mle_data_stack, StatsT &stats)
{
auto estimator = make_estimator(model,method);
estimator->estimate_max_stack(data_stack, theta_init_stack, mle_data_stack);
stats = estimator->get_stats();
}


template<class Model>
void estimate_profile_likelihood_stack(const Model &model, const ModelDataT<Model> &data, const std::string &method,
const ParamVecT<Model> &fixed_theta_init,
estimator::ProfileLikelihoodData &est)
{
auto estimator = make_estimator(model,method);
estimator->estimate_profile_max(data, fixed_theta_init, est);
}

template<class Model>
void estimate_profile_likelihood_stack(const Model &model, const ModelDataT<Model> &data, const std::string &method,
const ParamVecT<Model> &fixed_theta_init, estimator::ProfileLikelihoodData &est, StatsT &stats)
{
auto estimator = make_estimator(model,method);
estimator->estimate_profile_max(data, fixed_theta_init, est);
stats = estimator->get_stats();
}


template <class Model>
void estimate_posterior_stack(const Model &model, const ModelDataStackT<Model> &data_stack,
const ParamVecT<Model> &theta_init_stack, mcmc::MCMCDataStack &est)
{
auto Np = model.get_num_params();
est.Ndata = model.get_size_image_stack(data_stack);
est.initialize_arrays(Np);
auto Noversample = mcmc::num_oversample(est.Nsample,est.Nburnin,est.thin);
omp_exception_catcher::OMPExceptionCatcher catcher;
#pragma omp parallel
{
auto oversample = model.make_param_stack(Noversample);
VecT oversample_rllh(Noversample);
auto sample = model.make_param_stack(est.Nsample);
VecT sample_rllh(est.Nsample);
auto sample_mean = model.make_param();
auto sample_cov = model.make_param_mat();
auto credible_lb = model.make_param();
auto credible_ub = model.make_param();
#pragma omp for
for(IdxT n=0; n<est.Ndata; n++)
catcher.run([&]{
auto data = model.get_image_from_stack(data_stack,n);
mcmc::sample_posterior(model, data, model.initial_theta_estimate(data, theta_init_stack.col(n)), oversample, oversample_rllh);
mcmc::thin_sample(oversample, oversample_rllh, est.Nburnin, est.thin, sample, sample_rllh);
mcmc::estimate_sample_posterior(sample, sample_mean, sample_cov);
mcmc::compute_posterior_credible(sample, est.confidence, credible_lb, credible_ub);
est.credible_lb.col(n) = credible_lb;
est.credible_ub.col(n) = credible_ub;
est.sample.slice(n) = sample;
est.sample_rllh.col(n) = sample_rllh;
est.sample_mean.col(n) = sample_mean;
est.sample_cov.slice(n) = sample_cov;
});
}
catcher.rethrow();
}

template <class Model>
void estimate_posterior_stack(const Model &model, const ModelDataStackT<Model> &data_stack, mcmc::MCMCDataStack &est)
{
IdxT count = model.get_size_image_stack(data_stack);
auto theta_init_stack = model.make_param_stack(count,arma::fill::zeros);
estimate_posterior_stack(model, data_stack, theta_init_stack,est);
}

template<class Model>
void error_bounds_expected_stack(const Model &model, const MatT &theta_est_stack, double confidence,
MatT &theta_lb_stack, MatT &theta_ub_stack)
{
auto count = theta_est_stack.n_cols;
theta_lb_stack.set_size(model.get_num_params(), count);
theta_ub_stack.set_size(model.get_num_params(), count);
double z = normal_quantile_twosided(confidence);

omp_exception_catcher::OMPExceptionCatcher catcher;
#pragma omp parallel for
for(IdxT n=0; n<count; n++)
catcher.run([&]{
auto theta_est = theta_est_stack.col(n);
VecT bnd = z*arma::sqrt(cr_lower_bound(model, theta_est));
theta_lb_stack.col(n) = arma::max(model.get_lbound(), theta_est-bnd);
theta_ub_stack.col(n) = arma::min(model.get_ubound(), theta_est+bnd);
});
catcher.rethrow();
}

template<class Model>
void error_bounds_observed_stack(const Model &model, const MatT &theta_est_stack, CubeT &obsI_stack, double confidence,
MatT &theta_lb_stack, MatT &theta_ub_stack)
{
auto count = theta_est_stack.n_cols;
theta_lb_stack.set_size(model.get_num_params(), count);
theta_ub_stack.set_size(model.get_num_params(), count);
double z = normal_quantile_twosided(confidence);
if(obsI_stack.n_slices != count) {
std::ostringstream msg;
msg<<"Got inconsistent sizes.  Num theta_est:"<<count<<" #obsI:"<<obsI_stack.n_slices;
ArrayShapeError(msg.str());
}

omp_exception_catcher::OMPExceptionCatcher catcher;
#pragma omp parallel for
for(IdxT n=0; n<count; n++)
catcher.run([&]{
auto theta_est = theta_est_stack.col(n);
VecT bnd = z*arma::sqrt(arma::pinv(obsI_stack.slice(n)).eval().diag());
theta_lb_stack.col(n) = arma::max(model.get_lbound(), theta_est-bnd);
theta_ub_stack.col(n) = arma::min(model.get_ubound(), theta_est+bnd);
});
catcher.rethrow();
}


template<class Model>
void error_bounds_profile_likelihood_parallel(const Model &model, const ModelDataStackT<Model> &image, estimator::ProfileBoundsData &est, StatsT &stats)
{
estimator::NewtonMaximizer<Model> estimator(model);
if(!std::isfinite(est.target_rllh_delta)) est.target_rllh_delta =  -.5*chisq_quantile(est.confidence);
if(est.estimated_idxs.is_empty()) est.estimated_idxs = arma::regspace<IdxVecT>(0,model.get_num_params()-1);
estimator.estimate_profile_bounds_parallel(image,est);
stats = estimator.get_stats();
}

template<class Model>
void error_bounds_profile_likelihood_parallel(const Model &model, const ModelDataT<Model> &image, estimator::ProfileBoundsData &est)
{
estimator::NewtonMaximizer<Model> estimator(model);
if(!std::isfinite(est.target_rllh_delta)) est.target_rllh_delta =  -.5*chisq_quantile(est.confidence);
if(est.estimated_idxs.is_empty()) est.estimated_idxs = arma::regspace<IdxVecT>(0,model.get_num_params()-1);
estimator.estimate_profile_bounds_parallel(image,est);
}

template<class Model>
void error_bounds_profile_likelihood_stack(const Model &model, const ModelDataStackT<Model> &image, estimator::ProfileBoundsDataStack &est, StatsT &stats)
{
estimator::NewtonMaximizer<Model> estimator(model);
if(!std::isfinite(est.target_rllh_delta)) est.target_rllh_delta =  -.5*chisq_quantile(est.confidence);
if(est.estimated_idxs.is_empty()) est.estimated_idxs = arma::regspace<IdxVecT>(0,model.get_num_params()-1);
estimator.estimate_profile_bounds_stack(image,est);
stats = estimator.get_stats();
}

template<class Model>
void error_bounds_profile_likelihood_stack(const Model &model, const ModelDataStackT<Model> &image, estimator::ProfileBoundsDataStack &est)
{
estimator::NewtonMaximizer<Model> estimator(model);
if(!std::isfinite(est.target_rllh_delta)) est.target_rllh_delta =  -.5*chisq_quantile(est.confidence);
if(est.estimated_idxs.is_empty()) est.estimated_idxs = arma::regspace<IdxVecT>(0,model.get_num_params()-1);
estimator.estimate_profile_bounds_stack(image,est);
}

} 


namespace objective {

inline namespace openmp {


template<class Model>
void llh_stack(const Model &model, const ImageT<Model> &image, const ParamVecT<Model> &theta_stack, VecT &llh_stack)
{
auto nthetas = theta_stack.n_cols;
llh_stack.set_size(nthetas);
omp_exception_catcher::OMPExceptionCatcher catcher;
#pragma omp parallel for
for(IdxT n=0; n<theta_stack.n_cols; n++)
catcher.run([&]{
llh_stack(n) = methods::objective::llh(model, image, theta_stack.col(n));
});
catcher.rethrow();
}


template<class Model>
void llh_stack(const Model &model, const ImageStackT<Model> &image_stack, const ParamVecT<Model> &theta_stack, VecT &llh_stack)
{
IdxT nimages = model.get_size_image_stack(image_stack);
IdxT nthetas = theta_stack.n_cols;
model.check_param_shape(theta_stack);
model.check_image_shape(image_stack);
llh_stack.set_size(std::max(nthetas,nimages));
if (nimages==1 && nthetas==1) {
llh_stack(0) = objective::llh(model, model.get_image_from_stack(image_stack,0), theta_stack.col(0));
} else {
omp_exception_catcher::OMPExceptionCatcher catcher;
if (nthetas==1) {
auto s = model.make_stencil(theta_stack.col(0));
#pragma omp parallel for
for(IdxT n=0; n<nimages; n++)
catcher.run([&]{
llh_stack(n) = objective::llh(model, model.get_image_from_stack(image_stack,n), s);
});
} else if (nimages==1) {
#pragma omp parallel for
for(IdxT n=0; n<nthetas; n++)
catcher.run([&]{
llh_stack(n) = objective::llh(model, model.get_image_from_stack(image_stack,0), theta_stack.col(n));
});
} else {
#pragma omp parallel for
for(IdxT n=0; n<nimages; n++)
catcher.run([&]{
llh_stack(n) = objective::llh(model, model.get_image_from_stack(image_stack,n), theta_stack.col(n));
});
}
catcher.rethrow();
}
}


template<class Model>
void rllh_stack(const Model &model, const ImageStackT<Model> &image_stack, const ParamVecT<Model> &theta_stack, VecT &rllh_stack)
{
IdxT nimages = model.get_size_image_stack(image_stack);
IdxT nthetas = theta_stack.n_cols;
model.check_param_shape(theta_stack);
model.check_image_shape(image_stack);
rllh_stack.set_size(std::max(nthetas,nimages));
if (nimages==1 && nthetas==1) {
rllh_stack(0) = objective::rllh(model, model.get_image_from_stack(image_stack,0), theta_stack.col(0));
} else {
omp_exception_catcher::OMPExceptionCatcher catcher;
if (nthetas==1) {
auto s = model.make_stencil(theta_stack.col(0));
#pragma omp parallel for
for(IdxT n=0; n<nimages; n++)
catcher.run([&]{
rllh_stack(n) = objective::rllh(model, model.get_image_from_stack(image_stack,n), s);
});
} else if (nimages==1) {
#pragma omp parallel for
for(IdxT n=0; n<nthetas; n++)
catcher.run([&]{
rllh_stack(n) = objective::rllh(model, model.get_image_from_stack(image_stack,0), theta_stack.col(n));
});
} else {
#pragma omp parallel for
for(IdxT n=0; n<nimages; n++)
catcher.run([&]{
rllh_stack(n) = objective::rllh(model, model.get_image_from_stack(image_stack,n), theta_stack.col(n));
});
}
catcher.rethrow();
}
}

template<class Model>
void rllh_stack(const Model &model, const ImageT<Model> &image, const ParamVecT<Model> &theta_stack, VecT &rllh_stack)
{
IdxT nthetas = theta_stack.n_cols;
model.check_param_shape(theta_stack);
model.check_image_shape(image);
rllh_stack.set_size(nthetas);
omp_exception_catcher::OMPExceptionCatcher catcher;
#pragma omp parallel for
for(IdxT n=0; n<nthetas; n++)
catcher.run([&]{
rllh_stack(n) = objective::rllh(model, image, theta_stack.col(n));
});
catcher.rethrow();
}



template<class Model>
void grad_stack(const Model &model, const ImageStackT<Model> &image_stack, const ParamVecT<Model> &theta_stack,
ParamVecT<Model> &grad_stack)
{
IdxT nimages = model.get_size_image_stack(image_stack);
IdxT nthetas = theta_stack.n_cols;
model.check_param_shape(theta_stack);
model.check_image_shape(image_stack);
grad_stack.set_size(model.get_num_params(),std::max(nthetas,nimages));
if (nimages==1 && nthetas==1) {
grad_stack.col(0) = objective::grad(model, model.get_image_from_stack(image_stack,0), theta_stack.col(0));
} else {
omp_exception_catcher::OMPExceptionCatcher catcher;
if (nthetas==1) { 
auto s = model.make_stencil(theta_stack.col(0));
#pragma omp for
for(IdxT n=0; n<nimages; n++)
catcher.run([&]{
grad_stack.col(n) = objective::grad(model, model.get_image_from_stack(image_stack,n), s);
});
} else if (nimages==1) { 
#pragma omp parallel for
for(IdxT n=0; n<nthetas; n++)
catcher.run([&]{
grad_stack.col(n) = objective::grad(model, model.get_image_from_stack(image_stack,0), theta_stack.col(n));
});
} else {
#pragma omp parallel for
for(IdxT n=0; n<nthetas; n++)
catcher.run([&]{
grad_stack.col(n) = objective::grad(model, model.get_image_from_stack(image_stack,n), theta_stack.col(n));
});
}
catcher.rethrow();
}
}


template<class Model>
void hessian_stack(const Model &model, const ImageStackT<Model> &image_stack, const ParamVecT<Model> &theta_stack,
CubeT &hessian_stack)
{
IdxT nimages = model.get_size_image_stack(image_stack);
IdxT nthetas = theta_stack.n_cols;
model.check_param_shape(theta_stack);
model.check_image_shape(image_stack);
hessian_stack.set_size(model.get_num_params(),model.get_num_params(),std::max(nthetas,nimages));
if (nimages==1 && nthetas==1) {
hessian_stack.slice(0) = objective::hessian(model, model.get_image_from_stack(image_stack,0), theta_stack.col(0));
} else {
omp_exception_catcher::OMPExceptionCatcher catcher;
if (nthetas==1) { 
auto s = model.make_stencil(theta_stack.col(0));
#pragma omp for
for(IdxT n=0; n<nimages; n++)
catcher.run([&]{
hessian_stack.slice(n) = objective::hessian(model, model.get_image_from_stack(image_stack,n), s);
});
} else if (nimages==1) { 
#pragma omp parallel for
for(IdxT n=0; n<nthetas; n++)
catcher.run([&]{
hessian_stack.slice(n) = objective::hessian(model, model.get_image_from_stack(image_stack,0), theta_stack.col(n));
});
} else {
#pragma omp parallel for
for(IdxT n=0; n<nthetas; n++)
catcher.run([&]{
hessian_stack.slice(n) = objective::hessian(model, model.get_image_from_stack(image_stack,n), theta_stack.col(n));
});
}
catcher.rethrow();
}
}


template<class Model>
void negative_definite_hessian_stack(const Model &model, const ImageStackT<Model> &image_stack, const ParamVecT<Model> &theta_stack,
CubeT &hessian_stack)
{
IdxT nimages = model.get_size_image_stack(image_stack);
IdxT nthetas = theta_stack.n_cols;
model.check_param_shape(theta_stack);
model.check_image_shape(image_stack);
hessian_stack.set_size(model.get_num_params(),model.get_num_params(),std::max(nthetas,nimages));
if (nimages==1 && nthetas==1) {
hessian_stack.slice(0) = objective::negative_definite_hessian(model, model.get_image_from_stack(image_stack,0), theta_stack.col(0));
} else {
omp_exception_catcher::OMPExceptionCatcher catcher;
if (nthetas==1) { 
auto s=model.make_stencil(theta_stack.col(0));
#pragma omp parallel for
for(IdxT n=0; n<nimages; n++)
catcher.run([&]{
hessian_stack.slice(n) = objective::negative_definite_hessian(model, model.get_image_from_stack(image_stack,n), s);
});
} else if (nimages==1) { 
#pragma omp parallel for
for(IdxT n=0; n<nthetas; n++)
catcher.run([&]{
hessian_stack.slice(n) = objective::negative_definite_hessian(model, model.get_image_from_stack(image_stack,0), theta_stack.col(n));
});
} else {
#pragma omp parallel for
for(IdxT n=0; n<nthetas; n++)
catcher.run([&]{
hessian_stack.slice(n) = objective::negative_definite_hessian(model, model.get_image_from_stack(image_stack,n), theta_stack.col(n));
});
}
catcher.rethrow();
}
}

} 
} 
} 
} 

#endif 
