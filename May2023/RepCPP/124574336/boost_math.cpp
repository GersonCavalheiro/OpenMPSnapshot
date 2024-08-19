



#include "stdafx.h"

#ifdef _MSC_VER
#  pragma warning(disable: 4400) 
#  pragma warning(disable: 4244) 
#  pragma warning(disable: 4512) 
#  pragma warning(disable: 4127) 
#endif

#include "boost_math.h"

namespace boost_math
{

any_distribution::any_distribution(int t, double arg1, double arg2, double arg3)
{
TRANSLATE_EXCEPTIONS_BEGIN
switch(t) 
{  
case 0:
this->reset(new concrete_distribution<boost::math::bernoulli>(boost::math::bernoulli(arg1)));
break;
case 1:
this->reset(new concrete_distribution<boost::math::beta_distribution<> >(boost::math::beta_distribution<>(arg1, arg2)));
break; 
case 2:
this->reset(new concrete_distribution<boost::math::binomial_distribution<> >(boost::math::binomial_distribution<>(arg1, arg2)));
break; 
case 3:
this->reset(new concrete_distribution<boost::math::cauchy>(boost::math::cauchy(arg1, arg2)));
break;
case 4:
this->reset(new concrete_distribution<boost::math::chi_squared>(boost::math::chi_squared(arg1)));
break;
case 5:
this->reset(new concrete_distribution<boost::math::exponential>(boost::math::exponential(arg1)));
break;
case 6:
this->reset(new concrete_distribution<boost::math::extreme_value>(boost::math::extreme_value(arg1)));
break;
case 7:
this->reset(new concrete_distribution<boost::math::fisher_f >(boost::math::fisher_f(arg1, arg2)));
break;
case 8:
this->reset(new concrete_distribution<boost::math::gamma_distribution<> >(boost::math::gamma_distribution<>(arg1, arg2)));
break;
case 9:
this->reset(new concrete_distribution<boost::math::geometric_distribution<> >(boost::math::geometric_distribution<>(arg1)));
break;
case 10:
this->reset(new concrete_distribution<boost::math::hypergeometric_distribution<> >(boost::math::hypergeometric_distribution<>(arg1, arg2, arg3)));
break;
case 11:
this->reset(new concrete_distribution<boost::math::inverse_chi_squared_distribution<> >(boost::math::inverse_chi_squared_distribution<>(arg1, arg2)));
break;
case 12:
this->reset(new concrete_distribution<boost::math::inverse_gamma_distribution<> >(boost::math::inverse_gamma_distribution<>(arg1, arg2)));
break;
case 13:
this->reset(new concrete_distribution<boost::math::inverse_gaussian_distribution<> >(boost::math::inverse_gaussian_distribution<>(arg1, arg2)));
break;
case 14:
this->reset(new concrete_distribution<boost::math::laplace_distribution<> >(boost::math::laplace_distribution<>(arg1, arg2)));
break;
case 15:
this->reset(new concrete_distribution<boost::math::logistic_distribution<> >(boost::math::logistic_distribution<>(arg1, arg2)));
break;
case 16:
this->reset(new concrete_distribution<boost::math::lognormal_distribution<> >(boost::math::lognormal_distribution<>(arg1, arg2)));
break;
case 17:
this->reset(new concrete_distribution<boost::math::negative_binomial_distribution<> >(boost::math::negative_binomial_distribution<>(arg1, arg2)));
break;
case 18:
this->reset(new concrete_distribution<boost::math::non_central_beta_distribution<> >(boost::math::non_central_beta_distribution<>(arg1, arg2, arg3)));
break;
case 19:
this->reset(new concrete_distribution<boost::math::non_central_chi_squared_distribution<> >(boost::math::non_central_chi_squared_distribution<>(arg1, arg2)));
break;
case 20:
this->reset(new concrete_distribution<boost::math::non_central_f_distribution<> >(boost::math::non_central_f_distribution<>(arg1, arg2, arg3)));
break;
case 21:
this->reset(new concrete_distribution<boost::math::non_central_t_distribution<> >(boost::math::non_central_t_distribution<>(arg1, arg2)));
break;
case 22:
this->reset(new concrete_distribution<boost::math::normal_distribution<> >(boost::math::normal_distribution<>(arg1, arg2)));
break;
case 23:
this->reset(new concrete_distribution<boost::math::pareto>(boost::math::pareto(arg1, arg2)));
break;
case 24:
this->reset(new concrete_distribution<boost::math::poisson>(boost::math::poisson(arg1)));
break;
case 25:
this->reset(new concrete_distribution<boost::math::rayleigh>(boost::math::rayleigh(arg1)));
break;
case 26:
this->reset(new concrete_distribution<boost::math::skew_normal>(boost::math::skew_normal(arg1, arg2, arg3)));
break;
case 27:
this->reset(new concrete_distribution<boost::math::students_t>(boost::math::students_t(arg1)));
break;
case 28:
this->reset(new concrete_distribution<boost::math::triangular>(boost::math::triangular(arg1, arg2, arg3)));
break;
case 29:
this->reset(new concrete_distribution<boost::math::uniform>(boost::math::uniform(arg1, arg2)));
break;
case 30:
this->reset(new concrete_distribution<boost::math::weibull>(boost::math::weibull(arg1, arg2)));
break;


default:
BOOST_ASSERT(0);
}
TRANSLATE_EXCEPTIONS_END
} 

struct distribution_info
{
const char* name; 
const char* first_param; 
const char* second_param; 
const char* third_param; 
double first_default; 
double second_default; 
double third_default; 
};

distribution_info distributions[] =
{ 
{ "Bernoulli", "Probability", "", "",0.5, 0, 0}, 
{ "Beta", "Alpha", "Beta", "", 1, 1, 0}, 
{ "Binomial", "Trials", "Probability of success", "", 1, 0.5, 0}, 
{ "Cauchy", "Location", "Scale", "", 0, 1, 0}, 
{ "Chi_squared", "Degrees of freedom", "", "", 1, 0, 0}, 
{ "Exponential", "lambda", "", "", 1, 0, 0}, 
{ "Extreme value", "Location", "Scale", "", 0, 1, 0}, 
{ "Fisher-F", "Degrees of freedom 1", "Degrees of freedom 2", "", 1, 1, 0}, 
{ "Gamma (Erlang)", "Shape", "Scale", "", 1, 1, 0}, 
{ "Geometric", "Probability", "", "", 1, 0, 0}, 
{ "HyperGeometric", "Defects", "Samples", "Objects", 1, 0, 1}, 
{ "InverseChiSq", "Degrees of Freedom", "Scale", "", 1, 1, 0}, 
{ "InverseGamma", "Shape", "Scale", "", 1, 1, 0}, 
{ "InverseGaussian", "Mean", "Scale", "", 1, 1, 0}, 
{ "Laplace", "Location", "Scale", "", 0, 1, 0}, 
{ "Logistic", "Location", "Scale", "", 0, 1, 0}, 
{ "LogNormal", "Location", "Scale", "", 0, 1, 0}, 
{ "Negative Binomial", "Successes", "Probability of success", "", 1, 0.5, 0}, 
{ "Noncentral Beta", "Shape alpha", "Shape beta", "Non-centrality", 1, 1, 0}, 
{ "Noncentral ChiSquare", "Degrees of Freedom", "Non-centrality", "", 1, 0, 0}, 
{ "Noncentral F", "Degrees of Freedom 1", "Degrees of Freedom 2", "Non-centrality", 1, 1, 0}, 
{ "Noncentral t", "Degrees of Freedom", "Non-centrality", "", 1, 0, 0}, 
{ "Normal (Gaussian)", "Mean", "Standard Deviation", "", 0, 1, 0}, 
{ "Pareto", "Location", "Shape","", 1, 1, 0}, 
{ "Poisson", "Mean", "", "", 1, 0, 0}, 
{ "Rayleigh", "Shape", "", "", 1, 0, 0}, 
{ "Skew Normal", "Location", "Shape", "Skew", 0, 1, 0}, 
{ "Student's t", "Degrees of Freedom", "", "", 1, 0, 0}, 
{ "Triangular", "Lower", "Mode", "Upper", -1, 0, +1 }, 
{ "Uniform", "Lower", "Upper", "", 0, 1, 0}, 
{ "Weibull", "Shape", "Scale", "", 1, 1, 0}, 
};

int any_distribution::size()
{
return sizeof(distributions) / sizeof(distributions[0]);
}

System::String^ any_distribution::distribution_name(int i)
{
if(i >= size())
return "";
return gcnew System::String(distributions[i].name);
}
System::String^ any_distribution::first_param_name(int i)
{
if(i >= size())
return "";
return gcnew System::String(distributions[i].first_param);
}
System::String^ any_distribution::second_param_name(int i)
{
if(i >= size())
return "";
return gcnew System::String(distributions[i].second_param);
}
System::String^ any_distribution::third_param_name(int i)
{
if(i >= size())
return "";
return gcnew System::String(distributions[i].third_param);
}
double any_distribution::first_param_default(int i)
{
if(i >= size())
return 0;
return distributions[i].first_default;
}
double any_distribution::second_param_default(int i)
{
if(i >= size())
return 0;
return distributions[i].second_default;
}
double any_distribution::third_param_default(int i)
{
if(i >= size())
return 0;
return distributions[i].third_default;
}

} 


