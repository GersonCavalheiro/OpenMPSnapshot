



#ifdef _MSC_VER
#  pragma once
#  pragma warning (disable : 4127)
#endif

using namespace System;

#define TRANSLATE_EXCEPTIONS_BEGIN try{

#define TRANSLATE_EXCEPTIONS_END \
}catch(const std::exception& e){  \
System::String^ s = gcnew System::String(e.what());\
InvalidOperationException^ se = gcnew InvalidOperationException(s);\
throw se;  \
}

namespace boost_math {

class any_imp
{
public:
virtual double mean()const = 0;
virtual double mode()const = 0;
virtual double median()const = 0;
virtual double variance()const = 0;
virtual double standard_deviation()const = 0;
virtual double skewness()const = 0;
virtual double kurtosis()const = 0;
virtual double kurtosis_excess()const = 0;
virtual double coefficient_of_variation()const = 0;
virtual double hazard(double x)const = 0;
virtual double chf(double x)const = 0;
virtual double cdf(double x)const = 0;
virtual double ccdf(double x)const = 0;
virtual double pdf(double x)const = 0;
virtual double quantile(double x)const = 0;
virtual double quantile_c(double x)const = 0;
virtual double lowest()const = 0;
virtual double uppermost()const = 0;
virtual double lower()const = 0;
virtual double upper()const = 0;
};

template <class Distribution>
class concrete_distribution : public any_imp
{
public:
concrete_distribution(const Distribution& d) : m_dist(d) {}
virtual double mean()const
{
return boost::math::mean(m_dist);
}
virtual double median()const
{
return boost::math::median(m_dist);
}
virtual double mode()const
{
return boost::math::mode(m_dist);
}
virtual double variance()const
{
return boost::math::variance(m_dist);
}
virtual double skewness()const
{
return boost::math::skewness(m_dist);
}
virtual double standard_deviation()const
{
return boost::math::standard_deviation(m_dist);
}
virtual double coefficient_of_variation()const
{
return boost::math::coefficient_of_variation(m_dist);
}
virtual double kurtosis()const
{
return boost::math::kurtosis(m_dist);
}
virtual double kurtosis_excess()const
{
return boost::math::kurtosis_excess(m_dist);
}
virtual double lowest()const
{
return boost::math::range(m_dist).first;
}
virtual double uppermost()const
{
return boost::math::range(m_dist).second;
}
virtual double lower()const
{
return boost::math::support(m_dist).first;
}
virtual double upper()const
{
return boost::math::support(m_dist).second;
}

virtual double hazard(double x)const
{
return boost::math::hazard(m_dist, x);
}
virtual double chf(double x)const
{
return boost::math::chf(m_dist, x);
}
virtual double cdf(double x)const
{
return boost::math::cdf(m_dist, x);
}
virtual double ccdf(double x)const
{
return boost::math::cdf(complement(m_dist, x));
}
virtual double pdf(double x)const
{
return boost::math::pdf(m_dist, x);
}
virtual double quantile(double x)const
{
return boost::math::quantile(m_dist, x);
}
virtual double quantile_c(double x)const
{
return boost::math::quantile(complement(m_dist, x));
}
private:
Distribution m_dist;
};

public ref class any_distribution
{
public:
any_distribution(int t, double arg1, double arg2, double arg3);
~any_distribution()
{
reset(0);
}
double mean()
{
TRANSLATE_EXCEPTIONS_BEGIN
return pimpl->mean();
TRANSLATE_EXCEPTIONS_END
}
double median()
{
TRANSLATE_EXCEPTIONS_BEGIN
return pimpl->median();
TRANSLATE_EXCEPTIONS_END
}
double mode()
{
TRANSLATE_EXCEPTIONS_BEGIN
return pimpl->mode();
TRANSLATE_EXCEPTIONS_END
}
double variance()
{
TRANSLATE_EXCEPTIONS_BEGIN
return pimpl->variance();
TRANSLATE_EXCEPTIONS_END
}
double standard_deviation()
{
TRANSLATE_EXCEPTIONS_BEGIN
return pimpl->standard_deviation();
TRANSLATE_EXCEPTIONS_END
}
double coefficient_of_variation()
{ 
TRANSLATE_EXCEPTIONS_BEGIN
return pimpl->coefficient_of_variation();
TRANSLATE_EXCEPTIONS_END
}
double skewness()
{
TRANSLATE_EXCEPTIONS_BEGIN
return pimpl->skewness();
TRANSLATE_EXCEPTIONS_END
}
double kurtosis()
{
TRANSLATE_EXCEPTIONS_BEGIN
return pimpl->kurtosis();
TRANSLATE_EXCEPTIONS_END
}
double kurtosis_excess()
{
TRANSLATE_EXCEPTIONS_BEGIN
return pimpl->kurtosis_excess();
TRANSLATE_EXCEPTIONS_END
}
double hazard(double x)
{
TRANSLATE_EXCEPTIONS_BEGIN
return pimpl->hazard(x);
TRANSLATE_EXCEPTIONS_END
}
double chf(double x)
{
TRANSLATE_EXCEPTIONS_BEGIN
return pimpl->chf(x);
TRANSLATE_EXCEPTIONS_END
}
double cdf(double x)
{
TRANSLATE_EXCEPTIONS_BEGIN
return pimpl->cdf(x);
TRANSLATE_EXCEPTIONS_END
}
double ccdf(double x)
{
TRANSLATE_EXCEPTIONS_BEGIN
return pimpl->ccdf(x);
TRANSLATE_EXCEPTIONS_END
}
double pdf(double x)
{
TRANSLATE_EXCEPTIONS_BEGIN
return pimpl->pdf(x);
TRANSLATE_EXCEPTIONS_END
}
double quantile(double x)
{
TRANSLATE_EXCEPTIONS_BEGIN
return pimpl->quantile(x);
TRANSLATE_EXCEPTIONS_END
}
double quantile_c(double x)
{
TRANSLATE_EXCEPTIONS_BEGIN
return pimpl->quantile_c(x);
TRANSLATE_EXCEPTIONS_END
}

double lowest()
{
TRANSLATE_EXCEPTIONS_BEGIN
return pimpl->lowest();
TRANSLATE_EXCEPTIONS_END
}

double uppermost()
{
TRANSLATE_EXCEPTIONS_BEGIN
return pimpl->uppermost();
TRANSLATE_EXCEPTIONS_END
}

double lower()
{
TRANSLATE_EXCEPTIONS_BEGIN
return pimpl->lower();
TRANSLATE_EXCEPTIONS_END
}
double upper()
{
TRANSLATE_EXCEPTIONS_BEGIN
return pimpl->upper();
TRANSLATE_EXCEPTIONS_END
}

static int size();
static System::String^ distribution_name(int i);
static System::String^ first_param_name(int i);
static System::String^ second_param_name(int i);
static System::String^ third_param_name(int i);
static double first_param_default(int i);
static double second_param_default(int i);
static double third_param_default(int i);

private:
any_distribution(const any_distribution^)
{ 
}
const any_distribution^ operator=(const any_distribution^ d)
{ 
return d;
}
void reset(any_imp* p)
{
if(pimpl)
{ 
delete pimpl;
}
pimpl = p;
}
any_imp* pimpl;
};
}
