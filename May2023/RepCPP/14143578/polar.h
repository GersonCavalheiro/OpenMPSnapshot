#pragma once
#include "generator.h"

namespace dg {
namespace geo {


struct PolarGenerator : public aGenerator2d
{
private:
double r_min, r_max;

public:


PolarGenerator(double _r_min, double _r_max) : r_min(_r_min), r_max(_r_max) {}
virtual PolarGenerator* clone() const override final{return new PolarGenerator(*this); }

private:
virtual void do_generate(
const thrust::host_vector<double>& zeta1d,
const thrust::host_vector<double>& eta1d,
thrust::host_vector<double>& x,
thrust::host_vector<double>& y,
thrust::host_vector<double>& zetaX,
thrust::host_vector<double>& zetaY,
thrust::host_vector<double>& etaX,
thrust::host_vector<double>& etaY) const override final {

int size_r   = zeta1d.size();
int size_phi = eta1d.size();
int size     = size_r*size_phi;

x.resize(size); y.resize(size);
zetaX.resize(size); zetaY.resize(size);
etaX.resize(size); etaY.resize(size);

for(int j=0;j<size_phi;j++)
for(int i=0;i<size_r;i++) {
double r   = zeta1d[i] + r_min;
double phi = eta1d[j];

x[i+size_r*j] = r*cos(phi);
y[i+size_r*j] = r*sin(phi);

zetaX[i+size_r*j] = cos(phi);
zetaY[i+size_r*j] = sin(phi);
etaX[i+size_r*j] = -sin(phi)/r;
etaY[i+size_r*j] =  cos(phi)/r;
}

}

virtual double do_width() const override final{return r_max-r_min;}
virtual double do_height() const override final{return 2*M_PI;}
virtual bool do_isOrthogonal() const override final{return true;}
};



struct LogPolarGenerator : public aGenerator2d
{
private:
double r_min, r_max;

public:


LogPolarGenerator(double _r_min, double _r_max) : r_min(_r_min), r_max(_r_max) {}
virtual LogPolarGenerator* clone() const override final{return new LogPolarGenerator(*this); }

private:
virtual void do_generate(
const thrust::host_vector<double>& zeta1d,
const thrust::host_vector<double>& eta1d,
thrust::host_vector<double>& x,
thrust::host_vector<double>& y,
thrust::host_vector<double>& zetaX,
thrust::host_vector<double>& zetaY,
thrust::host_vector<double>& etaX,
thrust::host_vector<double>& etaY) const override final{

int size_r   = zeta1d.size();
int size_phi = eta1d.size();
int size     = size_r*size_phi;

x.resize(size); y.resize(size);
zetaX.resize(size); zetaY.resize(size);
etaX.resize(size); etaY.resize(size);

for(int j=0;j<size_phi;j++)
for(int i=0;i<size_r;i++) {
double l   = zeta1d[i] + log(r_min);
double phi = eta1d[j];

x[i+size_r*j] = exp(l)*cos(phi);
y[i+size_r*j] = exp(l)*sin(phi);

zetaX[i+size_r*j] = cos(phi)*exp(-l);
zetaY[i+size_r*j] = sin(phi)*exp(-l);
etaX[i+size_r*j] = -sin(phi)*exp(-l);
etaY[i+size_r*j] =  cos(phi)*exp(-l);
}

}

virtual double do_width() const override final{return log(r_max)-log(r_min);}
virtual double do_height() const override final{return 2*M_PI;}
virtual bool do_isOrthogonal() const override final{return true;}
};

}
}
