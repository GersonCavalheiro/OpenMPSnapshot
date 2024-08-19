#pragma once

#include "fluxfunctions.h"


namespace dg
{
namespace geo
{

namespace detail{

struct LaplaceAdaptPsi: public aCylindricalFunctor<LaplaceAdaptPsi>
{
LaplaceAdaptPsi( const CylindricalFunctorsLvl2& psi, const CylindricalFunctorsLvl1& chi) : psi_(psi), chi_(chi){}
double do_compute(double x, double y)const
{
return  psi_.dfx()(x,y)*chi_.dfx()(x,y) +
psi_.dfy()(x,y)*chi_.dfy()(x,y) +
chi_.f()(x,y)*( psi_.dfxx()(x,y) + psi_.dfyy()(x,y));
}
private:
CylindricalFunctorsLvl2 psi_;
CylindricalFunctorsLvl1 chi_;
};

struct LaplaceChiPsi: public aCylindricalFunctor<LaplaceChiPsi>
{
LaplaceChiPsi( const CylindricalFunctorsLvl2& psi, const CylindricalSymmTensorLvl1& chi):
psi_(psi), chi_(chi){}
double do_compute(double x, double y)const
{
return psi_.dfxx()(x,y)*chi_.xx()(x,y)+2.*psi_.dfxy()(x,y)*chi_.xy()(x,y)+psi_.dfyy()(x,y)*chi_.yy()(x,y)
+ chi_.divX()(x,y)*psi_.dfx()(x,y) + chi_.divY()(x,y)*psi_.dfy()(x,y);
}
private:

CylindricalFunctorsLvl2 psi_;
CylindricalSymmTensorLvl1 chi_;
};

struct LaplacePsi: public aCylindricalFunctor<LaplacePsi>
{
LaplacePsi( const CylindricalFunctorsLvl2& psi): psi_(psi){}
double do_compute(double x, double y)const{return psi_.dfxx()(x,y)+psi_.dfyy()(x,y);}
private:
CylindricalFunctorsLvl2 psi_;
};

}



struct NablaPsiInv: public aCylindricalFunctor<NablaPsiInv>
{

NablaPsiInv( const CylindricalFunctorsLvl1& psi): psi_(psi){}
double do_compute(double x, double y)const
{
double psiX = psi_.dfx()(x,y), psiY = psi_.dfy()(x,y);
return 1./sqrt(psiX*psiX+psiY*psiY);
}
private:
CylindricalFunctorsLvl1 psi_;
};


struct NablaPsiInvX: public aCylindricalFunctor<NablaPsiInvX>
{
NablaPsiInvX( const CylindricalFunctorsLvl2& psi):psi_(psi) {}
double do_compute(double x, double y)const
{
double psiX = psi_.dfx()(x,y), psiY = psi_.dfy()(x,y);
double psiXX = psi_.dfxx()(x,y), psiXY = psi_.dfxy()(x,y);
double psip = sqrt( psiX*psiX+psiY*psiY);
return -(psiX*psiXX+psiY*psiXY)/psip/psip/psip;
}
private:

CylindricalFunctorsLvl2 psi_;
};


struct NablaPsiInvY: public aCylindricalFunctor<NablaPsiInvY>
{
NablaPsiInvY( const CylindricalFunctorsLvl2& psi):psi_(psi) {}
double do_compute(double x, double y)const
{
double psiX = psi_.dfx()(x,y), psiY = psi_.dfy()(x,y);
double psiYY = psi_.dfyy()(x,y), psiXY = psi_.dfxy()(x,y);
double psip = sqrt( psiX*psiX+psiY*psiY);
return -(psiX*psiXY+psiY*psiYY)/psip/psip/psip;
}
private:

CylindricalFunctorsLvl2 psi_;
};


static inline CylindricalFunctorsLvl1 make_NablaPsiInvCollective( const CylindricalFunctorsLvl2& psi)
{
return CylindricalFunctorsLvl1( NablaPsiInv(psi), NablaPsiInvX(psi),
NablaPsiInvY( psi));
}




struct Liseikin_XX: public aCylindricalFunctor<Liseikin_XX>
{
Liseikin_XX(const CylindricalFunctorsLvl1& psi, double k, double eps):k_(k), eps_(eps), psi_(psi){}
double do_compute(double x, double y)const
{
double psiX = psi_.dfx()(x,y), psiY = psi_.dfy()(x,y), k2 = k_*k_;
double psip2 = psiX*psiX+psiY*psiY;
double sqrtG = sqrt((eps_+psip2)*(eps_+k2*psip2));
return (psiY*psiY+k2*psiX*psiX + eps_)/sqrtG;
}
private:

double k_, eps_;
CylindricalFunctorsLvl1 psi_;
};


struct Liseikin_XY: public aCylindricalFunctor<Liseikin_XY>
{
Liseikin_XY(const CylindricalFunctorsLvl1& psi, double k, double eps):k_(k), eps_(eps), psi_(psi){}
double do_compute(double x, double y)const
{
double psiX = psi_.dfx()(x,y), psiY = psi_.dfy()(x,y), k2 = k_*k_;
double psip2 = psiX*psiX+psiY*psiY;
double sqrtG = sqrt((eps_+psip2)*(eps_+k2*psip2));
return (-psiX*psiY+k2*psiX*psiY)/sqrtG;
}
private:

double k_, eps_;
CylindricalFunctorsLvl1 psi_;
};


struct Liseikin_YY: public aCylindricalFunctor<Liseikin_YY>
{
Liseikin_YY(const CylindricalFunctorsLvl1& psi, double k, double eps):k_(k), eps_(eps), psi_(psi){}
double do_compute(double x, double y)const
{
double psiX = psi_.dfx()(x,y), psiY = psi_.dfy()(x,y), k2 = k_*k_;
double psip2 = psiX*psiX+psiY*psiY;
double sqrtG = sqrt((eps_+psip2)*(eps_+k2*psip2));
return (eps_+psiX*psiX+k2*psiY*psiY)/sqrtG;
}
private:

double k_, eps_;
CylindricalFunctorsLvl1 psi_;
};


struct DivLiseikinX: public aCylindricalFunctor<DivLiseikinX>
{
DivLiseikinX(const CylindricalFunctorsLvl2& psi, double k, double eps): k_(k), eps_(eps), psi_(psi){}
double do_compute(double x, double y)const
{
double psiX = psi_.dfx()(x,y), psiY = psi_.dfy()(x,y), k2 = k_*k_;
double psiXX = psi_.dfxx()(x,y), psiXY = psi_.dfxy()(x,y), psiYY=psi_.dfyy()(x,y);
double psiY2 = psiY*psiY, psiY3=psiY*psiY2, psiY4=psiY2*psiY2, psiY5=psiY4*psiY;
double psiX2 = psiX*psiX, psiX4=psiX2*psiX2;
double psip2 = psiX*psiX+psiY*psiY;
double sqrtG = sqrt((eps_+psip2)*(eps_+k2*psip2));
return (k2-1.)*(k2*psiY5*psiXY-k2*psiX*psiY4*(psiYY-2.*psiXX) +
psiX*(eps_+2.*eps_*k2+2.*k2*psiX2)*psiY2*psiXX +
psiX*(eps_+k2*psiX2)*((eps_+psiX2)*psiYY+eps_*psiXX)+
psiY*((eps_*eps_-k2*psiX4)*psiXY-(eps_+2*psiX2)*(eps_+k2*psiX2)*psiXY) +
psiY3*(eps_*(1.+k2)*psiXY-(eps_+2.*k2*psiX2)*psiXY))/sqrtG/sqrtG/sqrtG;
}
private:

double k_, eps_;
CylindricalFunctorsLvl2 psi_;
};


struct DivLiseikinY : public aCylindricalFunctor<DivLiseikinY>
{
DivLiseikinY(const CylindricalFunctorsLvl2& psi, double k, double eps):k_(k), eps_(eps), psi_(psi){}
double do_compute(double x, double y)const
{
double psiX = psi_.dfx()(x,y), psiY = psi_.dfy()(x,y), k2 = k_*k_;
double psiXX = psi_.dfxx()(x,y), psiXY = psi_.dfxy()(x,y), psiYY=psi_.dfyy()(x,y);
double psiX2 = psiX*psiX, psiX3=psiX*psiX2, psiX4=psiX2*psiX2, psiX5=psiX4*psiX;
double psiY2 = psiY*psiY, psiY4 = psiY2*psiY2;
double psip2 = psiX*psiX+psiY*psiY;
double sqrtG = sqrt((eps_+psip2)*(eps_+k2*psip2));
return (k2-1.)*(psiX2*psiY*(eps_+2.*eps_*k2+2.*k2*psiY2)*psiYY +
k2*psiX4*psiY*(2.*psiYY-psiXX)+psiY*(eps_+k2*psiY2)
*(eps_*psiYY+(eps_+psiY2)*psiXX)+k2*psiX5*psiXY+
psiX3*(-(eps_+2.*k2*psiY2)*psiXY+eps_*(1.+k2)*psiXY) +
psiX*(-(eps_+2.*psiY2)*(eps_+k2*psiY2)*psiXY + (eps_*eps_-k2*psiY4)*psiXY))/sqrtG/sqrtG/sqrtG;
}
private:

double k_, eps_;
CylindricalFunctorsLvl2 psi_;
};

static inline CylindricalSymmTensorLvl1 make_LiseikinCollective( const CylindricalFunctorsLvl2& psi, double k, double eps)
{
return CylindricalSymmTensorLvl1( Liseikin_XX(psi,k,eps),
Liseikin_XY(psi,k,eps), Liseikin_YY(psi,k,eps),
DivLiseikinX(psi,k,eps), DivLiseikinY(psi,k,eps));
}

}
}
