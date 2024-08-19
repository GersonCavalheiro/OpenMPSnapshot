#pragma once

#include "operator.h"
#include "dg/functors.h"
#include "dg/blas1.h"
#include "tensor.h"



namespace dg
{

template<class value_type>
struct TensorMultiply2d{
DG_DEVICE
void operator() (
value_type lambda,
value_type t00, value_type t01,
value_type t10, value_type t11,
value_type in0, value_type in1,
value_type mu,
value_type& out0, value_type& out1) const
{
value_type tmp0 = DG_FMA(t00,in0 , t01*in1);
value_type tmp1 = DG_FMA(t10,in0 , t11*in1);
value_type temp = out1*mu;
out1 = DG_FMA( lambda, tmp1, temp);
temp = out0*mu;
out0 = DG_FMA( lambda, tmp0, temp);
}
};
template<class value_type>
struct TensorMultiply3d{
DG_DEVICE
void operator() ( value_type lambda,
value_type t00, value_type t01, value_type t02,
value_type t10, value_type t11, value_type t12,
value_type t20, value_type t21, value_type t22,
value_type in0, value_type in1, value_type in2,
value_type mu,
value_type& out0, value_type& out1, value_type& out2) const
{
value_type tmp0 = DG_FMA( t00,in0 , (DG_FMA( t01,in1 , t02*in2)));
value_type tmp1 = DG_FMA( t10,in0 , (DG_FMA( t11,in1 , t12*in2)));
value_type tmp2 = DG_FMA( t20,in0 , (DG_FMA( t21,in1 , t22*in2)));
value_type temp = out2*mu;
out2 = DG_FMA( lambda, tmp2, temp);
temp = out1*mu;
out1 = DG_FMA( lambda, tmp1, temp);
temp = out0*mu;
out0 = DG_FMA( lambda, tmp0, temp);
}
};
template<class value_type>
struct InverseTensorMultiply2d{
DG_DEVICE
void operator() (  value_type lambda,
value_type t00, value_type t01,
value_type t10, value_type t11,
value_type in0, value_type in1,
value_type mu, value_type& out0, value_type& out1) const
{
value_type dett = DG_FMA( t00,t11 , (-t10*t01));
value_type tmp0 = DG_FMA( in0,t11 , (-in1*t01));
value_type tmp1 = DG_FMA( t00,in1 , (-t10*in0));
value_type temp = out1*mu;
out1 = DG_FMA( lambda, tmp1/dett, temp);
temp = out0*mu;
out0 = DG_FMA( lambda, tmp0/dett, temp);
}
};
template<class value_type>
struct InverseTensorMultiply3d{
DG_DEVICE
void operator() ( value_type lambda,
value_type t00, value_type t01, value_type t02,
value_type t10, value_type t11, value_type t12,
value_type t20, value_type t21, value_type t22,
value_type in0, value_type in1, value_type in2,
value_type mu,
value_type& out0, value_type& out1, value_type& out2) const
{
value_type dett = det( t00,t01,t02, t10,t11,t12, t20,t21,t22);

value_type tmp0 = det( in0,t01,t02, in1,t11,t12, in2,t21,t22);
value_type tmp1 = det( t00,in0,t02, t10,in1,t12, t20,in2,t22);
value_type tmp2 = det( t00,t01,in0, t10,t11,in1, t20,t21,in2);
value_type temp = out2*mu;
out2 = DG_FMA( lambda, tmp2/dett, temp);
temp = out1*mu;
out1 = DG_FMA( lambda, tmp1/dett, temp);
temp = out0*mu;
out0 = DG_FMA( lambda, tmp0/dett, temp);
}
private:
DG_DEVICE
value_type det( value_type t00, value_type t01, value_type t02,
value_type t10, value_type t11, value_type t12,
value_type t20, value_type t21, value_type t22)const
{
return t00*DG_FMA(t11, t22, (-t12*t21))
-t01*DG_FMA(t10, t22, (-t20*t12))
+t02*DG_FMA(t10, t21, (-t20*t11));
}
};


template<class value_type>
struct TensorDot2d{
DG_DEVICE
value_type operator() (
value_type lambda,
value_type v0,  value_type v1,
value_type t00, value_type t01,
value_type t10, value_type t11,
value_type mu,
value_type w0, value_type w1
) const
{
value_type tmp0 = DG_FMA(t00,w0 , t01*w1);
value_type tmp1 = DG_FMA(t10,w0 , t11*w1);
return lambda*mu*DG_FMA(v0,tmp0  , v1*tmp1);
}
};
template<class value_type>
struct TensorDot3d{
DG_DEVICE
value_type operator() (
value_type lambda,
value_type v0,  value_type v1,  value_type v2,
value_type t00, value_type t01, value_type t02,
value_type t10, value_type t11, value_type t12,
value_type t20, value_type t21, value_type t22,
value_type mu,
value_type w0, value_type w1, value_type w2) const
{
value_type tmp0 = DG_FMA( t00,w0 , (DG_FMA( t01,w1 , t02*w2)));
value_type tmp1 = DG_FMA( t10,w0 , (DG_FMA( t11,w1 , t12*w2)));
value_type tmp2 = DG_FMA( t20,w0 , (DG_FMA( t21,w1 , t22*w2)));
return lambda*mu*DG_FMA(v0,tmp0 , DG_FMA(v1,tmp1 , v2*tmp2));
}
};

template<class value_type>
struct TensorDeterminant2d
{
DG_DEVICE
value_type operator() ( value_type t00, value_type t01,
value_type t10, value_type t11) const
{
return DG_FMA( t00,t11 , (-t10*t01));
}
};
template<class value_type>
struct TensorDeterminant3d
{
DG_DEVICE
value_type operator() ( value_type t00, value_type t01, value_type t02,
value_type t10, value_type t11, value_type t12,
value_type t20, value_type t21, value_type t22) const
{
return t00*m_t(t11, t12, t21, t22)
-t01*m_t(t10, t12, t20, t22)
+t02*m_t(t10, t11, t20, t21);
}
private:
TensorDeterminant2d<value_type> m_t;
};



namespace tensor
{



template<class ContainerType0, class ContainerType1>
void scal( SparseTensor<ContainerType0>& t, const ContainerType1& mu)
{
unsigned size=t.values().size();
for( unsigned i=0; i<size; i++)
dg::blas1::pointwiseDot( mu, t.values()[i], t.values()[i]);
}


template<class ContainerTypeL, class ContainerType0, class ContainerType1, class ContainerType2, class ContainerTypeM, class ContainerType3, class ContainerType4>
void multiply2d( const ContainerTypeL& lambda, const SparseTensor<ContainerType0>& t, const ContainerType1& in0, const ContainerType2& in1, const ContainerTypeM& mu, ContainerType3& out0, ContainerType4& out1)
{
dg::blas1::subroutine( dg::TensorMultiply2d<get_value_type<ContainerType0>>(),
lambda,      t.value(0,0), t.value(0,1),
t.value(1,0), t.value(1,1),
in0,  in1,
mu,          out0, out1);
}


template<class ContainerTypeL, class ContainerType0, class ContainerType1, class ContainerType2, class ContainerType3, class ContainerTypeM, class ContainerType4, class ContainerType5, class ContainerType6>
void multiply3d( const ContainerTypeL& lambda, const SparseTensor<ContainerType0>& t, const ContainerType1& in0, const ContainerType2& in1, const ContainerType3& in2, const ContainerTypeM& mu, ContainerType4& out0, ContainerType5& out1, ContainerType6& out2)
{
dg::blas1::subroutine( dg::TensorMultiply3d<get_value_type<ContainerType0>>(),
lambda,      t.value(0,0), t.value(0,1), t.value(0,2),
t.value(1,0), t.value(1,1), t.value(1,2),
t.value(2,0), t.value(2,1), t.value(2,2),
in0, in1, in2,
mu,          out0, out1, out2);
}


template<class ContainerTypeL, class ContainerType0, class ContainerType1, class ContainerType2, class ContainerTypeM, class ContainerType3, class ContainerType4>
void inv_multiply2d( const ContainerTypeL& lambda, const SparseTensor<ContainerType0>& t, const ContainerType1& in0, const ContainerType2& in1, const ContainerTypeM& mu, ContainerType3& out0, ContainerType4& out1)
{
dg::blas1::subroutine( dg::InverseTensorMultiply2d<get_value_type<ContainerType0>>(),
lambda,    t.value(0,0), t.value(0,1),
t.value(1,0), t.value(1,1),
in0,  in1,
mu,        out0, out1);
}


template<class ContainerTypeL, class ContainerType0, class ContainerType1, class ContainerType2, class ContainerType3, class ContainerTypeM, class ContainerType4, class ContainerType5, class ContainerType6>
void inv_multiply3d( const ContainerTypeL& lambda, const SparseTensor<ContainerType0>& t, const ContainerType1& in0, const ContainerType2& in1, const ContainerType3& in2, const ContainerTypeM& mu, ContainerType4& out0, ContainerType5& out1, ContainerType6& out2)
{
dg::blas1::subroutine( dg::InverseTensorMultiply3d<get_value_type<ContainerType0>>(),
lambda,       t.value(0,0), t.value(0,1), t.value(0,2),
t.value(1,0), t.value(1,1), t.value(1,2),
t.value(2,0), t.value(2,1), t.value(2,2),
in0, in1, in2,
mu,           out0, out1, out2);
}


template<class ContainerType>
ContainerType determinant2d( const SparseTensor<ContainerType>& t)
{
ContainerType det = t.value(0,0);
dg::blas1::evaluate( det, dg::equals(), dg::TensorDeterminant2d<get_value_type<ContainerType>>(),
t.value(0,0), t.value(0,1),
t.value(1,0), t.value(1,1));
return det;
}


template<class ContainerType>
ContainerType determinant( const SparseTensor<ContainerType>& t)
{
ContainerType det = t.value(0,0);
dg::blas1::evaluate( det, dg::equals(), dg::TensorDeterminant3d<get_value_type<ContainerType>>(),
t.value(0,0), t.value(0,1), t.value(0,2),
t.value(1,0), t.value(1,1), t.value(1,2),
t.value(2,0), t.value(2,1), t.value(2,2));
return det;
}


template<class ContainerType>
ContainerType volume2d( const SparseTensor<ContainerType>& t)
{
ContainerType vol=determinant2d(t);
dg::blas1::transform(vol, vol, dg::InvSqrt<get_value_type<ContainerType>>());
return vol;
}


template<class ContainerType>
ContainerType volume( const SparseTensor<ContainerType>& t)
{
ContainerType vol=determinant(t);
dg::blas1::transform(vol, vol, dg::InvSqrt<get_value_type<ContainerType>>());
return vol;
}


template<class ContainerType0, class ContainerType1, class ContainerType2, class ContainerType3, class ContainerType4>
void multiply2d( const SparseTensor<ContainerType0>& t, const ContainerType1& in0, const ContainerType2& in1, ContainerType3& out0, ContainerType4& out1)
{
multiply2d( 1, t, in0, in1, 0., out0, out1);
}


template<class ContainerType0, class ContainerType1, class ContainerType2, class ContainerType3, class ContainerType4, class ContainerType5, class ContainerType6>
void multiply3d( const SparseTensor<ContainerType0>& t, const ContainerType1& in0, const ContainerType2& in1, const ContainerType3& in2, ContainerType4& out0, ContainerType5& out1, ContainerType6& out2)
{
multiply3d( 1., t, in0, in1, in2, 0., out0, out1, out2);
}


template<class ContainerType0, class ContainerType1, class ContainerType2, class ContainerType3, class ContainerType4>
void inv_multiply2d( const SparseTensor<ContainerType0>& t, const ContainerType1& in0, const ContainerType2& in1, ContainerType3& out0, ContainerType4& out1)
{
inv_multiply2d( 1., t, in0, in1, out0, out1);
}


template<class ContainerType0, class ContainerType1, class ContainerType2, class ContainerType3, class ContainerType4, class ContainerType5, class ContainerType6>
void inv_multiply3d( const SparseTensor<ContainerType0>& t, const ContainerType1& in0, const ContainerType2& in1, const ContainerType3& in2, ContainerType4& out0, ContainerType5& out1, ContainerType6& out2)
{
inv_multiply3d( 1., t, in0, in1, in2, 0., out0, out1, out2);
}


template<class ContainerTypeL, class ContainerType0, class ContainerType1, class ContainerType2, class ContainerType3, class ContainerTypeM, class ContainerType4, class ContainerType5>
void scalar_product2d(
get_value_type<ContainerType0> alpha,
const ContainerTypeL& lambda,
const ContainerType0& v0,
const ContainerType1& v1,
const SparseTensor<ContainerType2>& t,
const ContainerTypeM& mu,
const ContainerType3& w0,
const ContainerType4& w1,
get_value_type<ContainerType0> beta,
ContainerType5& y)
{
dg::blas1::evaluate( y,
dg::Axpby<get_value_type<ContainerType0>>( alpha, beta),
dg::TensorDot2d<get_value_type<ContainerType0>>(),
lambda,
v0, v1,
t.value(0,0), t.value(0,1),
t.value(1,0), t.value(1,1),
mu,
w0, w1);
}


template<class ContainerTypeL, class ContainerType0, class ContainerType1, class ContainerType2, class ContainerType3, class ContainerTypeM, class ContainerType4, class ContainerType5, class ContainerType6, class ContainerType7>
void scalar_product3d(
get_value_type<ContainerType0> alpha,
const ContainerTypeL& lambda,
const ContainerType0& v0,
const ContainerType1& v1,
const ContainerType2& v2,
const SparseTensor<ContainerType3>& t,
const ContainerTypeM& mu,
const ContainerType4& w0,
const ContainerType5& w1,
const ContainerType6& w2,
get_value_type<ContainerType0> beta,
ContainerType7& y)
{
dg::blas1::evaluate( y,
dg::Axpby<get_value_type<ContainerType0>>( alpha, beta),
dg::TensorDot3d<get_value_type<ContainerType0>>(),
lambda,
v0, v1, v2,
t.value(0,0), t.value(0,1), t.value(0,2),
t.value(1,0), t.value(1,1), t.value(1,2),
t.value(2,0), t.value(2,1), t.value(2,2),
mu,
w0, w1, w2);
}

}
}
