#pragma once

#include "accelerator/accelerator.hpp"


namespace paracabs
{
namespace datatypes
{
template <typename type>
class Vector3D
{
public:
type data[3];   

accel inline Vector3D () {}

accel inline Vector3D (const type x, const type y, const type z)
{
data[0] = x;
data[1] = y;
data[2] = z;
}

accel inline Vector3D (const type v)
{
*this = Vector3D (v, v, v);
}







accel inline void operator= (const type v)
{
*this = Vector3D (v);
}

accel inline Vector3D operator+ (const Vector3D& vec) const
{
const type x = data[0] + vec.data[0];
const type y = data[1] + vec.data[1];
const type z = data[2] + vec.data[2];

return Vector3D (x, y, z);
}

accel inline Vector3D operator- (const Vector3D& vec) const
{
const type x = data[0] - vec.data[0];
const type y = data[1] - vec.data[1];
const type z = data[2] - vec.data[2];

return Vector3D (x, y, z);
}

accel inline void operator+= (const Vector3D& vec)
{
data[0] += vec.data[0];
data[1] += vec.data[1];
data[2] += vec.data[2];
}

accel inline void operator-= (const Vector3D& vec)
{
data[0] -= vec.data[0];
data[1] -= vec.data[1];
data[2] -= vec.data[2];
}

accel inline type dot (const Vector3D& vec) const
{
return   data[0] * vec.data[0]
+ data[1] * vec.data[1]
+ data[2] * vec.data[2];
}

accel inline type squaredNorm () const
{
return dot (*this);
}


accel inline type x () const {return data[0];}
accel inline type y () const {return data[1];}
accel inline type z () const {return data[2];}

accel inline void print () const {printf ("%le, %le, %le\n", x(), y(), z());}
};


template <typename type>
accel inline Vector3D<type> operator* (const Vector3D<type>& vec, const type& scalar)
{
const type x = vec.data[0] * scalar;
const type y = vec.data[1] * scalar;
const type z = vec.data[2] * scalar;

return Vector3D<type>(x, y, z);
}

template <typename type>
accel inline Vector3D<type> operator* (const type& scalar, const Vector3D<type>& vec)
{
const type x = vec.data[0] * scalar;
const type y = vec.data[1] * scalar;
const type z = vec.data[2] * scalar;

return Vector3D<type>(x, y, z);
}

}
}
