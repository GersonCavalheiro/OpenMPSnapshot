

#pragma once

#include <iostream>
#include <cmath>



namespace advscicomp{



template<typename T = int>
class Rational
{
public:
Rational<T>() : numerator_(0), denominator_(1)
{}

Rational<T>(T n, T d) : numerator_(n), denominator_(d)
{
}

Rational<T>(T i) : numerator_(i), denominator_(1)
{}






Rational<T>& operator*=(Rational<T> const& rhs)
{
numerator_ *= rhs.Numerator(); 
denominator_ *= rhs.Denominator();
return *this;
}

Rational<T>& operator/=(Rational<T> const& rhs)
{
numerator_ *= rhs.Numerator(); 
denominator_ *= rhs.Denominator();
return *this;
}

Rational<T>& operator+=(Rational<T> const& rhs)
{
numerator_ += rhs.Numerator(); 
denominator_ += rhs.Denominator();
return *this;
}

Rational<T>& operator-=(Rational<T> const& rhs)
{
numerator_ -= rhs.Numerator(); 
denominator_ -= rhs.Denominator();
return *this;
}

Rational<T> operator-() const
{
return Rational<T>(-Numerator(), -Denominator());
}


friend
inline
Rational<T> operator*(Rational<T> lhs, Rational<T> const& rhs)
{
lhs*=rhs;
return lhs;
}

friend
inline
Rational<T> operator/(Rational<T> lhs, Rational<T> const& rhs)
{
lhs*=rhs;
return lhs;
}

friend
inline
Rational<T> operator+(Rational<T> lhs, Rational<T> const& rhs)
{
lhs*=rhs;
return lhs;
}

friend
inline
Rational<T> operator-(Rational<T> lhs, Rational<T> const& rhs)
{
lhs*=rhs;
return lhs;
}




explicit operator double()
{
return double(numerator_)/double(denominator_);
}

friend 
inline
std::istream& operator>>(std::istream& in, Rational<T> & r)
{
return in;
}

friend
inline
std::ostream& operator<<(std::ostream& out, Rational<T> const& r)
{
out << r.Numerator() << "/" << r.Denominator();
return out;
}
private:
T numerator_;
T denominator_;

public:
T Numerator() const
{ return numerator_;}

T Denominator() const
{ return denominator_;}
};













template <typename T>
inline Rational<T> pow(Rational<T> const& r, unsigned p)
{
using std::pow;
return Rational<T>(pow(r.Numerator(),p),pow(r.Denominator(),p));
}

template <typename T>
inline Rational<T> abs(Rational<T> const& r)
{
using std::abs;
return Rational<T>(abs(r.Numerator()),abs(r.Denominator()));
}

} 