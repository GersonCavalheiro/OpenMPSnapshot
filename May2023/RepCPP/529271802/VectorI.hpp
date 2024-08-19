

template<typename T, typename Allocator>
inline void pFlow::Vector<T, Allocator>::fill( const T& val)
{
std::fill(this->begin(), this->end(), val);
}

#pragma hd_warning_disable 
template<typename T, typename Allocator>
inline void pFlow::Vector<T, Allocator>::operator +=( const T& val)
{
for( auto& v:(*this) )
{
v+= val;
}
}

#pragma hd_warning_disable 
template<typename T, typename Allocator>
inline void pFlow::Vector<T, Allocator>::operator -=( const T& val)
{
for( auto& v:(*this) )
{
v -= val;
}
}

#pragma hd_warning_disable 
template<typename T, typename Allocator>
inline void pFlow::Vector<T, Allocator>::operator *=( const T& val)
{
for( auto& v:(*this) )
{
v *= val;
}
}

#pragma hd_warning_disable 
template<typename T, typename Allocator>
inline void pFlow::Vector<T, Allocator>::operator /=( const T& val)
{
for( auto& v:(*this) )
{
v /= val;
}
}

#pragma hd_warning_disable 
template<typename T, typename Allocator>
inline void pFlow::Vector<T, Allocator>::operator +=( const Vector<T, Allocator>& v )
{

if(size() != v.size() )
{
fatalErrorInFunction <<
"the += operator is invalid, vector size of right side (" << v.size() <<
") is not equal to the left side (" << size() << ").\n";
fatalExit;
}

for(label i=0; i<v.size(); i++)
{
this->operator[](i) += v[i];
}
}

#pragma hd_warning_disable 
template<typename T, typename Allocator>
inline void pFlow::Vector<T, Allocator>::operator -=( const Vector<T, Allocator>& v )
{

if(size() != v.size() )
{
fatalErrorInFunction <<
"the -= operator is invalid, vector size of right side (" << v.size() <<
") is not equal to the left side (" << size() << ").\n";
fatalExit;
}

for(label i=0; i<v.size(); i++)
{
this->operator[](i) -= v[i];
}
}

#pragma hd_warning_disable 
template<typename T, typename Allocator>
inline void pFlow::Vector<T, Allocator>::operator *=( const Vector<T, Allocator>& v )
{

if(size() != v.size() )
{
fatalErrorInFunction <<
"the * operator is invalid, vector size of right side (" << v.size() <<
") is not equal to the left side (" << size() << ").\n";
fatalExit;
}

for(label i=0; i<v.size(); i++)
{
this->operator[](i) *= v[i];
}
}

#pragma hd_warning_disable 
template<typename T, typename Allocator>
inline void pFlow::Vector<T, Allocator>::operator /=( const Vector<T, Allocator>& v )
{

if(size() != v.size() )
{
fatalErrorInFunction <<
"the /= operator is invalid, vector size of right side (" << v.size() <<
") is not equal to the left side (" << size() << ").\n";
fatalExit;
}

for(label i=0; i<v.size(); i++)
{
this->operator[](i) /= v[i];
}
}

#pragma hd_warning_disable 
template<typename T, typename Allocator>
inline pFlow::Vector<T, Allocator> pFlow::Vector<T, Allocator>::operator -
(
)const
{
Vector<T, Allocator> res(*this);
for( auto& vi:res)
{
vi = -vi;
}
return res;
}

#pragma hd_warning_disable 
template<typename T, typename Allocator>
inline pFlow::Vector<T, Allocator> pFlow::operator+ (const Vector<T, Allocator>& op1, const T& op2 )
{
Vector<T, Allocator> res(op1);
res += op2;
return res;
}

#pragma hd_warning_disable 
template<typename T, typename Allocator>
inline pFlow::Vector<T, Allocator> pFlow::operator+ (const T& op1, const Vector<T, Allocator>& op2 )
{
Vector<T, Allocator> res(op2);
res += op1;

return res;

}

#pragma hd_warning_disable 
template<typename T, typename Allocator>
inline pFlow::Vector<T, Allocator> pFlow::operator+ (const Vector<T, Allocator>& op1, const Vector<T, Allocator>& op2 )
{
if( op1.size() != op2.size() )
{
fatalErrorInFunction <<
"the + operator is invalid, vector size of operand1 (" << op1.size() <<
") is not equal to vector size of operand2 (" << op1.size() << ").\n";
fatalExit;
}

Vector<T, Allocator> res(op1);
res += op2;
return res;
}


#pragma hd_warning_disable 
template<typename T, typename Allocator>
inline pFlow::Vector<T, Allocator> pFlow::operator - (const Vector<T, Allocator>& op1, const T& op2 )
{
Vector<T, Allocator> res(op1);
res -= op2;
return res;
}

#pragma hd_warning_disable 
template<typename T, typename Allocator>
inline pFlow::Vector<T, Allocator> pFlow::operator - (const T& op1, const Vector<T, Allocator>& op2 )
{
Vector<T, Allocator> res(op2.size(), op1);
res -= op2;
return res;

}

#pragma hd_warning_disable 
template<typename T, typename Allocator>
inline pFlow::Vector<T, Allocator> pFlow::operator - (const Vector<T, Allocator>& op1, const Vector<T, Allocator>& op2 )
{
if( op1.size() != op2.size() )
{
fatalErrorInFunction <<
"the - operator is invalid, vector size of operand1 (" << op1.size() <<
") is not equal to vector size of operand2 (" << op1.size() << ").\n";
fatalExit;
}

Vector<T, Allocator> res(op1);
res -= op2;
return res;
}

#pragma hd_warning_disable 
template<typename T, typename Allocator>
inline pFlow::Vector<T, Allocator> pFlow::operator* (const Vector<T, Allocator>& op1, const T& op2 )
{
Vector<T, Allocator> res(op1);
res *= op2;
return res;
}

#pragma hd_warning_disable 
template<typename T, typename Allocator>
inline pFlow::Vector<T, Allocator> pFlow::operator* (const T& op1, const Vector<T, Allocator>& op2 )
{
Vector<T, Allocator> res(op2);
res *= op1;

return res;

}

#pragma hd_warning_disable 
template<typename T, typename Allocator>
inline pFlow::Vector<T, Allocator> pFlow::operator* (const Vector<T, Allocator>& op1, const Vector<T, Allocator>& op2 )
{
if( op1.size() != op2.size() )
{
fatalErrorInFunction <<
"the * operator is invalid, vector size of operand1 (" << op1.size() <<
") is not equal to vector size of operand2 (" << op1.size() << ").\n";
fatalExit;
}

Vector<T, Allocator> res(op1);
res *= op2;
return res;
}

#pragma hd_warning_disable 
template<typename T, typename Allocator>
inline pFlow::Vector<T, Allocator> pFlow::operator / (const Vector<T, Allocator>& op1, const T& op2 )
{
Vector<T, Allocator> res(op1);
res /= op2;
return res;
}

#pragma hd_warning_disable 
template<typename T, typename Allocator>
inline pFlow::Vector<T, Allocator> pFlow::operator / (const T& op1, const Vector<T, Allocator>& op2 )
{
Vector<T, Allocator> res(op2.size(), op1);
res /= op2;
return res;

}

#pragma hd_warning_disable 
template<typename T, typename Allocator>
inline pFlow::Vector<T, Allocator> pFlow::operator / (const Vector<T, Allocator>& op1, const Vector<T, Allocator>& op2 )
{
if( op1.size() != op2.size() )
{
fatalErrorInFunction <<
"the / operator is invalid, vector size of operand1 (" << op1.size() <<
") is not equal to vector size of operand2 (" << op1.size() << ").\n";
fatalExit;
}

Vector<T, Allocator> res(op1);
res /= op2;
return res;
}