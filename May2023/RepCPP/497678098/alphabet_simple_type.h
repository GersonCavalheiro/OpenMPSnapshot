
#ifndef SEQAN_INCLUDE_SEQAN_BASIC_ALPHABET_SIMPLE_H_
#define SEQAN_INCLUDE_SEQAN_BASIC_ALPHABET_SIMPLE_H_

namespace seqan {










#pragma pack(push,1)
template <typename TValue, typename TSpec>
class SimpleType
{
public:

TValue value{0};


SimpleType() = default;
SimpleType(SimpleType const &) = default;
SimpleType(SimpleType &&) = default;
~SimpleType() = default;

template <typename T>
SimpleType(T const & other)
{
assign(*this, other);
}

SimpleType & operator=(SimpleType const &) = default;
SimpleType & operator=(SimpleType &&) = default;

template <typename T>
inline SimpleType &
operator=(T const & other)
{
assign(*this, other);
return *this;
}




operator int64_t() const
{
int64_t c;
assign(c, *this);
return c;
}


operator uint64_t() const
{
uint64_t c;
assign(c, *this);
return c;
}


operator int() const
{
int c;
assign(c, *this);
return c;
}


operator unsigned int() const
{
unsigned int c;
assign(c, *this);
return c;
}


operator short() const
{
short c;
assign(c, *this);
return c;
}


operator unsigned short() const
{
unsigned short c;
assign(c, *this);
return c;
}


operator char() const
{
char c;
assign(c, *this);
return c;
}


operator signed char() const
{
signed char c;
assign(c, *this);
return c;
}


operator unsigned char() const
{
unsigned char c;
assign(c, *this);
return c;
}
};
#pragma pack(pop)

} 

namespace std
{

template <typename TValue, typename TSpec>
class numeric_limits<seqan::SimpleType<TValue, TSpec> >
{
public:
static constexpr bool is_specialized    = true;
static constexpr bool is_signed         = false;
static constexpr bool is_integer        = false;
static constexpr bool is_exact          = true;
static constexpr bool has_infinity      = false;
static constexpr bool has_quiet_NaN     = false;
static constexpr bool has_signaling_NaN = false;
static constexpr float_denorm_style has_denorm = denorm_absent;
static constexpr bool has_denorm_loss   = false;
static constexpr float_round_style round_style = round_toward_zero;
static constexpr bool is_iec559         = false;
static constexpr bool is_bounded        = true;
static constexpr bool is_modulu         = false;
static constexpr int  digits            = seqan::BitsPerValue<seqan::SimpleType<TValue, TSpec>>::VALUE;
static constexpr int  digits10          = digits - 1;
static constexpr int  max_digits10      = 0;
static constexpr int  radix             = 2;
static constexpr int  min_exponent      = 0;
static constexpr int  min_exponent10    = 0;
static constexpr int  max_exponent      = 0;
static constexpr int  max_exponent10    = 0;
static constexpr bool traps             = false;
static constexpr bool tinyness_before   = false;

static constexpr seqan::SimpleType<TValue, TSpec> min()
{
return seqan::SimpleType<TValue, TSpec>(0);
}

static constexpr seqan::SimpleType<TValue, TSpec> max()
{
return seqan::SimpleType<TValue, TSpec>(((TValue)seqan::ValueSize<seqan::SimpleType<TValue, TSpec> >::VALUE - 1));
}

static constexpr seqan::SimpleType<TValue, TSpec> lowest()
{
return seqan::SimpleType<TValue, TSpec>(0);
}

static constexpr seqan::SimpleType<TValue, TSpec> infinity()
{
return seqan::SimpleType<TValue, TSpec>(((TValue)seqan::ValueSize<seqan::SimpleType<TValue, TSpec> >::VALUE - 1));
}
};

} 

namespace seqan
{


template <typename TValue, typename TSpec>
struct IsSimple<SimpleType<TValue, TSpec> >
{
typedef True Type;
};


template <typename TValue, typename TSpec, typename TSource>
struct Is< Convertible<SimpleType<TValue, TSpec>, TSource> > :
Is< FundamentalConcept<TSource> > {};

template <typename TTarget, typename TValue, typename TSpec>
struct Is< Convertible<TTarget, SimpleType<TValue, TSpec> > > :
Is< FundamentalConcept<TTarget> > {};



template <typename TValue, typename TSpec>
struct Value<SimpleType<TValue, TSpec> >
{
typedef TValue Type;
};

template <typename TValue, typename TSpec>
struct Value<SimpleType<TValue, TSpec> const>
{
typedef TValue const Type;
};


template <typename TValue, typename TSpec>
struct MinValue<SimpleType<TValue, TSpec> >
{
static const SimpleType<TValue, TSpec> VALUE;
};

template <typename TValue, typename TSpec>
const SimpleType<TValue, TSpec> MinValue<SimpleType<TValue, TSpec> >::VALUE = std::numeric_limits<SimpleType<TValue, TSpec>>::min();

template <typename TValue, typename TSpec>
inline SimpleType<TValue, TSpec> const &
infimumValueImpl(SimpleType<TValue, TSpec> *)
{
return MinValue<SimpleType<TValue, TSpec> >::VALUE;
}


template <typename TValue, typename TSpec>
struct MaxValue<SimpleType<TValue, TSpec> >
{
static const SimpleType<TValue, TSpec> VALUE;
};

template <typename TValue, typename TSpec>
const SimpleType<TValue, TSpec> MaxValue<SimpleType<TValue, TSpec> >::VALUE = std::numeric_limits<SimpleType<TValue, TSpec>>::max();

template <typename TValue, typename TSpec>
inline SimpleType<TValue, TSpec> const &
supremumValueImpl(SimpleType<TValue, TSpec> *)
{
return MaxValue<SimpleType<TValue, TSpec> >::VALUE;
}


template <typename TValue, typename TSpec>
struct Spec<SimpleType<TValue, TSpec> >
{
typedef TSpec Type;
};

template <typename TValue, typename TSpec>
struct Spec<SimpleType<TValue, TSpec> const>
{
typedef TSpec Type;
};



template <typename TValue, typename TSpec, typename TRight>
struct CompareTypeImpl<SimpleType<TValue, TSpec>, TRight>
{
typedef TRight Type;
};





template <typename TTarget, typename T, typename TSourceValue, typename TSourceSpec>
inline typename RemoveConst_<TTarget>::Type
convertImpl(Convert<TTarget, T> const,
SimpleType<TSourceValue, TSourceSpec> const & source_)
{
typename RemoveConst_<TTarget>::Type target_;
assign(target_, source_);
return target_;
}


template <typename TStream, typename TValue, typename TSpec>
inline TStream &
operator<<(TStream & stream,
SimpleType<TValue, TSpec> const & data)
{
stream << convert<char>(data);
return stream;
}


template <typename TStream, typename TValue, typename TSpec>
inline TStream &
operator>>(TStream & stream,
SimpleType<TValue, TSpec> & data)
{
char c;
stream >> c;
assign(data, c);
return stream;
}


template <typename TTargetValue, typename TTargetSpec, typename TSourceValue, typename TSourceSpec>
inline void
assign(SimpleType<TTargetValue, TTargetSpec> & target,
SimpleType<TSourceValue, TSourceSpec> & source)
{
target.value = source.value;
}

template <typename TTargetValue, typename TTargetSpec, typename TSourceValue, typename TSourceSpec>
inline void
assign(SimpleType<TTargetValue, TTargetSpec> & target,
SimpleType<TSourceValue, TSourceSpec> const & source)
{
target.value = source.value;
}

template <typename TTargetValue, typename TTargetSpec, typename TSource>
inline void
assign(SimpleType<TTargetValue, TTargetSpec> & target,
TSource & source)
{
target.value = source;
}

template <typename TTargetValue, typename TTargetSpec, typename TSource>
inline void
assign(SimpleType<TTargetValue, TTargetSpec> & target,
TSource const & source)
{
target.value = source;
}


template <typename TTargetValue, typename TTargetSpec, typename TSourceSpec>
inline void
assign(SimpleType<TTargetValue, TTargetSpec> & target,
Proxy<TSourceSpec> & source)
{
target.value = getValue(source);
}

template <typename TTargetValue, typename TTargetSpec, typename TSourceSpec>
inline void
assign(SimpleType<TTargetValue, TTargetSpec> & target,
Proxy<TSourceSpec> const & source)
{
target.value = getValue(source);
}


template <typename TValue, typename TSpec>
inline void
assign(int64_t & c_target,
SimpleType<TValue, TSpec> & source)
{
c_target = source.value;
}

template <typename TValue, typename TSpec>
inline void
assign(int64_t & c_target,
SimpleType<TValue, TSpec> const & source)
{
c_target = source.value;
}

template <typename TValue, typename TSpec>
inline void
assign(uint64_t & c_target,
SimpleType<TValue, TSpec> & source)
{
c_target = source.value;
}

template <typename TValue, typename TSpec>
inline void
assign(uint64_t & c_target,
SimpleType<TValue, TSpec> const & source)
{
c_target = source.value;
}

template <typename TValue, typename TSpec>
inline void
assign(int & c_target,
SimpleType<TValue, TSpec> & source)
{
c_target = source.value;
}

template <typename TValue, typename TSpec>
inline void
assign(int & c_target,
SimpleType<TValue, TSpec> const & source)
{
c_target = source.value;
}

template <typename TValue, typename TSpec>
inline void
assign(unsigned int & c_target,
SimpleType<TValue, TSpec> & source)
{
c_target = source.value;
}

template <typename TValue, typename TSpec>
inline void
assign(unsigned int & c_target,
SimpleType<TValue, TSpec> const & source)
{
c_target = source.value;
}

template <typename TValue, typename TSpec>
inline void
assign(short & c_target,
SimpleType<TValue, TSpec> & source)
{
c_target = source.value;
}

template <typename TValue, typename TSpec>
inline void
assign(short & c_target,
SimpleType<TValue, TSpec> const & source)
{
c_target = source.value;
}

template <typename TValue, typename TSpec>
inline void
assign(unsigned short & c_target,
SimpleType<TValue, TSpec> & source)
{
c_target = source.value;
}

template <typename TValue, typename TSpec>
inline void
assign(unsigned short & c_target,
SimpleType<TValue, TSpec> const & source)
{
c_target = source.value;
}

template <typename TValue, typename TSpec>
inline void
assign(char & c_target,
SimpleType<TValue, TSpec> & source)
{
c_target = source.value;
}

template <typename TValue, typename TSpec>
inline void
assign(char & c_target,
SimpleType<TValue, TSpec> const & source)
{
c_target = source.value;
}

template <typename TValue, typename TSpec>
inline void
assign(signed char & c_target,
SimpleType<TValue, TSpec> & source)
{
c_target = source.value;
}

template <typename TValue, typename TSpec>
inline void
assign(signed char & c_target,
SimpleType<TValue, TSpec> const & source)
{
c_target = source.value;
}

template <typename TValue, typename TSpec>
inline void
assign(unsigned char & c_target,
SimpleType<TValue, TSpec> & source)
{
c_target = source.value;
}

template <typename TValue, typename TSpec>
inline void
assign(unsigned char & c_target,
SimpleType<TValue, TSpec> const & source)
{
c_target = source.value;
}


template <typename TValue, typename TSpec, typename TRight>
inline bool
operator==(SimpleType<TValue, TSpec> const & left_,
TRight const & right_)
{
typedef SimpleType<TValue, TSpec> TLeft;
typedef typename CompareType<TLeft, TRight>::Type TCompareType;
return convert<TCompareType>(left_) == convert<TCompareType>(right_);
}

template <typename TLeft, typename TValue, typename TSpec>
inline bool
operator==(TLeft const & left_,
SimpleType<TValue, TSpec> const & right_)
{
typedef SimpleType<TValue, TSpec> TRight;
typedef typename CompareType<TRight, TLeft>::Type TCompareType;
return convert<TCompareType>(left_) == convert<TCompareType>(right_);
}

template <typename TLeftValue, typename TLeftSpec, typename TRightValue, typename TRightSpec>
inline bool
operator==(SimpleType<TLeftValue, TLeftSpec> const & left_,
SimpleType<TRightValue, TRightSpec> const & right_)
{
typedef SimpleType<TLeftValue, TLeftSpec> TLeft;
typedef SimpleType<TRightValue, TRightSpec> TRight;
typedef typename CompareType<TLeft, TRight>::Type TCompareType;
return convert<TCompareType>(left_) == convert<TCompareType>(right_);
}

template <typename TValue, typename TSpec>
inline bool
operator==(SimpleType<TValue, TSpec> const & left_,
SimpleType<TValue, TSpec> const & right_)
{
return convert<TValue>(left_) == convert<TValue>(right_);
}

template <typename TSpec, typename TValue, typename TSpec2>
inline bool
operator==(Proxy<TSpec> const & left_,
SimpleType<TValue, TSpec2> const & right_)
{
typedef Proxy<TSpec> TLeft;
typedef SimpleType<TValue, TSpec> TRight;
typedef typename CompareType<TLeft, TRight>::Type TCompareType;
return convert<TCompareType>(left_) == convert<TCompareType>(right_);
}

template <typename TSpec, typename TValue, typename TSpec2>
inline bool
operator==(SimpleType<TValue, TSpec2> const & left_,
Proxy<TSpec> const & right_)
{
typedef SimpleType<TValue, TSpec> TLeft;
typedef Proxy<TSpec> TRight;
typedef typename CompareType<TLeft, TRight>::Type TCompareType;
return convert<TCompareType>(left_) == convert<TCompareType>(right_);
}


template <typename TValue, typename TSpec, typename TRight>
inline bool
operator!=(SimpleType<TValue, TSpec> const & left_,
TRight const & right_)
{
typedef SimpleType<TValue, TSpec> TLeft;
typedef typename CompareType<TLeft, TRight>::Type TCompareType;
return convert<TCompareType>(left_) != convert<TCompareType>(right_);
}

template <typename TLeft, typename TValue, typename TSpec>
inline bool
operator!=(TLeft const & left_,
SimpleType<TValue, TSpec> const & right_)
{
typedef SimpleType<TValue, TSpec> TRight;
typedef typename CompareType<TRight, TLeft>::Type TCompareType;
return convert<TCompareType>(left_) != convert<TCompareType>(right_);
}

template <typename TLeftValue, typename TLeftSpec, typename TRightValue, typename TRightSpec>
inline bool
operator!=(SimpleType<TLeftValue, TLeftSpec> const & left_,
SimpleType<TRightValue, TRightSpec> const & right_)
{
typedef SimpleType<TLeftValue, TLeftSpec> TLeft;
typedef SimpleType<TRightValue, TRightSpec> TRight;
typedef typename CompareType<TLeft, TRight>::Type TCompareType;
return convert<TCompareType>(left_) != convert<TCompareType>(right_);
}

template <typename TValue, typename TSpec>
inline bool
operator!=(SimpleType<TValue, TSpec> const & left_,
SimpleType<TValue, TSpec> const & right_)
{
return convert<TValue>(left_) != convert<TValue>(right_);
}

template <typename TSpec, typename TValue, typename TSpec2>
inline bool
operator!=(Proxy<TSpec> const & left_,
SimpleType<TValue, TSpec2> const & right_)
{
typedef Proxy<TSpec> TLeft;
typedef SimpleType<TValue, TSpec> TRight;
typedef typename CompareType<TLeft, TRight>::Type TCompareType;
return convert<TCompareType>(left_) != convert<TCompareType>(right_);
}

template <typename TValue, typename TSpec, typename TProxySpec>
inline bool
operator!=(SimpleType<TValue, TSpec> const & left_,
Proxy<TProxySpec> const & right_)
{
typedef SimpleType<TValue, TSpec> TLeft;
typedef Proxy<TProxySpec> TRight;
typedef typename CompareType<TRight, TLeft>::Type TCompareType;
return convert<TCompareType>(left_) != convert<TCompareType>(right_);
}


template <typename TValue, typename TSpec, typename TRight>
inline bool
operator<(SimpleType<TValue, TSpec> const & left_,
TRight const & right_)
{
typedef SimpleType<TValue, TSpec> TLeft;
typedef typename CompareType<TLeft, TRight>::Type TCompareType;
return convert<TCompareType>(left_) < convert<TCompareType>(right_);
}

template <typename TLeft, typename TValue, typename TSpec>
inline bool
operator<(TLeft const & left_,
SimpleType<TValue, TSpec> const & right_)
{
typedef SimpleType<TValue, TSpec> TRight;
typedef typename CompareType<TRight, TLeft>::Type TCompareType;
return convert<TCompareType>(left_) < convert<TCompareType>(right_);
}

template <typename TLeftValue, typename TLeftSpec, typename TRightValue, typename TRightSpec>
inline bool
operator<(SimpleType<TLeftValue, TLeftSpec> const & left_,
SimpleType<TRightValue, TRightSpec> const & right_)
{
typedef SimpleType<TLeftValue, TLeftSpec> TLeft;
typedef SimpleType<TRightValue, TRightSpec> TRight;
typedef typename CompareType<TLeft, TRight>::Type TCompareType;
return convert<TCompareType>(left_) < convert<TCompareType>(right_);
}

template <typename TValue, typename TSpec>
inline bool
operator<(SimpleType<TValue, TSpec> const & left_,
SimpleType<TValue, TSpec> const & right_)
{
return convert<TValue>(left_) < convert<TValue>(right_);
}

template <typename TSpec, typename TValue, typename TSpec2>
inline bool
operator<(Proxy<TSpec> const & left_,
SimpleType<TValue, TSpec2> const & right_)
{
typedef Proxy<TSpec> TLeft;
typedef SimpleType<TValue, TSpec> TRight;
typedef typename CompareType<TLeft, TRight>::Type TCompareType;
return convert<TCompareType>(left_) < convert<TCompareType>(right_);
}

template <typename TSpec, typename TValue, typename TSpec2>
inline bool
operator<(SimpleType<TValue, TSpec2> const & left_,
Proxy<TSpec> const & right_)
{
typedef SimpleType<TValue, TSpec> TLeft;
typedef Proxy<TSpec> TRight;
typedef typename CompareType<TLeft, TRight>::Type TCompareType;
return convert<TCompareType>(left_) < convert<TCompareType>(right_);
}


template <typename TValue, typename TSpec, typename TRight>
inline bool
operator<=(SimpleType<TValue, TSpec> const & left_,
TRight const & right_)
{
typedef SimpleType<TValue, TSpec> TLeft;
typedef typename CompareType<TLeft, TRight>::Type TCompareType;
return convert<TCompareType>(left_) <= convert<TCompareType>(right_);
}

template <typename TLeft, typename TValue, typename TSpec>
inline bool
operator<=(TLeft const & left_,
SimpleType<TValue, TSpec> const & right_)
{
typedef SimpleType<TValue, TSpec> TRight;
typedef typename CompareType<TRight, TLeft>::Type TCompareType;
return convert<TCompareType>(left_) <= convert<TCompareType>(right_);
}

template <typename TLeftValue, typename TLeftSpec, typename TRightValue, typename TRightSpec>
inline bool
operator<=(SimpleType<TLeftValue, TLeftSpec> const & left_,
SimpleType<TRightValue, TRightSpec> const & right_)
{
typedef SimpleType<TLeftValue, TLeftSpec> TLeft;
typedef SimpleType<TRightValue, TRightSpec> TRight;
typedef typename CompareType<TLeft, TRight>::Type TCompareType;
return convert<TCompareType>(left_) <= convert<TCompareType>(right_);
}

template <typename TValue, typename TSpec>
inline bool
operator<=(SimpleType<TValue, TSpec> const & left_,
SimpleType<TValue, TSpec> const & right_)
{
return convert<TValue>(left_) <= convert<TValue>(right_);
}

template <typename TSpec, typename TValue, typename TSpec2>
inline bool
operator<=(Proxy<TSpec> const & left_,
SimpleType<TValue, TSpec2> const & right_)
{
typedef Proxy<TSpec> TLeft;
typedef SimpleType<TValue, TSpec> TRight;
typedef typename CompareType<TLeft, TRight>::Type TCompareType;
return convert<TCompareType>(left_) <= convert<TCompareType>(right_);
}
template <typename TSpec, typename TValue, typename TSpec2>
inline bool
operator<=(SimpleType<TValue, TSpec2> const & left_,
Proxy<TSpec> const & right_)
{
typedef SimpleType<TValue, TSpec> TLeft;
typedef Proxy<TSpec> TRight;
typedef typename CompareType<TLeft, TRight>::Type TCompareType;
return convert<TCompareType>(left_) <= convert<TCompareType>(right_);
}


template <typename TValue, typename TSpec, typename TRight>
inline bool
operator>(SimpleType<TValue, TSpec> const & left_,
TRight const & right_)
{
typedef SimpleType<TValue, TSpec> TLeft;
typedef typename CompareType<TLeft, TRight>::Type TCompareType;
return convert<TCompareType>(left_) > convert<TCompareType>(right_);
}

template <typename TLeft, typename TValue, typename TSpec>
inline bool
operator>(TLeft const & left_,
SimpleType<TValue, TSpec> const & right_)
{
typedef SimpleType<TValue, TSpec> TRight;
typedef typename CompareType<TRight, TLeft>::Type TCompareType;
return convert<TCompareType>(left_) > convert<TCompareType>(right_);
}

template <typename TLeftValue, typename TLeftSpec, typename TRightValue, typename TRightSpec>
inline bool
operator>(SimpleType<TLeftValue, TLeftSpec> const & left_,
SimpleType<TRightValue, TRightSpec> const & right_)
{
typedef SimpleType<TLeftValue, TLeftSpec> TLeft;
typedef SimpleType<TRightValue, TRightSpec> TRight;
typedef typename CompareType<TLeft, TRight>::Type TCompareType;
return convert<TCompareType>(left_) > convert<TCompareType>(right_);
}

template <typename TValue, typename TSpec>
inline bool
operator>(SimpleType<TValue, TSpec> const & left_,
SimpleType<TValue, TSpec> const & right_)
{
return convert<TValue>(left_) > convert<TValue>(right_);
}

template <typename TSpec, typename TValue, typename TSpec2>
inline bool
operator>(Proxy<TSpec> const & left_,
SimpleType<TValue, TSpec2> const & right_)
{
typedef Proxy<TSpec> TLeft;
typedef SimpleType<TValue, TSpec> TRight;
typedef typename CompareType<TLeft, TRight>::Type TCompareType;
return convert<TCompareType>(left_) > convert<TCompareType>(right_);
}

template <typename TSpec, typename TValue, typename TSpec2>
inline bool
operator>(SimpleType<TValue, TSpec2> const & left_,
Proxy<TSpec> const & right_)
{
typedef SimpleType<TValue, TSpec> TLeft;
typedef Proxy<TSpec> TRight;
typedef typename CompareType<TLeft, TRight>::Type TCompareType;
return convert<TCompareType>(left_) > convert<TCompareType>(right_);
}


template <typename TValue, typename TSpec, typename TRight>
inline bool
operator>=(SimpleType<TValue, TSpec> const & left_,
TRight const & right_)
{
typedef SimpleType<TValue, TSpec> TLeft;
typedef typename CompareType<TLeft, TRight>::Type TCompareType;
return convert<TCompareType>(left_) >= convert<TCompareType>(right_);
}

template <typename TLeft, typename TValue, typename TSpec>
inline bool
operator>=(TLeft const & left_,
SimpleType<TValue, TSpec> const & right_)
{
typedef SimpleType<TValue, TSpec> TRight;
typedef typename CompareType<TRight, TLeft>::Type TCompareType;
return convert<TCompareType>(left_) >= convert<TCompareType>(right_);
}

template <typename TLeftValue, typename TLeftSpec, typename TRightValue, typename TRightSpec>
inline bool
operator>=(SimpleType<TLeftValue, TLeftSpec> const & left_,
SimpleType<TRightValue, TRightSpec> const & right_)
{
typedef SimpleType<TLeftValue, TLeftSpec> TLeft;
typedef SimpleType<TRightValue, TRightSpec> TRight;
typedef typename CompareType<TLeft, TRight>::Type TCompareType;
return convert<TCompareType>(left_) >= convert<TCompareType>(right_);
}

template <typename TValue, typename TSpec>
inline bool
operator>=(SimpleType<TValue, TSpec> const & left_,
SimpleType<TValue, TSpec> const & right_)
{
return convert<TValue>(left_) >= convert<TValue>(right_);
}

template <typename TSpec, typename TValue, typename TSpec2>
inline bool
operator>=(Proxy<TSpec> const & left_,
SimpleType<TValue, TSpec2> const & right_)
{
typedef Proxy<TSpec> TLeft;
typedef SimpleType<TValue, TSpec> TRight;
typedef typename CompareType<TLeft, TRight>::Type TCompareType;
return convert<TCompareType>(left_) >= convert<TCompareType>(right_);
}
template <typename TSpec, typename TValue, typename TSpec2>
inline bool
operator>=(SimpleType<TValue, TSpec2> const & left_,
Proxy<TSpec> const & right_)
{
typedef SimpleType<TValue, TSpec> TLeft;
typedef Proxy<TSpec> TRight;
typedef typename CompareType<TLeft, TRight>::Type TCompareType;
return convert<TCompareType>(left_) >= convert<TCompareType>(right_);
}


template <typename TValue, typename TSpec>
inline SimpleType<TValue, TSpec> &
operator++(SimpleType<TValue, TSpec> & me)
{
++me.value;
return me;
}

template <typename TValue, typename TSpec>
inline SimpleType<TValue, TSpec>
operator++(SimpleType<TValue, TSpec> & me, int)
{
SimpleType<TValue, TSpec> dummy = me;
++me.value;
return dummy;
}


template <typename TValue, typename TSpec>
inline SimpleType<TValue, TSpec> &
operator--(SimpleType<TValue, TSpec> & me)
{
--me.value;
return me;
}

template <typename TValue, typename TSpec>
inline SimpleType<TValue, TSpec>
operator--(SimpleType<TValue, TSpec> & me, int)
{
SimpleType<TValue, TSpec> dummy = me;
--me.value;
return dummy;
}


template <typename TValue, typename TSpec>
inline SimpleType<TValue, TSpec>
operator+(SimpleType<TValue, TSpec> const & v)
{
return v;
}

template <typename TValue, typename TSpec>
inline typename ValueSize<SimpleType<TValue, TSpec> >::Type
_internalOrdValue(SimpleType<TValue, TSpec> const & c)
{
return c.value;
}

template <typename TValue, typename TSpec>
inline typename ValueSize<SimpleType<TValue, TSpec> >::Type
ordValue(SimpleType<TValue, TSpec> const & c)
{
return convert<unsigned>(c);
}

}  

#endif  
