

#ifndef SEQAN_MODIFIER_MODIFIER_VIEW_H_
#define SEQAN_MODIFIER_MODIFIER_VIEW_H_

namespace seqan
{








template <typename TFunctor>
struct ModView {};

template <typename TFunctor>
struct ModViewCargo
{
TFunctor    func;

ModViewCargo() : func()
{}
};

template <typename THost, typename TFunctor>
class ModifiedIterator<THost, ModView<TFunctor> >
{
public:
typedef typename Cargo<ModifiedIterator>::Type TCargo_;

THost _host;
TCargo_ _cargo;

mutable typename Value<ModifiedIterator>::Type tmp_value;

ModifiedIterator() : _host(), tmp_value()
{}

template <typename TOtherHost>
ModifiedIterator(ModifiedIterator<TOtherHost, ModView<TFunctor> > & origin) :
_host(origin._host), _cargo(origin._cargo), tmp_value()
{}

explicit
ModifiedIterator(THost const & host) :
_host(host), tmp_value()
{}

ModifiedIterator(THost const & host, TFunctor const & functor):
_host(host), tmp_value()
{
cargo(*this).func = functor;
}
};


template <typename THost, typename TFunctor>
class ModifiedString<THost, ModView<TFunctor> >
{
public:
typedef typename Pointer_<THost>::Type       THostPointer_;
typedef typename Cargo<ModifiedString>::Type TCargo_;

mutable THostPointer_ _host;
TCargo_ _cargo;

mutable typename Value<ModifiedString>::Type tmp_value;

ModifiedString() : _host(), tmp_value()
{}

explicit
ModifiedString(typename Parameter_<THost>::Type host):
_host(_toPointer(host)), tmp_value()
{}

explicit
ModifiedString(TFunctor const & functor):
_host(), tmp_value()
{
cargo(*this).func = functor;
}

template <typename THost_>
explicit
ModifiedString(THost_ & host,
SEQAN_CTOR_ENABLE_IF(IsConstructible<THost, THost_>)) :
_host(_toPointer(host)), tmp_value()
{
ignoreUnusedVariableWarning(dummy);
}

ModifiedString(typename Parameter_<THost>::Type host, TFunctor const & functor) :
_host(_toPointer(host)), tmp_value()
{
cargo(*this).func = functor;
}

template <typename THost_>
explicit
ModifiedString(THost_ & host,
TFunctor const & functor,
SEQAN_CTOR_ENABLE_IF(IsConstructible<THost, THost_>)) :
_host(_toPointer(host)), tmp_value()
{
ignoreUnusedVariableWarning(dummy);
cargo(*this).func = functor;
}

template <typename THost_>
explicit
ModifiedString(THost_ && host,
SEQAN_CTOR_ENABLE_IF(IsAnInnerHost<
typename RemoveReference<THost>::Type,
typename RemoveReference<THost_>::Type >)) :
_host(std::forward<THost_>(host)), tmp_value()
{
ignoreUnusedVariableWarning(dummy);
}

template <typename THost_>
explicit
ModifiedString(THost_ && host,
TFunctor const & functor,
SEQAN_CTOR_ENABLE_IF(IsAnInnerHost<
typename RemoveReference<THost>::Type,
typename RemoveReference<THost_>::Type >)) :
_host(std::forward<THost_>(host)), tmp_value()
{
ignoreUnusedVariableWarning(dummy);
cargo(*this).func = functor;
}

template <typename TPos>
inline typename Reference<ModifiedString>::Type
operator[](TPos pos)
{
return value(*this, pos);
}

template <typename TPos>
inline typename Reference<ModifiedString const>::Type
operator[](TPos pos) const
{
return value(*this, pos);
}
};



template <typename THost, typename TFunctor>
struct Cargo<ModifiedIterator<THost, ModView<TFunctor> > >
{
typedef ModViewCargo<TFunctor>    Type;
};


template <typename THost, typename TFunctor>
struct Value<ModifiedIterator<THost, ModView<TFunctor> > >
{
typedef typename TFunctor::result_type            TResult_;
typedef typename RemoveConst_<TResult_>::Type   Type;
};

template <typename THost, typename TFunctor>
struct Value<ModifiedIterator<THost, ModView<TFunctor> > const> :
Value<ModifiedIterator<THost, ModView<TFunctor> > >
{};


template <typename THost, typename TFunctor>
struct GetValue<ModifiedIterator<THost, ModView<TFunctor> > > :
Value<ModifiedIterator<THost, ModView<TFunctor> > >
{};

template <typename THost, typename TFunctor>
struct GetValue<ModifiedIterator<THost, ModView<TFunctor> > const> :
Value<ModifiedIterator<THost, ModView<TFunctor> > >
{};


template <typename THost, typename TFunctor>
struct Reference<ModifiedIterator<THost, ModView<TFunctor> > > :
Value<ModifiedIterator<THost, ModView<TFunctor> > >
{};

template <typename THost, typename TFunctor>
struct Reference<ModifiedIterator<THost, ModView<TFunctor> > const> :
Value<ModifiedIterator<THost, ModView<TFunctor> > >
{};



template <typename THost, typename TFunctor>
struct Cargo< ModifiedString<THost, ModView<TFunctor> > >
{
typedef ModViewCargo<TFunctor>    Type;
};



template <typename THost, typename TFunctor>
inline typename GetValue<ModifiedIterator<THost, ModView<TFunctor> > >::Type
getValue(ModifiedIterator<THost, ModView<TFunctor> > & me)
{
return cargo(me).func(*host(me));
}

template <typename THost, typename TFunctor>
inline typename GetValue<ModifiedIterator<THost, ModView<TFunctor> > const>::Type
getValue(ModifiedIterator<THost, ModView<TFunctor> > const & me)
{
return cargo(me).func(*host(me));
}


template <typename THost, typename TFunctor>
inline typename GetValue<ModifiedIterator<THost, ModView<TFunctor> > >::Type
value(ModifiedIterator<THost, ModView<TFunctor> > & me)
{
return getValue(me);
}

template <typename THost, typename TFunctor>
inline typename GetValue<ModifiedIterator<THost, ModView<TFunctor> > const>::Type
value(ModifiedIterator<THost, ModView<TFunctor> > const & me)
{
return getValue(me);
}


template <typename THost, typename TFunctor, typename TPos>
inline typename GetValue<ModifiedString<THost, ModView<TFunctor> > >::Type
getValue(ModifiedString<THost, ModView<TFunctor> > & me, TPos pos)
{
return cargo(me).func(getValue(host(me), pos));
}

template <typename THost, typename TFunctor, typename TPos>
inline typename GetValue<ModifiedString<THost, ModView<TFunctor> > const>::Type
getValue(ModifiedString<THost, ModView<TFunctor> > const & me, TPos pos)
{
return cargo(me).func(getValue(host(me), pos));
}


template <typename THost, typename TFunctor, typename TPos>
inline typename GetValue<ModifiedString<THost, ModView<TFunctor> > >::Type
value(ModifiedString<THost, ModView<TFunctor> > & me, TPos pos)
{
return getValue(me, pos);
}

template <typename THost, typename TFunctor, typename TPos>
inline typename GetValue<ModifiedString<THost, ModView<TFunctor> > const>::Type
value(ModifiedString<THost, ModView<TFunctor> > const & me, TPos pos)
{
return getValue(me, pos);
}


template <typename THost, typename TFunctor>
inline void
assignModViewFunctor(ModifiedString<THost, ModView<TFunctor> > & me, TFunctor const & functor)
{
cargo(me).func = functor;
}


template < typename TSequence, typename TFunctor >
inline void
convert(TSequence & sequence, TFunctor const &F)
{
#if defined (_OPENMP) && defined (SEQAN_PARALLEL)
typedef typename Position<TSequence>::Type    TPos;
typedef typename MakeSigned_<TPos>::Type    TSignedPos;

#pragma omp parallel for if(length(sequence) > 1000000)
for(TSignedPos p = 0; p < (TSignedPos)length(sequence); ++p)
sequence[p] = F(sequence[p]);

#else
typedef typename Iterator<TSequence, Standard>::Type    TIter;

TIter it = begin(sequence, Standard());
TIter itEnd = end(sequence, Standard());
for(; it != itEnd; ++it)
*it = F(*it);
#endif
}

template < typename TSequence, typename TFunctor >
inline void
convert(TSequence const & sequence, TFunctor const &F)
{
#if defined (_OPENMP) && defined (SEQAN_PARALLEL)
typedef typename Position<TSequence>::Type    TPos;
typedef typename MakeSigned_<TPos>::Type    TSignedPos;

#pragma omp parallel for if(length(sequence) > 1000000)
for(TSignedPos p = 0; p < (TSignedPos)length(sequence); ++p)
sequence[p] = F(sequence[p]);

#else
typedef typename Iterator<TSequence const, Standard>::Type    TIter;

TIter it = begin(sequence, Standard());
TIter itEnd = end(sequence, Standard());
for(; it != itEnd; ++it)
*it = F(*it);
#endif
}

}  

#endif  
