
#ifndef BOOST_XPRESSIVE_DETAIL_SYMBOLS_HPP_DRJ_06_11_2007
#define BOOST_XPRESSIVE_DETAIL_SYMBOLS_HPP_DRJ_06_11_2007

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/noncopyable.hpp>
#include <boost/range/begin.hpp>
#include <boost/range/end.hpp>
#include <boost/range/value_type.hpp>
#include <boost/range/const_iterator.hpp>
#include <boost/shared_ptr.hpp>

namespace boost { namespace xpressive { namespace detail
{

template<typename Map>
struct symbols
{
typedef typename range_value<Map>::type::first_type key_type;
typedef typename range_value<Map>::type::second_type value_type;
typedef typename range_value<key_type>::type char_type;
typedef typename range_const_iterator<Map>::type iterator;
typedef typename range_const_iterator<key_type>::type key_iterator;
typedef value_type const *result_type;


template<typename Trans>
void load(Map const &map, Trans trans)
{
iterator begin = boost::begin(map);
iterator end = boost::end(map);
node* root_p = this->root.get();
for(; begin != end; ++begin)
{
key_iterator kbegin = boost::begin(begin->first);
key_iterator kend = boost::end(begin->first);
root_p = this->insert(root_p, kbegin, kend, &begin->second, trans);
}
this->root.reset(root_p);
}

template<typename BidiIter, typename Trans>
result_type operator ()(BidiIter &begin, BidiIter end, Trans trans) const
{
return this->search(begin, end, trans, this->root.get());
}

template<typename Sink>
void peek(Sink const &sink) const
{
this->peek_(this->root.get(), sink);
}

private:
struct node
: boost::noncopyable
{
node(char_type c)
: ch(c)
, lo(0)
, eq(0)
, hi(0)
#ifdef BOOST_DISABLE_THREADS
, tau(0)
#endif
{}

~node()
{
delete lo;
if (ch)
delete eq;
delete hi;
}

void swap(node& that)
{
std::swap(ch, that.ch);
std::swap(lo, that.lo);
std::swap(eq, that.eq);
std::swap(hi, that.hi);
#ifdef BOOST_DISABLE_THREADS
std::swap(tau, that.tau);
#endif
}

char_type ch;
node* lo;
union
{
node* eq;
result_type result;
};
node* hi;
#ifdef BOOST_DISABLE_THREADS
long tau;
#endif
};

template<typename Trans>
node* insert(node* p, key_iterator &begin, key_iterator end, result_type r, Trans trans) const
{
char_type c1 = 0;

if(begin != end)
{
c1 = trans(*begin);
}

if(!p)
{
p = new node(c1);
}

if(c1 < p->ch)
{
p->lo = this->insert(p->lo, begin, end, r, trans);
}
else if(c1 == p->ch)
{
if(0 == c1)
{
p->result = r;
}
else
{
p->eq = this->insert(p->eq, ++begin, end, r, trans);
}
}
else
{
p->hi = this->insert(p->hi, begin, end, r, trans);
}

return p;
}

#ifdef BOOST_DISABLE_THREADS
bool cond_rotation(bool left, node* const i, node* const j) const
{
if (i == j)
return false;
node* const k = (left ? i->hi : i->lo);
long psi = 2*i->tau - j->tau - (k ? k->tau : 0);
if (psi <= 0)
return false;

j->tau += -i->tau + (k ? k->tau : 0);
i->tau +=  j->tau - (k ? k->tau : 0);
if (left)
{
j->lo = k;
i->hi = i;
}
else
{
j->hi = k;
i->lo = i;
}
(*i).swap(*j);
return true;
}
#endif

template<typename BidiIter, typename Trans>
result_type search(BidiIter &begin, BidiIter end, Trans trans, node* p) const
{
result_type r = 0;
#ifdef BOOST_DISABLE_THREADS
node* p2 = p;
bool left = false;
#endif
char_type c1 = (begin != end ? trans(*begin) : 0);
while(p)
{
#ifdef BOOST_DISABLE_THREADS
++p->tau;
#endif
if(c1 == p->ch)
{
#ifdef BOOST_DISABLE_THREADS
if (this->cond_rotation(left, p, p2))
p = p2;
#endif
if (0 == p->ch)
{
r = p->result;
}
if(begin == end)
break;
++begin;
p = p->eq;
r = search(begin,end,trans,p);
if (0 == r)
{
r = search(end,end,trans,p);
if (0 == r)
{
--begin;
}
}
break;
}
else if(c1 < p->ch)
{
#ifdef BOOST_DISABLE_THREADS
left = true;
p2 = p;
#endif
p = p->lo;
}
else 
{
#ifdef BOOST_DISABLE_THREADS
left = false;
p2 = p;
#endif
p = p->hi;
}
}
return r;
}

template<typename Sink>
void peek_(node const *const &p, Sink const &sink) const
{
if(p)
{
sink(p->ch);
this->peek_(p->lo, sink);
this->peek_(p->hi, sink);
}
}

boost::shared_ptr<node> root;
};

}}} 

#endif
