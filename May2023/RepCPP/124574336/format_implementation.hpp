




#ifndef BOOST_FORMAT_IMPLEMENTATION_HPP
#define BOOST_FORMAT_IMPLEMENTATION_HPP

#include <boost/config.hpp>
#include <boost/throw_exception.hpp>
#include <boost/assert.hpp>
#include <boost/format/format_class.hpp>
#include <algorithm> 

namespace boost {


template< class Ch, class Tr, class Alloc>
basic_format<Ch, Tr, Alloc>:: basic_format(const Ch* s)
: style_(0), cur_arg_(0), num_args_(0), dumped_(false),
exceptions_(io::all_error_bits)
{
if( s)
parse( s );
}

#if !defined(BOOST_NO_STD_LOCALE)
template< class Ch, class Tr, class Alloc>
basic_format<Ch, Tr, Alloc>:: basic_format(const Ch* s, const std::locale & loc)
: style_(0), cur_arg_(0), num_args_(0), dumped_(false),
exceptions_(io::all_error_bits), loc_(loc)
{
if(s) parse( s );
}

template< class Ch, class Tr, class Alloc>
basic_format<Ch, Tr, Alloc>:: basic_format(const string_type& s, const std::locale & loc)
: style_(0), cur_arg_(0), num_args_(0), dumped_(false),
exceptions_(io::all_error_bits), loc_(loc)
{
parse(s);  
}
#endif 
template< class Ch, class Tr, class Alloc>
io::detail::locale_t basic_format<Ch, Tr, Alloc>:: 
getloc() const {
return loc_ ? loc_.get() : io::detail::locale_t(); 
}

template< class Ch, class Tr, class Alloc>
basic_format<Ch, Tr, Alloc>:: basic_format(const string_type& s)
: style_(0), cur_arg_(0), num_args_(0), dumped_(false),
exceptions_(io::all_error_bits)
{
parse(s);  
}

template< class Ch, class Tr, class Alloc> 
basic_format<Ch, Tr, Alloc>:: basic_format(const basic_format& x)
: items_(x.items_), bound_(x.bound_), style_(x.style_),
cur_arg_(x.cur_arg_), num_args_(x.num_args_), dumped_(x.dumped_),
prefix_(x.prefix_), exceptions_(x.exceptions_), loc_(x.loc_)
{
}

template< class Ch, class Tr, class Alloc>  
basic_format<Ch, Tr, Alloc>& basic_format<Ch, Tr, Alloc>:: 
operator= (const basic_format& x) {
if(this == &x)
return *this;
(basic_format<Ch, Tr, Alloc>(x)).swap(*this);
return *this;
}
template< class Ch, class Tr, class Alloc>
void  basic_format<Ch, Tr, Alloc>:: 
swap (basic_format & x) {
std::swap(exceptions_, x.exceptions_);
std::swap(style_, x.style_); 
std::swap(cur_arg_, x.cur_arg_); 
std::swap(num_args_, x.num_args_);
std::swap(dumped_, x.dumped_);

items_.swap(x.items_);
prefix_.swap(x.prefix_);
bound_.swap(x.bound_);
}

template< class Ch, class Tr, class Alloc>
unsigned char basic_format<Ch,Tr, Alloc>:: exceptions() const {
return exceptions_; 
}

template< class Ch, class Tr, class Alloc>
unsigned char basic_format<Ch,Tr, Alloc>:: exceptions(unsigned char newexcept) { 
unsigned char swp = exceptions_; 
exceptions_ = newexcept; 
return swp; 
}

template<class Ch, class Tr, class Alloc>
void basic_format<Ch, Tr, Alloc>:: 
make_or_reuse_data (std::size_t nbitems) {
#if !defined(BOOST_NO_STD_LOCALE)
Ch fill = ( BOOST_USE_FACET(std::ctype<Ch>, getloc()) ). widen(' ');
#else
Ch fill = ' ';
#endif
if(items_.size() == 0)
items_.assign( nbitems, format_item_t(fill) );
else {
if(nbitems>items_.size())
items_.resize(nbitems, format_item_t(fill));
bound_.resize(0);
for(std::size_t i=0; i < nbitems; ++i)
items_[i].reset(fill); 
}
prefix_.resize(0);
}

template< class Ch, class Tr, class Alloc>
basic_format<Ch,Tr, Alloc>& basic_format<Ch,Tr, Alloc>:: 
clear () {

BOOST_ASSERT( bound_.size()==0 || num_args_ == static_cast<int>(bound_.size()) );

for(unsigned long i=0; i<items_.size(); ++i) {
if( bound_.size()==0 || items_[i].argN_<0 || !bound_[ items_[i].argN_ ] )
items_[i].res_.resize(0);
}
cur_arg_=0; dumped_=false;
if(bound_.size() != 0) {
for(; cur_arg_ < num_args_ && bound_[cur_arg_]; ++cur_arg_)
{}
}
return *this;
}

template< class Ch, class Tr, class Alloc>
basic_format<Ch,Tr, Alloc>& basic_format<Ch,Tr, Alloc>:: 
clear_binds () {
bound_.resize(0);
clear();
return *this;
}

template< class Ch, class Tr, class Alloc>
basic_format<Ch,Tr, Alloc>& basic_format<Ch,Tr, Alloc>:: 
clear_bind (int argN) {
if(argN<1 || argN > num_args_ || bound_.size()==0 || !bound_[argN-1] ) {
if( exceptions() & io::out_of_range_bit)
boost::throw_exception(io::out_of_range(argN, 1, num_args_+1 ) ); 
else return *this;
}
bound_[argN-1]=false;
clear();
return *this;
}

template< class Ch, class Tr, class Alloc>
int basic_format<Ch,Tr, Alloc>::
bound_args() const {
if(bound_.size()==0)
return 0;
int n=0;
for(int i=0; i<num_args_ ; ++i)
if(bound_[i])
++n;
return n;
}

template< class Ch, class Tr, class Alloc>
int basic_format<Ch,Tr, Alloc>::
fed_args() const {
if(bound_.size()==0)
return cur_arg_;
int n=0;
for(int i=0; i<cur_arg_ ; ++i)
if(!bound_[i])
++n;
return n;
}

template< class Ch, class Tr, class Alloc>
int basic_format<Ch,Tr, Alloc>::
cur_arg() const {
return cur_arg_+1; }

template< class Ch, class Tr, class Alloc>
int basic_format<Ch,Tr, Alloc>::
remaining_args() const {
if(bound_.size()==0)
return num_args_-cur_arg_;
int n=0;
for(int i=cur_arg_; i<num_args_ ; ++i)
if(!bound_[i])
++n;
return n;
}

template< class Ch, class Tr, class Alloc>
typename basic_format<Ch, Tr, Alloc>::string_type 
basic_format<Ch,Tr, Alloc>:: 
str () const {
if(items_.size()==0)
return prefix_;
if( cur_arg_ < num_args_)
if( exceptions() & io::too_few_args_bit )
boost::throw_exception(io::too_few_args(cur_arg_, num_args_)); 

unsigned long i;
string_type res;
res.reserve(size());
res += prefix_;
for(i=0; i < items_.size(); ++i) {
const format_item_t& item = items_[i];
res += item.res_;
if( item.argN_ == format_item_t::argN_tabulation) { 
BOOST_ASSERT( item.pad_scheme_ & format_item_t::tabulation);
if( static_cast<size_type>(item.fmtstate_.width_) > res.size() )
res.append( static_cast<size_type>(item.fmtstate_.width_) - res.size(),
item.fmtstate_.fill_ );
}
res += item.appendix_;
}
dumped_=true;
return res;
}
template< class Ch, class Tr, class Alloc>
typename std::basic_string<Ch, Tr, Alloc>::size_type  basic_format<Ch,Tr, Alloc>:: 
size () const {
#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable:4267)
#endif
BOOST_USING_STD_MAX();
size_type sz = prefix_.size();
unsigned long i;
for(i=0; i < items_.size(); ++i) {
const format_item_t& item = items_[i];
sz += item.res_.size();
if( item.argN_ == format_item_t::argN_tabulation)
sz = max BOOST_PREVENT_MACRO_SUBSTITUTION (sz,
static_cast<size_type>(item.fmtstate_.width_) );
sz += item.appendix_.size();
}
return sz;
#ifdef BOOST_MSVC
#pragma warning(pop)
#endif
}

namespace io {
namespace detail {

template<class Ch, class Tr, class Alloc, class T> 
basic_format<Ch, Tr, Alloc>&  
bind_arg_body (basic_format<Ch, Tr, Alloc>& self, int argN, const T& val) {
if(self.dumped_) 
self.clear(); 
if(argN<1 || argN > self.num_args_) {
if( self.exceptions() & io::out_of_range_bit )
boost::throw_exception(io::out_of_range(argN, 1, self.num_args_+1 ) );
else return self;
}
if(self.bound_.size()==0) 
self.bound_.assign(self.num_args_,false);
else 
BOOST_ASSERT( self.num_args_ == static_cast<signed int>(self.bound_.size()) );
int o_cur_arg = self.cur_arg_;
self.cur_arg_ = argN-1; 

self.bound_[self.cur_arg_]=false; 
self.operator%(val); 


self.cur_arg_ = o_cur_arg; 
self.bound_[argN-1]=true;
if(self.cur_arg_ == argN-1 ) {
while(self.cur_arg_ < self.num_args_ && self.bound_[self.cur_arg_])   
++self.cur_arg_;
}
BOOST_ASSERT( self.cur_arg_ >= self.num_args_ || ! self.bound_[self.cur_arg_]);
return self;
}

template<class Ch, class Tr, class Alloc, class T> basic_format<Ch, Tr, Alloc>&
modify_item_body (basic_format<Ch, Tr, Alloc>& self, int itemN, T manipulator) {
if(itemN<1 || itemN > static_cast<signed int>(self.items_.size() )) {
if( self.exceptions() & io::out_of_range_bit ) 
boost::throw_exception(io::out_of_range(itemN, 1, static_cast<int>(self.items_.size()) ));
else return self;
}
self.items_[itemN-1].fmtstate_. template apply_manip<T> ( manipulator );
return self;
}

} 
} 
} 



#endif  
