

#ifndef ABSL_STRINGS_INTERNAL_STR_SPLIT_INTERNAL_H_
#define ABSL_STRINGS_INTERNAL_STR_SPLIT_INTERNAL_H_

#include <array>
#include <initializer_list>
#include <iterator>
#include <map>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/macros.h"
#include "absl/base/port.h"
#include "absl/meta/type_traits.h"
#include "absl/strings/string_view.h"

#ifdef _GLIBCXX_DEBUG
#include "absl/strings/internal/stl_type_traits.h"
#endif  

namespace absl {
namespace strings_internal {

class ConvertibleToStringView {
public:
ConvertibleToStringView(const char* s)  
: value_(s) {}
ConvertibleToStringView(char* s) : value_(s) {}  
ConvertibleToStringView(absl::string_view s)     
: value_(s) {}
ConvertibleToStringView(const std::string& s)  
: value_(s) {}

ConvertibleToStringView(std::string&& s)  
: copy_(std::move(s)), value_(copy_) {}

ConvertibleToStringView(const ConvertibleToStringView& other)
: copy_(other.copy_),
value_(other.IsSelfReferential() ? copy_ : other.value_) {}

ConvertibleToStringView(ConvertibleToStringView&& other) {
StealMembers(std::move(other));
}

ConvertibleToStringView& operator=(ConvertibleToStringView other) {
StealMembers(std::move(other));
return *this;
}

absl::string_view value() const { return value_; }

private:
bool IsSelfReferential() const { return value_.data() == copy_.data(); }

void StealMembers(ConvertibleToStringView&& other) {
if (other.IsSelfReferential()) {
copy_ = std::move(other.copy_);
value_ = copy_;
other.value_ = other.copy_;
} else {
value_ = other.value_;
}
}

std::string copy_;
absl::string_view value_;
};

template <typename Splitter>
class SplitIterator {
public:
using iterator_category = std::input_iterator_tag;
using value_type = absl::string_view;
using difference_type = ptrdiff_t;
using pointer = const value_type*;
using reference = const value_type&;

enum State { kInitState, kLastState, kEndState };
SplitIterator(State state, const Splitter* splitter)
: pos_(0),
state_(state),
splitter_(splitter),
delimiter_(splitter->delimiter()),
predicate_(splitter->predicate()) {
if (splitter_->text().data() == nullptr) {
state_ = kEndState;
pos_ = splitter_->text().size();
return;
}

if (state_ == kEndState) {
pos_ = splitter_->text().size();
} else {
++(*this);
}
}

bool at_end() const { return state_ == kEndState; }

reference operator*() const { return curr_; }
pointer operator->() const { return &curr_; }

SplitIterator& operator++() {
do {
if (state_ == kLastState) {
state_ = kEndState;
return *this;
}
const absl::string_view text = splitter_->text();
const absl::string_view d = delimiter_.Find(text, pos_);
if (d.data() == text.data() + text.size()) state_ = kLastState;
curr_ = text.substr(pos_, d.data() - (text.data() + pos_));
pos_ += curr_.size() + d.size();
} while (!predicate_(curr_));
return *this;
}

SplitIterator operator++(int) {
SplitIterator old(*this);
++(*this);
return old;
}

friend bool operator==(const SplitIterator& a, const SplitIterator& b) {
return a.state_ == b.state_ && a.pos_ == b.pos_;
}

friend bool operator!=(const SplitIterator& a, const SplitIterator& b) {
return !(a == b);
}

private:
size_t pos_;
State state_;
absl::string_view curr_;
const Splitter* splitter_;
typename Splitter::DelimiterType delimiter_;
typename Splitter::PredicateType predicate_;
};

template <typename T, typename = void>
struct HasMappedType : std::false_type {};
template <typename T>
struct HasMappedType<T, absl::void_t<typename T::mapped_type>>
: std::true_type {};

template <typename T, typename = void>
struct HasValueType : std::false_type {};
template <typename T>
struct HasValueType<T, absl::void_t<typename T::value_type>> : std::true_type {
};

template <typename T, typename = void>
struct HasConstIterator : std::false_type {};
template <typename T>
struct HasConstIterator<T, absl::void_t<typename T::const_iterator>>
: std::true_type {};

std::false_type IsInitializerListDispatch(...);  
template <typename T>
std::true_type IsInitializerListDispatch(std::initializer_list<T>*);
template <typename T>
struct IsInitializerList
: decltype(IsInitializerListDispatch(static_cast<T*>(nullptr))) {};


template <typename C, bool has_value_type, bool has_mapped_type>
struct SplitterIsConvertibleToImpl : std::false_type {};

template <typename C>
struct SplitterIsConvertibleToImpl<C, true, false>
: std::is_constructible<typename C::value_type, absl::string_view> {};

template <typename C>
struct SplitterIsConvertibleToImpl<C, true, true>
: absl::conjunction<
std::is_constructible<typename C::key_type, absl::string_view>,
std::is_constructible<typename C::mapped_type, absl::string_view>> {};

template <typename C>
struct SplitterIsConvertibleTo
: SplitterIsConvertibleToImpl<
C,
#ifdef _GLIBCXX_DEBUG
!IsStrictlyBaseOfAndConvertibleToSTLContainer<C>::value &&
#endif  
!IsInitializerList<
typename std::remove_reference<C>::type>::value &&
HasValueType<C>::value && HasConstIterator<C>::value,
HasMappedType<C>::value> {
};

template <typename Delimiter, typename Predicate>
class Splitter {
public:
using DelimiterType = Delimiter;
using PredicateType = Predicate;
using const_iterator = strings_internal::SplitIterator<Splitter>;
using value_type = typename std::iterator_traits<const_iterator>::value_type;

Splitter(ConvertibleToStringView input_text, Delimiter d, Predicate p)
: text_(std::move(input_text)),
delimiter_(std::move(d)),
predicate_(std::move(p)) {}

absl::string_view text() const { return text_.value(); }
const Delimiter& delimiter() const { return delimiter_; }
const Predicate& predicate() const { return predicate_; }

const_iterator begin() const { return {const_iterator::kInitState, this}; }
const_iterator end() const { return {const_iterator::kEndState, this}; }

template <typename Container,
typename = typename std::enable_if<
SplitterIsConvertibleTo<Container>::value>::type>
operator Container() const {  
return ConvertToContainer<Container, typename Container::value_type,
HasMappedType<Container>::value>()(*this);
}

template <typename First, typename Second>
operator std::pair<First, Second>() const {  
absl::string_view first, second;
auto it = begin();
if (it != end()) {
first = *it;
if (++it != end()) {
second = *it;
}
}
return {First(first), Second(second)};
}

private:
template <typename Container, typename ValueType, bool is_map = false>
struct ConvertToContainer {
Container operator()(const Splitter& splitter) const {
Container c;
auto it = std::inserter(c, c.end());
for (const auto sp : splitter) {
*it++ = ValueType(sp);
}
return c;
}
};

template <typename A>
struct ConvertToContainer<std::vector<absl::string_view, A>,
absl::string_view, false> {
std::vector<absl::string_view, A> operator()(
const Splitter& splitter) const {
struct raw_view {
const char* data;
size_t size;
operator absl::string_view() const {  
return {data, size};
}
};
std::vector<absl::string_view, A> v;
std::array<raw_view, 16> ar;
for (auto it = splitter.begin(); !it.at_end();) {
size_t index = 0;
do {
ar[index].data = it->data();
ar[index].size = it->size();
++it;
} while (++index != ar.size() && !it.at_end());
v.insert(v.end(), ar.begin(), ar.begin() + index);
}
return v;
}
};

template <typename A>
struct ConvertToContainer<std::vector<std::string, A>, std::string, false> {
std::vector<std::string, A> operator()(const Splitter& splitter) const {
const std::vector<absl::string_view> v = splitter;
return std::vector<std::string, A>(v.begin(), v.end());
}
};

template <typename Container, typename First, typename Second>
struct ConvertToContainer<Container, std::pair<const First, Second>, true> {
Container operator()(const Splitter& splitter) const {
Container m;
typename Container::iterator it;
bool insert = true;
for (const auto sp : splitter) {
if (insert) {
it = Inserter<Container>::Insert(&m, First(sp), Second());
} else {
it->second = Second(sp);
}
insert = !insert;
}
return m;
}

template <typename Map>
struct Inserter {
using M = Map;
template <typename... Args>
static typename M::iterator Insert(M* m, Args&&... args) {
return m->insert(std::make_pair(std::forward<Args>(args)...)).first;
}
};

template <typename... Ts>
struct Inserter<std::map<Ts...>> {
using M = std::map<Ts...>;
template <typename... Args>
static typename M::iterator Insert(M* m, Args&&... args) {
return m->emplace(std::make_pair(std::forward<Args>(args)...)).first;
}
};

template <typename... Ts>
struct Inserter<std::multimap<Ts...>> {
using M = std::multimap<Ts...>;
template <typename... Args>
static typename M::iterator Insert(M* m, Args&&... args) {
return m->emplace(std::make_pair(std::forward<Args>(args)...));
}
};
};

ConvertibleToStringView text_;
Delimiter delimiter_;
Predicate predicate_;
};

}  
}  

#endif  
