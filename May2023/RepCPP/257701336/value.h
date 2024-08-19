
#ifndef CPPTL_JSON_H_INCLUDED
#define CPPTL_JSON_H_INCLUDED

#if !defined(JSON_IS_AMALGAMATION)
#include "forwards.h"
#endif 
#include <string>
#include <vector>
#include <exception>

#ifndef JSON_USE_CPPTL_SMALLMAP
#include <map>
#else
#include <cpptl/smallmap.h>
#endif
#ifdef JSON_USE_CPPTL
#include <cpptl/forwards.h>
#endif

#if !defined(JSONCPP_NORETURN)
#  if defined(_MSC_VER)
#    define JSONCPP_NORETURN __declspec(noreturn)
#  elif defined(__GNUC__)
#    define JSONCPP_NORETURN __attribute__ ((__noreturn__))
#  else
#    define JSONCPP_NORETURN
#  endif
#endif

#if defined(JSONCPP_DISABLE_DLL_INTERFACE_WARNING)
#pragma warning(push)
#pragma warning(disable : 4251)
#endif 

#pragma pack(push, 8)


namespace Json {


class JSON_API Exception : public std::exception {
public:
Exception(JSONCPP_STRING const& msg);
~Exception() JSONCPP_NOEXCEPT JSONCPP_OVERRIDE;
char const* what() const JSONCPP_NOEXCEPT JSONCPP_OVERRIDE;
protected:
JSONCPP_STRING msg_;
};


class JSON_API RuntimeError : public Exception {
public:
RuntimeError(JSONCPP_STRING const& msg);
};


class JSON_API LogicError : public Exception {
public:
LogicError(JSONCPP_STRING const& msg);
};

JSONCPP_NORETURN void throwRuntimeError(JSONCPP_STRING const& msg);
JSONCPP_NORETURN void throwLogicError(JSONCPP_STRING const& msg);


enum ValueType {
nullValue = 0, 
intValue,      
uintValue,     
realValue,     
stringValue,   
booleanValue,  
arrayValue,    
objectValue    
};

enum CommentPlacement {
commentBefore = 0,      
commentAfterOnSameLine, 
commentAfter, 
numberOfCommentPlacement
};



class JSON_API StaticString {
public:
explicit StaticString(const char* czstring) : c_str_(czstring) {}

operator const char*() const { return c_str_; }

const char* c_str() const { return c_str_; }

private:
const char* c_str_;
};


class JSON_API Value {
friend class ValueIteratorBase;
public:
typedef std::vector<JSONCPP_STRING> Members;
typedef ValueIterator iterator;
typedef ValueConstIterator const_iterator;
typedef Json::UInt UInt;
typedef Json::Int Int;
#if defined(JSON_HAS_INT64)
typedef Json::UInt64 UInt64;
typedef Json::Int64 Int64;
#endif 
typedef Json::LargestInt LargestInt;
typedef Json::LargestUInt LargestUInt;
typedef Json::ArrayIndex ArrayIndex;

typedef std::string value_type;

static const Value& null;  
static const Value& nullRef;  
static Value const& nullSingleton(); 

static const LargestInt minLargestInt;
static const LargestInt maxLargestInt;
static const LargestUInt maxLargestUInt;

static const Int minInt;
static const Int maxInt;
static const UInt maxUInt;

#if defined(JSON_HAS_INT64)
static const Int64 minInt64;
static const Int64 maxInt64;
static const UInt64 maxUInt64;
#endif 

private:
#ifndef JSONCPP_DOC_EXCLUDE_IMPLEMENTATION
class CZString {
public:
enum DuplicationPolicy {
noDuplication = 0,
duplicate,
duplicateOnCopy
};
CZString(ArrayIndex index);
CZString(char const* str, unsigned length, DuplicationPolicy allocate);
CZString(CZString const& other);
#if JSON_HAS_RVALUE_REFERENCES
CZString(CZString&& other);
#endif
~CZString();
CZString& operator=(const CZString& other);

#if JSON_HAS_RVALUE_REFERENCES
CZString& operator=(CZString&& other);
#endif

bool operator<(CZString const& other) const;
bool operator==(CZString const& other) const;
ArrayIndex index() const;
char const* data() const;
unsigned length() const;
bool isStaticString() const;

private:
void swap(CZString& other);

struct StringStorage {
unsigned policy_: 2;
unsigned length_: 30; 
};

char const* cstr_;  
union {
ArrayIndex index_;
StringStorage storage_;
};
};

public:
#ifndef JSON_USE_CPPTL_SMALLMAP
typedef std::map<CZString, Value> ObjectValues;
#else
typedef CppTL::SmallMap<CZString, Value> ObjectValues;
#endif 
#endif 

public:

Value(ValueType type = nullValue);
Value(Int value);
Value(UInt value);
#if defined(JSON_HAS_INT64)
Value(Int64 value);
Value(UInt64 value);
#endif 
Value(double value);
Value(const char* value); 
Value(const char* begin, const char* end); 

Value(const StaticString& value);
Value(const JSONCPP_STRING& value); 
#ifdef JSON_USE_CPPTL
Value(const CppTL::ConstString& value);
#endif
Value(bool value);
Value(const Value& other);
#if JSON_HAS_RVALUE_REFERENCES
Value(Value&& other);
#endif
~Value();

Value& operator=(Value other);

void swap(Value& other);
void swapPayload(Value& other);

void copy(const Value& other);
void copyPayload(const Value& other);

ValueType type() const;

bool operator<(const Value& other) const;
bool operator<=(const Value& other) const;
bool operator>=(const Value& other) const;
bool operator>(const Value& other) const;
bool operator==(const Value& other) const;
bool operator!=(const Value& other) const;
int compare(const Value& other) const;

const char* asCString() const; 
#if JSONCPP_USING_SECURE_MEMORY
unsigned getCStringLength() const; 
#endif
JSONCPP_STRING asString() const; 

bool getString(
char const** begin, char const** end) const;
#ifdef JSON_USE_CPPTL
CppTL::ConstString asConstString() const;
#endif
Int asInt() const;
UInt asUInt() const;
#if defined(JSON_HAS_INT64)
Int64 asInt64() const;
UInt64 asUInt64() const;
#endif 
LargestInt asLargestInt() const;
LargestUInt asLargestUInt() const;
float asFloat() const;
double asDouble() const;
bool asBool() const;

bool isNull() const;
bool isBool() const;
bool isInt() const;
bool isInt64() const;
bool isUInt() const;
bool isUInt64() const;
bool isIntegral() const;
bool isDouble() const;
bool isNumeric() const;
bool isString() const;
bool isArray() const;
bool isObject() const;

bool isConvertibleTo(ValueType other) const;

ArrayIndex size() const;

bool empty() const;

explicit operator bool() const;

void clear();

void resize(ArrayIndex size);

Value& operator[](ArrayIndex index);

Value& operator[](int index);

const Value& operator[](ArrayIndex index) const;

const Value& operator[](int index) const;

Value get(ArrayIndex index, const Value& defaultValue) const;
bool isValidIndex(ArrayIndex index) const;
Value& append(const Value& value);

#if JSON_HAS_RVALUE_REFERENCES
Value& append(Value&& value);
#endif

Value& operator[](const char* key);
const Value& operator[](const char* key) const;
Value& operator[](const JSONCPP_STRING& key);
const Value& operator[](const JSONCPP_STRING& key) const;

Value& operator[](const StaticString& key);
#ifdef JSON_USE_CPPTL
Value& operator[](const CppTL::ConstString& key);
const Value& operator[](const CppTL::ConstString& key) const;
#endif
Value get(const char* key, const Value& defaultValue) const;
Value get(const char* begin, const char* end, const Value& defaultValue) const;
Value get(const JSONCPP_STRING& key, const Value& defaultValue) const;
#ifdef JSON_USE_CPPTL
Value get(const CppTL::ConstString& key, const Value& defaultValue) const;
#endif
Value const* find(char const* begin, char const* end) const;
Value const* demand(char const* begin, char const* end);
void removeMember(const char* key);
void removeMember(const JSONCPP_STRING& key);
bool removeMember(const char* key, Value* removed);

bool removeMember(JSONCPP_STRING const& key, Value* removed);
bool removeMember(const char* begin, const char* end, Value* removed);

bool removeIndex(ArrayIndex i, Value* removed);

bool isMember(const char* key) const;
bool isMember(const JSONCPP_STRING& key) const;
bool isMember(const char* begin, const char* end) const;
#ifdef JSON_USE_CPPTL
bool isMember(const CppTL::ConstString& key) const;
#endif

Members getMemberNames() const;


JSONCPP_DEPRECATED("Use setComment(JSONCPP_STRING const&) instead.")
void setComment(const char* comment, CommentPlacement placement);
void setComment(const char* comment, size_t len, CommentPlacement placement);
void setComment(const JSONCPP_STRING& comment, CommentPlacement placement);
bool hasComment(CommentPlacement placement) const;
JSONCPP_STRING getComment(CommentPlacement placement) const;

JSONCPP_STRING toStyledString() const;

const_iterator begin() const;
const_iterator end() const;

iterator begin();
iterator end();

void setOffsetStart(ptrdiff_t start);
void setOffsetLimit(ptrdiff_t limit);
ptrdiff_t getOffsetStart() const;
ptrdiff_t getOffsetLimit() const;

private:
void initBasic(ValueType type, bool allocated = false);

Value& resolveReference(const char* key);
Value& resolveReference(const char* key, const char* end);

struct CommentInfo {
CommentInfo();
~CommentInfo();

void setComment(const char* text, size_t len);

char* comment_;
};


union ValueHolder {
LargestInt int_;
LargestUInt uint_;
double real_;
bool bool_;
char* string_;  
ObjectValues* map_;
} value_;
ValueType type_ : 8;
unsigned int allocated_ : 1; 
CommentInfo* comments_;

ptrdiff_t start_;
ptrdiff_t limit_;
};


class JSON_API PathArgument {
public:
friend class Path;

PathArgument();
PathArgument(ArrayIndex index);
PathArgument(const char* key);
PathArgument(const JSONCPP_STRING& key);

private:
enum Kind {
kindNone = 0,
kindIndex,
kindKey
};
JSONCPP_STRING key_;
ArrayIndex index_;
Kind kind_;
};


class JSON_API Path {
public:
Path(const JSONCPP_STRING& path,
const PathArgument& a1 = PathArgument(),
const PathArgument& a2 = PathArgument(),
const PathArgument& a3 = PathArgument(),
const PathArgument& a4 = PathArgument(),
const PathArgument& a5 = PathArgument());

const Value& resolve(const Value& root) const;
Value resolve(const Value& root, const Value& defaultValue) const;
Value& make(Value& root) const;

private:
typedef std::vector<const PathArgument*> InArgs;
typedef std::vector<PathArgument> Args;

void makePath(const JSONCPP_STRING& path, const InArgs& in);
void addPathInArg(const JSONCPP_STRING& path,
const InArgs& in,
InArgs::const_iterator& itInArg,
PathArgument::Kind kind);
void invalidPath(const JSONCPP_STRING& path, int location);

Args args_;
};


class JSON_API ValueIteratorBase {
public:
typedef std::bidirectional_iterator_tag iterator_category;
typedef unsigned int size_t;
typedef int difference_type;
typedef ValueIteratorBase SelfType;

bool operator==(const SelfType& other) const { return isEqual(other); }

bool operator!=(const SelfType& other) const { return !isEqual(other); }

difference_type operator-(const SelfType& other) const {
return other.computeDistance(*this);
}

Value key() const;

UInt index() const;

JSONCPP_STRING name() const;

JSONCPP_DEPRECATED("Use `key = name();` instead.")
char const* memberName() const;
char const* memberName(char const** end) const;

protected:
Value& deref() const;

void increment();

void decrement();

difference_type computeDistance(const SelfType& other) const;

bool isEqual(const SelfType& other) const;

void copy(const SelfType& other);

private:
Value::ObjectValues::iterator current_;
bool isNull_;

public:
ValueIteratorBase();
explicit ValueIteratorBase(const Value::ObjectValues::iterator& current);
};


class JSON_API ValueConstIterator : public ValueIteratorBase {
friend class Value;

public:
typedef const Value value_type;
typedef const Value& reference;
typedef const Value* pointer;
typedef ValueConstIterator SelfType;

ValueConstIterator();
ValueConstIterator(ValueIterator const& other);

private:

explicit ValueConstIterator(const Value::ObjectValues::iterator& current);
public:
SelfType& operator=(const ValueIteratorBase& other);

SelfType operator++(int) {
SelfType temp(*this);
++*this;
return temp;
}

SelfType operator--(int) {
SelfType temp(*this);
--*this;
return temp;
}

SelfType& operator--() {
decrement();
return *this;
}

SelfType& operator++() {
increment();
return *this;
}

reference operator*() const { return deref(); }

pointer operator->() const { return &deref(); }
};


class JSON_API ValueIterator : public ValueIteratorBase {
friend class Value;

public:
typedef Value value_type;
typedef unsigned int size_t;
typedef int difference_type;
typedef Value& reference;
typedef Value* pointer;
typedef ValueIterator SelfType;

ValueIterator();
explicit ValueIterator(const ValueConstIterator& other);
ValueIterator(const ValueIterator& other);

private:

explicit ValueIterator(const Value::ObjectValues::iterator& current);
public:
SelfType& operator=(const SelfType& other);

SelfType operator++(int) {
SelfType temp(*this);
++*this;
return temp;
}

SelfType operator--(int) {
SelfType temp(*this);
--*this;
return temp;
}

SelfType& operator--() {
decrement();
return *this;
}

SelfType& operator++() {
increment();
return *this;
}

reference operator*() const { return deref(); }

pointer operator->() const { return &deref(); }
};

} 


namespace std {
template<>
inline void swap(Json::Value& a, Json::Value& b) { a.swap(b); }
}

#pragma pack(pop)

#if defined(JSONCPP_DISABLE_DLL_INTERFACE_WARNING)
#pragma warning(pop)
#endif 

#endif 
