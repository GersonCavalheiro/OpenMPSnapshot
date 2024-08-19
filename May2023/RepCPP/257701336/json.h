










#ifndef AWS_JSON_AMALGATED_H_INCLUDED
# define AWS_JSON_AMALGATED_H_INCLUDED
#define AWS_JSON_IS_AMALGAMATION


#ifndef AWS_JSON_VERSION_H_INCLUDED
# define AWS_JSON_VERSION_H_INCLUDED

# define AWS_JSONCPP_VERSION_STRING "1.4.0"
# define AWS_JSONCPP_VERSION_MAJOR 1
# define AWS_JSONCPP_VERSION_MINOR 4
# define AWS_JSONCPP_VERSION_PATCH 0
# define AWS_JSONCPP_VERSION_QUALIFIER
# define AWS_JSONCPP_VERSION_HEXA ((JSONCPP_VERSION_MAJOR << 24) | (JSONCPP_VERSION_MINOR << 16) | (JSONCPP_VERSION_PATCH << 8))

#endif 









#ifndef AWS_JSON_CONFIG_H_INCLUDED
#define AWS_JSON_CONFIG_H_INCLUDED



#include <aws/core/SDKConfig.h>

#ifndef JSON_USE_EXCEPTION
#define JSON_USE_EXCEPTION 1
#endif


#ifdef AWS_JSON_IN_CPPTL
#include <cpptl/config.h>
#ifndef AWS_JSON_USE_CPPTL
#define AWS_JSON_USE_CPPTL 1
#endif
#endif

#ifdef AWS_JSON_IN_CPPTL
#define AWS_JSON_API CPPTL_API
#elif defined(USE_IMPORT_EXPORT)
#ifdef _MSC_VER
#ifdef AWS_CORE_EXPORTS
#define AWS_JSON_API __declspec(dllexport)
#define AWS_JSONCPP_DISABLE_DLL_INTERFACE_WARNING
#else
#define AWS_JSON_API __declspec(dllimport)
#define AWS_JSONCPP_DISABLE_DLL_INTERFACE_WARNING
#endif 
#endif 
#endif 
#if !defined(AWS_JSON_API)
#define AWS_JSON_API
#endif


#if defined(_MSC_VER) && _MSC_VER <= 1200 
#define AWS_JSON_USE_INT64_DOUBLE_CONVERSION 1
#pragma warning(disable : 4786)
#endif 

#if defined(_MSC_VER) && _MSC_VER >= 1500 
#define AWS_JSONCPP_DEPRECATED(message) __declspec(deprecated(message))
#pragma warning(push)
#pragma warning(disable : 4512)
#endif

#if !defined(AWS_JSONCPP_DEPRECATED)
#define AWS_JSONCPP_DEPRECATED(message)
#endif 

namespace Aws {
namespace External {
namespace Json {
typedef int Int;
typedef unsigned int UInt;
#if defined(AWS_JSON_NO_INT64)
typedef int LargestInt;
typedef unsigned int LargestUInt;
#undef AWS_JSON_HAS_INT64
#else                 
#if defined(_MSC_VER) 
typedef __int64 Int64;
typedef unsigned __int64 UInt64;
#else                 
typedef long long int Int64;
typedef unsigned long long int UInt64;
#endif 
typedef Int64 LargestInt;
typedef UInt64 LargestUInt;
#define AWS_JSON_HAS_INT64
#endif 
} 
} 
} 

#endif 









#ifndef AWS_JSON_FORWARDS_H_INCLUDED
#define AWS_JSON_FORWARDS_H_INCLUDED

#if !defined(AWS_JSON_IS_AMALGAMATION)
#include <config.h>
#endif 

namespace Aws {
namespace External {
namespace Json {

class FastWriter;
class StyledWriter;

class Reader;

class Features;

typedef unsigned int ArrayIndex;
class StaticString;
class Path;
class PathArgument;
class Value;
class ValueIteratorBase;
class ValueIterator;
class ValueConstIterator;
#ifdef AWS_JSON_VALUE_USE_INTERNAL_MAP
class ValueMapAllocator;
class ValueInternalLink;
class ValueInternalArray;
class ValueInternalMap;
#endif 

} 
} 
} 

#endif 









#ifndef AWS_CPPTL_JSON_FEATURES_H_INCLUDED
#define AWS_CPPTL_JSON_FEATURES_H_INCLUDED

#if !defined(AWS_JSON_IS_AMALGAMATION)
#include <forwards.h>
#endif 

namespace Aws {
namespace External {
namespace Json {


class AWS_JSON_API Features {
public:

static Features all();


static Features strictMode();


Features();

bool allowComments_;

bool strictRoot_;

bool allowDroppedNullPlaceholders_;

bool allowNumericKeys_;
};

} 
} 
} 

#endif 









#ifndef AWS_CPPTL_JSON_H_INCLUDED
#define AWS_CPPTL_JSON_H_INCLUDED

#if !defined(AWS_JSON_IS_AMALGAMATION)
#include <forwards.h>
#endif 

#include <aws/core/utils/memory/stl/AWSDeque.h>
#include <aws/core/utils/memory/stl/AWSMap.h>
#include <aws/core/utils/memory/stl/AWSStack.h>
#include <aws/core/utils/memory/stl/AWSStreamFwd.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/utils/memory/stl/AWSVector.h>

#ifndef AWS_JSON_USE_CPPTL_SMALLMAP
#include <map>
#else
#include <cpptl/smallmap.h>
#endif
#ifdef AWS_JSON_USE_CPPTL
#include <cpptl/forwards.h>
#endif

#if defined(AWS_JSONCPP_DISABLE_DLL_INTERFACE_WARNING)
#pragma warning(push)
#pragma warning(disable : 4251)
#endif 


namespace Aws {
namespace External {
namespace Json {


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



class AWS_JSON_API StaticString {
public:
explicit StaticString(const char* czstring) : str_(czstring) {}

operator const char*() const { return str_; }

const char* c_str() const { return str_; }

private:
const char* str_;
};


class AWS_JSON_API Value {
friend class ValueIteratorBase;
#ifdef AWS_JSON_VALUE_USE_INTERNAL_MAP
friend class ValueInternalLink;
friend class ValueInternalMap;
#endif
public:
typedef Aws::Vector<Aws::String> Members;
typedef ValueIterator iterator;
typedef ValueConstIterator const_iterator;
typedef Json::UInt UInt;
typedef Json::Int Int;
#if defined(AWS_JSON_HAS_INT64)
typedef Json::UInt64 UInt64;
typedef Json::Int64 Int64;
#endif 
typedef Json::LargestInt LargestInt;
typedef Json::LargestUInt LargestUInt;
typedef Json::ArrayIndex ArrayIndex;

static const Value& null;
static const LargestInt minLargestInt;
static const LargestInt maxLargestInt;
static const LargestUInt maxLargestUInt;

static const Int minInt;
static const Int maxInt;
static const UInt maxUInt;

#if defined(AWS_JSON_HAS_INT64)
static const Int64 minInt64;
static const Int64 maxInt64;
static const UInt64 maxUInt64;
#endif 

private:
#ifndef AWS_JSONCPP_DOC_EXCLUDE_IMPLEMENTATION
#ifndef AWS_JSON_VALUE_USE_INTERNAL_MAP
class CZString {
public:
enum DuplicationPolicy {
noDuplication = 0,
duplicate,
duplicateOnCopy
};
CZString(ArrayIndex index);
CZString(const char* cstr, DuplicationPolicy allocate);
CZString(const CZString& other);
~CZString();
CZString& operator=(CZString other);
bool operator<(const CZString& other) const;
bool operator==(const CZString& other) const;
ArrayIndex index() const;
const char* c_str() const;
bool isStaticString() const;

private:
void swap(CZString& other);
const char* cstr_;
ArrayIndex index_;
};

public:
#ifndef AWS_JSON_USE_CPPTL_SMALLMAP
typedef Aws::Map<CZString, Value> ObjectValues;
#else
typedef CppTL::SmallMap<CZString, Value> ObjectValues;
#endif 
#endif 
#endif 

public:

Value(ValueType type = nullValue);
Value(Int value);
Value(UInt value);
#if defined(AWS_JSON_HAS_INT64)
Value(Int64 value);
Value(UInt64 value);
#endif 
Value(double value);
Value(const char* value);
Value(const char* beginValue, const char* endValue);

Value(const StaticString& value);
Value(const Aws::String& value);
#ifdef AWS_JSON_USE_CPPTL
Value(const CppTL::ConstString& value);
#endif
Value(bool value);
Value(const Value& other);
~Value();

Value& operator=(Value other);
void swap(Value& other);
void swapPayload(Value& other);

ValueType type() const;

bool operator<(const Value& other) const;
bool operator<=(const Value& other) const;
bool operator>=(const Value& other) const;
bool operator>(const Value& other) const;
bool operator==(const Value& other) const;
bool operator!=(const Value& other) const;
int compare(const Value& other) const;

const char* asCString() const;
Aws::String asString() const;
#ifdef AWS_JSON_USE_CPPTL
CppTL::ConstString asConstString() const;
#endif
Int asInt() const;
UInt asUInt() const;
#if defined(AWS_JSON_HAS_INT64)
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

bool operator!() const;

void clear();

void resize(ArrayIndex size);

Value& operator[](ArrayIndex index);

Value& operator[](int index);

const Value& operator[](ArrayIndex index) const;

const Value& operator[](int index) const;

Value get(ArrayIndex index, const Value& defaultValue) const;
bool isValidIndex(ArrayIndex index) const;
Value& append(const Value& value);

Value& operator[](const char* key);
const Value& operator[](const char* key) const;
Value& operator[](const Aws::String& key);
const Value& operator[](const Aws::String& key) const;

Value& operator[](const StaticString& key);
#ifdef AWS_JSON_USE_CPPTL
Value& operator[](const CppTL::ConstString& key);
const Value& operator[](const CppTL::ConstString& key) const;
#endif
Value get(const char* key, const Value& defaultValue) const;
Value get(const Aws::String& key, const Value& defaultValue) const;
#ifdef AWS_JSON_USE_CPPTL
Value get(const CppTL::ConstString& key, const Value& defaultValue) const;
#endif
Value removeMember(const char* key);
Value removeMember(const Aws::String& key);

bool removeMember(const char* key, Value* removed);

bool removeIndex(ArrayIndex i, Value* removed);

bool isMember(const char* key) const;
bool isMember(const Aws::String& key) const;
#ifdef AWS_JSON_USE_CPPTL
bool isMember(const CppTL::ConstString& key) const;
#endif

Members getMemberNames() const;


void setComment(const char* comment, CommentPlacement placement);
void setComment(const char* comment, size_t len, CommentPlacement placement);
void setComment(const Aws::String& comment, CommentPlacement placement);
bool hasComment(CommentPlacement placement) const;
Aws::String getComment(CommentPlacement placement) const;

Aws::String toStyledString() const;

const_iterator begin() const;
const_iterator end() const;

iterator begin();
iterator end();

void setOffsetStart(size_t start);
void setOffsetLimit(size_t limit);
size_t getOffsetStart() const;
size_t getOffsetLimit() const;

private:
void initBasic(ValueType type, bool allocated = false);

Value& resolveReference(const char* key, bool isStatic);

#ifdef AWS_JSON_VALUE_USE_INTERNAL_MAP
inline bool isItemAvailable() const { return itemIsUsed_ == 0; }

inline void setItemUsed(bool isUsed = true) { itemIsUsed_ = isUsed ? 1 : 0; }

inline bool isMemberNameStatic() const { return memberNameIsStatic_ == 0; }

inline void setMemberNameIsStatic(bool isStatic) {
memberNameIsStatic_ = isStatic ? 1 : 0;
}
#endif 

private:
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
#ifdef AWS_JSON_VALUE_USE_INTERNAL_MAP
ValueInternalArray* array_;
ValueInternalMap* map_;
#else
ObjectValues* map_;
#endif
} value_;
ValueType type_ : 8;
unsigned int allocated_ : 1; 
#ifdef AWS_JSON_VALUE_USE_INTERNAL_MAP
unsigned int itemIsUsed_ : 1; 
unsigned int memberNameIsStatic_ : 1;  
#endif
CommentInfo* comments_;

size_t start_;
size_t limit_;
};


class AWS_JSON_API PathArgument {
public:
friend class Path;

PathArgument();
PathArgument(ArrayIndex index);
PathArgument(const char* key);
PathArgument(const Aws::String& key);

private:
enum Kind {
kindNone = 0,
kindIndex,
kindKey
};
Aws::String key_;
ArrayIndex index_;
Kind kind_;
};


class AWS_JSON_API Path {
public:
Path(const Aws::String& path,
const PathArgument& a1 = PathArgument(),
const PathArgument& a2 = PathArgument(),
const PathArgument& a3 = PathArgument(),
const PathArgument& a4 = PathArgument(),
const PathArgument& a5 = PathArgument());

const Value& resolve(const Value& root) const;
Value resolve(const Value& root, const Value& defaultValue) const;
Value& make(Value& root) const;

private:
typedef Aws::Vector<const PathArgument*> InArgs;
typedef Aws::Vector<PathArgument> Args;

void makePath(const Aws::String& path, const InArgs& in);
void addPathInArg(const Aws::String& path,
const InArgs& in,
InArgs::const_iterator& itInArg,
PathArgument::Kind kind);
void invalidPath(const Aws::String& path, int location);

Args args_;
};

#ifdef AWS_JSON_VALUE_USE_INTERNAL_MAP

class AWS_JSON_API ValueMapAllocator {
public:
virtual ~ValueMapAllocator();
virtual ValueInternalMap* newMap() = 0;
virtual ValueInternalMap* newMapCopy(const ValueInternalMap& other) = 0;
virtual void destructMap(ValueInternalMap* map) = 0;
virtual ValueInternalLink* allocateMapBuckets(unsigned int size) = 0;
virtual void releaseMapBuckets(ValueInternalLink* links) = 0;
virtual ValueInternalLink* allocateMapLink() = 0;
virtual void releaseMapLink(ValueInternalLink* link) = 0;
};


class AWS_JSON_API ValueInternalLink {
public:
enum {
itemPerLink = 6
}; 
enum InternalFlags {
flagAvailable = 0,
flagUsed = 1
};

ValueInternalLink();

~ValueInternalLink();

Value items_[itemPerLink];
char* keys_[itemPerLink];
ValueInternalLink* previous_;
ValueInternalLink* next_;
};


class AWS_JSON_API ValueInternalMap {
friend class ValueIteratorBase;
friend class Value;

public:
typedef unsigned int HashKey;
typedef unsigned int BucketIndex;

#ifndef AWS_JSONCPP_DOC_EXCLUDE_IMPLEMENTATION
struct IteratorState {
IteratorState() : map_(0), link_(0), itemIndex_(0), bucketIndex_(0) {}
ValueInternalMap* map_;
ValueInternalLink* link_;
BucketIndex itemIndex_;
BucketIndex bucketIndex_;
};
#endif 

ValueInternalMap();
ValueInternalMap(const ValueInternalMap& other);
ValueInternalMap& operator=(ValueInternalMap other);
~ValueInternalMap();

void swap(ValueInternalMap& other);

BucketIndex size() const;

void clear();

bool reserveDelta(BucketIndex growth);

bool reserve(BucketIndex newItemCount);

const Value* find(const char* key) const;

Value* find(const char* key);

Value& resolveReference(const char* key, bool isStatic);

void remove(const char* key);

void doActualRemove(ValueInternalLink* link,
BucketIndex index,
BucketIndex bucketIndex);

ValueInternalLink*& getLastLinkInBucket(BucketIndex bucketIndex);

Value& setNewItem(const char* key,
bool isStatic,
ValueInternalLink* link,
BucketIndex index);

Value& unsafeAdd(const char* key, bool isStatic, HashKey hashedKey);

HashKey hash(const char* key) const;

int compare(const ValueInternalMap& other) const;

private:
void makeBeginIterator(IteratorState& it) const;
void makeEndIterator(IteratorState& it) const;
static bool equals(const IteratorState& x, const IteratorState& other);
static void increment(IteratorState& iterator);
static void incrementBucket(IteratorState& iterator);
static void decrement(IteratorState& iterator);
static const char* key(const IteratorState& iterator);
static const char* key(const IteratorState& iterator, bool& isStatic);
static Value& value(const IteratorState& iterator);
static int distance(const IteratorState& x, const IteratorState& y);

private:
ValueInternalLink* buckets_;
ValueInternalLink* tailLink_;
BucketIndex bucketsSize_;
BucketIndex itemCount_;
};


class AWS_JSON_API ValueInternalArray {
friend class Value;
friend class ValueIteratorBase;

public:
enum {
itemsPerPage = 8
}; 
typedef Value::ArrayIndex ArrayIndex;
typedef unsigned int PageIndex;

#ifndef AWS_JSONCPP_DOC_EXCLUDE_IMPLEMENTATION
struct IteratorState 
{
IteratorState() : array_(0), currentPageIndex_(0), currentItemIndex_(0) {}
ValueInternalArray* array_;
Value** currentPageIndex_;
unsigned int currentItemIndex_;
};
#endif 

ValueInternalArray();
ValueInternalArray(const ValueInternalArray& other);
ValueInternalArray& operator=(ValueInternalArray other);
~ValueInternalArray();
void swap(ValueInternalArray& other);

void clear();
void resize(ArrayIndex newSize);

Value& resolveReference(ArrayIndex index);

Value* find(ArrayIndex index) const;

ArrayIndex size() const;

int compare(const ValueInternalArray& other) const;

private:
static bool equals(const IteratorState& x, const IteratorState& other);
static void increment(IteratorState& iterator);
static void decrement(IteratorState& iterator);
static Value& dereference(const IteratorState& iterator);
static Value& unsafeDereference(const IteratorState& iterator);
static int distance(const IteratorState& x, const IteratorState& y);
static ArrayIndex indexOf(const IteratorState& iterator);
void makeBeginIterator(IteratorState& it) const;
void makeEndIterator(IteratorState& it) const;
void makeIterator(IteratorState& it, ArrayIndex index) const;

void makeIndexValid(ArrayIndex index);

Value** pages_;
ArrayIndex size_;
PageIndex pageCount_;
};


class AWS_JSON_API ValueArrayAllocator {
public:
virtual ~ValueArrayAllocator();
virtual ValueInternalArray* newArray() = 0;
virtual ValueInternalArray* newArrayCopy(const ValueInternalArray& other) = 0;
virtual void destructArray(ValueInternalArray* array) = 0;

virtual void
reallocateArrayPageIndex(Value**& indexes,
ValueInternalArray::PageIndex& indexCount,
ValueInternalArray::PageIndex minNewIndexCount) = 0;
virtual void
releaseArrayPageIndex(Value** indexes,
ValueInternalArray::PageIndex indexCount) = 0;
virtual Value* allocateArrayPage() = 0;
virtual void releaseArrayPage(Value* value) = 0;
};
#endif 


class AWS_JSON_API ValueIteratorBase {
public:
typedef std::bidirectional_iterator_tag iterator_category;
typedef unsigned int size_t;
typedef int difference_type;
typedef ValueIteratorBase SelfType;

ValueIteratorBase();
#ifndef AWS_JSON_VALUE_USE_INTERNAL_MAP
explicit ValueIteratorBase(const Value::ObjectValues::iterator& current);
#else
ValueIteratorBase(const ValueInternalArray::IteratorState& state);
ValueIteratorBase(const ValueInternalMap::IteratorState& state);
#endif

bool operator==(const SelfType& other) const { return isEqual(other); }

bool operator!=(const SelfType& other) const { return !isEqual(other); }

difference_type operator-(const SelfType& other) const {
return computeDistance(other);
}

Value key() const;

UInt index() const;

const char* memberName() const;

protected:
Value& deref() const;

void increment();

void decrement();

difference_type computeDistance(const SelfType& other) const;

bool isEqual(const SelfType& other) const;

void copy(const SelfType& other);

private:
#ifndef AWS_JSON_VALUE_USE_INTERNAL_MAP
Value::ObjectValues::iterator current_;
bool isNull_;
#else
union {
ValueInternalArray::IteratorState array_;
ValueInternalMap::IteratorState map_;
} iterator_;
bool isArray_;
#endif
};


class AWS_JSON_API ValueConstIterator : public ValueIteratorBase {
friend class Value;

public:
typedef const Value value_type;
typedef unsigned int size_t;
typedef int difference_type;
typedef const Value& reference;
typedef const Value* pointer;
typedef ValueConstIterator SelfType;

ValueConstIterator();

private:

#ifndef AWS_JSON_VALUE_USE_INTERNAL_MAP
explicit ValueConstIterator(const Value::ObjectValues::iterator& current);
#else
ValueConstIterator(const ValueInternalArray::IteratorState& state);
ValueConstIterator(const ValueInternalMap::IteratorState& state);
#endif
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


class AWS_JSON_API ValueIterator : public ValueIteratorBase {
friend class Value;

public:
typedef Value value_type;
typedef unsigned int size_t;
typedef int difference_type;
typedef Value& reference;
typedef Value* pointer;
typedef ValueIterator SelfType;

ValueIterator();
ValueIterator(const ValueConstIterator& other);
ValueIterator(const ValueIterator& other);

private:

#ifndef AWS_JSON_VALUE_USE_INTERNAL_MAP
explicit ValueIterator(const Value::ObjectValues::iterator& current);
#else
ValueIterator(const ValueInternalArray::IteratorState& state);
ValueIterator(const ValueInternalMap::IteratorState& state);
#endif
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
} 
} 

namespace std {
template<>
inline void swap(Aws::External::Json::Value& a, Aws::External::Json::Value& b) { a.swap(b); }
}


#if defined(AWS_JSONCPP_DISABLE_DLL_INTERFACE_WARNING)
#pragma warning(pop)
#endif 

#endif 









#ifndef AWS_CPPTL_JSON_READER_H_INCLUDED
#define AWS_CPPTL_JSON_READER_H_INCLUDED

#if !defined(AWS_JSON_IS_AMALGAMATION)
#include <features.h>
#include <value.h>
#endif 

#if defined(AWS_JSONCPP_DISABLE_DLL_INTERFACE_WARNING)
#pragma warning(push)
#pragma warning(disable : 4251)
#endif 

namespace Aws {
namespace External {
namespace Json {


class AWS_JSON_API Reader {
public:
typedef char Char;
typedef const Char* Location;


struct StructuredError {
size_t offset_start;
size_t offset_limit;
Aws::String message;
};


Reader();


Reader(const Features& features);


bool
parse(const Aws::String& document, Value& root, bool collectComments = true);


bool parse(const char* beginDoc,
const char* endDoc,
Value& root,
bool collectComments = true);

bool parse(Aws::IStream& is, Value& root, bool collectComments = true);


AWS_JSONCPP_DEPRECATED("Use getFormattedErrorMessages instead")
Aws::String getFormatedErrorMessages() const;


Aws::String getFormattedErrorMessages() const;


Aws::Vector<StructuredError> getStructuredErrors() const;


bool pushError(const Value& value, const Aws::String& message);


bool pushError(const Value& value, const Aws::String& message, const Value& extra);


bool good() const;

private:
enum TokenType {
tokenEndOfStream = 0,
tokenObjectBegin,
tokenObjectEnd,
tokenArrayBegin,
tokenArrayEnd,
tokenString,
tokenNumber,
tokenTrue,
tokenFalse,
tokenNull,
tokenArraySeparator,
tokenMemberSeparator,
tokenComment,
tokenError
};

class Token {
public:
TokenType type_;
Location start_;
Location end_;
};

class ErrorInfo {
public:
Token token_;
Aws::String message_;
Location extra_;
};

typedef Aws::Deque<ErrorInfo> Errors;

bool readToken(Token& token);
void skipSpaces();
bool match(Location pattern, int patternLength);
bool readComment();
bool readCStyleComment();
bool readCppStyleComment();
bool readString();
void readNumber();
bool readValue();
bool readObject(Token& token);
bool readArray(Token& token);
bool decodeNumber(Token& token);
bool decodeNumber(Token& token, Value& decoded);
bool decodeString(Token& token);
bool decodeString(Token& token, Aws::String& decoded);
bool decodeDouble(Token& token);
bool decodeDouble(Token& token, Value& decoded);
bool decodeUnicodeCodePoint(Token& token,
Location& current,
Location end,
unsigned int& unicode);
bool decodeUnicodeEscapeSequence(Token& token,
Location& current,
Location end,
unsigned int& unicode);
bool addError(const Aws::String& message, Token& token, Location extra = 0);
bool recoverFromError(TokenType skipUntilToken);
bool addErrorAndRecover(const Aws::String& message,
Token& token,
TokenType skipUntilToken);
void skipUntilSpace();
Value& currentValue();
Char getNextChar();
void
getLocationLineAndColumn(Location location, int& line, int& column) const;
Aws::String getLocationLineAndColumn(Location location) const;
void addComment(Location begin, Location end, CommentPlacement placement);
void skipCommentTokens(Token& token);

typedef Aws::Stack<Value*> Nodes;
Nodes nodes_;
Errors errors_;
Aws::String document_;
Location begin_;
Location end_;
Location current_;
Location lastValueEnd_;
Value* lastValue_;
Aws::String commentsBefore_;
Features features_;
bool collectComments_;
};


AWS_JSON_API Aws::IStream& operator>>(Aws::IStream&, Value&);

} 
} 
} 

#if defined(AWS_JSONCPP_DISABLE_DLL_INTERFACE_WARNING)
#pragma warning(pop)
#endif 

#endif 









#ifndef AWS_JSON_WRITER_H_INCLUDED
#define AWS_JSON_WRITER_H_INCLUDED

#if !defined(AWS_JSON_IS_AMALGAMATION)
#include <value.h>
#endif 

#if defined(AWS_JSONCPP_DISABLE_DLL_INTERFACE_WARNING)
#pragma warning(push)
#pragma warning(disable : 4251)
#endif 

namespace Aws {
namespace External {
namespace Json {

class Value;


class AWS_JSON_API StreamWriter {
protected:
Aws::OStream& sout_;  
public:
enum class CommentStyle {
None,  
Most,  
All  
};

StreamWriter(Aws::OStream* sout);
virtual ~StreamWriter();
virtual int write(Value const& root) = 0;


class AWS_JSON_API Factory {
public:
virtual ~Factory();
virtual StreamWriter* newStreamWriter(Aws::OStream* sout) const = 0;
};  
};  

Aws::String writeString(Value const& root, StreamWriter::Factory const& factory);



class AWS_JSON_API StreamWriterBuilder : public StreamWriter::Factory {
public:


StreamWriter::CommentStyle cs_;

Aws::String indentation_;

StreamWriterBuilder();

StreamWriter* newStreamWriter(Aws::OStream* sout) const;
};


class AWS_JSON_API OldCompressingStreamWriterBuilder : public StreamWriter::Factory
{
public:


bool dropNullPlaceholders_;

bool omitEndingLineFeed_;

bool enableYAMLCompatibility_;

OldCompressingStreamWriterBuilder()
: dropNullPlaceholders_(false)
, omitEndingLineFeed_(false)
, enableYAMLCompatibility_(false)
{}
virtual StreamWriter* newStreamWriter(Aws::OStream*) const;
};


class AWS_JSON_API Writer {
public:
virtual ~Writer();

virtual Aws::String write(const Value& root) = 0;
};


class AWS_JSON_API FastWriter : public Writer {
public:
FastWriter();
virtual ~FastWriter() {}

void enableYAMLCompatibility();


void dropNullPlaceholders();

void omitEndingLineFeed();

public: 
virtual Aws::String write(const Value& root);

private:
void writeValue(const Value& value);

Aws::String document_;
bool yamlCompatiblityEnabled_;
bool dropNullPlaceholders_;
bool omitEndingLineFeed_;
};


class AWS_JSON_API StyledWriter : public Writer {
public:
StyledWriter();
virtual ~StyledWriter() {}

public: 

virtual Aws::String write(const Value& root);

private:
void writeValue(const Value& value);
void writeArrayValue(const Value& value);
bool isMultineArray(const Value& value);
void pushValue(const Aws::String& value);
void writeIndent();
void writeWithIndent(const Aws::String& value);
void indent();
void unindent();
void writeCommentBeforeValue(const Value& root);
void writeCommentAfterValueOnSameLine(const Value& root);
bool hasCommentForValue(const Value& value);
static Aws::String normalizeEOL(const Aws::String& text);

typedef Aws::Vector<Aws::String> ChildValues;

ChildValues childValues_;
Aws::String document_;
Aws::String indentString_;
int rightMargin_;
int indentSize_;
bool addChildValues_;
};


class AWS_JSON_API StyledStreamWriter {
public:
StyledStreamWriter(Aws::String indentation = "\t");
~StyledStreamWriter() {}

public:

void write(Aws::OStream& out, const Value& root);

private:
void writeValue(const Value& value);
void writeArrayValue(const Value& value);
bool isMultineArray(const Value& value);
void pushValue(const Aws::String& value);
void writeIndent();
void writeWithIndent(const Aws::String& value);
void indent();
void unindent();
void writeCommentBeforeValue(const Value& root);
void writeCommentAfterValueOnSameLine(const Value& root);
bool hasCommentForValue(const Value& value);
static Aws::String normalizeEOL(const Aws::String& text);

typedef Aws::Vector<Aws::String> ChildValues;

ChildValues childValues_;
Aws::OStream* document_;
Aws::String indentString_;
int rightMargin_;
Aws::String indentation_;
bool addChildValues_ : 1;
bool indented_ : 1;
};

#if defined(AWS_JSON_HAS_INT64)
Aws::String AWS_JSON_API valueToString(Int value);
Aws::String AWS_JSON_API valueToString(UInt value);
#endif 
Aws::String AWS_JSON_API valueToString(LargestInt value);
Aws::String AWS_JSON_API valueToString(LargestUInt value);
Aws::String AWS_JSON_API valueToString(double value);
Aws::String AWS_JSON_API valueToString(bool value);
Aws::String AWS_JSON_API valueToQuotedString(const char* value);

AWS_JSON_API Aws::OStream& operator<<(Aws::OStream&, const Value& root);

} 
} 
} 

#if defined(AWS_JSONCPP_DISABLE_DLL_INTERFACE_WARNING)
#pragma warning(pop)
#endif 

#endif 









#ifndef AWS_CPPTL_JSON_ASSERTIONS_H_INCLUDED
#define AWS_CPPTL_JSON_ASSERTIONS_H_INCLUDED

#include <stdlib.h>
#include <sstream>

#if !defined(AWS_JSON_IS_AMALGAMATION)
#include <config.h>
#endif 

#if JSON_USE_EXCEPTION
#include <stdexcept>
#define AWS_JSON_ASSERT(condition)                                                 \
assert(condition); 
#define AWS_JSON_FAIL_MESSAGE(message) do{Aws::OStringStream oss; oss << message; throw std::runtime_error(oss.str());}while(0)
#else 
#define AWS_JSON_ASSERT(condition) assert(condition);

#define AWS_JSON_FAIL_MESSAGE(message)                                             \
{                                                                            \
Aws::OStringStream oss;                                                    \
oss << message;                                                            \
assert(false && oss.str().c_str());                                        \
abort();                                                                   \
}


#endif

#define AWS_JSON_ASSERT_MESSAGE(condition, message)                                \
if (!(condition)) {                                                          \
AWS_JSON_FAIL_MESSAGE(message);                                                \
}

#endif 



#if defined(_MSC_VER) && _MSC_VER >= 1500 
#pragma warning(pop)
#endif


#endif 
