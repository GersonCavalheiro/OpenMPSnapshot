

#ifndef TINYXML2_INCLUDED
#define TINYXML2_INCLUDED

#if defined(ANDROID_NDK) || defined(__BORLANDC__) || defined(__QNXNTO__)
#   include <ctype.h>
#   include <limits.h>
#   include <stdio.h>
#   include <stdlib.h>
#   include <string.h>
#   include <stdarg.h>
#else
#   include <cctype>
#   include <climits>
#   include <cstdio>
#   include <cstdlib>
#   include <cstring>
#   include <cstdarg>
#endif

#include <aws/core/utils/memory/AWSMemory.h>




#if defined( _DEBUG ) || defined( DEBUG ) || defined (__DEBUG__)
#   ifndef DEBUG
#       define DEBUG
#   endif
#endif

#ifdef _MSC_VER
#   pragma warning(push)
#   pragma warning(disable: 4251)
#endif

#ifdef _WIN32
#ifdef USE_IMPORT_EXPORT
#ifdef AWS_CORE_EXPORTS
#define TINYXML2_LIB __declspec(dllexport)
#else
#define TINYXML2_LIB __declspec(dllimport)
#endif 
#endif 
#endif

#ifndef TINYXML2_LIB
#define TINYXML2_LIB
#endif 


#if defined(DEBUG)
#   if defined(_MSC_VER)
#       
#       define TIXMLASSERT( x )           if ( !((void)0,(x))) { __debugbreak(); } 
#   elif defined (ANDROID_NDK)
#       include <android/log.h>
#       define TIXMLASSERT( x )           if ( !(x)) { __android_log_assert( "assert", "grinliz", "ASSERT in '%s' at %d.", __FILE__, __LINE__ ); }
#   else
#       include <assert.h>
#       define TIXMLASSERT                assert
#   endif
#   else
#       define TIXMLASSERT( x )           {}
#endif


#if defined(_MSC_VER) && (_MSC_VER >= 1400 ) && (!defined WINCE)

inline int TIXML_SNPRINTF(char* buffer, size_t size, const char* format, ...)
{
va_list va;
va_start(va, format);
int result = vsnprintf_s(buffer, size, _TRUNCATE, format, va);
va_end(va);
return result;
}
#define TIXML_SSCANF   sscanf_s
#elif defined WINCE
#define TIXML_SNPRINTF _snprintf
#define TIXML_SSCANF   sscanf
#else
#define TIXML_SNPRINTF snprintf
#define TIXML_SSCANF   sscanf
#endif


static const int TIXML2_MAJOR_VERSION = 2;
static const int TIXML2_MINOR_VERSION = 2;
static const int TIXML2_PATCH_VERSION = 0;

namespace Aws
{
namespace External
{
namespace tinyxml2
{
class XMLDocument;
class XMLElement;
class XMLAttribute;
class XMLComment;
class XMLText;
class XMLDeclaration;
class XMLUnknown;
class XMLPrinter;

static const char* ALLOCATION_TAG = "AWS::TinyXML";


class TINYXML2_LIB StrPair
{
public:
enum {
NEEDS_ENTITY_PROCESSING = 0x01,
NEEDS_NEWLINE_NORMALIZATION = 0x02,
COLLAPSE_WHITESPACE = 0x04,

TEXT_ELEMENT = NEEDS_ENTITY_PROCESSING | NEEDS_NEWLINE_NORMALIZATION,
TEXT_ELEMENT_LEAVE_ENTITIES = NEEDS_NEWLINE_NORMALIZATION,
ATTRIBUTE_NAME = 0,
ATTRIBUTE_VALUE = NEEDS_ENTITY_PROCESSING | NEEDS_NEWLINE_NORMALIZATION,
ATTRIBUTE_VALUE_LEAVE_ENTITIES = NEEDS_NEWLINE_NORMALIZATION,
COMMENT = NEEDS_NEWLINE_NORMALIZATION
};

StrPair() : _flags(0), _start(0), _end(0) {}
~StrPair();

void Set(char* start, char* end, int flags) {
Reset();
_start = start;
_end = end;
_flags = flags | NEEDS_FLUSH;
}

const char* GetStr();

bool Empty() const {
return _start == _end;
}

void SetInternedStr(const char* str) {
Reset();
_start = const_cast<char*>(str);
}

void SetStr(const char* str, int flags = 0);

char* ParseText(char* in, const char* endTag, int strFlags);
char* ParseName(char* in);

void TransferTo(StrPair* other);

private:
void Reset();
void CollapseWhitespace();

enum {
NEEDS_FLUSH = 0x100,
NEEDS_DELETE = 0x200
};

int     _flags;
char*   _start;
char*   _end;

StrPair(const StrPair& other);	
void operator=(StrPair& other);	
};



template <class T, int INIT>
class DynArray
{
public:
DynArray() {
_mem = _pool;
_allocated = INIT;
_size = 0;
}

~DynArray() {
if (_mem != _pool) {
Aws::DeleteArray<T>(_mem);
}
}

void Clear() {
_size = 0;
}

void Push(T t) {
TIXMLASSERT(_size < INT_MAX);
EnsureCapacity(_size + 1);
_mem[_size++] = t;
}

T* PushArr(int count) {
TIXMLASSERT(count >= 0);
TIXMLASSERT(_size <= INT_MAX - count);
EnsureCapacity(_size + count);
T* ret = &_mem[_size];
_size += count;
return ret;
}

T Pop() {
TIXMLASSERT(_size > 0);
return _mem[--_size];
}

void PopArr(int count) {
TIXMLASSERT(_size >= count);
_size -= count;
}

bool Empty() const					{
return _size == 0;
}

T& operator[](int i)				{
TIXMLASSERT(i >= 0 && i < _size);
return _mem[i];
}

const T& operator[](int i) const	{
TIXMLASSERT(i >= 0 && i < _size);
return _mem[i];
}

const T& PeekTop() const            {
TIXMLASSERT(_size > 0);
return _mem[_size - 1];
}

int Size() const					{
TIXMLASSERT(_size >= 0);
return _size;
}

int Capacity() const				{
return _allocated;
}

const T* Mem() const				{
return _mem;
}

T* Mem()							{
return _mem;
}

private:
DynArray(const DynArray&); 
void operator=(const DynArray&); 

void EnsureCapacity(int cap) {
TIXMLASSERT(cap > 0);
if (cap > _allocated) {
TIXMLASSERT(cap <= INT_MAX / 2);
int newAllocated = cap * 2;
T* newMem = Aws::NewArray<T>(newAllocated, ALLOCATION_TAG);
memcpy(newMem, _mem, sizeof(T)*_size);	
if (_mem != _pool) {
Aws::DeleteArray<T>(_mem);
}
_mem = newMem;
_allocated = newAllocated;
}
}

T*  _mem;
T   _pool[INIT];
int _allocated;		
int _size;			
};



class MemPool
{
public:
MemPool() {}
virtual ~MemPool() {}

virtual int ItemSize() const = 0;
virtual void* Alloc() = 0;
virtual void Free(void*) = 0;
virtual void SetTracked() = 0;
virtual void Clear() = 0;
};



template< int SIZE >
class MemPoolT : public MemPool
{
public:
MemPoolT() : _root(0), _currentAllocs(0), _nAllocs(0), _maxAllocs(0), _nUntracked(0)	{}
~MemPoolT() {
Clear();
}

void Clear() {
while (!_blockPtrs.Empty()) {
Block* b = _blockPtrs.Pop();
Aws::Delete(b);
}
_root = 0;
_currentAllocs = 0;
_nAllocs = 0;
_maxAllocs = 0;
_nUntracked = 0;
}

virtual int ItemSize() const	{
return SIZE;
}
int CurrentAllocs() const		{
return _currentAllocs;
}

virtual void* Alloc() {
if (!_root) {
Block* block = Aws::New<Block>(ALLOCATION_TAG);
_blockPtrs.Push(block);

for (int i = 0; i<COUNT - 1; ++i) {
block->chunk[i].next = &block->chunk[i + 1];
}
block->chunk[COUNT - 1].next = 0;
_root = block->chunk;
}
void* result = _root;
_root = _root->next;

++_currentAllocs;
if (_currentAllocs > _maxAllocs) {
_maxAllocs = _currentAllocs;
}
_nAllocs++;
_nUntracked++;
return result;
}

virtual void Free(void* mem) {
if (!mem) {
return;
}
--_currentAllocs;
Chunk* chunk = static_cast<Chunk*>(mem);
#ifdef DEBUG
memset(chunk, 0xfe, sizeof(Chunk));
#endif
chunk->next = _root;
_root = chunk;
}
void Trace(const char* name) {
printf("Mempool %s watermark=%d [%dk] current=%d size=%d nAlloc=%d blocks=%d\n",
name, _maxAllocs, _maxAllocs*SIZE / 1024, _currentAllocs, SIZE, _nAllocs, _blockPtrs.Size());
}

void SetTracked() {
_nUntracked--;
}

int Untracked() const {
return _nUntracked;
}

enum { COUNT = (4 * 1024) / SIZE }; 

private:
MemPoolT(const MemPoolT&); 
void operator=(const MemPoolT&); 

union Chunk {
Chunk*  next;
char    mem[SIZE];
};
struct Block {
Chunk chunk[COUNT];
};
DynArray< Block*, 10 > _blockPtrs;
Chunk* _root;

int _currentAllocs;
int _nAllocs;
int _maxAllocs;
int _nUntracked;
};




class TINYXML2_LIB XMLVisitor
{
public:
virtual ~XMLVisitor() {}

virtual bool VisitEnter(const XMLDocument& )			{
return true;
}
virtual bool VisitExit(const XMLDocument& )			{
return true;
}

virtual bool VisitEnter(const XMLElement& , const XMLAttribute* )	{
return true;
}
virtual bool VisitExit(const XMLElement& )			{
return true;
}

virtual bool Visit(const XMLDeclaration& )		{
return true;
}
virtual bool Visit(const XMLText& )					{
return true;
}
virtual bool Visit(const XMLComment& )				{
return true;
}
virtual bool Visit(const XMLUnknown& )				{
return true;
}
};

enum XMLError {
XML_SUCCESS = 0,
XML_NO_ERROR = 0,
XML_NO_ATTRIBUTE,
XML_WRONG_ATTRIBUTE_TYPE,
XML_ERROR_FILE_NOT_FOUND,
XML_ERROR_FILE_COULD_NOT_BE_OPENED,
XML_ERROR_FILE_READ_ERROR,
XML_ERROR_ELEMENT_MISMATCH,
XML_ERROR_PARSING_ELEMENT,
XML_ERROR_PARSING_ATTRIBUTE,
XML_ERROR_IDENTIFYING_TAG,
XML_ERROR_PARSING_TEXT,
XML_ERROR_PARSING_CDATA,
XML_ERROR_PARSING_COMMENT,
XML_ERROR_PARSING_DECLARATION,
XML_ERROR_PARSING_UNKNOWN,
XML_ERROR_EMPTY_DOCUMENT,
XML_ERROR_MISMATCHED_ELEMENT,
XML_ERROR_PARSING,
XML_CAN_NOT_CONVERT_TEXT,
XML_NO_TEXT_NODE,

XML_ERROR_COUNT
};



class XMLUtil
{
public:
static const char* SkipWhiteSpace(const char* p)	{
TIXMLASSERT(p);
while (IsWhiteSpace(*p)) {
++p;
}
TIXMLASSERT(p);
return p;
}
static char* SkipWhiteSpace(char* p)				{
return const_cast<char*>(SkipWhiteSpace(const_cast<const char*>(p)));
}

static bool IsWhiteSpace(char p)					{
return !IsUTF8Continuation(p) && isspace(static_cast<unsigned char>(p));
}

inline static bool IsNameStartChar(unsigned char ch) {
if (ch >= 128) {
return true;
}
if (isalpha(ch)) {
return true;
}
return ch == ':' || ch == '_';
}

inline static bool IsNameChar(unsigned char ch) {
return IsNameStartChar(ch)
|| isdigit(ch)
|| ch == '.'
|| ch == '-';
}

inline static bool StringEqual(const char* p, const char* q, int nChar = INT_MAX)  {
if (p == q) {
return true;
}
int n = 0;
while (*p && *q && *p == *q && n<nChar) {
++p;
++q;
++n;
}
if ((n == nChar) || (*p == 0 && *q == 0)) {
return true;
}
return false;
}

inline static bool IsUTF8Continuation(const char p) {
return (p & 0x80) != 0;
}

static const char* ReadBOM(const char* p, bool* hasBOM);
static const char* GetCharacterRef(const char* p, char* value, int* length);
static void ConvertUTF32ToUTF8(unsigned long input, char* output, int* length);

static void ToStr(int v, char* buffer, int bufferSize);
static void ToStr(unsigned v, char* buffer, int bufferSize);
static void ToStr(bool v, char* buffer, int bufferSize);
static void ToStr(float v, char* buffer, int bufferSize);
static void ToStr(double v, char* buffer, int bufferSize);

static bool	ToInt(const char* str, int* value);
static bool ToUnsigned(const char* str, unsigned* value);
static bool	ToBool(const char* str, bool* value);
static bool	ToFloat(const char* str, float* value);
static bool ToDouble(const char* str, double* value);
};



class TINYXML2_LIB XMLNode
{
friend class XMLDocument;
friend class XMLElement;
public:

const XMLDocument* GetDocument() const	{
return _document;
}
XMLDocument* GetDocument()				{
return _document;
}

virtual XMLElement*		ToElement()		{
return 0;
}
virtual XMLText*		ToText()		{
return 0;
}
virtual XMLComment*		ToComment()		{
return 0;
}
virtual XMLDocument*	ToDocument()	{
return 0;
}
virtual XMLDeclaration*	ToDeclaration()	{
return 0;
}
virtual XMLUnknown*		ToUnknown()		{
return 0;
}

virtual const XMLElement*		ToElement() const		{
return 0;
}
virtual const XMLText*			ToText() const			{
return 0;
}
virtual const XMLComment*		ToComment() const		{
return 0;
}
virtual const XMLDocument*		ToDocument() const		{
return 0;
}
virtual const XMLDeclaration*	ToDeclaration() const	{
return 0;
}
virtual const XMLUnknown*		ToUnknown() const		{
return 0;
}


const char* Value() const;


void SetValue(const char* val, bool staticMem = false);

const XMLNode*	Parent() const			{
return _parent;
}

XMLNode* Parent()						{
return _parent;
}

bool NoChildren() const					{
return !_firstChild;
}

const XMLNode*  FirstChild() const		{
return _firstChild;
}

XMLNode*		FirstChild()			{
return _firstChild;
}


const XMLElement* FirstChildElement(const char* value = 0) const;

XMLElement* FirstChildElement(const char* value = 0)	{
return const_cast<XMLElement*>(const_cast<const XMLNode*>(this)->FirstChildElement(value));
}

const XMLNode*	LastChild() const						{
return _lastChild;
}

XMLNode*		LastChild()								{
return const_cast<XMLNode*>(const_cast<const XMLNode*>(this)->LastChild());
}


const XMLElement* LastChildElement(const char* value = 0) const;

XMLElement* LastChildElement(const char* value = 0)	{
return const_cast<XMLElement*>(const_cast<const XMLNode*>(this)->LastChildElement(value));
}

const XMLNode*	PreviousSibling() const					{
return _prev;
}

XMLNode*	PreviousSibling()							{
return _prev;
}

const XMLElement*	PreviousSiblingElement(const char* value = 0) const;

XMLElement*	PreviousSiblingElement(const char* value = 0) {
return const_cast<XMLElement*>(const_cast<const XMLNode*>(this)->PreviousSiblingElement(value));
}

const XMLNode*	NextSibling() const						{
return _next;
}

XMLNode*	NextSibling()								{
return _next;
}

const XMLElement*	NextSiblingElement(const char* value = 0) const;

XMLElement*	NextSiblingElement(const char* value = 0)	{
return const_cast<XMLElement*>(const_cast<const XMLNode*>(this)->NextSiblingElement(value));
}


XMLNode* InsertEndChild(XMLNode* addThis);

XMLNode* LinkEndChild(XMLNode* addThis)	{
return InsertEndChild(addThis);
}

XMLNode* InsertFirstChild(XMLNode* addThis);

XMLNode* InsertAfterChild(XMLNode* afterThis, XMLNode* addThis);


void DeleteChildren();


void DeleteChild(XMLNode* node);


virtual XMLNode* ShallowClone(XMLDocument* document) const = 0;


virtual bool ShallowEqual(const XMLNode* compare) const = 0;


virtual bool Accept(XMLVisitor* visitor) const = 0;

virtual char* ParseDeep(char*, StrPair*);

protected:
XMLNode(XMLDocument*);
virtual ~XMLNode();

XMLDocument*	_document;
XMLNode*		_parent;
mutable StrPair	_value;

XMLNode*		_firstChild;
XMLNode*		_lastChild;

XMLNode*		_prev;
XMLNode*		_next;

private:
MemPool*		_memPool;
void Unlink(XMLNode* child);
static void DeleteNode(XMLNode* node);
void InsertChildPreamble(XMLNode* insertThis) const;

XMLNode(const XMLNode&);	
XMLNode& operator=(const XMLNode&);	
};



class TINYXML2_LIB XMLText : public XMLNode
{
friend class XMLBase;
friend class XMLDocument;
public:
virtual bool Accept(XMLVisitor* visitor) const;

virtual XMLText* ToText()			{
return this;
}
virtual const XMLText* ToText() const	{
return this;
}

void SetCData(bool isCData)			{
_isCData = isCData;
}
bool CData() const						{
return _isCData;
}

char* ParseDeep(char*, StrPair* endTag);
virtual XMLNode* ShallowClone(XMLDocument* document) const;
virtual bool ShallowEqual(const XMLNode* compare) const;

protected:
XMLText(XMLDocument* doc) : XMLNode(doc), _isCData(false)	{}
virtual ~XMLText()												{}

private:
bool _isCData;

XMLText(const XMLText&);	
XMLText& operator=(const XMLText&);	
};



class TINYXML2_LIB XMLComment : public XMLNode
{
friend class XMLDocument;
public:
virtual XMLComment*	ToComment()					{
return this;
}
virtual const XMLComment* ToComment() const		{
return this;
}

virtual bool Accept(XMLVisitor* visitor) const;

char* ParseDeep(char*, StrPair* endTag);
virtual XMLNode* ShallowClone(XMLDocument* document) const;
virtual bool ShallowEqual(const XMLNode* compare) const;

protected:
XMLComment(XMLDocument* doc);
virtual ~XMLComment();

private:
XMLComment(const XMLComment&);	
XMLComment& operator=(const XMLComment&);	
};



class TINYXML2_LIB XMLDeclaration : public XMLNode
{
friend class XMLDocument;
public:
virtual XMLDeclaration*	ToDeclaration()					{
return this;
}
virtual const XMLDeclaration* ToDeclaration() const		{
return this;
}

virtual bool Accept(XMLVisitor* visitor) const;

char* ParseDeep(char*, StrPair* endTag);
virtual XMLNode* ShallowClone(XMLDocument* document) const;
virtual bool ShallowEqual(const XMLNode* compare) const;

protected:
XMLDeclaration(XMLDocument* doc);
virtual ~XMLDeclaration();

private:
XMLDeclaration(const XMLDeclaration&);	
XMLDeclaration& operator=(const XMLDeclaration&);	
};



class TINYXML2_LIB XMLUnknown : public XMLNode
{
friend class XMLDocument;
public:
virtual XMLUnknown*	ToUnknown()					{
return this;
}
virtual const XMLUnknown* ToUnknown() const		{
return this;
}

virtual bool Accept(XMLVisitor* visitor) const;

char* ParseDeep(char*, StrPair* endTag);
virtual XMLNode* ShallowClone(XMLDocument* document) const;
virtual bool ShallowEqual(const XMLNode* compare) const;

protected:
XMLUnknown(XMLDocument* doc);
virtual ~XMLUnknown();

private:
XMLUnknown(const XMLUnknown&);	
XMLUnknown& operator=(const XMLUnknown&);	
};




class TINYXML2_LIB XMLAttribute
{
friend class XMLElement;
public:
const char* Name() const;

const char* Value() const;

const XMLAttribute* Next() const {
return _next;
}


int		 IntValue() const				{
int i = 0;
QueryIntValue(&i);
return i;
}
unsigned UnsignedValue() const			{
unsigned i = 0;
QueryUnsignedValue(&i);
return i;
}
bool	 BoolValue() const				{
bool b = false;
QueryBoolValue(&b);
return b;
}
double 	 DoubleValue() const			{
double d = 0;
QueryDoubleValue(&d);
return d;
}
float	 FloatValue() const				{
float f = 0;
QueryFloatValue(&f);
return f;
}


XMLError QueryIntValue(int* value) const;
XMLError QueryUnsignedValue(unsigned int* value) const;
XMLError QueryBoolValue(bool* value) const;
XMLError QueryDoubleValue(double* value) const;
XMLError QueryFloatValue(float* value) const;

void SetAttribute(const char* value);
void SetAttribute(int value);
void SetAttribute(unsigned value);
void SetAttribute(bool value);
void SetAttribute(double value);
void SetAttribute(float value);

private:
enum { BUF_SIZE = 200 };

XMLAttribute() : _next(0), _memPool(0) {}
virtual ~XMLAttribute()	{}

XMLAttribute(const XMLAttribute&);	
void operator=(const XMLAttribute&);	
void SetName(const char* name);

char* ParseDeep(char* p, bool processEntities);

mutable StrPair _name;
mutable StrPair _value;
XMLAttribute*   _next;
MemPool*        _memPool;
};



class TINYXML2_LIB XMLElement : public XMLNode
{
friend class XMLBase;
friend class XMLDocument;
public:
const char* Name() const		{
return Value();
}
void SetName(const char* str, bool staticMem = false)	{
SetValue(str, staticMem);
}

virtual XMLElement* ToElement()				{
return this;
}
virtual const XMLElement* ToElement() const {
return this;
}
virtual bool Accept(XMLVisitor* visitor) const;


const char* Attribute(const char* name, const char* value = 0) const;


int		 IntAttribute(const char* name) const		{
int i = 0;
QueryIntAttribute(name, &i);
return i;
}
unsigned UnsignedAttribute(const char* name) const {
unsigned i = 0;
QueryUnsignedAttribute(name, &i);
return i;
}
bool	 BoolAttribute(const char* name) const	{
bool b = false;
QueryBoolAttribute(name, &b);
return b;
}
double 	 DoubleAttribute(const char* name) const	{
double d = 0;
QueryDoubleAttribute(name, &d);
return d;
}
float	 FloatAttribute(const char* name) const	{
float f = 0;
QueryFloatAttribute(name, &f);
return f;
}


XMLError QueryIntAttribute(const char* name, int* value) const				{
const XMLAttribute* a = FindAttribute(name);
if (!a) {
return XML_NO_ATTRIBUTE;
}
return a->QueryIntValue(value);
}
XMLError QueryUnsignedAttribute(const char* name, unsigned int* value) const	{
const XMLAttribute* a = FindAttribute(name);
if (!a) {
return XML_NO_ATTRIBUTE;
}
return a->QueryUnsignedValue(value);
}
XMLError QueryBoolAttribute(const char* name, bool* value) const				{
const XMLAttribute* a = FindAttribute(name);
if (!a) {
return XML_NO_ATTRIBUTE;
}
return a->QueryBoolValue(value);
}
XMLError QueryDoubleAttribute(const char* name, double* value) const			{
const XMLAttribute* a = FindAttribute(name);
if (!a) {
return XML_NO_ATTRIBUTE;
}
return a->QueryDoubleValue(value);
}
XMLError QueryFloatAttribute(const char* name, float* value) const			{
const XMLAttribute* a = FindAttribute(name);
if (!a) {
return XML_NO_ATTRIBUTE;
}
return a->QueryFloatValue(value);
}



int QueryAttribute(const char* name, int* value) const {
return QueryIntAttribute(name, value);
}

int QueryAttribute(const char* name, unsigned int* value) const {
return QueryUnsignedAttribute(name, value);
}

int QueryAttribute(const char* name, bool* value) const {
return QueryBoolAttribute(name, value);
}

int QueryAttribute(const char* name, double* value) const {
return QueryDoubleAttribute(name, value);
}

int QueryAttribute(const char* name, float* value) const {
return QueryFloatAttribute(name, value);
}

void SetAttribute(const char* name, const char* value)	{
XMLAttribute* a = FindOrCreateAttribute(name);
a->SetAttribute(value);
}
void SetAttribute(const char* name, int value)			{
XMLAttribute* a = FindOrCreateAttribute(name);
a->SetAttribute(value);
}
void SetAttribute(const char* name, unsigned value)		{
XMLAttribute* a = FindOrCreateAttribute(name);
a->SetAttribute(value);
}
void SetAttribute(const char* name, bool value)			{
XMLAttribute* a = FindOrCreateAttribute(name);
a->SetAttribute(value);
}
void SetAttribute(const char* name, double value)		{
XMLAttribute* a = FindOrCreateAttribute(name);
a->SetAttribute(value);
}
void SetAttribute(const char* name, float value)		{
XMLAttribute* a = FindOrCreateAttribute(name);
a->SetAttribute(value);
}


void DeleteAttribute(const char* name);

const XMLAttribute* FirstAttribute() const {
return _rootAttribute;
}
const XMLAttribute* FindAttribute(const char* name) const;


const char* GetText() const;


void SetText(const char* inText);
void SetText(int value);
void SetText(unsigned value);
void SetText(bool value);
void SetText(double value);
void SetText(float value);


XMLError QueryIntText(int* ival) const;
XMLError QueryUnsignedText(unsigned* uval) const;
XMLError QueryBoolText(bool* bval) const;
XMLError QueryDoubleText(double* dval) const;
XMLError QueryFloatText(float* fval) const;

enum {
OPEN,		
CLOSED,		
CLOSING		
};
int ClosingType() const {
return _closingType;
}
char* ParseDeep(char* p, StrPair* endTag);
virtual XMLNode* ShallowClone(XMLDocument* document) const;
virtual bool ShallowEqual(const XMLNode* compare) const;

private:
XMLElement(XMLDocument* doc);
virtual ~XMLElement();
XMLElement(const XMLElement&);	
void operator=(const XMLElement&);	

XMLAttribute* FindAttribute(const char* name) {
return const_cast<XMLAttribute*>(const_cast<const XMLElement*>(this)->FindAttribute(name));
}
XMLAttribute* FindOrCreateAttribute(const char* name);
char* ParseAttributes(char* p);
static void DeleteAttribute(XMLAttribute* attribute);

enum { BUF_SIZE = 200 };
int _closingType;
XMLAttribute* _rootAttribute;
};


enum Whitespace {
PRESERVE_WHITESPACE,
COLLAPSE_WHITESPACE
};



class TINYXML2_LIB XMLDocument : public XMLNode
{
friend class XMLElement;
public:
XMLDocument(bool processEntities = true, Whitespace = PRESERVE_WHITESPACE);
~XMLDocument();

virtual XMLDocument* ToDocument()				{
return this;
}
virtual const XMLDocument* ToDocument() const	{
return this;
}


XMLError Parse(const char* xml, size_t nBytes = (size_t)(-1));


XMLError LoadFile(const char* filename);


XMLError LoadFile(FILE*);


XMLError SaveFile(const char* filename, bool compact = false);


XMLError SaveFile(FILE* fp, bool compact = false);

bool ProcessEntities() const		{
return _processEntities;
}
Whitespace WhitespaceMode() const	{
return _whitespace;
}


bool HasBOM() const {
return _writeBOM;
}

void SetBOM(bool useBOM) {
_writeBOM = useBOM;
}


XMLElement* RootElement()				{
return FirstChildElement();
}
const XMLElement* RootElement() const	{
return FirstChildElement();
}


void Print(XMLPrinter* streamer = 0) const;
virtual bool Accept(XMLVisitor* visitor) const;


XMLElement* NewElement(const char* name);

XMLComment* NewComment(const char* comment);

XMLText* NewText(const char* text);

XMLDeclaration* NewDeclaration(const char* text = 0);

XMLUnknown* NewUnknown(const char* text);


void DeleteNode(XMLNode* node);

void SetError(XMLError error, const char* str1, const char* str2);

bool Error() const {
return _errorID != XML_NO_ERROR;
}
XMLError  ErrorID() const {
return _errorID;
}
const char* ErrorName() const;

const char* GetErrorStr1() const {
return _errorStr1;
}
const char* GetErrorStr2() const {
return _errorStr2;
}
void PrintError() const;

void Clear();

char* Identify(char* p, XMLNode** node);

virtual XMLNode* ShallowClone(XMLDocument* ) const	{
return 0;
}
virtual bool ShallowEqual(const XMLNode* ) const	{
return false;
}

private:
XMLDocument(const XMLDocument&);	
void operator=(const XMLDocument&);	

bool        _writeBOM;
bool        _processEntities;
XMLError    _errorID;
Whitespace  _whitespace;
const char* _errorStr1;
const char* _errorStr2;
char*       _charBuffer;

MemPoolT< sizeof(XMLElement) >	 _elementPool;
MemPoolT< sizeof(XMLAttribute) > _attributePool;
MemPoolT< sizeof(XMLText) >		 _textPool;
MemPoolT< sizeof(XMLComment) >	 _commentPool;

static const char* _errorNames[XML_ERROR_COUNT];

void Parse();
};



class TINYXML2_LIB XMLHandle
{
public:
XMLHandle(XMLNode* node)												{
_node = node;
}
XMLHandle(XMLNode& node)												{
_node = &node;
}
XMLHandle(const XMLHandle& ref)										{
_node = ref._node;
}
XMLHandle& operator=(const XMLHandle& ref)							{
_node = ref._node;
return *this;
}

XMLHandle FirstChild() 													{
return XMLHandle(_node ? _node->FirstChild() : 0);
}
XMLHandle FirstChildElement(const char* value = 0)						{
return XMLHandle(_node ? _node->FirstChildElement(value) : 0);
}
XMLHandle LastChild()													{
return XMLHandle(_node ? _node->LastChild() : 0);
}
XMLHandle LastChildElement(const char* _value = 0)						{
return XMLHandle(_node ? _node->LastChildElement(_value) : 0);
}
XMLHandle PreviousSibling()												{
return XMLHandle(_node ? _node->PreviousSibling() : 0);
}
XMLHandle PreviousSiblingElement(const char* _value = 0)				{
return XMLHandle(_node ? _node->PreviousSiblingElement(_value) : 0);
}
XMLHandle NextSibling()													{
return XMLHandle(_node ? _node->NextSibling() : 0);
}
XMLHandle NextSiblingElement(const char* _value = 0)					{
return XMLHandle(_node ? _node->NextSiblingElement(_value) : 0);
}

XMLNode* ToNode()							{
return _node;
}
XMLElement* ToElement() 					{
return ((_node == 0) ? 0 : _node->ToElement());
}
XMLText* ToText() 							{
return ((_node == 0) ? 0 : _node->ToText());
}
XMLUnknown* ToUnknown() 					{
return ((_node == 0) ? 0 : _node->ToUnknown());
}
XMLDeclaration* ToDeclaration() 			{
return ((_node == 0) ? 0 : _node->ToDeclaration());
}

private:
XMLNode* _node;
};



class TINYXML2_LIB XMLConstHandle
{
public:
XMLConstHandle(const XMLNode* node)											{
_node = node;
}
XMLConstHandle(const XMLNode& node)											{
_node = &node;
}
XMLConstHandle(const XMLConstHandle& ref)										{
_node = ref._node;
}

XMLConstHandle& operator=(const XMLConstHandle& ref)							{
_node = ref._node;
return *this;
}

const XMLConstHandle FirstChild() const											{
return XMLConstHandle(_node ? _node->FirstChild() : 0);
}
const XMLConstHandle FirstChildElement(const char* value = 0) const				{
return XMLConstHandle(_node ? _node->FirstChildElement(value) : 0);
}
const XMLConstHandle LastChild()	const										{
return XMLConstHandle(_node ? _node->LastChild() : 0);
}
const XMLConstHandle LastChildElement(const char* _value = 0) const				{
return XMLConstHandle(_node ? _node->LastChildElement(_value) : 0);
}
const XMLConstHandle PreviousSibling() const									{
return XMLConstHandle(_node ? _node->PreviousSibling() : 0);
}
const XMLConstHandle PreviousSiblingElement(const char* _value = 0) const		{
return XMLConstHandle(_node ? _node->PreviousSiblingElement(_value) : 0);
}
const XMLConstHandle NextSibling() const										{
return XMLConstHandle(_node ? _node->NextSibling() : 0);
}
const XMLConstHandle NextSiblingElement(const char* _value = 0) const			{
return XMLConstHandle(_node ? _node->NextSiblingElement(_value) : 0);
}


const XMLNode* ToNode() const				{
return _node;
}
const XMLElement* ToElement() const			{
return ((_node == 0) ? 0 : _node->ToElement());
}
const XMLText* ToText() const				{
return ((_node == 0) ? 0 : _node->ToText());
}
const XMLUnknown* ToUnknown() const			{
return ((_node == 0) ? 0 : _node->ToUnknown());
}
const XMLDeclaration* ToDeclaration() const	{
return ((_node == 0) ? 0 : _node->ToDeclaration());
}

private:
const XMLNode* _node;
};



class TINYXML2_LIB XMLPrinter : public XMLVisitor
{
public:

XMLPrinter(FILE* file = 0, bool compact = false, int depth = 0);
virtual ~XMLPrinter()	{}


void PushHeader(bool writeBOM, bool writeDeclaration);

void OpenElement(const char* name, bool compactMode = false);
void PushAttribute(const char* name, const char* value);
void PushAttribute(const char* name, int value);
void PushAttribute(const char* name, unsigned value);
void PushAttribute(const char* name, bool value);
void PushAttribute(const char* name, double value);
virtual void CloseElement(bool compactMode = false);

void PushText(const char* text, bool cdata = false);
void PushText(int value);
void PushText(unsigned value);
void PushText(bool value);
void PushText(float value);
void PushText(double value);

void PushComment(const char* comment);

void PushDeclaration(const char* value);
void PushUnknown(const char* value);

virtual bool VisitEnter(const XMLDocument& );
virtual bool VisitExit(const XMLDocument& )			{
return true;
}

virtual bool VisitEnter(const XMLElement& element, const XMLAttribute* attribute);
virtual bool VisitExit(const XMLElement& element);

virtual bool Visit(const XMLText& text);
virtual bool Visit(const XMLComment& comment);
virtual bool Visit(const XMLDeclaration& declaration);
virtual bool Visit(const XMLUnknown& unknown);


const char* CStr() const {
return _buffer.Mem();
}

int CStrSize() const {
return _buffer.Size();
}

void ClearBuffer() {
_buffer.Clear();
_buffer.Push(0);
}

protected:
virtual bool CompactMode(const XMLElement&)	{ return _compactMode; }


virtual void PrintSpace(int depth);
void Print(const char* format, ...);

void SealElementIfJustOpened();
bool _elementJustOpened;
DynArray< const char*, 10 > _stack;

private:
void PrintString(const char*, bool restrictedEntitySet);	

bool _firstElement;
FILE* _fp;
int _depth;
int _textDepth;
bool _processEntities;
bool _compactMode;

enum {
ENTITY_RANGE = 64,
BUF_SIZE = 200
};
bool _entityFlag[ENTITY_RANGE];
bool _restrictedEntityFlag[ENTITY_RANGE];

DynArray< char, 20 > _buffer;
};


}	
}   
}   

#if defined(_MSC_VER)
#   pragma warning(pop)
#endif

#endif 
