
#ifndef JSON_WRITER_H_INCLUDED
#define JSON_WRITER_H_INCLUDED

#if !defined(JSON_IS_AMALGAMATION)
#include "value.h"
#endif 
#include <vector>
#include <string>
#include <ostream>

#if defined(JSONCPP_DISABLE_DLL_INTERFACE_WARNING) && defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4251)
#endif 

#pragma pack(push, 8)

namespace Json {

class Value;


class JSON_API StreamWriter {
protected:
JSONCPP_OSTREAM* sout_;  
public:
StreamWriter();
virtual ~StreamWriter();

virtual int write(Value const& root, JSONCPP_OSTREAM* sout) = 0;


class JSON_API Factory {
public:
virtual ~Factory();

virtual StreamWriter* newStreamWriter() const = 0;
};  
};  


JSONCPP_STRING JSON_API writeString(StreamWriter::Factory const& factory, Value const& root);



class JSON_API StreamWriterBuilder : public StreamWriter::Factory {
public:

Json::Value settings_;

StreamWriterBuilder();
~StreamWriterBuilder() JSONCPP_OVERRIDE;


StreamWriter* newStreamWriter() const JSONCPP_OVERRIDE;


bool validate(Json::Value* invalid) const;

Value& operator[](JSONCPP_STRING key);


static void setDefaults(Json::Value* settings);
};


class JSONCPP_DEPRECATED("Use StreamWriter instead") JSON_API Writer {
public:
virtual ~Writer();

virtual JSONCPP_STRING write(const Value& root) = 0;
};


#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable:4996) 
#endif
class JSONCPP_DEPRECATED("Use StreamWriterBuilder instead") JSON_API FastWriter : public Writer {
public:
FastWriter();
~FastWriter() JSONCPP_OVERRIDE {}

void enableYAMLCompatibility();


void dropNullPlaceholders();

void omitEndingLineFeed();

public: 
JSONCPP_STRING write(const Value& root) JSONCPP_OVERRIDE;

private:
void writeValue(const Value& value);

JSONCPP_STRING document_;
bool yamlCompatibilityEnabled_;
bool dropNullPlaceholders_;
bool omitEndingLineFeed_;
};
#if defined(_MSC_VER)
#pragma warning(pop)
#endif


#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable:4996) 
#endif
class JSONCPP_DEPRECATED("Use StreamWriterBuilder instead") JSON_API StyledWriter : public Writer {
public:
StyledWriter();
~StyledWriter() JSONCPP_OVERRIDE {}

public: 

JSONCPP_STRING write(const Value& root) JSONCPP_OVERRIDE;

private:
void writeValue(const Value& value);
void writeArrayValue(const Value& value);
bool isMultilineArray(const Value& value);
void pushValue(const JSONCPP_STRING& value);
void writeIndent();
void writeWithIndent(const JSONCPP_STRING& value);
void indent();
void unindent();
void writeCommentBeforeValue(const Value& root);
void writeCommentAfterValueOnSameLine(const Value& root);
bool hasCommentForValue(const Value& value);
static JSONCPP_STRING normalizeEOL(const JSONCPP_STRING& text);

typedef std::vector<JSONCPP_STRING> ChildValues;

ChildValues childValues_;
JSONCPP_STRING document_;
JSONCPP_STRING indentString_;
unsigned int rightMargin_;
unsigned int indentSize_;
bool addChildValues_;
};
#if defined(_MSC_VER)
#pragma warning(pop)
#endif


#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable:4996) 
#endif
class JSONCPP_DEPRECATED("Use StreamWriterBuilder instead") JSON_API StyledStreamWriter {
public:

StyledStreamWriter(JSONCPP_STRING indentation = "\t");
~StyledStreamWriter() {}

public:

void write(JSONCPP_OSTREAM& out, const Value& root);

private:
void writeValue(const Value& value);
void writeArrayValue(const Value& value);
bool isMultilineArray(const Value& value);
void pushValue(const JSONCPP_STRING& value);
void writeIndent();
void writeWithIndent(const JSONCPP_STRING& value);
void indent();
void unindent();
void writeCommentBeforeValue(const Value& root);
void writeCommentAfterValueOnSameLine(const Value& root);
bool hasCommentForValue(const Value& value);
static JSONCPP_STRING normalizeEOL(const JSONCPP_STRING& text);

typedef std::vector<JSONCPP_STRING> ChildValues;

ChildValues childValues_;
JSONCPP_OSTREAM* document_;
JSONCPP_STRING indentString_;
unsigned int rightMargin_;
JSONCPP_STRING indentation_;
bool addChildValues_ : 1;
bool indented_ : 1;
};
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(JSON_HAS_INT64)
JSONCPP_STRING JSON_API valueToString(Int value);
JSONCPP_STRING JSON_API valueToString(UInt value);
#endif 
JSONCPP_STRING JSON_API valueToString(LargestInt value);
JSONCPP_STRING JSON_API valueToString(LargestUInt value);
JSONCPP_STRING JSON_API valueToString(double value);
JSONCPP_STRING JSON_API valueToString(bool value);
JSONCPP_STRING JSON_API valueToQuotedString(const char* value);

JSON_API JSONCPP_OSTREAM& operator<<(JSONCPP_OSTREAM&, const Value& root);

} 

#pragma pack(pop)

#if defined(JSONCPP_DISABLE_DLL_INTERFACE_WARNING)
#pragma warning(pop)
#endif 

#endif 
