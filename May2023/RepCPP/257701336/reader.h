
#ifndef CPPTL_JSON_READER_H_INCLUDED
#define CPPTL_JSON_READER_H_INCLUDED

#if !defined(JSON_IS_AMALGAMATION)
#include "features.h"
#include "value.h"
#endif 
#include <deque>
#include <iosfwd>
#include <stack>
#include <string>
#include <istream>

#if defined(JSONCPP_DISABLE_DLL_INTERFACE_WARNING)
#pragma warning(push)
#pragma warning(disable : 4251)
#endif 

#pragma pack(push, 8)

namespace Json {


class JSONCPP_DEPRECATED("Use CharReader and CharReaderBuilder instead") JSON_API Reader {
public:
typedef char Char;
typedef const Char* Location;


struct StructuredError {
ptrdiff_t offset_start;
ptrdiff_t offset_limit;
JSONCPP_STRING message;
};


Reader();


Reader(const Features& features);


bool
parse(const std::string& document, Value& root, bool collectComments = true);


bool parse(const char* beginDoc,
const char* endDoc,
Value& root,
bool collectComments = true);

bool parse(JSONCPP_ISTREAM& is, Value& root, bool collectComments = true);


JSONCPP_DEPRECATED("Use getFormattedErrorMessages() instead.")
JSONCPP_STRING getFormatedErrorMessages() const;


JSONCPP_STRING getFormattedErrorMessages() const;


std::vector<StructuredError> getStructuredErrors() const;


bool pushError(const Value& value, const JSONCPP_STRING& message);


bool pushError(const Value& value, const JSONCPP_STRING& message, const Value& extra);


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
JSONCPP_STRING message_;
Location extra_;
};

typedef std::deque<ErrorInfo> Errors;

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
bool decodeString(Token& token, JSONCPP_STRING& decoded);
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
bool addError(const JSONCPP_STRING& message, Token& token, Location extra = 0);
bool recoverFromError(TokenType skipUntilToken);
bool addErrorAndRecover(const JSONCPP_STRING& message,
Token& token,
TokenType skipUntilToken);
void skipUntilSpace();
Value& currentValue();
Char getNextChar();
void
getLocationLineAndColumn(Location location, int& line, int& column) const;
JSONCPP_STRING getLocationLineAndColumn(Location location) const;
void addComment(Location begin, Location end, CommentPlacement placement);
void skipCommentTokens(Token& token);

static bool containsNewLine(Location begin, Location end);
static JSONCPP_STRING normalizeEOL(Location begin, Location end);

typedef std::stack<Value*> Nodes;
Nodes nodes_;
Errors errors_;
JSONCPP_STRING document_;
Location begin_;
Location end_;
Location current_;
Location lastValueEnd_;
Value* lastValue_;
JSONCPP_STRING commentsBefore_;
Features features_;
bool collectComments_;
};  


class JSON_API CharReader {
public:
virtual ~CharReader() {}

virtual bool parse(
char const* beginDoc, char const* endDoc,
Value* root, JSONCPP_STRING* errs) = 0;

class JSON_API Factory {
public:
virtual ~Factory() {}

virtual CharReader* newCharReader() const = 0;
};  
};  


class JSON_API CharReaderBuilder : public CharReader::Factory {
public:

Json::Value settings_;

CharReaderBuilder();
~CharReaderBuilder() JSONCPP_OVERRIDE;

CharReader* newCharReader() const JSONCPP_OVERRIDE;


bool validate(Json::Value* invalid) const;


Value& operator[](JSONCPP_STRING key);


static void setDefaults(Json::Value* settings);

static void strictMode(Json::Value* settings);
};


bool JSON_API parseFromStream(
CharReader::Factory const&,
JSONCPP_ISTREAM&,
Value* root, std::string* errs);


JSON_API JSONCPP_ISTREAM& operator>>(JSONCPP_ISTREAM&, Value&);

} 

#pragma pack(pop)

#if defined(JSONCPP_DISABLE_DLL_INTERFACE_WARNING)
#pragma warning(pop)
#endif 

#endif 
