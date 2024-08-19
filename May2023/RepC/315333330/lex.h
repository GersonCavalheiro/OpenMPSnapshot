#ifndef GO_LEX_H
#define GO_LEX_H
#include <mpfr.h>
#include "operator.h"
#include "go-linemap.h"
struct Unicode_range;
enum Keyword
{
KEYWORD_INVALID,	
KEYWORD_ASM,
KEYWORD_BREAK,
KEYWORD_CASE,
KEYWORD_CHAN,
KEYWORD_CONST,
KEYWORD_CONTINUE,
KEYWORD_DEFAULT,
KEYWORD_DEFER,
KEYWORD_ELSE,
KEYWORD_FALLTHROUGH,
KEYWORD_FOR,
KEYWORD_FUNC,
KEYWORD_GO,
KEYWORD_GOTO,
KEYWORD_IF,
KEYWORD_IMPORT,
KEYWORD_INTERFACE,
KEYWORD_MAP,
KEYWORD_PACKAGE,
KEYWORD_RANGE,
KEYWORD_RETURN,
KEYWORD_SELECT,
KEYWORD_STRUCT,
KEYWORD_SWITCH,
KEYWORD_TYPE,
KEYWORD_VAR
};
enum GoPragma
{
GOPRAGMA_NOINTERFACE = 1 << 0,	
GOPRAGMA_NOESCAPE = 1 << 1,		
GOPRAGMA_NORACE = 1 << 2,		
GOPRAGMA_NOSPLIT = 1 << 3,		
GOPRAGMA_NOINLINE = 1 << 4,		
GOPRAGMA_SYSTEMSTACK = 1 << 5,	
GOPRAGMA_NOWRITEBARRIER = 1 << 6,	
GOPRAGMA_NOWRITEBARRIERREC = 1 << 7,	
GOPRAGMA_CGOUNSAFEARGS = 1 << 8,	
GOPRAGMA_UINTPTRESCAPES = 1 << 9,	
GOPRAGMA_NOTINHEAP = 1 << 10		
};
class Token
{
public:
enum Classification
{
TOKEN_INVALID,
TOKEN_EOF,
TOKEN_KEYWORD,
TOKEN_IDENTIFIER,
TOKEN_STRING,
TOKEN_OPERATOR,
TOKEN_CHARACTER,
TOKEN_INTEGER,
TOKEN_FLOAT,
TOKEN_IMAGINARY
};
~Token();
Token(const Token&);
Token& operator=(const Token&);
Classification
classification() const
{ return this->classification_; }
static Token
make_invalid_token(Location location)
{ return Token(TOKEN_INVALID, location); }
static Token
make_eof_token(Location location)
{ return Token(TOKEN_EOF, location); }
static Token
make_keyword_token(Keyword keyword, Location location)
{
Token tok(TOKEN_KEYWORD, location);
tok.u_.keyword = keyword;
return tok;
}
static Token
make_identifier_token(const std::string& value, bool is_exported,
Location location)
{
Token tok(TOKEN_IDENTIFIER, location);
tok.u_.identifier_value.name = new std::string(value);
tok.u_.identifier_value.is_exported = is_exported;
return tok;
}
static Token
make_string_token(const std::string& value, Location location)
{
Token tok(TOKEN_STRING, location);
tok.u_.string_value = new std::string(value);
return tok;
}
static Token
make_operator_token(Operator op, Location location)
{
Token tok(TOKEN_OPERATOR, location);
tok.u_.op = op;
return tok;
}
static Token
make_character_token(mpz_t val, Location location)
{
Token tok(TOKEN_CHARACTER, location);
mpz_init(tok.u_.integer_value);
mpz_swap(tok.u_.integer_value, val);
return tok;
}
static Token
make_integer_token(mpz_t val, Location location)
{
Token tok(TOKEN_INTEGER, location);
mpz_init(tok.u_.integer_value);
mpz_swap(tok.u_.integer_value, val);
return tok;
}
static Token
make_float_token(mpfr_t val, Location location)
{
Token tok(TOKEN_FLOAT, location);
mpfr_init(tok.u_.float_value);
mpfr_swap(tok.u_.float_value, val);
return tok;
}
static Token
make_imaginary_token(mpfr_t val, Location location)
{
Token tok(TOKEN_IMAGINARY, location);
mpfr_init(tok.u_.float_value);
mpfr_swap(tok.u_.float_value, val);
return tok;
}
Location
location() const
{ return this->location_; }
bool
is_invalid() const
{ return this->classification_ == TOKEN_INVALID; }
bool
is_eof() const
{ return this->classification_ == TOKEN_EOF; }
Keyword
keyword() const
{
go_assert(this->classification_ == TOKEN_KEYWORD);
return this->u_.keyword;
}
bool
is_identifier() const
{ return this->classification_ == TOKEN_IDENTIFIER; }
const std::string&
identifier() const
{
go_assert(this->classification_ == TOKEN_IDENTIFIER);
return *this->u_.identifier_value.name;
}
bool
is_identifier_exported() const
{
go_assert(this->classification_ == TOKEN_IDENTIFIER);
return this->u_.identifier_value.is_exported;
}
bool
is_string() const
{
return this->classification_ == TOKEN_STRING;
}
std::string
string_value() const
{
go_assert(this->classification_ == TOKEN_STRING);
return *this->u_.string_value;
}
const mpz_t*
character_value() const
{
go_assert(this->classification_ == TOKEN_CHARACTER);
return &this->u_.integer_value;
}
const mpz_t*
integer_value() const
{
go_assert(this->classification_ == TOKEN_INTEGER);
return &this->u_.integer_value;
}
const mpfr_t*
float_value() const
{
go_assert(this->classification_ == TOKEN_FLOAT);
return &this->u_.float_value;
}
const mpfr_t*
imaginary_value() const
{
go_assert(this->classification_ == TOKEN_IMAGINARY);
return &this->u_.float_value;
}
Operator
op() const
{
go_assert(this->classification_ == TOKEN_OPERATOR);
return this->u_.op;
}
bool
is_keyword(Keyword keyword) const
{
return (this->classification_ == TOKEN_KEYWORD
&& this->u_.keyword == keyword);
}
bool
is_op(Operator op) const
{ return this->classification_ == TOKEN_OPERATOR && this->u_.op == op; }
void
print(FILE*) const;
private:
Token(Classification, Location);
void
clear();
Classification classification_;
union
{
Keyword keyword;
struct
{
std::string* name;
bool is_exported;
} identifier_value;
std::string* string_value;
mpz_t integer_value;
mpfr_t float_value;
Operator op;
} u_;
Location location_;
};
class Lex
{
public:
Lex(const char* input_file_name, FILE* input_file, Linemap *linemap);
~Lex();
Token
next_token();
const std::string&
extern_name() const
{ return this->extern_; }
unsigned int
get_and_clear_pragmas()
{
unsigned int ret = this->pragmas_;
this->pragmas_ = 0;
return ret;
}
struct Linkname
{
std::string ext_name;	
bool is_exported;		
Location loc;		
Linkname()
: ext_name(), is_exported(false), loc()
{ }
Linkname(const std::string& ext_name_a, bool is_exported_a, Location loc_a)
: ext_name(ext_name_a), is_exported(is_exported_a), loc(loc_a)
{ }
};
typedef std::map<std::string, Linkname> Linknames;
Linknames*
get_and_clear_linknames()
{
Linknames* ret = this->linknames_;
this->linknames_ = NULL;
return ret;
}
static bool
is_exported_name(const std::string& name);
static bool
is_invalid_identifier(const std::string& name);
static void
append_char(unsigned int v, bool is_charater, std::string* str,
Location);
static int
fetch_char(const char* str, unsigned int *value);
static bool
is_unicode_space(unsigned int c);
private:
ssize_t
get_line();
bool
require_line();
Location
location() const;
Location
earlier_location(int chars) const;
static bool
is_hex_digit(char);
static unsigned char
octal_value(char c)
{ return c - '0'; }
static unsigned
hex_val(char c);
Token
make_invalid_token()
{ return Token::make_invalid_token(this->location()); }
Token
make_eof_token()
{ return Token::make_eof_token(this->location()); }
Token
make_operator(Operator op, int chars)
{ return Token::make_operator_token(op, this->earlier_location(chars)); }
Token
gather_identifier();
static bool
could_be_exponent(const char*, const char*);
Token
gather_number();
Token
gather_character();
Token
gather_string();
Token
gather_raw_string();
const char*
advance_one_utf8_char(const char*, unsigned int*, bool*);
const char*
advance_one_char(const char*, bool, unsigned int*, bool*);
static bool
is_unicode_digit(unsigned int c);
static bool
is_unicode_letter(unsigned int c);
static bool
is_unicode_uppercase(unsigned int c);
static bool
is_in_unicode_range(unsigned int C, const Unicode_range* ranges,
size_t range_size);
Operator
three_character_operator(char, char, char);
Operator
two_character_operator(char, char);
Operator
one_character_operator(char);
bool
skip_c_comment(bool* found_newline);
void
skip_cpp_comment();
const char* input_file_name_;
FILE* input_file_;
Linemap* linemap_;
char* linebuf_;
size_t linebufsize_;
size_t linesize_;
size_t lineoff_;
size_t lineno_;
bool add_semi_at_eol_;
unsigned int pragmas_;
std::string extern_;
Linknames* linknames_;
};
#endif 
