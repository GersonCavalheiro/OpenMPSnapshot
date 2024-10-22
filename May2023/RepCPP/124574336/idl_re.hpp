

#if !defined(BOOST_IDL_RE_HPP_BD62775D_1659_4684_872C_03C02543C9A5_INCLUDED)
#define BOOST_IDL_RE_HPP_BD62775D_1659_4684_872C_03C02543C9A5_INCLUDED

#include <cstdio>

#include <string>
#include <boost/config.hpp>

#if defined(BOOST_HAS_UNISTD_H)
#include <unistd.h>
#else
#include <io.h>
#endif

#include <boost/assert.hpp>
#include <boost/detail/workaround.hpp>

#include <boost/wave/token_ids.hpp>
#include <boost/wave/cpplexer/re2clex/aq.hpp>
#include <boost/wave/cpplexer/re2clex/scanner.hpp>
#include <boost/wave/cpplexer/cpplexer_exceptions.hpp>

#define BOOST_WAVE_BSIZE     196608

#define RE2C_ASSERT BOOST_ASSERT

#if defined(_MSC_VER) && !defined(__COMO__)
#pragma warning (disable: 4101)     
#pragma warning (disable: 4102)     
#endif

#define YYCTYPE   uchar
#define YYCURSOR  cursor
#define YYLIMIT   s->lim
#define YYMARKER  s->ptr
#define YYFILL(n) {cursor = fill(s, cursor);}

#define BOOST_WAVE_RET(i)    \
{ \
s->line += count_backslash_newlines(s, cursor); \
s->cur = cursor; \
return (i); \
} \




namespace boost {
namespace wave {
namespace idllexer {
namespace re2clex {

template<typename Iterator>
int
get_one_char(boost::wave::cpplexer::re2clex::Scanner<Iterator> *s)
{
using namespace boost::wave::cpplexer::re2clex;
RE2C_ASSERT(s->first <= s->act && s->act <= s->last);
if (s->act < s->last)
return *(s->act)++;
return -1;
}

template<typename Iterator>
std::ptrdiff_t
rewind_stream (boost::wave::cpplexer::re2clex::Scanner<Iterator> *s, int cnt)
{
s->act += cnt;
RE2C_ASSERT(s->first <= s->act && s->act <= s->last);
return s->act - s->first;
}

template<typename Iterator>
std::size_t
get_first_eol_offset(boost::wave::cpplexer::re2clex::Scanner<Iterator>* s)
{
if (!AQ_EMPTY(s->eol_offsets))
{
return s->eol_offsets->queue[s->eol_offsets->head];
}
else
{
return (unsigned int)-1;
}
}

template<typename Iterator>
void
adjust_eol_offsets(boost::wave::cpplexer::re2clex::Scanner<Iterator>* s,
std::size_t adjustment)
{
boost::wave::cpplexer::re2clex::aq_queue q;
std::size_t i;

if (!s->eol_offsets)
s->eol_offsets = boost::wave::cpplexer::re2clex::aq_create();

q = s->eol_offsets;

if (AQ_EMPTY(q))
return;

i = q->head;
while (i != q->tail)
{
if (adjustment > q->queue[i])
q->queue[i] = 0;
else
q->queue[i] -= adjustment;
++i;
if (i == q->max_size)
i = 0;
}
if (adjustment > q->queue[i])
q->queue[i] = 0;
else
q->queue[i] -= adjustment;
}

template<typename Iterator>
int
count_backslash_newlines(boost::wave::cpplexer::re2clex::Scanner<Iterator> *s,
boost::wave::cpplexer::re2clex::uchar *cursor)
{
using namespace boost::wave::cpplexer::re2clex;

std::size_t diff, offset;
int skipped = 0;


diff = cursor - s->bot;
offset = get_first_eol_offset(s);
while (offset <= diff && offset != (unsigned int)-1)
{
skipped++;
boost::wave::cpplexer::re2clex::aq_pop(s->eol_offsets);
offset = get_first_eol_offset(s);
}
return skipped;
}

bool
is_backslash(
boost::wave::cpplexer::re2clex::uchar *p,
boost::wave::cpplexer::re2clex::uchar *end, int &len);

template<typename Iterator>
boost::wave::cpplexer::re2clex::uchar *
fill(boost::wave::cpplexer::re2clex::Scanner<Iterator> *s,
boost::wave::cpplexer::re2clex::uchar *cursor)
{
using namespace std;    
using namespace boost::wave::cpplexer::re2clex;

if(!s->eof)
{
uchar* p;
std::ptrdiff_t cnt = s->tok - s->bot;
if(cnt)
{
memcpy(s->bot, s->tok, s->lim - s->tok);
s->tok = s->bot;
s->ptr -= cnt;
cursor -= cnt;
s->lim -= cnt;
adjust_eol_offsets(s, cnt);
}

if((s->top - s->lim) < BOOST_WAVE_BSIZE)
{
uchar *buf = (uchar*) malloc(((s->lim - s->bot) + BOOST_WAVE_BSIZE)*sizeof(uchar));
if (buf == 0)
{
using namespace std;      
if (0 != s->error_proc) {
(*s->error_proc)(s,
cpplexer::lexing_exception::unexpected_error,
"Out of memory!");
}
else
printf("Out of memory!\n");


*cursor = 0;
return cursor;
}

memcpy(buf, s->tok, s->lim - s->tok);
s->tok = buf;
s->ptr = &buf[s->ptr - s->bot];
cursor = &buf[cursor - s->bot];
s->lim = &buf[s->lim - s->bot];
s->top = &s->lim[BOOST_WAVE_BSIZE];
free(s->bot);
s->bot = buf;
}

cnt = std::distance(s->act, s->last);
if (cnt > BOOST_WAVE_BSIZE)
cnt = BOOST_WAVE_BSIZE;
uchar * dst = s->lim;
for (std::ptrdiff_t idx = 0; idx < cnt; ++idx)
{
*dst++ = *s->act++;
}
if (cnt != BOOST_WAVE_BSIZE) {
s->eof = &s->lim[cnt];
*(s->eof)++ = '\0';
}




for (p = s->lim; p < s->lim + cnt - 2; ++p)
{
int len = 0;
if (is_backslash(p, s->lim + cnt, len))
{
if (*(p+len) == '\n')
{
int offset = len + 1;
memmove(p, p + offset, s->lim + cnt - p - offset);
cnt -= offset;
--p;
aq_enqueue(s->eol_offsets, p - s->bot + 1);
}
else if (*(p+len) == '\r')
{
if (*(p+len+1) == '\n')
{
int offset = len + 2;
memmove(p, p + offset, s->lim + cnt - p - offset);
cnt -= offset;
--p;
}
else
{
int offset = len + 1;
memmove(p, p + offset, s->lim + cnt - p - offset);
cnt -= offset;
--p;
}
aq_enqueue(s->eol_offsets, p - s->bot + 1);
}
}
}




if (cnt >= 2)
{
uchar last = s->lim[cnt-1];
uchar last2 = s->lim[cnt-2];

if (last == '\\')
{
int next = get_one_char(s);

if (next == '\n')
{
--cnt; 
boost::wave::cpplexer::re2clex::aq_enqueue(s->eol_offsets,
cnt + (s->lim - s->bot));
}
else if (next == '\r')
{
int next2 = get_one_char(s);
if (next2 == '\n')
{
--cnt; 
}
else
{

rewind_stream(s, -1);
--cnt;
}
boost::wave::cpplexer::re2clex::aq_enqueue(s->eol_offsets,
cnt + (s->lim - s->bot));
}
else if (next != -1) 
{

rewind_stream(s, -1);
}
}

else if (last == '\r' && last2 == '\\')
{
int next = get_one_char(s);
if (next == '\n')
{
cnt -= 2; 
}
else
{

rewind_stream(s, -1);
cnt -= 2;
}
boost::wave::cpplexer::re2clex::aq_enqueue(s->eol_offsets,
cnt + (s->lim - s->bot));
}

else if (last == '\n' && last2 == '\\')
{
cnt -= 2;
boost::wave::cpplexer::re2clex::aq_enqueue(s->eol_offsets,
cnt + (s->lim - s->bot));
}
}

s->lim += cnt;
if (s->eof) 
{
s->eof = s->lim;
*(s->eof)++ = '\0';
}
}
return cursor;
}

template<typename Iterator>
BOOST_WAVE_DECL boost::wave::token_id scan(
boost::wave::cpplexer::re2clex::Scanner<Iterator> *s)
{

using namespace boost::wave::cpplexer::re2clex;

uchar *cursor = s->tok = s->cur;

#include "idl.inc"


} 

}   
}   
}   
}   

#undef RE2C_ASSERT

#endif 
