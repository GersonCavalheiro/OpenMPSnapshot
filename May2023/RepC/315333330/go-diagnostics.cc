#include "go-diagnostics.h"
static std::string
mformat_value()
{
return std::string(xstrerror(errno));
}
static std::string
expand_format(const char* fmt)
{
std::stringstream ss;
for (const char* c = fmt; *c; ++c)
{
if (*c != '%')
{
ss << *c;
continue;
}
c++;
switch (*c)
{
case '\0':
{
go_unreachable();
}
case '%':
{
ss << "%";
break;
}
case 'm':
{
ss << mformat_value();
break;
}
case '<':
{
ss << go_open_quote();
break;
}
case '>':
{
ss << go_close_quote();
break;
}
case 'q':
{
ss << go_open_quote();
c++;
if (*c == 'm')
{
ss << mformat_value();
}
else
{
ss << "%" << *c;
}
ss << go_close_quote();
break;
}
default:
{
ss << "%" << *c;
}
}
}
return ss.str();
}
static std::string
expand_message(const char* fmt, va_list ap) GO_ATTRIBUTE_GCC_DIAG(1,0);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-attribute=format"
static std::string
expand_message(const char* fmt, va_list ap)
{
char* mbuf = 0;
std::string expanded_fmt = expand_format(fmt);
int nwr = vasprintf(&mbuf, expanded_fmt.c_str(), ap);
if (nwr == -1)
{
go_be_error_at(Linemap::unknown_location(),
"memory allocation failed in vasprintf");
go_assert(0);
}
std::string rval = std::string(mbuf);
free(mbuf);
return rval;
}
#pragma GCC diagnostic pop
static const char* cached_open_quote = NULL;
static const char* cached_close_quote = NULL;
const char*
go_open_quote()
{
if (cached_open_quote == NULL)
go_be_get_quotechars(&cached_open_quote, &cached_close_quote);
return cached_open_quote;
}
const char*
go_close_quote()
{
if (cached_close_quote == NULL)
go_be_get_quotechars(&cached_open_quote, &cached_close_quote);
return cached_close_quote;
}
void
go_error_at(const Location location, const char* fmt, ...)
{
va_list ap;
va_start(ap, fmt);
go_be_error_at(location, expand_message(fmt, ap));
va_end(ap);
}
void
go_warning_at(const Location location, int opt, const char* fmt, ...)
{
va_list ap;
va_start(ap, fmt);
go_be_warning_at(location, opt, expand_message(fmt, ap));
va_end(ap);
}
void
go_fatal_error(const Location location, const char* fmt, ...)
{
va_list ap;
va_start(ap, fmt);
go_be_fatal_error(location, expand_message(fmt, ap));
va_end(ap);
}
void
go_inform(const Location location, const char* fmt, ...)
{
va_list ap;
va_start(ap, fmt);
go_be_inform(location, expand_message(fmt, ap));
va_end(ap);
}
