

#pragma once

#include <aws/core/Core_EXPORTS.h>

#include <aws/core/http/Scheme.h>
#include <aws/core/utils/memory/stl/AWSMap.h>
#include <aws/core/utils/memory/stl/AWSString.h>

#include <stdint.h>

namespace Aws
{
namespace Http
{
extern AWS_CORE_API const char* SEPARATOR;
static const uint16_t HTTP_DEFAULT_PORT = 80;
static const uint16_t HTTPS_DEFAULT_PORT = 443;

typedef Aws::MultiMap<Aws::String, Aws::String> QueryStringParameterCollection;


class AWS_CORE_API URI
{
public:

URI();

URI(const Aws::String&);

URI(const char*);

URI& operator = (const Aws::String&);
URI& operator = (const char*);

bool operator == (const URI&) const;
bool operator == (const Aws::String&) const;
bool operator == (const char*) const;
bool operator != (const URI&) const;
bool operator != (const Aws::String&) const;
bool operator != (const char*) const;


inline Scheme GetScheme() const { return m_scheme; }


void SetScheme(Scheme value);


inline const Aws::String& GetAuthority() const { return m_authority; }


inline void SetAuthority(const Aws::String& value) { m_authority = value; }


inline uint16_t GetPort() const { return m_port; }


inline void SetPort(uint16_t value) { m_port = value; }


inline const Aws::String& GetPath() const { return m_path; }


inline Aws::String GetURLEncodedPath() const { return URLEncodePath(m_path); }


void SetPath(const Aws::String& value);


inline const Aws::String& GetQueryString() const { return m_queryString; }


void SetQueryString(const Aws::String& str);

Aws::String GetFormParameters() const;


void CanonicalizeQueryString();


QueryStringParameterCollection GetQueryStringParameters(bool decode = true) const;


void AddQueryStringParameter(const char* key, const Aws::String& value);


Aws::String GetURIString(bool includeQueryString = true) const;


static Aws::String URLEncodePath(const Aws::String& path);


static Aws::String URLEncodePathRFC3986(const Aws::String& path);

private:
void ParseURIParts(const Aws::String& uri);
void ExtractAndSetScheme(const Aws::String& uri);
void ExtractAndSetAuthority(const Aws::String& uri);
void ExtractAndSetPort(const Aws::String& uri);
void ExtractAndSetPath(const Aws::String& uri);
void ExtractAndSetQueryString(const Aws::String& uri);
bool CompareURIParts(const URI& other) const;

Scheme m_scheme;
Aws::String m_authority;
uint16_t m_port;
Aws::String m_path;
Aws::String m_queryString;
};

} 
} 

