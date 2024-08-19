

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSVector.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <utility>

namespace Aws
{
namespace Utils
{
namespace Xml
{
class XmlNode;
} 
} 
namespace S3
{
namespace Model
{

class AWS_S3_API CORSRule
{
public:
CORSRule();
CORSRule(const Aws::Utils::Xml::XmlNode& xmlNode);
CORSRule& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const Aws::Vector<Aws::String>& GetAllowedHeaders() const{ return m_allowedHeaders; }


inline void SetAllowedHeaders(const Aws::Vector<Aws::String>& value) { m_allowedHeadersHasBeenSet = true; m_allowedHeaders = value; }


inline void SetAllowedHeaders(Aws::Vector<Aws::String>&& value) { m_allowedHeadersHasBeenSet = true; m_allowedHeaders = std::move(value); }


inline CORSRule& WithAllowedHeaders(const Aws::Vector<Aws::String>& value) { SetAllowedHeaders(value); return *this;}


inline CORSRule& WithAllowedHeaders(Aws::Vector<Aws::String>&& value) { SetAllowedHeaders(std::move(value)); return *this;}


inline CORSRule& AddAllowedHeaders(const Aws::String& value) { m_allowedHeadersHasBeenSet = true; m_allowedHeaders.push_back(value); return *this; }


inline CORSRule& AddAllowedHeaders(Aws::String&& value) { m_allowedHeadersHasBeenSet = true; m_allowedHeaders.push_back(std::move(value)); return *this; }


inline CORSRule& AddAllowedHeaders(const char* value) { m_allowedHeadersHasBeenSet = true; m_allowedHeaders.push_back(value); return *this; }



inline const Aws::Vector<Aws::String>& GetAllowedMethods() const{ return m_allowedMethods; }


inline void SetAllowedMethods(const Aws::Vector<Aws::String>& value) { m_allowedMethodsHasBeenSet = true; m_allowedMethods = value; }


inline void SetAllowedMethods(Aws::Vector<Aws::String>&& value) { m_allowedMethodsHasBeenSet = true; m_allowedMethods = std::move(value); }


inline CORSRule& WithAllowedMethods(const Aws::Vector<Aws::String>& value) { SetAllowedMethods(value); return *this;}


inline CORSRule& WithAllowedMethods(Aws::Vector<Aws::String>&& value) { SetAllowedMethods(std::move(value)); return *this;}


inline CORSRule& AddAllowedMethods(const Aws::String& value) { m_allowedMethodsHasBeenSet = true; m_allowedMethods.push_back(value); return *this; }


inline CORSRule& AddAllowedMethods(Aws::String&& value) { m_allowedMethodsHasBeenSet = true; m_allowedMethods.push_back(std::move(value)); return *this; }


inline CORSRule& AddAllowedMethods(const char* value) { m_allowedMethodsHasBeenSet = true; m_allowedMethods.push_back(value); return *this; }



inline const Aws::Vector<Aws::String>& GetAllowedOrigins() const{ return m_allowedOrigins; }


inline void SetAllowedOrigins(const Aws::Vector<Aws::String>& value) { m_allowedOriginsHasBeenSet = true; m_allowedOrigins = value; }


inline void SetAllowedOrigins(Aws::Vector<Aws::String>&& value) { m_allowedOriginsHasBeenSet = true; m_allowedOrigins = std::move(value); }


inline CORSRule& WithAllowedOrigins(const Aws::Vector<Aws::String>& value) { SetAllowedOrigins(value); return *this;}


inline CORSRule& WithAllowedOrigins(Aws::Vector<Aws::String>&& value) { SetAllowedOrigins(std::move(value)); return *this;}


inline CORSRule& AddAllowedOrigins(const Aws::String& value) { m_allowedOriginsHasBeenSet = true; m_allowedOrigins.push_back(value); return *this; }


inline CORSRule& AddAllowedOrigins(Aws::String&& value) { m_allowedOriginsHasBeenSet = true; m_allowedOrigins.push_back(std::move(value)); return *this; }


inline CORSRule& AddAllowedOrigins(const char* value) { m_allowedOriginsHasBeenSet = true; m_allowedOrigins.push_back(value); return *this; }



inline const Aws::Vector<Aws::String>& GetExposeHeaders() const{ return m_exposeHeaders; }


inline void SetExposeHeaders(const Aws::Vector<Aws::String>& value) { m_exposeHeadersHasBeenSet = true; m_exposeHeaders = value; }


inline void SetExposeHeaders(Aws::Vector<Aws::String>&& value) { m_exposeHeadersHasBeenSet = true; m_exposeHeaders = std::move(value); }


inline CORSRule& WithExposeHeaders(const Aws::Vector<Aws::String>& value) { SetExposeHeaders(value); return *this;}


inline CORSRule& WithExposeHeaders(Aws::Vector<Aws::String>&& value) { SetExposeHeaders(std::move(value)); return *this;}


inline CORSRule& AddExposeHeaders(const Aws::String& value) { m_exposeHeadersHasBeenSet = true; m_exposeHeaders.push_back(value); return *this; }


inline CORSRule& AddExposeHeaders(Aws::String&& value) { m_exposeHeadersHasBeenSet = true; m_exposeHeaders.push_back(std::move(value)); return *this; }


inline CORSRule& AddExposeHeaders(const char* value) { m_exposeHeadersHasBeenSet = true; m_exposeHeaders.push_back(value); return *this; }



inline int GetMaxAgeSeconds() const{ return m_maxAgeSeconds; }


inline void SetMaxAgeSeconds(int value) { m_maxAgeSecondsHasBeenSet = true; m_maxAgeSeconds = value; }


inline CORSRule& WithMaxAgeSeconds(int value) { SetMaxAgeSeconds(value); return *this;}

private:

Aws::Vector<Aws::String> m_allowedHeaders;
bool m_allowedHeadersHasBeenSet;

Aws::Vector<Aws::String> m_allowedMethods;
bool m_allowedMethodsHasBeenSet;

Aws::Vector<Aws::String> m_allowedOrigins;
bool m_allowedOriginsHasBeenSet;

Aws::Vector<Aws::String> m_exposeHeaders;
bool m_exposeHeadersHasBeenSet;

int m_maxAgeSeconds;
bool m_maxAgeSecondsHasBeenSet;
};

} 
} 
} 
