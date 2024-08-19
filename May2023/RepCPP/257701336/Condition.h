

#pragma once
#include <aws/s3/S3_EXPORTS.h>
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

class AWS_S3_API Condition
{
public:
Condition();
Condition(const Aws::Utils::Xml::XmlNode& xmlNode);
Condition& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const Aws::String& GetHttpErrorCodeReturnedEquals() const{ return m_httpErrorCodeReturnedEquals; }


inline void SetHttpErrorCodeReturnedEquals(const Aws::String& value) { m_httpErrorCodeReturnedEqualsHasBeenSet = true; m_httpErrorCodeReturnedEquals = value; }


inline void SetHttpErrorCodeReturnedEquals(Aws::String&& value) { m_httpErrorCodeReturnedEqualsHasBeenSet = true; m_httpErrorCodeReturnedEquals = std::move(value); }


inline void SetHttpErrorCodeReturnedEquals(const char* value) { m_httpErrorCodeReturnedEqualsHasBeenSet = true; m_httpErrorCodeReturnedEquals.assign(value); }


inline Condition& WithHttpErrorCodeReturnedEquals(const Aws::String& value) { SetHttpErrorCodeReturnedEquals(value); return *this;}


inline Condition& WithHttpErrorCodeReturnedEquals(Aws::String&& value) { SetHttpErrorCodeReturnedEquals(std::move(value)); return *this;}


inline Condition& WithHttpErrorCodeReturnedEquals(const char* value) { SetHttpErrorCodeReturnedEquals(value); return *this;}



inline const Aws::String& GetKeyPrefixEquals() const{ return m_keyPrefixEquals; }


inline void SetKeyPrefixEquals(const Aws::String& value) { m_keyPrefixEqualsHasBeenSet = true; m_keyPrefixEquals = value; }


inline void SetKeyPrefixEquals(Aws::String&& value) { m_keyPrefixEqualsHasBeenSet = true; m_keyPrefixEquals = std::move(value); }


inline void SetKeyPrefixEquals(const char* value) { m_keyPrefixEqualsHasBeenSet = true; m_keyPrefixEquals.assign(value); }


inline Condition& WithKeyPrefixEquals(const Aws::String& value) { SetKeyPrefixEquals(value); return *this;}


inline Condition& WithKeyPrefixEquals(Aws::String&& value) { SetKeyPrefixEquals(std::move(value)); return *this;}


inline Condition& WithKeyPrefixEquals(const char* value) { SetKeyPrefixEquals(value); return *this;}

private:

Aws::String m_httpErrorCodeReturnedEquals;
bool m_httpErrorCodeReturnedEqualsHasBeenSet;

Aws::String m_keyPrefixEquals;
bool m_keyPrefixEqualsHasBeenSet;
};

} 
} 
} 
