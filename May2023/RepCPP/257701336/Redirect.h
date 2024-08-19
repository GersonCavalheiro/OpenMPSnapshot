

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/s3/model/Protocol.h>
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

class AWS_S3_API Redirect
{
public:
Redirect();
Redirect(const Aws::Utils::Xml::XmlNode& xmlNode);
Redirect& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const Aws::String& GetHostName() const{ return m_hostName; }


inline void SetHostName(const Aws::String& value) { m_hostNameHasBeenSet = true; m_hostName = value; }


inline void SetHostName(Aws::String&& value) { m_hostNameHasBeenSet = true; m_hostName = std::move(value); }


inline void SetHostName(const char* value) { m_hostNameHasBeenSet = true; m_hostName.assign(value); }


inline Redirect& WithHostName(const Aws::String& value) { SetHostName(value); return *this;}


inline Redirect& WithHostName(Aws::String&& value) { SetHostName(std::move(value)); return *this;}


inline Redirect& WithHostName(const char* value) { SetHostName(value); return *this;}



inline const Aws::String& GetHttpRedirectCode() const{ return m_httpRedirectCode; }


inline void SetHttpRedirectCode(const Aws::String& value) { m_httpRedirectCodeHasBeenSet = true; m_httpRedirectCode = value; }


inline void SetHttpRedirectCode(Aws::String&& value) { m_httpRedirectCodeHasBeenSet = true; m_httpRedirectCode = std::move(value); }


inline void SetHttpRedirectCode(const char* value) { m_httpRedirectCodeHasBeenSet = true; m_httpRedirectCode.assign(value); }


inline Redirect& WithHttpRedirectCode(const Aws::String& value) { SetHttpRedirectCode(value); return *this;}


inline Redirect& WithHttpRedirectCode(Aws::String&& value) { SetHttpRedirectCode(std::move(value)); return *this;}


inline Redirect& WithHttpRedirectCode(const char* value) { SetHttpRedirectCode(value); return *this;}



inline const Protocol& GetProtocol() const{ return m_protocol; }


inline void SetProtocol(const Protocol& value) { m_protocolHasBeenSet = true; m_protocol = value; }


inline void SetProtocol(Protocol&& value) { m_protocolHasBeenSet = true; m_protocol = std::move(value); }


inline Redirect& WithProtocol(const Protocol& value) { SetProtocol(value); return *this;}


inline Redirect& WithProtocol(Protocol&& value) { SetProtocol(std::move(value)); return *this;}



inline const Aws::String& GetReplaceKeyPrefixWith() const{ return m_replaceKeyPrefixWith; }


inline void SetReplaceKeyPrefixWith(const Aws::String& value) { m_replaceKeyPrefixWithHasBeenSet = true; m_replaceKeyPrefixWith = value; }


inline void SetReplaceKeyPrefixWith(Aws::String&& value) { m_replaceKeyPrefixWithHasBeenSet = true; m_replaceKeyPrefixWith = std::move(value); }


inline void SetReplaceKeyPrefixWith(const char* value) { m_replaceKeyPrefixWithHasBeenSet = true; m_replaceKeyPrefixWith.assign(value); }


inline Redirect& WithReplaceKeyPrefixWith(const Aws::String& value) { SetReplaceKeyPrefixWith(value); return *this;}


inline Redirect& WithReplaceKeyPrefixWith(Aws::String&& value) { SetReplaceKeyPrefixWith(std::move(value)); return *this;}


inline Redirect& WithReplaceKeyPrefixWith(const char* value) { SetReplaceKeyPrefixWith(value); return *this;}



inline const Aws::String& GetReplaceKeyWith() const{ return m_replaceKeyWith; }


inline void SetReplaceKeyWith(const Aws::String& value) { m_replaceKeyWithHasBeenSet = true; m_replaceKeyWith = value; }


inline void SetReplaceKeyWith(Aws::String&& value) { m_replaceKeyWithHasBeenSet = true; m_replaceKeyWith = std::move(value); }


inline void SetReplaceKeyWith(const char* value) { m_replaceKeyWithHasBeenSet = true; m_replaceKeyWith.assign(value); }


inline Redirect& WithReplaceKeyWith(const Aws::String& value) { SetReplaceKeyWith(value); return *this;}


inline Redirect& WithReplaceKeyWith(Aws::String&& value) { SetReplaceKeyWith(std::move(value)); return *this;}


inline Redirect& WithReplaceKeyWith(const char* value) { SetReplaceKeyWith(value); return *this;}

private:

Aws::String m_hostName;
bool m_hostNameHasBeenSet;

Aws::String m_httpRedirectCode;
bool m_httpRedirectCodeHasBeenSet;

Protocol m_protocol;
bool m_protocolHasBeenSet;

Aws::String m_replaceKeyPrefixWith;
bool m_replaceKeyPrefixWithHasBeenSet;

Aws::String m_replaceKeyWith;
bool m_replaceKeyWithHasBeenSet;
};

} 
} 
} 
