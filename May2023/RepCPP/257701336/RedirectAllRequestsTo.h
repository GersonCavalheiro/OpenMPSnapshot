

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

class AWS_S3_API RedirectAllRequestsTo
{
public:
RedirectAllRequestsTo();
RedirectAllRequestsTo(const Aws::Utils::Xml::XmlNode& xmlNode);
RedirectAllRequestsTo& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const Aws::String& GetHostName() const{ return m_hostName; }


inline void SetHostName(const Aws::String& value) { m_hostNameHasBeenSet = true; m_hostName = value; }


inline void SetHostName(Aws::String&& value) { m_hostNameHasBeenSet = true; m_hostName = std::move(value); }


inline void SetHostName(const char* value) { m_hostNameHasBeenSet = true; m_hostName.assign(value); }


inline RedirectAllRequestsTo& WithHostName(const Aws::String& value) { SetHostName(value); return *this;}


inline RedirectAllRequestsTo& WithHostName(Aws::String&& value) { SetHostName(std::move(value)); return *this;}


inline RedirectAllRequestsTo& WithHostName(const char* value) { SetHostName(value); return *this;}



inline const Protocol& GetProtocol() const{ return m_protocol; }


inline void SetProtocol(const Protocol& value) { m_protocolHasBeenSet = true; m_protocol = value; }


inline void SetProtocol(Protocol&& value) { m_protocolHasBeenSet = true; m_protocol = std::move(value); }


inline RedirectAllRequestsTo& WithProtocol(const Protocol& value) { SetProtocol(value); return *this;}


inline RedirectAllRequestsTo& WithProtocol(Protocol&& value) { SetProtocol(std::move(value)); return *this;}

private:

Aws::String m_hostName;
bool m_hostNameHasBeenSet;

Protocol m_protocol;
bool m_protocolHasBeenSet;
};

} 
} 
} 
