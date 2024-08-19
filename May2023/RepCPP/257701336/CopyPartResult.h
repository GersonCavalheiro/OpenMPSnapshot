

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/utils/DateTime.h>
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

class AWS_S3_API CopyPartResult
{
public:
CopyPartResult();
CopyPartResult(const Aws::Utils::Xml::XmlNode& xmlNode);
CopyPartResult& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const Aws::String& GetETag() const{ return m_eTag; }


inline void SetETag(const Aws::String& value) { m_eTagHasBeenSet = true; m_eTag = value; }


inline void SetETag(Aws::String&& value) { m_eTagHasBeenSet = true; m_eTag = std::move(value); }


inline void SetETag(const char* value) { m_eTagHasBeenSet = true; m_eTag.assign(value); }


inline CopyPartResult& WithETag(const Aws::String& value) { SetETag(value); return *this;}


inline CopyPartResult& WithETag(Aws::String&& value) { SetETag(std::move(value)); return *this;}


inline CopyPartResult& WithETag(const char* value) { SetETag(value); return *this;}



inline const Aws::Utils::DateTime& GetLastModified() const{ return m_lastModified; }


inline void SetLastModified(const Aws::Utils::DateTime& value) { m_lastModifiedHasBeenSet = true; m_lastModified = value; }


inline void SetLastModified(Aws::Utils::DateTime&& value) { m_lastModifiedHasBeenSet = true; m_lastModified = std::move(value); }


inline CopyPartResult& WithLastModified(const Aws::Utils::DateTime& value) { SetLastModified(value); return *this;}


inline CopyPartResult& WithLastModified(Aws::Utils::DateTime&& value) { SetLastModified(std::move(value)); return *this;}

private:

Aws::String m_eTag;
bool m_eTagHasBeenSet;

Aws::Utils::DateTime m_lastModified;
bool m_lastModifiedHasBeenSet;
};

} 
} 
} 
