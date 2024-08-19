

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/DateTime.h>
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

class AWS_S3_API Part
{
public:
Part();
Part(const Aws::Utils::Xml::XmlNode& xmlNode);
Part& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline int GetPartNumber() const{ return m_partNumber; }


inline void SetPartNumber(int value) { m_partNumberHasBeenSet = true; m_partNumber = value; }


inline Part& WithPartNumber(int value) { SetPartNumber(value); return *this;}



inline const Aws::Utils::DateTime& GetLastModified() const{ return m_lastModified; }


inline void SetLastModified(const Aws::Utils::DateTime& value) { m_lastModifiedHasBeenSet = true; m_lastModified = value; }


inline void SetLastModified(Aws::Utils::DateTime&& value) { m_lastModifiedHasBeenSet = true; m_lastModified = std::move(value); }


inline Part& WithLastModified(const Aws::Utils::DateTime& value) { SetLastModified(value); return *this;}


inline Part& WithLastModified(Aws::Utils::DateTime&& value) { SetLastModified(std::move(value)); return *this;}



inline const Aws::String& GetETag() const{ return m_eTag; }


inline void SetETag(const Aws::String& value) { m_eTagHasBeenSet = true; m_eTag = value; }


inline void SetETag(Aws::String&& value) { m_eTagHasBeenSet = true; m_eTag = std::move(value); }


inline void SetETag(const char* value) { m_eTagHasBeenSet = true; m_eTag.assign(value); }


inline Part& WithETag(const Aws::String& value) { SetETag(value); return *this;}


inline Part& WithETag(Aws::String&& value) { SetETag(std::move(value)); return *this;}


inline Part& WithETag(const char* value) { SetETag(value); return *this;}



inline long long GetSize() const{ return m_size; }


inline void SetSize(long long value) { m_sizeHasBeenSet = true; m_size = value; }


inline Part& WithSize(long long value) { SetSize(value); return *this;}

private:

int m_partNumber;
bool m_partNumberHasBeenSet;

Aws::Utils::DateTime m_lastModified;
bool m_lastModifiedHasBeenSet;

Aws::String m_eTag;
bool m_eTagHasBeenSet;

long long m_size;
bool m_sizeHasBeenSet;
};

} 
} 
} 
