

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

class AWS_S3_API CompletedPart
{
public:
CompletedPart();
CompletedPart(const Aws::Utils::Xml::XmlNode& xmlNode);
CompletedPart& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const Aws::String& GetETag() const{ return m_eTag; }


inline void SetETag(const Aws::String& value) { m_eTagHasBeenSet = true; m_eTag = value; }


inline void SetETag(Aws::String&& value) { m_eTagHasBeenSet = true; m_eTag = std::move(value); }


inline void SetETag(const char* value) { m_eTagHasBeenSet = true; m_eTag.assign(value); }


inline CompletedPart& WithETag(const Aws::String& value) { SetETag(value); return *this;}


inline CompletedPart& WithETag(Aws::String&& value) { SetETag(std::move(value)); return *this;}


inline CompletedPart& WithETag(const char* value) { SetETag(value); return *this;}



inline int GetPartNumber() const{ return m_partNumber; }


inline void SetPartNumber(int value) { m_partNumberHasBeenSet = true; m_partNumber = value; }


inline CompletedPart& WithPartNumber(int value) { SetPartNumber(value); return *this;}

private:

Aws::String m_eTag;
bool m_eTagHasBeenSet;

int m_partNumber;
bool m_partNumberHasBeenSet;
};

} 
} 
} 
