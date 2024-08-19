

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

class AWS_S3_API Bucket
{
public:
Bucket();
Bucket(const Aws::Utils::Xml::XmlNode& xmlNode);
Bucket& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const Aws::String& GetName() const{ return m_name; }


inline void SetName(const Aws::String& value) { m_nameHasBeenSet = true; m_name = value; }


inline void SetName(Aws::String&& value) { m_nameHasBeenSet = true; m_name = std::move(value); }


inline void SetName(const char* value) { m_nameHasBeenSet = true; m_name.assign(value); }


inline Bucket& WithName(const Aws::String& value) { SetName(value); return *this;}


inline Bucket& WithName(Aws::String&& value) { SetName(std::move(value)); return *this;}


inline Bucket& WithName(const char* value) { SetName(value); return *this;}



inline const Aws::Utils::DateTime& GetCreationDate() const{ return m_creationDate; }


inline void SetCreationDate(const Aws::Utils::DateTime& value) { m_creationDateHasBeenSet = true; m_creationDate = value; }


inline void SetCreationDate(Aws::Utils::DateTime&& value) { m_creationDateHasBeenSet = true; m_creationDate = std::move(value); }


inline Bucket& WithCreationDate(const Aws::Utils::DateTime& value) { SetCreationDate(value); return *this;}


inline Bucket& WithCreationDate(Aws::Utils::DateTime&& value) { SetCreationDate(std::move(value)); return *this;}

private:

Aws::String m_name;
bool m_nameHasBeenSet;

Aws::Utils::DateTime m_creationDate;
bool m_creationDateHasBeenSet;
};

} 
} 
} 
