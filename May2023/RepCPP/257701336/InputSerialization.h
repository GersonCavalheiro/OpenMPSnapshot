

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/model/CSVInput.h>
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


class AWS_S3_API InputSerialization
{
public:
InputSerialization();
InputSerialization(const Aws::Utils::Xml::XmlNode& xmlNode);
InputSerialization& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const CSVInput& GetCSV() const{ return m_cSV; }


inline void SetCSV(const CSVInput& value) { m_cSVHasBeenSet = true; m_cSV = value; }


inline void SetCSV(CSVInput&& value) { m_cSVHasBeenSet = true; m_cSV = std::move(value); }


inline InputSerialization& WithCSV(const CSVInput& value) { SetCSV(value); return *this;}


inline InputSerialization& WithCSV(CSVInput&& value) { SetCSV(std::move(value)); return *this;}

private:

CSVInput m_cSV;
bool m_cSVHasBeenSet;
};

} 
} 
} 
