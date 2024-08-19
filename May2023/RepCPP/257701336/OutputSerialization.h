

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/model/CSVOutput.h>
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


class AWS_S3_API OutputSerialization
{
public:
OutputSerialization();
OutputSerialization(const Aws::Utils::Xml::XmlNode& xmlNode);
OutputSerialization& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const CSVOutput& GetCSV() const{ return m_cSV; }


inline void SetCSV(const CSVOutput& value) { m_cSVHasBeenSet = true; m_cSV = value; }


inline void SetCSV(CSVOutput&& value) { m_cSVHasBeenSet = true; m_cSV = std::move(value); }


inline OutputSerialization& WithCSV(const CSVOutput& value) { SetCSV(value); return *this;}


inline OutputSerialization& WithCSV(CSVOutput&& value) { SetCSV(std::move(value)); return *this;}

private:

CSVOutput m_cSV;
bool m_cSVHasBeenSet;
};

} 
} 
} 
