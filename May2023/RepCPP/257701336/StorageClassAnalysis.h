

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/model/StorageClassAnalysisDataExport.h>
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

class AWS_S3_API StorageClassAnalysis
{
public:
StorageClassAnalysis();
StorageClassAnalysis(const Aws::Utils::Xml::XmlNode& xmlNode);
StorageClassAnalysis& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const StorageClassAnalysisDataExport& GetDataExport() const{ return m_dataExport; }


inline void SetDataExport(const StorageClassAnalysisDataExport& value) { m_dataExportHasBeenSet = true; m_dataExport = value; }


inline void SetDataExport(StorageClassAnalysisDataExport&& value) { m_dataExportHasBeenSet = true; m_dataExport = std::move(value); }


inline StorageClassAnalysis& WithDataExport(const StorageClassAnalysisDataExport& value) { SetDataExport(value); return *this;}


inline StorageClassAnalysis& WithDataExport(StorageClassAnalysisDataExport&& value) { SetDataExport(std::move(value)); return *this;}

private:

StorageClassAnalysisDataExport m_dataExport;
bool m_dataExportHasBeenSet;
};

} 
} 
} 
