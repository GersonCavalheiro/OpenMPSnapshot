

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/model/StorageClassAnalysisSchemaVersion.h>
#include <aws/s3/model/AnalyticsExportDestination.h>
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

class AWS_S3_API StorageClassAnalysisDataExport
{
public:
StorageClassAnalysisDataExport();
StorageClassAnalysisDataExport(const Aws::Utils::Xml::XmlNode& xmlNode);
StorageClassAnalysisDataExport& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const StorageClassAnalysisSchemaVersion& GetOutputSchemaVersion() const{ return m_outputSchemaVersion; }


inline void SetOutputSchemaVersion(const StorageClassAnalysisSchemaVersion& value) { m_outputSchemaVersionHasBeenSet = true; m_outputSchemaVersion = value; }


inline void SetOutputSchemaVersion(StorageClassAnalysisSchemaVersion&& value) { m_outputSchemaVersionHasBeenSet = true; m_outputSchemaVersion = std::move(value); }


inline StorageClassAnalysisDataExport& WithOutputSchemaVersion(const StorageClassAnalysisSchemaVersion& value) { SetOutputSchemaVersion(value); return *this;}


inline StorageClassAnalysisDataExport& WithOutputSchemaVersion(StorageClassAnalysisSchemaVersion&& value) { SetOutputSchemaVersion(std::move(value)); return *this;}



inline const AnalyticsExportDestination& GetDestination() const{ return m_destination; }


inline void SetDestination(const AnalyticsExportDestination& value) { m_destinationHasBeenSet = true; m_destination = value; }


inline void SetDestination(AnalyticsExportDestination&& value) { m_destinationHasBeenSet = true; m_destination = std::move(value); }


inline StorageClassAnalysisDataExport& WithDestination(const AnalyticsExportDestination& value) { SetDestination(value); return *this;}


inline StorageClassAnalysisDataExport& WithDestination(AnalyticsExportDestination&& value) { SetDestination(std::move(value)); return *this;}

private:

StorageClassAnalysisSchemaVersion m_outputSchemaVersion;
bool m_outputSchemaVersionHasBeenSet;

AnalyticsExportDestination m_destination;
bool m_destinationHasBeenSet;
};

} 
} 
} 
