

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/s3/model/AnalyticsFilter.h>
#include <aws/s3/model/StorageClassAnalysis.h>
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

class AWS_S3_API AnalyticsConfiguration
{
public:
AnalyticsConfiguration();
AnalyticsConfiguration(const Aws::Utils::Xml::XmlNode& xmlNode);
AnalyticsConfiguration& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const Aws::String& GetId() const{ return m_id; }


inline void SetId(const Aws::String& value) { m_idHasBeenSet = true; m_id = value; }


inline void SetId(Aws::String&& value) { m_idHasBeenSet = true; m_id = std::move(value); }


inline void SetId(const char* value) { m_idHasBeenSet = true; m_id.assign(value); }


inline AnalyticsConfiguration& WithId(const Aws::String& value) { SetId(value); return *this;}


inline AnalyticsConfiguration& WithId(Aws::String&& value) { SetId(std::move(value)); return *this;}


inline AnalyticsConfiguration& WithId(const char* value) { SetId(value); return *this;}



inline const AnalyticsFilter& GetFilter() const{ return m_filter; }


inline void SetFilter(const AnalyticsFilter& value) { m_filterHasBeenSet = true; m_filter = value; }


inline void SetFilter(AnalyticsFilter&& value) { m_filterHasBeenSet = true; m_filter = std::move(value); }


inline AnalyticsConfiguration& WithFilter(const AnalyticsFilter& value) { SetFilter(value); return *this;}


inline AnalyticsConfiguration& WithFilter(AnalyticsFilter&& value) { SetFilter(std::move(value)); return *this;}



inline const StorageClassAnalysis& GetStorageClassAnalysis() const{ return m_storageClassAnalysis; }


inline void SetStorageClassAnalysis(const StorageClassAnalysis& value) { m_storageClassAnalysisHasBeenSet = true; m_storageClassAnalysis = value; }


inline void SetStorageClassAnalysis(StorageClassAnalysis&& value) { m_storageClassAnalysisHasBeenSet = true; m_storageClassAnalysis = std::move(value); }


inline AnalyticsConfiguration& WithStorageClassAnalysis(const StorageClassAnalysis& value) { SetStorageClassAnalysis(value); return *this;}


inline AnalyticsConfiguration& WithStorageClassAnalysis(StorageClassAnalysis&& value) { SetStorageClassAnalysis(std::move(value)); return *this;}

private:

Aws::String m_id;
bool m_idHasBeenSet;

AnalyticsFilter m_filter;
bool m_filterHasBeenSet;

StorageClassAnalysis m_storageClassAnalysis;
bool m_storageClassAnalysisHasBeenSet;
};

} 
} 
} 
