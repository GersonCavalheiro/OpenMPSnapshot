

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/s3/model/MetricsFilter.h>
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

class AWS_S3_API MetricsConfiguration
{
public:
MetricsConfiguration();
MetricsConfiguration(const Aws::Utils::Xml::XmlNode& xmlNode);
MetricsConfiguration& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const Aws::String& GetId() const{ return m_id; }


inline void SetId(const Aws::String& value) { m_idHasBeenSet = true; m_id = value; }


inline void SetId(Aws::String&& value) { m_idHasBeenSet = true; m_id = std::move(value); }


inline void SetId(const char* value) { m_idHasBeenSet = true; m_id.assign(value); }


inline MetricsConfiguration& WithId(const Aws::String& value) { SetId(value); return *this;}


inline MetricsConfiguration& WithId(Aws::String&& value) { SetId(std::move(value)); return *this;}


inline MetricsConfiguration& WithId(const char* value) { SetId(value); return *this;}



inline const MetricsFilter& GetFilter() const{ return m_filter; }


inline void SetFilter(const MetricsFilter& value) { m_filterHasBeenSet = true; m_filter = value; }


inline void SetFilter(MetricsFilter&& value) { m_filterHasBeenSet = true; m_filter = std::move(value); }


inline MetricsConfiguration& WithFilter(const MetricsFilter& value) { SetFilter(value); return *this;}


inline MetricsConfiguration& WithFilter(MetricsFilter&& value) { SetFilter(std::move(value)); return *this;}

private:

Aws::String m_id;
bool m_idHasBeenSet;

MetricsFilter m_filter;
bool m_filterHasBeenSet;
};

} 
} 
} 
