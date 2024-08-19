

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/utils/memory/stl/AWSVector.h>
#include <aws/s3/model/TargetGrant.h>
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

class AWS_S3_API LoggingEnabled
{
public:
LoggingEnabled();
LoggingEnabled(const Aws::Utils::Xml::XmlNode& xmlNode);
LoggingEnabled& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const Aws::String& GetTargetBucket() const{ return m_targetBucket; }


inline void SetTargetBucket(const Aws::String& value) { m_targetBucketHasBeenSet = true; m_targetBucket = value; }


inline void SetTargetBucket(Aws::String&& value) { m_targetBucketHasBeenSet = true; m_targetBucket = std::move(value); }


inline void SetTargetBucket(const char* value) { m_targetBucketHasBeenSet = true; m_targetBucket.assign(value); }


inline LoggingEnabled& WithTargetBucket(const Aws::String& value) { SetTargetBucket(value); return *this;}


inline LoggingEnabled& WithTargetBucket(Aws::String&& value) { SetTargetBucket(std::move(value)); return *this;}


inline LoggingEnabled& WithTargetBucket(const char* value) { SetTargetBucket(value); return *this;}



inline const Aws::Vector<TargetGrant>& GetTargetGrants() const{ return m_targetGrants; }


inline void SetTargetGrants(const Aws::Vector<TargetGrant>& value) { m_targetGrantsHasBeenSet = true; m_targetGrants = value; }


inline void SetTargetGrants(Aws::Vector<TargetGrant>&& value) { m_targetGrantsHasBeenSet = true; m_targetGrants = std::move(value); }


inline LoggingEnabled& WithTargetGrants(const Aws::Vector<TargetGrant>& value) { SetTargetGrants(value); return *this;}


inline LoggingEnabled& WithTargetGrants(Aws::Vector<TargetGrant>&& value) { SetTargetGrants(std::move(value)); return *this;}


inline LoggingEnabled& AddTargetGrants(const TargetGrant& value) { m_targetGrantsHasBeenSet = true; m_targetGrants.push_back(value); return *this; }


inline LoggingEnabled& AddTargetGrants(TargetGrant&& value) { m_targetGrantsHasBeenSet = true; m_targetGrants.push_back(std::move(value)); return *this; }



inline const Aws::String& GetTargetPrefix() const{ return m_targetPrefix; }


inline void SetTargetPrefix(const Aws::String& value) { m_targetPrefixHasBeenSet = true; m_targetPrefix = value; }


inline void SetTargetPrefix(Aws::String&& value) { m_targetPrefixHasBeenSet = true; m_targetPrefix = std::move(value); }


inline void SetTargetPrefix(const char* value) { m_targetPrefixHasBeenSet = true; m_targetPrefix.assign(value); }


inline LoggingEnabled& WithTargetPrefix(const Aws::String& value) { SetTargetPrefix(value); return *this;}


inline LoggingEnabled& WithTargetPrefix(Aws::String&& value) { SetTargetPrefix(std::move(value)); return *this;}


inline LoggingEnabled& WithTargetPrefix(const char* value) { SetTargetPrefix(value); return *this;}

private:

Aws::String m_targetBucket;
bool m_targetBucketHasBeenSet;

Aws::Vector<TargetGrant> m_targetGrants;
bool m_targetGrantsHasBeenSet;

Aws::String m_targetPrefix;
bool m_targetPrefixHasBeenSet;
};

} 
} 
} 
