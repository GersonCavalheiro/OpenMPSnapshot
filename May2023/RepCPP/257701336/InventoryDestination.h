

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/model/InventoryS3BucketDestination.h>
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

class AWS_S3_API InventoryDestination
{
public:
InventoryDestination();
InventoryDestination(const Aws::Utils::Xml::XmlNode& xmlNode);
InventoryDestination& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const InventoryS3BucketDestination& GetS3BucketDestination() const{ return m_s3BucketDestination; }


inline void SetS3BucketDestination(const InventoryS3BucketDestination& value) { m_s3BucketDestinationHasBeenSet = true; m_s3BucketDestination = value; }


inline void SetS3BucketDestination(InventoryS3BucketDestination&& value) { m_s3BucketDestinationHasBeenSet = true; m_s3BucketDestination = std::move(value); }


inline InventoryDestination& WithS3BucketDestination(const InventoryS3BucketDestination& value) { SetS3BucketDestination(value); return *this;}


inline InventoryDestination& WithS3BucketDestination(InventoryS3BucketDestination&& value) { SetS3BucketDestination(std::move(value)); return *this;}

private:

InventoryS3BucketDestination m_s3BucketDestination;
bool m_s3BucketDestinationHasBeenSet;
};

} 
} 
} 
