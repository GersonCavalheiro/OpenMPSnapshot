

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/model/MFADelete.h>
#include <aws/s3/model/BucketVersioningStatus.h>
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

class AWS_S3_API VersioningConfiguration
{
public:
VersioningConfiguration();
VersioningConfiguration(const Aws::Utils::Xml::XmlNode& xmlNode);
VersioningConfiguration& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const MFADelete& GetMFADelete() const{ return m_mFADelete; }


inline void SetMFADelete(const MFADelete& value) { m_mFADeleteHasBeenSet = true; m_mFADelete = value; }


inline void SetMFADelete(MFADelete&& value) { m_mFADeleteHasBeenSet = true; m_mFADelete = std::move(value); }


inline VersioningConfiguration& WithMFADelete(const MFADelete& value) { SetMFADelete(value); return *this;}


inline VersioningConfiguration& WithMFADelete(MFADelete&& value) { SetMFADelete(std::move(value)); return *this;}



inline const BucketVersioningStatus& GetStatus() const{ return m_status; }


inline void SetStatus(const BucketVersioningStatus& value) { m_statusHasBeenSet = true; m_status = value; }


inline void SetStatus(BucketVersioningStatus&& value) { m_statusHasBeenSet = true; m_status = std::move(value); }


inline VersioningConfiguration& WithStatus(const BucketVersioningStatus& value) { SetStatus(value); return *this;}


inline VersioningConfiguration& WithStatus(BucketVersioningStatus&& value) { SetStatus(std::move(value)); return *this;}

private:

MFADelete m_mFADelete;
bool m_mFADeleteHasBeenSet;

BucketVersioningStatus m_status;
bool m_statusHasBeenSet;
};

} 
} 
} 
