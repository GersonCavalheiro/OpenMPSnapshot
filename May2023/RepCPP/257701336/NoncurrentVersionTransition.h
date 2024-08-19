

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/model/TransitionStorageClass.h>
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


class AWS_S3_API NoncurrentVersionTransition
{
public:
NoncurrentVersionTransition();
NoncurrentVersionTransition(const Aws::Utils::Xml::XmlNode& xmlNode);
NoncurrentVersionTransition& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline int GetNoncurrentDays() const{ return m_noncurrentDays; }


inline void SetNoncurrentDays(int value) { m_noncurrentDaysHasBeenSet = true; m_noncurrentDays = value; }


inline NoncurrentVersionTransition& WithNoncurrentDays(int value) { SetNoncurrentDays(value); return *this;}



inline const TransitionStorageClass& GetStorageClass() const{ return m_storageClass; }


inline void SetStorageClass(const TransitionStorageClass& value) { m_storageClassHasBeenSet = true; m_storageClass = value; }


inline void SetStorageClass(TransitionStorageClass&& value) { m_storageClassHasBeenSet = true; m_storageClass = std::move(value); }


inline NoncurrentVersionTransition& WithStorageClass(const TransitionStorageClass& value) { SetStorageClass(value); return *this;}


inline NoncurrentVersionTransition& WithStorageClass(TransitionStorageClass&& value) { SetStorageClass(std::move(value)); return *this;}

private:

int m_noncurrentDays;
bool m_noncurrentDaysHasBeenSet;

TransitionStorageClass m_storageClass;
bool m_storageClassHasBeenSet;
};

} 
} 
} 
