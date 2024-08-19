

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/model/SseKmsEncryptedObjects.h>
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


class AWS_S3_API SourceSelectionCriteria
{
public:
SourceSelectionCriteria();
SourceSelectionCriteria(const Aws::Utils::Xml::XmlNode& xmlNode);
SourceSelectionCriteria& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const SseKmsEncryptedObjects& GetSseKmsEncryptedObjects() const{ return m_sseKmsEncryptedObjects; }


inline void SetSseKmsEncryptedObjects(const SseKmsEncryptedObjects& value) { m_sseKmsEncryptedObjectsHasBeenSet = true; m_sseKmsEncryptedObjects = value; }


inline void SetSseKmsEncryptedObjects(SseKmsEncryptedObjects&& value) { m_sseKmsEncryptedObjectsHasBeenSet = true; m_sseKmsEncryptedObjects = std::move(value); }


inline SourceSelectionCriteria& WithSseKmsEncryptedObjects(const SseKmsEncryptedObjects& value) { SetSseKmsEncryptedObjects(value); return *this;}


inline SourceSelectionCriteria& WithSseKmsEncryptedObjects(SseKmsEncryptedObjects&& value) { SetSseKmsEncryptedObjects(std::move(value)); return *this;}

private:

SseKmsEncryptedObjects m_sseKmsEncryptedObjects;
bool m_sseKmsEncryptedObjectsHasBeenSet;
};

} 
} 
} 
