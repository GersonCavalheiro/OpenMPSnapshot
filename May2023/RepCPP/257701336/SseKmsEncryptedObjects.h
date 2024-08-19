

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/model/SseKmsEncryptedObjectsStatus.h>
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


class AWS_S3_API SseKmsEncryptedObjects
{
public:
SseKmsEncryptedObjects();
SseKmsEncryptedObjects(const Aws::Utils::Xml::XmlNode& xmlNode);
SseKmsEncryptedObjects& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const SseKmsEncryptedObjectsStatus& GetStatus() const{ return m_status; }


inline void SetStatus(const SseKmsEncryptedObjectsStatus& value) { m_statusHasBeenSet = true; m_status = value; }


inline void SetStatus(SseKmsEncryptedObjectsStatus&& value) { m_statusHasBeenSet = true; m_status = std::move(value); }


inline SseKmsEncryptedObjects& WithStatus(const SseKmsEncryptedObjectsStatus& value) { SetStatus(value); return *this;}


inline SseKmsEncryptedObjects& WithStatus(SseKmsEncryptedObjectsStatus&& value) { SetStatus(std::move(value)); return *this;}

private:

SseKmsEncryptedObjectsStatus m_status;
bool m_statusHasBeenSet;
};

} 
} 
} 
