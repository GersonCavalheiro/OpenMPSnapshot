

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSString.h>
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


class AWS_S3_API EncryptionConfiguration
{
public:
EncryptionConfiguration();
EncryptionConfiguration(const Aws::Utils::Xml::XmlNode& xmlNode);
EncryptionConfiguration& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const Aws::String& GetReplicaKmsKeyID() const{ return m_replicaKmsKeyID; }


inline void SetReplicaKmsKeyID(const Aws::String& value) { m_replicaKmsKeyIDHasBeenSet = true; m_replicaKmsKeyID = value; }


inline void SetReplicaKmsKeyID(Aws::String&& value) { m_replicaKmsKeyIDHasBeenSet = true; m_replicaKmsKeyID = std::move(value); }


inline void SetReplicaKmsKeyID(const char* value) { m_replicaKmsKeyIDHasBeenSet = true; m_replicaKmsKeyID.assign(value); }


inline EncryptionConfiguration& WithReplicaKmsKeyID(const Aws::String& value) { SetReplicaKmsKeyID(value); return *this;}


inline EncryptionConfiguration& WithReplicaKmsKeyID(Aws::String&& value) { SetReplicaKmsKeyID(std::move(value)); return *this;}


inline EncryptionConfiguration& WithReplicaKmsKeyID(const char* value) { SetReplicaKmsKeyID(value); return *this;}

private:

Aws::String m_replicaKmsKeyID;
bool m_replicaKmsKeyIDHasBeenSet;
};

} 
} 
} 
