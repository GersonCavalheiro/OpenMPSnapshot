

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/model/ServerSideEncryptionByDefault.h>
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


class AWS_S3_API ServerSideEncryptionRule
{
public:
ServerSideEncryptionRule();
ServerSideEncryptionRule(const Aws::Utils::Xml::XmlNode& xmlNode);
ServerSideEncryptionRule& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const ServerSideEncryptionByDefault& GetApplyServerSideEncryptionByDefault() const{ return m_applyServerSideEncryptionByDefault; }


inline void SetApplyServerSideEncryptionByDefault(const ServerSideEncryptionByDefault& value) { m_applyServerSideEncryptionByDefaultHasBeenSet = true; m_applyServerSideEncryptionByDefault = value; }


inline void SetApplyServerSideEncryptionByDefault(ServerSideEncryptionByDefault&& value) { m_applyServerSideEncryptionByDefaultHasBeenSet = true; m_applyServerSideEncryptionByDefault = std::move(value); }


inline ServerSideEncryptionRule& WithApplyServerSideEncryptionByDefault(const ServerSideEncryptionByDefault& value) { SetApplyServerSideEncryptionByDefault(value); return *this;}


inline ServerSideEncryptionRule& WithApplyServerSideEncryptionByDefault(ServerSideEncryptionByDefault&& value) { SetApplyServerSideEncryptionByDefault(std::move(value)); return *this;}

private:

ServerSideEncryptionByDefault m_applyServerSideEncryptionByDefault;
bool m_applyServerSideEncryptionByDefaultHasBeenSet;
};

} 
} 
} 
