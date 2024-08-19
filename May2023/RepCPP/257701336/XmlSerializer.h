

#pragma once

#include <aws/core/Core_EXPORTS.h>

#include <aws/core/utils/Outcome.h>
#include <aws/core/utils/memory/stl/AWSStreamFwd.h>
#include <aws/core/utils/memory/stl/AWSString.h>

namespace Aws
{
namespace External
{
namespace tinyxml2
{
class XMLNode;

class XMLDocument;
} 
} 
} 

namespace Aws
{
template<typename PAYLOAD_TYPE>
class AmazonWebServiceResult;
namespace Client
{
enum class CoreErrors;
template<typename ERROR_TYPE>
class AWSError;
class AWSXMLClient;
} 
namespace Utils
{
namespace Xml
{

AWS_CORE_API Aws::String DecodeEscapedXmlText(const Aws::String& textToDecode);

class XmlDocument;


class AWS_CORE_API XmlNode
{
public:

XmlNode(const XmlNode& other);

XmlNode& operator=(const XmlNode& other);

const Aws::String GetName() const;

void SetName(const Aws::String& name);

const Aws::String GetAttributeValue(const Aws::String& name) const;

void SetAttributeValue(const Aws::String& name, const Aws::String& value);

Aws::String GetText() const;

void SetText(const Aws::String& textValue);

bool HasNextNode() const;

XmlNode NextNode() const;

XmlNode NextNode(const char* name) const;

XmlNode NextNode(const Aws::String& name) const;

XmlNode FirstChild() const;

XmlNode FirstChild(const char* name) const;

XmlNode FirstChild(const Aws::String& name) const;

bool HasChildren() const;

XmlNode Parent() const;

XmlNode CreateChildElement(const Aws::String& name);

XmlNode CreateSiblingElement(const Aws::String& name);

bool IsNull();

private:
XmlNode(Aws::External::tinyxml2::XMLNode* node, const XmlDocument& document) :
m_node(node), m_doc(&document)
{
}

Aws::External::tinyxml2::XMLNode* m_node;
const XmlDocument* m_doc;

friend class XmlDocument;
};


class AWS_CORE_API XmlDocument
{
public:

XmlDocument(XmlDocument&& doc); 
XmlDocument(const XmlDocument& other) = delete;

~XmlDocument();


XmlNode GetRootElement() const;

Aws::String ConvertToString() const;

bool WasParseSuccessful() const;

Aws::String GetErrorMessage() const;

static XmlDocument CreateFromXmlStream(Aws::IOStream&);

static XmlDocument CreateFromXmlString(const Aws::String&);

static XmlDocument CreateWithRootNode(const Aws::String&);

private:
XmlDocument();

Aws::External::tinyxml2::XMLDocument* m_doc;

friend class XmlNode;

friend class Aws::Utils::Outcome<Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>, Aws::Client::AWSError<Aws::Client::CoreErrors>>;
friend class Aws::AmazonWebServiceResult<Aws::Utils::Xml::XmlDocument>;
friend class Client::AWSXMLClient;
};

} 
} 
} 

