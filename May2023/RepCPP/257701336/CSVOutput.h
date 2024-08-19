

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/model/QuoteFields.h>
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


class AWS_S3_API CSVOutput
{
public:
CSVOutput();
CSVOutput(const Aws::Utils::Xml::XmlNode& xmlNode);
CSVOutput& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const QuoteFields& GetQuoteFields() const{ return m_quoteFields; }


inline void SetQuoteFields(const QuoteFields& value) { m_quoteFieldsHasBeenSet = true; m_quoteFields = value; }


inline void SetQuoteFields(QuoteFields&& value) { m_quoteFieldsHasBeenSet = true; m_quoteFields = std::move(value); }


inline CSVOutput& WithQuoteFields(const QuoteFields& value) { SetQuoteFields(value); return *this;}


inline CSVOutput& WithQuoteFields(QuoteFields&& value) { SetQuoteFields(std::move(value)); return *this;}



inline const Aws::String& GetQuoteEscapeCharacter() const{ return m_quoteEscapeCharacter; }


inline void SetQuoteEscapeCharacter(const Aws::String& value) { m_quoteEscapeCharacterHasBeenSet = true; m_quoteEscapeCharacter = value; }


inline void SetQuoteEscapeCharacter(Aws::String&& value) { m_quoteEscapeCharacterHasBeenSet = true; m_quoteEscapeCharacter = std::move(value); }


inline void SetQuoteEscapeCharacter(const char* value) { m_quoteEscapeCharacterHasBeenSet = true; m_quoteEscapeCharacter.assign(value); }


inline CSVOutput& WithQuoteEscapeCharacter(const Aws::String& value) { SetQuoteEscapeCharacter(value); return *this;}


inline CSVOutput& WithQuoteEscapeCharacter(Aws::String&& value) { SetQuoteEscapeCharacter(std::move(value)); return *this;}


inline CSVOutput& WithQuoteEscapeCharacter(const char* value) { SetQuoteEscapeCharacter(value); return *this;}



inline const Aws::String& GetRecordDelimiter() const{ return m_recordDelimiter; }


inline void SetRecordDelimiter(const Aws::String& value) { m_recordDelimiterHasBeenSet = true; m_recordDelimiter = value; }


inline void SetRecordDelimiter(Aws::String&& value) { m_recordDelimiterHasBeenSet = true; m_recordDelimiter = std::move(value); }


inline void SetRecordDelimiter(const char* value) { m_recordDelimiterHasBeenSet = true; m_recordDelimiter.assign(value); }


inline CSVOutput& WithRecordDelimiter(const Aws::String& value) { SetRecordDelimiter(value); return *this;}


inline CSVOutput& WithRecordDelimiter(Aws::String&& value) { SetRecordDelimiter(std::move(value)); return *this;}


inline CSVOutput& WithRecordDelimiter(const char* value) { SetRecordDelimiter(value); return *this;}



inline const Aws::String& GetFieldDelimiter() const{ return m_fieldDelimiter; }


inline void SetFieldDelimiter(const Aws::String& value) { m_fieldDelimiterHasBeenSet = true; m_fieldDelimiter = value; }


inline void SetFieldDelimiter(Aws::String&& value) { m_fieldDelimiterHasBeenSet = true; m_fieldDelimiter = std::move(value); }


inline void SetFieldDelimiter(const char* value) { m_fieldDelimiterHasBeenSet = true; m_fieldDelimiter.assign(value); }


inline CSVOutput& WithFieldDelimiter(const Aws::String& value) { SetFieldDelimiter(value); return *this;}


inline CSVOutput& WithFieldDelimiter(Aws::String&& value) { SetFieldDelimiter(std::move(value)); return *this;}


inline CSVOutput& WithFieldDelimiter(const char* value) { SetFieldDelimiter(value); return *this;}



inline const Aws::String& GetQuoteCharacter() const{ return m_quoteCharacter; }


inline void SetQuoteCharacter(const Aws::String& value) { m_quoteCharacterHasBeenSet = true; m_quoteCharacter = value; }


inline void SetQuoteCharacter(Aws::String&& value) { m_quoteCharacterHasBeenSet = true; m_quoteCharacter = std::move(value); }


inline void SetQuoteCharacter(const char* value) { m_quoteCharacterHasBeenSet = true; m_quoteCharacter.assign(value); }


inline CSVOutput& WithQuoteCharacter(const Aws::String& value) { SetQuoteCharacter(value); return *this;}


inline CSVOutput& WithQuoteCharacter(Aws::String&& value) { SetQuoteCharacter(std::move(value)); return *this;}


inline CSVOutput& WithQuoteCharacter(const char* value) { SetQuoteCharacter(value); return *this;}

private:

QuoteFields m_quoteFields;
bool m_quoteFieldsHasBeenSet;

Aws::String m_quoteEscapeCharacter;
bool m_quoteEscapeCharacterHasBeenSet;

Aws::String m_recordDelimiter;
bool m_recordDelimiterHasBeenSet;

Aws::String m_fieldDelimiter;
bool m_fieldDelimiterHasBeenSet;

Aws::String m_quoteCharacter;
bool m_quoteCharacterHasBeenSet;
};

} 
} 
} 
