

#pragma once
#include <aws/s3/S3_EXPORTS.h>
#include <aws/s3/model/FileHeaderInfo.h>
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


class AWS_S3_API CSVInput
{
public:
CSVInput();
CSVInput(const Aws::Utils::Xml::XmlNode& xmlNode);
CSVInput& operator=(const Aws::Utils::Xml::XmlNode& xmlNode);

void AddToNode(Aws::Utils::Xml::XmlNode& parentNode) const;



inline const FileHeaderInfo& GetFileHeaderInfo() const{ return m_fileHeaderInfo; }


inline void SetFileHeaderInfo(const FileHeaderInfo& value) { m_fileHeaderInfoHasBeenSet = true; m_fileHeaderInfo = value; }


inline void SetFileHeaderInfo(FileHeaderInfo&& value) { m_fileHeaderInfoHasBeenSet = true; m_fileHeaderInfo = std::move(value); }


inline CSVInput& WithFileHeaderInfo(const FileHeaderInfo& value) { SetFileHeaderInfo(value); return *this;}


inline CSVInput& WithFileHeaderInfo(FileHeaderInfo&& value) { SetFileHeaderInfo(std::move(value)); return *this;}



inline const Aws::String& GetComments() const{ return m_comments; }


inline void SetComments(const Aws::String& value) { m_commentsHasBeenSet = true; m_comments = value; }


inline void SetComments(Aws::String&& value) { m_commentsHasBeenSet = true; m_comments = std::move(value); }


inline void SetComments(const char* value) { m_commentsHasBeenSet = true; m_comments.assign(value); }


inline CSVInput& WithComments(const Aws::String& value) { SetComments(value); return *this;}


inline CSVInput& WithComments(Aws::String&& value) { SetComments(std::move(value)); return *this;}


inline CSVInput& WithComments(const char* value) { SetComments(value); return *this;}



inline const Aws::String& GetQuoteEscapeCharacter() const{ return m_quoteEscapeCharacter; }


inline void SetQuoteEscapeCharacter(const Aws::String& value) { m_quoteEscapeCharacterHasBeenSet = true; m_quoteEscapeCharacter = value; }


inline void SetQuoteEscapeCharacter(Aws::String&& value) { m_quoteEscapeCharacterHasBeenSet = true; m_quoteEscapeCharacter = std::move(value); }


inline void SetQuoteEscapeCharacter(const char* value) { m_quoteEscapeCharacterHasBeenSet = true; m_quoteEscapeCharacter.assign(value); }


inline CSVInput& WithQuoteEscapeCharacter(const Aws::String& value) { SetQuoteEscapeCharacter(value); return *this;}


inline CSVInput& WithQuoteEscapeCharacter(Aws::String&& value) { SetQuoteEscapeCharacter(std::move(value)); return *this;}


inline CSVInput& WithQuoteEscapeCharacter(const char* value) { SetQuoteEscapeCharacter(value); return *this;}



inline const Aws::String& GetRecordDelimiter() const{ return m_recordDelimiter; }


inline void SetRecordDelimiter(const Aws::String& value) { m_recordDelimiterHasBeenSet = true; m_recordDelimiter = value; }


inline void SetRecordDelimiter(Aws::String&& value) { m_recordDelimiterHasBeenSet = true; m_recordDelimiter = std::move(value); }


inline void SetRecordDelimiter(const char* value) { m_recordDelimiterHasBeenSet = true; m_recordDelimiter.assign(value); }


inline CSVInput& WithRecordDelimiter(const Aws::String& value) { SetRecordDelimiter(value); return *this;}


inline CSVInput& WithRecordDelimiter(Aws::String&& value) { SetRecordDelimiter(std::move(value)); return *this;}


inline CSVInput& WithRecordDelimiter(const char* value) { SetRecordDelimiter(value); return *this;}



inline const Aws::String& GetFieldDelimiter() const{ return m_fieldDelimiter; }


inline void SetFieldDelimiter(const Aws::String& value) { m_fieldDelimiterHasBeenSet = true; m_fieldDelimiter = value; }


inline void SetFieldDelimiter(Aws::String&& value) { m_fieldDelimiterHasBeenSet = true; m_fieldDelimiter = std::move(value); }


inline void SetFieldDelimiter(const char* value) { m_fieldDelimiterHasBeenSet = true; m_fieldDelimiter.assign(value); }


inline CSVInput& WithFieldDelimiter(const Aws::String& value) { SetFieldDelimiter(value); return *this;}


inline CSVInput& WithFieldDelimiter(Aws::String&& value) { SetFieldDelimiter(std::move(value)); return *this;}


inline CSVInput& WithFieldDelimiter(const char* value) { SetFieldDelimiter(value); return *this;}



inline const Aws::String& GetQuoteCharacter() const{ return m_quoteCharacter; }


inline void SetQuoteCharacter(const Aws::String& value) { m_quoteCharacterHasBeenSet = true; m_quoteCharacter = value; }


inline void SetQuoteCharacter(Aws::String&& value) { m_quoteCharacterHasBeenSet = true; m_quoteCharacter = std::move(value); }


inline void SetQuoteCharacter(const char* value) { m_quoteCharacterHasBeenSet = true; m_quoteCharacter.assign(value); }


inline CSVInput& WithQuoteCharacter(const Aws::String& value) { SetQuoteCharacter(value); return *this;}


inline CSVInput& WithQuoteCharacter(Aws::String&& value) { SetQuoteCharacter(std::move(value)); return *this;}


inline CSVInput& WithQuoteCharacter(const char* value) { SetQuoteCharacter(value); return *this;}

private:

FileHeaderInfo m_fileHeaderInfo;
bool m_fileHeaderInfoHasBeenSet;

Aws::String m_comments;
bool m_commentsHasBeenSet;

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
