import io
import docx
from pypdf import PdfReader

def extract_text(file):
    if file.type == "application/pdf":
        reader = PdfReader(io.BytesIO(file.read()))
        texts = []
        for page in reader.pages:
            t = page.extract_text() or ""
            if t: texts.append(t)
        return "\n".join(texts)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        return "\n".join([p.text for p in doc.paragraphs if p.text])
    else:
        raise ValueError("Unsupported file type")
