import io
from pathlib import Path

import pandas as pd
from pypdf import PdfReader
from fastapi import UploadFile

SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.csv', '.xlsx', '.xls'}


class DocumentIngestionService:
    def __init__(self, upload_dir: Path) -> None:
        self.upload_dir = upload_dir
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    async def save_upload(self, upload: UploadFile) -> Path:
        suffix = Path(upload.filename or '').suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(f'Unsupported file type: {suffix}')

        destination = self.upload_dir / (upload.filename or 'uploaded_file')
        data = await upload.read()
        destination.write_bytes(data)
        return destination

    def parse_file(self, file_path: Path) -> str:
        suffix = file_path.suffix.lower()
        if suffix == '.pdf':
            return self._parse_pdf(file_path)
        if suffix == '.txt':
            return file_path.read_text(encoding='utf-8', errors='ignore')
        if suffix == '.csv':
            df = pd.read_csv(file_path)
            return df.to_csv(index=False)
        if suffix in {'.xlsx', '.xls'}:
            excel = pd.ExcelFile(file_path)
            pieces: list[str] = []
            for sheet_name in excel.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                pieces.append(f'Sheet: {sheet_name}\n{df.to_csv(index=False)}')
            return '\n\n'.join(pieces)
        raise ValueError(f'Unsupported file type: {suffix}')

    def _parse_pdf(self, file_path: Path) -> str:
        reader = PdfReader(str(file_path))
        pages: list[str] = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ''
            pages.append(f'[Page {page_number}]\n{text}')
        return '\n\n'.join(pages)
