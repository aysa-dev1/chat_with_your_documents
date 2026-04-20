import io
from src.ingestion.loader import load_pdf


def test_load_pdf_valid_content(mocker):
    test_text: str = "Hello, this is a test text."
    mock_page = mocker.Mock()
    mock_page.extract_text.return_value = test_text

    mock_reader = mocker.Mock()
    mock_reader.pages = [mock_page]
    mocker.patch("src.ingestion.loader.PdfReader", return_value=mock_reader)

    fake_file = io.BytesIO(b"fake_binary_data")
    fake_file.name = "test_document.pdf"

    result = load_pdf(fake_file)

    assert len(result) == 1
    assert result[0].page_content == test_text
    assert result[0].metadata["source"] == "test_document.pdf"
    assert result[0].metadata["page"] == 1


def test_load_pdf_skips_empty_pages(mocker):
    valid_page_text: str = "Valid text on page 1"
    page_valid = mocker.Mock()
    page_valid.extract_text.return_value = valid_page_text

    page_empty = mocker.Mock()
    page_empty.extract_text.return_value = "    "

    mock_reader = mocker.Mock()
    mock_reader.pages = [page_valid, page_empty]
    mocker.patch("src.ingestion.loader.PdfReader", return_value=mock_reader)

    result = load_pdf(io.BytesIO(b"fake_binary_data"))

    assert len(result) == 1
    assert result[0].page_content == valid_page_text


def test_load_pdf_source_fallback(mocker):
    content_text: str = "Some content" 
    mock_page = mocker.Mock()
    mock_page.extract_text.return_value = content_text

    mock_reader = mocker.Mock()
    mock_reader.pages = [mock_page]
    mocker.patch("src.ingestion.loader.PdfReader", return_value=mock_reader)

    fake_file = io.BytesIO(b"fake_binary_data")

    result = load_pdf(fake_file)

    assert len(result) == 1
    assert result[0].metadata["source"] == "uploaded_file"