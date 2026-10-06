from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.chunking.title import chunk_by_title


def parse_pdf(
    file_path: str,
    max_characters: int = 4000,
    new_after_n_chars: int = 3800,
    combine_text_under_n_chars: int = 250,
):
    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res"
    )

    return chunk_by_title(
        elements,
        max_characters=max_characters,
        new_after_n_chars=new_after_n_chars,
        combine_text_under_n_chars=combine_text_under_n_chars
    )


def parse_docx(
    file_path: str,
    max_characters: int = 1200,
    new_after_n_chars: int = 1000,
    combine_text_under_n_chars: int = 150,
):
    elements = partition_docx(
        filename=file_path
    )

    return chunk_by_title(
        elements,
        max_characters=max_characters,
        new_after_n_chars=new_after_n_chars,
        combine_text_under_n_chars=combine_text_under_n_chars
    )