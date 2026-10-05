"""Builders for synthetic statement files that attack the shape tools' side channels.

Ported from the second-round privacy review (B-090 onward). Every file here is synthetic:
the "secrets" are fixed markers a test searches the output for. Loaded by the shape-tool
tests with importlib (``tests`` is not a package), and built into ``tmp_path`` at test time
so the bytes each test relies on are visible here rather than hidden in a binary fixture.

* :func:`xlsx_with_metadata` - a valid one-sheet workbook whose METADATA carries a marker:
  a custom document property of an unknown type (openpyxl warns, quoting its name), a
  print-area defined name (openpyxl warns, quoting its value) and a defined name for a sheet
  index that does not exist (openpyxl warns). All three reach stderr through
  ``warnings.warn`` in openpyxl 3.1, in read-only mode too (B-090).
"""

from __future__ import annotations

import io
import zipfile

#: The marker every metadata channel carries. Digits included, so a digit-run check fires too.
META_SECRET = "HOLDER Jane Q Public SSN 123-45-6789"
PRINT_AREA_SECRET = "JANE_Q_PUBLIC_ACCT_987654321"

_CUSTOM_XML = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/custom-properties"'
    ' xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
    '<property fmtid="{D5CDD505-2E9C-101B-9397-08002B2CF9AE}" pid="2" name="'
    + META_SECRET
    + '"><vt:blob>AAAA</vt:blob></property></Properties>'
)


def _base_xlsx(rows: list[list[str]]) -> bytes:
    import openpyxl

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Sheet1"
    for row in rows:
        sheet.append(row)
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def xlsx_with_metadata(
    rows: list[list[str]] | None = None,
    *,
    custom_property: bool = True,
    print_area: bool = True,
    bad_sheet_index: bool = True,
) -> bytes:
    """A readable workbook (``rows``) with markers planted in the chosen metadata channels."""
    rows = rows or [["Date", "Description", "Amount"], ["2024-01-02", "SECRETMERCHANT", "12.50"]]
    source = zipfile.ZipFile(io.BytesIO(_base_xlsx(rows)))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as target:
        for name in source.namelist():
            data = source.read(name)
            if name == "[Content_Types].xml" and custom_property:
                data = data.replace(
                    b"</Types>",
                    b'<Override PartName="/docProps/custom.xml" ContentType="application/'
                    b'vnd.openxmlformats-officedocument.custom-properties+xml"/></Types>',
                )
            elif name == "_rels/.rels" and custom_property:
                data = data.replace(
                    b"</Relationships>",
                    b'<Relationship Id="rIdC" Type="http://schemas.openxmlformats.org/'
                    b'officeDocument/2006/relationships/custom-properties" '
                    b'Target="docProps/custom.xml"/></Relationships>',
                )
                target.writestr("docProps/custom.xml", _CUSTOM_XML)
            elif name == "xl/workbook.xml":
                names = b""
                if print_area:
                    names += (
                        b'<definedName name="_xlnm.Print_Area" localSheetId="0">'
                        + PRINT_AREA_SECRET.encode()
                        + b"</definedName>"
                    )
                if bad_sheet_index:
                    names += (
                        b'<definedName name="Holder987654321" localSheetId="7">'
                        b"Sheet1!$A$1</definedName>"
                    )
                if names:
                    # openpyxl writes an empty <definedNames /> already; a second element
                    # would be ignored, so the planted names replace it.
                    data = data.replace(b"<definedNames />", b"").replace(
                        b"</sheets>", b"</sheets><definedNames>" + names + b"</definedNames>"
                    )
            target.writestr(name, data)
    return out.getvalue()
