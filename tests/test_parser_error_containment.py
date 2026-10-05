"""No parser message reaches a caller: a lazily-raised openpyxl error quoted cell text
(``invalid literal for int() with base 10: 'SSN-123-45-6789'``) and the MCP server returned it
verbatim to the calling agent. Synthetic fixtures (from the round-4 review) hold planted
values; neither the message nor the formatted traceback may contain them."""

from __future__ import annotations

import shutil
import traceback
from pathlib import Path

import pytest

from hearth.config import Settings
from hearth.mcp.files import FileAccessError, read_table, read_text_file

FIXTURES = Path(__file__).parent / "fixtures" / "hostile_xlsx"
PLANTED = ("SSN", "123-45-6789", "99887766", "ACCT")


@pytest.mark.parametrize("reader", [read_table, read_text_file])
@pytest.mark.parametrize("name", ["badnum.xlsx", "baddate.xlsx", "badxml.xlsx"])
def test_a_parser_error_never_carries_file_content(tmp_path, reader, name):
    target = tmp_path / name
    shutil.copy(FIXTURES / name, target)
    settings = Settings(file_roots=str(tmp_path), home=tmp_path / "h")
    with pytest.raises(FileAccessError) as excinfo:
        reader(target, settings)
    rendered = "".join(traceback.format_exception(excinfo.value))
    for secret in PLANTED:
        assert secret not in str(excinfo.value), secret
        assert secret not in rendered, f"{secret} leaked through the traceback"
    assert "details withheld" in str(excinfo.value)
