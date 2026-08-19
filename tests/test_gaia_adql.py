"""ESA Gaia ADQL with Heidelberg ARI TAP fallback."""

from darkhunter_rv import gaia_utils


class _FakeResults:
    def __init__(self, rows: list[dict]):
        self.colnames = list(rows[0].keys()) if rows else []
        self._rows = [tuple(row[c] for c in self.colnames) for row in rows]

    def __iter__(self):
        return iter(self._rows)


class _FakeJob:
    def __init__(self, rows: list[dict]):
        self._results = _FakeResults(rows)

    def get_results(self):
        return self._results


class _FakeEsaGaia:
    def __init__(self, *, error: Exception | None = None, rows: list[dict] | None = None):
        self.error = error
        self.rows = rows or []
        self.calls = 0

    def launch_job_async(self, query, dump_to_file=False):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return _FakeJob(self.rows)


class _FakeAriTap:
    def __init__(self, *, error: Exception | None = None, rows: list[dict] | None = None):
        self.error = error
        self.rows = rows or []
        self.calls = 0
        self.queries: list[str] = []

    def launch_job(self, query, dump_to_file=False):
        self.calls += 1
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        return _FakeJob(self.rows)


def test_execute_gaia_adql_uses_esa_when_ok(monkeypatch):
    esa = _FakeEsaGaia(rows=[{"source_id": 1, "plx": 2.5}])
    ari = _FakeAriTap(rows=[{"source_id": 99}])
    monkeypatch.setattr(gaia_utils, "_gaia_class", lambda: esa)
    monkeypatch.setattr(gaia_utils, "_ari_tap", lambda: ari)

    rows = gaia_utils.execute_gaia_adql("SELECT 1", "unit")
    assert rows == [{"source_id": 1, "plx": 2.5}]
    assert esa.calls == 1
    assert ari.calls == 0


def test_execute_gaia_adql_falls_back_to_ari(monkeypatch):
    esa = _FakeEsaGaia(error=ConnectionError("Remote end closed connection without response"))
    ari = _FakeAriTap(rows=[{"source_id": 1, "plx": 2.5}])
    monkeypatch.setattr(gaia_utils, "_gaia_class", lambda: esa)
    monkeypatch.setattr(gaia_utils, "_ari_tap", lambda: ari)

    query = "SELECT source_id FROM gaiadr3.gaia_source WHERE source_id = 1"
    rows = gaia_utils.execute_gaia_adql(query, "Gaia Core")
    assert rows == [{"source_id": 1, "plx": 2.5}]
    assert esa.calls == 1
    assert ari.calls == 1
    assert ari.queries == [query]


def test_execute_gaia_adql_returns_empty_when_both_fail(monkeypatch):
    esa = _FakeEsaGaia(error=RuntimeError("esa down"))
    ari = _FakeAriTap(error=RuntimeError("ari down"))
    monkeypatch.setattr(gaia_utils, "_gaia_class", lambda: esa)
    monkeypatch.setattr(gaia_utils, "_ari_tap", lambda: ari)

    assert gaia_utils.execute_gaia_adql("SELECT 1", "unit") == []


def test_execute_gaia_adql_uses_ari_if_esa_missing(monkeypatch):
    ari = _FakeAriTap(rows=[{"ok": 1}])
    monkeypatch.setattr(gaia_utils, "_gaia_class", lambda: None)
    monkeypatch.setattr(gaia_utils, "_ari_tap", lambda: ari)

    assert gaia_utils.execute_gaia_adql("SELECT 1", "unit") == [{"ok": 1}]
    assert ari.calls == 1
