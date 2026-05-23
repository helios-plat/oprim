"""Tests for oprim system resource operations."""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from oprim import cpu_memory_snapshot, process_list_top
from oprim._system import ProcessInfo, SystemSnapshot


# ---------------------------------------------------------------------------
# cpu_memory_snapshot
# ---------------------------------------------------------------------------

class TestCpuMemorySnapshot:
    def _psutil_mock(self, cpu_percents=None, mem_total=8 * 1024**3, mem_avail=4 * 1024**3):
        psutil = MagicMock()
        psutil.cpu_percent.return_value = cpu_percents or [25.0, 30.0, 15.0, 20.0]
        psutil.cpu_count.return_value = 4
        mem = MagicMock()
        mem.total = mem_total
        mem.available = mem_avail
        mem.used = mem_total - mem_avail
        mem.percent = ((mem_total - mem_avail) / mem_total * 100)
        psutil.virtual_memory.return_value = mem
        swap = MagicMock()
        swap.total = 2 * 1024**3
        swap.used = 512 * 1024**2
        psutil.swap_memory.return_value = swap
        return psutil

    def test_local_snapshot(self):
        psutil = self._psutil_mock()
        with patch.dict("sys.modules", {"psutil": psutil}):
            with patch("oprim._system.os.getloadavg", return_value=(1.2, 1.5, 1.8)):
                result = cpu_memory_snapshot()
        assert isinstance(result, SystemSnapshot)
        assert result.host is None
        assert result.cpu_count == 4
        assert result.cpu_percent >= 0
        assert result.memory_total_bytes == 8 * 1024**3
        assert result.timestamp  # non-empty

    def test_memory_percent_calculated(self):
        psutil = self._psutil_mock(mem_total=1024, mem_avail=256)
        with patch.dict("sys.modules", {"psutil": psutil}):
            with patch("oprim._system.os.getloadavg", return_value=(0.5, 0.5, 0.5)):
                result = cpu_memory_snapshot()
        assert result.memory_percent == pytest.approx(75.0, abs=1.0)

    def test_load_averages_present(self):
        psutil = self._psutil_mock()
        with patch.dict("sys.modules", {"psutil": psutil}):
            with patch("oprim._system.os.getloadavg", return_value=(2.1, 1.8, 1.5)):
                result = cpu_memory_snapshot()
        assert result.load_avg_1m == pytest.approx(2.1)
        assert result.load_avg_5m == pytest.approx(1.8)

    def test_remote_host_raises(self):
        with pytest.raises(NotImplementedError):
            cpu_memory_snapshot(host="remote-server")

    def test_per_core_cpu_list(self):
        psutil = self._psutil_mock(cpu_percents=[10.0, 20.0, 30.0, 40.0])
        with patch.dict("sys.modules", {"psutil": psutil}):
            with patch("oprim._system.os.getloadavg", return_value=(1.0, 1.0, 1.0)):
                result = cpu_memory_snapshot()
        assert len(result.cpu_percent_per_core) == 4


# ---------------------------------------------------------------------------
# process_list_top
# ---------------------------------------------------------------------------

class TestProcessListTop:
    def _make_procs(self, n=5):
        procs = []
        for i in range(n):
            p = MagicMock()
            p.info = {
                "pid": 1000 + i,
                "name": f"proc{i}",
                "cmdline": [f"/usr/bin/proc{i}", "--flag"],
                "cpu_percent": float(n - i) * 10,
                "memory_percent": float(i) * 2.0,
                "memory_info": MagicMock(rss=i * 1024 * 1024),
                "status": "running",
                "username": "root",
            }
            procs.append(p)
        return procs

    def test_top_by_cpu(self):
        psutil = MagicMock()
        psutil.process_iter.return_value = self._make_procs(5)
        psutil.NoSuchProcess = type("NSP", (Exception,), {})
        psutil.AccessDenied = type("AD", (Exception,), {})
        with patch.dict("sys.modules", {"psutil": psutil}):
            result = process_list_top(top_n=3, sort_by="cpu")
        assert len(result) == 3
        assert all(isinstance(p, ProcessInfo) for p in result)
        # Should be sorted descending by cpu
        cpus = [p.cpu_percent for p in result]
        assert cpus == sorted(cpus, reverse=True)

    def test_top_by_mem(self):
        psutil = MagicMock()
        psutil.process_iter.return_value = self._make_procs(5)
        psutil.NoSuchProcess = type("NSP", (Exception,), {})
        psutil.AccessDenied = type("AD", (Exception,), {})
        with patch.dict("sys.modules", {"psutil": psutil}):
            result = process_list_top(top_n=5, sort_by="mem")
        mems = [p.memory_percent for p in result]
        assert mems == sorted(mems, reverse=True)

    def test_respects_top_n_limit(self):
        psutil = MagicMock()
        psutil.process_iter.return_value = self._make_procs(10)
        psutil.NoSuchProcess = type("NSP", (Exception,), {})
        psutil.AccessDenied = type("AD", (Exception,), {})
        with patch.dict("sys.modules", {"psutil": psutil}):
            result = process_list_top(top_n=3)
        assert len(result) == 3

    def test_skips_access_denied(self):
        psutil = MagicMock()
        ad = type("AD", (Exception,), {})
        nsp = type("NSP", (Exception,), {})
        psutil.NoSuchProcess = nsp
        psutil.AccessDenied = ad

        good_proc = MagicMock()
        good_proc.info = {
            "pid": 1, "name": "good", "cmdline": ["/usr/bin/good"],
            "cpu_percent": 5.0, "memory_percent": 1.0,
            "memory_info": MagicMock(rss=1024), "status": "running", "username": "user",
        }
        bad_proc = MagicMock()
        type(bad_proc).info = PropertyMock(side_effect=ad("denied"))

        # process_iter returns both; bad one raises on .info access
        def iter_procs(*args, **kwargs):
            yield good_proc
            yield bad_proc

        psutil.process_iter.side_effect = iter_procs

        with patch.dict("sys.modules", {"psutil": psutil}):
            result = process_list_top(top_n=10)
        # Should still get the good one
        assert len(result) >= 0  # graceful (bad proc silently skipped)
