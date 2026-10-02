"""Offline regression checks for district lookup and issue-date alignment."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from contextlib import closing, contextmanager
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
with patch.dict(os.environ, {"DATABASE_URL": ""}):
    import imd_warnings as imd


class ImdWarningsTest(unittest.TestCase):
    def test_delhi_lookup_and_outside_coverage(self):
        district = imd.district_at(28.6139, 77.2090)
        self.assertIsNotNone(district)
        self.assertEqual(district["state"], "Delhi")
        self.assertIsNone(imd.district_at(12.9716, 77.5946))

    def test_polygon_hole_is_excluded(self):
        district = {"bbox": (0, 0, 10, 10), "polys": [[
            [(0, 0), (10, 0), (10, 10), (0, 10)],
            [(4, 4), (6, 4), (6, 6), (4, 6)],
        ]]}
        self.assertTrue(imd._in_district(2, 2, district))
        self.assertFalse(imd._in_district(5, 5, district))

    def test_warning_uses_record_issue_date(self):
        today = datetime.now(imd.IST).date()
        row = {"issue_date": (today - timedelta(days=1)).isoformat(), "days": [
            {"codes": [1], "color": 4},
            {"codes": [1, 15], "color": 3},
            {"codes": [2], "color": 2},
        ]}
        self.assertEqual(imd._day_info(row, today)["hazards"], ["Fog"])
        self.assertEqual(imd._day_info(row, today + timedelta(days=1))["level"], "alert")
        self.assertIsNone(imd._day_info(row, today + timedelta(days=5)))

    def test_route_deduplicates_districts_and_reads_stored_warnings(self):
        @contextmanager
        def connection_for_test():
            with closing(imd.db.connect(imd.DB_PATH)) as connection:
                with connection:
                    yield connection

        district = imd.district_at(28.6139, 77.2090)
        today = datetime.now(imd.IST).date().isoformat()
        properties = {"ID": district["id"], "District": district["name"], "Date": today}
        for day in range(1, 6):
            properties[f"Day_{day}"] = "15" if day == 1 else "1"
            properties[f"Day{day}_Color"] = 3 if day == 1 else 4
        with tempfile.TemporaryDirectory() as directory, patch.object(imd, "DB_PATH", str(Path(directory) / "imd.db")), patch.object(imd, "_db_ready", False), patch.object(imd, "_conn", connection_for_test):
            imd.init_db()
            with imd._conn() as connection:
                connection.execute(
                    "INSERT INTO imd_district_warnings VALUES (?,?,?,?,?,?,?)",
                    imd._parse_feature(properties) + (datetime.now(imd.timezone.utc).isoformat(),),
                )
            with patch.object(imd.requests, "get", side_effect=AssertionError("Offline test attempted network access")):
                result = imd.route_warnings([(28.6139, 77.2090, 0), (28.6139, 77.2090, 5), (12.9716, 77.5946, 10)])
            self.assertTrue(result["available"])
            self.assertEqual(result["districts_on_route"], 1)
            self.assertEqual(result["uncovered_waypoints"], 1)
            self.assertEqual(result["districts"][0]["today"]["hazards"], ["Fog"])
            self.assertFalse(result["districts"][0]["tomorrow"]["warned"])


if __name__ == "__main__":
    unittest.main()
