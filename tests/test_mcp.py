import unittest

import jsonschema

from geograph2.mcp import GeoGraph2MCPDispatcher


class TestMCPAgentSkill(unittest.TestCase):
    def test_mcp_geo_contract_and_tool_dispatch(self):
        mcp = GeoGraph2MCPDispatcher()
        self.assertEqual(mcp.geo_contract["default_crs"], "EPSG:4326")
        self.assertTrue(mcp.geo_contract["offline_deterministic"])
        self.assertEqual(len(mcp.list_tools()), 4)

        dggs_res = mcp.invoke(
            "geograph2.build_spherical_dggs",
            {"bbox_lonlat": [0.0, 50.0, 3.0, 52.0], "hex_radius_km": 30.0},
        )
        self.assertLess(dggs_res["max_area_distortion_pct"], 0.5)

        case_res = mcp.invoke("geograph2.run_case_study", {"case_name": "protected-area-30x30"})
        self.assertTrue(case_res["prot_compliant"])

        # Invalid arguments rejected by JSON schema contract
        with self.assertRaises(jsonschema.ValidationError):
            mcp.invoke("geograph2.build_spherical_dggs", {"bbox_lonlat": [0.0, 50.0]})


if __name__ == "__main__":
    unittest.main()
