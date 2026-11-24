"""QGIS Processing Provider & Algorithms for GeoGraph 2.0 (D1-P8).

Works natively inside QGIS 3.28+ (PyQGIS) and gracefully degrades to a lightweight
offline-testable base class when ``qgis.core`` is unavailable in headless CI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from geograph2.audit import run_gbf_30x30_audit
from geograph2.io import load_graphjson
from geograph2.rsp import build_reference_transition
from geograph2.rsp_adjoint import rsp_adjoint_sensitivity

try:  # pragma: no cover - active inside QGIS desktop
    from qgis.core import (
        QgsProcessingAlgorithm,
        QgsProcessingParameterFile,
        QgsProcessingParameterFolderDestination,
        QgsProcessingParameterNumber,
        QgsProcessingParameterString,
        QgsProcessingProvider,
    )
    HAS_QGIS = True
except ImportError:
    HAS_QGIS = False

    class QgsProcessingProvider:  # type: ignore[no-redef]
        def __init__(self) -> None:
            self._algorithms: list[Any] = []

        def addAlgorithm(self, alg: Any) -> None:
            self._algorithms.append(alg)

        def algorithms(self) -> list[Any]:
            return list(self._algorithms)

    class QgsProcessingAlgorithm:  # type: ignore[no-redef]
        def __init__(self) -> None:
            self._params: list[dict[str, Any]] = []

        def addParameter(self, param: dict[str, Any]) -> None:
            self._params.append(param)

        def parameterDefinitions(self) -> list[dict[str, Any]]:
            return list(self._params)

    def QgsProcessingParameterFile(name: str, description: str, extension: str = "") -> dict[str, Any]:  # type: ignore[misc]
        return {"name": name, "description": description, "type": "file", "extension": extension}

    def QgsProcessingParameterNumber(name: str, description: str, defaultValue: float = 0.0) -> dict[str, Any]:  # type: ignore[misc]
        return {"name": name, "description": description, "type": "number", "default": defaultValue}

    def QgsProcessingParameterString(name: str, description: str, defaultValue: str = "") -> dict[str, Any]:  # type: ignore[misc]
        return {"name": name, "description": description, "type": "string", "default": defaultValue}

    def QgsProcessingParameterFolderDestination(name: str, description: str) -> dict[str, Any]:  # type: ignore[misc]
        return {"name": name, "description": description, "type": "folder"}


class GBF30x30AuditAlgorithm(QgsProcessingAlgorithm):
    """QGIS Processing Algorithm: GBF Target 3 (30x30) Connectivity Audit."""

    INPUT_GRAPH = "INPUT_GRAPH"
    REGION_AREA_HA = "REGION_AREA_HA"
    SITE_NAME = "SITE_NAME"
    OUTPUT_DIR = "OUTPUT_DIR"

    def name(self) -> str:
        return "gbf_30x30_audit"

    def displayName(self) -> str:
        return "GBF 30x30 Protected Connectivity Audit"

    def group(self) -> str:
        return "Policy Audit"

    def groupId(self) -> str:
        return "policy_audit"

    def createInstance(self) -> GBF30x30AuditAlgorithm:
        return GBF30x30AuditAlgorithm()

    def initAlgorithm(self, config: dict[str, Any] | None = None) -> None:
        self.addParameter(QgsProcessingParameterFile(self.INPUT_GRAPH, "Habitat GraphJSON", extension="json"))
        self.addParameter(QgsProcessingParameterNumber(self.REGION_AREA_HA, "Region Area (ha)", defaultValue=1000.0))
        self.addParameter(QgsProcessingParameterString(self.SITE_NAME, "Site Name", defaultValue="Protected Network"))
        self.addParameter(QgsProcessingParameterFolderDestination(self.OUTPUT_DIR, "Audit Output Folder"))

    def processAlgorithm(
        self,
        parameters: dict[str, Any],
        context: Any = None,
        feedback: Any = None,
    ) -> dict[str, Any]:
        graph = load_graphjson(Path(parameters[self.INPUT_GRAPH]))
        region_area = float(parameters[self.REGION_AREA_HA])
        site_name = str(parameters.get(self.SITE_NAME, "Protected Network"))
        out_dir = Path(parameters[self.OUTPUT_DIR])

        report = run_gbf_30x30_audit(graph, region_area_ha=region_area, site_name=site_name)
        md_p, json_p = report.export(out_dir, stem="qgis_gbf_30x30_audit")
        return {
            "MARKDOWN_REPORT": str(md_p),
            "JSON_REPORT": str(json_p),
            "PROT_PCT": report.protconn.prot_pct,
            "PROTCONN_PCT": report.protconn.protconn_pct,
            "PC": report.pc.value,
        }


class RSPAdjointSensitivityAlgorithm(QgsProcessingAlgorithm):
    """QGIS Processing Algorithm: RSP Expected Cost & Adjoint Corridor Sensitivity."""

    INPUT_GRAPH = "INPUT_GRAPH"
    BETA = "BETA"
    ALPHA = "ALPHA"

    def name(self) -> str:
        return "rsp_adjoint_sensitivity"

    def displayName(self) -> str:
        return "RSP Adjoint Corridor Sensitivity"

    def group(self) -> str:
        return "Connectivity Solvers"

    def groupId(self) -> str:
        return "solvers"

    def createInstance(self) -> RSPAdjointSensitivityAlgorithm:
        return RSPAdjointSensitivityAlgorithm()

    def initAlgorithm(self, config: dict[str, Any] | None = None) -> None:
        self.addParameter(QgsProcessingParameterFile(self.INPUT_GRAPH, "Habitat GraphJSON", extension="json"))
        self.addParameter(QgsProcessingParameterNumber(self.BETA, "Inverse Temperature (beta)", defaultValue=1.0))
        self.addParameter(QgsProcessingParameterNumber(self.ALPHA, "Dispersal Decay (alpha)", defaultValue=0.25))

    def processAlgorithm(
        self,
        parameters: dict[str, Any],
        context: Any = None,
        feedback: Any = None,
    ) -> dict[str, Any]:
        graph = load_graphjson(Path(parameters[self.INPUT_GRAPH]))
        P, C = build_reference_transition(graph)
        adj = rsp_adjoint_sensitivity(
            P,
            C,
            beta=float(parameters.get(self.BETA, 1.0)),
            alpha=float(parameters.get(self.ALPHA, 0.25)),
            q=graph.node_qualities(),
        )
        return {
            "CONNECTIVITY_K": adj.connectivity_K,
            "MEAN_EXPECTED_COST": float(adj.rsp.expected_cost.mean()),
            "NODE_SENSITIVITIES": {nid: float(v) for nid, v in zip(graph.node_ids, adj.dK_dq)},
        }


class GeoGraph2Provider(QgsProcessingProvider):
    """QGIS Processing Provider exposing GeoGraph 2.0 algorithms."""

    def id(self) -> str:
        return "geograph2"

    def name(self) -> str:
        return "GeoGraph 2.0"

    def loadAlgorithms(self) -> None:
        self.addAlgorithm(GBF30x30AuditAlgorithm())
        self.addAlgorithm(RSPAdjointSensitivityAlgorithm())
