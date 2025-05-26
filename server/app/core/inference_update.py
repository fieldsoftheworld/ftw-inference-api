import json
import logging
import time
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.core.inference import process_inference_queue
from app.models.project import InferenceResult, Project

logger = logging.getLogger(__name__)


def process_polygonize_queue(
    project_id: str, parameters: dict[str, Any], db: Session
) -> InferenceResult:
    """
    Process polygonization for a project, either on existing inference results
    or by running inference first

    Args:
        project_id: The ID of the project to process
        parameters: The parameters for the polygonization
        db: The database session

    Returns:
        The created inference result
    """
    # Get project
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise ValueError(f"Project with ID {project_id} not found")

    # Update project status
    project.status = "running"
    project.progress = 0
    db.commit()

    try:
        # Create results directory if it doesn't exist
        results_dir = Path("data/results") / project_id
        results_dir.mkdir(exist_ok=True, parents=True)

        # First check if we need to run inference or if we can use existing results
        inference_result = (
            db.query(InferenceResult)
            .filter(
                InferenceResult.project_id == project_id,
                InferenceResult.result_type == "image",
            )
            .order_by(InferenceResult.created_at.desc())
            .first()
        )

        # If no inference result exists, run inference first
        if not inference_result:
            logger.info(
                f"No inference result found for project {project_id}, "
                + "running inference first"
            )
            inference_result = process_inference_queue(project_id, parameters, db)

        # Update progress
        project.progress = 50
        db.commit()

        # Run polygonization on the inference result
        logger.info(f"Running polygonization for project {project_id}")

        try:
            # from ftw_tools import (
            #     run_polygonization,
            # )

            # Run the actual polygonization
            output_path = results_dir / f"polygons_{int(time.time())}.geojson"

            # polygonize_params = parameters.get("polygons", {})
            # In a real implementation, you would call the actual function
            # with appropriate parameters
            #
            # For example:
            # run_polygonization(
            #    inference_result.file_path,
            #    str(output_path),
            #    simplify=polygonize_params.get("simplify", 15),
            #    min_size=polygonize_params.get("min_size", 500),
            #    close_interiors=polygonize_params.get("close_interiors", False)
            # )

            # For now, create a mock result
            mock_geojson = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"id": 1},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]],
                        },
                    }
                ],
            }

            # Write GeoJSON to file
            with open(output_path, "w") as f:
                json.dump(mock_geojson, f)

        except ImportError:
            # Mock implementation when ftw_tools is not available
            logger.warning("ftw_tools not installed. Using mock implementation.")

            # Create a mock result
            output_path = results_dir / f"polygons_{int(time.time())}.geojson"
            mock_geojson = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"id": 1},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]],
                        },
                    }
                ],
            }

            # Write GeoJSON to file
            with open(output_path, "w") as f:
                json.dump(mock_geojson, f)

        # Create new inference result record
        geojson_result = InferenceResult(
            project_id=project_id,
            result_type="geojson",
            file_path=str(output_path),
            parameters=parameters,
        )
        db.add(geojson_result)

        # Update project status
        project.status = "completed"
        project.progress = 100
        db.commit()

        return geojson_result

    except Exception as e:
        # Update project status to failed
        project.status = "failed"
        project.progress = None
        db.commit()
        raise e
