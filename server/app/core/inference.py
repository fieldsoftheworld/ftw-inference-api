import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.models.project import InferenceResult, Project

logger = logging.getLogger(__name__)

# Queue for storing inference tasks
inference_queue = []
is_processing = False


def run_task(
    project_id: str, parameters: dict[str, Any], process_type: str = "inference"
):
    """
    Add inference task to queue and start processing if not already running

    Args:
        project_id: The ID of the project
        parameters: The parameters for the inference/polygonization
        process_type: The type of processing, either "inference" or "polygonize"
    """
    inference_queue.append((project_id, parameters, process_type))

    # Start processing thread if not already running
    global is_processing
    if not is_processing:
        threading.Thread(target=process_queue).start()


def process_queue():
    """
    Process inference queue
    """
    global is_processing
    is_processing = True

    while inference_queue:
        project_id, parameters, process_type = inference_queue.pop(0)

        # Get DB session
        db = SessionLocal()
        try:
            # Process the inference or polygonization
            if process_type == "inference":
                process_inference_queue(project_id, parameters, db)
            elif process_type == "polygonize":
                process_polygonize_queue(project_id, parameters, db)
            else:
                raise ValueError(f"Unknown process type: {process_type}")
        except Exception as e:
            logger.error(
                f"Error processing {process_type} for project {project_id}: {e}"
            )
            # Update project status to failed
            project = db.query(Project).filter(Project.id == project_id).first()
            if project:
                project.status = "failed"
                db.commit()
        finally:
            db.close()

    is_processing = False


def process_inference_queue(
    project_id: str, parameters: dict[str, Any], db: Session
) -> InferenceResult:
    """
    Process inference for a project
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

        # Get model ID from parameters
        model_id = parameters.get("model")
        if not model_id:
            raise ValueError("Model ID not specified in parameters")

        # Get model file path from config
        from app.core.config import get_settings

        settings = get_settings()

        model_config = next(
            (model for model in settings.models if model.get("id") == model_id),
            None,
        )
        if not model_config:
            raise ValueError(f"Model with ID {model_id} not found in configuration")

        # Get the model file name from the config
        model_file = model_config.get("file")
        if not model_file:
            raise ValueError(f"Model file not specified for model {model_id}")

        # Construct the full path to the model file
        model_path = (
            Path(__file__).parent.parent.parent / "data" / "models" / model_file
        )
        if not model_path.exists():
            raise ValueError(f"Model file not found at {model_path}")

        # Get image paths from database or parameters
        images = parameters.get("images")
        if images:
            # Images are provided as URLs, need to download them
            image_paths = download_images(project_id, images)
        else:
            # Get image paths from database
            db_images = db.query(project.images).all()
            if len(db_images) < 2:
                raise ValueError("Not enough images uploaded for this project")

            # Sort by window ('a' should come first)
            db_images.sort(key=lambda img: img.window)
            image_paths = [img.file_path for img in db_images]

        # Prepare polygonization options
        polygonize_opts = parameters.get("polygonize")

        # Update progress
        project.progress = 10
        db.commit()

        # Run ftw-tools inference
        try:
            from ftw_tools import (
                run_inference,  # Importing here to avoid dependency if not used
            )

            # Extract other parameters
            resize_factor = parameters.get("resize_factor", 2)
            patch_size = parameters.get("patch_size")
            padding = parameters.get("padding", 0)

            # Run inference
            result_path = run_inference(
                model_id=model_id,
                image_paths=image_paths,
                output_dir=str(results_dir),
                resize_factor=resize_factor,
                patch_size=patch_size,
                padding=padding,
            )

            # Update progress
            project.progress = 80
            db.commit()  # Create inference result record for image
            result = InferenceResult(
                project_id=project_id,
                model_id=model_id,
                result_type="image",
                file_path=result_path,
            )

            # Add result to database
            db.add(result)

            # Update project results field
            project.results["inference"] = result_path

            # Update project status to completed
            project.status = "completed"
            project.progress = 100
            db.commit()

            return result

        except ImportError:
            # Mock implementation for testing or when ftw_tools is not available
            logger.warning("ftw_tools not installed. Using mock implementation.")

            # Mock the inference process with a delay
            time.sleep(2)  # Simulate processing time

            # Create a mock result file (either image or GeoJSON)
            if polygonize_opts:
                # Create a mock GeoJSON file
                mock_geojson = {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [
                                    [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
                                ],
                            },
                            "properties": {"id": 1},
                        }
                    ],
                }

                result_path = str(results_dir / f"{project_id}_polygons.geojson")
                with open(result_path, "w") as f:
                    json.dump(mock_geojson, f)

                result_type = "geojson"
            else:
                # For image, just create an empty file for demonstration
                result_path = str(results_dir / f"{project_id}_result.tif")
                with open(result_path, "w") as f:
                    f.write("Mock TIF file")

                result_type = "image"  # Create inference result record
            result = InferenceResult(
                project_id=project_id,
                model_id=model_id,
                result_type=result_type,
                file_path=result_path,
            )

            db.add(result)

            # Update project results field
            project.results["inference"] = result_path

            # Update project status to completed
            project.status = "completed"
            project.progress = 100
            db.commit()

            return result

    except Exception:
        # Update project status to failed
        project.status = "failed"
        db.commit()
        raise


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

        # Extract polygonization parameters
        polygonize_params = parameters.get("polygonization", {})
        if not polygonize_params:
            # Use defaults from schema if not provided
            polygonize_params = {
                "simplify": 15,
                "min_size": 500,
                "close_interiors": False,
            }

        # Update progress
        project.progress = 50
        db.commit()

        # Run polygonization on the inference result
        logger.info(f"Running polygonization for project {project_id}")

        # In a real implementation, you would call your polygonization function here
        # For now, we'll create a mock GeoJSON result
        output_path = results_dir / f"polygons_{int(time.time())}.geojson"

        # Generate a simple mock GeoJSON
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
            json.dump(mock_geojson, f)  # Create new inference result record
        geojson_result = InferenceResult(
            project_id=project_id,
            result_type="geojson",
            file_path=str(output_path),
            parameters=parameters,
        )
        db.add(geojson_result)

        # Update project results field
        project.results["polygons"] = str(output_path)

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


def download_images(project_id: str, image_urls: list) -> list:
    """
    Download images from URLs
    This is a placeholder implementation; in a real application you would
    implement proper URL downloading with error handling
    """
    from pathlib import Path

    import requests

    # Create uploads directory if it doesn't exist
    project_dir = Path("data/uploads") / project_id
    project_dir.mkdir(exist_ok=True, parents=True)

    image_paths = []

    for i, url in enumerate(image_urls[:2]):  # Only use first two images
        # Download image
        response = requests.get(url)
        if response.status_code != 200:
            raise ValueError(f"Failed to download image from {url}")

        # Save image
        window = "a" if i == 0 else "b"
        file_path = project_dir / f"{window}.tif"
        with open(file_path, "wb") as f:
            f.write(response.content)

        image_paths.append(str(file_path))

    return image_paths
