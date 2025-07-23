import asyncio

from app.core.logging import get_logger

logger = get_logger(__name__)


async def run_async(cmd):
    """Run subprocess command asynchronously"""
    logger.debug(f"Running command: {' '.join(cmd)}")

    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()

    stdout_text = stdout.decode().strip()
    stderr_text = stderr.decode().strip()

    if stdout_text:
        logger.debug(f"Command stdout: {stdout_text}")
    if stderr_text:
        logger.debug(f"Command stderr: {stderr_text}")

    logger.debug("Command completed", extra={"return_code": process.returncode})

    if process.returncode != 0:
        lines = stderr_text.splitlines()
        error_message = lines[-1] if lines else "Unknown error"
        logger.error(
            f"Command failed: {error_message}",
            extra={"return_code": process.returncode},
        )
        raise ValueError(error_message)

    return process
