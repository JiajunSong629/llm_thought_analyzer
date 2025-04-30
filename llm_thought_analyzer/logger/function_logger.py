import logging
import os
from datetime import datetime
from pathlib import Path


class FunctionLogger:
    def __init__(self, log_dir: str = "logs", log_file: str = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Use provided log file or create a new one with timestamp
        if log_file:
            self.log_file = self.log_dir / log_file
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = self.log_dir / f"main_{timestamp}.log"

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler(),  # Also print to console
            ],
        )

        self.logger = logging.getLogger("function_reasoning")

        # Log the start of a new session
        # self.log_step(
        #     "SESSION_START", f"Starting new logging session in {self.log_file}", "INFO"
        # )

    def log_step(self, step_name: str, details: str, level: str = "INFO"):
        """Log a step in the function reasoning process.

        Args:
            step_name: Name of the step being logged
            details: Detailed information about the step
            level: Log level (INFO, DEBUG, WARNING, ERROR)
        """
        log_message = f"[{step_name}] {details}"

        if level.upper() == "INFO":
            self.logger.info(log_message)
        elif level.upper() == "DEBUG":
            self.logger.debug(log_message)
        elif level.upper() == "WARNING":
            self.logger.warning(log_message)
        elif level.upper() == "ERROR":
            self.logger.error(log_message)
        else:
            self.logger.info(log_message)  # Default to INFO

    def log_function_generation(self, function_name: str, reasoning_steps: list[str]):
        """Log the generation of a function with its reasoning steps.

        Args:
            function_name: Name of the function being generated
            reasoning_steps: List of reasoning steps taken
        """
        self.log_step(
            "FUNCTION_GENERATION",
            f"Starting generation of function: {function_name}",
            "INFO",
        )

        for i, step in enumerate(reasoning_steps, 1):
            self.log_step("REASONING_STEP", f"Step {i}: {step}", "INFO")

    def log_pool_creation(self, pool_name: str, functions: list[str]):
        """Log the creation of a function pool.

        Args:
            pool_name: Name of the function pool
            functions: List of functions in the pool
        """
        self.log_step("POOL_CREATION", f"Creating function pool: {pool_name}", "INFO")

        for func in functions:
            self.log_step("POOL_FUNCTION", f"Added function to pool: {func}", "INFO")

    def get_log_file_path(self) -> str:
        """Get the path to the current log file."""
        return str(self.log_file)
