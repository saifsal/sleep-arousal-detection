"""
Neptune helper functions
"""
import neptune.new as neptune


class NeptuneHelper:
    """
    Neptune encapsulater class
    """

    def __init__(self, use_neptune: bool):
        """
        Initialize neptune if used
        """
        self.run = None

        if use_neptune:
            self.run = neptune.init(
                source_files=[
                    "../*.py",
                    "*.sh",
                    "../dataset/*.py",
                    "../li/*.py",
                    "../unet/*.py",
                    "../utils/*.py",
                ],
            )

    def log(self, string: str, file: str = None):
        """
        Log to stdout and neptune if used
        """
        if file and self.run is not None:
            self.run[file].log(string)

        if file:
            print(f"{file}: {string}")
        else:
            print(string)

    def upload_model(self, model_name: str):
        """
        Uploads model to neptune if used
        """
        if self.run is not None:
            self.run[model_name].upload(f"./{model_name}")
