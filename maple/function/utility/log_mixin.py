class LogMixin:
    """
    Mixin that provides standardised log_info / log_error helpers.

    Any class that mixes this in must set ``self.output`` (path to the
    output file) before calling the logging methods.
    """

    def log_error(self, error_message: str) -> None:
        """Append an ERROR line to the output file."""
        with open(self.output, 'a') as f:
            f.write(f"ERROR: {error_message}\n")

    def log_info(self, info_message: list) -> None:
        """Append a sequence of info lines to the output file."""
        with open(self.output, 'a') as f:
            for info in info_message:
                f.write(f"{info}")
