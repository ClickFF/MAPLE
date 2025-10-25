def log_error(error_message: str, output:str) -> None:
    """
    Logs error messages to the output file.

    Args:
        error_message: The error message to log.
        output: The output file.
    """
    with open(output, 'a') as file:
        file.write(f"ERROR: {error_message}\n")

def log_info(info_message: list, output:str) -> None:
    """
    Logs info messages to the output file.

    Args:
        info_message: The info message to log.
        output: The output file.
    """
    with open(output, 'a') as file:
        for info in info_message:   
            file.write(f"{info}")