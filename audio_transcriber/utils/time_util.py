def format_timestamp(seconds: float) -> str:
    """Convert seconds (float) to hh:mm:ss.s format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"
