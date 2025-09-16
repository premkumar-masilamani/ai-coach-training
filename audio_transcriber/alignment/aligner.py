import logging

logger = logging.getLogger()

class Aligner:
    def __init__(self):
        pass

    def align(self, transcribed_json: str, diarized_json: str) -> str:
        # TODO: Implement alignment logic
        logger.info("Simply returning the transcribed JSON, without alignment")
        return transcribed_json
