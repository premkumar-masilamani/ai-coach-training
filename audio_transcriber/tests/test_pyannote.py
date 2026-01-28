from pyannote.core import Annotation, Segment

def test_crop():
    diarization = Annotation()
    diarization[Segment(0.0, 10.0)] = "A"

    query = Segment(5.0, 15.0)
    overlap = diarization.crop(query)

    print(f"Overlap Annotation: {overlap}")
    for segment, track, label in overlap.itertracks(yield_label=True):
        print(f"Segment: {segment}, Duration: {segment.duration}, Label: {label}")

if __name__ == "__main__":
    test_crop()
