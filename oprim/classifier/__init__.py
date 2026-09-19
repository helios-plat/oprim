from oprim.classifier.detect_image_exif import ImageExif, detect_image_exif
from oprim.classifier.detect_mime import detect_mime


def __getattr__(name: str):
    if name in {"PDFFeatures", "detect_pdf_features"}:
        from oprim.classifier.detect_pdf_features import PDFFeatures, detect_pdf_features

        return {"PDFFeatures": PDFFeatures, "detect_pdf_features": detect_pdf_features}[name]
    if name == "extract_text_sample":
        from oprim.classifier.extract_text_sample import extract_text_sample

        return extract_text_sample
    raise AttributeError(name)


__all__ = [
    "detect_mime",
    "detect_pdf_features",
    "PDFFeatures",
    "detect_image_exif",
    "ImageExif",
    "extract_text_sample",
]
