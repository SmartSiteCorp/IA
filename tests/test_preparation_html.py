from html.parser import HTMLParser

from smartsite_ia.preparation_html import write_preparation_page


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.targets = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self.targets.extend(value for name, value in attrs if name in ("src", "href"))


def test_report_escapes_explanations_and_links_masks_and_compared_images(tmp_path):
    report = {
        "dataset": "concrete_crack_segmentation_v1",
        "split_counts": {"external_candidate": 1},
        "similar_pairs": [{"left": "001", "right": "001", "binary_mask_iou": 0.99}],
        "records": [
            {
                "id": "001",
                "image": "images/001.png",
                "mask": "masks/001.png",
                "preview": "overlays/001.jpg",
                "split": "external_candidate",
                "content_group": "001",
                "reason": '<script>alert("x")</script>',
            }
        ],
    }
    write_preparation_page(tmp_path, report)
    page = (tmp_path / "index.html").read_text()
    parser = Links()
    parser.feed(page)
    assert "script" not in parser.tags
    assert "&lt;script&gt;" in page
    assert "masks/001.png" in parser.targets
    assert "overlays/001.jpg" in parser.targets
    assert "99.00%" in page
    assert "<details>" in page
