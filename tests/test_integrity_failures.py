"""Non-vacuous integrity checks, including missing/corrupt image fault injection."""
from pathlib import Path
import tempfile,shutil,unittest
from analysis.reproduce import image_audit
ROOT=Path(__file__).resolve().parents[1]
class Integrity(unittest.TestCase):
    def test_missing_and_corrupt_image_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            shutil.copytree(ROOT/'config',root/'config')
            shutil.copytree(ROOT/'provenance',root/'provenance')
            shutil.copytree(ROOT/'source_data/inputs',root/'source_data/inputs')
            image=root/'source_data/inputs/images/fir/Q001.png';original=image.read_bytes();image.unlink()
            with self.assertRaisesRegex(ValueError,'Missing or changed original image'):image_audit(root)
            image.write_bytes(original+b'corruption')
            with self.assertRaisesRegex(ValueError,'Missing or changed original image'):image_audit(root)
