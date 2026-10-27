import unittest

from vision import VisionLanguageModel


class _FakeCuda:
    def __init__(self, bf16_supported):
        self.bf16_supported = bf16_supported

    def is_available(self):
        return True

    def is_bf16_supported(self):
        return self.bf16_supported


class _FakeTorch:
    bfloat16 = "bfloat16"
    float16 = "float16"
    float32 = "float32"

    def __init__(self, bf16_supported):
        self.cuda = _FakeCuda(bf16_supported)


class VisionConfigurationTests(unittest.TestCase):
    def test_auto_4bit_compute_uses_bf16_when_supported(self):
        name, dtype = VisionLanguageModel._resolve_4bit_compute_dtype(
            _FakeTorch(bf16_supported=True),
            "auto",
        )
        self.assertEqual(name, "bfloat16")
        self.assertEqual(dtype, "bfloat16")

    def test_auto_4bit_compute_falls_back_to_fp32_when_bf16_is_unavailable(self):
        name, dtype = VisionLanguageModel._resolve_4bit_compute_dtype(
            _FakeTorch(bf16_supported=False),
            "auto",
        )
        self.assertEqual(name, "float32")
        self.assertEqual(dtype, "float32")

    def test_explicit_unavailable_bf16_uses_documented_fp32_fallback(self):
        name, dtype = VisionLanguageModel._resolve_4bit_compute_dtype(
            _FakeTorch(bf16_supported=False),
            "bf16",
        )
        self.assertEqual(name, "float32")
        self.assertEqual(dtype, "float32")


if __name__ == "__main__":
    unittest.main()
