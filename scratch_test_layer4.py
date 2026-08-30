import unittest
from stage4_analyze.keyword_emotion import analyze_keyword_emotion
import os

class TestEmotionAnalysis(unittest.TestCase):
    def test_basic_logic(self):
        matches = [
            {"detected_as": "scam", "matched_keyword": "scam", "start": 1.0, "end": 1.5, "confidence": 0.99, "segment_text": "this is a scam"}
        ]
        # We need an audio file. If isolated audio exists, use it.
        isolated = "data/04_isolated_audio/target_speaker_isolated.wav"
        if os.path.exists(isolated):
            res = analyze_keyword_emotion(isolated, matches, pre_context_seconds=1.0, post_context_seconds=1.0)
            print(res)
            self.assertEqual(res[0]["emotion_status"], "analyzed")
            self.assertIn("emotion", res[0])

if __name__ == '__main__':
    unittest.main()
