import os
import sys
# --- Add project root ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# -------------------------

def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    try:
        from transformers import WhisperProcessor, WhisperConfig
        from src.av_dicow.av_dicow_model import DiCoWForConditionalGeneration
        from src.dataset.avwhisper_dataset import load_audio, load_video, cut_or_pad, AudioTransform, VideoTransform, DataCollator
        from src.tokenizer.spm_tokenizer import TextTransform
        print("All imports successful!")
        return True
    except Exception as e:
        print(f"Import error: {e}")
        return False

def test_model_creation():
    """Test that the AVWhisper model can be created."""
    print("\nTesting model creation...")
    try:
        from transformers import WhisperConfig
        from src.av_dicow.av_dicow_model import DiCoWForConditionalGeneration
        from transformers import AutoModel
        import math
        from torchsummary import summary
        from transformers import AutoConfig
        config = 
        avsr_model = DiCoWForConditionalGeneration.from_pretrained("BUT-FIT/DiCoW_v3_2")
        vision_model = AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
        avsr_model.model.encoder.set_vision_encoder(vision_model)
        print("DiCoW model created successfully!")
        summary(avsr_model)
        # print(f"   Whisper Model hidden size: {config.d_model}")
        print(f"   Model parameters: {sum(p.numel() for p in avsr_model.parameters()):,}")
        return True
    except Exception as e:
        print(f"Model creation error: {e}")
        return False

def main():
    """Run all tests."""
    print("AVWhisper Setup Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_model_creation
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! You're ready to train AVWhisper.")
    else:
        print("⚠️  Some tests failed. Please fix the issues before training.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)