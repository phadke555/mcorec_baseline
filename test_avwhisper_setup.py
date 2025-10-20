#!/usr/bin/env python3
"""
Test script to verify AVWhisper setup works correctly.
Run this before training to check for any issues.
"""

import os
import sys
os.sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))

def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    try:
        from transformers import WhisperProcessor, WhisperConfig
        from src.av_whisper.av_whisper_model import AVWhisperForConditionalGeneration
        from src.dataset.avwhisper_dataset import create_avwhisper_collator
        from src.tokenizer.spm_tokenizer import TextTransform
        print("✅ All imports successful!")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_model_creation():
    """Test that the AVWhisper model can be created."""
    print("\nTesting model creation...")
    try:
        from transformers import WhisperConfig
        from src.av_whisper.av_whisper_model import AVWhisperForConditionalGeneration
        
        config = WhisperConfig.from_pretrained("openai/whisper-small")
        model = AVWhisperForConditionalGeneration(config)
        print("✅ AVWhisper model created successfully!")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        return True
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False

def test_processor_creation():
    """Test that WhisperProcessor can be created."""
    print("\nTesting processor creation...")
    try:
        from transformers import WhisperProcessor
        processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        print("✅ WhisperProcessor created successfully!")
        return True
    except Exception as e:
        print(f"❌ Processor creation error: {e}")
        return False

def test_collator_creation():
    """Test that the data collator can be created."""
    print("\nTesting collator creation...")
    try:
        from transformers import WhisperProcessor
        from src.dataset.avwhisper_dataset import create_avwhisper_collator
        from src.tokenizer.spm_tokenizer import TextTransform
        
        # Create dummy paths (they don't need to exist for this test)
        sp_model_path = "src/tokenizer/spm/unigram/unigram5000.model"
        dict_path = "src/tokenizer/spm/unigram/unigram5000_units.txt"
        
        text_transform = TextTransform(
            sp_model_path=sp_model_path,
            dict_path=dict_path,
        )
        
        processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        collator = create_avwhisper_collator(
            processor=processor,
            text_transform=text_transform,
            subset="train"
        )
        print("✅ Data collator created successfully!")
        return True
    except Exception as e:
        print(f"❌ Collator creation error: {e}")
        return False

def test_dinov3_loading():
    """Test that DINOv3 can be loaded."""
    print("\nTesting DINOv3 loading...")
    try:
        from transformers import AutoModel
        dinov3 = AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
        print("✅ DINOv3 model loaded successfully!")
        print(f"   DINOv3 hidden size: {dinov3.config.hidden_size}")
        return True
    except Exception as e:
        print(f"❌ DINOv3 loading error: {e}")
        return False

def main():
    """Run all tests."""
    print("AVWhisper Setup Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_processor_creation,
        test_dinov3_loading,
        test_model_creation,
        test_collator_creation,
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
