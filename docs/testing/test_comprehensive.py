"""
Comprehensive Test Suite for New Features

Tests:
1. Batch Processing
2. Benchmarking Modules
3. Rate Limiting
4. End-to-End Integration
"""

import unittest
import os
import sys
import time
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_processing.batch_processor import BatchProcessor, SmartBatchProcessor, BatchRequest, create_batch_processor
from video_processing.batch_parameters import BatchParameters, LLMProvider, CVModel
from video_processing.video_processor import RateLimiter
import numpy as np


class TestBatchProcessor(unittest.TestCase):
    """Test BatchProcessor functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = BatchProcessor(batch_size=3, requests_per_minute=60)
    
    def test_initialization(self):
        """Test BatchProcessor initialization"""
        self.assertEqual(self.processor.batch_size, 3)
        self.assertEqual(self.processor.requests_per_minute, 60)
        self.assertEqual(len(self.processor.pending_requests), 0)
        self.assertEqual(len(self.processor.results), 0)
    
    def test_add_request(self):
        """Test adding requests to queue"""
        request = BatchRequest(
            request_id="test_1",
            frames=[np.zeros((100, 100, 3), dtype=np.uint8)],
            prompt_text="Test prompt",
            context={}
        )
        
        self.processor.add_request(request)
        self.assertEqual(len(self.processor.pending_requests), 1)
        self.assertEqual(self.processor.pending_requests[0].request_id, "test_1")
    
    def test_batch_grouping(self):
        """Test that requests are grouped into correct batch sizes"""
        # Add 10 requests
        for i in range(10):
            request = BatchRequest(
                request_id=f"test_{i}",
                frames=[np.zeros((100, 100, 3), dtype=np.uint8)],
                prompt_text=f"Test prompt {i}",
                context={}
            )
            self.processor.add_request(request)
        
        # With batch_size=3, should create 4 batches (3+3+3+1)
        total_requests = len(self.processor.pending_requests)
        num_batches = (total_requests + self.processor.batch_size - 1) // self.processor.batch_size
        
        self.assertEqual(total_requests, 10)
        self.assertEqual(num_batches, 4)
    
    def test_clear(self):
        """Test clearing processor state"""
        request = BatchRequest(
            request_id="test_1",
            frames=[np.zeros((100, 100, 3), dtype=np.uint8)],
            prompt_text="Test prompt",
            context={}
        )
        
        self.processor.add_request(request)
        self.processor.results["test_1"] = "result"
        
        self.processor.clear()
        
        self.assertEqual(len(self.processor.pending_requests), 0)
        self.assertEqual(len(self.processor.results), 0)


class TestSmartBatchProcessor(unittest.TestCase):
    """Test SmartBatchProcessor functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = SmartBatchProcessor(batch_size=5, requests_per_minute=60)
    
    def test_request_separation(self):
        """Test that keyframes and intervals are separated"""
        # Add keyframe requests
        for i in range(5):
            request = BatchRequest(
                request_id=f"keyframe_{i}",
                frames=[np.zeros((100, 100, 3), dtype=np.uint8)],
                prompt_text=f"Keyframe {i}",
                context={}
            )
            self.processor.add_request(request)
        
        # Add interval requests
        for i in range(3):
            request = BatchRequest(
                request_id=f"interval_{i}_{i+1}",
                frames=[np.zeros((100, 100, 3), dtype=np.uint8)],
                prompt_text=f"Interval {i}",
                context={}
            )
            self.processor.add_request(request)
        
        # Check separation
        keyframe_requests = [r for r in self.processor.pending_requests if 'keyframe' in r.request_id]
        interval_requests = [r for r in self.processor.pending_requests if 'interval' in r.request_id]
        
        self.assertEqual(len(keyframe_requests), 5)
        self.assertEqual(len(interval_requests), 3)


class TestRateLimiter(unittest.TestCase):
    """Test RateLimiter functionality"""
    
    def test_initialization(self):
        """Test RateLimiter initialization"""
        limiter = RateLimiter(requests_per_minute=60, buffer=0.1)
        
        self.assertEqual(limiter.requests_per_minute, 60)
        self.assertEqual(limiter.buffer, 0.1)
        self.assertAlmostEqual(limiter.min_interval, 1.1, places=1)
        self.assertEqual(limiter.request_count, 0)
    
    def test_wait_if_needed(self):
        """Test rate limiting delays"""
        limiter = RateLimiter(requests_per_minute=120, buffer=0.0)  # Fast for testing
        
        start_time = time.time()
        
        # First request should not wait
        limiter.wait_if_needed()
        first_request_time = time.time() - start_time
        self.assertLess(first_request_time, 0.1)
        
        # Second request should wait
        limiter.wait_if_needed()
        second_request_time = time.time() - start_time
        self.assertGreater(second_request_time, 0.4)  # Should wait ~0.5s
    
    def test_request_counting(self):
        """Test request count tracking"""
        limiter = RateLimiter(requests_per_minute=120, buffer=0.0)
        
        for i in range(3):
            limiter.wait_if_needed()
        
        stats = limiter.get_stats()
        self.assertEqual(stats['total_requests'], 3)


class TestBatchParameters(unittest.TestCase):
    """Test BatchParameters configuration"""
    
    def test_batch_processing_defaults(self):
        """Test that batch processing is enabled by default"""
        params = BatchParameters()
        
        self.assertTrue(params.enable_batch_processing)
        self.assertEqual(params.batch_size, 5)
        self.assertTrue(params.use_smart_batching)
    
    def test_rate_limiting_defaults(self):
        """Test rate limiting defaults"""
        params = BatchParameters()
        
        self.assertTrue(params.enable_rate_limiting)
        self.assertEqual(params.api_requests_per_minute, 10)
        self.assertEqual(params.rate_limit_buffer, 0.1)
    
    def test_gemini_configuration(self):
        """Test Gemini-specific configuration"""
        params = BatchParameters(
            llm_provider=LLMProvider.GEMINI,
            llm_model="gemini-2.0-flash-exp",
            enable_batch_processing=True,
            batch_size=5
        )
        
        self.assertEqual(params.llm_provider, LLMProvider.GEMINI)
        self.assertEqual(params.llm_model, "gemini-2.0-flash-exp")
        self.assertTrue(params.enable_batch_processing)


class TestBatchProcessorFactory(unittest.TestCase):
    """Test batch processor factory function"""
    
    def test_create_basic_processor(self):
        """Test creating basic BatchProcessor"""
        params = BatchParameters(use_smart_batching=False)
        processor = create_batch_processor(params)
        
        self.assertIsInstance(processor, BatchProcessor)
        self.assertNotIsInstance(processor, SmartBatchProcessor)
    
    def test_create_smart_processor(self):
        """Test creating SmartBatchProcessor"""
        params = BatchParameters(use_smart_batching=True)
        processor = create_batch_processor(params)
        
        self.assertIsInstance(processor, SmartBatchProcessor)


class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.params = BatchParameters(
            llm_provider=LLMProvider.GEMINI,
            llm_model="gemini-2.0-flash-exp",
            enable_batch_processing=True,
            batch_size=3,
            api_requests_per_minute=10,
            video_directory=self.test_dir,
            csv_directory=os.path.join(self.test_dir, "outputs"),
            batch_tracking_directory=os.path.join(self.test_dir, "tracking")
        )
    
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_batch_id_generation(self):
        """Test that batch_id is auto-generated"""
        self.assertIsNotNone(self.params.batch_id)
        self.assertTrue(self.params.batch_id.startswith("batch_"))
    
    def test_batch_config_save(self):
        """Test saving batch configuration"""
        config_path = self.params.save_batch_config()
        
        self.assertTrue(os.path.exists(config_path))
        
        # Load and verify
        loaded_params = BatchParameters.from_batch_id(
            self.params.batch_id,
            tracking_directory=self.params.batch_tracking_directory
        )
        
        self.assertEqual(loaded_params.batch_id, self.params.batch_id)
        self.assertEqual(loaded_params.llm_provider, self.params.llm_provider)
        self.assertEqual(loaded_params.batch_size, self.params.batch_size)


def run_tests():
    """Run all tests and return results"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestBatchProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestSmartBatchProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestRateLimiter))
    suite.addTests(loader.loadTestsFromTestCase(TestBatchParameters))
    suite.addTests(loader.loadTestsFromTestCase(TestBatchProcessorFactory))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    print("="*70)
    print("COMPREHENSIVE TEST SUITE - NEW FEATURES")
    print("="*70)
    print()
    
    result = run_tests()
    
    print()
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print()
    
    if result.wasSuccessful():
        print("✓ ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("✗ SOME TESTS FAILED")
        sys.exit(1)
