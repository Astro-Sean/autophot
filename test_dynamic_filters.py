#!/usr/bin/env python3
"""
Test script to verify that the dynamic filter system correctly accepts arbitrary filter names.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from functions import normalize_photometric_filter_name, sanitize_photometric_filters

def test_dynamic_filter_system():
    """Test that arbitrary filter names are correctly accepted."""
    
    print("Testing Dynamic Filter System")
    print("=" * 50)
    
    # Test 1: Standard filters should still work
    print("\n1. Testing standard filters...")
    standard_filters = ['U', 'B', 'V', 'R', 'I', 'u', 'g', 'r', 'i', 'z', 'J', 'H', 'K']
    available_standard = list(standard_filters)
    
    for filt in standard_filters:
        result = normalize_photometric_filter_name(filt, available_filters=available_standard)
        assert result == filt, f"Standard filter {filt} failed: got {result}"
    print("✓ All standard filters work")
    
    # Test 2: Arbitrary filter names should be accepted
    print("\n2. Testing arbitrary filter names...")
    arbitrary_filters = ['my_custom_filter', 'NIR_band', 'special_survey', 'weird_filter', 'Y', 'w']
    available_arbitrary = list(arbitrary_filters)
    
    for filt in arbitrary_filters:
        result = normalize_photometric_filter_name(filt, available_filters=available_arbitrary)
        assert result == filt, f"Arbitrary filter {filt} failed: got {result}"
    print("✓ All arbitrary filters accepted")
    
    # Test 3: Mixed standard and arbitrary filters
    print("\n3. Testing mixed standard and arbitrary filters...")
    mixed_filters = ['U', 'B', 'my_custom_filter', 'g', 'NIR_band', 'K', 'special_survey']
    available_mixed = list(mixed_filters)
    
    for filt in mixed_filters:
        result = normalize_photometric_filter_name(filt, available_filters=available_mixed)
        assert result == filt, f"Mixed filter {filt} failed: got {result}"
    print("✓ Mixed filters work correctly")
    
    # Test 4: sanitize_photometric_filters with arbitrary filters
    print("\n4. Testing sanitize_photometric_filters with arbitrary filters...")
    test_filters = ['U', 'my_custom', 'g', 'NIR_band', 'invalid_filter', 'K']
    available_test = ['U', 'my_custom', 'g', 'NIR_band', 'K']  # Note: invalid_filter not in available
    
    cleaned, dropped = sanitize_photometric_filters(test_filters, available_filters=available_test)
    
    expected_cleaned = ['U', 'my_custom', 'g', 'NIR_band', 'K']
    expected_dropped = ['invalid_filter']
    
    assert set(cleaned) == set(expected_cleaned), f"Cleaned mismatch: got {cleaned}, expected {expected_cleaned}"
    assert set(dropped) == set(expected_dropped), f"Dropped mismatch: got {dropped}, expected {expected_dropped}"
    print("✓ sanitize_photometric_filters works correctly")
    
    # Test 5: Case insensitive matching
    print("\n5. Testing case insensitive matching...")
    case_filters = ['My_Custom_Filter', 'nir_band', 'Y']
    available_case = ['my_custom_filter', 'NIR_band', 'Y']
    
    for filt in case_filters:
        result = normalize_photometric_filter_name(filt, available_filters=available_case)
        assert result in available_case, f"Case insensitive filter {filt} failed: got {result}"
    print("✓ Case insensitive matching works")
    
    # Test 6: Fuzzy matching
    print("\n6. Testing fuzzy matching...")
    fuzzy_filters = ['my_custm_filter', 'NIR_bnd']  # Small typos
    available_fuzzy = ['my_custom_filter', 'NIR_band']
    
    for filt in fuzzy_filters:
        result = normalize_photometric_filter_name(filt, available_filters=available_fuzzy)
        assert result is not None, f"Fuzzy matching failed for {filt}"
        print(f"  {filt} -> {result}")
    print("✓ Fuzzy matching works")
    
    print("\n" + "=" * 50)
    print("✅ ALL TESTS PASSED!")
    print("The dynamic filter system is correctly set up to accept arbitrary filter names.")
    
    return True

def test_edge_cases():
    """Test edge cases and error conditions."""
    
    print("\nTesting Edge Cases")
    print("=" * 30)
    
    # Test with None
    result = normalize_photometric_filter_name(None)
    assert result is None, "None should return None"
    print("✓ None handling works")
    
    # Test with empty string
    result = normalize_photometric_filter_name("")
    assert result is None, "Empty string should return None"
    print("✓ Empty string handling works")
    
    # Test with non-photometric keys
    non_photo = ['RA', 'DEC', 'name', 'mag_err']
    available = ['U', 'B', 'V']
    
    for filt in non_photo:
        result = normalize_photometric_filter_name(filt, available_filters=available)
        assert result is None, f"Non-photometric {filt} should return None"
    print("✓ Non-photometric keys correctly rejected")
    
    # Test with no available_filters (backward compatibility)
    result = normalize_photometric_filter_name('U')
    assert result == 'U', "Backward compatibility failed"
    print("✓ Backward compatibility maintained")
    
    print("✅ ALL EDGE CASE TESTS PASSED!")

if __name__ == "__main__":
    try:
        test_dynamic_filter_system()
        test_edge_cases()
        print("\n🎉 Dynamic filter system is working correctly!")
        print("Users can now use ANY filter name with custom catalogs or transmission curves.")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
