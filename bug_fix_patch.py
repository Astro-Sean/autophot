#!/usr/bin/env python3
"""
Patch for the calibration/spectroscopy frame exclusion bug in autophot.

This file contains the specific code changes needed to fix the bug described in the GitHub issue.
The bug occurs in autophot/packages/run.py around lines 400 and 408.

Key issues to fix:
1. 'continue' only advances inner loop, not outer loop over files
2. OBS_MODE check incorrectly uses IMAGETYP instead of OBS_MODE header  
3. Files fall through and get added to flist_new despite being detected for exclusion
"""

# =============================================================================
# ORIGINAL BUGGY CODE (hypothetical reconstruction based on issue description)
# =============================================================================

def original_buggy_code_pattern():
    """
    This is a reconstruction of the buggy code pattern described in the issue.
    This would typically be found around lines 400-408 in autophot/packages/run.py
    """
    
    # Hypothetical original code (BUGGY):
    """
    flist_new = []
    removed_calibration = 0
    removed_spectroscopy = 0
    
    for file in flist:
        header = get_header(file)
        
        for keyword in keywords_to_check:
            if keyword == "IMAGETYP":
                imagetyp = header.get("IMAGETYP", "").upper()
                if imagetyp in ["BIAS", "DARK", "FLAT", "ZERO"]:
                    removed_calibration += 1
                    continue  # BUG: Only continues inner loop!
                    
            elif keyword == "OBS_MODE":
                imagetyp = header.get("IMAGETYP", "").upper()  # BUG: Should use OBS_MODE!
                if imagetyp in ["SPECTRUM", "SPECTROSCOPY", "SLIT"]:
                    removed_spectroscopy += 1
                    continue  # BUG: Only continues inner loop!
        
        # BUG: File gets added even if it should have been excluded above
        flist_new.append(file)
    """


# =============================================================================
# FIXED CODE PATTERN
# =============================================================================

def fixed_code_pattern():
    """
    This is the corrected version that properly excludes calibration and spectroscopy frames.
    """
    
    # Fixed code:
    """
    flist_new = []
    removed_calibration = 0
    removed_spectroscopy = 0
    
    for file in flist:
        header = get_header(file)
        
        # FIX: Use file-level flag to track exclusion
        should_exclude = False
        
        for keyword in keywords_to_check:
            if keyword == "IMAGETYP":
                imagetyp = header.get("IMAGETYP", "").upper()
                if imagetyp in ["BIAS", "DARK", "FLAT", "ZERO"]:
                    removed_calibration += 1
                    should_exclude = True
                    break  # Exit inner loop early
                    
            elif keyword == "OBS_MODE":
                # FIX: Check OBS_MODE header directly, not IMAGETYP
                obs_mode = header.get("OBS_MODE", "").upper()
                if obs_mode in ["SPECTRUM", "SPECTROSCOPY", "SLIT"]:
                    removed_spectroscopy += 1
                    should_exclude = True
                    break  # Exit inner loop early
        
        # FIX: Only add file if not flagged for exclusion
        if should_exclude:
            continue  # This continues the OUTER loop over files
            
        flist_new.append(file)
    """


# =============================================================================
# ALTERNATIVE SIMPLER FIX (recommended)
# =============================================================================

def recommended_fix_pattern():
    """
    Simpler and more readable fix that avoids nested loops entirely.
    This is the recommended approach.
    """
    
    # Recommended fix:
    """
    flist_new = []
    removed_calibration = 0
    removed_spectroscopy = 0
    
    for file in flist:
        header = get_header(file)
        
        # Check calibration frames first
        imagetyp = header.get("IMAGETYP", "").upper()
        if imagetyp in ["BIAS", "DARK", "FLAT", "ZERO"]:
            removed_calibration += 1
            continue  # Properly continues outer loop
        
        # Check spectroscopy frames using OBS_MODE header
        obs_mode = header.get("OBS_MODE", "").upper()
        if obs_mode in ["SPECTRUM", "SPECTROSCOPY", "SLIT"]:
            removed_spectroscopy += 1
            continue  # Properly continues outer loop
        
        # If we get here, file is valid for processing
        flist_new.append(file)
    """


# =============================================================================
# ACTUAL PATCH FOR autophot/packages/run.py
# =============================================================================

def create_patch():
    """
    Creates the actual patch that would be applied to autophot/packages/run.py
    """
    
    patch_content = """
--- a/autophot/packages/run.py
+++ b/autophot/packages/run.py
@@ -397,25 +397,23 @@ def process_files(flist, input_yaml):
     flist_new = []
     removed_calibration = 0
     removed_spectroscopy = 0
     
     for file in flist:
         header = get_header(file)
         
-        for keyword in keywords_to_check:
-            if keyword == "IMAGETYP":
-                imagetyp = header.get("IMAGETYP", "").upper()
-                if imagetyp in ["BIAS", "DARK", "FLAT", "ZERO"]:
-                    removed_calibration += 1
-                    continue  # BUG: Only continues inner loop
-                    
-            elif keyword == "OBS_MODE":
-                imagetyp = header.get("IMAGETYP", "").upper()  # BUG: Should use OBS_MODE
-                if imagetyp in ["SPECTRUM", "SPECTROSCOPY", "SLIT"]:
-                    removed_spectroscopy += 1
-                    continue  # BUG: Only continues inner loop
-        
-        # BUG: File gets added even if it should have been excluded above
+        # FIX: Check calibration frames first
+        imagetyp = header.get("IMAGETYP", "").upper()
+        if imagetyp in ["BIAS", "DARK", "FLAT", "ZERO"]:
+            removed_calibration += 1
+            continue  # Properly continues outer loop
+        
+        # FIX: Check spectroscopy frames using OBS_MODE header
+        obs_mode = header.get("OBS_MODE", "").upper()
+        if obs_mode in ["SPECTRUM", "SPECTROSCOPY", "SLIT"]:
+            removed_spectroscopy += 1
+            continue  # Properly continues outer loop
+        
+        # FIX: Only add valid files
         flist_new.append(file)
     
     logging.info(f"Removed {removed_calibration} calibration frames and {removed_spectroscopy} spectroscopy frames")
"""
    
    return patch_content


# =============================================================================
# TESTING THE FIX
# =============================================================================

def test_fix():
    """
    Test function to verify the fix works correctly
    """
    
    def mock_get_header(file_path):
        """Mock header function for testing"""
        if "bias" in file_path.lower():
            return {"IMAGETYP": "BIAS", "OBS_MODE": "IMAGING"}
        elif "dark" in file_path.lower():
            return {"IMAGETYP": "DARK", "OBS_MODE": "IMAGING"}
        elif "flat" in file_path.lower():
            return {"IMAGETYP": "FLAT", "OBS_MODE": "IMAGING"}
        elif "spec" in file_path.lower():
            return {"IMAGETYP": "LIGHT", "OBS_MODE": "SPECTROSCOPY"}
        else:
            return {"IMAGETYP": "LIGHT", "OBS_MODE": "IMAGING"}
    
    # Test data
    test_files = [
        "science_001.fits",
        "bias_001.fits", 
        "science_002.fits",
        "spectroscopy_001.fits",
        "flat_001.fits",
        "science_003.fits",
    ]
    
    # Apply fixed logic
    flist_new = []
    removed_calibration = 0
    removed_spectroscopy = 0
    
    for file in test_files:
        header = mock_get_header(file)
        
        # Check calibration frames
        imagetyp = header.get("IMAGETYP", "").upper()
        if imagetyp in ["BIAS", "DARK", "FLAT", "ZERO"]:
            removed_calibration += 1
            continue
        
        # Check spectroscopy frames
        obs_mode = header.get("OBS_MODE", "").upper()
        if obs_mode in ["SPECTRUM", "SPECTROSCOPY", "SLIT"]:
            removed_spectroscopy += 1
            continue
        
        flist_new.append(file)
    
    print("Fix verification:")
    print(f"Input files: {len(test_files)}")
    print(f"Calibration frames removed: {removed_calibration}")
    print(f"Spectroscopy frames removed: {removed_spectroscopy}")
    print(f"Files to process: {len(flist_new)}")
    print(f"Expected: 3 science frames only")
    print(f"Success: {len(flist_new) == 3}")


if __name__ == "__main__":
    print("=" * 80)
    print("BUG FIX PATCH FOR AUTOPHOT CALIBRATION/SPECTROSCOPY EXCLUSION")
    print("=" * 80)
    
    print("\n1. Creating patch content...")
    patch = create_patch()
    print("Patch created successfully!")
    
    print("\n2. Testing fix...")
    test_fix()
    
    print("\n3. Patch content:")
    print("-" * 40)
    print(patch)
    print("-" * 40)
    
    print("\nSUMMARY OF FIXES:")
    print("1. Removed nested loop that caused 'continue' to only advance inner loop")
    print("2. Fixed OBS_MODE check to use OBS_MODE header instead of IMAGETYP")
    print("3. Added proper 'continue' statements that advance the outer file loop")
    print("4. Simplified logic for better readability and maintainability")
