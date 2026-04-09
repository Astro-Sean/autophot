#!/usr/bin/env python3
"""
Debug script to test TNS lookup and identify why coordinates aren't being resolved.
"""

import sys
sys.path.append('/home/sbrennan/Documents/autophot_object')

import autophot_tokens
from prepare import Prepare

# Create a minimal input to test TNS
test_input = {
    'target_name': 'SN2026gzf',
    'catalog': {
        'MASTcasjobs_wsid': autophot_tokens.MASTcasjobs_wsid,
        'MASTcasjobs_pwd': autophot_tokens.MASTcasjobs_pwd,
    }
}

print("Testing TNS lookup for SN2026gzf...")
print(f"TNS WSID: {autophot_tokens.MASTcasjobs_wsid}")
print(f"TNS Password: {'*' * len(autophot_tokens.MASTcasjobs_pwd) if autophot_tokens.MASTcasjobs_pwd else 'None'}")

try:
    # Create Prepare instance and test TNS
    prep = Prepare(test_input)
    tns_result = prep.check_tns()
    
    print("✅ TNS lookup successful!")
    print(f"Result: {tns_result}")
    
    if 'radeg' in tns_result and 'decdeg' in tns_result:
        print(f"Coordinates: RA={tns_result['radeg']}, Dec={tns_result['decdeg']}")
    else:
        print("❌ No coordinates in TNS result")
        
except Exception as e:
    print(f"❌ TNS lookup failed: {e}")
    print(f"Error type: {type(e).__name__}")
    
    import traceback
    print("Full traceback:")
    traceback.print_exc()

print("\n" + "="*50)
print("SOLUTIONS:")
print("1. If TNS failed, add explicit coordinates to your script:")
print("   autophot_input['target_ra'] = 152.207497")
print("   autophot_input['target_dec'] = -67.047493")
print("2. Check TNS credentials in autophot_tokens.py")
print("3. Verify network connection to TNS API")
