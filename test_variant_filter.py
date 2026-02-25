#!/usr/bin/env python3
"""Test variant filtering logic for visualize_combined_results.py"""

# Test cases
test_features = [
    'Graph',
    'Graph+Desc',
    'Graph+RDKit',
    'Graph+Desc+RDKit',
    'Graph (BN)',
    'Graph+Desc (BN)',
    'Graph (FiLM)',
    'Graph+Desc (FiLM)',
    'Graph (FiLM-fllast)',
    'Graph+Desc (FiLM-fllast)',
    'Graph (BN) (FiLM)',
    'Graph+Desc (BN) (FiLM-fllast)',
    'Graph (Aux)',
    'Graph (NoFusion)',
    'Baseline_DMPNN',
    'Baseline_DMPNN+RDKit',
]

def matches_pattern(feature_str, pattern):
    """Check if feature string matches a pattern."""
    feature_lower = feature_str.lower()
    pattern_lower = pattern.lower()
    
    if pattern_lower == "graph-only":
        # Match Graph models without any mode suffixes
        if "graph" in feature_lower:
            if not any(mode in feature_lower for mode in ["(film", "(aux", "(nofusion"]):
                if "(" not in feature_lower or feature_lower.count("(") == 0:
                    return True
                if "baseline" not in feature_lower:
                    return True
    elif pattern_lower in ["film", "aux", "nofusion", "bn"]:
        # Match if the pattern appears in parentheses (mode suffix)
        if f"({pattern_lower}" in feature_lower:
            return True
    else:
        # General partial match
        if pattern_lower in feature_lower:
            return True
    
    return False

def filter_features(features, patterns):
    """Filter features by patterns."""
    return [f for f in features if any(matches_pattern(f, p) for p in patterns)]

print("=" * 70)
print("Test 1: graph-only filter (Graph without mode suffixes)")
print("=" * 70)
result = filter_features(test_features, ['graph-only'])
for f in result:
    print(f"  ✓ {f}")
print()

print("=" * 70)
print("Test 2: FiLM filter (any FiLM variant)")
print("=" * 70)
result = filter_features(test_features, ['FiLM'])
for f in result:
    print(f"  ✓ {f}")
print()

print("=" * 70)
print("Test 3: FiLM-fllast filter (specific FiLM last layer)")
print("=" * 70)
result = filter_features(test_features, ['FiLM-fllast'])
for f in result:
    print(f"  ✓ {f}")
print()

print("=" * 70)
print("Test 4: Multiple filters (graph-only + FiLM)")
print("=" * 70)
result = filter_features(test_features, ['graph-only', 'FiLM'])
for f in result:
    print(f"  ✓ {f}")
print()

print("=" * 70)
print("Test 5: BN filter (batch normalization)")
print("=" * 70)
result = filter_features(test_features, ['BN'])
for f in result:
    print(f"  ✓ {f}")
print()

print("=" * 70)
print("Test 6: Aux filter")
print("=" * 70)
result = filter_features(test_features, ['Aux'])
for f in result:
    print(f"  ✓ {f}")
