#!/usr/bin/env python3
"""
Simple YAML parser helper for bash scripts.
Replacement for yq command using PyYAML.

Usage:
    python3 yaml_parser.py <yaml_file> <query>
    
Examples:
    python3 yaml_parser.py config.yaml '.global.queue'
    python3 yaml_parser.py config.yaml '.datasets | keys'
    python3 yaml_parser.py config.yaml '.datasets.opv.targets[]'
"""

import sys
import yaml
from pathlib import Path

def get_value(data, query):
    """Parse a simple query path like '.global.queue' or '.datasets.opv.targets[]'"""
    
    # Remove leading dot
    if query.startswith('.'):
        query = query[1:]
    
    # Handle special queries
    if ' | keys' in query:
        # Get keys of a dict
        path = query.replace(' | keys', '').strip()
        parts = path.split('.') if path else []
        current = data
        for part in parts:
            if part:
                current = current[part]
        if isinstance(current, dict):
            return list(current.keys())
        return []
    
    # Handle array access with []
    if query.endswith('[]'):
        query = query[:-2]
        parts = query.split('.')
        current = data
        for part in parts:
            if part:
                current = current[part]
        if isinstance(current, list):
            return current
        return []
    
    # Regular path traversal
    parts = query.split('.')
    current = data
    for part in parts:
        if part:
            if isinstance(current, dict):
                current = current.get(part)
                if current is None:
                    return 'null'
            else:
                return 'null'
    
    return current

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 yaml_parser.py <yaml_file> <query>", file=sys.stderr)
        sys.exit(1)
    
    yaml_file = sys.argv[1]
    query = sys.argv[2]
    
    # Load YAML file
    try:
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {yaml_file}", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Get value from query
    result = get_value(data, query)
    
    # Output result
    if isinstance(result, list):
        for item in result:
            print(item)
    elif isinstance(result, bool):
        print(str(result).lower())
    elif result is None:
        print('null')
    else:
        print(result)

if __name__ == "__main__":
    main()
